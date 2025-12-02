# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch

# from Isaac Sim 4.2 onwards, pxr.Semantics is deprecated
try:
    import Semantics
except ModuleNotFoundError:
    from pxr import Semantics

from isaacsim.core.utils.stage import get_current_stage

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply
from isaaclab.utils.buffers import CircularBuffer

from isaaclab_tasks.direct.inhand_manipulation.inhand_manipulation_env import InHandManipulationEnv, unscale

from .feature_extractor import FeatureExtractor, FeatureExtractorCfg
from .shadow_hand_tactile_env_cfg import ShadowHandTactileEnvCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
import glob
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate
from isaaclab.sensors import ContactSensorCfg, ContactSensor
import numpy as np

usd_list = sorted(glob.glob('/home/ubuntu/IsaacLab/asset/mix/train/*.usd', recursive=True))




@configclass
class ShadowHandTactileEnvPlayCfg(ShadowHandTactileEnvCfg):
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=2.0, replicate_physics=False)
    # inference for CNN
    # feature_extractor = FeatureExtractorCfg(train=False, load_checkpoint=True)


class ShadowHandTactileEnv(InHandManipulationEnv):
    cfg: ShadowHandTactileEnvCfg

    def __init__(self, cfg: ShadowHandTactileEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # Use the log directory from the configuration
        self.feature_extractor = FeatureExtractor(self.cfg.feature_extractor, self.device, self.cfg.log_dir)

        # hide goal cubes
        self.goal_pos[:, :] = torch.tensor([-0.2, 0.1, 0.6], device=self.device)
        # keypoints buffer
        self.gt_keypoints = torch.ones(self.num_envs, 8, 3, dtype=torch.float32, device=self.device)
        self.goal_keypoints = torch.ones(self.num_envs, 8, 3, dtype=torch.float32, device=self.device)

        self.num_tactile_observations = 68
        self.dtype = torch.float16
        self.binary_tactile = True
        self.binary_threshold = 0.01


        self.tactile = torch.zeros((self.num_envs, self.num_tactile_observations), device=self.device)
        self.last_tactile = torch.zeros((self.num_envs, self.num_tactile_observations), device=self.device)

        self.buffer_length = 5
        # self.buffer = CircularBuffer(max_len=self.buffer_length, batch_size=(24+24+self.num_tactile_observations), device=self.device)
        self.buffer = CircularBuffer(max_len=self.buffer_length, batch_size=self.num_envs, device=self.device)
        self.fell_off_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.npy_list = []
        for _i_ in range(self.num_envs):
            npy_path = (usd_list[(_i_% len(usd_list))].replace('asset', 'features')).replace('.usd', '.npy')
            self.npy_list.append(np.load(npy_path))


    def _setup_scene(self):
        # add hand, in-hand object, and goal object
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        self.object_stage = RigidObject(self.cfg.object_stage_cfg)
        # self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        # get stage
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate (no need to filter for this environment)
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene - we must register to scene to randomize with EventManager
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        self.scene.rigid_objects["object_stage"] = self.object_stage
        # self.scene.sensors["tiled_camera"] = self._tiled_camera
        self.distal_sensor = ContactSensor(self.cfg.distal_contact_cfg)
        self.proximal_sensor = ContactSensor(self.cfg.proximal_contact_cfg)
        self.middle_sensor = ContactSensor(self.cfg.middle_contact_cfg)
        self.palm_sensor = ContactSensor(self.cfg.palm_contact_cfg)
        self.metacarpal_sensor = ContactSensor(self.cfg.metacarpal_contact_cfg)

        self.scene.sensors["distal_sensor"] = self.distal_sensor
        self.scene.sensors["proximal_sensor"] = self.proximal_sensor
        self.scene.sensors["middle_sensor"] = self.middle_sensor
        self.scene.sensors["palm_sensor"] = self.palm_sensor
        self.scene.sensors["metacarpal_sensor"] = self.metacarpal_sensor
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


    def _compute_image_observations(self):
        # generate ground truth keypoints for in-hand cube
        compute_keypoints(pose=torch.cat((self.object_pos, self.object_rot), dim=1), out=self.gt_keypoints)

        object_pose = torch.cat([self.object_pos, self.gt_keypoints.view(-1, 24)], dim=-1)

        # train CNN to regress on keypoint positions
        pose_loss, embeddings = self.feature_extractor.step(
            self._tiled_camera.data.output["rgb"],
            self._tiled_camera.data.output["depth"],
            self._tiled_camera.data.output["semantic_segmentation"][..., :3],
            object_pose,
        )

        self.embeddings = embeddings.clone().detach()
        # compute keypoints for goal cube
        compute_keypoints(
            pose=torch.cat((torch.zeros_like(self.goal_pos), self.goal_rot), dim=-1), out=self.goal_keypoints
        )

        obs = torch.cat(
            (
                self.embeddings,
                self.goal_keypoints.view(-1, 24),
            ),
            dim=-1,
        )

        # log pose loss from CNN training
        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["pose_loss"] = pose_loss.mean()

        return obs

    def _get_object_feature(self, input_obs):
        # feature
        features_list = []
        for _i_ in range(self.num_envs):
            # npy_path = (usd_list[(_i_% len(usd_list))].replace('asset', 'features')).replace('.usd', '.npy')
            # feature = torch.from_numpy(np.load(npy_path)).to(self.device)
            feature = torch.from_numpy(self.npy_list[(_i_% len(usd_list))]).to(self.device)
            features_list.append(feature)
        features = torch.stack(features_list, dim=0)
        self.embeddings = features

        obs_gt = torch.reshape(self.embeddings, (self.num_envs, 128))

        """
        # train feature extractor
        state_obs = self._compute_proprio_observations()
        # get_tactile
        tactile_obs = self._get_tactile()
        obs_now = torch.cat((state_obs, tactile_obs), dim=-1)
        concat_list = [obs_now.clone().detach()]
        for _i_ in range(self.buffer_length):
            concat_list.append(self.buffer.__getitem__(self.buffer_length - _i_ - 1))
        # obs_prev = self.buffer.__getitem__(0)
        obs_concat = torch.cat((concat_list), dim=-1)
        """

        feature_loss = self.feature_extractor.step(
            obs_gt,
            input_obs,
        )


        # log pose loss from CNN training
        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["feature_loss"] = feature_loss[0].mean()

        return obs_gt

    def _compute_proprio_observations(self):
        """Proprioception observations from physics."""
        # print(self.hand_dof_pos.shape, (self.cfg.vel_obs_scale * self.hand_dof_vel).shape, self.in_hand_pos.shape, self.goal_rot.shape, self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3).shape, self.fingertip_rot.view(self.num_envs, self.num_fingertips * 4).shape, self.fingertip_velocities.view(self.num_envs, self.num_fingertips * 6).shape, self.object_pos.shape, self.object_rot.shape,self.object_velocities.shape, self.object_linvel.shape, self.object_angvel.shape, self.actions.shape)
        print('hand_vel', self.cfg.vel_obs_scale * self.hand_dof_vel)
        print('fin_pos', self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3))
        print('fin_rot', self.fingertip_rot.view(self.num_envs, self.num_fingertips * 4))
        print('fin_vel', self.fingertip_velocities.view(self.num_envs, self.num_fingertips * 6))
        print('obj_pos', self.object_pos)
        print('obj_rot', self.object_rot)
        print('obj_vel', self.object_velocities)
        print('obj_linvel', self.object_linvel)
        print('obj_angvel', self.object_angvel)
        obs = torch.cat(
            (
                # hand
                # unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits), # 24
                self.hand_dof_pos,
                self.cfg.vel_obs_scale * self.hand_dof_vel, # 24
                # goal
                # self.in_hand_pos, # 3
                # self.goal_rot, # 4
                # fingertips
                self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3), # 15
                self.fingertip_rot.view(self.num_envs, self.num_fingertips * 4), # 20
                self.fingertip_velocities.view(self.num_envs, self.num_fingertips * 6), # 30
                # object
                self.object_pos, # 3
                self.object_rot, # 4
                self.object_velocities, # 6
                self.object_linvel, # 3
                self.object_angvel, # 3
                # actions
                self.actions, # 24
            ),
            dim=-1,
        )
        return obs

    def _compute_states(self):
        """Asymmetric states for the critic."""
        sim_states = self.compute_full_state()
        state = torch.cat((sim_states, self.embeddings), dim=-1)
        return state

    def _get_observations(self) -> dict:
        # proprioception observations
        state_obs = self._compute_proprio_observations()
        # get_tactile
        tactile_obs = self._get_tactile()
        obs_now = torch.cat((state_obs, tactile_obs), dim=-1)
        self.buffer.append(obs_now)

        just_reset_indices = (self.buffer._num_pushes == 1).nonzero(as_tuple=True)[0]
        if len(just_reset_indices) > 0:
            # その環境だけ、バッファの全履歴を「今のデータ」で上書きする
            # shape: [N_reset, obs_dim] -> [N_reset, MaxLen, obs_dim]
            fill_val = obs_now[just_reset_indices].unsqueeze(1).expand(-1, self.buffer_length, -1)

            self.buffer.buffer[just_reset_indices] = fill_val.clone()

            # 内部カウンタも満タン扱いにしておく（推奨）
            self.buffer._num_pushes[just_reset_indices] = self.buffer_length

        # previous observations
        concat_list = []
        for _i_ in range(self.buffer_length):
            concat_list.append(self.buffer.buffer[:, (self.buffer_length - _i_ - 1), :48])
            concat_list.append(self.buffer.buffer[:, (self.buffer_length - _i_ - 1), -17:])
        # obs_prev = self.buffer.__getitem__(0)
        obs_concat = torch.reshape(torch.cat(concat_list, dim=-1), (self.num_envs, -1)).to(device=self.device)
        # vision observations from CMM
        # image_obs = self._compute_image_observations()
        # obs = torch.cat((state_obs, image_obs), dim=-1)
        # feature_obs = self._get_object_feature()
        feature_obs = self._get_object_feature(obs_concat)
        # print(feature_obs.shape, obs_concat.shape)
        obs_all = torch.cat((obs_now, torch.reshape(feature_obs, (self.num_envs, 128))), dim=-1)
        # asymmetric critic states
        self.fingertip_force_sensors = self.hand.root_physx_view.get_link_incoming_joint_force()[:, self.finger_bodies]
        # state = self._compute_states()
        # self.buffer.append(obs_now)

        observations = {"policy": obs_all.to(device=self.device), "critic": obs_all.to(device=self.device)}
        return observations

    def _get_tactile(self):

        distal_forces = self.distal_sensor.data.net_forces_w[:].clone() #.reshape(self.num_envs, 3 * 5)
        proximal_forces = self.proximal_sensor.data.net_forces_w[:].clone()
        middle_forces = self.middle_sensor.data.net_forces_w[:].clone()
        palm_forces = self.palm_sensor.data.net_forces_w[:].clone()
        metacarpal_forces = self.metacarpal_sensor.data.net_forces_w[:].clone()

        distal_norm = torch.norm(distal_forces, dim=-1)
        proximal_norm = torch.norm(proximal_forces, dim=-1)
        middle_norm = torch.norm(middle_forces, dim=-1)
        palm_norm = torch.norm(palm_forces, dim=-1)
        metacarpal_norm = torch.norm(metacarpal_forces, dim=-1)

        
        if self.dtype == torch.float16:
            distal_norm = (distal_norm > self.binary_threshold).half()
            proximal_norm = (proximal_norm > self.binary_threshold).half()
            middle_norm = (middle_norm > self.binary_threshold).half()
            palm_norm = (palm_norm > self.binary_threshold).half()
            metacarpal_norm = (metacarpal_norm > self.binary_threshold).half()
        else:
            distal_norm = (distal_norm > self.binary_threshold).float()
            proximal_norm = (proximal_norm > self.binary_threshold).float()
            middle_norm = (middle_norm > self.binary_threshold).float()
            palm_norm = (palm_norm > self.binary_threshold).float()
            metacarpal_norm = (metacarpal_norm > self.binary_threshold).float()

        tactile = torch.cat((
            distal_norm,
            proximal_norm,
            middle_norm,
            palm_norm,
            metacarpal_norm
            ), 
            dim=-1
        )

        self.last_tactile = self.tactile
        self.tactile = tactile
        # print(tactile.shape)
        return tactile

    def _get_rewards(self) -> torch.Tensor:
        (
            total_reward,self.reset_goal_buf,
            self.successes[:],
            self.consecutive_successes[:],
        ) = compute_rewards(
            self.reset_buf,
            self.reset_goal_buf,
            self.successes,
            self.consecutive_successes,
            self.max_episode_length,
            self.object_pos,
            self.object_rot,
            self.in_hand_pos,
            self.goal_rot,
            self.cfg.dist_reward_scale,
            self.cfg.rot_reward_scale,
            self.cfg.rot_eps,
            self.actions,
            self.cfg.action_penalty_scale,
            self.cfg.success_tolerance,
            self.cfg.reach_goal_bonus,
            self.cfg.fall_dist,
            self.cfg.fall_penalty,
            self.cfg.av_factor,
            self.fell_off_buf,
            self.cfg.rew_scale_alive,
            self.object_velocities,
            self.buffer.buffer[:, :, :],
            self.cfg.thr_obj_move,
            self.cfg.rew_obj_move
        )

        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["consecutive_successes"] = self.consecutive_successes.mean()

        # reset goals if the goal has been reached
        # goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)
        # if len(goal_env_ids) > 0:
        #     self._reset_target_pose(goal_env_ids)

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()

        # reset when cube has fallen
        self.fell_off_buf = self.object_pos[:, 2] < self.cfg.fall_height

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return self.fell_off_buf, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        # resets articulation and rigid body attributes
        # env_ids = env_ids.to(device=self.device, dtype=torch.long).view(-1)
        super()._reset_idx(env_ids)

        # reset goals
        # self._reset_target_pose(env_ids)

        # reset object
        object_default_state = self.object.data.default_root_state.clone()[env_ids]
        pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device)
        # global object positions
        object_default_state[:, 0:3] = (
            object_default_state[:, 0:3] + self.cfg.reset_position_noise * pos_noise + self.scene.env_origins[env_ids]
        )

        rot_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)  # noise for X and Y rotation
        object_default_state[:, 3:7] = randomize_rotation(
            rot_noise[:, 0], rot_noise[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        )

        object_default_state[:, 7:] = torch.zeros_like(self.object.data.default_root_state[env_ids, 7:])
        self.object.write_root_pose_to_sim(object_default_state[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(object_default_state[:, 7:], env_ids)

        # reset hand
        delta_max = self.hand_dof_upper_limits[env_ids] - self.hand.data.default_joint_pos[env_ids]
        delta_min = self.hand_dof_lower_limits[env_ids] - self.hand.data.default_joint_pos[env_ids]

        dof_pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        rand_delta = delta_min + (delta_max - delta_min) * 0.5 * dof_pos_noise
        dof_pos = self.hand.data.default_joint_pos[env_ids] + self.cfg.reset_dof_pos_noise * rand_delta

        dof_vel_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        dof_vel = self.hand.data.default_joint_vel[env_ids] + self.cfg.reset_dof_vel_noise * dof_vel_noise

        self.prev_targets[env_ids] = dof_pos
        self.cur_targets[env_ids] = dof_pos
        self.hand_dof_targets[env_ids] = dof_pos

        self.hand.set_joint_position_target(dof_pos, env_ids=env_ids)
        self.hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)

        self.successes[env_ids] = 0
        self._compute_intermediate_values()

        self.buffer.reset(env_ids)

    def _reset_target_pose(self, env_ids):
        # reset goal rotation
        rand_floats = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)
        new_rot = randomize_rotation(
            rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        )

        # update goal pose and markers
        self.goal_rot[env_ids] = new_rot
        goal_pos = self.goal_pos + self.scene.env_origins
        self.goal_markers.visualize(self.goal_rot[:,:3])

        self.reset_goal_buf[env_ids] = 0

@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor)
    )

@torch.jit.script
def compute_keypoints(
    pose: torch.Tensor,
    num_keypoints: int = 8,
    size: tuple[float, float, float] = (2 * 0.03, 2 * 0.03, 2 * 0.03),
    out: torch.Tensor | None = None,
):
    """Computes positions of 8 corner keypoints of a cube.

    Args:
        pose: Position and orientation of the center of the cube. Shape is (N, 7)
        num_keypoints: Number of keypoints to compute. Default = 8
        size: Length of X, Y, Z dimensions of cube. Default = [0.06, 0.06, 0.06]
        out: Buffer to store keypoints. If None, a new buffer will be created.
    """
    num_envs = pose.shape[0]
    if out is None:
        out = torch.ones(num_envs, num_keypoints, 3, dtype=torch.float32, device=pose.device)
    else:
        out[:] = 1.0
    for i in range(num_keypoints):
        # which dimensions to negate
        n = [((i >> k) & 1) == 0 for k in range(3)]
        corner_loc = ([(1 if n[k] else -1) * s / 2 for k, s in enumerate(size)],)
        corner = torch.tensor(corner_loc, dtype=torch.float32, device=pose.device) * out[:, i, :]
        # express corner position in the world frame
        out[:, i, :] = pose[:, :3] + quat_apply(pose[:, 3:7], corner)

    return out

@torch.jit.script
def rotation_distance(object_rot, target_rot):
    # Orientation alignment for the cube in hand and goal cube
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    return 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0))  # changed quat convention

@torch.jit.script
def compute_rewards(
    reset_buf: torch.Tensor,
    reset_goal_buf: torch.Tensor,
    successes: torch.Tensor,
    consecutive_successes: torch.Tensor,
    max_episode_length: float,
    object_pos: torch.Tensor,
    object_rot: torch.Tensor,
    target_pos: torch.Tensor,
    target_rot: torch.Tensor,
    dist_reward_scale: float,
    rot_reward_scale: float,
    rot_eps: float,
    actions: torch.Tensor,
    action_penalty_scale: float,
    success_tolerance: float,
    reach_goal_bonus: float,
    fall_dist: float,
    fall_penalty: float,
    av_factor: float,
    fell_off_buf: torch.Tensor,
    rew_scale_alive: float,
    object_velocities: torch.Tensor,
    buffer: torch.Tensor,
    thr_obj_move: float,
    rew_obj_move: float
):
    rot_dist = rotation_distance(object_rot, target_rot)
    rot_rew = 1.0 / (torch.abs(rot_dist) + rot_eps) * rot_reward_scale



    # 1. 生存報酬（落としていない場合の報酬）
    reward_alive = torch.full_like(fell_off_buf, 
                                    rew_scale_alive, 
                                    dtype=torch.float32)
    
    # 2. ドロップペナルティ
    penalty_dropped = torch.full_like(fell_off_buf,
                                        fall_penalty,
                                        dtype=torch.float32)
    
    # 全ステップのobjectの変化（動いていればよし）
    obj_move  = torch.sum(torch.abs(object_velocities - buffer[:, -2, 120:126]), dim=1)
    obj_move_buf = obj_move > thr_obj_move
    reward_move = torch.where(obj_move_buf,
                              torch.full_like(obj_move_buf,
                                    rew_obj_move,
                                    dtype=torch.float32), 0.0)


    reward = torch.where(fell_off_buf, 
                            penalty_dropped, 
                            reward_alive) + reward_move

    # Find out which envs hit the goal and update successes count
    goal_resets = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.ones_like(reset_goal_buf), reset_goal_buf)
    successes = successes + goal_resets

    # Success bonus: orientation is within `success_tolerance` of goal orientation
    # reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)

    # Fall penalty: distance to the goal is larger than a threshold
    # reward = torch.where(goal_dist >= fall_dist, reward + fall_penalty, reward)

    # Check env termination conditions, including maximum success number
    # resets = torch.where(goal_dist >= fall_dist, torch.ones_like(reset_buf), reset_buf)

    # num_resets = torch.sum(resets)
    # finished_cons_successes = torch.sum(successes * resets.float())
    # Check env termination conditions, including maximum success number
    resets = torch.where(fell_off_buf, torch.ones_like(reset_buf), reset_buf)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(
        num_resets > 0,
        av_factor * finished_cons_successes / num_resets + (1.0 - av_factor) * consecutive_successes,
        consecutive_successes,
    )

    return reward, goal_resets, successes, cons_successes

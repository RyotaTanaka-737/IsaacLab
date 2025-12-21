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
from collections import deque
from pxr import Usd, UsdGeom
import omni.usd
from isaaclab.utils.math import quat_from_euler_xyz
# from omni.isaac.lab.utils.math import sample_uniform, randomize_rotation

usd_list = sorted(glob.glob('/home/ubuntu/IsaacLab/asset/mix/train/*.usd', recursive=True))



"""
@configclass
class ShadowHandTactileEnvPlayCfg(ShadowHandTactileEnvCfg):
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=2.0, replicate_physics=False)
    # inference for CNN
    feature_extractor = FeatureExtractorCfg(train=False, load_checkpoint=True)
"""

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
            npy_path = (usd_list[(_i_% len(usd_list))].replace('asset/mix/train/', 'features/mix/train/')).replace('.usd', '.npy')
            self.npy_list.append(np.load(npy_path))

        # --- カリキュラム状態管理 ---
        self.curriculum_level = self.cfg.curr_initial_level
        # 成功判定の履歴バッファ (直近100エピソード)
        self.success_buf = deque(maxlen=100)

        # --- 時間・イベント管理用バッファ ---
        # 各環境が最後にリセットされた絶対時刻
        self._env_reset_times = torch.zeros(self.num_envs, device=self.device)
        # 各環境ごとの「リセットから外乱開始までの遅延時間」
        self._force_trigger_delays = torch.zeros(self.num_envs, dtype=torch.uint8, device=self.device)
        self.active_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.active_drop = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # 実際の初期位置を保存するバッファを作成
        self._object_start_pos = torch.zeros((self.num_envs, 3), device=self.device)

        self.success_rate = 0
        self.lift_count = torch.zeros(self.num_envs, device=self.device)

        self.steps_per_loop = 96

        self.prev_actions = torch.zeros(self.num_envs, 20, device=self.device)
        palm_ids, _ = self.hand.find_bodies(".*palm.*")
        self.palm_link_idx = palm_ids[0] # 通常は1つだけ


    def _setup_scene(self):
        # add hand, in-hand object, and goal object
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        # self.object_stage = RigidObject(self.cfg.object_stage_cfg)
        self.floor = RigidObject(self.cfg.floor_cfg)
        # self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        # get stage
        # self.scene.register("object_stage", self.object_stage)
        '''
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                restitution=0.0,      # 反発係数 (0.0 = 跳ねない, 1.0 = よく跳ねる)
                static_friction=1.0,  # 静止摩擦 (滑りにくくする)
                dynamic_friction=1.0, # 動摩擦
                ),
            ))
        '''
        # clone and replicate (no need to filter for this environment)
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene - we must register to scene to randomize with EventManager
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        self.scene.rigid_objects["floor"] = self.floor
        # self.scene.rigid_objects["object_stage"] = self.object_stage
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

        # self.fingertip_indices, _ = self.hand.find_bodies(".*distal")


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
        if self.cfg.feature_train == True:
            self.extras["log"]["feature_loss"] = feature_loss[0].mean()

        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["curriculum"] = self.curriculum_level

        return obs_gt

    def _compute_proprio_observations(self):
        """Proprioception observations from physics."""
        # print(self.hand_dof_pos.shape, (self.cfg.vel_obs_scale * self.hand_dof_vel).shape, self.in_hand_pos.shape, self.goal_rot.shape, self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3).shape, self.fingertip_rot.view(self.num_envs, self.num_fingertips * 4).shape, self.fingertip_velocities.view(self.num_envs, self.num_fingertips * 6).shape, self.object_pos.shape, self.object_rot.shape,self.object_velocities.shape, self.object_linvel.shape, self.object_angvel.shape, self.actions.shape)
        # print('hand_dof', unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits))
        # print('hand_vel', self.cfg.vel_obs_scale * self.hand_dof_vel)
        # print('fin_pos', self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3))
        # print('fin_rot', self.fingertip_rot.view(self.num_envs, self.num_fingertips * 4))
        # print('fin_vel', self.fingertip_velocities.view(self.num_envs, self.num_fingertips * 6))
        # print('obj_pos', self.object_pos)
        # print('obj_rot', self.object_rot)
        # print('obj_vel', self.object_velocities)
        # print('obj_linvel', self.object_linvel)
        # print('obj_angvel', self.object_angvel)
        obs = torch.cat(
            (
                # hand
                unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits), # 24
                # self.hand_dof_pos,
                torch.clamp(self.cfg.vel_obs_scale * self.hand_dof_vel, -100.0, 100.0), # 24
                # goal
                # self.in_hand_pos, # 3
                self.goal_rot, # 4
                # fingertips
                self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3), # 15
                self.fingertip_rot.view(self.num_envs, self.num_fingertips * 4), # 20
                torch.clamp(self.fingertip_velocities.view(self.num_envs, self.num_fingertips * 6), -100.0, 100.0), # 30
                # object
                self.object_pos, # 3
                self.object_rot, # 4
                self.object_velocities * 100, # 6
                self.object_linvel* 100, # 3
                self.object_angvel* 100, # 3
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
            self.consecutive_successes[:], self.lift_count
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
            self.cfg.rew_obj_move,
            self.curriculum_level,
            self._object_start_pos,
            self.cfg.rew_height,
            self.hand.data.applied_torque,
            self.hand.data.joint_vel,
            self.cfg.penalty_torque_scale,
            self.cfg.penalty_dof_vel_scale,
            self.cfg.total_scale,
            self.fingertip_pos,
            self.tactile,
            self.lift_count,
            self.object_linvel[: ,2],
            self.prev_actions,
            self.hand.data.body_pos_w[:, self.palm_link_idx, :],
            self.object.data.root_pos_w,
        )
        self.prev_actions = self.actions.clone()

        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["consecutive_successes"] = self.consecutive_successes.mean()

        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["successes_rate"] = self.success_rate

        # reset goals if the goal has been reached
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(goal_env_ids) > 0:
            self._reset_target_pose(goal_env_ids)

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()

        # reset when cube has fallen
        goal_dist = torch.norm(self.object_pos - self.in_hand_pos, p=2, dim=-1)
        out_of_reach = goal_dist >= self.cfg.fall_dist

        palm_distance = torch.norm(self.hand.data.body_pos_w[:, self.palm_link_idx, :] - self.object.data.root_pos_w, dim=-1)
        # print(palm_distance)
        is_too_far = palm_distance > 0.2

        # reset when cube has fallen
        # if self.curriculum_level > 0.0:
        # self.fell_off_buf_force = (self.object_pos[:, 2] < self._object_start_pos) & (self.active_mask)
        # self.fell_off_buf = (self.object_pos[:, 2] < self._object_start_pos * (0.8 + self.curriculum_level))
        # self.fell_off_buf = (self.object_pos[:, 2] < 0) & (self.active_drop)
        # self.fell_off_buf = ((self.object_pos[:, 2] < self._object_start_pos[:, 2]) & (self.active_drop)) | is_too_far
        self.fell_off_buf = ((self.object.data.root_pos_w[:, 2] < self._object_start_pos[:, 2]) & (self.active_drop)) | is_too_far
        # print('pos', self.object.data.root_pos_w, self._object_start_pos)
        # print(self.fell_off_buf)
        # else:
        #     self.fell_off_buf = self.active_mask

        if self.cfg.max_consecutive_success > 0:
            # Reset progress (episode length buf) on goal envs if max_consecutive_success > 0
            rot_dist = rotation_distance(self.object_rot, self.goal_rot)
            self.episode_length_buf = torch.where(
                torch.abs(rot_dist) <= self.cfg.success_tolerance,
                torch.zeros_like(self.episode_length_buf),
                self.episode_length_buf,
            )
            max_success_reached = self.successes >= self.cfg.max_consecutive_success

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self.cfg.max_consecutive_success > 0:
            time_out = time_out | max_success_reached

        # max_vel = torch.max(torch.abs(self.hand.data.joint_vel), dim=1)[0] > 20.0
        # time_out = time_out | max_vel


        return self.fell_off_buf, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        # resets articulation and rigid body attributes
        # env_ids = env_ids.to(device=self.device, dtype=torch.long).view(-1)
        super()._reset_idx(env_ids)

        # reset goals
        self._reset_target_pose(env_ids)

        # reset object
        # -----------------------------------------------------
        # 2. データの準備
        # -----------------------------------------------------
        # デフォルト状態をコピー
        object_default_state = self.object.data.default_root_state.clone()[env_ids]
        robot_default_state = self.hand.data.default_root_state.clone()[env_ids]
        robot_default_state[:, 0:3] = robot_default_state[: ,0:3] + self.scene.env_origins[env_ids]
        # パラメータ設定
        floor_z = 0.0
        margin_floor = 0.005  # 床との隙間 (埋まり防止)
        margin_robot = 0.05   # オブジェクトとロボットの隙間 (5cm)
        
        pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device)
        # global object positions
        object_default_state[:, 0:3] = (
            object_default_state[:, 0:3] + self.cfg.reset_position_noise * pos_noise + self.scene.env_origins[env_ids]
        )

        # rot_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)  # noise for X and Y rotation
        # object_default_state[:, 3:7] = randomize_rotation(
        #     rot_noise[:, 0], rot_noise[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        # )
        roll_obj = torch.zeros(len(env_ids), device=self.device)
        pitch_obj = torch.zeros(len(env_ids), device=self.device)

        # Yaw (Z軸回転) だけ 0 ～ 2π (360度) の範囲でランダムに
        yaw_obj = torch.rand(len(env_ids), device=self.device) * 2 * torch.pi

        # オイラー角(roll, pitch, yaw) から クォータニオン(w, x, y, z) に変換
        object_quat = quat_from_euler_xyz(roll_obj, pitch_obj, yaw_obj)
        object_default_state[:, 3:7] = object_quat

        # object_default_state[:, 7:] = torch.zeros_like(self.object.data.default_root_state[env_ids, 7:])
        # self.object.write_root_pose_to_sim(object_default_state[:, :7], env_ids)
        # self.object.write_root_velocity_to_sim(object_default_state[:, 7:], env_ids)
        # self._object_start_pos[env_ids] = object_default_state[:, 2]

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

        self.lift_count[env_ids] = 0.0
    
        
        # -----------------------------------------------------
        # 4. Z座標(高さ)の計算
        # -----------------------------------------------------
        for i, env_id in enumerate(env_ids):
            # Primパスを取得
            prim_path = f"/World/envs/env_{int(env_id)}/object"
            
            # 形状情報を取得
            bottom_offset, obj_height = get_prim_geometry_info(prim_path)
            
            # --- A. オブジェクトのZ ---
            # 式: 床高さ + マージン - (原点から底面のズレ)
            # offsetがマイナスの場合、マイナスを引くのでプラスになり持ち上がります
            obj_target_z = floor_z + margin_floor - bottom_offset
            
            # --- B. ロボットのZ ---
            # 式: オブジェクトのZ(原点) + 底面への距離(戻す) + 全高 + マージン
            # これで「オブジェクトの天面 + マージン」の位置になります
            robot_target_z = obj_target_z + bottom_offset + (obj_height/2) + margin_robot
            
            # 適用
            object_default_state[i, 2] = obj_target_z
            robot_default_state[i, 2] = robot_target_z
            
        self.object.write_root_pose_to_sim(object_default_state[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(object_default_state[:, 7:], env_ids)
        self._object_start_pos[env_ids] = object_default_state[:, :3]
        
        self.hand.write_root_pose_to_sim(robot_default_state[:, :7], env_ids)
        self.hand.write_root_velocity_to_sim(robot_default_state[:, 7:], env_ids)

        if len(env_ids) > 0:
            success_results = (self.successes[env_ids] > 0).float().cpu().tolist()
            self.success_buf.extend(success_results)
        # 2. 成功率計算とレベル調整
        if len(self.success_buf) > 0:
            success_rate = sum(self.success_buf) / len(self.success_buf)

            # レベルアップ
            self.current_iter = self.common_step_counter // self.steps_per_loop
            if (self.current_iter != 0) & ((self.current_iter % 1000) == 0):
                self.curriculum_level += 0.05
            if success_rate > self.cfg.curr_threshold_up:
                self.curriculum_level += self.cfg.curr_step_size
            # レベルダウン (難しすぎる場合)
            # elif success_rate < self.cfg.curr_threshold_down:
                # self.curriculum_level -= self.cfg.curr_step_size

            # クリップ (0.0 ~ 1.0)
            self.curriculum_level = max(0.0, min(1.0, self.curriculum_level))

        self.successes[env_ids] = 0
        self._compute_intermediate_values()

        self.buffer.reset(env_ids)

        # 1. 現在の絶対時刻 (シミュレーション時間) を取得
        current_time = self.common_step_counter * self.step_dt

        # 2. リセット時刻を更新
        self._env_reset_times[env_ids] = current_time

        # 3. 次の外乱イベントまでの遅延時間をランダムに決定
        # min_s 〜 max_s の一様乱数
        # delays = (torch.rand(len(env_ids), device=self.device) * (self.cfg.force_delay_max_s - self.cfg.force_delay_min_s) + self.cfg.force_delay_min_s)
        delays = (torch.rand(len(env_ids), device=self.device) * (self.cfg.force_delay_max_s - self.cfg.force_delay_min_s) + self.cfg.force_delay_min_s)
        self._force_trigger_delays[env_ids] = delays.to(torch.uint8)

        self.active_drop[env_ids] = False
        self.active_mask[env_ids] = False
        self.success_rate = success_rate

        self.prev_actions[env_ids] = 0.0
        default_pos = torch.tensor([0.0, 0.0, -0.05], device=self.device)
        floor_poses = self.scene["floor"].data.root_pose_w[env_ids].clone()

        floor_poses[:, 0:3] = default_pos + self.scene.env_origins[env_ids]

        # バッファ書き換え
        self.scene["floor"].data.root_pos_w[env_ids] = floor_poses[:, 0:3]

        # シミュレータへ適用
        self.scene["floor"].write_root_pose_to_sim(root_pose=floor_poses, env_ids=env_ids)


    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # 1. 外乱の適用 (カリキュラムレベル > 0 の場合のみ計算)
        if self.curriculum_level > 0.0:
            # self._apply_curriculum_forces()
            self._apply_drop_floor()

        self.actions = actions.clone()

    def _apply_drop_floor(self):
        # 現在時刻
        current_time = self.common_step_counter * self.step_dt

        # リセットからの経過時間
        time_since_reset = current_time - self._env_reset_times
        # print(time_since_reset)

        # 外乱を開始する時刻 = 0 + delay
        start_times = self._force_trigger_delays
        # 外乱を終了する時刻 = start + duratioe
        end_times = start_times + self.cfg.force_duration_s

        # 現在時刻が [start, end] の区間に入っている環境を特定
        # self.active_drop = (time_since_reset >= start_times) & (time_since_reset <= end_times)
        self.active_drop = self.episode_length_buf == self._force_trigger_delays

        env_ids_active = self.active_drop.nonzero(as_tuple=False).flatten()
        # print(self.episode_length_buf, self._force_trigger_delays)

        # 対象環境があれば力を加える
        if len(env_ids_active) > 0:
            self._remove_floor(env_ids_active)


    def _apply_curriculum_forces(self):
        # 現在時刻
        current_time = self.common_step_counter * self.step_dt

        # リセットからの経過時間
        time_since_reset = current_time - self._env_reset_times

        # 外乱を開始する時刻 = 0 + delay
        start_times = self._force_trigger_delays
        # 外乱を終了する時刻 = start + duration
        end_times = start_times + self.cfg.force_duration_s

        # 現在時刻が [start, end] の区間に入っている環境を特定
        self.active_mask = (time_since_reset >= start_times) & (time_since_reset <= end_times)
        env_ids_active = self.active_mask.nonzero(as_tuple=False).flatten()

        # 対象環境があれば力を加える
        if len(env_ids_active) > 0:
            # 現在のレベルに応じた力の強さ (Linear Scaling)
            current_mag = self.curriculum_level * self.cfg.force_max_magnitude

            # 上方向 (+Z) の力を作成
            external_forces = torch.zeros((len(env_ids_active), 3), device=self.device)
            external_forces[:, 2] = current_mag

            external_torques = torch.zeros((len(env_ids_active), 3), device=self.device)

            # ロボットのRootリンクに適用
            # ※固定アームの場合は手先リンクのindexを指定する必要があります
            #  self.hand.root_physx_view.apply_forces(forces, indices=env_ids_active)
            root_id = torch.tensor([0], device=self.device)
            self.hand.set_external_force_and_torque(forces=external_forces.unsqueeze(1), torques=external_torques.unsqueeze(1), body_ids=root_id, env_ids=env_ids_active)

    def _reset_target_pose(self, env_ids):
        # reset goal rotation
        rand_floats = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)
        new_rot = randomize_rotation(
            rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        )

        # update goal pose and markers
        self.goal_rot[env_ids] = new_rot
        goal_pos = self.goal_pos + self.scene.env_origins
        # self.goal_markers.visualize(self.goal_rot[:,:3])

        self.reset_goal_buf[env_ids] = 0


    def _remove_floor(self, env_ids):
        # 1. 現在の床の位置を取得 (num_envs, 3)
        floor_pose = self.scene["floor"].data.root_pose_w.clone()

        # 2. Z座標を -100.0 に書き換え (奈落の底へ)
        floor_pose[env_ids, 2] = -100.0

        self.scene["floor"].write_root_pose_to_sim(
        root_pose=floor_pose[env_ids], # 変更する分だけ渡す
        env_ids=env_ids
    )


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
    return 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), min=-1.0,max=1.0))  # changed quat convention

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
    rew_obj_move: float,
    curriculum_level: float,
    object_start_pos: torch.Tensor,
    rew_height: float,
    applied_torque: torch.Tensor,
    joint_vel: torch.Tensor,
    penalty_torque_scale: float,
    penalty_dof_vel_scale: float,
    total_scale: float,
    fingertip_pos: torch.Tensor,
    contact_sensor: torch.Tensor,
    lift_count: torch.Tensor,
    object_vel_z: torch.Tensor,
    prev_actions: torch.Tensor,
    palm_pos: torch.Tensor,
    obj_pos_w: torch.Tensor,
):


    # Check env termination conditions, including maximum success number
    # resets = torch.where(goal_dist >= fall_dist, torch.ones_like(reset_buf), reset_buf)


    # 1. 生存報酬（落としていない場合の報酬）
    reward_alive = torch.full_like(fell_off_buf, 
                                    # rew_scale_alive, 
                                    0.0,
                                    dtype=torch.float32)
    
    # 2. ドロップペナルティ
    penalty_dropped = torch.full_like(fell_off_buf,
                                        fall_penalty,
                                        dtype=torch.float32)

    # dist_from_start_z = torch.clamp((object_pos[:, 2] - object_start_pos[:, 2]), min=0)
    dist_from_start_z = torch.clamp((obj_pos_w[:, 2] - object_start_pos[:, 2]), min=0)
    # height_buf = dist_from_start_z > 0.05
    # height_buf = object_pos[:, 2] > object_start_pos[:, 2]
    height_buf = obj_pos_w[:, 2] > object_start_pos[:, 2]
    is_contact = (torch.sum(contact_sensor, dim=1) > 1).float()
    lift_count = torch.where(height_buf, lift_count + 1.0, 0.0)
    reward_height = rew_height * dist_from_start_z * is_contact * lift_count
    
    action_penalty = torch.sum(actions**2, dim=-1) * action_penalty_scale
    penalty_torque = torch.sum(torch.square(applied_torque), dim=1) * penalty_torque_scale
    # penalty_dof_vel = torch.sum(torch.square(joint_vel), dim=1) * penalty_dof_vel_scale
    joint_vel = torch.clamp(joint_vel, min=-100.0, max=100.0)

    # 2. 丸めた値を使ってペナルティを計算する
    penalty_dof_vel = torch.sum(torch.square(joint_vel), dim=1) * penalty_dof_vel_scale
    # 3. 念のため結果もチェック (scaleが NaN だったりする場合の対策)
    if not torch.isfinite(penalty_dof_vel).all():
        penalty_dof_vel = torch.nan_to_num(penalty_dof_vel, nan=0.0, posinf=100.0, neginf=-100.0)

    tip_dists = torch.norm(fingertip_pos - object_pos.unsqueeze(1), dim=-1)
    mean_tip_dist = torch.mean(tip_dists, dim=1)
    # sigma_tip = 0.5
    # reward_reach = torch.exp(-torch.square(mean_tip_dist) / sigma_tip)
    alpha = 100.0 # 係数
    reward_reach = 1.0 / (1.0 + alpha * torch.square(mean_tip_dist))

    # contact_sensor
    reward_contact = torch.sum(contact_sensor, dim=1) * 0.1

    sigma = 0.5
    rot_dist = rotation_distance(object_rot, target_rot)
    # rot_rew = (1.0 / (torch.abs(rot_dist) + rot_eps))* rot_reward_scale
    # rot_rew = torch.exp(-torch.square(rot_dist) / sigma)
    rot_rew = 1.0 / (1.0 + torch.square(rot_dist))
    rot_rew = rot_rew + (rot_rew * reward_height * 5.0)

    # Find out which envs hit the goal and update successes count
    # goal_resets = torch.where((torch.abs(rot_dist) <= success_tolerance) | ((dist_from_start_z > 0.05) & ((torch.sum(contact_sensor, dim=1) > 1)) & (lift_count > 10)), torch.ones_like(reset_goal_buf), reset_goal_buf)
    # goal_resets = torch.where(((dist_from_start_z > 0.05) & ((torch.sum(contact_sensor, dim=1) > 1)) & (lift_count > 10)), torch.ones_like(reset_goal_buf), reset_goal_buf)
    goal_resets = torch.where((torch.abs(rot_dist) <= success_tolerance) & ((dist_from_start_z > 0.05) & ((torch.sum(contact_sensor, dim=1) > 1)) & (lift_count > 10)), torch.ones_like(reset_goal_buf), reset_goal_buf)
    successes = successes + goal_resets

    # Success bonus: orientation is within `success_tolerance` of goal orientation
    reward_goal = torch.where(goal_resets == 1, reach_goal_bonus, 0.0)


    dropping_vel = torch.minimum(object_vel_z, torch.tensor(0.0, device=object_vel_z.device))

    # 高さが 5cm 以上ある場合のみ、落下ペナルティを適用
    is_lifted = dist_from_start_z > 0.05
    dropping_vel_penalty = dropping_vel * is_lifted.float()

    action_diff = actions - prev_actions
    rate_penalty = torch.sum(action_diff**2, dim=-1) * -0.001

    # palm_distance = torch.norm(palm_pos - object_pos, dim=-1)
    palm_distance = torch.norm(palm_pos - obj_pos_w, dim=-1)
    beta = 100.0
    reward_reach_palm = 1.0 / (1.0 + beta * torch.square(palm_distance))

    combined_reward = (
        reward_goal +
        rot_rew +
        reward_height +
        reward_reach +
        reward_contact +
        action_penalty +
        penalty_torque +
        penalty_dof_vel +
        dropping_vel_penalty +
        rate_penalty +
        reward_reach_palm
        # + reward_alive # 必要なら正の値にして足す
    )
    # print(reward_goal, rot_rew, reward_height, reward_reach, reward_contact, reward_reach_palm)
    debug_info = {
            "Goal": reward_goal,
            "Reach": reward_reach,
            "Rot": rot_rew,
            "Height": reward_height,
            "Action": action_penalty, # もしあれば
            "VelPenalty": penalty_dof_vel, # もしあれば
            "TorquePenalty": penalty_torque, # もしあれば
            "drop": penalty_dropped
            }
    for name, val in debug_info.items():
        if not torch.isfinite(val).all():
            print(f"!!! EXPLOSION DETECTED in [{name}] !!!")
            print(f"    Min: {val.min().item()}, Max: {val.max().item()}")
            print(f"    NaN count: {torch.isnan(val).sum().item()}")
            print(f"    Inf count: {torch.isinf(val).sum().item()}")

    reward = torch.where(fell_off_buf, 
                            penalty_dropped, 
                            combined_reward)
    
    if not torch.isfinite(reward).all():
        # デバッグ用に原因を表示（一度だけ表示など工夫しても良い）
        print("Reward explosion detected! Clipping...")
        pass

    reward = reward * total_scale

    resets = torch.where(fell_off_buf, torch.ones_like(reset_buf), reset_buf)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(
        num_resets > 0,
        av_factor * finished_cons_successes / num_resets + (1.0 - av_factor) * consecutive_successes,
        consecutive_successes,
    )

    return reward, goal_resets, successes, cons_successes, lift_count
    
def get_prim_geometry_info(prim_path: str):
    """
    指定されたPrimの幾何情報を取得します。
    
    Returns:
        bottom_offset (float): 原点(0,0,0)から底面(Min Z)までの距離。通常は負の値か0.0。
        height (float): オブジェクトのZ軸方向の全長。
    """
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        # エラー時のデフォルト値 (適宜調整)
        return 0.0, 0.1

    # Collision (proxy) も含めたバウンディングボックスを計算
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.proxy])
    bound = bbox_cache.ComputeWorldBound(prim)
    range3d = bound.GetRange()
    
    # 1. 高さ (Size Z)
    height = range3d.GetSize()[2]
    
    # 2. 底面のWorld Z座標
    world_bottom_z = range3d.GetMin()[2]

    # 3. 原点(Pivot)のWorld Z座標
    xformable = UsdGeom.Xformable(prim)
    world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    world_pivot_z = world_transform.ExtractTranslation()[2]
    
    # 4. オフセット (底面 - 原点)
    # 例: 中心原点なら -0.05, 底面原点なら 0.0
    bottom_offset = world_bottom_z - world_pivot_z
    
    return bottom_offset, height

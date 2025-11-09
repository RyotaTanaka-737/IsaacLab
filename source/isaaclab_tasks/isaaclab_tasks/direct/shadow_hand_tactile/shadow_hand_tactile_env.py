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
from isaaclab.utils.buffer import CircularBuffer

from isaaclab_tasks.direct.inhand_manipulation.inhand_manipulation_env import InHandManipulationEnv, unscale

from .feature_extractor import FeatureExtractor, FeatureExtractorCfg
from .shadow_hand_tactile_env_cfg import ShadowHandEnvCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane



@configclass
class ShadowHandTactileEnvCfg(ShadowHandEnvCfg):
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1225, env_spacing=2.0, replicate_physics=True)
    """
    # camera
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0, -0.35, 1.0), rot=(0.7071, 0.0, 0.7071, 0.0), convention="world"),
        data_types=["rgb", "depth", "semantic_segmentation"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=120,
        height=120,
    )
    """
    feature_extractor = FeatureExtractorCfg()

    # env
    observation_space = 164 + 27  # state observation + vision CNN embedding
    state_space = 187 + 27  # asymettric states + vision CNN embedding

    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
    marker_cfg.prim_path = "/Visuals/ContactCfg"
    distal_contact_cfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/robot0_.*distal",
        update_period=0.0,
        history_length=1,

    )
    middle_contact_cfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/robot0_.*middle",
        update_period=0.0,
        history_length=1,
    )
    proximal_contact_cfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/robot0_.*proximal",
        update_period=0.0,
        history_length=1,

    )
    palm_contact_cfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/robot0_palm",
        update_period=0.0,
        history_length=1,
    )
    metacarpal_contact_cfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/robot0_lfmetacarpal",
        update_period=0.0,
        history_length=1,
    )


@configclass
class ShadowHandTactileEnvPlayCfg(ShadowHandTactileEnvCfg):
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=2.0, replicate_physics=True)
    # inference for CNN
    feature_extractor = FeatureExtractorCfg(train=False, load_checkpoint=True)


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

        self.tactile = torch.zeros((self.num_envs, self.num_tactile_observations), device=self.device)
        self.last_tactile = torch.zeros((self.num_envs, self.num_tactile_observations), device=self.device)

        self.extras["log"] = {
            "tactile_penalty": None,
            "success_reward": None,
            "action_penalty": None,
            "fall_penalty": None,
            "object_height": None,
            "object_z_linvel": None,
            "object_z_angvel": None,
            "sum_forces": None,
            "total_rotations": None,
            "cumulative_rotations": None,
            "ball_1_vel": None,
            "ball_2_vel": None,
            "ball_dist": None,
            "dist_penalty": None,
            "tactile_reward": None,
            "transition_reward": None,
            "bounce_reward": None,
            "air_reward": None
        }

        self.buffer = CircularBuffer(size=5, shape=(self.num_envs, self.num_tactile_observations), device=self.device)

    def _setup_scene(self):
        # add hand, in-hand object, and goal object
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        # self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        # get stage
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        stage = get_current_stage()
        # add semantics for in-hand cube
        prim = stage.GetPrimAtPath("/World/envs/env_0/object")
        sem = Semantics.SemanticsAPI.Apply(prim, "Semantics")
        sem.CreateSemanticTypeAttr()
        sem.CreateSemanticDataAttr()
        sem.GetSemanticTypeAttr().Set("class")
        sem.GetSemanticDataAttr().Set("cube")
        # clone and replicate (no need to filter for this environment)
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene - we must register to scene to randomize with EventManager
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.object
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
        self.extras["log"]["pose_loss"] = pose_loss

        return obs

    def _compute_proprio_observations(self):
        """Proprioception observations from physics."""
        obs = torch.cat(
            (
                # hand
                unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
                self.cfg.vel_obs_scale * self.hand_dof_vel,
                # goal
                self.in_hand_pos,
                self.goal_rot,
                # fingertips
                self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                self.fingertip_rot.view(self.num_envs, self.num_fingertips * 4),
                self.fingertip_velocities.view(self.num_envs, self.num_fingertips * 6),
                # actions
                self.actions,
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
        obs = torch.cat((state_obs, tactile_obs), dim=-1)
        # vision observations from CMM
        # image_obs = self._compute_image_observations()
        # obs = torch.cat((state_obs, image_obs), dim=-1)
        # previous observations
        prev_obs = self.buffer.__getitem__(0)
        obs_concat = torch.cat((obs, prev_obs), dim=-1)
        self.buffer.append(obs)
        # asymmetric critic states
        self.fingertip_force_sensors = self.hand.root_physx_view.get_link_incoming_joint_force()[:, self.finger_bodies]
        state = self._compute_states()

        observations = {"policy": obs, "critic": state}
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
        return tactile


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

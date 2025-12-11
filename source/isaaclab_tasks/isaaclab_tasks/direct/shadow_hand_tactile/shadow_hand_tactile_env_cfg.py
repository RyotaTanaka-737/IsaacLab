# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab_assets.robots.shadow_hand import SHADOW_HAND_CFG

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from .feature_extractor import FeatureExtractor, FeatureExtractorCfg
import glob


@configclass
class EventCfg:
    """Configuration for randomization."""

    # -- robot
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        min_step_count_between_reset=720,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        },
    )
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.75, 1.5),
            "damping_distribution_params": (0.3, 3.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    robot_joint_pos_limits = EventTerm(
        func=mdp.randomize_joint_parameters,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "lower_limit_distribution_params": (0.00, 0.01),
            "upper_limit_distribution_params": (0.00, 0.01),
            "operation": "add",
            "distribution": "gaussian",
        },
    )
    robot_tendon_properties = EventTerm(
        func=mdp.randomize_fixed_tendon_parameters,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", fixed_tendon_names=".*"),
            "stiffness_distribution_params": (0.75, 1.5),
            "damping_distribution_params": (0.3, 3.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    # -- object
    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        },
    )
    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    reset_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="interval",
        is_global_time=True,
        interval_range_s=(36.0, 36.0),  # time_s = num_steps * (decimation * dt)
        params={
            "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.4]),
            "operation": "add",
            "distribution": "gaussian",
        },
    )
    ''' 
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.5, 1), "yaw": (10, 10), "pitch": (10, 10), "roll": (-10, 10)},
                "asset_cfg": SceneEntityCfg("robot")},
    )
    '''
    down_stage_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="interval",
        interval_range_s=(2.0, 2.5),
        params={
            "pose_range":{"z": (-1.0, 0.0)},
            "asset_cfg": SceneEntityCfg("object_stage"),
        },
    )
    '''
    reset_object_z_events =  EventTerm(
        func=mdp.spawn_object_under_palm_down,  # 自作関数を指定
        mode="reset",                  # "reset": エピソードリセット時に実行
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "asset_robot_cfg": SceneEntityCfg("robot"),
        },
    )

    start_object_z_events =  EventTerm(
        func=mdp.spawn_object_under_palm_down,  # 自作関数を指定
        mode="startup",                  # "reset": エピソードリセット時に実行
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "asset_robot_cfg": SceneEntityCfg("robot"),
        },
    )
    '''
    # -- scene
    reset_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="interval",
        is_global_time=True,
        interval_range_s=(36.0, 36.0),  # time_s = num_steps * (decimation * dt)
        params={
            "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.4]),
            "operation": "add",
            "distribution": "gaussian",
        },
    )

@configclass
class ShadowHandTactileEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    episode_length_s = 10.0
    action_space = 20
    observation_space = 157  # (full)
    state_space = 0
    asymmetric_obs = False
    obs_type = "full"

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=12,
            max_velocity_iteration_count=1,
            bounce_threshold_velocity=0.2,
            gpu_found_lost_pairs_capacity=8 * 1024 * 1024,
            gpu_temp_buffer_capacity=16 * 1024 * 1024,
        ),
    )
    # robot
    robot_cfg: ArticulationCfg = SHADOW_HAND_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 2.0),
            rot=(1.0, 1.0, 0.0, 0.0),
            joint_pos={".*": 0.0},
        )
    )
    robot_cfg.spawn.activate_contact_sensors = True
    actuated_joint_names = [
        "robot0_WRJ1",
        "robot0_WRJ0",
        "robot0_FFJ3",
        "robot0_FFJ2",
        "robot0_FFJ1",
        "robot0_MFJ3",
        "robot0_MFJ2",
        "robot0_MFJ1",
        "robot0_RFJ3",
        "robot0_RFJ2",
        "robot0_RFJ1",
        "robot0_LFJ4",
        "robot0_LFJ3",
        "robot0_LFJ2",
        "robot0_LFJ1",
        "robot0_THJ4",
        "robot0_THJ3",
        "robot0_THJ2",
        "robot0_THJ1",
        "robot0_THJ0",
    ]
    fingertip_body_names = [
        "robot0_ffdistal",
        "robot0_mfdistal",
        "robot0_rfdistal",
        "robot0_lfdistal",
        "robot0_thdistal",
    ]
    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
    marker_cfg.prim_path = "/Visuals/ContactCfg"
    """
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
    """
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=128, env_spacing=2.0, replicate_physics=False)
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
    feature_extractor = FeatureExtractorCfg(train=True, load_checkpoint=False)

    # env
    observation_space_ = 152 + 17 + 128  # state observation + tactile + PointCloud embedding
    state_space = 152 + 17 + 128  # asymettric states + vision CNN embedding
    num_observations: int = observation_space_

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

    usd_list = sorted(glob.glob('/home/ubuntu/IsaacLab/asset/mix/train/*.usd', recursive=True))[:128]
    # in-hand object
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.MultiUsdFileCfg(
            usd_path=usd_list,
            random_choice=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=567.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.39, 1.9), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # goal object
    '''
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.MultiUsdFileCfg(
                usd_path=usd_list,
                random_choice=False,
                scale=(1.0, 1.0, 1.0),
            )
        },
    )
    '''
    '''
    object_stage_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/stageobject",
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 1.0, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=True,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
                ),
            mass_props=sim_utils.MassPropertiesCfg(mass=10000.0),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True  # 必須
                ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.0, 0.8)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.45), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    '''
    # scene
    # scene: InteractiveSceneCfg = InteractiveSceneCfg(
    #     num_envs=8192, env_spacing=0.75, replicate_physics=False, clone_in_fabric=True
    # )

    # --- カリキュラム(ACL)設定 ---
    curr_threshold_up: float = 0.8     # レベルを上げる成功率
    curr_threshold_down: float = 0.6   # レベルを下げる成功率
    curr_step_size: float = 0.05       # 1回の更新でのレベル変動幅
    curr_initial_level: float = 0.0    # 初期レベル

    # --- 外乱 (Upward Force) 設定 ---
    # レベル1.0 (最大) 時の力 [N]
    force_max_magnitude: float = 200.0
    # 外乱の継続時間 [秒]
    force_duration_s: float = 0.5
    # リセット後、外乱が発生するまでのランダム遅延範囲 [秒]
    force_delay_min_s: float = 1.0
    force_delay_max_s: float = 4.0

    # --- 報酬スケーリング設定 ---
    # 回転タスクの報酬 (基本)
    rew_scale_rotate: float = 1.0
    rew_height: float = 1.0

    # reset
    reset_position_noise = 0.01  # range of position at reset
    reset_dof_pos_noise = 0.2  # range of dof pos at reset
    reset_dof_vel_noise = 0.0  # range of dof vel at reset
    # reward scales
    fall_height = 1.0
    dist_reward_scale = -10.0
    rot_reward_scale = 0.010
    rot_eps = 0.1
    action_penalty_scale = -0.02
    reach_goal_bonus = 10
    rew_scale_alive = 0.5
    fall_penalty = -50.0
    fall_dist = 0.24
    fall_height = 1.0
    thr_obj_move = 0.1
    obj_move = 0.1
    rew_obj_move = 0.1
    vel_obs_scale = 0.2
    success_tolerance = 0.1
    max_consecutive_success = 0
    av_factor = 0.1
    act_moving_average = 1.0
    force_torque_obs_scale = 10.0
    penalty_torque_scale = -0.0005
    penalty_dof_vel_scale = -0.001
    total_scale = 0.05

    feature_train = True

    def __post_init__(self):
        super().__post_init__()

        # ここで物理バッファを強制的に増やします (32MB)
        # エラーで要求されているのは0.2MB程度ですが、足りないとまた止まるので大きく取ります
        self.sim.physx.gpu_patch_buffer_capacity = 32 * 1024 * 1024
        self.sim.physx.gpu_found_lost_pairs_capacity = 32 * 1024 * 1024

        # 念のため他も増やしておくと安心です
        self.sim.physx.gpu_heap_capacity = 64 * 1024 * 1024

        self.sim.physx.gpu_max_rigid_patch_count = 163840 * 10  # 例: 約160万

        # もし Contact Count (接触点数) のエラーも出るようなら以下も増やします
        self.sim.physx.gpu_max_rigid_contact_count = 524288 * 10 # 例: 約500万


@configclass
class ShadowHandTactileEnvPlayCfg(ShadowHandTactileEnvCfg):
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=2.0, replicate_physics=False)
    # inference for CNN
    feature_extractor = FeatureExtractorCfg(train=False, load_checkpoint=True)
    feature_train = False


import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from humanoid_dreamwaq.assets.g1 import UNITREE_G1_29DOF_CFG as ROBOT_CFG
from humanoid_dreamwaq.locomotion import mdp


WALKING_EASY_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=5,
    num_cols=21,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 0.5),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.7),
        "wave": terrain_gen.HfWaveTerrainCfg(
            proportion=0.3, amplitude_range=[0.05, 0.2], num_waves=3, border_width=0.5
        ),
    },
)


STAIR_EASY_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=5,
    num_cols=21,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 0.5),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.4),
        "stairs_up": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.5,
            step_height_range=[0.05, 0.12],
            step_width=0.45,
            platform_width=3.0,
            border_width=0.5,
        ),
        "stairs_down": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=[0.05, 0.12],
            step_width=0.45,
            platform_width=3.0,
            border_width=0.5,
        ),
    },
)


STAIR_MED_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=7,
    num_cols=21,
    difficulty_range=(0.0, 0.8),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.30),
        "wave": terrain_gen.HfWaveTerrainCfg(
            proportion=0.20, amplitude_range=[0.05, 0.2], num_waves=3, border_width=0.5
        ),
        "stairs_up": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=[0.05, 0.12],
            step_width=0.45,
            platform_width=3.0,
            border_width=0.5,
        ),
        "stairs_down": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=[0.05, 0.12],
            step_width=0.45,
            platform_width=3.0,
            border_width=0.5,
        ),
    },
)


BLIND_COMPLEX_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=7,
    num_cols=21,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 0.75),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.10),
        "wave": terrain_gen.HfWaveTerrainCfg(
            proportion=0.15,
            amplitude_range=[0.10, 0.25],
            num_waves=3,
            border_width=0.5,
        ),
        "rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.15,
            noise_range=(0.02, 0.08),
            noise_step=0.02,
            border_width=0.25,
        ),
        "stairs_up": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.30,
            step_height_range=[0.05, 0.20],
            step_width=0.35,
            platform_width=2.0,
            border_width=0.5,
        ),
        "stairs_down": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.30,
            step_height_range=[0.05, 0.20],
            step_width=0.35,
            platform_width=2.0,
            border_width=0.5,
        ),
    },
)


DEPLOY_STABLE_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=7,
    num_cols=21,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 0.70),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.15),
        "wave": terrain_gen.HfWaveTerrainCfg(
            proportion=0.20,
            amplitude_range=[0.05, 0.20],
            num_waves=3,
            border_width=0.5,
        ),
        "rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.15,
            noise_range=(0.02, 0.06),
            noise_step=0.02,
            border_width=0.25,
        ),
        "stairs_up": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.30,
            step_height_range=[0.05, 0.16],
            step_width=0.35,
            platform_width=2.5,
            border_width=0.5,
        ),
        "stairs_down": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.20,
            step_height_range=[0.05, 0.16],
            step_width=0.35,
            platform_width=2.5,
            border_width=0.5,
        ),
    },
)

def set_terrain_preset(env_cfg: ManagerBasedRLEnvCfg, name: str, *, for_play: bool = False):
    """Copy a terrain preset while retaining the playback grid size."""
    presets = {
        "walk": WALKING_EASY_CFG,
        "stairs": STAIR_EASY_CFG,
        "stairs-medium": STAIR_MED_CFG,
        "complex": BLIND_COMPLEX_CFG,
        "stable": DEPLOY_STABLE_CFG,
    }
    if name not in presets:
        raise ValueError(f"Unknown terrain preset {name!r}. Choose from {', '.join(presets)}.")

    terrain = env_cfg.scene.terrain
    previous = terrain.terrain_generator
    generator = presets[name].copy()
    generator.curriculum = previous.curriculum
    if for_play:
        generator.num_rows = previous.num_rows
        generator.num_cols = previous.num_cols
    terrain.terrain_generator = generator
    terrain.max_init_terrain_level = generator.num_rows - 1


# Starting at the bottom of the inverted stairs makes outward travel an ascent.
PURE_UPSTAIRS_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=7,
    num_cols=21,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 0.70),
    use_cache=False,
    sub_terrains={
        "stairs_down": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=1.0,
            step_height_range=[0.05, 0.16],
            step_width=0.35,
            platform_width=2.5,
            border_width=0.5,
        ),
    },
)

COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=9,
    num_cols=21,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    color_scheme="height",
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.3,
        ),
        "wave": terrain_gen.HfWaveTerrainCfg(
            proportion = 0.3,
            amplitude_range = [0.05,0.3],
            num_waves = 3,
            border_width=0.5,
        ),
        "stairs_up": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion = 0.15,
            step_height_range = [0.05,0.2],
            step_width = 0.3,
            platform_width = 1.0,
            border_width = 0.5,
        ),
        "stairs_down": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion = 0.15,
            step_height_range = [0.05,0.2],
            step_width = 0.3,
            platform_width = 1.0,
            border_width = 0.5,
        ),
    },
)


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """G1 articulation, terrain, and simulation sensors."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=STAIR_EASY_CFG,
        max_init_terrain_level=STAIR_EASY_CFG.num_rows - 1,

        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),

        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),

        debug_vis=False,
    )

    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # The height scan is used by the critic and terrain-dependent rewards.
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        update_period = 0.02,
    )

    force_detect = ContactSensorCfg(
        max_contact_data_count_per_prim = 2,
        prim_path="{ENV_REGEX_NS}/Robot/.*_ankle_roll_link",
        history_length=3,
        debug_vis=False,
        filter_prim_paths_expr = ["/World/ground"],
    )

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    """Configuration for events."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (0, 0),
            "torque_range": (0, 0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )

    randomize_foot_material = EventTerm(
    func=mdp.randomize_rigid_body_material,
    mode="reset",
    params={
        "asset_cfg": SceneEntityCfg(
            "robot",
            body_names=".*ankle_roll.*"
        ),
        "static_friction_range": (0.3, 1.5),
        "dynamic_friction_range": (0.3, 1.5),
        "restitution_range": (0.0, 0.1),
        "num_buckets": 64,
        },
    )

    randomize_external_force = EventTerm(
        func=mdp.apply_random_body_force,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "force_range": (-20.0,20.0),
        }
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 5.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.2, 0.6), lin_vel_y=(-0.06, 0.06), ang_vel_z=(-0.08, 0.08)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.2, 0.9), lin_vel_y=(-0.10, 0.10), ang_vel_z=(-0.12, 0.12)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Proprioceptive history consumed by the actor."""

        # Observation term order defines the input layout of saved checkpoints.
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2), clip=(-5.0, 5.0)
        )
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05),clip=(-1.05, 1.05), )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01),clip=(-3.15, 3.15),)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5),clip=(-5.0, 5.0), )
        last_action = ObsTerm(func=mdp.last_action, clip=(-20.0, 20.0))

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Training observations with simulated velocity, terrain heights, and external force."""
        # CENet uses the first three critic values as its velocity target.
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action, clip=(-20.0, 20.0))

        height_scanner = ObsTerm(
            func=mdp.height_scan_safe,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise = Unoise(n_min=-0.05,n_max=0.05),
            clip=(-1.0, 1.5),
        )
        external_force = ObsTerm(func=mdp.applied_external_force)

        def __post_init__(self):
            self.history_length = 1

    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:

    """Velocity tracking, stability, and foot placement rewards."""
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    alive = RewTerm(
        func=mdp.is_alive,
        weight=0.15,
    )

    base_linear_velocity = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-2.0,
    )

    base_angular_velocity = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.05,
    )

    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-2.0,
    )

    base_height = RewTerm(
    func=mdp.base_height_above_terrain,
    weight=-4.0,
    params={
        "target_height": 0.78,
        "asset_cfg": SceneEntityCfg("robot"),
        "feet_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.001,
    )
    joint_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
    )
    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.05,
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-5.0,
    )
    energy = RewTerm(
        func=mdp.energy,
        weight=-2e-5,
    )
    stand_still = RewTerm(
        func=mdp.stand_still,
        weight=-1.0,
    )
    feet_contact_without_cmd = RewTerm(
        func=mdp.feet_contact_without_cmd,
        weight=1.0,
        params={
            "sensor_cfg" : SceneEntityCfg(
                "contact_forces",
                body_names=".*ankle_roll.*",
            )
        }
    )

    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[
            ".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*"])},
    )
    joint_deviation_waists = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist.*"])},
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[
            ".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )

    gait = RewTerm(
        func=mdp.feet_gait,
        weight=0.4,
        params={"period": 0.8, "offset": [0.0, 0.5], "threshold": 0.55,
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                )},
    )
    air_time_variance = RewTerm(
        func=mdp.air_time_variance_penalty,
        weight=-0.8,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
            ),
            "max_time": 0.5,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.3,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"])},
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg(
            "contact_forces",
            body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
        )},
    )
    feet_contact_orientation = RewTerm(
        func=mdp.feet_contact_orientation_l2,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
            ),
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
            ),
            "command_name": "base_velocity",
        },
    )

    foot_clearance_stance_based = RewTerm(
        func=mdp.foot_target_height_from_stance,
        weight=1.0,  # The term returns a nonpositive clearance score.
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
            "target_clearance": 0.12,
            "stair_height_thresh": 0.04,
            "command_name": "base_velocity",
        },
    )

    feet_too_near = RewTerm(
        func=mdp.feet_too_near,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
            "threshold":0.2,
        }
    )

    feet_touchdown_sagittal_gap = RewTerm(
        func=mdp.feet_touchdown_sagittal_gap,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
            "target_gap": 0.20,
            "command_name": "base_velocity",
            "dynamic_target_gap": True,
            "step_interval": 0.4,
            "target_gap_scale": 1.0,
            "min_target_gap": 0.06,
        },
    )

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.8,
        params={"threshold": 1,
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"])},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(
        func=mdp.root_height_below_feet,
        params={
            "minimum_height": 0.25,
            "asset_cfg": SceneEntityCfg("robot"),
            "feet_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
            ),
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
            ),
        },
    )
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    scene: RobotSceneCfg = RobotSceneCfg(num_envs=1024, env_spacing=2.5)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""

        self.decimation = 4
        self.episode_length_s = 20.0

        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.force_detect.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 7
        self.scene.terrain.terrain_generator.num_cols = 10
        self.scene.terrain.max_init_terrain_level = 6
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.curriculum = None
        self.commands.base_velocity.ranges = mdp.UniformLevelVelocityCommandCfg.Ranges(
        lin_vel_x=(0.3, 0.9),
        lin_vel_y=(-0.10, 0.10),
        ang_vel_z=(-0.12, 0.12),
        )

@configclass
class RobotSymEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.feet_touchdown_sagittal_gap.weight = -2.0
        self.rewards.air_time_variance.weight = -0.4


@configclass
class RobotSymPlayEnvCfg(RobotPlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.feet_touchdown_sagittal_gap.weight = -2.0
        self.rewards.air_time_variance.weight = -0.4


@configclass
class PureUpstairsRewardsCfg(RewardsCfg):
    feet_sagittal_separation = RewTerm(
        func=mdp.feet_contact_sagittal_separation,
        weight=-3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
            "target_gap": 0.18,
        },
    )


@configclass
class RobotPureUpstairsEnvCfg(RobotEnvCfg):
    rewards: PureUpstairsRewardsCfg = PureUpstairsRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = PURE_UPSTAIRS_CFG.copy()
        self.scene.terrain.max_init_terrain_level = PURE_UPSTAIRS_CFG.num_rows - 1
        self.scene.terrain.terrain_generator.curriculum = True
        self.commands.base_velocity.ranges = mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.2, 0.6),
            lin_vel_y=(-0.04, 0.04),
            ang_vel_z=(-0.06, 0.06),
        )
        self.commands.base_velocity.limit_ranges = mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.2, 0.8),
            lin_vel_y=(-0.06, 0.06),
            ang_vel_z=(-0.08, 0.08),
        )
        self.rewards.feet_touchdown_sagittal_gap.weight = -4.0
        self.rewards.air_time_variance.weight = -0.5


@configclass
class RobotPureUpstairsPlayEnvCfg(RobotPureUpstairsEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 7
        self.scene.terrain.terrain_generator.num_cols = 10
        self.scene.terrain.max_init_terrain_level = 6
        self.curriculum = None
        self.commands.base_velocity.ranges = mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.3, 0.8),
            lin_vel_y=(-0.06, 0.06),
            ang_vel_z=(-0.08, 0.08),
        )

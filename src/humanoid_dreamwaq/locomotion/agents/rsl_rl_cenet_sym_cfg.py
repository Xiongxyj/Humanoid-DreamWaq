from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlSymmetryCfg

from humanoid_dreamwaq.locomotion.mdp.symmetry import g1

from .rsl_rl_cenet_cfg import CENetOnPolicyRunner, CENetPpoAlgorithm


@configclass
class CENetSymmetryPpoAlgorithm(CENetPpoAlgorithm):
    symmetry_cfg = RslRlSymmetryCfg(
        use_data_augmentation=False,
        use_mirror_loss=True,
        data_augmentation_func=g1.compute_symmetric_states,
        mirror_loss_coeff=0.01,
    )


@configclass
class CENetSymmetryOnPolicyRunner(CENetOnPolicyRunner):
    experiment_name = "unitree_g1_29dof_velocity_ce_sym"
    algorithm = CENetSymmetryPpoAlgorithm()


@configclass
class CENetStairSymmetryPpoAlgorithm(CENetPpoAlgorithm):
    symmetry_cfg = RslRlSymmetryCfg(
        use_data_augmentation=False,
        use_mirror_loss=True,
        data_augmentation_func=g1.compute_symmetric_states,
        mirror_loss_coeff=0.05,
    )


@configclass
class CENetStairSymmetryOnPolicyRunner(CENetOnPolicyRunner):
    experiment_name = "unitree_g1_29dof_velocity_ce_sym_stairs"
    algorithm = CENetStairSymmetryPpoAlgorithm()

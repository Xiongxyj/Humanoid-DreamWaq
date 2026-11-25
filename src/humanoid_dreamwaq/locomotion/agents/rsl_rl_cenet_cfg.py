from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg, RslRlMLPModelCfg

@configclass
class CENetMLPModelCfg(RslRlMLPModelCfg):
    class_name = "humanoid_dreamwaq.locomotion.ce_net_mlp_model.CENetMLPModel"
    hidden_dims = [512,256,128]
    activation = "elu"
    distribution_cfg = RslRlMLPModelCfg.GaussianDistributionCfg(
        init_std=1.0,
        std_type="scalar"
    )

@configclass
class CENetPpoAlgorithm(RslRlPpoAlgorithmCfg):
    class_name = "humanoid_dreamwaq.locomotion.ce_net_ppo.CENetPPO"
    value_loss_coef=1.0
    use_clipped_value_loss=True
    clip_param=0.2
    entropy_coef=0.01
    num_learning_epochs=5
    num_mini_batches=4
    learning_rate=1.0e-3
    schedule="adaptive"
    gamma=0.99
    lam=0.95
    desired_kl=0.01
    max_grad_norm=1.0

    beta_kl = 0.001
    lambda_recon = 1.0
    lambda_vel = 1.0

@configclass
class CENetOnPolicyRunner(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = ""  # same as task name
    empirical_normalization = False
    clip_actions = 20.0
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}

    algorithm = CENetPpoAlgorithm()
    actor = CENetMLPModelCfg()
    critic = RslRlMLPModelCfg(
        hidden_dims = [512,256,128],
        activation = "elu",
        distribution_cfg = None,
    )

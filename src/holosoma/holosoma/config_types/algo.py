from __future__ import annotations

from dataclasses import field
from typing import Any, List, Union

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class OptimizerConfig:
    """Configuration for optimizer settings."""

    _target_: str
    """Target optimizer class (e.g., torch.optim.AdamW)."""

    weight_decay: float = 0.001
    """Weight decay parameter for the optimizer."""


@dataclass(frozen=True)
class LayerConfig:
    """Configuration for neural network layer settings."""

    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    """List of hidden layer dimensions."""

    activation: str = "ELU"
    """Activation function name."""

    dropout_prob: float = 0.0
    """Dropout probability."""

    use_layer_norm: bool = False
    """Whether to use layer normalization."""

    encoder_activation: str = "ELU"
    """Activation function name for encoder layers."""

    encoder_output_dim: int | None = None
    """Output dimension for encoder. Only used for encoder modules."""

    encoder_hidden_dims: List[int] | None = None
    """Hidden dimensions for encoder. Only used for encoder modules."""

    encoder_input_name: str = ""
    """Input name for encoder. Only used for encoder modules."""

    input_channels: int = 1
    """Number of input channels. Only used for CNN modules."""

    input_height: int = 1
    """Height of input feature maps. Only used for CNN modules."""

    input_width: int = 1
    """Width of input feature maps. Only used for CNN modules."""

    hidden_channels: tuple[int, ...] | None = None
    """Hidden channel dimensions. Only used for CNN modules."""

    kernel_size: int | tuple[int, ...] = 3
    """Kernel size for convolutions. Only used for CNN modules."""

    stride: int | tuple[int, ...] = 1
    """Stride for convolutions. Only used for CNN modules."""

    padding: str | int | tuple[str | int, ...] = "same"
    """Padding mode for convolutions. Only used for CNN modules."""

    module_input_name: tuple[str, ...] = ()
    """Input names for module. Only used for encoder modules."""

    # --- VisionTransformerEncoder fields ---

    embed_dim: int = 64
    """Embedding dimension for VisionTransformerEncoder."""

    num_heads: int = 4
    """Number of attention heads."""

    num_blocks: int = 1
    """Number of transformer blocks."""

    attention_type: str = "cross"
    """Attention type: 'self' or 'cross'."""

    use_ffn: bool = True
    """Whether to use feed-forward network in transformer blocks."""

    ffn_hidden_dim: int | None = None
    """Hidden dimension for FFN. None defaults to embed_dim * 4."""

    use_pos_encoding: bool = True
    """Whether to use learnable positional encodings."""

    use_final_ln: bool = False
    """Whether to apply final layer norm after transformer blocks."""

    aggregation_method: str = "last"
    """History aggregation: 'last', 'mean', 'max', or 'attention'."""

    # --- Multi-camera & vision mask fields ---

    num_cameras: int = 1
    """Number of camera views. >1 creates per-camera CNN branches."""

    use_vision_mask: bool = False
    """Enable velocity-conditioned masking of per-camera features."""

    vision_mask_input_name: str = ""
    """Observation key providing the velocity command for vision masking (e.g. 'command_lin_vel_obs')."""

    vision_mask_k: float = 4.0
    """Sigmoid steepness for vision mask."""

    vision_mask_v_th: float = 0.2
    """Velocity threshold for vision mask."""

    vision_mask_hard_cutoff: float = 0.0
    """Zero out camera scales below this value (0 = disabled)."""

    vision_tokens: str = "spatial"
    """Token layout of the multi-camera CNN tokeniser fed to the VisionTransformerEncoder.

    ``"spatial"``: one token per spatial cell (per camera), features = CNN output channels.
    ``"channel"``: one token per CNN output channel, features = spatial cells of all cameras
    concatenated (FAR-FALCON-Terrains layout; vision pos-encoding is ``[1, C_out, D]``).
    """

    independent_camera_init: bool = False
    """Initialise each camera's CNN branch independently (FAR-FALCON-Terrains).

    ``False`` deep-copies the first branch, so all cameras start from identical weights.
    """

    # --- Terrain MoE fields ---

    terrain_proportions: List[float] = field(default_factory=list)
    """Per-terrain-type proportions used to select which experts to instantiate in a MoE actor.
    Only terrain indices with a strictly positive proportion become experts.
    Used only by ``MoE*`` module types."""

    terrain_types: List[str] = field(default_factory=list)
    """Human-readable terrain type names indexed alongside ``terrain_proportions``.
    Used for loading per-terrain expert checkpoints and per-terrain loss logging."""


@dataclass(frozen=True)
class ModuleConfig:
    """Configuration for neural network modules."""

    type: str
    """Module type (e.g., MLP)."""

    input_dim: List[str] = field(default_factory=list)
    """Input dimension specification."""

    output_dim: List[str | int] = field(default_factory=list)
    """Output dimension specification."""

    layer_config: LayerConfig = field(default_factory=LayerConfig)
    """Layer configuration settings."""


@dataclass(frozen=True)
class PPOModuleDictConfig:
    """Configuration for PPO module dictionary."""

    actor: ModuleConfig
    """Actor module configuration."""

    critic: ModuleConfig
    """Critic module configuration."""


@dataclass(frozen=True)
class PPOMultiModuleDictConfig:
    """Separate actor-critic MLPs for lower and upper body (decoupled PPO)."""

    actor_lower_body: ModuleConfig
    critic_lower_body: ModuleConfig
    actor_upper_body: ModuleConfig
    critic_upper_body: ModuleConfig


@dataclass(frozen=True)
class DAggerModuleDictConfig:
    """Student + teacher actor pair for single-actor DAgger."""

    actor: ModuleConfig
    actor_teacher: ModuleConfig
    actor_encoder_teacher: ModuleConfig | None = None
    critic: ModuleConfig | None = None


@dataclass(frozen=True)
class DAggerMAModuleDictConfig:
    """Student + teacher actor pair per body for decoupled DAgger."""

    actor_lower_body: ModuleConfig
    actor_teacher_lower_body: ModuleConfig
    actor_upper_body: ModuleConfig
    actor_teacher_upper_body: ModuleConfig
    actor_encoder_teacher_lower_body: ModuleConfig | None = None
    actor_encoder_teacher_upper_body: ModuleConfig | None = None
    critic_lower_body: ModuleConfig | None = None
    critic_upper_body: ModuleConfig | None = None


@dataclass(frozen=True)
class WarmupParams:
    """Warmup scheduler parameters."""

    warmup_steps: int = 100
    warmup_start_factor: float = 1e-3
    warmup_end_factor: float = 1.0


@dataclass(frozen=True)
class BodyPartFloatConfig:
    """Per-body-part floating-point value."""

    lower_body: float = 0.0
    upper_body: float = 0.0


@dataclass(frozen=True)
class BodyPartBoolConfig:
    """Per-body-part boolean value."""

    lower_body: bool = False
    upper_body: bool = False


@dataclass(frozen=True)
class BodyPartLossTypeConfig:
    """Per-body-part loss type selection."""

    lower_body: str = "mse"
    upper_body: str = "mse"


@dataclass(frozen=True)
class PPOConfig:
    """Configuration for PPO algorithm."""

    module_dict: PPOModuleDictConfig
    """PPO module configurations (actor, critic)."""

    num_learning_epochs: int = 8
    """Number of learning epochs per update."""

    num_mini_batches: int = 4
    """Number of mini-batches per epoch."""

    clip_param: float = 0.2
    """PPO clipping parameter."""

    gamma: float = 0.99
    """Discount factor for future rewards."""

    lam: float = 0.95
    """GAE lambda parameter."""

    value_loss_coef: float = 1.0
    """Value loss coefficient."""

    entropy_coef: float = 0.01
    """Entropy coefficient for exploration."""

    actor_learning_rate: float = 1e-5
    """Learning rate for actor network."""

    actor_optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(_target_="torch.optim.AdamW"))
    """Actor optimizer configuration."""

    critic_learning_rate: float = 1e-5
    """Learning rate for critic network."""

    critic_optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(_target_="torch.optim.AdamW"))
    """Critic optimizer configuration."""

    max_grad_norm: float = 1.0
    """Maximum gradient norm for clipping."""

    schedule: str = "adaptive"
    """Learning rate schedule type."""

    desired_kl: float = 0.01
    """Desired KL divergence for adaptive learning rate."""

    use_symmetry: bool = False
    """Whether to use symmetry in training."""

    symmetry_actor_coef: float = 1.0
    """Symmetry coefficient for actor."""

    symmetry_critic_coef: float = 0.0
    """Symmetry coefficient for critic."""

    num_steps_per_env: int = 24
    """Number of steps per environment."""

    save_interval: int = 100
    """Interval for saving model checkpoints."""

    load_optimizer: bool = True
    """Whether to load optimizer state."""

    init_noise_std: float = 0.8
    """Initial noise standard deviation."""

    num_learning_iterations: int = 1000000
    """Total number of learning iterations."""

    init_at_random_ep_len: bool = True
    """Whether to initialize at random episode length."""

    empirical_normalization: bool = False
    """Whether to apply empirical normalization to actor and critic observations."""

    eval_callbacks: Any = None
    """Evaluation callbacks configuration."""

    max_actor_learning_rate: float | None = None
    min_actor_learning_rate: float | None = None
    max_critic_learning_rate: float | None = None
    min_critic_learning_rate: float | None = None

    log_raw_episode: bool = True
    """Log unscaled per-term rewards (``raw_rew_*``) to console and TensorBoard."""


@dataclass(frozen=True)
class PPOMAConfig:
    """PPO with two Gaussian policies (lower / upper) and two value heads (RPL ``ppo_ma`` style)."""

    module_dict: PPOMultiModuleDictConfig
    body_keys: tuple[str, ...] = ("lower_body", "upper_body")
    init_noise_std_lower_body: float = 0.8
    init_noise_std_upper_body: float = 0.8

    num_learning_epochs: int = 8
    num_mini_batches: int = 4
    clip_param: float = 0.2
    gamma: float = 0.99
    lam: float = 0.95
    value_loss_coef: float = 1.0
    entropy_coef: float = 0.01
    actor_learning_rate: float = 1e-5
    actor_optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(_target_="torch.optim.AdamW"))
    critic_learning_rate: float = 1e-5
    critic_optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(_target_="torch.optim.AdamW"))
    max_grad_norm: float = 1.0
    schedule: str = "adaptive"
    desired_kl: float = 0.01
    use_symmetry: bool = False
    symmetry_actor_coef: float = 1.0
    symmetry_critic_coef: float = 0.0
    num_steps_per_env: int = 24
    save_interval: int = 100
    load_optimizer: bool = True
    num_learning_iterations: int = 1000000
    init_at_random_ep_len: bool = True
    eval_callbacks: Any = None
    max_actor_learning_rate: float | None = None
    min_actor_learning_rate: float | None = None
    max_critic_learning_rate: float | None = None
    min_critic_learning_rate: float | None = None

    log_raw_episode: bool = True
    """Log unscaled per-term rewards (``raw_rew_*``) to console and TensorBoard."""


@dataclass(frozen=True)
class FastSACConfig:
    num_learning_iterations: int = 25000
    """total timesteps of the experiments"""

    critic_learning_rate: float = 3e-4
    """the learning rate of the critic"""

    actor_learning_rate: float = 3e-4
    """the learning rate for the actor"""

    alpha_learning_rate: float = 3e-4
    """the learning rate for the alpha"""

    buffer_size: int = 1024
    """the replay memory buffer size per environment"""

    num_steps: int = 1
    """the number of steps to use for the multi-step return"""

    gamma: float = 0.97
    """the discount factor gamma"""

    tau: float = 0.125
    """target smoothing coefficient (default: 0.005)"""

    batch_size: int = 8192
    """the batch size of sample from the replay memory"""

    learning_starts: int = 10
    """timestep to start learning"""

    policy_frequency: int = 4
    """the frequency of training policy (delayed)"""

    num_updates: int = 8
    """the number of updates to perform per step"""

    target_entropy_ratio: float = 0.0
    """the ratio of the target entropy to the number of actions"""

    num_atoms: int = 101
    """the number of atoms"""

    v_min: float = -20.0
    """the minimum value of the support"""

    v_max: float = 20.0
    """the maximum value of the support"""

    critic_hidden_dim: int = 768
    """the hidden dimension of the critic network"""

    actor_hidden_dim: int = 512
    """the hidden dimension of the actor network"""

    use_symmetry: bool = False
    """whether to use symmetry"""

    alpha_init: float = 0.001
    """the initial value of the alpha"""

    use_autotune: bool = True
    """whether to use autotune for the alpha"""

    use_tanh: bool = True
    """whether to use tanh for the action"""

    log_std_max: float = 0.0
    """the maximum value of the log std"""

    log_std_min: float = -5.0
    """the minimum value of the log std"""

    compile: bool = True
    """whether to use torch.compile."""

    obs_normalization: bool = True
    """whether to enable observation normalization"""

    use_layer_norm: bool = True
    """whether to use layer normalization"""

    num_q_networks: int = 2
    """number of Q-networks to ensemble"""

    max_grad_norm: float = 0.0
    """the maximum gradient norm"""

    amp: bool = True
    """whether to use amp"""

    amp_dtype: str = "bf16"
    """the dtype of the amp"""

    weight_decay: float = 0.001
    """the weight decay of the optimizer"""

    save_interval: int = 1000
    """the interval to save the model"""

    logging_interval: int = 100
    """the interval to log the metrics"""

    log_raw_episode: bool = True
    """Log unscaled per-term rewards (``raw_rew_*``) to console and TensorBoard."""

    encoder_obs_key: str = "perception_obs"
    """the key of the encoder observation. only valid if use_cnn_encoder is True"""

    encoder_obs_shape: tuple[int, int, int] = (1, 13, 9)
    """the shape of the encoder observation. only valid if use_cnn_encoder is True"""

    use_cnn_encoder: bool = False
    """whether to use CNN for the encoder"""

    actor_obs_keys: List[str] = field(default_factory=lambda: ["actor_obs"])
    critic_obs_keys: List[str] = field(default_factory=lambda: ["critic_obs"])

    eval_callbacks: Any = None
    """Evaluation callbacks configuration."""


@dataclass(frozen=True)
class PPOAlgoConfig:
    """Configuration for algorithm wrapper."""

    _target_: str
    """Target algorithm class."""

    _recursive_: bool
    """Whether to recursively instantiate."""

    config: PPOConfig
    """Algorithm-specific configuration."""


@dataclass(frozen=True)
class PPOMAAlgoConfig:
    """Decoupled two-policy PPO (lower + upper body)."""

    _target_: str
    _recursive_: bool
    config: PPOMAConfig


@dataclass(frozen=True)
class FastSACAlgoConfig:
    """Configuration for algorithm wrapper."""

    _target_: str
    """Target algorithm class."""

    _recursive_: bool
    """Whether to recursively instantiate."""

    config: FastSACConfig
    """Algorithm-specific configuration."""


@dataclass(frozen=True)
class StudentTeacherModuleConfig:
    """Configuration for a DepthStudentTeacherCritic module composite.

    Slots into :class:`DistillationPPOConfig`. Dimensions are resolved at
    runtime from the observation manager; only hidden-layer shapes and the
    depth-backbone class are declared here.
    """

    student_hidden_dims: List[int] = field(default_factory=lambda: [2048, 1024, 512, 256, 128])
    """Student MLP hidden layer widths."""

    teacher_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    """Teacher MLP hidden layer widths. Must match the teacher checkpoint being loaded."""

    critic_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    """Critic MLP hidden layer widths."""

    activation: str = "ELU"
    """Activation function name."""

    depth_backbone: str = "holosoma.agents.modules.depth_backbone.DepthOnlyFCBackbone58x87Small"
    """Dotted path to the depth-backbone nn.Module class, instantiated as ``cls(depth_output_dim)``."""

    depth_output_dim: int = 32
    """Latent dim produced by the depth backbone and concatenated onto the student input."""


@dataclass(frozen=True)
class DistillationPPOConfig:
    """Configuration for the :class:`DistillationPPO` algorithm.

    Mirrors :class:`PPOConfig` but replaces ``module_dict`` with a single
    :class:`StudentTeacherModuleConfig` and adds DAgger schedule knobs.
    """

    module: StudentTeacherModuleConfig
    """Student/teacher/critic module configuration."""

    num_learning_epochs: int = 2
    """Number of learning epochs per update."""

    num_mini_batches: int = 96
    """Number of mini-batches per epoch."""

    clip_param: float = 0.2
    """PPO clipping parameter."""

    gamma: float = 0.99
    """Discount factor for future rewards."""

    lam: float = 0.95
    """GAE lambda parameter."""

    value_loss_coef: float = 1.0
    """Value loss coefficient."""

    entropy_coef: float = 0.001
    """Entropy coefficient for exploration."""

    learning_rate: float = 3e-4
    """Learning rate applied to all parameter groups."""

    max_grad_norm: float = 1.0
    """Maximum gradient norm for clipping."""

    schedule: str = "adaptive"
    """Learning rate schedule type."""

    desired_kl: float = 0.01
    """Desired KL divergence for adaptive learning rate."""

    num_steps_per_env: int = 24
    """Number of rollout steps per environment."""

    save_interval: int = 500
    """Interval for saving model checkpoints."""

    init_noise_std: float = 0.01
    """Initial action noise standard deviation (forwarded to the student-teacher module)."""

    num_learning_iterations: int = 20000
    """Total number of learning iterations."""

    empirical_normalization: bool = False
    """Whether to apply empirical normalization. Not supported for distillation."""

    ppo_start_epoch: int = 0
    """Iteration at which PPO coefficient begins ramping up from 0."""

    dagger_end_epoch: int = 10000
    """Iteration at which PPO coefficient reaches its plateau (0.9)."""

    distill_loss_type: str = "mse"
    """``mse`` or ``huber``."""

    dagger_loss_coef: float = 10.0
    """Multiplier applied to the (1 - ppo_coef) * distill_loss term."""

    distillation_warmup_steps: int = 0
    """Initial training iterations during which the teacher drives the env.
    Updates are DAgger-only because PPO cannot use rewards generated by a
    different behavior policy. ``0`` disables warmup."""

    weight_decay: float = 0.0
    """Weight decay for the student/critic AdamW param groups."""

    depth_weight_decay: float = 1e-2
    """Weight decay for the depth-backbone AdamW parameter group."""

    eval_callbacks: Any = None
    """Evaluation callbacks configuration."""


@dataclass(frozen=True)
class DistillationPPOAlgoConfig:
    """Wrapper paralleling :class:`PPOAlgoConfig`."""

    _target_: str
    _recursive_: bool
    config: DistillationPPOConfig


@dataclass(frozen=True)
class DistillationConfig:
    """Configuration for the :class:`Distillation` (pure-DAgger) algorithm.

    Strips all PPO-related fields from :class:`DistillationPPOConfig`. The
    student is trained purely against teacher actions; there is no critic,
    no value loss, and no KL-adaptive LR schedule.
    """

    module: StudentTeacherModuleConfig
    """Student/teacher module configuration. ``critic_hidden_dims`` on the
    shared module cfg is ignored here since Distillation doesn't build a critic."""

    num_learning_epochs: int = 2
    """Number of learning epochs per update."""

    num_mini_batches: int = 96
    """Number of mini-batches per epoch."""

    learning_rate: float = 3e-4
    """Learning rate for the Adam optimizer (applied uniformly to student +
    depth backbone)."""

    max_grad_norm: float = 1.0
    """Maximum gradient norm for clipping."""

    num_steps_per_env: int = 24
    """Number of rollout steps per environment between updates."""

    save_interval: int = 500
    """Interval for saving model checkpoints."""

    init_noise_std: float = 0.001
    """Initial action noise std forwarded to the student module. Pure DAgger
    uses near-zero noise so the student mostly mirrors the teacher during
    rollouts."""

    num_learning_iterations: int = 10000
    """Total number of learning iterations."""

    empirical_normalization: bool = False
    """Not supported; must remain False."""

    distill_loss_type: str = "mse"
    """``mse`` or ``huber``."""

    distillation_warmup_steps: int = 0
    """Initial training iterations during which the teacher drives the env.
    The count is absolute, so resuming or calling ``learn()`` again does not
    restart warmup."""

    eval_callbacks: Any = None
    """Evaluation callbacks configuration."""


@dataclass(frozen=True)
class DistillationAlgoConfig:
    """Wrapper paralleling :class:`DistillationPPOAlgoConfig`."""

    _target_: str
    _recursive_: bool
    config: DistillationConfig


@dataclass(frozen=True)
class DAggerConfig:
    """Single-actor DAgger: one student distilled from one teacher."""

    module_dict: DAggerModuleDictConfig

    # Training schedule
    num_learning_iterations: int = 5000
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    num_steps_per_env: int = 24
    save_interval: int = 100
    init_at_random_ep_len: bool = True

    # Actor optimization
    actor_learning_rate: float = 1e-4
    actor_optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(_target_="torch.optim.AdamW", weight_decay=0.01)
    )
    max_grad_norm: float = 1.0
    bc_loss_coef: float = 1.0
    init_noise_std: float = 0.8

    # Loss type ("l1_norm", "l2_norm", "mse", "mae", "bce")
    loss_type: str = "mse"

    # Encoder-distillation loss (optional)
    use_encoder_loss: bool = False
    encoder_loss_coef: float = 0.0

    # LR scheduling
    use_scheduler: bool = True
    scheduler_type: str = "CosineAnnealingLR"
    num_scheduler_iterations: int | None = None
    scheduler_params: dict[str, Any] = field(default_factory=dict)

    use_warmup: bool = True
    warmup_params: WarmupParams = field(default_factory=WarmupParams)

    # Checkpointing
    ckpt_teacher: str | None = None
    ckpt_encoder_teacher: str | None = None
    load_optimizer: bool = True

    eval_callbacks: Any = None
    log_raw_episode: bool = True


@dataclass(frozen=True)
class DAggerAlgoConfig:
    """Single-actor DAgger wrapper."""

    _target_: str
    _recursive_: bool
    config: DAggerConfig


@dataclass(frozen=True)
class DAggerMAConfig:
    """Decoupled DAgger: student/teacher actors for lower + upper body."""

    module_dict: DAggerMAModuleDictConfig

    body_keys: tuple[str, ...] = ("lower_body", "upper_body")

    # Training schedule
    num_learning_iterations: int = 5000
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    num_steps_per_env: int = 24
    save_interval: int = 100
    init_at_random_ep_len: bool = True

    # Actor optimization
    actor_learning_rate: float = 1e-4
    actor_optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(_target_="torch.optim.AdamW", weight_decay=0.01)
    )
    max_grad_norm: float = 1.0
    bc_loss_coef: float = 1.0

    # Initial exploration noise for the student actor.
    init_noise_std_lower_body: float = 0.8
    init_noise_std_upper_body: float = 0.6

    # Loss selection ("l1_norm", "l2_norm", "mse", "mae", "bce")
    loss_types: BodyPartLossTypeConfig = field(default_factory=BodyPartLossTypeConfig)

    # Encoder-distillation loss (optional, per body part)
    use_encoder_loss: BodyPartBoolConfig = field(default_factory=BodyPartBoolConfig)
    encoder_loss_coef: BodyPartFloatConfig = field(default_factory=BodyPartFloatConfig)

    # Optional critic head (pure DAgger defaults to False).
    use_critic: bool = False
    value_loss_coef: float = 1.0
    gamma: float = 0.99
    lam: float = 0.95
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    critic_learning_rate: float = 1e-5
    critic_optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(_target_="torch.optim.AdamW", weight_decay=0.001)
    )

    # LR scheduling
    use_scheduler: bool = True
    scheduler_type: str = "CosineAnnealingLR"
    num_scheduler_iterations: int | None = None
    scheduler_params: dict[str, Any] = field(default_factory=dict)

    use_warmup: bool = True
    warmup_params: WarmupParams = field(default_factory=WarmupParams)

    # Checkpointing
    ckpt_teacher: str | None = None
    ckpt_actor_module: str | None = None
    ckpt_encoder_teacher: str | None = None
    load_optimizer: bool = True

    eval_callbacks: Any = None
    log_raw_episode: bool = True


@dataclass(frozen=True)
class DAggerMAAlgoConfig:
    """Decoupled DAgger wrapper."""

    _target_: str
    _recursive_: bool
    config: DAggerMAConfig


def _default_terrain_wandb_ckpts() -> dict[str, str]:
    return {
        "slopes": "",
        "stairs_up": "",
        "stairs_down": "",
        "stepping_stones_columns": "",
    }


@dataclass(frozen=True)
class DAggerMAMoEConfig(DAggerMAConfig):
    """Decoupled DAgger with a terrain-routed MoE teacher."""

    # Name of the observation that carries the terrain index per env.
    route_obs_name: str = "terrain_idx_obs"

    # Per-terrain-type W&B URIs for the MoE teacher experts. Keys are terrain
    # names (matching ``module_dict.actor_teacher_lower_body.layer_config.terrain_types``)
    # and values are full ``wandb://entity/project/run_id/model_xxx.pt`` URIs.
    # Takes precedence over the directory layout used with ``ckpt_teacher``.
    ckpt_teacher_wandb: dict[str, str] = field(default_factory=_default_terrain_wandb_ckpts)


@dataclass(frozen=True)
class DAggerMAMoEAlgoConfig:
    """Decoupled DAgger with MoE teacher wrapper."""

    _target_: str
    _recursive_: bool
    config: DAggerMAMoEConfig


AlgoInitConfig = Union[
    PPOConfig,
    PPOMAConfig,
    FastSACConfig,
    DistillationPPOConfig,
    DistillationConfig,
    DAggerConfig,
    DAggerMAConfig,
    DAggerMAMoEConfig,
]

AlgoConfig = Union[
    PPOAlgoConfig,
    PPOMAAlgoConfig,
    FastSACAlgoConfig,
    DistillationPPOAlgoConfig,
    DistillationAlgoConfig,
    DAggerAlgoConfig,
    DAggerMAAlgoConfig,
    DAggerMAMoEAlgoConfig,
]

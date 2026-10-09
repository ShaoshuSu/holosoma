from __future__ import annotations

from typing import cast

import torch

from holosoma.agents.modules.moe import MoETerrainEncoderActor, MoETerrainMLPActor
from holosoma.agents.modules.ppo_modules import (
    PPOActor,
    PPOActorEncoder,
    PPOActorVisionTransformerEncoder,
    PPOCritic,
    PPOCriticEncoder,
    PPOCriticVisionTransformerEncoder,
)
from holosoma.agents.modules.student_teacher_modules import (
    DepthStudentTeacher,
    DepthStudentTeacherCritic,
    STActor,
    STActorEncoder,
    STActorVisionTransformerEncoder,
)
from holosoma.config_types.algo import StudentTeacherModuleConfig
from holosoma.utils.helpers import get_class


def setup_ppo_actor_module(
    obs_dim_dict,
    module_config,
    num_actions,
    init_noise_std,
    device,
    history_length: dict[str, int],
):
    module_type = module_config.type
    if module_type == "VisionTransformerEncoder":
        return PPOActorVisionTransformerEncoder(
            obs_dim_dict=obs_dim_dict,
            module_config_dict=module_config,
            num_actions=num_actions,
            init_noise_std=init_noise_std,
            history_length=history_length,
        ).to(device)
    if module_type in ["MLPEncoder", "CNNEncoder"]:
        return PPOActorEncoder(
            obs_dim_dict=obs_dim_dict,
            module_config_dict=module_config,
            num_actions=num_actions,
            init_noise_std=init_noise_std,
            history_length=history_length,
        ).to(device)
    if module_type == "MLP":
        return PPOActor(
            obs_dim_dict=obs_dim_dict,
            module_config_dict=module_config,
            num_actions=num_actions,
            init_noise_std=init_noise_std,
            history_length=history_length,
        ).to(device)

    raise ValueError(f"Invalid actor type: {module_type}")


def setup_ppo_critic_module(
    obs_dim_dict,
    module_config,
    device,
    history_length: dict[str, int],
):
    module_type = module_config.type
    if module_type == "VisionTransformerEncoder":
        return PPOCriticVisionTransformerEncoder(
            obs_dim_dict=obs_dim_dict,
            module_config_dict=module_config,
            history_length=history_length,
        ).to(device)
    if module_type in ["MLPEncoder", "CNNEncoder"]:
        return PPOCriticEncoder(
            obs_dim_dict=obs_dim_dict,
            module_config_dict=module_config,
            history_length=history_length,
        ).to(device)
    if module_type == "MLP":
        return PPOCritic(
            obs_dim_dict=obs_dim_dict,
            module_config_dict=module_config,
            history_length=history_length,
        ).to(device)
    raise ValueError(f"Invalid critic type: {module_type}")


def setup_student_teacher_module(
    num_actor_obs: int,
    num_teacher_obs: int,
    num_critic_obs: int,
    num_actions: int,
    module_config: StudentTeacherModuleConfig,
    device: torch.device | str,
    init_noise_std: float,
    num_teachers: int = 1,
    module_cls: type | None = None,
) -> DepthStudentTeacherCritic:
    """Build a ``DepthStudentTeacherCritic``, or a drop-in replacement.

    ``module_config`` is a :class:`StudentTeacherModuleConfig`. The depth
    backbone is resolved from ``module_config.depth_backbone`` (dotted path)
    and instantiated with ``(depth_output_dim,)``. ``num_teachers`` controls
    how many frozen teacher MLPs are built.

    ``module_cls`` lets an application substitute a subclass — e.g. one that
    derives the per-env teacher index from the observation rather than taking
    it as an argument — without restating this construction.
    """
    backbone_cls = get_class(module_config.depth_backbone)
    depth_backbone = backbone_cls(module_config.depth_output_dim)

    cls = module_cls if module_cls is not None else DepthStudentTeacherCritic
    return cast(
        "DepthStudentTeacherCritic",
        cls(
            num_actor_obs=num_actor_obs,
            num_teacher_obs=num_teacher_obs,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            depth_backbone=depth_backbone,
            depth_output_dim=module_config.depth_output_dim,
            student_hidden_dims=list(module_config.student_hidden_dims),
            teacher_hidden_dims=list(module_config.teacher_hidden_dims),
            critic_hidden_dims=list(module_config.critic_hidden_dims),
            activation=module_config.activation,
            init_noise_std=init_noise_std,
            num_teachers=num_teachers,
        ).to(device),
    )


def setup_depth_student_teacher_module(
    num_actor_obs: int,
    num_teacher_obs: int,
    num_actions: int,
    module_config: StudentTeacherModuleConfig,
    device: torch.device | str,
    init_noise_std: float,
    num_teachers: int = 1,
    module_cls: type | None = None,
) -> DepthStudentTeacher:
    """Build a :class:`DepthStudentTeacher` (no critic). Used by the pure-DAgger
    :class:`Distillation` algorithm.

    Shares :class:`StudentTeacherModuleConfig` with the PPO variant;
    ``critic_hidden_dims`` on the cfg is silently ignored since this module
    has no critic MLP. ``module_cls`` substitutes a drop-in subclass, as in
    :func:`setup_student_teacher_module`.
    """
    backbone_cls = get_class(module_config.depth_backbone)
    depth_backbone = backbone_cls(module_config.depth_output_dim)

    cls = module_cls if module_cls is not None else DepthStudentTeacher
    return cast(
        "DepthStudentTeacher",
        cls(
            num_actor_obs=num_actor_obs,
            num_teacher_obs=num_teacher_obs,
            num_actions=num_actions,
            depth_backbone=depth_backbone,
            depth_output_dim=module_config.depth_output_dim,
            student_hidden_dims=list(module_config.student_hidden_dims),
            teacher_hidden_dims=list(module_config.teacher_hidden_dims),
            activation=module_config.activation,
            init_noise_std=init_noise_std,
            num_teachers=num_teachers,
        ).to(device),
    )
def setup_student_teacher_actor_module(
    obs_dim_dict,
    module_config,
    num_actions,
    init_noise_std,
    device,
    history_length: dict[str, int],
):
    """Build the deterministic actor used as DAgger student or teacher.

    Module types prefixed with ``MoE`` are wrapped in a terrain-routed mixture
    of experts; the MoE wrapper rebuilds each expert from the same
    ``module_config`` with the ``MoE`` prefix stripped.
    """
    kwargs = dict(
        obs_dim_dict=obs_dim_dict,
        module_config_dict=module_config,
        num_actions=num_actions,
        init_noise_std=init_noise_std,
        history_length=history_length,
    )
    module_type = module_config.type
    if module_type == "MoEMLP":
        return MoETerrainMLPActor(**kwargs).to(device)
    if module_type in ("MoEMLPEncoder", "MoECNNEncoder"):
        return MoETerrainEncoderActor(**kwargs).to(device)
    if module_type == "MLP":
        return STActor(**kwargs).to(device)
    if module_type in ("MLPEncoder", "CNNEncoder"):
        return STActorEncoder(**kwargs).to(device)
    if module_type == "VisionTransformerEncoder":
        return STActorVisionTransformerEncoder(**kwargs).to(device)
    raise ValueError(f"Invalid student/teacher actor type: {module_type}")

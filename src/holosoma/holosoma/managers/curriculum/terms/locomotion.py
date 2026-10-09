"""Curriculum hooks for locomotion tasks."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import torch
from loguru import logger

from holosoma.managers.curriculum.base import CurriculumTermBase


class AverageEpisodeLengthTracker(CurriculumTermBase):
    """Track moving average of episode length for locomotion tasks."""

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)
        params = cfg.params or {}
        base_num_compute_average_epl = float(params.get("num_compute_average_epl", 1000))
        base_denominator = getattr(env, "BASE_NUM_ENVS", env.num_envs)
        self.num_compute_average_epl = max(1, int(base_num_compute_average_epl * env.num_envs / base_denominator))
        self.average_episode_length = torch.tensor(0.0, device=env.device, dtype=torch.float)
        self._suppress_next_update = False

    def setup(self) -> None:
        self.average_episode_length = torch.as_tensor(
            float(self.average_episode_length), device=self.env.device, dtype=torch.float
        )

    def reset(self, env_ids) -> None:
        if env_ids is None:
            return

        if not torch.is_tensor(env_ids):
            env_ids_tensor = torch.as_tensor(env_ids, device=self.env.device, dtype=torch.long)
        else:
            env_ids_tensor = env_ids.to(device=self.env.device, dtype=torch.long)

        if env_ids_tensor.numel() == 0:
            return

        pending = self.env._pending_episode_lengths
        mask_tensor = self.env._pending_episode_update_mask

        update_mask = mask_tensor.index_select(0, env_ids_tensor)
        if not torch.any(update_mask):
            return
        active_ids = env_ids_tensor[update_mask]

        if active_ids.numel() == 0:
            return

        episode_lengths = pending.index_select(0, active_ids).to(dtype=torch.float)
        if episode_lengths.numel() == 0:
            return

        self.update(active_ids, episode_lengths)

        zeros_long = torch.zeros(active_ids.shape, device=pending.device, dtype=pending.dtype)
        pending.index_copy_(0, active_ids, zeros_long)

        zeros_bool = torch.zeros(active_ids.shape, device=mask_tensor.device, dtype=mask_tensor.dtype)
        mask_tensor.index_copy_(0, active_ids, zeros_bool)

    def step(self) -> None:
        return

    def update(self, env_ids: torch.Tensor, episode_lengths: torch.Tensor) -> None:
        if self._suppress_next_update:
            self._suppress_next_update = False
            return

        num = env_ids.numel()
        if num == 0:
            return
        current_average = torch.mean(episode_lengths.to(dtype=torch.float), dtype=torch.float)
        weight = min(num / self.num_compute_average_epl, 1.0)
        self.average_episode_length = self.average_episode_length * (1 - weight) + current_average * weight

    def suppress_next_update(self) -> None:
        self._suppress_next_update = True

    def get_average(self) -> torch.Tensor:
        return self.average_episode_length

    def set_average(self, value: float | torch.Tensor, *, suppress_update: bool = True) -> None:
        self.average_episode_length = torch.as_tensor(float(value), device=self.env.device, dtype=torch.float)
        if suppress_update:
            self._suppress_next_update = True

    def state_dict(self) -> dict[str, Any]:
        return {
            "average_episode_length": self.average_episode_length.detach().to("cpu"),
            "suppress_next_update": self._suppress_next_update,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        avg = state.get("average_episode_length")
        if avg is not None:
            self.average_episode_length = torch.as_tensor(float(avg), device=self.env.device, dtype=torch.float)
        self._suppress_next_update = bool(state.get("suppress_next_update", False))


class PenaltyCurriculum(CurriculumTermBase):
    """Stateful penalty curriculum that scales reward term weights based on episode length.

    This curriculum term adaptively scales penalty reward weights during training.
    When episodes are short (robot falls quickly), penalties are reduced to make
    learning easier. As episodes get longer (robot stays up), penalties gradually
    increase to refine behavior.
    """

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)

        # Get parameters from config
        params = cfg.params
        self.enabled = params.get("enabled", True)
        self.tag = params.get("tag", "penalty_curriculum")
        self.min_scale = float(params.get("min_scale", 0.0))
        self.max_scale = float(params.get("max_scale", 1.0))
        self.level_down_threshold = float(params.get("level_down_threshold", 150.0))
        self.level_up_threshold = float(params.get("level_up_threshold", 850.0))
        self.degree = float(params.get("degree", 0.0))

        # State variables (previously stored on env)
        self.current_scale = float(params.get("initial_scale", 1.0))
        self.penalty_reward_names: list[str] = []
        self.original_weights: dict[str, float] = {}

    def setup(self) -> None:
        """Setup penalty curriculum - identify rewards and apply initial scaling."""
        if not self.enabled or not hasattr(self.env, "reward_manager"):
            return

        # Identify penalty rewards by tag using public reward manager APIs
        for term_name in self.env.reward_manager.active_terms:
            term_cfg = self.env.reward_manager.get_term_cfg(term_name)
            if self.tag in term_cfg.tags:
                self.penalty_reward_names.append(term_name)

        # The user explicitly enabled the curriculum and named a tag; if no active reward term carries
        # it (typo or tag/term mismatch), the curriculum scales nothing yet reports itself enabled
        # below. Warn rather than run a silently inert curriculum.
        if not self.penalty_reward_names:
            logger.warning(
                f"PenaltyCurriculum enabled but no active reward term carries tag '{self.tag}'; "
                f"the penalty curriculum will have no effect."
            )

        # Store original weights and apply initial scaling
        for name in self.penalty_reward_names:
            if name not in self.env.reward_manager.active_terms:
                continue

            term_cfg = self.env.reward_manager.get_term_cfg(name)
            # Store original weight
            self.original_weights[name] = float(term_cfg.weight)
            # Apply initial scale
            scaled_cfg = replace(term_cfg, weight=term_cfg.weight * self.current_scale)
            self.env.reward_manager.set_term_cfg(name, scaled_cfg)

        # Set flag for logging compatibility
        self.env.use_reward_penalty_curriculum = True
        self.env.reward_penalty_scale = self.current_scale

    def reset(self, env_ids) -> None:
        """Update penalty scale based on average episode length."""
        if not self.enabled or not hasattr(self.env, "reward_manager"):
            return

        if not self.penalty_reward_names or not self.original_weights:
            return

        average_length = float(self.env.average_episode_length)

        # Update current scale based on episode length
        if average_length < self.level_down_threshold:
            self.current_scale *= 1.0 - self.degree
        elif average_length > self.level_up_threshold:
            self.current_scale *= 1.0 + self.degree

        # Clamp scale
        self.current_scale = float(np.clip(self.current_scale, self.min_scale, self.max_scale))

        # Apply scale to each penalty reward's weight
        for name in self.penalty_reward_names:
            if name not in self.original_weights or name not in self.env.reward_manager.active_terms:
                continue
            term_cfg = self.env.reward_manager.get_term_cfg(name)
            # Set weight = original_weight * current_scale
            scaled_cfg = replace(term_cfg, weight=self.original_weights[name] * self.current_scale)
            self.env.reward_manager.set_term_cfg(name, scaled_cfg)

        # Update for logging
        self.env.reward_penalty_scale = self.current_scale

        # Update log_dict for WandB logging
        if hasattr(self.env, "log_dict"):
            self.env.log_dict["penalty_scale"] = torch.tensor(self.current_scale, dtype=torch.float)

    def step(self) -> None:
        """Clamp penalty scale within bounds each step."""
        if not self.enabled or not hasattr(self.env, "reward_manager"):
            return

        if not self.penalty_reward_names or not self.original_weights:
            return

        # Clamp current_scale
        self.current_scale = float(np.clip(self.current_scale, self.min_scale, self.max_scale))

        # Re-apply clamped scale to weights
        for name in self.penalty_reward_names:
            if name not in self.original_weights or name not in self.env.reward_manager.active_terms:
                continue
            term_cfg = self.env.reward_manager.get_term_cfg(name)
            scaled_cfg = replace(term_cfg, weight=self.original_weights[name] * self.current_scale)
            self.env.reward_manager.set_term_cfg(name, scaled_cfg)


class TerrainCurriculum(CurriculumTermBase):
    """RPL-style locomotion terrain curriculum based on traveled distance."""

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)
        params = cfg.params or {}
        self.enabled = bool(params.get("enabled", True))
        self.move_up_ratio = float(params.get("terrain_move_up_ratio", 0.5))
        self.move_down_ratio = float(params.get("terrain_move_down_ratio", 0.5))

    def setup(self) -> None:
        return

    def reset(self, env_ids) -> None:
        if not self.enabled:
            return
        idx = self._ensure_index_tensor(env_ids)
        if idx.numel() == 0 or not getattr(self.env, "init_done", False):
            return

        terrain_state = self.env.terrain_manager.get_state("locomotion_terrain")
        terrain_cfg = self.env.terrain_manager.cfg.terrain_term
        if terrain_state is None or str(terrain_cfg.mesh_type) == "plane":
            return

        distance = torch.norm(
            self.env.simulator.robot_root_states[idx, :2] - terrain_state.env_origins[idx, :2],
            dim=1,
        )
        move_up = distance > terrain_cfg.terrain_length * self.move_up_ratio
        move_down = (
            distance
            < torch.norm(self.env.command_manager.commands[idx, :2], dim=1)
            * self.env.max_episode_length_s
            * self.move_down_ratio
        ) & ~move_up

        new_levels = terrain_state.terrain_levels[idx] + move_up.to(torch.long) - move_down.to(torch.long)
        max_level = max(int(terrain_state.max_terrain_level), 1)
        new_levels = torch.where(
            new_levels >= max_level,
            torch.randint_like(new_levels, max_level),
            torch.clamp(new_levels, min=0),
        )
        terrain_state.set_terrain_levels(idx, new_levels)

    def step(self) -> None:
        return

    def _ensure_index_tensor(self, env_ids) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.env.num_envs, device=self.env.device, dtype=torch.long)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.env.device, dtype=torch.long)
        return torch.as_tensor(list(env_ids), device=self.env.device, dtype=torch.long)


class DomainRandScaleCurriculum(CurriculumTermBase):
    """Scale domain randomization ranges based on average episode length.

    Mirrors FAR-FALCON's ``domain_rand_scale_curriculum``: when episodes are
    short, shrink the DR ranges to make the task easier; as episodes get
    longer, expand the ranges up to ``max_scale``. The scale is exposed on
    the env as ``env.domain_rand_scale`` so randomization terms can read it.

    Terms that honor ``env.domain_rand_scale`` (if present):
      - Height-measurement XY offset range (:class:`HeightMeasurementXYOffsetRandomizer`),
        via an explicit ``set_curriculum_scale`` propagation.
      - Control delay step range (``randomize_action_delay`` reads
        ``env.domain_rand_scale`` directly each reset, matching FALCON-Terrains).

    Tuning params (mirrors FAR-FALCON's YAML knobs):
      - enabled: toggle the curriculum
      - initial_scale: starting value of ``domain_rand_scale``
      - min_scale / max_scale: clamp
      - level_down_threshold: shrink range below this epl
      - level_up_threshold: grow range above this epl
      - degree: multiplicative rate per update
    """

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)
        params = cfg.params or {}
        self.enabled: bool = bool(params.get("enabled", False))
        self.initial_scale: float = float(params.get("initial_scale", 0.1))
        self.min_scale: float = float(params.get("min_scale", 0.0))
        self.max_scale: float = float(params.get("max_scale", 1.0))
        self.level_down_threshold: float = float(params.get("level_down_threshold", 40.0))
        self.level_up_threshold: float = float(params.get("level_up_threshold", 210.0))
        self.degree: float = float(params.get("degree", 0.000025))
        self.current_scale: float = self.initial_scale

    def setup(self) -> None:
        if not self.enabled:
            # Still advertise scale=1 so consumers get a consistent attribute.
            self.env.use_domain_rand_scale_curriculum = False
            self.env.domain_rand_scale = 1.0
            return
        self.env.use_domain_rand_scale_curriculum = True
        self.env.domain_rand_scale = self.current_scale
        self._propagate()

    def reset(self, env_ids) -> None:
        if not self.enabled:
            return
        avg = float(self.env.average_episode_length)
        if avg < self.level_down_threshold:
            self.current_scale *= 1.0 - self.degree
        elif avg > self.level_up_threshold:
            self.current_scale *= 1.0 + self.degree
        self.current_scale = float(np.clip(self.current_scale, self.min_scale, self.max_scale))
        self.env.domain_rand_scale = self.current_scale
        self._propagate()
        if hasattr(self.env, "log_dict"):
            self.env.log_dict["domain_rand_scale"] = torch.tensor(self.current_scale, dtype=torch.float)

    def step(self) -> None:
        return

    def _propagate(self) -> None:
        """Push the current scale to any randomization term that cares."""
        rm = getattr(self.env, "randomization_manager", None)
        if rm is None:
            return
        hm_term = rm.get_state("height_measurements_xy_offset")
        if hm_term is not None and hasattr(hm_term, "set_curriculum_scale"):
            hm_term.set_curriculum_scale(self.current_scale)


# ================================================================================================
# Legacy stateless functions (backward compatibility)
# ================================================================================================


def configure_reward_penalty(
    env,
    *,
    enabled: bool = True,
    tag: str = "penalty_curriculum",
    initial_scale: float = 1.0,
    min_scale: float = 0.0,
    max_scale: float = 1.0,
    level_down_threshold: float = 150.0,
    level_up_threshold: float = 750.0,
    degree: float = 0.0,
) -> None:
    """Configure reward-penalty curriculum parameters.

    This modifies the reward term weights directly in the reward manager,
    scaling them by initial_scale and storing the original weights for reference.

    Args:
        enabled: Whether to enable penalty curriculum
        tag: Tag to filter reward terms (e.g., "penalty_curriculum").
        initial_scale: Initial scaling factor
        min_scale: Minimum scaling factor
        max_scale: Maximum scaling factor
        level_down_threshold: Episode length threshold for decreasing penalty scale
        level_up_threshold: Episode length threshold for increasing penalty scale
        degree: Adjustment rate when updating penalty scale
    """
    env.use_reward_penalty_curriculum = bool(enabled)

    # Determine which rewards to apply curriculum to
    # Use tag-based selection
    penalty_names = []
    for term_name in env.reward_manager.active_terms:
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        if tag in term_cfg.tags:
            penalty_names.append(term_name)

    # Enabled but nothing matched the explicitly-given tag -> a permanently inert curriculum that
    # still reports itself enabled. Warn instead of silently doing nothing (mirrors PenaltyCurriculum).
    if enabled and not penalty_names:
        logger.warning(
            f"configure_reward_penalty: no active reward term carries tag '{tag}'; "
            f"the penalty curriculum is enabled but will have no effect."
        )

    env._curriculum_penalty_reward_names = penalty_names

    # Store original weights and apply initial scaling
    env._curriculum_penalty_original_weights = {}
    if env.use_reward_penalty_curriculum and hasattr(env, "reward_manager"):
        for name in penalty_names:
            if name not in env.reward_manager.active_terms:
                continue
            term_cfg = env.reward_manager.get_term_cfg(name)
            # Store original weight
            env._curriculum_penalty_original_weights[name] = float(term_cfg.weight)
            # Apply initial scale
            scaled_cfg = replace(term_cfg, weight=term_cfg.weight * initial_scale)
            env.reward_manager.set_term_cfg(name, scaled_cfg)

    env._curriculum_penalty_cfg = {
        "min_scale": float(min_scale),
        "max_scale": float(max_scale),
        "level_down_threshold": float(level_down_threshold),
        "level_up_threshold": float(level_up_threshold),
        "degree": float(degree),
        "current_scale": float(initial_scale),
    }

    # Set reward_penalty_scale for logging compatibility
    env.reward_penalty_scale = float(initial_scale)


def update_reward_penalty(env, env_ids, **_) -> None:
    """Update penalty scale based on average episode length.

    Modifies reward term weights directly in the reward manager.
    """
    if not getattr(env, "use_reward_penalty_curriculum", False):
        return

    cfg = getattr(env, "_curriculum_penalty_cfg", None)
    penalty_names = getattr(env, "_curriculum_penalty_reward_names", [])
    original_weights = getattr(env, "_curriculum_penalty_original_weights", {})
    if not cfg or not penalty_names or not hasattr(env, "reward_manager"):
        return

    average_length = float(env.average_episode_length)
    degree = cfg["degree"]

    # Update current scale based on episode length
    current_scale = cfg["current_scale"]
    if average_length < cfg["level_down_threshold"]:
        current_scale *= 1.0 - degree
    elif average_length > cfg["level_up_threshold"]:
        current_scale *= 1.0 + degree

    # Clamp scale
    current_scale = float(np.clip(current_scale, cfg["min_scale"], cfg["max_scale"]))
    cfg["current_scale"] = current_scale

    # Apply scale to each penalty reward's weight
    for name in penalty_names:
        if name not in original_weights or name not in env.reward_manager.active_terms:
            continue
        term_cfg = env.reward_manager.get_term_cfg(name)
        # Set weight = original_weight * current_scale
        scaled_cfg = replace(term_cfg, weight=original_weights[name] * current_scale)
        env.reward_manager.set_term_cfg(name, scaled_cfg)

    # Update reward_penalty_scale for logging
    env.reward_penalty_scale = current_scale

    # Update log_dict for WandB logging
    if hasattr(env, "log_dict"):
        import torch

        env.log_dict["penalty_scale"] = torch.tensor(env.reward_penalty_scale, dtype=torch.float)


def clamp_reward_penalty(env, **_) -> None:
    """Ensure penalty scale stays within configured bounds each step.

    Re-applies clamping to reward weights in case of any drift.
    """
    if not getattr(env, "use_reward_penalty_curriculum", False):
        return

    cfg = getattr(env, "_curriculum_penalty_cfg", None)
    penalty_names = getattr(env, "_curriculum_penalty_reward_names", [])
    original_weights = getattr(env, "_curriculum_penalty_original_weights", {})
    if not cfg or not penalty_names or not hasattr(env, "reward_manager"):
        return

    # Clamp current_scale
    current_scale = cfg.get("current_scale", 1.0)
    current_scale = float(np.clip(current_scale, cfg["min_scale"], cfg["max_scale"]))
    cfg["current_scale"] = current_scale

    # Re-apply clamped scale to weights
    for name in penalty_names:
        if name not in original_weights or name not in env.reward_manager.active_terms:
            continue
        term_cfg = env.reward_manager.get_term_cfg(name)
        scaled_cfg = replace(term_cfg, weight=original_weights[name] * current_scale)
        env.reward_manager.set_term_cfg(name, scaled_cfg)


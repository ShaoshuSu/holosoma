"""Configuration types for observation manager."""

from __future__ import annotations

from dataclasses import field
from typing import Any

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class ObsTermCfg:
    """Configuration for a single observation term."""

    func: str
    """Import path to the observation function, e.g.
    ``holosoma.managers.observation.terms.locomotion:base_lin_vel``."""

    params: dict[str, Any] = field(default_factory=dict)
    """Additional keyword arguments forwarded to ``func``."""

    scale: float | tuple[float, ...] = 1.0
    """Scaling factor(s) applied after noise and clipping."""

    noise: float = 0.0
    """Noise magnitude applied before scaling when the group enables noise."""

    clip: tuple[float, float] | None = None
    """Optional min/max clip bounds applied after scaling."""


@dataclass(frozen=True)
class ObsGroupCfg:
    """Configuration for an observation group (e.g. actor or critic input)."""

    terms: dict[str, ObsTermCfg] = field(default_factory=dict)
    """Mapping of term name to its configuration."""

    concatenate: bool = True
    """Whether to concatenate term outputs into a single tensor."""

    enable_noise: bool = False
    """If ``True``, apply each term's noise setting before scaling."""

    history_length: int = 1
    """Number of timesteps to retain for history stacking (``1`` disables history)."""

    interval: int = 1
    """Push a new frame into the group only every ``interval`` env steps (per env).

    Matches FAR-FALCON-Terrains ``obs.obs_intervals``: a frame is pushed when the
    per-env counter satisfies ``counter % interval == 0`` and the counter then
    always advances. The counter is zeroed on env reset, so the first step of an
    episode always pushes. On the other steps the group returns its previous value.
    """

    reset_history_on_env_reset: bool = True
    """Zero this group's history buffer for reset envs.

    FAR-FALCON-Terrains never clears its observation buffers on reset (the
    previous episode's frames shift out naturally); set ``False`` to match it.
    """

    history_time_major: bool = False
    """Flatten history time-major (``[t-N: all terms, ..., t: all terms]``) instead of term-major.

    Only affects concatenated groups with ``history_length > 1``. FAR-FALCON-Terrains / RPL
    policies use time-major order.
    """


@dataclass(frozen=True)
class ObservationManagerCfg:
    """Configuration for the observation manager."""

    groups: dict[str, ObsGroupCfg] = field(default_factory=dict)
    """Mapping of group name to its configuration."""

    clip_observations: float = 100.0
    """Global observation clipping threshold (applied to all non-image observations)."""

"""Base classes and protocols for observation terms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, final

from holosoma.utils.safe_torch_import import torch

if TYPE_CHECKING:
    from holosoma.config_types.observation import ObsTermCfg


class ObservationTermBase(ABC):
    """Base class for observation terms.

    Subclass it for terms that keep internal state (e.g. latency buffers, filters). A plain function
    in a term config is wrapped in :class:`FunctionTerm`, so the manager handles every term the same way.

    The manager reads a term once per frame for all envs, and may additionally read it for the
    envs about to reset to observe their terminal state.
    """

    def __init__(self, cfg: ObsTermCfg, env: Any):
        """Initialize observation term.

        Args:
            cfg: Configuration for this observation term
            env: Environment instance (typically a ``BaseTask`` subclass)
        """
        self.cfg = cfg
        self.env = env

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset internal state for specified environments. Does nothing by default.

        Args:
            env_ids: Environment IDs to reset. If None, reset all environments.
        """
        del env_ids

    @abstractmethod
    def read(self, env: Any, env_ids: torch.Tensor | None = None, **kwargs: Any) -> torch.Tensor:
        """Compute the observation of ``env_ids`` and advance only their internal state.

        Envs not in ``env_ids`` must keep their state untouched. An empty ``env_ids`` is how
        :meth:`shape` measures the output: return ``[0, *obs_shape]`` and change nothing.

        Args:
            env: Environment instance
            env_ids: Environment IDs to compute and advance. If None, all environments.
            **kwargs: Additional parameters from config

        Returns:
            Observation tensor of shape [num_envs, *obs_shape], or [len(env_ids), *obs_shape]
        """

    @final
    def observe(self, env: Any, env_ids: torch.Tensor | None = None, **kwargs: Any) -> torch.Tensor:
        """Read the observation of ``env_ids`` as the manager serves it.

        Args:
            env: Environment instance
            env_ids: Environment IDs to compute and advance. If None, all environments.
            **kwargs: Additional parameters from config

        Returns:
            A copy of :meth:`read`'s output; the term may return a view of state that ``reset`` overwrites.
        """
        return self._checked_read(env, env_ids, **kwargs).clone()

    @final
    def shape(self, env: Any, **kwargs: Any) -> tuple[int, ...]:
        """Per-env output shape, measured by reading zero envs.

        Args:
            env: Environment instance
            **kwargs: Additional parameters from config
        """
        no_envs = torch.zeros(0, dtype=torch.long, device=env.device)
        return tuple(self._checked_read(env, no_envs, **kwargs).shape[1:])

    @final
    def _checked_read(self, env: Any, env_ids: torch.Tensor | None, **kwargs: Any) -> torch.Tensor:
        """:meth:`read`, raising if a row subset comes back with the wrong number of rows."""
        obs = self.read(env, env_ids=env_ids, **kwargs)
        if env_ids is not None and obs.shape[0] != len(env_ids):
            raise ValueError(
                f"Observation term {type(self).__name__} ({self.cfg.func}) returned {obs.shape[0]} rows for "
                f"{len(env_ids)} env_ids; ObservationTermBase.read must compute and advance only env_ids."
            )
        return obs


class FunctionTerm(ObservationTermBase):
    """A stateless observation term defined by a plain function ``func(env, **params)``."""

    def __init__(self, cfg: ObsTermCfg, env: Any, func: Callable[..., torch.Tensor]):
        super().__init__(cfg, env)
        self.func = func

    def read(self, env: Any, env_ids: torch.Tensor | None = None, **kwargs: Any) -> torch.Tensor:
        obs = self.func(env, **kwargs)
        return obs if env_ids is None else obs[env_ids]

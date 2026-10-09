"""Observation manager for computing and processing observations."""

from __future__ import annotations

import inspect
from collections import deque
from typing import Any, Callable

import torch

from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg
from holosoma.managers.utils import resolve_callable

from .base import ObservationTermBase


class ObservationManager:
    """Manages observation computation with groups, terms, and processing.

    The manager orchestrates the computation of observations from multiple terms,
    applying scaling, noise, clipping, and history buffering to produce the final
    observation tensors. It maintains exact equivalence with the direct observation
    system while providing a more modular structure.

    Parameters
    ----------
    cfg : ObservationManagerCfg
        Configuration specifying observation groups and terms.
    env : Any
        Environment instance (typically a ``BaseTask`` subclass).
    device : str
        Device to place tensors on.
    """

    def __init__(self, cfg: ObservationManagerCfg, env: Any, device: str):
        self.cfg = cfg
        self.env = env
        self.device = device
        self.logger = getattr(env, "logger", None)

        # Storage for resolved functions and stateful terms
        self._term_funcs: dict[str, dict[str, Callable]] = {}
        self._term_instances: dict[str, dict[str, ObservationTermBase]] = {}
        self._term_supports_modify_history: dict[str, dict[str, bool]] = {}

        # History buffers: group_name -> term_name -> deque
        self._history_buffers: dict[str, dict[str, deque]] = {}

        # FALCON-style frame buffers, used only by groups with ``interval > 1`` or
        # ``reset_history_on_env_reset=False``: group_name -> term_name ->
        # ``[num_envs, history_length, term_dim]`` (zero-initialised, allocated lazily),
        # plus a per-env step counter per group.
        self._frame_buffers: dict[str, dict[str, torch.Tensor]] = {}
        self._interval_counters: dict[str, torch.Tensor] = {}

        # Initialize groups
        self._initialize_groups()

    def _initialize_groups(self) -> None:
        """Initialize observation groups and resolve term functions."""
        for group_name, group_cfg in self.cfg.groups.items():
            self._term_funcs[group_name] = {}
            self._term_instances[group_name] = {}
            self._term_supports_modify_history[group_name] = {}
            self._history_buffers[group_name] = {}

            for term_name, term_cfg in group_cfg.terms.items():
                # Resolve function
                func = resolve_callable(term_cfg.func, context="observation term")

                # Check if it's a class (stateful) or function (stateless)
                if isinstance(func, type) and issubclass(func, ObservationTermBase):
                    # Stateful term - instantiate
                    instance = func(term_cfg, self.env)
                    self._term_instances[group_name][term_name] = instance
                    self._term_supports_modify_history[group_name][term_name] = self._accepts_modify_history(instance)
                else:
                    # Stateless function
                    self._term_funcs[group_name][term_name] = func

                # Initialize history buffer if needed (using group-level history_length)
                if self._uses_frame_buffer(group_cfg):
                    continue
                if group_cfg.history_length > 1:
                    self._history_buffers[group_name][term_name] = deque(maxlen=group_cfg.history_length)

            if self._uses_frame_buffer(group_cfg):
                if group_cfg.interval < 1:
                    raise ValueError(
                        f"Observation group '{group_name}' has interval={group_cfg.interval}; must be >= 1"
                    )
                self._frame_buffers[group_name] = {}
                self._interval_counters[group_name] = torch.zeros(
                    self.env.num_envs, dtype=torch.long, device=self.device
                )

    @staticmethod
    def _uses_frame_buffer(group_cfg: ObsGroupCfg) -> bool:
        return group_cfg.interval != 1 or not group_cfg.reset_history_on_env_reset
    @staticmethod
    def _accepts_modify_history(instance: ObservationTermBase) -> bool:
        """Whether a stateful term opts into manager-controlled history reads."""
        try:
            params = inspect.signature(instance.__call__).parameters.values()
        except (TypeError, ValueError):
            return False
        return any(param.name == "modify_history" or param.kind is inspect.Parameter.VAR_KEYWORD for param in params)

    def compute(self, *, modify_history: bool = True) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Compute all observation groups.

        Parameters
        ----------
        modify_history : bool, optional
            If ``True``, update history buffers; if ``False``, preserve them
            for bootstrapping. Stateful terms receive the same flag so they can
            preserve any internal buffers. Defaults to ``True``.

        Returns
        -------
        dict[str, torch.Tensor | dict[str, torch.Tensor]]
            Mapping from group names to observation tensors or dictionaries of tensors.
        """
        obs_dict = {}
        for group_name in self.cfg.groups:
            obs_dict[group_name] = self.compute_group(group_name, modify_history=modify_history)
        return obs_dict

    def compute_group(self, group_name: str, *, modify_history: bool = True) -> torch.Tensor | dict[str, torch.Tensor]:
        """Compute observations for a specific group.

        This method replicates the exact behavior of the direct
        ``parse_observation()`` function to ensure equivalence.

        Parameters
        ----------
        group_name : str
            Name of the observation group to compute.
        modify_history : bool, optional
            If ``True``, update history buffers; if ``False``, preserve them
            for bootstrapping. Defaults to ``True``.

        Returns
        -------
        torch.Tensor | dict[str, torch.Tensor]
            Concatenated tensor when ``group.concatenate`` is ``True``; otherwise
            a dictionary of tensors keyed by term name.
        """
        group_cfg = self.cfg.groups[group_name]
        obs_tensors = {}
        interval_counters = getattr(self, "_interval_counters", {})

        for term_name, term_cfg in group_cfg.terms.items():
            # 1. Compute base observation
            obs = self._compute_term(
                group_name,
                term_name,
                term_cfg,
                modify_history=modify_history,
            )

            # 2. Apply noise (matches direct: noise before scaling)
            if group_cfg.enable_noise and term_cfg.noise > 0:
                obs = self._apply_noise(obs, term_cfg.noise)

            # 3. Apply scaling (matches direct: scale after noise). Skip the multiply when the
            # scale is the default 1.0 (numerically a no-op) so a non-float term (e.g. a uint8
            # RGB image) keeps its native dtype instead of being promoted to float.
            if term_cfg.scale != 1.0:
                obs = self._apply_scale(obs, term_cfg.scale)

            # 4. Apply clipping (if specified)
            if term_cfg.clip is not None:
                obs = obs.clip(term_cfg.clip[0], term_cfg.clip[1])

            # 5. Handle history buffering
            # NOTE: history is configured at the group level but the buffers are kept
            # per term. When history is active each term becomes a 3D tensor
            # ``[num_envs, history_length, term_dim]``; the group flattens it below in
            # term-major (default) or time-major order (``history_time_major``).
            if group_name in interval_counters:
                obs = self._apply_frame_buffer(group_name, term_name, obs, group_cfg, modify_buffer=modify_history)
            elif group_cfg.history_length > 1:
                obs = self._apply_history(group_name, term_name, obs, group_cfg, modify_buffer=modify_history)

            obs_tensors[term_name] = obs

        # FALCON advances the interval counter on every step, pushed or not.
        if modify_history and group_name in interval_counters:
            interval_counters[group_name] += 1

        # Concatenate or return dict
        if group_cfg.concatenate:
            # Concatenate in alphabetically sorted order (to match direct system behavior)
            # Direct system does: sorted(obs_config) before concatenation
            sorted_keys = sorted(obs_tensors.keys())
            if group_cfg.history_length > 1:
                # Each term is ``[num_envs, history_length, term_dim]``.
                if group_cfg.history_time_major:
                    # Concatenate terms along the feature dimension, then flatten in TIME-MAJOR
                    # order: ``[term_a_{t-N}, term_b_{t-N}, ..., term_a_{t}, term_b_{t}, ...]``.
                    stacked = torch.cat([obs_tensors[key] for key in sorted_keys], dim=-1)
                    return stacked.reshape(stacked.shape[0], -1)
                # TERM-MAJOR order: ``[term_a_{t-N..t}, term_b_{t-N..t}, ...]``.
                return torch.cat([obs_tensors[key].reshape(obs_tensors[key].shape[0], -1) for key in sorted_keys], dim=-1)
            for key in sorted_keys:
                if obs_tensors[key].ndim != 2:
                    raise ValueError(
                        f"Observation term '{key}' in concatenate=True group '{group_name}' must be "
                        f"2-D [num_envs, feature], got shape {tuple(obs_tensors[key].shape)}. Image terms "
                        f"(e.g. camera_rgb/camera_depth) belong only in a concatenate=False (dict) group."
                    )
            return torch.cat([obs_tensors[key] for key in sorted_keys], dim=-1)
        # Non-concatenated group: flatten any per-term history to 2D ``[num_envs, history_length * term_dim]``.
        if group_cfg.history_length > 1:
            return {key: value.reshape(value.shape[0], -1) for key, value in obs_tensors.items()}
        return obs_tensors

    def _compute_term(
        self,
        group_name: str,
        term_name: str,
        term_cfg: ObsTermCfg,
        *,
        modify_history: bool = True,
    ) -> torch.Tensor:
        """Compute a single observation term.

        Parameters
        ----------
        group_name : str
            Name of the observation group.
        term_name : str
            Name of the observation term.
        term_cfg : ObsTermCfg
            Configuration for the observation term.
        modify_history : bool
            Whether a stateful term may advance its internal buffers.

        Returns
        -------
        torch.Tensor
            Observation tensor with shape ``[num_envs, obs_dim]``.
        """
        # Check if stateful or stateless
        if term_name in self._term_instances[group_name]:
            # Stateful term
            instance = self._term_instances[group_name][term_name]
            if self._term_supports_modify_history[group_name][term_name]:
                obs = instance(self.env, modify_history=modify_history, **term_cfg.params)
            else:
                obs = instance(self.env, **term_cfg.params)
        else:
            # Stateless function
            func = self._term_funcs[group_name][term_name]
            obs = func(self.env, **term_cfg.params)

        return obs.clone()

    def _apply_noise(self, obs: torch.Tensor, noise_scale: float) -> torch.Tensor:
        """Apply uniform observation noise.

        The implementation matches the direct path, sampling uniform noise in
        ``[-noise_scale, noise_scale]``.

        Parameters
        ----------
        obs : torch.Tensor
            Observation tensor.
        noise_scale : float
            Magnitude of the uniform noise.

        Returns
        -------
        torch.Tensor
            Noisy observation tensor.
        """
        # Direct uses: (torch.rand_like(obs) * 2.0 - 1.0) * noise_scale
        noise = (torch.rand_like(obs) * 2.0 - 1.0) * noise_scale
        return obs + noise

    def _apply_scale(self, obs: torch.Tensor, scale: float | tuple) -> torch.Tensor:
        """Apply scaling to an observation tensor.

        Parameters
        ----------
        obs : torch.Tensor
            Observation tensor.
        scale : float or tuple
            Scalar or per-dimension scale factors.

        Returns
        -------
        torch.Tensor
            Scaled observation tensor.
        """
        return obs * scale

    def _apply_history(
        self, group_name: str, term_name: str, obs: torch.Tensor, group_cfg: ObsGroupCfg, *, modify_buffer: bool = True
    ) -> torch.Tensor:
        """Apply history buffering to an observation term.

        Maintains a circular buffer of past observations and returns them
        concatenated along the feature dimension.

        Parameters
        ----------
        group_name : str
            Name of the observation group.
        term_name : str
            Name of the observation term.
        obs : torch.Tensor
            Current observation tensor with shape ``[num_envs, obs_dim]``.
        group_cfg : ObsGroupCfg
            Configuration for the observation group (contains ``history_length``).
        modify_buffer : bool, optional
            If ``True``, append to the history buffer; if ``False``, preserve the
            buffer contents. Defaults to ``True``.

        Returns
        -------
        torch.Tensor
            Historical observations with shape ``[num_envs, history_length, obs_dim]``.
            The caller flattens this in time-major order at the group level (oldest
            timestep first), matching FAR-FALCON-Terrains.
        """
        buffer = self._history_buffers[group_name][term_name]

        if modify_buffer:
            # Add current observation to buffer
            buffer.append(obs)
            history = list(buffer)
        else:
            # Don't modify buffer - create temporary history with current obs
            history = list(buffer) + [obs]
            # Trim to history_length if needed
            if len(history) > group_cfg.history_length:
                history = history[-group_cfg.history_length :]

        # If buffer not full yet, pad with zeros (same as direct behavior)
        if len(history) < group_cfg.history_length:
            num_missing = group_cfg.history_length - len(history)
            obs_dim = obs.shape[1]
            padding = [torch.zeros(self.env.num_envs, obs_dim, device=self.device) for _ in range(num_missing)]
            history = padding + history

        # Stack along time dimension: [num_envs, history_length, obs_dim].
        # Flattening is deferred to compute_group so terms can be interleaved
        # per-timestep (time-major) rather than grouped per-term.
        return torch.stack(history, dim=1)

    def _apply_frame_buffer(
        self, group_name: str, term_name: str, obs: torch.Tensor, group_cfg: ObsGroupCfg, *, modify_buffer: bool = True
    ) -> torch.Tensor:
        """Interval-gated history buffer matching FAR-FALCON-Terrains ``_post_config_observation_callback``.

        Envs whose counter satisfies ``counter % interval == 0`` shift their buffer
        by one frame and append ``obs``; the others keep their previous frames.

        Returns
        -------
        torch.Tensor
            ``[num_envs, history_length, obs_dim]`` when ``history_length > 1``,
            otherwise ``[num_envs, obs_dim]``.
        """
        buffer = self._frame_buffers[group_name].get(term_name)
        if buffer is None:
            buffer = torch.zeros(
                obs.shape[0], group_cfg.history_length, obs.shape[1], device=obs.device, dtype=obs.dtype
            )
        push = (self._interval_counters[group_name] % group_cfg.interval) == 0
        pushed = torch.cat([buffer[:, 1:], obs.unsqueeze(1)], dim=1)
        buffer = torch.where(push[:, None, None], pushed, buffer)
        if modify_buffer:
            self._frame_buffers[group_name][term_name] = buffer
        if group_cfg.history_length > 1:
            return buffer
        return buffer[:, 0]

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset observation history and stateful terms.

        Parameters
        ----------
        env_ids : torch.Tensor or None, optional
            Environment IDs to reset. If ``None``, reset all environments.
        """
        # Normalize environment ids
        env_ids_tensor: torch.Tensor | None
        if env_ids is None:
            env_ids_tensor = None
        elif isinstance(env_ids, torch.Tensor):
            env_ids_tensor = env_ids.to(device=self.device, dtype=torch.long)
        else:
            env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        # Reset or clear history buffers
        for group_buffers in self._history_buffers.values():
            for buffer in group_buffers.values():
                if env_ids_tensor is None:
                    buffer.clear()
                else:
                    if env_ids_tensor.numel() == 0:
                        continue
                    for history in buffer:
                        history[env_ids_tensor] = 0.0

        for group_name, counter in getattr(self, "_interval_counters", {}).items():
            if env_ids_tensor is None:
                counter.zero_()
            else:
                counter[env_ids_tensor] = 0
            if not self.cfg.groups[group_name].reset_history_on_env_reset:
                continue
            for frames in self._frame_buffers[group_name].values():
                if env_ids_tensor is None:
                    frames.zero_()
                else:
                    frames[env_ids_tensor] = 0.0

        # Reset stateful term instances
        for group_instances in self._term_instances.values():
            for instance in group_instances.values():
                instance.reset(env_ids_tensor)

    def get_obs_dims(self) -> dict[str, int | dict[str, int | tuple[int, ...]]]:
        """Get observation dimensions for each group.

        Concatenate groups require 2-D terms and report a single summed int. Dict groups
        return per-term feature size: an int for 2-D terms, or the trailing-dim shape tuple
        (``obs.shape[1:]``) for image terms.

        Returns
        -------
        dict[str, int or dict[str, int or tuple]]
            Mapping from group names to observation dimensions. Values are integers
            when the group concatenates terms, otherwise dictionaries of per-term
            dimensions.
        """
        dims: dict[str, int | dict[str, int | tuple[int, ...]]] = {}
        for group_name, group_cfg in self.cfg.groups.items():
            if group_cfg.concatenate:
                # Sum up all term dimensions
                total_dim = 0
                for term_name, term_cfg in group_cfg.terms.items():
                    # Compute term once to get its dimension
                    obs = self._compute_term(group_name, term_name, term_cfg, modify_history=False)
                    if obs.ndim != 2:
                        raise ValueError(
                            f"Observation term '{term_name}' in concatenate=True group '{group_name}' must be "
                            f"2-D [num_envs, feature], got shape {tuple(obs.shape)}. Image terms "
                            f"(e.g. camera_rgb/camera_depth) belong only in a concatenate=False (dict) group."
                        )
                    term_dim = obs.shape[1]

                    # Account for history at group level
                    if group_cfg.history_length > 1:
                        term_dim *= group_cfg.history_length

                    total_dim += term_dim
                dims[group_name] = total_dim
            else:
                # Return dict of per-term feature sizes: an int for 2-D terms, or the
                # trailing-dim shape tuple for image terms ([N, H, W, C] -> (H, W, C)).
                term_dims: dict[str, int | tuple[int, ...]] = {}
                for term_name, term_cfg in group_cfg.terms.items():
                    obs = self._compute_term(group_name, term_name, term_cfg, modify_history=False)
                    term_dims[term_name] = obs.shape[1] if obs.ndim == 2 else tuple(obs.shape[1:])
                dims[group_name] = term_dims
        return dims

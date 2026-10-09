"""Base class for camera terms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

    from holosoma.config_types.camera import CameraTermCfg


class CameraTermBase(ABC):
    """Base class for stateful camera terms."""

    def __init__(self, cfg: CameraTermCfg, env: Any):
        """Initialize camera term.

        Parameters
        ----------
        cfg : CameraTermCfg
            Configuration for this camera term
        env : Any
            Environment instance (typically BaseTask or subclass)
        """
        self._cfg = cfg
        self.env = env
        self.num_envs = env.num_envs
        self.device = env.device
        self.camera_rendered: bool = False
        """Set to ``True`` by the manager when this term produced a new frame."""

        self.capture_fps: float = 10.0
        """Capture rate in Hz.  Subclasses should set this from their props."""

        self.camera_names: list[str] = ["default"]
        """Ordered names of cameras managed by this term.  Subclasses with
        multi-camera support should override this in ``__init__``."""

        self.needs_reset: bool = False
        """When ``True`` the manager calls :meth:`reset` every env step with the
        envs flagged in ``env.reset_buf`` (i.e. envs reset during this step)."""

    @abstractmethod
    def setup(self) -> None:
        """Setup hook called once during environment initialization."""

    def reset(self, env_ids: torch.Tensor) -> None:  # noqa: B027
        """Clear per-env state for the given (just reset) environments.

        Default implementation is a no-op.
        """

    def step(self) -> None:  # noqa: B027
        """Per-step update (e.g. refresh camera pose from rigid-body state).

        Called by ``CameraManager.step()`` every environment step.
        Override in subclasses that need per-step bookkeeping; the default
        implementation is a no-op.
        """

    @abstractmethod
    def capture(self) -> None:
        """Capture camera outputs for the current simulation step."""

    @abstractmethod
    def draw_debug_viz(self) -> None:
        """Draw debug visualization."""

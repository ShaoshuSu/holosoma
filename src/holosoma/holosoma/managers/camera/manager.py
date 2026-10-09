"""Camera manager coordinating camera hooks."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from loguru import logger

from holosoma.config_types.camera import CameraManagerCfg
from holosoma.managers.utils import resolve_callable

from .base import CameraTermBase


class CameraManager:
    """Drive camera terms at setup and capture boundaries.

    Supports per-term FPS-based capture decimation: ``render_cameras()``
    should be called every environment step.  Internally the manager counts
    steps per term and only calls ``step()`` + ``capture()`` when each
    term's ``capture_fps`` (sourced from ``CameraProps.frame_rate``)
    dictates a new frame is due.

    Parameters
    ----------
    cfg : CameraManagerCfg
        Camera manager configuration specifying terms and parameters.
    env : Any
        Environment instance operated on by camera terms.
    device : str
        Device identifier used by camera terms.
    """

    def __init__(self, cfg: CameraManagerCfg, env: Any, device: str):
        self.cfg = cfg
        self.env = env
        self.device = device

        self.camera_rendered: bool = False
        """``True`` when the most recent ``render_cameras()`` produced at least one new frame."""

        self.camera_terms: dict[str, CameraTermBase] = {}
        self._capture_counters: dict[str, int] = {}
        self._capture_intervals: dict[str, int] = {}
        self._term_initialized: dict[str, bool] = {}

        for term_name, term_cfg in self.cfg.terms.items():
            self.camera_terms[term_name] = resolve_callable(
                term_cfg.func,
                context=f"camera term '{term_name}'",
            )(term_cfg, self.env)
            self._capture_counters[term_name] = 0
            self._capture_intervals[term_name] = 1
            self._term_initialized[term_name] = False

    @property
    def has_cameras(self) -> bool:
        return len(self.camera_terms) > 0

    def setup(self) -> None:
        """Run setup hooks and compute per-term capture intervals."""
        for term in self.camera_terms.values():
            term.setup()

        if not self.has_cameras:
            return

        env_dt: float = getattr(self.env, "dt", None)

        for term_name, term in self.camera_terms.items():
            fps = term.capture_fps
            if env_dt is not None and env_dt > 0 and fps > 0:
                interval = max(1, round(1.0 / (fps * env_dt)))
            else:
                interval = 1
            self._capture_intervals[term_name] = interval
            logger.info(
                f"CameraManager[{term_name}]: capture_fps={fps}, env_dt={env_dt}, capture_interval={interval} env steps"
            )

    # ------------------------------------------------------------------
    # Main entry point - call once per env step
    # ------------------------------------------------------------------

    def render_cameras(self) -> None:
        """Update camera poses and conditionally capture new frames.

        This method should be called **once per environment step** (i.e. after
        physics + env resets, before observations).  It increments per-term
        counters and only performs the (expensive) ray-cast capture when each
        term's ``capture_fps`` dictates a new frame is due.

        After this call, ``self.camera_rendered`` indicates whether *any* term
        produced a new frame.  Each term's ``camera_rendered`` flag is updated
        independently.
        """
        if not self.has_cameras:
            return

        self.camera_rendered = False

        # Envs reset during this step are still flagged in ``reset_buf`` here
        # (resets run before rendering); clear their camera state before capture,
        # as FAR-FALCON-Terrains does in ``reset_envs_idx``.
        if any(term.needs_reset for term in self.camera_terms.values()):
            reset_buf = getattr(self.env, "reset_buf", None)
            if reset_buf is not None:
                env_ids = reset_buf.nonzero(as_tuple=False).flatten()
                if env_ids.numel() > 0:
                    self.reset(env_ids)

        for term_name, term in self.camera_terms.items():
            self._capture_counters[term_name] += 1
            should_capture = (
                self._capture_counters[term_name] >= self._capture_intervals[term_name]
                or not self._term_initialized[term_name]
            )

            if should_capture:
                self.camera_rendered = True
                self._capture_counters[term_name] = 0
                self._term_initialized[term_name] = True

                term.step()
                term.capture()
                term.camera_rendered = True

                if self.cfg.image_show:
                    self._show_depth(term_name, term)
            else:
                term.camera_rendered = False

    # ------------------------------------------------------------------
    # Debug visualization
    # ------------------------------------------------------------------

    def _show_depth(self, term_name: str, term: CameraTermBase) -> None:
        """Display depth images from env 0 using OpenCV (one window per camera)."""
        try:
            import cv2
        except ImportError:
            logger.warning("image_show requires opencv-python (`pip install opencv-python`)")
            return

        buf = getattr(term, "depth_buffer", None)
        if buf is None:
            return

        num_sensors = buf.shape[1]
        camera_names = getattr(term, "camera_names", [str(i) for i in range(num_sensors)])
        far = getattr(term, "far_plane", 1.0)

        # Sync before read so async CUDA errors are reported here; clone to avoid
        # reading buffer that may be written by Warp/other kernels.
        if buf.is_cuda:
            torch.cuda.synchronize()
        for s in range(num_sensors):
            frame = buf[0, s].clone().detach().cpu().numpy()
            if far > 0:
                frame_norm = np.clip(frame / far, 0.0, 1.0)
            else:
                frame_norm = np.zeros_like(frame)
            frame_u8 = (frame_norm * 255).astype(np.uint8)
            cam_label = camera_names[s] if s < len(camera_names) else str(s)
            cv2.imshow(f"depth/{term_name}/{cam_label}", frame_u8)
        cv2.waitKey(1)

    def step(self) -> None:
        """Run step hooks."""
        # camera manager will be called every env step, but one should be able to control
        for term in self.camera_terms.values():
            term.step()

    def reset(self, env_ids: torch.Tensor) -> None:
        """Clear per-env camera state (e.g. depth delay buffers) for ``env_ids``."""
        for term in self.camera_terms.values():
            if term.needs_reset:
                term.reset(env_ids)

    def capture(self) -> None:
        """Capture camera outputs for the current step."""
        for term in self.camera_terms.values():
            term.capture()

    def draw_debug_viz(self) -> None:
        for term in self.camera_terms.values():
            term.draw_debug_viz()

    def get_state(self, term_name: str) -> CameraTermBase:
        """Retrieve a stateful camera term by name.

        Parameters
        ----------
        term_name : str
            Name of the camera term.

        Returns
        -------
        CameraTermBase
            Stateful camera term instance.
        """
        return self.camera_terms[term_name]

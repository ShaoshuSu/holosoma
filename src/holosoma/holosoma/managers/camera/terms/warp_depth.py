"""Warp-based depth camera term.

Uses NVIDIA Warp ray casting against terrain (and optionally robot body) meshes
to produce per-environment depth images at GPU speed, following the approach from
RPL/humanoidverse/simulator/warp_sensors/camera_sensor.py.

A single ``WarpDepthCamera`` instance can handle **multiple** cameras (poses)
rendered in one kernel launch via the ``cam_id`` dimension, avoiding the overhead
of sequential per-camera kernel dispatches.

Params require ``poses``: a ``dict[str, CameraPose]`` mapping camera names to
their mounting specifications.
"""

from __future__ import annotations

import math
import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
import warp as wp
from loguru import logger

from holosoma.config_types.camera import CameraPose, CameraProps
from holosoma.managers.camera.base import CameraTermBase
from holosoma.managers.camera.terms.warp_kernels import (
    depth_range_dynamic_kernel,
    depth_range_dynamic_per_env_fov_kernel,
    depth_range_kernel,
    depth_range_per_env_fov_kernel,
)
from holosoma.utils.rotations import quat_apply, quat_from_euler_xyz, quat_mul

DEFAULT_EULER_FRAME_ROT_DEG = [-90.0, 0.0, -90.0]


def _coerce_camera_props(props: CameraProps | dict[str, Any]) -> CameraProps:
    """Restore CameraProps after YAML/checkpoint round-trip leaves a plain dict."""
    if isinstance(props, CameraProps):
        return props
    if isinstance(props, dict):
        return CameraProps(**props)
    raise TypeError(f"Expected CameraProps or dict, got {type(props)}")


def _coerce_camera_poses(poses: dict[str, CameraPose | dict[str, Any]]) -> list[CameraPose]:
    """Restore CameraPose entries after YAML/checkpoint round-trip."""
    return [pose if isinstance(pose, CameraPose) else CameraPose(**pose) for pose in poses.values()]


def _convert_to_warp_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    device: str,
) -> wp.Mesh:
    return wp.Mesh(
        points=wp.array(vertices.astype(np.float32), dtype=wp.vec3, device=device),
        indices=wp.array(faces.astype(np.int32).flatten(), dtype=wp.int32, device=device),
    )


def resize_depth(
    depth: torch.Tensor,
    size: tuple[int, int],
    mode: str = "bicubic",
    antialias: bool = True,
) -> torch.Tensor:
    """Resize ``(N, C, H, W)`` depth images to ``size = (h, w)``.

    Equivalent to ``torchvision.transforms.Resize(size, interpolation=mode)``
    applied to a float tensor (torchvision >= 0.17 defaults to
    ``antialias=True`` for tensors), which is what FAR-FALCON-Terrains uses.
    """
    align_corners = False if mode in ("linear", "bilinear", "bicubic", "trilinear") else None
    return F.interpolate(
        depth,
        size=size,
        mode=mode,
        align_corners=align_corners,
        antialias=antialias and mode in ("bilinear", "bicubic"),
    )


class WarpDepthCamera(CameraTermBase):
    """Warp ray-cast depth camera term.

    Produces depth (or range) images each time :meth:`capture` is called by
    casting rays from a pinhole camera model against the terrain warp mesh.

    Supports **multiple cameras** in a single term so they are rendered in one
    GPU kernel launch (parallel).  Use ``poses`` to specify a name -> CameraPose
    mapping; the depth buffer will have shape ``(N, S, H, W)`` where ``S`` is
    the number of cameras.

    Expected ``CameraTermCfg.params`` keys
    ---------------------------------------
    poses : dict[str, CameraPose]
        Named camera mounting specifications.  Each camera can reference a
        different body link, offset, and rotation.  Cameras are rendered in
        parallel via the ``cam_id`` kernel dimension.
    props : CameraProps
        Camera intrinsics and capture settings (shared across all cameras).
    euler_frame_rot_deg : list[float], optional
        Euler angles ``[roll, pitch, yaw]`` in degrees that rotate from the
        warp camera convention (z-forward) into the robot body convention
        (x-forward).  Default ``[-90, 0, -90]``.
    calculate_depth : bool, optional
        If ``True`` (default), project ray distances onto the principal axis to
        obtain depth; ``False`` returns raw range.
    fov_randomization : dict, optional
        Per-environment horizontal-FOV randomization.  Keys:
        ``enable`` (bool, default ``False``),
        ``min_fov_deg`` (float, lower bound in degrees),
        ``max_fov_deg`` (float, upper bound in degrees).
        When enabled each environment gets its own intrinsic matrix
        sampled uniformly from ``[min_fov_deg, max_fov_deg]``.
        Call :meth:`resample_fov` to re-draw (e.g. on episode reset).
    dynamic_meshes : dict, optional
        Robot-body ray casting for self-occlusion.  Keys:
        ``enable`` (bool, default ``False``),
        ``body_meshes`` (dict mapping *mesh_file* -> *body_name*),
        ``meshes_root`` (str, base directory for the mesh files).
        When enabled the depth kernel also casts against per-environment
        body meshes whose poses are updated from the rigid-body state
        every :meth:`step`.
    noise : dict, optional
        Sensor noise parameters.  Keys: ``enable`` (bool), ``std_multiplier``
        (float), ``dropout_prob`` (float), ``dropout_value`` (float).
    random_depth_occlusion : dict, optional
        Border-strip depth occlusion (ports FAR-FALCON-Terrains
        ``random_depth_occlusion``).  Per env, a mode is sampled from
        ``{none, small, large}`` with terrain-biased probabilities; ``small``
        masks reduced left/right strips, ``large`` masks all four borders.
        Masked pixels are overwritten with uniform random depth in
        ``[near_clip, far_clip]``.
        Keys: ``enable`` (bool, default ``False``),
        ``mask`` (list[int], terrain-type indices that bias toward ``large``;
        complement biases toward ``small``),
        ``box`` ([x1, x2, y1, y2] in **native** camera pixels — auto-scaled to
        the ray-cast resolution),
        ``prob`` (float, probability that occlusion is applied at all),
        ``diff`` (int, native-pixel reduction applied to ``x1``/``x2`` for the
        ``small`` mode),
        ``terrain_bias``, ``p_none``, ``p_small``, ``p_large`` (floats),
        ``share_across_cameras`` (bool, default ``False``; ``True`` fills the
        masked strips of every camera of an env from one random image, as
        FALCON does).
    pose_randomization : dict, optional
        Per-env, per-camera mounting randomization sampled once at init
        (FALCON ``randomize_placement``).  Keys: ``enable`` (bool, default
        ``False``), ``translation_min`` / ``translation_max`` (xyz offsets in
        m added to ``camera_offset``), ``rotation_min_deg`` /
        ``rotation_max_deg`` (roll/pitch/yaw offsets in degrees added to
        ``camera_rotation`` before converting to a quaternion).
    resize : dict, optional
        Post-processing of the ray-cast image.  Keys: ``enable`` (bool,
        default ``False``), ``mode`` (default ``"bicubic"``), ``antialias``
        (default ``True``).  When enabled each capture clips the (noised,
        occluded) ray-cast image to ``[near_clip, far_clip]`` and resizes it
        to ``props.resized_height x props.resized_width`` (see
        :func:`resize_depth`).
    ray_cast_width, ray_cast_height : int, optional
        Ray-cast resolution.  Defaults to ``props.resized_*``.
    ray_far_plane : float, optional
        Maximum ray distance.  Defaults to ``props.far_clip`` (FALCON uses
        ``far_clip + 0.5``).

    The depth delay ring buffer is configured through
    ``props.depth_buffer_len`` / ``props.depth_delay`` /
    ``props.depth_delay_range``.  Observations should read
    :attr:`output_depth`, which is the delayed, post-processed frame when
    those features are enabled and ``depth_buffer`` otherwise.
    """

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)
        params = self._cfg.params

        poses_dict: dict[str, CameraPose | dict[str, Any]] = params["poses"]
        self.camera_names: list[str] = list(poses_dict.keys())
        self.poses = _coerce_camera_poses(poses_dict)

        self.props = _coerce_camera_props(params["props"])
        self.euler_frame_rot_deg = params.get("euler_frame_rot_deg", DEFAULT_EULER_FRAME_ROT_DEG)
        self.calculate_depth = params.get("calculate_depth", True)
        self.noise_cfg = params.get("noise", {"enable": False})
        self.fov_cfg = params.get("fov_randomization", {"enable": False})
        self.dyna_cfg = params.get("dynamic_meshes", {"enable": False})
        self.occlusion_cfg = params.get("random_depth_occlusion", {"enable": False})
        self.pose_rand_cfg = params.get("pose_randomization", {"enable": False})
        self.resize_cfg = params.get("resize", {"enable": False})

        self.ray_width = params.get("ray_cast_width", self.props.resized_width)
        self.ray_height = params.get("ray_cast_height", self.props.resized_height)
        self.far_plane = float(params.get("ray_far_plane", self.props.far_clip))
        self.capture_fps = float(self.props.frame_rate)

        self._init_postprocess()

        self._num_sensors = len(self.poses)
        self._randomize_fov: bool = self.fov_cfg.get("enable", False)
        self._use_dynamic_meshes: bool = self.dyna_cfg.get("enable", False)
        self._graph: wp.Graph | None = None

        self._camera_name_to_idx: dict[str, int] = {name: i for i, name in enumerate(self.camera_names)}

        self._init_camera_matrices()
        self._init_tensors()

    # ------------------------------------------------------------------
    # Post-processing / delay configuration
    # ------------------------------------------------------------------

    def _init_postprocess(self) -> None:
        self._use_resize: bool = bool(self.resize_cfg.get("enable", False))
        self._resize_mode: str = self.resize_cfg.get("mode", "bicubic")
        self._resize_antialias: bool = bool(self.resize_cfg.get("antialias", True))

        self._delay_buffer_len = int(self.props.depth_buffer_len)
        self._use_delay: bool = self._delay_buffer_len > 1
        delay_range = self.props.depth_delay_range
        if delay_range is None:
            delay_range = [self.props.depth_delay, self.props.depth_delay]
        self._delay_min, self._delay_max = int(delay_range[0]), int(delay_range[1])
        if self._use_delay:
            if not 0 <= self._delay_min <= self._delay_max:
                raise ValueError(f"Invalid depth delay range {list(delay_range)}; need 0 <= min <= max")
            self._delay_buffer_len = max(self._delay_buffer_len, self._delay_max + 1)

        self._postprocess: bool = self._use_resize or self._use_delay
        self.output_preclipped: bool = self._postprocess
        """``True`` when :attr:`output_depth` was clipped to ``[near, far]``
        before resizing, so observation terms must not clip it again."""
        self.needs_reset = self._use_delay

        if self._use_resize:
            self.out_height, self.out_width = int(self.props.resized_height), int(self.props.resized_width)
        else:
            self.out_height, self.out_width = int(self.ray_height), int(self.ray_width)

    # ------------------------------------------------------------------
    # Camera intrinsics
    # ------------------------------------------------------------------

    def _init_camera_matrices(self) -> None:
        W, H = float(self.ray_width), float(self.ray_height)
        self._c_x = int(W / 2.0)
        self._c_y = int(H / 2.0)

        if self._randomize_fov:
            self._K_inv = None  # not used in per-env mode
            self._init_per_env_fov()
        else:
            self._K_inv_array = None
            self._wp_K_inv_array = None

            u_0, v_0 = W / 2.0, H / 2.0
            hfov_rad = math.radians(self.props.horizontal_fov)
            f = W / 2.0 / math.tan(hfov_rad / 2.0)
            vfov = 2.0 * math.atan(H / (2.0 * f))
            alpha_u = u_0 / math.tan(hfov_rad / 2.0)
            alpha_v = v_0 / math.tan(vfov / 2.0)

            K = wp.mat44(
                alpha_u,
                0.0,
                u_0,
                0.0,
                0.0,
                alpha_v,
                v_0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            )
            self._K_inv = wp.inverse(K)

    # ------------------------------------------------------------------
    # Per-environment FOV randomisation
    # ------------------------------------------------------------------

    def _init_per_env_fov(self) -> None:
        """Create per-env FOV tensors and sample initial values."""
        N, S = self.num_envs, self._num_sensors
        dev = self.device

        min_deg = self.fov_cfg.get("min_fov_deg", self.props.horizontal_fov - 2.0)
        max_deg = self.fov_cfg.get("max_fov_deg", self.props.horizontal_fov + 2.0)

        self._min_fov_rad = torch.full((N, S), math.radians(min_deg), device=dev)
        self._max_fov_rad = torch.full((N, S), math.radians(max_deg), device=dev)
        self._fov_rad = torch.zeros(N, S, device=dev)

        self.resample_fov()

    def resample_fov(self, env_ids: torch.Tensor | None = None) -> None:
        """(Re-)sample per-environment FOV and recompute K_inv matrices.

        Parameters
        ----------
        env_ids : torch.Tensor or None
            Environment indices to resample.  ``None`` resamples all envs.
            Pass a 1-D long tensor of env indices after episode resets to
            re-randomise only those environments.
        """
        if not self._randomize_fov:
            return

        if env_ids is None:
            self._fov_rad[:] = (self._max_fov_rad - self._min_fov_rad) * torch.rand_like(
                self._fov_rad
            ) + self._min_fov_rad
        else:
            n = env_ids.shape[0]
            self._fov_rad[env_ids] = (self._max_fov_rad[env_ids] - self._min_fov_rad[env_ids]) * torch.rand(
                n, self._num_sensors, device=self.device
            ) + self._min_fov_rad[env_ids]

        self._rebuild_K_inv_array()
        # Per-env FOV changes the intrinsic matrix so the CUDA graph must be
        # recaptured on the next capture() call.
        self._graph = None

    def _rebuild_K_inv_array(self) -> None:  # noqa: N802
        """Vectorised K_inv computation from the current ``_fov_rad`` tensor."""
        W, H = float(self.ray_width), float(self.ray_height)
        u_0, v_0 = W / 2.0, H / 2.0

        fov_half = self._fov_rad / 2.0
        tan_half = torch.tan(fov_half)
        alpha_u = u_0 / tan_half
        f = (W / 2.0) / tan_half
        vfov = 2.0 * torch.atan(H / (2.0 * f))
        alpha_v = v_0 / torch.tan(vfov / 2.0)

        N, S = self._fov_rad.shape
        K = torch.zeros(N, S, 4, 4, device=self.device, dtype=torch.float32)
        K[..., 0, 0] = alpha_u
        K[..., 0, 2] = u_0
        K[..., 1, 1] = alpha_v
        K[..., 1, 2] = v_0
        K[..., 2, 2] = 1.0
        K[..., 3, 3] = 1.0

        K_inv = torch.linalg.inv(K)  # (N, S, 4, 4)
        self._K_inv_array = K_inv.reshape(N, S, 16).contiguous()
        self._wp_K_inv_array = wp.from_torch(self._K_inv_array, dtype=wp.float32)

    # ------------------------------------------------------------------
    # Tensor / buffer initialization
    # ------------------------------------------------------------------

    def _init_tensors(self) -> None:
        N = self.num_envs
        S = self._num_sensors
        dev = self.device

        self.depth_buffer = torch.zeros(
            N,
            S,
            self.ray_height,
            self.ray_width,
            device=dev,
            dtype=torch.float32,
        )

        # Frame-convention quaternion (camera z-fwd -> robot x-fwd)
        frame_rad = torch.deg2rad(torch.tensor(self.euler_frame_rot_deg, device=dev))
        frame_quat = quat_from_euler_xyz(frame_rad[0:1], frame_rad[1:2], frame_rad[2:3])  # (1, 4)
        self._frame_quat = frame_quat.expand(N, -1).contiguous()  # (N, 4)

        # Per-camera local poses relative to their body links
        self._local_pos_list: list[torch.Tensor] = []
        self._local_quat_list: list[torch.Tensor] = []
        for pose in self.poses:
            if self.pose_rand_cfg.get("enable", False):
                lp, lq = self._sample_local_pose(pose)
                self._local_pos_list.append(lp)
                self._local_quat_list.append(lq)
                continue
            lp = (
                torch.tensor(
                    pose.camera_offset,
                    device=dev,
                    dtype=torch.float32,
                )
                .unsqueeze(0)
                .expand(N, -1)
                .contiguous()
            )  # (N, 3)
            self._local_pos_list.append(lp)

            rot_deg = torch.tensor(pose.camera_rotation, device=dev, dtype=torch.float32)
            rot_rad = torch.deg2rad(rot_deg)
            lq = quat_from_euler_xyz(rot_rad[0:1], rot_rad[1:2], rot_rad[2:3])  # (1, 4)
            self._local_quat_list.append(lq.expand(N, -1).contiguous())  # (N, 4)

        # Global camera pose (written each step, read by warp kernel)
        self._global_pos = torch.zeros(N, S, 3, device=dev, dtype=torch.float32)
        self._global_quat = torch.zeros(N, S, 4, device=dev, dtype=torch.float32)
        self._global_quat[..., 3] = 1.0  # identity

        # Warp views (zero-copy over the torch tensors)
        self._wp_pos = wp.from_torch(self._global_pos, dtype=wp.vec3)
        self._wp_quat = wp.from_torch(self._global_quat, dtype=wp.quat)
        self._wp_pixels = wp.from_torch(self.depth_buffer, dtype=wp.float32)

        if self._postprocess:
            self.processed_buffer = torch.zeros(N, S, self.out_height, self.out_width, device=dev, dtype=torch.float32)
        if self._use_delay:
            # Value that ``cam_depth`` normalizes to 0, i.e. FALCON's zeroed
            # (already normalized) delay buffer.
            self._delay_fill_value = 0.5 * (self.props.near_clip + self.props.far_clip)
            self.delay_buffer = torch.full(
                (N, self._delay_buffer_len, S, self.out_height, self.out_width),
                self._delay_fill_value,
                device=dev,
                dtype=torch.float32,
            )
            self.delay_idx = torch.randint(self._delay_min, self._delay_max + 1, (N,), device=dev)
            self._env_arange = torch.arange(N, device=dev)

    def _sample_local_pose(self, pose: CameraPose) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a per-env local pose around ``pose`` (FALCON ``randomize_placement``).

        Offsets are drawn uniformly from ``[min, max]`` and added to the
        nominal translation / Euler angles; the perturbed Euler angles are then
        converted to a quaternion (not composed with the nominal rotation).
        """
        N, dev = self.num_envs, self.device
        cfg = self.pose_rand_cfg

        def _uniform(lo: list[float], hi: list[float]) -> torch.Tensor:
            lo_t = torch.tensor(lo, device=dev, dtype=torch.float32)
            hi_t = torch.tensor(hi, device=dev, dtype=torch.float32)
            return (hi_t - lo_t) * torch.rand(N, 3, device=dev) + lo_t

        nominal_pos = torch.tensor(pose.camera_offset, device=dev, dtype=torch.float32)
        nominal_rot = torch.tensor(pose.camera_rotation, device=dev, dtype=torch.float32)
        lp = nominal_pos + _uniform(
            cfg.get("translation_min", [0.0, 0.0, 0.0]), cfg.get("translation_max", [0.0, 0.0, 0.0])
        )
        rot_rad = torch.deg2rad(
            nominal_rot
            + _uniform(cfg.get("rotation_min_deg", [0.0, 0.0, 0.0]), cfg.get("rotation_max_deg", [0.0, 0.0, 0.0]))
        )
        lq = quat_from_euler_xyz(rot_rad[:, 0], rot_rad[:, 1], rot_rad[:, 2])
        return lp.contiguous(), lq.contiguous()

    # ------------------------------------------------------------------
    # CameraTermBase interface
    # ------------------------------------------------------------------

    def setup(self) -> None:
        terrain_state = self.env.terrain_manager.get_state("locomotion_terrain")
        self._terrain_wp_mesh = terrain_state.warp_mesh
        self._terrain_mesh_id = self._terrain_wp_mesh.id
        self._terrain_state = terrain_state

        sim = self.env.simulator
        self._body_indices: list[int] = [sim.find_rigid_body_indice(pose.camera_body_link) for pose in self.poses]

        if self._use_dynamic_meshes:
            self._setup_dynamic_meshes()

        if self.occlusion_cfg.get("enable", False):
            self._setup_occlusion()

        fov_info = "shared"
        if self._randomize_fov:
            min_d = self.fov_cfg.get("min_fov_deg", self.props.horizontal_fov - 2.0)
            max_d = self.fov_cfg.get("max_fov_deg", self.props.horizontal_fov + 2.0)
            fov_info = f"per-env [{min_d:.1f}, {max_d:.1f}] deg"
        dyna_info = f", dynamic_bodies={self._num_dyna_bodies}" if self._use_dynamic_meshes else ""
        cam_details = ", ".join(
            f"{name}@'{pose.camera_body_link}'(idx={idx})"
            for name, pose, idx in zip(self.camera_names, self.poses, self._body_indices)
        )
        logger.info(
            f"WarpDepthCamera setup: cameras=[{cam_details}], "
            f"resolution={self.ray_width}x{self.ray_height}, "
            f"far={self.far_plane}m, fov={fov_info}{dyna_info}"
        )

    def _setup_dynamic_meshes(self) -> None:
        """Load robot body meshes and resolve their rigid-body indices."""
        body_meshes: dict[str, str] = self.dyna_cfg.get("body_meshes", {})
        meshes_root: str = self.dyna_cfg.get("meshes_root", "")

        self._robot_wp_meshes: list[wp.Mesh] = []
        self._dyna_body_indices: list[int] = []

        sim = self.env.simulator
        use_global = hasattr(sim, "find_rigid_body_indice_global")

        for mesh_file, body_name in body_meshes.items():
            mesh_path = os.path.join(meshes_root, mesh_file)
            mesh = trimesh.load(mesh_path)
            wp_mesh = _convert_to_warp_mesh(mesh.vertices, mesh.faces, self.device)
            self._robot_wp_meshes.append(wp_mesh)

            if use_global:
                idx = sim.find_rigid_body_indice_global(body_name)
            else:
                idx = sim.find_rigid_body_indice(body_name)
            self._dyna_body_indices.append(idx)
            logger.debug(f"WarpDepthCamera: loaded dynamic mesh '{mesh_file}' -> body '{body_name}'")

        self._num_dyna_bodies = len(self._robot_wp_meshes)
        if self._num_dyna_bodies == 0:
            self._use_dynamic_meshes = False
            logger.warning("WarpDepthCamera: dynamic_meshes enabled but body_meshes is empty; falling back to static")
            return

        self._wp_robot_mesh_ids = wp.array(
            [m.id for m in self._robot_wp_meshes],
            dtype=wp.uint64,
            device=self.device,
        )

        self._dyna_body_idx_tensor = torch.tensor(
            self._dyna_body_indices,
            dtype=torch.long,
            device=self.device,
        )

        N = self.num_envs
        B = self._num_dyna_bodies
        self._dyna_body_pos = torch.zeros(N, B, 3, device=self.device, dtype=torch.float32)
        self._dyna_body_quat = torch.zeros(N, B, 4, device=self.device, dtype=torch.float32)
        self._dyna_body_quat[..., 3] = 1.0

        self._wp_dyna_body_pos = wp.from_torch(self._dyna_body_pos, dtype=wp.vec3)
        self._wp_dyna_body_quat = wp.from_torch(self._dyna_body_quat, dtype=wp.quat)

    def step(self) -> None:
        """Update global camera poses from the current rigid-body state."""
        sim = self.env.simulator

        for s, body_idx in enumerate(self._body_indices):
            body_pos = sim._rigid_body_pos[:, body_idx, :]  # (N, 3)
            body_quat = sim._rigid_body_rot[:, body_idx, :]  # (N, 4)

            cam_pos = quat_apply(body_quat, self._local_pos_list[s], w_last=True) + body_pos
            combined_local = quat_mul(self._local_quat_list[s], self._frame_quat, w_last=True)
            cam_quat = quat_mul(body_quat, combined_local, w_last=True)

            self._global_pos[:, s, :] = cam_pos
            self._global_quat[:, s, :] = cam_quat

        if self._use_dynamic_meshes:
            if hasattr(sim, "_rigid_body_pos_global"):
                self._dyna_body_pos[:] = sim._rigid_body_pos_global[:, self._dyna_body_idx_tensor, :]
                self._dyna_body_quat[:] = sim._rigid_body_rot_global[:, self._dyna_body_idx_tensor, :]
            else:
                self._dyna_body_pos[:] = sim._rigid_body_pos[:, self._dyna_body_idx_tensor, :]
                self._dyna_body_quat[:] = sim._rigid_body_rot[:, self._dyna_body_idx_tensor, :]

    def capture(self) -> torch.Tensor:
        """Ray-cast depth images for all cameras and return the buffer.

        Returns
        -------
        torch.Tensor
            Depth buffer of shape ``(num_envs, num_cameras, ray_height, ray_width)``.
        """
        if self._graph is None:
            self._build_render_graph()

        if self._graph is not None:
            wp.capture_launch(self._graph)
        else:
            self._launch_kernel()

        if self.noise_cfg.get("enable", False):
            self._apply_noise()

        if self.occlusion_cfg.get("enable", False):
            self._apply_random_depth_occlusion()

        if self._postprocess:
            self._update_processed()

        return self.depth_buffer

    @property
    def output_depth(self) -> torch.Tensor:
        """Depth frame observations should read, shape ``(N, S, out_h, out_w)``.

        The per-env delayed frame from the ring buffer when the delay buffer is
        enabled, else the clipped/resized frame when resizing is enabled, else
        the raw ``depth_buffer``.
        """
        if self._use_delay:
            return self.delay_buffer[self._env_arange, self.delay_idx]
        if self._postprocess:
            return self.processed_buffer
        return self.depth_buffer

    def reset(self, env_ids: torch.Tensor) -> None:
        """Clear the delay buffer and resample the delay index for ``env_ids``."""
        if not self._use_delay or env_ids.numel() == 0:
            return
        self.delay_buffer[env_ids] = self._delay_fill_value
        self.delay_idx[env_ids] = torch.randint(
            self._delay_min, self._delay_max + 1, (env_ids.numel(),), device=self.device
        )

    def _update_processed(self) -> None:
        """Clip -> resize the captured frame and push it into the delay buffer."""
        frame = self.depth_buffer.clamp(self.props.near_clip, self.props.far_clip)
        if self._use_resize:
            frame = resize_depth(
                frame, (self.out_height, self.out_width), mode=self._resize_mode, antialias=self._resize_antialias
            )
        self.processed_buffer[:] = frame
        if self._use_delay:
            # Shift older frames towards higher indices; index 0 is the latest.
            self.delay_buffer[:, 1:] = self.delay_buffer[:, :-1].clone()
            self.delay_buffer[:, 0] = frame

    def get_depth_buffer_by_name(self, camera_name: str) -> torch.Tensor:
        """Return the depth buffer slice for a named camera.

        Parameters
        ----------
        camera_name : str
            Name of the camera as defined in the ``poses`` dict.

        Returns
        -------
        torch.Tensor
            Depth buffer of shape ``(num_envs, 1, ray_height, ray_width)``.
        """
        idx = self._camera_name_to_idx[camera_name]
        return self.depth_buffer[:, idx : idx + 1, :, :]

    def draw_debug_viz(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _launch_kernel(self) -> None:
        dims = (self.num_envs, self._num_sensors, self.ray_width, self.ray_height)
        fov_per_env = self._randomize_fov and self._wp_K_inv_array is not None
        dyna = self._use_dynamic_meshes

        if dyna and fov_per_env:
            wp.launch(
                kernel=depth_range_dynamic_per_env_fov_kernel,
                dim=dims,
                inputs=[
                    self._terrain_mesh_id,
                    self._wp_robot_mesh_ids,
                    self._wp_dyna_body_pos,
                    self._wp_dyna_body_quat,
                    self._wp_pos,
                    self._wp_quat,
                    self._wp_K_inv_array,
                    self.far_plane,
                    self._wp_pixels,
                    self._c_x,
                    self._c_y,
                    self.calculate_depth,
                    self._num_dyna_bodies,
                ],
                device=self.device,
            )
        elif dyna:
            wp.launch(
                kernel=depth_range_dynamic_kernel,
                dim=dims,
                inputs=[
                    self._terrain_mesh_id,
                    self._wp_robot_mesh_ids,
                    self._wp_dyna_body_pos,
                    self._wp_dyna_body_quat,
                    self._wp_pos,
                    self._wp_quat,
                    self._K_inv,
                    self.far_plane,
                    self._wp_pixels,
                    self._c_x,
                    self._c_y,
                    self.calculate_depth,
                    self._num_dyna_bodies,
                ],
                device=self.device,
            )
        elif fov_per_env:
            wp.launch(
                kernel=depth_range_per_env_fov_kernel,
                dim=dims,
                inputs=[
                    self._terrain_mesh_id,
                    self._wp_pos,
                    self._wp_quat,
                    self._wp_K_inv_array,
                    self.far_plane,
                    self._wp_pixels,
                    self._c_x,
                    self._c_y,
                    self.calculate_depth,
                ],
                device=self.device,
            )
        else:
            wp.launch(
                kernel=depth_range_kernel,
                dim=dims,
                inputs=[
                    self._terrain_mesh_id,
                    self._wp_pos,
                    self._wp_quat,
                    self._K_inv,
                    self.far_plane,
                    self._wp_pixels,
                    self._c_x,
                    self._c_y,
                    self.calculate_depth,
                ],
                device=self.device,
            )

    def _build_render_graph(self) -> None:
        try:
            wp.capture_begin(device=self.device)
            self._launch_kernel()
            self._graph = wp.capture_end(device=self.device)
            logger.info("WarpDepthCamera: CUDA graph captured successfully")
        except Exception as e:
            logger.warning(f"WarpDepthCamera: CUDA graph capture failed ({e}), falling back to eager launch")
            self._graph = None

    def _apply_noise(self) -> None:
        std_mul = self.noise_cfg.get("std_multiplier", 0.01)
        dropout_prob = self.noise_cfg.get("dropout_prob", 0.01)
        dropout_val = self.noise_cfg.get("dropout_value", 0.0)

        self.depth_buffer[:] = torch.normal(
            mean=self.depth_buffer,
            std=std_mul * self.depth_buffer.abs(),
        )
        drop_mask = torch.bernoulli(torch.full_like(self.depth_buffer, dropout_prob)).bool()
        self.depth_buffer[drop_mask] = dropout_val

    def _setup_occlusion(self) -> None:
        box = list(self.occlusion_cfg.get("box", [0, 0, 0, 0]))
        diff = int(self.occlusion_cfg.get("diff", 20))
        x_scale = self.ray_width / float(self.props.width)
        y_scale = self.ray_height / float(self.props.height)
        self._occl_x1 = max(0, min(self.ray_width, round(box[0] * x_scale)))
        self._occl_x2 = max(0, min(self.ray_width, round(box[1] * x_scale)))
        self._occl_y1 = max(0, min(self.ray_height, round(box[2] * y_scale)))
        self._occl_y2 = max(0, min(self.ray_height, round(box[3] * y_scale)))
        self._occl_diff_x = round(diff * x_scale)

        mask_list = list(self.occlusion_cfg.get("mask", []))
        type_names = list(getattr(self._terrain_state, "terrain_type_names", []))
        num_terrain_types = max(1, len(type_names))
        all_idx = set(range(num_terrain_types))
        occluded_idx = sorted(i for i in mask_list if 0 <= i < num_terrain_types)
        unoccluded_idx = sorted(all_idx - set(occluded_idx))
        self._occl_occluded_idx = torch.tensor(
            occluded_idx,
            dtype=torch.long,
            device=self.device,
        )
        self._occl_unoccluded_idx = torch.tensor(
            unoccluded_idx,
            dtype=torch.long,
            device=self.device,
        )

    def _apply_random_depth_occlusion(self) -> None:
        depth = self.depth_buffer
        N, S, H, W = depth.shape
        device = depth.device

        terrain_idx = getattr(self._terrain_state, "terrain_type_idx", None)
        if terrain_idx is None:
            terrain_idx = torch.zeros(N, dtype=torch.long, device=device)
        else:
            terrain_idx = terrain_idx.to(device=device, dtype=torch.long)

        prob = float(self.occlusion_cfg.get("prob", 0.0))
        bias = float(self.occlusion_cfg.get("terrain_bias", 0.0))
        p_none_base = float(self.occlusion_cfg.get("p_none", 0.1))
        p_small_base = float(self.occlusion_cfg.get("p_small", 0.45))
        p_large_base = float(self.occlusion_cfg.get("p_large", 0.45))

        p_none = torch.full((N,), p_none_base, device=device)
        p_small = torch.full((N,), p_small_base, device=device)
        p_large = torch.full((N,), p_large_base, device=device)

        if self._occl_occluded_idx.numel() > 0:
            occluded_mask = torch.isin(terrain_idx, self._occl_occluded_idx)
            p_small[occluded_mask] -= bias
            p_large[occluded_mask] += bias
        if self._occl_unoccluded_idx.numel() > 0:
            unoccluded_mask = torch.isin(terrain_idx, self._occl_unoccluded_idx)
            p_small[unoccluded_mask] += bias
            p_large[unoccluded_mask] -= bias

        rand_vals = torch.rand(N, device=device)
        mode_small = (rand_vals >= p_none) & (rand_vals < p_none + p_small)
        mode_large = rand_vals >= (p_none + p_small)
        apply_occlusion = torch.rand(N, device=device) < prob

        x1, x2, y1, y2 = self._occl_x1, self._occl_x2, self._occl_y1, self._occl_y2
        # FALCON draws one (N, 1, H, W) image shared by all cameras of an env.
        img_s = 1 if self.occlusion_cfg.get("share_across_cameras", False) else S
        random_depth = self.props.near_clip + (self.props.far_clip - self.props.near_clip) * torch.rand(
            N, img_s, H, W, device=device
        )

        large_flags = mode_large & apply_occlusion
        if large_flags.any():
            if y1 > 0:
                depth[large_flags, :, :y1, :] = random_depth[large_flags, :, :y1, :]
            if y2 > 0:
                depth[large_flags, :, H - y2 : H, :] = random_depth[large_flags, :, H - y2 : H, :]
            if x1 > 0:
                depth[large_flags, :, :, :x1] = random_depth[large_flags, :, :, :x1]
            if x2 > 0:
                depth[large_flags, :, :, W - x2 : W] = random_depth[large_flags, :, :, W - x2 : W]

        small_flags = mode_small & apply_occlusion
        if small_flags.any():
            x1_small = max(0, x1 - self._occl_diff_x)
            x2_small = max(0, x2 - self._occl_diff_x)
            if x1_small > 0:
                depth[small_flags, :, :, :x1_small] = random_depth[small_flags, :, :, :x1_small]
            if x2_small > 0:
                depth[small_flags, :, :, W - x2_small : W] = random_depth[small_flags, :, :, W - x2_small : W]

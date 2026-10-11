"""Terrain hooks for locomotion tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Tuple

from holosoma.managers.terrain.base import TerrainTermBase
from holosoma.simulator.shared.terrain import Terrain
from holosoma.utils import draw, warp_utils
from holosoma.utils.rotations import quat_apply_yaw
from holosoma.utils.safe_torch_import import torch

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

ATTACHMENT_POS = (0.0, 0.0, 0.25)


class TerrainLocomotion(TerrainTermBase):
    """Stateful terrain term that owns terrain buffers and updates them each step."""

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)
        self._terrain = Terrain(self._cfg, self.num_envs)
        assert hasattr(self._terrain, "mesh")
        self._warp_mesh = warp_utils.convert_to_wp_mesh(
            self._terrain.mesh.vertices, self._terrain.mesh.faces, self.device
        )
        self._env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        self._terrain_origins = torch.as_tensor(self.terrain._env_origins, device=self.device, dtype=torch.float32)
        self._terrain_levels = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._terrain_types = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._max_terrain_levels = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._max_terrain_level = int(self._terrain_origins.shape[0])
        # Per-(row, col) terrain-type indices (int64 on device). For terrains that don't
        # carry a type grid (plane, load_obj), fall back to an all-zeros grid.
        grid = getattr(self._terrain, "terrain_type_grid", None)
        if grid is not None and grid.size > 0:
            self._terrain_type_grid = torch.as_tensor(grid, device=self.device, dtype=torch.long)
        else:
            self._terrain_type_grid = torch.zeros(
                (
                    max(1, self._terrain_origins.shape[0]),
                    max(1, self._terrain_origins.shape[1] if self._terrain_origins.ndim > 1 else 1),
                ),
                device=self.device,
                dtype=torch.long,
            )
        self._terrain_type_names: list[str] = list(getattr(self._terrain, "terrain_type_names", []))
        # Per-env terrain type index (into _terrain_type_names). Refreshed alongside env_origins.
        self._terrain_type_idx = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._get_env_origins()

        if self._terrain.x_edge_mask is not None:
            self._x_edge_mask = torch.tensor(self._terrain.x_edge_mask, dtype=torch.bool, device=self.device)
            self._y_edge_mask = torch.tensor(self._terrain.y_edge_mask, dtype=torch.bool, device=self.device)
        else:
            self._x_edge_mask = None
            self._y_edge_mask = None

        # Optional per-env heightfield-cell offset for height-scan sampling.
        # Populated by the ``HeightMeasurementXYOffsetRandomizer`` randomization
        # term; see :meth:`set_measured_heights_xy_offset_points`.
        self._measured_heights_xy_offset_points: torch.Tensor | None = None

    @property
    def terrain(self):
        return self._terrain

    @property
    def mesh(self):
        return self._terrain.mesh

    @property
    def warp_mesh(self):
        return self._warp_mesh

    @property
    def x_edge_mask(self):
        return self._x_edge_mask

    @property
    def y_edge_mask(self):
        return self._y_edge_mask

    @property
    def has_edge_masks(self) -> bool:
        return self._x_edge_mask is not None

    def setup(self) -> None:
        self._base_heights = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        self._offset_pos = torch.tensor(list(ATTACHMENT_POS), device=self.device)

        # Single-ray buffers for base height
        self._ray_directions_base = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._ray_directions_base[..., 2] = -1.0
        self._ray_hits_world_base = torch.zeros(self.num_envs, 1, 3, device=self.device, requires_grad=False)

        # Scan grid for height measurements
        self._enable_height_scan = getattr(self._cfg, "enable_height_scan", False)
        if self._enable_height_scan:
            self._height_scan_points, self._num_height_scan_points = self._init_height_scan_points()
        else:
            self._num_height_scan_points = 0

        # Per-step cached height-scan values. Populated lazily by
        # :meth:`height_measurements` on first call each step; invalidated at
        # the start of ``update_heights``. Subsequent callers in the same step
        # (observation + reward) reuse the cached tensor rather than issuing
        # duplicate ray-casts.
        self._height_measurements_cache: torch.Tensor | None = None

        # ``height_scan_mode="falcon"``: sample the raw heightfield like
        # FAR-FALCON's ``_get_heights`` instead of ray-casting the mesh.
        self._height_scan_mode = getattr(self._cfg, "height_scan_mode", "raycast")
        self._body_height_mode = getattr(self._cfg, "body_height_mode", "raycast")
        self._height_samples: torch.Tensor | None = None
        self._heightfield: torch.Tensor | None = None
        if (self._enable_height_scan and self._height_scan_mode == "falcon") or self._body_height_mode == "falcon":
            hf = self._terrain.height_field_raw
            if hf is None:
                raise ValueError("FALCON heightfield lookups require a heightfield terrain (not load_obj).")
            self._heightfield = torch.as_tensor(hf, device=self.device)
        if self._enable_height_scan and self._height_scan_mode == "falcon":
            self._height_samples = self._heightfield
        if self._body_height_mode == "falcon":
            # FAR-FALCON ``_init_base_height_points``: meshgrid(x, y) with x major.
            bx = torch.tensor([-0.05, 0.0, 0.05], device=self.device)
            by = torch.tensor([-0.1, -0.05, 0.0, 0.05, 0.1], device=self.device)
            gx, gy = torch.meshgrid(bx, by, indexing="ij")
            self._base_height_points = torch.zeros(self.num_envs, gx.numel(), 3, device=self.device)
            self._base_height_points[:, :, 0] = gx.flatten()
            self._base_height_points[:, :, 1] = gy.flatten()

        # Metric scale for converting the per-env integer cell offset stored
        # in :attr:`_measured_heights_xy_offset_points` into XY metres, to
        # emulate FAR-FALCON's ``randomize_height_measurements_xy_offset_points``.
        # Sampled under five XY positions (center, +/- x*scale, +/- y*scale),
        # the min is returned so the policy must tolerate lateral misalignment.
        # Use ``height_scan_resolution`` (mirrors FALCON's ``measured_resolution``)
        # rather than the terrain's ``horizontal_scale``: the latter varies with
        # terrain type (1.0 m on plane, 0.05-0.1 m on heightfields), which would
        # make a 1-cell offset mean different things in different runs.
        self._xy_offset_scale: float = float(
            getattr(self._cfg, "height_scan_resolution", self._terrain.horizontal_scale)
        )

        if hasattr(self.env, "feet_height_indices"):
            # Environment requires feet heights
            self._compute_feet_heights = True
            self._feet_heights = torch.zeros(
                self.num_envs, len(self.env.feet_height_indices), device=self.device, requires_grad=False
            )
            self._ray_directions_feet = torch.zeros(
                self.num_envs, len(self.env.feet_height_indices), 3, device=self.device, requires_grad=False
            )
            self._ray_directions_feet[..., :] = torch.tensor([0.0, 0.0, -1.0], device=self.device)
            self._ray_hits_world_feet = torch.zeros(
                self.num_envs, len(self.env.feet_height_indices), 3, device=self.device, requires_grad=False
            )
        else:
            # Initialize not-used feet heights tensors with empty tensors
            self._compute_feet_heights = False
            self._feet_heights = torch.zeros(0, device=self.device, requires_grad=False)
            self._ray_directions_feet = torch.zeros(0, 0, 3, device=self.device, requires_grad=False)
            self._ray_hits_world_feet = torch.zeros(0, 0, 3, device=self.device, requires_grad=False)

        # Precompute the "use raw world-Z instead of ray-cast height" mask
        # (FAR-FALCON's ``not_measure_heights_mask``). Applies to stepping stones
        # / gaps where the downward ray would land in a pit.
        self._raw_base_height_mask: torch.Tensor | None = None
        mask_names = getattr(self._cfg, "raw_base_height_terrain_mask", None)
        if mask_names and self._terrain_type_names:
            idx_list = [self._terrain_type_names.index(n) for n in mask_names if n in self._terrain_type_names]
            if idx_list:
                allowed = torch.tensor(idx_list, device=self.device, dtype=torch.long)
                self._raw_base_height_mask = torch.isin(self._terrain_type_idx, allowed)

    @property
    def env_origins(self) -> torch.Tensor:
        return self._env_origins

    @property
    def terrain_origins(self) -> torch.Tensor:
        return self._terrain_origins

    @property
    def terrain_levels(self) -> torch.Tensor:
        return self._terrain_levels

    @property
    def terrain_types(self) -> torch.Tensor:
        return self._terrain_types

    @property
    def terrain_type_idx(self) -> torch.Tensor:
        """Per-env terrain type index into :attr:`terrain_type_names`."""
        return self._terrain_type_idx

    @property
    def terrain_type_names(self) -> list[str]:
        """Ordered names matching :attr:`terrain_type_idx`."""
        return self._terrain_type_names

    @property
    def max_terrain_levels(self) -> torch.Tensor:
        return self._max_terrain_levels

    @property
    def max_terrain_level(self) -> int:
        return self._max_terrain_level

    @property
    def custom_origins(self) -> bool:
        return self._custom_origins

    @property
    def base_heights(self) -> torch.Tensor:
        return self._base_heights

    @property
    def feet_heights(self) -> torch.Tensor:
        return self._feet_heights

    def update_heights(self, env_ids=None):
        # Invalidate the per-step height-scan cache so the first caller this
        # step (observation or reward) recomputes it.
        self._height_measurements_cache = None
        idx = env_ids if env_ids is not None else slice(None)
        self._base_heights[idx], self._ray_hits_world_base[idx] = self._get_base_heights(env_ids)
        if self._compute_feet_heights:
            self._feet_heights[idx], self._ray_hits_world_feet[idx] = self._get_feet_heights(env_ids)

        # FAR-FALCON's ``not_measure_heights_mask`` hook: on stepping-stones /
        # gap-like terrains, overwrite the ray-cast heights with raw world-Z so
        # downstream code doesn't see pit-bottom heights. ``env_ids`` may be
        # ``None``, a ``slice`` (e.g. ``slice(None)`` for all envs), or a tensor
        # / sequence of ids.
        if self._raw_base_height_mask is not None:
            all_envs = env_ids is None or isinstance(env_ids, slice)
            if all_envs:
                mask_env_ids = self._raw_base_height_mask.nonzero(as_tuple=False).flatten()
            else:
                ids = (
                    env_ids
                    if isinstance(env_ids, torch.Tensor)
                    else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
                )
                mask_env_ids = ids[self._raw_base_height_mask[ids]]
            if mask_env_ids.numel() > 0:
                self._base_heights[mask_env_ids] = self.env.simulator.robot_root_states[mask_env_ids, 2].clone()
                if self._compute_feet_heights:
                    self._feet_heights[mask_env_ids] = self.env.simulator._rigid_body_pos[mask_env_ids][
                        :, self.env.feet_height_indices, 2
                    ].clone()

    def _get_env_origins(self):
        """Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
        Otherwise create a grid.
        """
        self._custom_origins = True

        if self._cfg.spawn.randomize_tiles:
            max_init_level = self._cfg.max_init_terrain_level
            if max_init_level is None:
                max_init_level = max(self._max_terrain_level - 1, 0)
            max_init_level = min(int(max_init_level), max(self._max_terrain_level - 1, 0))
            self._terrain_levels.random_(0, max_init_level + 1)
            num_cols = int(self._terrain_origins.shape[1])
            if num_cols > 0:
                denom = max(int(self.num_envs / num_cols), 1)
                self._terrain_types[:] = torch.div(
                    torch.arange(self.num_envs, device=self.device),
                    denom,
                    rounding_mode="floor",
                ).clamp_(max=num_cols - 1)
            self.refresh_env_origins()
        else:
            # Eval mode: all robots at tile (0,0) for deterministic evaluation.
            # A LOAD_OBJ terrain has no generated height field, so it never fills ``_env_origins``;
            # its per-tile origins are derived from the mesh bounds instead.
            if self._cfg.mesh_type == "load_obj":
                origin_grid: npt.NDArray[np.floating[Any]] = self.terrain._get_load_obj_env_origin_grid()
                origin_0_0 = torch.from_numpy(origin_grid[0, 0]).to(self.device).to(torch.float)
                self._env_origins[:] = origin_0_0  # Broadcast to all robots
            else:
                self._terrain_levels.zero_()
                self._terrain_types.zero_()
                self.refresh_env_origins()
        self._max_terrain_levels.copy_(self._terrain_levels)

    def refresh_env_origins(self, env_ids=None) -> None:
        idx = env_ids if env_ids is not None else slice(None)
        self._env_origins[idx] = self._terrain_origins[self._terrain_levels[idx], self._terrain_types[idx]]
        # Keep per-env terrain-type index in sync with (level, type-column) assignment.
        if self._terrain_type_grid.numel() > 0:
            self._terrain_type_idx[idx] = self._terrain_type_grid[self._terrain_levels[idx], self._terrain_types[idx]]
        self._sync_simulator_env_origins(env_ids)

    def set_terrain_levels(self, env_ids: torch.Tensor, levels: torch.Tensor) -> None:
        self._terrain_levels[env_ids] = levels.to(device=self.device, dtype=torch.long)
        self._max_terrain_levels[env_ids] = torch.maximum(
            self._max_terrain_levels[env_ids], self._terrain_levels[env_ids]
        )
        self.refresh_env_origins(env_ids)

    def _sync_simulator_env_origins(self, env_ids=None) -> None:
        simulator = getattr(self.env, "simulator", None)
        if simulator is None:
            return
        idx = env_ids if env_ids is not None else slice(None)
        sim_env_origins = getattr(simulator, "env_origins", None)
        if isinstance(sim_env_origins, torch.Tensor):
            sim_env_origins[idx] = self._env_origins[idx]
        scene = getattr(simulator, "scene", None)
        scene_env_origins = getattr(scene, "env_origins", None)
        if isinstance(scene_env_origins, torch.Tensor):
            scene_env_origins[idx] = self._env_origins[idx]

    def _init_height_scan_points(self):
        """Initialize the scan grid for height measurements (in base frame).

        If ``cfg.height_scan_points_x`` / ``cfg.height_scan_points_y`` are set, they
        are used verbatim.  Otherwise the grid is generated as
        ``linspace(-N/2, N/2, N) * resolution`` for each axis, mirroring FAR-FALCON's
        ``measured_points_x_num`` / ``measured_points_y_num`` / ``measured_resolution``.

        Returns:
            Tuple of (points ``[num_envs, num_points, 3]``, num_points)
        """
        x = self._build_scan_axis(self._cfg.height_scan_points_x, self._cfg.height_scan_points_x_num, "x")
        y = self._build_scan_axis(self._cfg.height_scan_points_y, self._cfg.height_scan_points_y_num, "y")
        self._num_height_scan_points_x = int(x.numel())
        self._num_height_scan_points_y = int(y.numel())
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
        num_points = grid_x.numel()
        points = torch.zeros(self.num_envs, num_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points, num_points

    def _build_scan_axis(self, explicit_points, num_points: int, axis: str) -> torch.Tensor:
        """Build a 1-D tensor of scan offsets for one axis, preferring the explicit list."""
        if explicit_points is not None:
            return torch.tensor(list(explicit_points), device=self.device, dtype=torch.float)
        if num_points <= 0:
            raise ValueError(
                f"enable_height_scan=True requires height_scan_points_{axis} or height_scan_points_{axis}_num > 0"
            )
        resolution = float(self._cfg.height_scan_resolution)
        return torch.linspace(-num_points / 2, num_points / 2, num_points, device=self.device) * resolution

    @property
    def num_height_scan_points_x(self) -> int:
        """Number of scan points along the x-axis (0 if ``enable_height_scan`` is False)."""
        return getattr(self, "_num_height_scan_points_x", 0)

    @property
    def num_height_scan_points_y(self) -> int:
        """Number of scan points along the y-axis (0 if ``enable_height_scan`` is False)."""
        return getattr(self, "_num_height_scan_points_y", 0)

    def _heightfield_cells(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """FAR-FALCON cell index of world ``points`` ``[..., 3]``: truncation, clipped to ``shape - 2``."""
        hf = self._heightfield
        assert hf is not None
        cells = ((points + self._terrain.border_size) / self._terrain.horizontal_scale).long()
        px = torch.clip(cells[..., 0].reshape(-1), 0, hf.shape[0] - 2)
        py = torch.clip(cells[..., 1].reshape(-1), 0, hf.shape[1] - 2)
        return px, py

    def _falcon_base_heights(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """FAR-FALCON ``IsaacGym._get_base_heights``; also returns the ground point for debug viz."""
        hf = self._heightfield
        root_pos = self.env.simulator.robot_root_states[idx, :3]
        base_quat = self.env.base_quat[idx]
        points = self._base_height_points[idx]
        n = points.shape[0]
        points = quat_apply_yaw(base_quat.repeat(1, points.shape[1]), points, True) + root_pos.unsqueeze(1)
        px, py = self._heightfield_cells(points)
        heights = torch.min(torch.min(hf[px, py], hf[px + 1, py]), hf[px, py + 1])
        ground = (heights.view(n, -1).to(torch.float32) * self._terrain.vertical_scale).mean(dim=1)
        hits = root_pos.clone()
        hits[:, 2] = ground
        return root_pos[:, 2] - ground, hits.unsqueeze(1)

    def _falcon_feet_heights(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """FAR-FALCON ``IsaacGym._get_feet_heights``; also returns the ground points for debug viz."""
        hf = self._heightfield
        foot_positions = self.env.simulator._rigid_body_pos[idx][:, self.env.feet_height_indices, :].clone()
        px, py = self._heightfield_cells(foot_positions)
        heights = torch.max(torch.max(hf[px, py], hf[px + 1, py]), hf[px, py + 1])
        ground = heights.view(foot_positions.shape[:2]).to(torch.float32) * self._terrain.vertical_scale
        hits = foot_positions.clone()
        hits[..., 2] = ground
        return foot_positions[..., 2] - ground, hits

    def _get_base_heights(self, env_ids=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get base height above terrain via single ray cast from root position."""
        idx = env_ids if env_ids is not None else slice(None)
        if self._body_height_mode == "falcon":
            return self._falcon_base_heights(idx)
        root_pos = self.env.simulator.robot_root_states[idx, :3].unsqueeze(1)
        ray_starts = root_pos + self._offset_pos
        ray_hits = warp_utils.ray_cast(ray_starts, self._ray_directions_base[idx], self.warp_mesh)
        base_heights = (root_pos - ray_hits)[..., 2].squeeze(1)
        return base_heights, ray_hits

    def _get_feet_heights(self, env_ids=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get feet heights from the terrain."""
        idx = env_ids if env_ids is not None else slice(None)
        if self._body_height_mode == "falcon":
            return self._falcon_feet_heights(idx)
        foot_positions = self.env.simulator._rigid_body_pos[idx][:, self.env.feet_height_indices, :].clone()
        ray_starts_world = foot_positions + self._offset_pos[None, None, ...]
        ray_directions_world = self._ray_directions_feet[idx]
        ray_hits_world = warp_utils.ray_cast(ray_starts_world, ray_directions_world, self.warp_mesh)
        return (foot_positions - ray_hits_world)[..., 2], ray_hits_world

    def query_terrain_heights(
        self,
        xy_positions: torch.Tensor,
        use_grid_sampling: bool = False,
        grid_size: int = 3,
        grid_spacing: float = 0.3,
    ) -> torch.Tensor:
        """Query terrain height at arbitrary XY positions using ray casting.

        This method casts rays straight down from high above the terrain to find
        the actual terrain height at each XY position. Useful for determining
        spawn heights, checking clearances, or any arbitrary height query.

        Args:
            xy_positions: XY coordinates to query. Shape: (N, 2)
            use_grid_sampling: If True, sample a grid around each position and return max height.
                              This ensures the robot clears all terrain within its footprint.
            grid_size: Number of points per dimension in the grid (e.g., 3 = 3x3 = 9 points)
            grid_spacing: Spacing between grid points in meters (e.g., 0.3m)

        Returns:
            Terrain heights (Z coordinates). Shape: (N,)

        Example:
            >>> xy = torch.tensor([[0.5, 0.5], [1.0, 1.0]], device="cuda")
            >>> heights = terrain.query_terrain_heights(xy)
            >>> heights.shape  # (2,)

            >>> # With grid sampling for safer spawn heights
            >>> heights = terrain.query_terrain_heights(xy, use_grid_sampling=True)
            >>> heights.shape  # (2,) - max height from 3x3 grid around each position
        """
        num_robots = xy_positions.shape[0]

        if use_grid_sampling:
            # Generate grid offsets centered at origin
            # For grid_size=3, spacing=0.3: offsets = [-0.3, 0.0, 0.3]
            half_extent = (grid_size - 1) * grid_spacing / 2.0
            offsets_1d = torch.linspace(-half_extent, half_extent, grid_size, device=self.device)
            grid_x, grid_y = torch.meshgrid(offsets_1d, offsets_1d, indexing="ij")
            grid_offsets = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
            # Shape: (grid_size^2, 2) e.g., (9, 2) for 3x3

            # Expand positions to include all grid points
            # xy_positions: (N, 2) -> (N, 1, 2)
            # grid_offsets: (9, 2) -> (1, 9, 2)
            # Result: (N, 9, 2) via broadcasting
            xy_expanded = xy_positions.unsqueeze(1) + grid_offsets.unsqueeze(0)

            # Flatten for batched ray casting
            xy_flat = xy_expanded.reshape(-1, 2)  # (N*grid_size^2, 2)
            num_points = len(xy_flat)
        else:
            xy_flat = xy_positions
            num_points = num_robots

        # Create ray starts high above terrain (100m should be above any realistic terrain)
        ray_starts = torch.zeros(num_points, 3, device=self.device)
        ray_starts[:, :2] = xy_flat
        ray_starts[:, 2] = 100.0  # Start 100m above ground for safety

        # Ray directions pointing straight down
        ray_directions = torch.zeros(num_points, 3, device=self.device)
        ray_directions[:, 2] = -1.0

        # Cast rays to find terrain height
        ray_hits = warp_utils.ray_cast(ray_starts, ray_directions, self.warp_mesh)

        if use_grid_sampling:
            # Reshape and take max per robot.
            terrain_heights_raw = ray_hits[:, 2]
            grid_points_per_robot = grid_size * grid_size
            terrain_heights_grid = terrain_heights_raw.reshape(num_robots, grid_points_per_robot)
            # Takes max height across grid to improve likelihood robot clears all terrain
            return terrain_heights_grid.max(dim=1)[0]  # Shape: (N,)
        # Extract Z coordinates (assumes above terrain heights)
        return ray_hits[:, 2]

    def set_measured_heights_xy_offset_points(self, offsets: torch.Tensor | None) -> None:
        """Register the per-env XY offset tensor used by ``height_measurements``.

        ``offsets`` is expected to be a 1D integer tensor of shape ``[num_envs]``
        giving the number of heightfield cells to shift the scan points by.
        Pass ``None`` to disable the offset perturbation.
        """
        if offsets is None:
            self._measured_heights_xy_offset_points = None
            return
        if offsets.device != self.device:
            offsets = offsets.to(self.device)
        self._measured_heights_xy_offset_points = offsets
        # Invalidate cache so the next call recomputes with the new offsets.
        self._height_measurements_cache = None

    def height_measurements(self) -> torch.Tensor:
        """Terrain height at scan grid points around the base. Shape ``[N, num_scan_points]``.

        Returns zeros if ``enable_height_scan`` is False in the terrain config.

        Cached per step: the first call after each :meth:`update_heights`
        computes the ray-cast and stores the result; subsequent calls in the
        same step (e.g. observation group + reward term) return the cached
        tensor. The cache is invalidated at the top of ``update_heights``.

        When ``_measured_heights_xy_offset_points`` is set (via the
        ``randomize_height_measurements_xy_offset_points`` randomization term),
        the scan grid is sampled at five XY positions per point (center, +x,
        -x, +y, -y) and the minimum terrain height is returned. Mirrors
        FAR-FALCON's ``randomize_height_measurements_xy_offset_points`` hook.
        """
        if not self._enable_height_scan:
            return torch.zeros(self.env.num_envs, 0, device=self.device)
        if self._height_measurements_cache is not None:
            return self._height_measurements_cache
        env = self.env
        p = self._num_height_scan_points
        scan_positions = quat_apply_yaw(
            env.base_quat.repeat(1, p),
            self._height_scan_points,
            True,
        ) + env.simulator.robot_root_states[:, :3].unsqueeze(1)
        xy = scan_positions[..., :2].reshape(-1, 2)

        offsets = self._measured_heights_xy_offset_points
        if self._height_samples is not None:
            terrain_z = self._falcon_heightfield_heights(scan_positions, offsets)
            self._height_measurements_cache = terrain_z
            return self._height_measurements_cache

        if offsets is not None:
            # Per-env offset in metres, shape [N, 1] for broadcasting over scan points.
            delta = (offsets.to(torch.float32) * self._xy_offset_scale).unsqueeze(1)
            zeros = torch.zeros_like(delta)
            # [N, 1, 2] for broadcasting with xy_grid of shape [N, p, 2].
            dx_pos = torch.stack([delta, zeros], dim=-1)
            dy_pos = torch.stack([zeros, delta], dim=-1)
            xy_grid = xy.view(env.num_envs, p, 2)
            variants = torch.stack(
                [
                    xy_grid,
                    xy_grid + dx_pos,
                    xy_grid - dx_pos,
                    xy_grid + dy_pos,
                    xy_grid - dy_pos,
                ],
                dim=0,
            )  # [5, N, p, 2]
            xy_flat = variants.reshape(-1, 2)
            terrain_z = self.query_terrain_heights(xy_flat).view(5, env.num_envs, p)
            terrain_z = terrain_z.min(dim=0).values
            self._height_measurements_cache = terrain_z
            return self._height_measurements_cache

        terrain_z = self.query_terrain_heights(xy)
        self._height_measurements_cache = terrain_z.view(env.num_envs, p)
        return self._height_measurements_cache

    def _falcon_heightfield_heights(self, scan_positions: torch.Tensor, offsets: torch.Tensor | None) -> torch.Tensor:
        """FAR-FALCON ``IsaacGym._get_heights`` on the raw heightfield. Shape ``[N, p]``.

        Cell index is ``((p + border_size) / horizontal_scale).long()`` (truncation,
        no interpolation), clipped to ``[0, shape - 2]``. With per-env cell
        ``offsets`` the min over the center and the four +/-offset neighbours is
        taken (FALCON indexes the neighbours unclipped; they are clipped here to
        stay in bounds, which only matters at the outer border).
        """
        hs = self._height_samples
        assert hs is not None
        num_envs, p = scan_positions.shape[:2]
        points = (scan_positions + self._terrain.border_size) / self._terrain.horizontal_scale
        points = points.long()
        px = torch.clip(points[:, :, 0].reshape(-1), 0, hs.shape[0] - 2)
        py = torch.clip(points[:, :, 1].reshape(-1), 0, hs.shape[1] - 2)
        if offsets is None:
            heights = hs[px, py]
        else:
            n = offsets.to(torch.long).view(num_envs, 1)
            pxv, pyv = px.view(num_envs, p), py.view(num_envs, p)
            px_1 = torch.clip(pxv + n, 0, hs.shape[0] - 1).reshape(-1)
            px_2 = torch.clip(pxv - n, 0, hs.shape[0] - 1).reshape(-1)
            py_1 = torch.clip(pyv + n, 0, hs.shape[1] - 1).reshape(-1)
            py_2 = torch.clip(pyv - n, 0, hs.shape[1] - 1).reshape(-1)
            heights = torch.stack([hs[px, py], hs[px, py_1], hs[px, py_2], hs[px_1, py], hs[px_2, py]]).min(dim=0)[0]
        return heights.view(num_envs, p).to(torch.float32) * self._terrain.vertical_scale

    def draw_debug_viz(self):
        if not hasattr(self, "_ray_hits_world_base"):
            return  # setup() not run (e.g. run_sim, which uses the terrain only for the scene)
        env_id = 0

        # Base height hit (single red sphere)
        position = self._ray_hits_world_base[env_id, 0].detach().cpu().numpy()
        draw.draw_sphere(self.env.simulator, position, 0.03, (1, 0, 0), env_id)

        # Feet height hits (green spheres)
        if self._compute_feet_heights:
            for j in range(len(self.env.feet_height_indices)):
                position = self._ray_hits_world_feet[env_id, j].detach().cpu().numpy()
                draw.draw_sphere(self.env.simulator, position, 0.02, (0, 1, 0), env_id)

        # Height scan grid (yellow spheres at terrain surface)
        if self._enable_height_scan:
            p = self._num_height_scan_points
            scan_world = quat_apply_yaw(
                self.env.base_quat[env_id : env_id + 1].repeat(1, p),
                self._height_scan_points[env_id : env_id + 1],
                True,
            ) + self.env.simulator.robot_root_states[env_id : env_id + 1, :3].unsqueeze(1)
            xy = scan_world[0, :, :2]
            terrain_z = self.query_terrain_heights(xy)
            for j in range(p):
                pos = [scan_world[0, j, 0].item(), scan_world[0, j, 1].item(), terrain_z[j].item()]
                draw.draw_sphere(self.env.simulator, pos, 0.02, (1, 1, 0), env_id)

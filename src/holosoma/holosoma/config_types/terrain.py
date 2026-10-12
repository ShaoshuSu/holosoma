"""Configuration types for the terrain manager."""

from __future__ import annotations

from dataclasses import field
from enum import Enum
from typing import Literal

from pydantic.dataclasses import dataclass


class MeshType(str, Enum):
    """Terrain mesh type enumeration.

    Defines the different types of terrain representations available in the simulator.
    Each type has different computational costs and fidelity levels.
    """

    NONE = "none"
    """No terrain mesh.

    Used when terrain height measurement is not needed. Raises errors if height measurement is attempted.
    """

    PLANE = "plane"
    """Simple flat plane terrain.

    Minimal computational cost with uniform height. Good for basic locomotion tasks without terrain variation.
    """

    TRIMESH = "trimesh"
    """Triangle mesh terrain representation.

    Highest fidelity and computational cost. Supports complex terrain geometry and detailed collision detection.
    """

    LOAD_OBJ = "load_obj"
    """Load an OBJ file as a terrain mesh (used by the on-path-obstacle pipeline)."""

    FAKE = "fake"
    """Mock terrain for testing purposes.

    Behaves like plane but with different internal handling for development/testing scenarios.
    """

    def __str__(self) -> str:
        """Return the string value for backward compatibility with existing configs."""
        return self.value

    def __hash__(self) -> int:
        """Return hash of the enum value for use in sets and as dict keys."""
        return hash(self.value)

    def __eq__(self, other) -> bool:
        """Enable comparison with strings for backward compatibility."""
        if isinstance(other, str):
            return self.value == other
        return super().__eq__(other)


@dataclass(frozen=True)
class SpawnCfg:
    """Configuration for robot spawning behavior."""

    randomize_tiles: bool = True
    """Randomize terrain tile selection for curriculum learning.
    When True, robots spawn at random tiles across the terrain grid.
    When False, all robots spawn at tile (0,0) for deterministic evaluation."""

    xy_offset_range: float = 1.0
    """Random XY offset range within terrain tile (±meters from tile center).
    Adds positional variety within the selected tile.
    Set to 0.0 for deterministic spawning at exact tile center."""

    xy_offset_range_x_masked: list[float] | None = None
    """Per-terrain masked x-offset range ``[min, max]`` (meters). Applied to envs
    whose terrain type is *not* in :attr:`xy_offset_terrain_mask`. ``None`` falls
    back to the symmetric :attr:`xy_offset_range`.

    Mirrors FAR-FALCON's ``env_rand_x_masked``: used for stepping-stones / beams
    where the center-column layout survives x jitter."""

    xy_offset_range_y_masked: list[float] | None = None
    """Per-terrain masked y-offset range ``[min, max]`` (meters). Applied to envs
    whose terrain type is *not* in :attr:`xy_offset_terrain_mask`. ``None`` falls
    back to the symmetric :attr:`xy_offset_range`.

    Mirrors FAR-FALCON's ``env_rand_y_masked``: used for stepping-stones where
    y-jitter must stay tight or the robot spawns between rows."""

    xy_offset_terrain_mask: list[str] | None = None
    """Terrain type names that receive the full symmetric ``xy_offset_range`` at
    reset. Envs on terrain types not in this list use
    :attr:`xy_offset_range_x_masked` / :attr:`xy_offset_range_y_masked` instead.
    ``None`` disables masking (all envs use the symmetric range)."""

    query_terrain_height: bool = False
    """Query terrain height after applying XY offset to set correct spawn Z position.

    When False (default - faster):
      - Uses original simple spawning: applies XY offset but keeps original Z from env_origins
      - Fastest option, works well for plane/flat terrain
      - May cause robots to spawn slightly below/above terrain on rough terrain

    When True (More accurate for rough terrain):
      - Queries actual terrain height at new XY position via ray casting
      - Updates Z position to place robot correctly on terrain surface
      - Slower due to ray casting, but necessary for uneven terrain
      - Use with use_grid_sampling for maximum safety on rough terrain"""

    use_grid_sampling: bool = False
    """Use grid sampling when querying terrain height (requires query_terrain_height=True).

    Only applies if query_terrain_height=True:
      - False: Single ray cast at new XY position (1 ray per robot)
                Good for smooth terrain with gentle slopes
      - True:  Sample grid of points around position, take max height (9 rays per robot)
               Ensures robot clears all terrain within footprint
               Essential for rough terrain with obstacles/steep features

    If query_terrain_height=False, this parameter is ignored."""

    grid_size: int = 3
    """Grid sampling: number of sample points per dimension (e.g., 3 = 3x3 = 9 points).
    Only used if query_terrain_height=True and use_grid_sampling=True."""

    grid_spacing: float = 0.2
    """Grid sampling: spacing between sample points in meters.
    Should roughly match robot footprint size.
    Only used if query_terrain_height=True and use_grid_sampling=True."""


@dataclass(frozen=True)
class TerrainTermCfg:
    """Configuration for the terrain manager.

    This class defines all parameters needed to configure the terrain manager.
    """

    func: str
    """Import path for the terrain hook (function or callable class)."""

    static_friction: float
    """Static friction coefficient between robot and terrain."""

    dynamic_friction: float
    """Dynamic friction coefficient between robot and terrain."""

    restitution: float
    """Bounce/elasticity coefficient (0=no bounce, 1=perfect bounce)."""

    mesh_type: MeshType = MeshType.PLANE
    """Type of terrain mesh representation.

    See MeshType enum for options:
    - "trimesh": Triangle mesh (default, highest fidelity)
    - "plane": Simple flat plane
    - "none": No terrain mesh
    - "fake": Mock terrain for testing
    """

    hide_visual: bool = False
    """Render the terrain invisible while keeping its collider (default ``False``). Set ``True`` when
    a scene supplies its own visible floor; the robot still collides with the terrain."""

    horizontal_scale: float = 0.1
    """Horizontal resolution of terrain grid in meters."""

    vertical_scale: float = 0.005
    """Vertical scaling factor for height variations."""

    border_size: int = 40
    """Size of border around terrain in meters."""

    terrain_move_down_ratio: float = 0.0
    """Ratio for moving to easier terrain levels."""

    terrain_move_up_ratio: float = 0.0
    """Ratio for moving to harder terrain levels."""

    terrain_length: float = 8.0
    """Length of each terrain tile in meters."""

    terrain_width: float = 8.0
    """Width of each terrain tile in meters."""

    num_rows: int = 10
    """Number of terrain rows (difficulty levels)."""

    num_cols: int = 20
    """Number of terrain columns (terrain types)."""

    max_init_terrain_level: int | None = None
    """Maximum initial terrain level sampled at reset.

    If ``None``, the full terrain range ``[0, num_rows - 1]`` is used.
    This mirrors RPL's curriculum start-level control.
    """

    curriculum: bool = True
    """Whether to generate a curriculum grid (one terrain type per column,
    difficulty = ``row / num_rows``).

    When True (matches FAR-FALCON ``terrain.curriculum: True``): each column is
    a single terrain type and row index is difficulty; level 0 is nearly flat.
    When False: each cell samples a type at random from ``terrain_config`` and
    a difficulty from ``{0.5, 0.75, 0.9}`` (legacy Holosoma behaviour).
    """

    spawn: SpawnCfg = field(default_factory=SpawnCfg)
    """Spawn behavior configuration."""

    # Dictionary keys must match terrain generation function names (without '_terrain_func' suffix)
    # See terrain.py for available terrain types and their corresponding functions
    terrain_config: dict[str, float] = field(default_factory=dict)
    """Dictionary mapping terrain types to their proportions.

    This implements a dynamic function dispatch pattern where each key
    corresponds to a terrain generation function with the naming pattern
    '{terrain_type}_terrain_func()'. For example:
    - "flat" → flat_terrain_func()
    - "rough" → rough_terrain_func()
    - "low_obst" → low_obstacles_terrain_func()
    Values represent the proportion of terrain tiles that should use
    that terrain type (must sum to ≤ 1.0).
    """

    max_slope: float = 0.3
    """Maximum slope angle allowed before correction to vertical."""

    platform_size: float = 2.0
    """Size of flat platforms in complex terrain."""

    step_width_range: list[float] = field(default_factory=lambda: [0.30, 0.40])
    """Range of step widths for stair-like terrain."""

    fixed_step_height: float = 0.0
    """Fixed step height for stair-like terrain.

    if fixed_step_height == 0.0, use the difficulty-dependent step height.
    if fixed_step_height > 0.0, use the fixed step height.
    """

    stairs_offset: float = 0.05
    """Constant offset added to stair step height, in meters.

    Mirrors FAR-FALCON's ``stairs_offset``. Combined with ``stairs_ratio`` via
    ``step_height = stairs_offset + stairs_ratio * difficulty`` for the
    directional stair terrains."""

    stairs_ratio: float = 0.18
    """Difficulty multiplier for stair step height, in meters.

    Mirrors FAR-FALCON's ``stairs_ratio``. See :attr:`stairs_offset`."""

    random_step_height_range: list[float] = field(default_factory=lambda: [0.0, 0.0])
    """Per-step random height delta ``[min, max]`` (meters) added to each step.

    Mirrors FAR-FALCON's ``random_step_height_range``. Defaults to ``[0, 0]``
    (no per-step randomization)."""

    stone_size_range: list[float] = field(default_factory=lambda: [0.2, 0.8])
    """Stepping-stone size range ``[min, max]`` (meters). Larger at easy difficulty."""

    stone_distance_row_range: list[float] = field(default_factory=lambda: [0.05, 0.4])
    """Gap between stepping-stone rows ``[min, max]`` (meters). Smaller at easy difficulty."""

    stone_distance_col_range: list[float] = field(default_factory=lambda: [0.05, 0.4])
    """Gap between stepping stones within a row ``[min, max]`` (meters)."""

    stone_height_range: list[float] = field(default_factory=lambda: [0.0, 0.0])
    """Stepping-stone height variation ``[min, max]`` (meters)."""

    stone_depth_range: list[float] = field(default_factory=lambda: [0.4, 0.6])
    """Depth of gaps between stepping stones ``[min, max]`` (meters)."""

    gap_size_max: float = 0.5
    """Maximum gap (moat) width for the ``gap`` terrain, in meters.

    The gap width scales linearly with difficulty as
    ``gap_size_max * difficulty`` (so difficulty 1.0 yields ``gap_size_max``)."""

    gap_depth_range: list[float] = field(default_factory=lambda: [0.4, 0.6])
    """Depth of the ``gap`` terrain's moats ``[min, max]`` (meters), sampled
    per tile."""

    raw_base_height_terrain_mask: list[str] | None = None
    """Terrain-type names whose envs should bypass the ray-cast pelvis/feet
    heights and use raw world-Z instead. Mirrors FAR-FALCON's
    ``not_measure_heights_mask`` (indices ``[2, 7]`` for ``stepping_stone`` and
    ``gap``): on these terrains the downward ray can land in a pit between
    stones and return a meaningless height, so we fall back to world-Z.

    Names absent from the active terrain are silently ignored. ``None`` disables
    the override (all envs use the ray-cast height)."""

    amplitude_range: list[float] = field(default_factory=lambda: [0.01, 0.05])
    """Range of height variations for rough terrain."""

    slopes_roughness: bool = False
    """Add FAR-FALCON ``rough_slope`` noise to the ``slopes`` terrain.

    When True, each ``slopes`` tile is built like FALCON's
    ``rough_slope_terrain_func``: a pyramid slope (direction drawn with
    ``randint(0, 2) * 2 - 1``) plus ``random_uniform_terrain`` noise with
    amplitude ``U(amplitude_range)``, step 0.005 m and downsampled scale 0.2 m.
    False (default) keeps the smooth 50/50 up/down pyramid.

    Also adds the same noise to the directional ``slope_up`` / ``slope_down``
    tiles, like FALCON's ``rough_terrain`` flag on its directional slopes."""

    spacing_size: float = 0.0
    """Flat gap [m] between neighbouring sub-terrain tiles (FAR-FALCON ``spacing_size``).

    Each of the ``num_rows - 1`` / ``num_cols - 1`` interior gaps is
    ``int(spacing_size / horizontal_scale)`` heightfield cells of zero height,
    and tile origins shift by ``spacing_size`` per row / column. 0 (default)
    packs the tiles edge to edge."""

    slope_treshold: float = 0.75
    """Slope threshold for trimesh correction to vertical surfaces."""

    obj_file_path: str = ""
    """Path to OBJ file for custom terrain mesh."""

    scale_factor: float = 1.0
    """Use for performance to scale border_size, terrain_length, terrain_width, num_ros and num_cols."""

    name: str = "floor"
    """Mujoco-only: object (geom) name to assign to terrain. Use to match MJCF files <contacts> e.g, 'floor'."""

    enable_height_scan: bool = False
    """Enable the height scan grid for terrain height observations.

    When False (default), ``height_measurements()`` is a no-op that returns zeros.
    Set to True when the observation group includes a height-measurement term."""

    height_scan_points_x_num: int = 16
    """Number of scan points along the robot's x axis.

    Combined with :attr:`height_scan_resolution` this produces x offsets
    ``linspace(-N/2, N/2, N) * resolution`` (in metres, base frame), matching
    FAR-FALCON's ``measured_points_x_num``.  Default mirrors FAR-FALCON's
    ``measured_points_x_num=16``."""

    height_scan_points_y_num: int = 10
    """Number of scan points along the robot's y axis.

    Default mirrors FAR-FALCON's ``measured_points_y_num=10``."""

    height_scan_resolution: float = 0.1
    """Spacing (in metres) between adjacent scan points along both axes.

    Mirrors FAR-FALCON's ``measured_resolution``."""

    height_scan_points_x: list[float] | None = None
    """Optional explicit list of x offsets (metres, base frame).

    When set, overrides the ``height_scan_points_x_num`` / ``height_scan_resolution``
    linspace and is used verbatim.  Leave as ``None`` for the default grid."""

    height_scan_points_y: list[float] | None = None
    """Optional explicit list of y offsets (metres, base frame).

    When set, overrides the ``height_scan_points_y_num`` / ``height_scan_resolution``
    linspace and is used verbatim.  Leave as ``None`` for the default grid."""

    height_scan_mode: Literal["raycast", "falcon"] = "raycast"
    """How ``height_measurements()`` samples terrain height at the scan points.

    - ``"raycast"`` (default): downward ray-cast against the terrain mesh
      (exact surface height, including the trimesh's vertical-wall correction).
    - ``"falcon"``: FAR-FALCON's ``_get_heights`` lookup on the raw heightfield:
      ``idx = ((p + border_size) / horizontal_scale).long()`` (truncation, no
      interpolation), clipped to ``[0, shape - 2]``, height
      ``height_field_raw[idx] * vertical_scale``. The XY-offset randomizer then
      shifts the index by whole heightfield cells (FALCON semantics) instead of
      ``height_scan_resolution`` metres. This is the teacher heightmap the
      FALCON MoE-CNN teachers were trained on."""

    body_height_mode: Literal["raycast", "falcon"] = "raycast"
    """How the base / feet heights above terrain are measured.

    - ``"raycast"`` (default): one downward ray under the root / each foot.
    - ``"falcon"``: FAR-FALCON's ``_get_base_heights`` / ``_get_feet_heights`` on the
      raw heightfield. Base: root z minus the mean, over a yaw-rotated 3x5 grid
      (x in {-0.05, 0, 0.05}, y in {-0.1, ..., 0.1}), of the min of the cell and its
      +x / +y neighbours. Feet: max of the cell and its +x / +y neighbours. Biases the
      ground towards the lower (base) / upper (feet) step at stair edges, like FALCON.
      ``raw_base_height_terrain_mask`` still overrides both with raw world z."""


@dataclass(frozen=True)
class TerrainManagerCfg:
    terrain_term: TerrainTermCfg
    """Configuration for the terrain term. Only one terrain term is supported."""

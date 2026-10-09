"""Camera-manager presets (ray-cast camera terms driven by :class:`CameraManager`).

Distinct from :mod:`holosoma.config_values.sensor`, whose cameras are rendered by the simulator.
Extensions register presets into ``CAMERA_MANAGER_REGISTRY`` (e.g. via a
``holosoma.config.camera_manager`` entry point).
"""

from holosoma.config_types.camera import CameraManagerCfg
from holosoma.utils.config_registry import ConfigRegistry

CAMERA_MANAGER_REGISTRY = ConfigRegistry(CameraManagerCfg, group="holosoma.config.camera_manager")

none = CAMERA_MANAGER_REGISTRY.add("none", CameraManagerCfg())

__all__ = ["CAMERA_MANAGER_REGISTRY", "none"]

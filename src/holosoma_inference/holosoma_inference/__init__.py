"""Holosoma's device-free inference API.

Robot transport is initialized only by the explicit run_policy entry point.
"""

from holosoma_inference.native import (
    InferenceConfig,
    LocomotionPolicy,
    WholeBodyTrackingPolicy,
    get_config,
    get_policy_from_preset,
)

__all__ = ["InferenceConfig", "LocomotionPolicy", "WholeBodyTrackingPolicy", "get_config", "get_policy_from_preset"]

_LIVE_TYPES = {"LiveWbtPolicy", "LiveRobotObservation", "ReferenceFrame", "ReferenceSequence"}
__all__ += sorted(_LIVE_TYPES)


def __getattr__(name):
    if name in _LIVE_TYPES:
        from holosoma_inference import _native

        return getattr(_native, name)
    raise AttributeError(name)

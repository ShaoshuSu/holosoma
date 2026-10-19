"""Compatibility import for native whole-body tracking with Python transport."""

from holosoma_inference.policies.native_runner import NativeWbtRunner as WholeBodyTrackingPolicy

__all__ = ["WholeBodyTrackingPolicy"]

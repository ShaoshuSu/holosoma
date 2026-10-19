"""In-process native policy API, without robot SDK or input-device initialization."""

from __future__ import annotations

import copy
import json
import threading
from dataclasses import asdict, replace

import numpy as np

from holosoma_inference.config.config_types.inference import InferenceConfig


def get_config(name: str) -> InferenceConfig:
    """Return an independent registered preset, including extension presets."""
    from holosoma_inference.config.config_values.inference import INFERENCE_REGISTRY
    from holosoma_inference.utils.config_registry import load_entrypoint_presets

    load_entrypoint_presets(INFERENCE_REGISTRY)
    return copy.deepcopy(INFERENCE_REGISTRY[name[len("inference:"):] if name.startswith("inference:") else name])


def _native_config(config: InferenceConfig, *, use_gpu=False) -> str:
    document = asdict(config)
    # These transports are serviced by the Python runner, not by the native
    # standalone executable. Preserve the original dataclass for those callers.
    document["secondary"] = None
    task = document["task"]
    task["velocity_input"] = task["state_input"] = "keyboard"
    for shortcut in ("use_keyboard", "use_joystick", "use_usb_joystick"):
        task[shortcut] = False
    task.get("debug", {})["record_rosbag"] = False
    document["runtime"] = {"use_gpu": use_gpu}
    return json.dumps(document)


class NativePolicy:
    """A stateful C++ policy. step accepts one packed actor observation.

    Returns absolute joint targets with shape (1, num_joints). Use step_state
    to build observations/history from robot state inside the same native engine.
    Construction and stepping never initialize a robot SDK or send motor I/O.
    """

    def __init__(self, config: InferenceConfig, *, use_gpu=False, autostart=True):
        from holosoma_inference import _native

        self.config = config
        self._lock = threading.RLock()
        paths = config.task.model_path
        if isinstance(paths, str):
            paths = [paths]
        resolved = []
        for path in paths or []:
            if str(path).startswith(("wandb://", "https://")):
                from holosoma_inference.utils.wandb import load_checkpoint

                path = load_checkpoint(None, str(path))
            resolved.append(str(path))
        native_config = replace(config, task=replace(config.task, model_path=resolved))
        self._engine = _native.InferenceEngine(_native_config(native_config, use_gpu=use_gpu), autostart)

    def reset(self):
        with self._lock:
            self._engine.reset()

    def step(self, obs):
        with self._lock:
            return self._engine.step(np.asarray(obs, dtype=np.float32))

    def _state(self, values, tick_ms=0):
        from holosoma_inference import _native

        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 2 and values.shape[0] == 1:
            values = values[0]
        n = self.config.robot.num_joints
        if values.shape != (13 + 2 * n,):
            raise ValueError(f"Expected robot state shape ({13 + 2 * n},) or (1,{13 + 2 * n})")
        state = _native.RobotState()
        state.base_pos = values[:3]
        state.base_quat_wxyz = values[3:7]
        state.dof_pos = values[7 : 7 + n]
        state.base_lin_vel = values[7 + n : 10 + n]
        state.base_ang_vel = values[10 + n : 13 + n]
        state.dof_vel = values[13 + n :]
        state.tick_ms = tick_ms
        return state

    def step_state(self, state, *, tick_ms=0):
        with self._lock:
            return self._engine.step_state(self._state(state, tick_ms))

    def set_state(self, state, *, tick_ms=0):
        with self._lock:
            self._engine.set_state(self._state(state, tick_ms))

    def command(self, command):
        with self._lock:
            self._engine.command(getattr(command, "name", command))

    def velocity(self, x, y, yaw):
        with self._lock:
            self._engine.velocity(x, y, yaw)

    @property
    def last_observation(self):
        with self._lock:
            return np.asarray(self._engine.last_observation, dtype=np.float32)


class LocomotionPolicy(NativePolicy):
    """Configurable native locomotion inference without a hardware interface."""


class WholeBodyTrackingPolicy(NativePolicy):
    """Embedded-motion native WBT inference without a hardware interface."""


def get_policy_from_preset(config: InferenceConfig | str, **kwargs) -> NativePolicy:
    if isinstance(config, str):
        config = get_config(config)
    cls = WholeBodyTrackingPolicy if "motion_command" in config.observation.obs_dict.get("actor_obs", []) else LocomotionPolicy
    return cls(config, **kwargs)

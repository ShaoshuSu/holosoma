# Native inference library

The C++ library owns model execution, observation history, action scaling and
policy state. Python binds that same implementation. Constructing a library
policy opens no robot, socket, keyboard listener or SDK interface.

## Python

Build the extension with a C++17 compiler and CMake:

```sh
uv pip install ./src/holosoma_inference
```

Dependencies are resolved by CMake from installed packages or pinned downloads.
Set `ONNXRUNTIME_ROOT` for an existing SDK, including a CUDA-enabled SDK. The
default source build uses CPU inference. Robot SDKs are optional extras for the
explicit command-line runner, not dependencies of the device-free API.

```python
from dataclasses import replace
import numpy as np
import holosoma_inference as inference

config = inference.get_config("g1-29dof-loco")
config = replace(config, task=replace(config.task, model_path="policy.onnx"))
policy = inference.get_policy_from_preset(config)
policy.reset()

# One already-packed actor observation, in the model's configured term order.
obs_config = config.observation
width = sum(obs_config.obs_dims[t] for t in obs_config.obs_dict["actor_obs"])
width *= obs_config.history_length_dict.get("actor_obs", 1)
targets = policy.step(np.zeros((1, width), dtype=np.float32))
```

`get_config` returns an independent copy and includes registered extension
presets. SDK-specific fields on robot-config subclasses remain available to the
Python transport owner; the native engine receives the common robot contract.
`InferenceConfig` and `LocomotionPolicy(config)` are also exported for
composition without a named preset. Model paths may be local or supported W&B
URLs, resolved before native construction. Packed observations must have shape
`(N,)` or `(1, N)` and contain finite values. The returned `(1, num_joints)` array
contains absolute joint-position targets, including defaults and calibration
offsets, not raw neural-network actions.

For native observation construction, `policy.step_state(state)` accepts the
existing joint-ordered flat state:
`[base_pos(3), quaternion_wxyz(4), joint_pos(N), base_lin_vel(3),
base_ang_vel(3), joint_vel(N)]`, optionally followed by corrected gravity (3).
It returns a command with `q`, `dq`, `tau`, `kp` and `kd`. Start with
`autostart=False` for explicit `policy.command("START")` / `"STOP"` / `"INIT"`
control. `velocity(x, y, yaw)` supplies a locomotion request.
`step_state` advances the gait phase once per call; an external loop that already
advances it passes `advance_phase=False`. `infer_offsets(state)` warms the native
model and history without changing engagement or publishing a command.

## Live-reference whole-body tracking

Embedded-motion policies read their reference outputs from the model. A
live-reference policy instead receives reference frames from its caller and
derives its observation layout from ONNX `experiment_config` metadata. These
contracts remain explicit; neither guesses the other's layout from its width.

SONIC-aligned layout names refer to NVIDIA's public
[GEAR-SONIC model family](https://github.com/NVlabs/GR00T-WholeBodyControl).
The parser retains the corresponding exporter metadata keys for compatibility.

```python
from holosoma_inference import (
    LiveWbtPolicy, LiveRobotObservation, ReferenceFrame, ReferenceSequence,
)

policy = LiveWbtPolicy("live-policy.onnx")
observation = LiveRobotObservation()
reference = ReferenceSequence()
reference.current = ReferenceFrame()
offsets = policy.lookahead_offsets(8)
# Supply actual caller-buffered frames at these offsets; zero is the cursor.
reference.lookahead = [ReferenceFrame() for _ in offsets]
targets = policy.step(observation, reference, elapsed_seconds=0.0)
```

The snippet illustrates the types, not a robot motion sequence. Live frames use
canonical G1 joint order, wxyz orientation, body-local angular velocity and SI
units. `reset_heading()` reanchors heading and motion phase while preserving
history; `reset()` starts a new episode and clears history/action feedback.
With no explicit elapsed clock, the live API advances one model timestep per
call. `set_control_interval(seconds)` sets the elapsed-clock conversion.

## C++

```sh
cmake -S src/holosoma_inference/holosoma_cpp -B build/native \
  -DHOLOSOMA_BUILD_RUNNER=OFF -DHOLOSOMA_BUILD_TESTS=OFF
cmake --build build/native -j
cmake --install build/native --prefix "$HOME/.local"
```

Consumers use `find_package(holosomaInference CONFIG REQUIRED)` and link
`holosoma::inference`. Include `holosoma_cpp/inference_engine.hpp`, construct an
`InferenceEngine` from `InferenceConfig` or `config_from_yaml(text)`, and call
`step` or `step_state`. The live-reference API is in
`holosoma_cpp/holosoma_inference_core.hpp`. No robot SDK is needed to link either
API. Each C++ engine has a single owner; serialize calls to an instance. The
Python facade serializes its calls and releases the GIL during native inference.

## Command-line compatibility

The Python `python -m holosoma_inference.run_policy` entry point retains Tyro
arguments, presets, secondary policies, W&B resolution, SDK plugins and input
providers. Built-in policies use the native engine. The Python transport layer
supports explicit `interface`, `velocity_input`, `command_provider` and
`shared_hardware` injection instead of constructing hardware and then patching
it out. SDK command overrides remain in joint order; the SDK owns motor
permutation. A configured simulation clock continues to use the Python clock
subscriber, while the C++ engine owns timestep conversion.

For editable development, reinstall after changing C++ source. Container
dependency stages compile that source before the final Python-source overlay,
so a Python-only edit can reuse the native build and a C++ edit invalidates it.

The former Python policy implementations are retained only under
`holosoma_cpp/tests/python_reference` as independent replay oracles. They are
excluded from the installed package. Custom policies should compose the native
library and keep transport-specific logic outside the inference engine.

## Verification

```sh
uv pip install './src/holosoma_inference[dev]'
cmake -S src/holosoma_inference/holosoma_cpp -B build/native-tests \
  -DHOLOSOMA_WITH_UNITREE_SDK=OFF -DHOLOSOMA_BUILD_TESTS=ON \
  -DPython_EXECUTABLE="$(command -v python)"
cmake --build build/native-tests -j
ctest --test-dir build/native-tests --output-on-failure
python -m pytest --import-mode=importlib src/holosoma_inference/tests -q
```

Tests cover both model contracts, observation history, heading/FK, lookahead,
clocks, reload failure, native Python loading and SDK-boundary joint ordering.
Synthetic models and recorded golden vectors do not establish physical robot
stability; evaluate the selected checkpoint on the intended robot separately.

"""Exercise the installed native library with a real ONNX model, no robot SDK."""

import gc
import json
import runpy
import subprocess
import sys
import weakref
from dataclasses import replace
from importlib.resources import files
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper
from pydantic.dataclasses import dataclass

import holosoma_inference as inference
from holosoma_inference.config.config_types.robot import RobotConfig


@pytest.fixture
def native_config(tmp_path):
    config = inference.get_config("g1-29dof-loco")
    obs = config.observation
    width = sum(obs.obs_dims[name] for name in obs.obs_dict["actor_obs"]) * obs.history_length_dict.get("actor_obs", 1)
    weights = np.zeros((width, 29), dtype=np.float32)
    weights[:29] = np.eye(29, dtype=np.float32)
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["actor_obs", "weights"], ["actions"])],
        "native-api-fixture",
        [helper.make_tensor_value_info("actor_obs", TensorProto.FLOAT, [1, width])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 29])],
        [numpy_helper.from_array(weights, "weights")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    helper.set_model_props(model, {"kp": json.dumps([40.0] * 29), "kd": json.dumps([2.0] * 29)})
    path = tmp_path / "policy.onnx"
    onnx.save(model, path)
    return replace(config, task=replace(config.task, model_path=str(path)), secondary=None), width


def test_library_import_has_no_robot_side_effects():
    code = (
        "import sys; import holosoma_inference; from holosoma_inference import _native; "
        "assert not any(n in sys.modules for n in "
        "('unitree_interface','rclpy','pinocchio','holosoma_inference.sdk'))"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_packed_observation_uses_native_graph_scaling_and_defaults(native_config):
    config, width = native_config
    policy = inference.LocomotionPolicy(config)
    obs = np.arange(width, dtype=np.float32)[None, :]
    expected = np.minimum(obs[:, :29], 100) * np.float32(config.task.policy_action_scale) + np.asarray(
        config.robot.default_dof_angles
    )
    np.testing.assert_allclose(policy.step(obs), expected, rtol=1e-6, atol=1e-7)
    policy.reset()
    np.testing.assert_allclose(policy.step(obs), expected, rtol=1e-6, atol=1e-7)


def test_native_rejects_invalid_shapes_and_nonfinite_input(native_config):
    config, width = native_config
    policy = inference.get_policy_from_preset(config)
    for obs in (np.zeros((2, width)), np.zeros((1, width - 1)), np.full((1, width), np.nan)):
        with pytest.raises(ValueError, match="observation"):
            policy.step(obs)
    with pytest.raises(ValueError, match="orientation"):
        policy.step_state(np.zeros((1, 71)))  # invalid all-zero orientation
    assert np.isfinite(policy.step(np.zeros((1, width)))).all()


def test_state_api_preserves_hold_and_explicit_engagement(native_config):
    config, _ = native_config
    policy = inference.get_policy_from_preset(config, autostart=False)
    state = np.zeros((1, 71))
    state[0, 3] = 1
    state[0, 7:36] = np.linspace(-0.2, 0.2, 29)
    held = policy.step_state(state)
    np.testing.assert_allclose(held.q, state[0, 7:36])
    assert policy.last_observation.size == 0
    policy.command("START")
    active = policy.step_state(state)
    assert policy.last_observation.size > 0
    assert not np.allclose(active.q, held.q)
    policy.command("STOP")
    np.testing.assert_allclose(policy.step_state(state).q, held.q)


def test_state_api_preserves_explicit_corrected_gravity(native_config):
    config, _ = native_config
    policy = inference.get_policy_from_preset(config)
    state = np.zeros((1, 74))
    state[0, 3] = 1
    state[0, -3:] = [0.2, -0.1, -0.97]
    policy.step_state(state)
    terms = sorted(config.observation.obs_dict["actor_obs"])
    history = config.observation.history_length_dict.get("actor_obs", 1)
    start = sum(config.observation.obs_dims[t] * history for t in terms[: terms.index("projected_gravity")])
    latest = start + (history - 1) * 3
    np.testing.assert_allclose(policy.last_observation[latest : latest + 3], state[0, -3:], atol=1e-6)


def test_state_api_advances_locomotion_phase_once_per_tick(native_config):
    config, _ = native_config
    config = replace(config, task=replace(config.task, auto_walk_on_vel_cmd=True, use_phase=True))
    policy = inference.get_policy_from_preset(config)
    state = np.zeros((1, 71))
    state[0, 3] = 1
    policy.velocity(0.2, 0.0, 0.0)
    policy.step_state(state)
    first = policy.last_observation.copy()
    policy.step_state(state)
    second = policy.last_observation.copy()
    terms = sorted(config.observation.obs_dict["actor_obs"])
    start = sum(config.observation.obs_dims[t] for t in terms[: terms.index("sin_phase")])
    dt = 2 * np.pi / (config.task.rl_rate * config.task.gait_period)
    np.testing.assert_allclose(first[start : start + 2], [np.sin(dt), -np.sin(dt)], atol=1e-6)
    np.testing.assert_allclose(second[start : start + 2], [np.sin(2 * dt), -np.sin(2 * dt)], atol=1e-6)


def test_preset_lookup_does_not_share_mutable_observation_maps():
    config = inference.get_config("g1-29dof-loco")
    config.observation.obs_dims["native_test_only"] = 9
    assert "native_test_only" not in inference.get_config("g1-29dof-loco").observation.obs_dims


def test_python_factory_uses_the_public_policy_interface(native_config):
    config, _ = native_config
    policy = inference.get_policy_from_preset(config)
    assert isinstance(policy._engine, inference.Policy)
    state = inference.RobotState()
    state.dof_pos = [0.0] * 29
    state.dof_vel = [0.0] * 29
    command = policy._engine.step(state)
    assert isinstance(command, inference.JointCommand)
    assert len(command.q) == policy._engine.num_dofs() == len(policy._engine.dof_names) == 29
    with pytest.raises(RuntimeError, match="own reference"):
        policy._engine.set_reference(inference.ReferenceWindow())


def test_public_policy_borrows_one_reference_and_returns_command_snapshots(tmp_path):
    generator = Path(__file__).parents[1] / "holosoma_cpp/tests/gen_live_stub_models.py"
    write_model = runpy.run_path(str(generator))["_write_model"]
    checkpoint = tmp_path / "live.onnx"
    write_model(checkpoint, history=9, lookahead_layout="flat_0p2s")
    policy = inference.Policy.from_live_model(str(checkpoint))
    assert policy.lookahead_offsets == [0, 2, 4, 6, 8]
    state = inference.RobotState()
    state.dof_pos = state.dof_vel = [0.0] * 29
    with pytest.raises(RuntimeError, match="set_reference"):
        policy.step(state)

    def window():
        reference = inference.ReferenceWindow()
        reference.q = reference.dq = [0.0] * (5 * 29)
        reference.quat = [[1, 0, 0, 0]] * 5
        return reference

    first = window()
    retained = weakref.ref(first)
    policy.set_reference(first)
    del first
    gc.collect()
    assert retained() is not None
    command = policy.step(state)
    saved = list(command.q)
    for _ in range(9):
        policy.step(state)
    assert command.q == saved
    assert policy.step(state).q != saved
    assert len(policy.last_observation) == 2267
    policy.set_reference(window())
    gc.collect()
    assert retained() is None
    assert len(policy.step(state).q) == 29
    policy.set_reference(None)
    with pytest.raises(RuntimeError, match="set_reference"):
        policy.step(state)


def test_library_accepts_robot_config_with_host_transport_fields(native_config):
    from dataclasses import asdict

    @dataclass(frozen=True)
    class TransportRobot(RobotConfig):
        command_topic: str = "/example/commands"

    config, width = native_config
    robot = TransportRobot(**asdict(config.robot))
    config = replace(config, robot=robot)
    policy = inference.get_policy_from_preset(config)
    assert policy.config.robot.command_topic == "/example/commands"
    assert np.isfinite(policy.step(np.zeros((1, width)))).all()


def test_embedded_motion_resumes_progress_after_a_library_reset():
    config = inference.get_config("g1-29dof-wbt")
    path = files("holosoma_inference").joinpath("models/wbt/fastsac_g1_29dof_dancing.onnx")
    config = replace(config, task=replace(config.task, model_path=str(path), use_sim_time=False), secondary=None)
    policy = inference.get_policy_from_preset(config)
    state = np.zeros((1, 71))
    state[0, 3] = 1
    state[0, 7:36] = config.robot.default_dof_angles
    policy.step_state(state)
    policy.step_state(state)
    assert policy.motion_timestep == 2
    policy.reset()
    policy.step_state(state)
    policy.step_state(state)
    assert policy.motion_timestep == 2


def test_library_autostart_captures_the_first_robot_heading():
    config = inference.get_config("g1-29dof-wbt")
    path = files("holosoma_inference").joinpath("models/wbt/fastsac_g1_29dof_dancing.onnx")
    config = replace(config, task=replace(config.task, model_path=str(path), use_sim_time=False), secondary=None)
    observations = []
    for yaw in (0.0, 1.0):
        policy = inference.get_policy_from_preset(config)
        state = np.zeros((1, 71))
        state[0, 3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        state[0, 7:36] = config.robot.default_dof_angles
        policy.step_state(state)
        observations.append(policy.last_observation)
    np.testing.assert_allclose(observations[0], observations[1], atol=2e-5)


def test_python_runner_passes_joint_ordered_gains_to_the_sdk(native_config):
    from holosoma_inference.policies.native_runner import NativeLocomotionRunner

    config, _ = native_config
    permutation = tuple(reversed(range(29)))
    config = replace(
        config,
        robot=replace(
            config.robot,
            joint2motor=permutation,
            motor2joint=permutation,
            motor_kp=tuple(range(10, 39)),
            motor_kd=tuple(range(1, 30)),
        ),
    )

    class Interface:
        kp_level = kd_level = 1.0

        def get_low_state(self):
            state = np.zeros((1, 71))
            state[0, 3] = 1
            return state

        def send_low_command(self, *args, **kwargs):
            self.sent = kwargs

    class Input:
        def poll_velocity(self):
            return None

        def poll_commands(self):
            return []

    interface, inputs = Interface(), Input()
    runner = NativeLocomotionRunner(config, interface=interface, velocity_input=inputs, command_provider=inputs)
    runner._handle_start_policy()
    runner.policy_action()
    np.testing.assert_allclose(interface.sent["kp_override"], np.asarray(config.robot.motor_kp)[list(permutation)])


def test_injected_runner_can_warm_up_without_publishing_or_engaging(native_config):
    from holosoma_inference.policies.native_runner import NativeLocomotionRunner

    config, width = native_config

    class Interface:
        def get_low_state(self):
            state = np.zeros((1, 71))
            state[0, 3] = 1
            return state

        def send_low_command(self, *_args, **_kwargs):
            pytest.fail("warm-up published a command")

    class Input:
        def poll_velocity(self):
            return None

        def poll_commands(self):
            return []

    interface, provider = Interface(), Input()
    runner = NativeLocomotionRunner(config, interface=interface, velocity_input=provider, command_provider=provider)
    result = runner.rl_inference(interface.get_low_state())
    assert result.shape == (1, 29) and np.isfinite(result).all()
    assert runner.native.last_observation.size == width
    assert not runner.use_policy_action


@pytest.mark.parametrize(
    ("override", "message"),
    [({"q": [np.nan] * 29}, "finite"), ({"q": [0.0]}, "finite"), ({"kp": [-1.0] * 29}, "nonnegative")],
)
def test_invalid_robot_output_transform_never_reaches_transport(native_config, override, message):
    from holosoma_inference.policies.native_runner import NativeLocomotionRunner

    class Runner(NativeLocomotionRunner):
        def transform_command(self, _command, _state):
            return override

    class Interface:
        def get_low_state(self):
            state = np.zeros((1, 71))
            state[0, 3] = 1
            return state

        def send_low_command(self, *_args, **_kwargs):
            pytest.fail("invalid transformed command reached the SDK")

    class Inputs:
        def poll_velocity(self):
            return None

        def poll_commands(self):
            return []

    config, _ = native_config
    inputs = Inputs()
    runner = Runner(config, interface=Interface(), velocity_input=inputs, command_provider=inputs)
    with pytest.raises(ValueError, match=message):
        runner.policy_action()

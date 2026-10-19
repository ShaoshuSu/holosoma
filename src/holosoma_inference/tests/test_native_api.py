"""Exercise the installed native library with a real ONNX model, no robot SDK."""

import json
import subprocess
import sys
from dataclasses import replace

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

import holosoma_inference as inference


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
    code = "import sys; import holosoma_inference; from holosoma_inference import _native; assert not any(n in sys.modules for n in ('unitree_interface','rclpy','pinocchio','holosoma_inference.sdk'))"
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
        with pytest.raises(ValueError):
            policy.step(obs)
    with pytest.raises(ValueError):
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


def test_preset_lookup_does_not_share_mutable_observation_maps():
    config = inference.get_config("g1-29dof-loco")
    config.observation.obs_dims["native_test_only"] = 9
    assert "native_test_only" not in inference.get_config("g1-29dof-loco").observation.obs_dims


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

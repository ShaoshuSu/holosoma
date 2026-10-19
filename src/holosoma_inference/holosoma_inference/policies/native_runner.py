"""Python SDK/input transport around the shared native control engine."""

from __future__ import annotations

import sys

import numpy as np

from holosoma_inference.inputs.api.commands import StateCommand
from holosoma_inference.native import NativePolicy
from holosoma_inference.policies.transport import RunnerTransport


class NativeRunnerPolicy(RunnerTransport):
    """Retain Python SDK plugins and input providers; execute policy logic in C++.

    Explicit injection is supported without pre-initialization attribute patches:
    interface owns I/O, velocity_input and command_provider own operator inputs.
    """

    is_wbt = False

    def __init__(
        self, config, *, interface=None, velocity_input=None, command_provider=None, shared_hardware=None, clock=None
    ):
        self.config = config
        self._init_robot_config(config.robot)
        self.native = NativePolicy(config, autostart=False)
        self.model_paths = (
            list(config.task.model_path)
            if isinstance(config.task.model_path, (list, tuple))
            else [config.task.model_path]
        )
        self._init_rate_handler()
        self._init_latency_tracking()
        self.use_phase = config.task.use_phase
        self.use_keyboard = False
        self.use_joystick = bool({"interface", "joystick"} & {config.task.velocity_input, config.task.state_input})
        self.cmd_q = np.zeros(self.num_dofs)
        self._clock = clock
        self._owns_clock = False
        self._owns_inputs = velocity_input is None and command_provider is None and shared_hardware is None
        if shared_hardware is not None:
            interface = shared_hardware.interface
            velocity_input = shared_hardware._velocity_input
            command_provider = shared_hardware._command_provider
        if interface is None:
            self._init_sdk_components()
            self._init_communication_components()
        else:
            self.interface = interface
            self.sdk_type = config.robot.sdk_type
        self._resolve_control_gains()
        if velocity_input is not None and command_provider is not None:
            self._velocity_input = velocity_input
            self._command_provider = command_provider
        elif velocity_input is not None or command_provider is not None:
            raise ValueError("Supply both velocity_input and command_provider")
        else:
            self._init_input_device()
        if self.is_wbt and config.task.use_sim_time and self._clock is None:
            if shared_hardware is not None and getattr(shared_hardware, "_clock", None) is not None:
                self._clock = shared_hardware._clock
            else:
                from holosoma_inference.utils.clock import ClockSub

                self._clock = ClockSub()
                self._clock.start()
                self._owns_clock = True
        if self.is_wbt and shared_hardware is None and not config.task.skip_stiff_prompt and sys.stdin.isatty():
            input("Ready to enter stiff hold mode. Press Enter to continue...")

    @property
    def use_policy_action(self):
        return self.native._engine.active

    @use_policy_action.setter
    def use_policy_action(self, value):
        self.native.command("START" if value else "STOP")

    @property
    def get_ready_state(self):
        return self.native._engine.ready

    @property
    def active_policy_index(self):
        return self.native._engine.active_index

    @property
    def active_model_path(self):
        return self.model_paths[self.active_policy_index]

    def _resolve_control_gains(self):
        # Both native commands and Python SDK overrides are joint-ordered.
        # SDK transport owns motor permutation and its operator gain scale.
        self._kp = np.asarray(self.native._engine.joint_kp)
        self._kd = np.asarray(self.native._engine.joint_kd)

    def _tick_ms(self):
        return self._clock.get_clock() if self._clock is not None else 0

    def _apply_velocity(self, command):
        self.native.velocity(float(command.lin_vel[0]), float(command.lin_vel[1]), float(command.ang_vel))

    def update_phase_time(self):
        self.native._engine.update_phase()

    def _init_phase_components(self):
        self.native.reset()

    def _dispatch_command(self, command):
        if command == StateCommand.KILL:
            raise SystemExit(0)
        if command in {
            StateCommand.KP_UP,
            StateCommand.KP_DOWN,
            StateCommand.KP_UP_FINE,
            StateCommand.KP_DOWN_FINE,
            StateCommand.KP_RESET,
        }:
            increments = {
                StateCommand.KP_UP: 0.1,
                StateCommand.KP_DOWN: -0.1,
                StateCommand.KP_UP_FINE: 0.01,
                StateCommand.KP_DOWN_FINE: -0.01,
            }
            self.interface.kp_level = (
                1.0 if command == StateCommand.KP_RESET else max(0.0, self.interface.kp_level + increments[command])
            )
            return
        previous_index = self.active_policy_index
        if command == StateCommand.START:
            self._handle_start_policy()
        elif command == StateCommand.STOP:
            self._handle_stop_policy()
        elif command == StateCommand.INIT:
            self._handle_init_state()
        else:
            if self.is_wbt:
                self.native.set_state(self.interface.get_low_state(), tick_ms=self._tick_ms())
            self.native.command(command)
        self._resolve_control_gains()
        if self.active_policy_index != previous_index:
            self._on_policy_switched(self.active_model_path)

    def _on_policy_switched(self, model_path):
        """Observe a native model switch without replacing inference."""

    def _handle_start_policy(self):
        self.native.set_state(self.interface.get_low_state(), tick_ms=self._tick_ms())
        self.native.command("START")
        if hasattr(self.interface, "no_action"):
            self.interface.no_action = 0

    def _handle_stop_policy(self):
        self.native.command("STOP")
        if hasattr(self.interface, "no_action"):
            self.interface.no_action = 0 if self.is_wbt else 1

    def _handle_init_state(self):
        self.native.command("INIT")
        if hasattr(self.interface, "no_action"):
            self.interface.no_action = 0

    def policy_action(self):
        state = self.interface.get_low_state()
        command = self.native.step_state(state, tick_ms=self._tick_ms(), advance_phase=False)
        transformed = self.transform_command(command, state) or {}
        q = np.asarray(transformed.get("q", command.q), dtype=float).reshape(-1)
        kp = np.asarray(transformed.get("kp", command.kp), dtype=float).reshape(-1)
        kd = np.asarray(transformed.get("kd", command.kd), dtype=float).reshape(-1)
        if any(value.shape != (self.num_dofs,) or not np.isfinite(value).all() for value in (q, kp, kd)):
            raise ValueError("Command transform must return finite joint-ordered q/kp/kd arrays")
        if (kp < 0).any() or (kd < 0).any():
            raise ValueError("Command transform gains must be nonnegative")
        self.cmd_q[:] = q
        self.interface.send_low_command(
            self.cmd_q,
            np.asarray(command.dq),
            np.asarray(command.tau),
            state[0, 7 : 7 + self.num_dofs],
            kp_override=kp,
            kd_override=kd,
        )
        self._on_command_sent(self.cmd_q, state)

    def transform_command(self, command, robot_state_data):
        """Optional robot-specific output adaptation; inference remains native.

        Return overrides for final, calibrated joint-order q/kp/kd, or None.
        The SDK still owns motor permutation and transmission.
        """
        return

    def rl_inference(self, robot_state_data):
        """Compatibility warm-up hook: compute offsets, publish nothing."""
        return self.native.infer_offsets(robot_state_data, tick_ms=self._tick_ms())

    def close(self):
        if self._owns_clock and self._clock is not None:
            self._clock.close()
            self._clock = None
        if self._owns_inputs:
            for provider in {
                id(self._velocity_input): self._velocity_input,
                id(self._command_provider): self._command_provider,
            }.values():
                stop = getattr(provider, "stop", None)
                if callable(stop):
                    stop()
            self._owns_inputs = False

    def run(self):
        try:
            super().run()
        finally:
            self.close()


class NativeLocomotionRunner(NativeRunnerPolicy):
    """Native locomotion with Python-owned transport."""


class NativeWbtRunner(NativeRunnerPolicy):
    """Native embedded-motion tracking with Python-owned transport."""

    is_wbt = True

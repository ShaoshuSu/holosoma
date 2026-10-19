"""Python SDK/input transport around the shared native control engine."""

from __future__ import annotations

import sys

import numpy as np

from holosoma_inference.inputs.api.commands import StateCommand
from holosoma_inference.native import NativePolicy
from holosoma_inference.policies.base import BasePolicy


class NativeRunnerPolicy(BasePolicy):
    """Retain Python SDK plugins and input providers; execute policy logic in C++.

    Explicit injection is supported without pre-initialization attribute patches:
    interface owns I/O, velocity_input and command_provider own operator inputs.
    """

    is_wbt = False

    def __init__(self, config, *, interface=None, velocity_input=None, command_provider=None, shared_hardware=None):
        self.config = config
        self._init_robot_config(config.robot)
        self.native = NativePolicy(config, autostart=False)
        self.model_paths = list(config.task.model_path) if isinstance(config.task.model_path, (list, tuple)) else [config.task.model_path]
        self._init_latency_tracking()
        self._init_rate_handler()
        self.use_phase = config.task.use_phase
        self.use_keyboard = False
        self.use_joystick = bool({"interface", "joystick"} & {config.task.velocity_input, config.task.state_input})
        self.cmd_q = np.zeros(self.num_dofs)
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
        # Native gains are joint-ordered. Python SDK overrides take motor order
        # and apply their own operator gain scale, so do not apply it twice.
        order = self.robot_config.motor2joint
        self._kp = np.asarray(self.native._engine.joint_kp)[order]
        self._kd = np.asarray(self.native._engine.joint_kd)[order]

    def _tick_ms(self):
        clock = getattr(self.interface, "get_sim_time", None)
        return int(clock() * 1000) if callable(clock) else 0

    def _apply_velocity(self, command):
        self.native.velocity(float(command.lin_vel[0]), float(command.lin_vel[1]), float(command.ang_vel))

    def update_phase_time(self):
        self.native._engine.update_phase()

    def _init_phase_components(self):
        self.native.reset()

    def _dispatch_command(self, command):
        if command == StateCommand.KILL:
            raise SystemExit(0)
        if command in {StateCommand.KP_UP, StateCommand.KP_DOWN, StateCommand.KP_UP_FINE, StateCommand.KP_DOWN_FINE, StateCommand.KP_RESET}:
            increments = {StateCommand.KP_UP: 0.1, StateCommand.KP_DOWN: -0.1, StateCommand.KP_UP_FINE: 0.01, StateCommand.KP_DOWN_FINE: -0.01}
            self.interface.kp_level = 1.0 if command == StateCommand.KP_RESET else max(0.0, self.interface.kp_level + increments[command])
            return
        if command == StateCommand.START:
            self._handle_start_policy()
        elif command == StateCommand.STOP:
            self._handle_stop_policy()
        else:
            self.native.command(command)
        self._resolve_control_gains()

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
        command = self.native.step_state(state, tick_ms=self._tick_ms())
        self.cmd_q[:] = command.q
        order = self.robot_config.motor2joint
        self.interface.send_low_command(self.cmd_q, np.asarray(command.dq), np.asarray(command.tau),
                                       state[0, 7:7+self.num_dofs],
                                       kp_override=np.asarray(command.kp)[order],
                                       kd_override=np.asarray(command.kd)[order])
        self._on_command_sent(self.cmd_q, state)


class NativeLocomotionRunner(NativeRunnerPolicy):
    """Native locomotion with Python-owned transport."""


class NativeWbtRunner(NativeRunnerPolicy):
    """Native embedded-motion tracking with Python-owned transport."""

    is_wbt = True

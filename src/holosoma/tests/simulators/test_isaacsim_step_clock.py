"""Check the IsaacSim stepping contract without requiring an IsaacSim install."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace


def load_step_method():
    source = Path(__file__).parents[2] / "holosoma/simulator/isaacsim/isaacsim.py"
    tree = ast.parse(source.read_text())
    simulator = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "IsaacSim")
    method = next(
        node
        for node in simulator.body
        if isinstance(node, ast.FunctionDef) and node.name == "simulate_at_each_physics_step"
    )
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["simulate_at_each_physics_step"]


class FakeSimulationContext:
    """IsaacSim 5.1's render=True advances an app/render interval, not one physics tick."""

    def __init__(self, gui=False, sensor=False):
        self.gui = gui
        self.sensor = sensor
        self.physics_ticks = 0
        self.render_calls = 0

    def has_gui(self):
        return self.gui

    def has_rtx_sensors(self):
        return self.sensor

    def step(self, render=True):
        self.physics_ticks += 4 if render else 1

    def render(self):
        self.render_calls += 1


class IsaacSimStepClockTest(unittest.TestCase):
    def test_rendering_preserves_control_dt(self):
        step = load_step_method()
        for mode in ("headless", "video", "gui", "sensor"):
            with self.subTest(mode=mode):
                sim = FakeSimulationContext(gui=mode == "gui", sensor=mode == "sensor")
                updates = []
                subject = SimpleNamespace(
                    _sim_step_counter=0,
                    sim=sim,
                    video_recorder=SimpleNamespace(is_recording=mode == "video"),
                    scene=SimpleNamespace(write_data_to_sim=lambda: None, update=lambda dt: updates.append(dt)),
                    simulator_config=SimpleNamespace(
                        sim=SimpleNamespace(fps=200, render_interval_steps=4),
                        bridge=SimpleNamespace(enabled=False),
                    ),
                    debug_viz_enabled=False,
                    _robot=SimpleNamespace(
                        data=SimpleNamespace(joint_pos=ArrayStub(), joint_vel=ArrayStub()),
                    ),
                    dof_ids=[0],
                )
                for _ in range(8):
                    step(subject)
                self.assertEqual(sim.physics_ticks, 8, "Two control steps must advance exactly 40 ms")
                self.assertEqual(updates, [0.005] * 8)
                self.assertEqual(sim.render_calls, 0 if mode == "headless" else 2)


class ArrayStub:
    def __getitem__(self, key):
        return self


if __name__ == "__main__":
    unittest.main()

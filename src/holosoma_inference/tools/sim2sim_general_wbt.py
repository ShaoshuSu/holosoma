"""Sim-to-sim for an actor-only (general tracker) WBT export, in-process MuJoCo.

The split-process workflow in docs/workflows/sim-to-sim-wbt.md runs run_sim.py and
run_policy.py over loopback DDS and is driven by keyboard. That is the right thing for a human
and the wrong thing for a pass/fail check: there is no way to assert "it finished the clip".

This is the same loop with the transport removed -- one process, MuJoCo stepped directly, the
policy at task rate, the reference read from a retargeted npz. It reports how much of the clip
the robot completed before falling, which is the thing worth gating on.

    python sim2sim_general_wbt.py --model model_49000.onnx \
        --motion .../sub3_largebox_003_mj.npz

Everything robot-specific comes from the export's metadata (dof order, gains, action scale,
default pose), so this does not read an inference preset at all. The joint order of the MJCF, the
motion file and the export all differ; all three are mapped by name.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import mujoco
import numpy as np
import onnx
import onnxruntime as ort
from scipy.spatial.transform import Rotation

SCENE = "src/holosoma/holosoma/data/robots/g1/scenes/scene_g1_29dof_wbt_plane.xml"
FUTURE_OFFSETS = (0, 2, 4, 6, 8)
HISTORY = 9
ACTION_CLIP = 20.0
OBS_DIM = 2267
INACTIVE = 240 + 9 + 12 + 6 + 720 + 60 + 60  # teleop then SMPL, zero for this export


def metadata(path: pathlib.Path) -> dict:
    return {p.key: p.value for p in onnx.load(path, load_external_data=False).metadata_props}


class Clip:
    """A 50 Hz retargeted wbt npz, permuted into the export's dof order."""

    def __init__(self, path: pathlib.Path, dof_names: list[str]):
        with np.load(path, allow_pickle=True) as z:
            names = [str(n) for n in z["joint_names"]]
            fps = float(np.asarray(z["fps"]).reshape(-1)[0])
            joint_pos, joint_vel = np.asarray(z["joint_pos"]), np.asarray(z["joint_vel"])
        if sorted(names) != sorted(dof_names):
            raise SystemExit("motion joint_names do not match the export's joints")
        perm = [names.index(n) for n in dof_names]
        self.fps = fps
        self.q = joint_pos[:, 7:][:, perm]
        self.dq = joint_vel[:, 6:][:, perm]
        self.root_pos = joint_pos[:, :3]
        self.root_quat = joint_pos[:, 3:7]  # wxyz
        self.frames = len(self.q)

    def window(self, index: int):
        picked = np.clip(np.add(index, FUTURE_OFFSETS), 0, self.frames - 1)
        return self.q[picked], self.dq[picked], self.root_quat[picked]


class Tracker:
    """The 2267-D observation and the PD decode. Mirrors the deployed contract."""

    def __init__(self, model_path: pathlib.Path):
        meta = metadata(model_path)
        if meta.get("wbt_onnx_export_mode") != "actor_only":
            raise SystemExit(f"{model_path}: not an actor_only export")
        if meta.get("anchor_body_name", "pelvis") != "pelvis":
            raise SystemExit(f"{model_path}: only pelvis-anchored exports are supported")
        self.dof_names = json.loads(meta["dof_names"])
        self.kp = np.asarray(json.loads(meta["kp"]), float)
        self.kd = np.asarray(json.loads(meta["kd"]), float)
        self.action_scale = np.asarray(json.loads(meta["action_scale"]), float)
        self.default_pos = np.asarray(json.loads(meta["default_joint_pos"]), float)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
        if self.session.get_inputs()[0].shape != [1, OBS_DIM]:
            raise SystemExit(f"{model_path}: expected a {OBS_DIM}-D 0.2 s export")
        self.reset()

    def reset(self):
        self._history = None
        self._yaw0 = None
        self.last_action = np.zeros(len(self.dof_names))
        self.step_index = 0

    def act(self, quat_wxyz, ang_vel_b, q, dq, window):
        rot = Rotation.from_quat(np.asarray(quat_wxyz)[[1, 2, 3, 0]])
        frame = [
            rot.inv().apply([0.0, 0.0, -1.0]),
            np.asarray(ang_vel_b, float),
            np.asarray(q, float) - self.default_pos,
            np.asarray(dq, float),
            self.last_action,
        ]
        if self._history is None:  # repeat_first
            self._history = [[f.copy() for _ in range(HISTORY)] for f in frame]
        else:
            for buffer, f in zip(self._history, frame):
                buffer.pop(0)
                buffer.append(np.asarray(f).copy())
        proprio = np.concatenate([np.concatenate(b) for b in self._history])

        ref_q, ref_dq, ref_quat = window
        ref_rot = Rotation.from_quat(np.asarray(ref_quat)[:, [1, 2, 3, 0]])
        if self._yaw0 is None:
            self._yaw0 = (_heading(rot), _heading(ref_rot[0]))
        relative = _strip(rot, self._yaw0[0]).inv() * _strip(ref_rot, self._yaw0[1])
        obs = np.concatenate(
            [
                proprio,
                np.asarray(ref_q).reshape(-1),
                np.asarray(ref_dq).reshape(-1),
                relative.as_matrix()[..., :2].reshape(-1),
                np.zeros(INACTIVE),
                [1.0, 0.0, 0.0],
            ]
        ).astype(np.float32)[None]
        if obs.shape != (1, OBS_DIM):
            raise SystemExit(f"assembled {obs.shape}, expected (1, {OBS_DIM})")

        raw = self.session.run(
            ["actions"], {"obs": obs, "time_step": np.array([[float(self.step_index)]], np.float32)}
        )[0][0]
        self.last_action = np.clip(raw.astype(float), -ACTION_CLIP, ACTION_CLIP)
        self.step_index += 1
        return self.default_pos + self.last_action * self.action_scale


def _heading(rot: Rotation) -> float:
    forward = rot.apply([1.0, 0.0, 0.0])
    return float(np.arctan2(forward[1], forward[0]))


def _strip(rot: Rotation, yaw: float) -> Rotation:
    return Rotation.from_euler("z", -yaw) * rot


class Plant:
    """MuJoCo with explicit PD, as the deployed driver does. Joints mapped by name."""

    def __init__(self, scene: str, dof_names: list[str], rl_rate: float):
        self.model = mujoco.MjModel.from_xml_path(scene)
        self.data = mujoco.MjData(self.model)
        self.decimation = round((1.0 / rl_rate) / self.model.opt.timestep)

        def joint_id(name):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise SystemExit(f"{name} not in the MJCF")
            return jid

        self.qpos_adr = np.array([self.model.jnt_qposadr[joint_id(n)] for n in dof_names])
        self.qvel_adr = np.array([self.model.jnt_dofadr[joint_id(n)] for n in dof_names])
        self.ctrl_adr = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in dof_names])
        if (self.ctrl_adr < 0).any():
            raise SystemExit("an export joint has no actuator in the MJCF")

    def settle(self, q_target, kp, kd, seconds):
        for _ in range(int(seconds / self.model.opt.timestep)):
            self._apply(q_target, kp, kd)
            mujoco.mj_step(self.model, self.data)

    def hold(self, q_target, kp, kd):
        for _ in range(self.decimation):
            self._apply(q_target, kp, kd)
            mujoco.mj_step(self.model, self.data)

    def _apply(self, q_target, kp, kd):
        q = self.data.qpos[self.qpos_adr]
        dq = self.data.qvel[self.qvel_adr]
        self.data.ctrl[self.ctrl_adr] = kp * (q_target - q) - kd * dq

    def reset_to(self, root_pos, root_quat_wxyz, joints):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = root_pos
        self.data.qpos[3:7] = root_quat_wxyz
        self.data.qpos[self.qpos_adr] = joints
        mujoco.mj_forward(self.model, self.data)

    @property
    def state(self):
        return (
            self.data.qpos[3:7].copy(),  # wxyz
            # A free joint's qvel[3:6] is ALREADY in the body frame (qvel[0:3] is world-frame
            # linear). Rotating it again is a double transform and corrupts the gyro the policy
            # balances on.
            self.data.qvel[3:6].copy(),
            self.data.qpos[self.qpos_adr].copy(),
            self.data.qvel[self.qvel_adr].copy(),
        )

    @property
    def pelvis_height(self) -> float:
        return float(self.data.qpos[2])

    @property
    def upright(self) -> float:
        """z of the pelvis up-axis in world: 1 upright, 0 on its side, negative inverted."""
        return float(Rotation.from_quat(self.data.qpos[3:7][[1, 2, 3, 0]]).apply([0, 0, 1])[2])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=pathlib.Path)
    ap.add_argument("--motion", required=True, type=pathlib.Path)
    ap.add_argument("--scene", default=SCENE)
    ap.add_argument("--rl-rate", type=float, default=50.0)
    ap.add_argument(
        "--freeze",
        type=float,
        default=1.0,
        help="seconds of policy stepping with the reference held at frame 0, before the clip "
        "advances -- the scripted equivalent of pressing ] and waiting before m. A static PD "
        "hold is not enough: the export's gains are tracking gains, and the policy has to be "
        "running to balance.",
    )
    # Absolute height is a poor detector for a clip that legitimately crouches; this one's
    # reference dips to 0.45 m. Tilt is unambiguous, and height is judged against the reference.
    ap.add_argument("--fall-upright", type=float, default=0.3, help="pelvis up-axis z below this is a fall")
    ap.add_argument("--fall-sag", type=float, default=0.25, help="metres below the reference pelvis height")
    args = ap.parse_args()

    tracker = Tracker(args.model)
    clip = Clip(args.motion, tracker.dof_names)
    if not np.isclose(clip.fps, args.rl_rate):
        raise SystemExit(f"clip is {clip.fps} Hz but rl-rate is {args.rl_rate}")
    plant = Plant(args.scene, tracker.dof_names, args.rl_rate)

    # Start at the clip's first frame, then run the policy against a frozen reference so it
    # takes over balance before the clip advances.
    plant.reset_to(clip.root_pos[0], clip.root_quat[0], clip.q[0])
    frozen = clip.window(0)
    for _ in range(int(args.freeze * args.rl_rate)):
        quat, ang_vel_b, q, dq = plant.state
        plant.hold(tracker.act(quat, ang_vel_b, q, dq, frozen), tracker.kp, tracker.kd)
    print(
        f"frozen hold {args.freeze:g}s: pelvis {plant.pelvis_height:.3f} m "
        f"(started {clip.root_pos[0][2]:.3f}); {clip.frames} frames at {clip.fps:g} Hz"
    )

    completed = 0
    for frame in range(clip.frames):
        quat, ang_vel_b, q, dq = plant.state
        target = tracker.act(quat, ang_vel_b, q, dq, clip.window(frame))
        plant.hold(target, tracker.kp, tracker.kd)
        completed = frame + 1
        sag = clip.root_pos[frame][2] - plant.pelvis_height
        if plant.upright < args.fall_upright or sag > args.fall_sag:
            print(
                f"FELL at frame {frame + 1}/{clip.frames}: upright {plant.upright:+.3f}, "
                f"pelvis {plant.pelvis_height:.3f} m vs reference {clip.root_pos[frame][2]:.3f}"
            )
            break

    pct = 100.0 * completed / clip.frames
    print(
        f"completed {completed}/{clip.frames} frames ({pct:.1f}%), "
        f"pelvis {plant.pelvis_height:.3f} m, upright {plant.upright:+.3f}"
    )
    return 0 if completed == clip.frames else 1


if __name__ == "__main__":
    raise SystemExit(main())

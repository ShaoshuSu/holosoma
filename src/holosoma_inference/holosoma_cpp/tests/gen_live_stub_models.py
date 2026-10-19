#!/usr/bin/env python3
"""Emit small metadata-complete holosoma WBT ONNX models for C++ tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

DOF_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
ACTION_DIM = len(DOF_NAMES)

STANDARD_TERMS = {
    "obs0_motion_command": {"func": "holosoma:wbt.motion_command"},
    "obs1_motion_ref_ori_b": {"func": "holosoma:wbt.motion_ref_ori_b"},
    "obs2_projected_gravity": {"func": "holosoma:wbt.projected_gravity"},
    "obs3_base_ang_vel": {"func": "holosoma:wbt.base_ang_vel"},
    "obs4_dof_pos": {"func": "holosoma:wbt.dof_pos"},
    "obs5_dof_vel": {"func": "holosoma:wbt.dof_vel"},
    "obs6_actions": {"func": "holosoma:wbt.actions"},
}
LOOKAHEAD_TERM = {"func": "exporter:motion_command_lookahead_sonic_aligned"}
STANDARD_FRAME_DIM = 2 * ACTION_DIM + 6 + 3 + 3 + ACTION_DIM * 3
LOOKAHEAD_DIM = 10 * 2 * ACTION_DIM


def _urdf() -> str:
    links = ['<link name="pelvis"/>']
    joints = []
    parent = "pelvis"
    for index, name in enumerate(DOF_NAMES):
        child = f"link_{index}"
        links.append(f'<link name="{child}"/>')
        joints.append(
            f'<joint name="{name}" type="revolute">'
            f'<parent link="{parent}"/><child link="{child}"/>'
            '<axis xyz="0 0 1"/>'
            '<limit lower="-3" upper="3" effort="1" velocity="1"/>'
            "</joint>"
        )
        parent = child
    return '<?xml version="1.0"?><robot name="wbt_stub">' + "".join(links + joints) + "</robot>"


def _experiment(body_name: str, history: int, lookahead_layout: str | None) -> tuple[dict, int]:
    actor_terms = dict(STANDARD_TERMS)
    groups: dict[str, dict] = {"actor_obs": {"terms": actor_terms, "history_length": history}}
    input_groups = ["actor_obs"]
    width = STANDARD_FRAME_DIM * history

    if lookahead_layout == "legacy":
        actor_terms["obs7_motion_command_lookahead"] = LOOKAHEAD_TERM
        width += LOOKAHEAD_DIM * history
    elif lookahead_layout == "split":
        groups["actor_targets"] = {"terms": {"obs7_motion_command_lookahead": LOOKAHEAD_TERM}, "history_length": 1}
        input_groups.append("actor_targets")
        width += LOOKAHEAD_DIM
    elif lookahead_layout in ("flat", "flat_0p2s"):
        short = lookahead_layout == "flat_0p2s"
        slots = 5 if short else 10
        actor_terms.clear()
        for i, leaf in enumerate(("projected_gravity", "base_ang_vel", "dof_pos", "dof_vel", "actions")):
            actor_terms[f"obs{i}_{leaf}"] = {"func": f"holosoma:wbt.{leaf}"}
        futures = (
            ("motion_joint_pos_future_0p2s", "motion_joint_vel_future_0p2s", "motion_future_ref_ori_b_0p2s")
            if short
            else (
                "motion_joint_pos_future_sonic_release",
                "motion_joint_vel_future_sonic_release",
                "motion_future_ref_ori_b_sonic_aligned",
            )
        )
        leaves = (
            *futures,
            "sonic_teleop_lower_body_future",
            "sonic_teleop_tracker_positions",
            "sonic_teleop_tracker_quaternions_wxyz",
            "sonic_teleop_anchor_rotation",
            "sonic_smpl_joints_local_future",
            "sonic_smpl_root_rotation_future",
            "sonic_smpl_wrist_joint_positions_future",
            "sonic_encoder_index",
        )
        groups["actor_targets"] = {
            "history_length": 1,
            "terms": {f"obs{i:02d}_target": {"func": f"exporter:{leaf}"} for i, leaf in enumerate(leaves, 5)},
        }
        input_groups.append("actor_targets")
        width = (6 + 3 * ACTION_DIM) * history + slots * (2 * ACTION_DIM + 6) + 1107 + 3

    defaults = {name: 0.01 * index for index, name in enumerate(DOF_NAMES)}
    experiment = {
        "robot": {"init_state": {"default_joint_angles": defaults}},
        "command": {"setup_terms": {"motion_command": {"params": {"motion_config": {"body_name_ref": [body_name]}}}}},
        "observation": {"clip_observations": 100.0, "groups": groups},
        "algo": {"config": {"module_dict": {"actor": {"input_dim": input_groups}}}},
    }
    return experiment, width


def _write_model(
    path: Path,
    *,
    body_name: str = "pelvis",
    history: int = 1,
    lookahead_layout: str | None = None,
    width_delta: int = 0,
    flat_error: str = "",
) -> None:
    experiment, width = _experiment(body_name, history, lookahead_layout)
    width += width_delta
    if flat_error:
        targets = experiment["observation"]["groups"]["actor_targets"]
        if flat_error == "missing":
            del targets["terms"]["obs07_target"]
            width -= 30
        elif flat_error == "mixed":
            targets["terms"]["obs06_target"]["func"] = "exporter:motion_joint_vel_future_sonic_release"
            width += 145
        elif flat_error == "history":
            targets["history_length"] = 2
            width += 1430
        else:
            raise ValueError(flat_error)

    obs = helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, width])
    time_step = helper.make_tensor_value_info("time_step", TensorProto.FLOAT, [1, 1])
    actions = helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, ACTION_DIM])
    indices = numpy_helper.from_array(np.arange(ACTION_DIM, dtype=np.int64), name="indices")
    gather = helper.make_node("Gather", ["obs", "indices"], ["actions"], axis=1)
    graph = helper.make_graph([gather], "wbt_stub", [obs, time_step], [actions], initializer=[indices])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9

    metadata = {
        "robot_urdf": json.dumps(_urdf()),
        "kp": json.dumps([40.0 + index for index in range(ACTION_DIM)]),
        "kd": json.dumps([4.0 + 0.1 * index for index in range(ACTION_DIM)]),
        "action_scale": json.dumps([1.0] * ACTION_DIM),
        "dof_names": json.dumps(DOF_NAMES),
        "experiment_config": json.dumps(experiment),
    }
    for key, value in metadata.items():
        item = model.metadata_props.add()
        item.key = key
        item.value = value

    onnx.checker.check_model(model)
    onnx.save(model, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    _write_model(output / "split_history.onnx", history=2, lookahead_layout="split")
    _write_model(output / "legacy_history.onnx", history=2, lookahead_layout="legacy")
    _write_model(output / "pelvis.onnx")
    _write_model(output / "torso.onnx", body_name="link_1")
    _write_model(output / "bad_shape.onnx", width_delta=1)
    _write_model(output / "sonic_flat.onnx", history=10, lookahead_layout="flat")
    _write_model(output / "sonic_flat_0p2s.onnx", history=9, lookahead_layout="flat_0p2s")
    for error in ("missing", "mixed", "history"):
        _write_model(output / f"flat_bad_{error}.onnx", history=9, lookahead_layout="flat_0p2s", flat_error=error)


if __name__ == "__main__":
    main()

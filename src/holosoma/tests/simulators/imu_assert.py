# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

"""Cross-backend live IMU assertion harness.

The in-sim oracles available without a harness (comparing the gyro and the orientation against
``robot_root_states``) only hold for a zero-offset identity mount, and they say nothing about the
accelerometer's magnitude under motion. This harness covers exactly that gap with closed-form
expectations:

  settled on the plane   |a| == g for every mount; gyro == 0
                         offset mount == identity mount EXACTLY (a lever arm is inert at rest)
                         rotated mount == identity rotated by the mount quaternion
                         orientation == root quat, and == root (x) mount for the rotated one
  in free fall           |a| == 0 for every mount, including the offset one

Known limitation — the settled scenario assumes an unactuated robot goes quiet, and how quiet is
both ``dt``- and engine-dependent. At ``--fps 1600`` on Isaac Sim the collapsed robot still carries
genuine residual contact acceleration (``|a|`` around 11.4), so the settled checks fail while
free-fall and the lever-arm checks still pass. That is the robot moving, not the sensor misreading.
Run the settled checks at the backend's default physics rate; ``--fps`` is for the convergence
sweep, where free-fall and the lever term are the meaningful signals. Loosening REST_ACCEL_TOL to
accommodate it would defeat the check, whose job is to catch gravity being wrong.

Exits 0 PASS, 1 FAIL, 77 SKIP.

  python imu_assert.py --simulator mujoco
  python imu_assert.py --simulator mjwarp   --num-envs 3
  python imu_assert.py --simulator isaacsim
"""

from __future__ import annotations

import argparse
import dataclasses
import sys

if sys.path and sys.path[0].endswith("simulators"):
    sys.path.pop(0)

from holosoma.utils.sim_utils import setup_simulation_environment  # noqa: E402
from tests.simulators._sim_harness import build_run_sim_config, run_and_hard_exit, step, steps_for_seconds  # noqa: E402

SKIP_EXIT_CODE = 77

GRAVITY = 9.81
# Settled-state tolerances. The robot is unactuated and collapses onto the plane, so it is never
# perfectly static: these allow residual contact jitter while staying far below the signal.
# m/s^2 on |a| vs g. Deliberately tight: this check's job is to catch gravity being wrong, so it
# must not be widened to accommodate a robot that has not actually settled (see the module note on
# --fps 1600).
REST_ACCEL_TOL = 0.30
REST_GYRO_TOL = 0.60  # rad/s
SAME_BODY_TOL = 0.05  # m/s^2 floor for numerical noise between two mounts on one body
ANGULAR_ACCEL_ALLOWANCE = 2.0  # rad/s^2, the unmeasured alpha in the rigid-body difference bound
QUAT_TOL = 2e-3
# Free fall: unactuated joints still exchange internal forces, so |a| is small but not zero. The
# assertion that matters is that it is a small FRACTION of g, i.e. gravity has been removed.
FREEFALL_ACCEL_FRACTION = 0.20
# Commanded body-z rate for the spin scenario. Fast enough that w^2*r (= 4 m/s^2 at r=0.25) is well
# clear of numerical noise, slow enough that one step does not deform the articulation much.
SPIN_RATE = 4.0
SPIN_DELTA_MIN_FRACTION = 0.5  # below this, the offset is being ignored
SPIN_DELTA_MAX_FRACTION = 2.5  # above this, it is not the centripetal term (alpha x r also lands here)
# Steps to run after a teleport before reading. ``set_actor_states`` changes velocity instantly, and
# a differencing IMU (IsaacLab's) divides that discontinuity by dt — a 1 m/s jump over a 5 ms step
# reads as 200 m/s^2. Stepping first flushes the velocity history so both samples come from the same
# regime. MuJoCo is analytic and indifferent to this, so the wait is what makes the harness
# backend-neutral rather than an Isaac Sim workaround.
SETTLE_STEPS_AFTER_TELEPORT = 8

FIELDS = ("linear_acceleration", "angular_velocity", "orientation")


def _fail(failures: list[str], message: str) -> None:
    failures.append(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulator", required=True, choices=["mujoco", "mjwarp", "isaacsim"])
    parser.add_argument("--robot", default="g1-29dof")
    parser.add_argument("--terrain", default="terrain_locomotion_plane")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--result-file", default=None, help="write OK/FAIL here before teardown")
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="override the physics rate; free-fall |a| should shrink toward 0 on a differencing "
        "backend as dt does, and stay flat on an analytic one",
    )
    args = parser.parse_args()

    from tests.simulators import _imu_presets

    sim_arg = "mujoco" if args.simulator == "mjwarp" else args.simulator
    config = build_run_sim_config(sim_arg, "empty", args.robot, args.terrain, sensors=_imu_presets.IMU_RIG)
    if args.simulator == "mjwarp":
        from tests.simulators import _camera_presets

        config = _camera_presets.as_mjwarp(config)

    device = "cpu" if args.simulator == "mujoco" else "cuda:0"
    n = args.num_envs
    config = dataclasses.replace(
        config, device=device, training=dataclasses.replace(config.training, num_envs=n)
    )
    if args.fps is not None:
        config = dataclasses.replace(
            config,
            simulator=dataclasses.replace(
                config.simulator,
                config=dataclasses.replace(
                    config.simulator.config,
                    sim=dataclasses.replace(config.simulator.config.sim, fps=args.fps),
                ),
            ),
        )

    env, device, _app = setup_simulation_environment(config, device=device)
    sim = env.sim
    sim.set_headless(True)
    sim.setup()
    sim.setup_terrain()
    sim.load_assets()

    import torch

    from holosoma.utils.rotations import quat_mul, quat_rotate_inverse

    env_origins = torch.zeros(n, 3, device=device)
    if n > 1:
        env_origins[:, 0] = torch.arange(n, device=device, dtype=torch.float32) * 5.0
    init = config.robot.init_state
    base_init = torch.tensor(
        list(init.pos) + list(init.rot) + list(init.lin_vel) + list(init.ang_vel), device=device, dtype=torch.float32
    )
    sim.create_envs(n, env_origins, base_init)
    sim.prepare_sim()
    sim.install_plugins()

    failures: list[str] = []
    created = sim.get_sensor_names()
    missing = sorted(set(_imu_presets.IMU_RIG) - set(created))
    if missing:
        print(f"[{args.simulator}] FAIL: IMUs not created: {missing} (created {sorted(created)})")
        return 1

    def read(name: str, data_type: str) -> torch.Tensor:
        return sim.get_imu_data(name, data_type)

    # ----- settled on the plane -----------------------------------------------------------------
    step(sim, steps_for_seconds(sim, 2.0))
    sim.render_sensors()

    rest = {key: {field: read(key, field).clone() for field in FIELDS} for key in _imu_presets.IMU_RIG}

    for key, data in rest.items():
        magnitude = data["linear_acceleration"].norm(dim=-1)
        if (magnitude - GRAVITY).abs().max() > REST_ACCEL_TOL:
            _fail(failures, f"{key}: settled |a|={magnitude.tolist()} is not {GRAVITY} +/- {REST_ACCEL_TOL}")
        if data["angular_velocity"].abs().max() > REST_GYRO_TOL:
            _fail(failures, f"{key}: settled gyro={data['angular_velocity'].tolist()} is not ~0")

    # A lever arm is inert at rest, so the two mounts must agree to within what the residual motion
    # can explain: |a_offset - a_identity| <= (w^2 + |alpha|) * r for a rigid body. Only w is
    # measured, so the bound uses it plus an allowance for the unmeasured angular acceleration.
    # NOTE this is a *corruption* check, not proof the offset is applied — an implementation that
    # dropped the offset would also pass it. The spin scenario below is what proves application.
    residual_omega = rest["identity"]["angular_velocity"].norm(dim=-1).max()
    offset_bound = SAME_BODY_TOL + (residual_omega**2 + ANGULAR_ACCEL_ALLOWANCE) * _imu_presets.LEVER_ARM
    offset_delta = (rest["offset"]["linear_acceleration"] - rest["identity"]["linear_acceleration"]).abs().max()
    if offset_delta > offset_bound:
        _fail(
            failures,
            f"offset vs identity at rest differ by {offset_delta:.4f} > {offset_bound:.4f} m/s^2, "
            f"more than residual motion (|w|={residual_omega:.3f} rad/s) can explain",
        )

    # The rotated mount must read the identity mount's vector rotated into its own frame.
    mount_quat_wxyz = torch.tensor(_imu_presets.MOUNT_ROTATION_WXYZ, device=device, dtype=torch.float32)
    mount_quat = mount_quat_wxyz[[1, 2, 3, 0]].expand(n, 4)
    expected_rotated = quat_rotate_inverse(mount_quat, rest["identity"]["linear_acceleration"], w_last=True)
    rotated_delta = (rest["rotated"]["linear_acceleration"] - expected_rotated).abs().max()
    if rotated_delta > SAME_BODY_TOL:
        _fail(
            failures,
            f"rotated mount a={rest['rotated']['linear_acceleration'].tolist()} != identity rotated by the "
            f"mount quat {expected_rotated.tolist()} (max delta {rotated_delta:.4f})",
        )

    # Orientation: identity mount == root attitude; rotated mount == root (x) mount.
    root_quat = sim.robot_root_states[:, 3:7]
    identity_quat_delta = (rest["identity"]["orientation"] - root_quat).abs().max()
    if identity_quat_delta > QUAT_TOL:
        _fail(failures, f"identity orientation differs from root quat by {identity_quat_delta:.5f}")
    expected_quat = quat_mul(root_quat, mount_quat, w_last=True)
    rotated_quat_delta = (rest["rotated"]["orientation"] - expected_quat).abs().max()
    if rotated_quat_delta > QUAT_TOL:
        _fail(failures, f"rotated orientation != root (x) mount, max delta {rotated_quat_delta:.5f}")

    # ----- free fall ----------------------------------------------------------------------------
    all_ids = torch.arange(n, device=device)
    states = sim.get_actor_states(["robot"], all_ids).clone()
    states[:, :3] = env_origins + torch.tensor(list(init.pos), device=device)
    states[:, 2] += 6.0
    states[:, 3:7] = torch.tensor(list(init.rot), device=device)
    states[:, 7:] = 0.0
    sim.set_actor_states(["robot"], all_ids, states)
    sim.write_state_updates()
    step(sim, SETTLE_STEPS_AFTER_TELEPORT)
    sim.render_sensors()  # discard: this sample may span the state-write velocity discontinuity
    step(sim, SETTLE_STEPS_AFTER_TELEPORT)
    sim.render_sensors()

    threshold = FREEFALL_ACCEL_FRACTION * GRAVITY
    freefall = {key: read(key, "linear_acceleration").norm(dim=-1).max().item() for key in _imu_presets.IMU_RIG}
    for key in _imu_presets.IMU_RIG:
        magnitude = read(key, "linear_acceleration").norm(dim=-1)
        if magnitude.max() > threshold:
            _fail(failures, f"{key}: free-fall |a|={magnitude.tolist()} exceeds {threshold:.2f} (should be ~0)")

    # ----- free fall + spin: the only check that proves the mount offset is applied --------------
    # Airborne with a known body-z rate, the offset mount measures the centripetal term w^2*r that
    # the zero-offset mount cannot see. Gravity and contact are both absent, so this is the clean
    # signal. The expectation uses the MEASURED w (the unactuated articulation starts deforming
    # immediately, so the commanded rate is not what the pelvis actually has a step later).
    states = sim.get_actor_states(["robot"], all_ids).clone()
    states[:, :3] = env_origins + torch.tensor(list(init.pos), device=device)
    states[:, 2] += 6.0
    states[:, 3:7] = torch.tensor(list(init.rot), device=device)
    states[:, 7:10] = 0.0
    states[:, 10:13] = torch.tensor([0.0, 0.0, SPIN_RATE], device=device)
    sim.set_actor_states(["robot"], all_ids, states)
    sim.write_state_updates()
    step(sim, SETTLE_STEPS_AFTER_TELEPORT)
    sim.render_sensors()  # discard: this sample may span the state-write velocity discontinuity
    step(sim, SETTLE_STEPS_AFTER_TELEPORT)
    sim.render_sensors()

    spin_omega = read("identity", "angular_velocity").norm(dim=-1)
    centripetal = spin_omega**2 * _imu_presets.LEVER_ARM
    spin_delta = (read("offset", "linear_acceleration") - read("identity", "linear_acceleration")).norm(dim=-1)
    if (spin_omega < 0.5 * SPIN_RATE).any():
        _fail(failures, f"spin did not take: measured |w|={spin_omega.tolist()} vs commanded {SPIN_RATE}")
    elif (spin_delta < SPIN_DELTA_MIN_FRACTION * centripetal).any():
        _fail(
            failures,
            f"offset mount shows no centripetal term: |a_offset - a_identity|={spin_delta.tolist()} "
            f"but w^2*r={centripetal.tolist()} — the mount offset looks ignored",
        )
    elif (spin_delta > SPIN_DELTA_MAX_FRACTION * centripetal).any():
        _fail(
            failures,
            f"offset mount centripetal term={spin_delta.tolist()} is far from w^2*r={centripetal.tolist()}",
        )

    label = f"[{args.simulator}/imu]"
    if failures:
        for message in failures:
            print(f"{label} FAIL: {message}")
    else:
        settled = [round(value, 3) for value in rest["identity"]["linear_acceleration"][0].tolist()]
        print(
            f"{label} CONVERGENCE fps={config.simulator.config.sim.fps} dt={sim.sim_dt * 1e3:.3f}ms "
            f"freefall|a|={freefall['identity']:.4f} (true 0) "
            f"lever={spin_delta[0]:.4f} vs w^2r={centripetal[0]:.4f} "
            f"|w|={spin_omega[0]:.4f} (cmd {SPIN_RATE})\n"
            f"{label} PASS: settled identity a={settled} (|a|={rest['identity']['linear_acceleration'][0].norm():.3f}); "
            f"spin |w|={spin_omega[0]:.3f} rad/s -> measured lever term {spin_delta[0]:.3f} m/s^2 "
            f"vs w^2*r={centripetal[0]:.3f}"
        )
    if args.result_file:
        with open(args.result_file, "w") as result:
            result.write("FAIL" if failures else "OK")
    return 1 if failures else 0


if __name__ == "__main__":
    run_and_hard_exit(main)

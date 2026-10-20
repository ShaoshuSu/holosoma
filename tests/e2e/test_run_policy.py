"""Sanity check tests for holosoma_inference/holosoma_inference/run_policy.py commands.

The goal is to test the run_policy commands as closely as possible to how the user would run them,
ensuring they start up correctly and emit inference output within 15 seconds.
"""

from __future__ import annotations

import os
import re
import shlex
import signal
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent.absolute()
_SETUP_TIMEOUT = 300
_STOP_TIMEOUT = 5


def run_mini_training(workflow_name: str) -> str:
    """Run a mini training and return the checkpoint path."""
    env = os.environ.copy()
    env["EXPORT_ONNX"] = "1"

    result = subprocess.run(
        [
            "python",
            f"{REPO_ROOT}/src/holosoma/holosoma/train_agent.py",
            f"exp:{workflow_name}",
            "--algo.config.num-learning-iterations=2",
            "--training.num-envs=4",
            "--logger.video.enabled=False",
        ],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert result.returncode == 0, f"Training failed: {result.stderr}"

    # Parse ONNX path from stdout/stderr
    output = result.stdout + result.stderr
    print(output)

    pt_matches = re.findall(r"Saving checkpoint to (.+/model_\d+\.pt)", output)
    if pt_matches:
        # Return the latest checkpoint (last one saved)
        return pt_matches[-1].replace(".pt", ".onnx")

    raise FileNotFoundError("No ONNX checkpoint path found in training output")


def _signal_process_group(process: subprocess.Popen, sig: int) -> None:
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


def _run_inference_command(command: str, timeout: float) -> tuple[int, str, bool]:
    """Collect output while bounding both execution and whole-group cleanup."""
    process = subprocess.Popen(
        ["/bin/bash", "-c", command],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        start_new_session=True,
    )
    timed_out = False
    try:
        try:
            process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            # Killing bash alone leaves Python (or pip) holding the output pipe.
            _signal_process_group(process, signal.SIGTERM)
            try:
                output, _ = process.communicate(timeout=_STOP_TIMEOUT)
            except subprocess.TimeoutExpired:
                _signal_process_group(process, signal.SIGKILL)
                output, _ = process.communicate(timeout=_STOP_TIMEOUT)
        return process.returncode, output, timed_out
    finally:
        # Also stop descendants that closed their pipes or outlived the leader.
        _signal_process_group(process, signal.SIGKILL)


def assert_run_policy_with_hsinference(config_name: str, model_path: str, timeout: float = 15) -> None:
    """Run policy in hsinference environment with local checkpoint.

    Parameters
    ----------
    config_name : str
        Tyro subcommand name (e.g., 'g1-29dof-loco', 't1-29dof-loco')
    model_path : str
        Path to the ONNX model checkpoint
    timeout : float
        Policy observation window in seconds, excluding dependency installation
    """
    activate = f"source {shlex.quote(str(REPO_ROOT / 'scripts/source_inference_setup.sh'))} && "
    setup_code, setup_output, setup_timed_out = _run_inference_command(
        activate + f"pip install -e {shlex.quote(str(REPO_ROOT / 'src/holosoma_inference') + '[unitree,booster]')}",
        timeout=_SETUP_TIMEOUT,
    )
    assert not setup_timed_out, f"Setup timed out after {_SETUP_TIMEOUT}s. Output: {setup_output}"
    assert setup_code == 0, f"Setup exited with {setup_code}. Output: {setup_output}"
    policy_code, output, timed_out = _run_inference_command(
        activate
        + shlex.join(
            [
                "python",
                str(REPO_ROOT / "src/holosoma_inference/holosoma_inference/run_policy.py"),
                f"inference:{config_name}",
                f"--task.model-path={model_path}",
                "--secondary",
                "none",
            ]
        ),
        timeout=timeout,
    )
    assert timed_out or policy_code == 0, f"Policy exited with {policy_code}. Output: {output}"
    assert "RL FPS:" in output, f"Didn't find RL FPS in output! Output: {output}"


@pytest.mark.requires_inference
@pytest.mark.parametrize(
    ("workflow_name", "config_name"),
    [
        ("g1-29dof", "g1-29dof-loco"),
        # ("t1-29dof", "t1-29dof-loco"),
    ],
)
def test_run_policy_with_trained_checkpoint(workflow_name: str, config_name: str):
    """Train a mini model and test running it in hsinference environment.

    Parameters
    ----------
    workflow_name : str
        Training workflow name (e.g., 'g1-29dof')
    config_name : str
        Tyro subcommand name for run_policy (e.g., 'g1-29dof-loco')
    """
    checkpoint_path = run_mini_training(workflow_name)
    assert_run_policy_with_hsinference(config_name, checkpoint_path)

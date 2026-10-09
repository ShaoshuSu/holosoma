"""Tests for ``ObsGroupCfg.interval`` / ``reset_history_on_env_reset`` (FAR-FALCON-Terrains parity)."""

from __future__ import annotations

import sys
import types

import pytest
import torch

from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg
from holosoma.managers.observation.manager import ObservationManager

_TERMS_MODULE = "_holosoma_test_obs_interval_terms"


class _Env:
    def __init__(self, num_envs: int, dim: int = 2):
        self.num_envs = num_envs
        self.value = torch.zeros(num_envs, dim)


@pytest.fixture(autouse=True)
def _terms_module():
    module = types.ModuleType(_TERMS_MODULE)
    module.value = lambda env: env.value  # type: ignore[attr-defined]
    module.value_x10 = lambda env: env.value * 10.0  # type: ignore[attr-defined]
    sys.modules[_TERMS_MODULE] = module
    yield
    del sys.modules[_TERMS_MODULE]


def _manager(env: _Env, **group_kwargs) -> ObservationManager:
    group = ObsGroupCfg(
        terms={
            "a": ObsTermCfg(func=f"{_TERMS_MODULE}:value"),
            "b": ObsTermCfg(func=f"{_TERMS_MODULE}:value_x10", scale=0.5),
        },
        **{"history_time_major": True, **group_kwargs},
    )
    return ObservationManager(ObservationManagerCfg(groups={"g": group}), env, device="cpu")


class _FalconReference:
    """Line-for-line port of FALCON ``_post_config_observation_callback`` + reset bookkeeping."""

    def __init__(self, num_envs: int, frame_dim: int, history_length: int, interval: int):
        self.frame_dim = frame_dim
        self.history_length = history_length
        self.interval = interval
        self.buf = torch.zeros(num_envs, frame_dim * history_length)
        self.counter = torch.zeros(num_envs, dtype=torch.int)

    def reset(self, env_ids: torch.Tensor) -> None:
        self.counter[env_ids] = 0  # FALCON leaves obs_buf_dict untouched on reset

    def step(self, current: torch.Tensor) -> torch.Tensor:
        should_update = (self.counter % self.interval) == 0
        if should_update.any():
            self.buf[should_update] = torch.cat(
                (
                    self.buf[should_update][:, self.frame_dim : self.frame_dim * self.history_length],
                    current[should_update],
                ),
                dim=-1,
            )
        self.counter[:] += 1
        return self.buf.clone()


def test_default_group_keeps_deque_path():
    env = _Env(num_envs=2)
    mgr = _manager(env, history_length=3)
    assert mgr._interval_counters == {}
    assert set(mgr._history_buffers["g"]) == {"a", "b"}


@pytest.mark.parametrize(("history_length", "interval"), [(2, 5), (3, 2), (1, 4), (4, 1)])
def test_matches_falcon_reference_with_resets(history_length: int, interval: int):
    num_envs, dim = 4, 2
    env = _Env(num_envs, dim)
    mgr = _manager(env, history_length=history_length, interval=interval, reset_history_on_env_reset=False)
    ref = _FalconReference(num_envs, frame_dim=2 * dim, history_length=history_length, interval=interval)

    gen = torch.Generator().manual_seed(0)
    for step in range(40):
        # base_task order: reset done envs, then compute observations.
        reset_ids = torch.nonzero(torch.rand(num_envs, generator=gen) < 0.15).flatten()
        if step == 0:
            reset_ids = torch.arange(num_envs)
        if reset_ids.numel() > 0:
            mgr.reset(reset_ids)
            ref.reset(reset_ids)
        env.value = torch.randn(num_envs, dim, generator=gen)
        # Sorted terms: a = value, b = value * 10 * 0.5.
        current = torch.cat([env.value, env.value * 5.0], dim=-1)
        expected = ref.step(current)
        got = mgr.compute()["g"]
        torch.testing.assert_close(got, expected, msg=f"step {step}")


def test_interval_holds_stale_frames_and_counter_resets():
    env = _Env(num_envs=2, dim=1)
    mgr = _manager(env, history_length=2, interval=5, reset_history_on_env_reset=False)

    # Layout per env: [a_old, b_old, a_new, b_new]; record term ``a`` for both frames.
    a_frames = []
    for step in range(7):
        if step == 3:
            mgr.reset(torch.tensor([1]))
        env.value = torch.full((2, 1), float(step + 1))
        out = mgr.compute()["g"]
        a_frames.append(out[:, [0, 2]].tolist())
    # env 0 pushes at steps 0 and 5 only.
    assert [f[0] for f in a_frames] == [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [1, 6], [1, 6]]
    # env 1 is reset before step 3: its counter restarts so it pushes at 0 and 3, and the
    # previous episode's frame (1) is kept rather than zeroed.
    assert [f[1] for f in a_frames] == [[0, 1], [0, 1], [0, 1], [1, 4], [1, 4], [1, 4], [1, 4]]


def test_modify_history_false_does_not_mutate():
    env = _Env(num_envs=3)
    mgr = _manager(env, history_length=2, interval=3, reset_history_on_env_reset=False)
    env.value = torch.ones(3, 2)
    first = mgr.compute()["g"]
    counter = mgr._interval_counters["g"].clone()
    frames = {k: v.clone() for k, v in mgr._frame_buffers["g"].items()}

    env.value = torch.full((3, 2), 7.0)
    peek = mgr.compute(modify_history=False)["g"]
    torch.testing.assert_close(peek, first)  # counter == 1 -> no push, stale value
    torch.testing.assert_close(mgr._interval_counters["g"], counter)
    for k, v in mgr._frame_buffers["g"].items():
        torch.testing.assert_close(v, frames[k])


def test_reset_history_on_env_reset_zeroes_reset_envs():
    env = _Env(num_envs=2)
    mgr = _manager(env, history_length=3, interval=2, reset_history_on_env_reset=True)
    env.value = torch.ones(2, 2)
    mgr.compute()
    mgr.reset(torch.tensor([0]))
    frames = mgr._frame_buffers["g"]["a"]
    assert torch.all(frames[0] == 0)
    assert torch.all(frames[1, -1] == 1)
    assert mgr._interval_counters["g"].tolist() == [0, 1]


def test_get_obs_dims_unchanged_by_interval():
    env = _Env(num_envs=2)
    assert _manager(env, history_length=2, interval=5).get_obs_dims() == {"g": 8}
    assert _manager(env, history_length=1, interval=5).get_obs_dims() == {"g": 4}

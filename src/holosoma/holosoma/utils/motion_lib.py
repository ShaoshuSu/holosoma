"""Minimal RPL-style motion library for loco-manip converted motions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from holosoma.config_types.motion import MotionLibConfig
from holosoma.utils.path import resolve_data_file_path
from holosoma.utils.safe_torch_import import torch


class MotionLibRobot:
    """Loads a file or directory of converted `*_mj*.npz` motions with RPL-like APIs."""

    def __init__(self, motion_lib_cfg: MotionLibConfig, num_envs: int, device: str):
        self.m_cfg = motion_lib_cfg
        self.num_envs = num_envs
        self._device = device
        self._sim_fps = 1.0 / float(motion_lib_cfg.step_dt)

        self._motion_data_paths = self._collect_motion_paths(motion_lib_cfg.motion_file)
        self._num_unique_motions = len(self._motion_data_paths)
        self._motion_fps: torch.Tensor | None = None
        self._motion_dt: torch.Tensor | None = None
        self._motion_lengths: torch.Tensor | None = None
        self._motion_num_frames: torch.Tensor | None = None
        self.length_starts: torch.Tensor | None = None
        self.joint_pos: torch.Tensor | None = None
        self.joint_vel: torch.Tensor | None = None
        self.body_pos_w: torch.Tensor | None = None
        self.body_quat_w: torch.Tensor | None = None
        self.body_lin_vel_w: torch.Tensor | None = None
        self.body_ang_vel_w: torch.Tensor | None = None
        self.motion_ids: torch.Tensor | None = None
        self._curr_motion_ids: torch.Tensor | None = None

    def _collect_motion_paths(self, motion_file: str) -> list[Path]:
        resolved = Path(resolve_data_file_path(motion_file))
        if resolved.is_dir():
            paths = sorted(resolved.glob("*.npz"))
        elif resolved.is_file():
            paths = [resolved]
        else:
            raise FileNotFoundError(f"Motion path does not exist: {motion_file}")

        if not paths:
            raise FileNotFoundError(f"No motion files found under: {resolved}")
        return paths

    def load_motions(self, random_sample: bool = True, start_idx: int = 0, **_) -> list[str]:
        selected_ids = self._select_motion_ids(random_sample=random_sample, start_idx=start_idx)

        joint_pos = []
        joint_vel = []
        body_pos_w = []
        body_quat_w = []
        body_lin_vel_w = []
        body_ang_vel_w = []
        motion_fps = []
        motion_dt = []
        motion_lengths = []
        motion_num_frames = []

        ids_list = selected_ids.tolist()
        iter_ids = tqdm(
            ids_list,
            desc="Loading motions",
            unit="motion",
            disable=len(ids_list) <= 1,
        )
        for motion_id in iter_ids:
            motion_path = self._motion_data_paths[motion_id]
            with np.load(motion_path, allow_pickle=False) as data:
                curr_joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=self._device)
                curr_joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=self._device)
                curr_body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=self._device)
                curr_body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=self._device)
                curr_body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=self._device)
                curr_body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=self._device)
                fps = float(data["fps"])

            num_frames = int(curr_joint_pos.shape[0])
            dt = 1.0 / fps
            length = max((num_frames - 1) * dt, 0.0)
            if self.m_cfg.standardize_motion_length:
                length = float(self.m_cfg.standardize_motion_length_value)
                fps = max((num_frames - 1) / max(length, 1e-6), 1e-6)
                dt = 1.0 / fps

            joint_pos.append(curr_joint_pos)
            joint_vel.append(curr_joint_vel)
            body_pos_w.append(curr_body_pos_w)
            body_quat_w.append(curr_body_quat_w)
            body_lin_vel_w.append(curr_body_lin_vel_w)
            body_ang_vel_w.append(curr_body_ang_vel_w)
            motion_fps.append(fps)
            motion_dt.append(dt)
            motion_lengths.append(length)
            motion_num_frames.append(num_frames)

        self._curr_motion_ids = selected_ids
        self._motion_fps = torch.tensor(motion_fps, dtype=torch.float32, device=self._device)
        self._motion_dt = torch.tensor(motion_dt, dtype=torch.float32, device=self._device)
        self._motion_lengths = torch.tensor(motion_lengths, dtype=torch.float32, device=self._device)
        self._motion_num_frames = torch.tensor(motion_num_frames, dtype=torch.long, device=self._device)
        lengths_shifted = self._motion_num_frames.roll(1)
        lengths_shifted[0] = 0
        self.length_starts = lengths_shifted.cumsum(0)
        self.motion_ids = torch.arange(len(selected_ids), dtype=torch.long, device=self._device)
        self.joint_pos = torch.cat(joint_pos, dim=0)
        self.joint_vel = torch.cat(joint_vel, dim=0)
        self.body_pos_w = torch.cat(body_pos_w, dim=0)
        self.body_quat_w = torch.cat(body_quat_w, dim=0)
        self.body_lin_vel_w = torch.cat(body_lin_vel_w, dim=0)
        self.body_ang_vel_w = torch.cat(body_ang_vel_w, dim=0)
        return [str(self._motion_data_paths[mid]) for mid in ids_list]

    def _select_motion_ids(self, random_sample: bool, start_idx: int) -> torch.Tensor:
        if self._num_unique_motions == 1:
            return torch.zeros(self.num_envs, dtype=torch.long, device=self._device)
        if random_sample:
            return torch.randint(0, self._num_unique_motions, (self.num_envs,), device=self._device, dtype=torch.long)
        return torch.remainder(
            torch.arange(start_idx, start_idx + self.num_envs, device=self._device, dtype=torch.long),
            self._num_unique_motions,
        )

    def sample_time(self, motion_ids: torch.Tensor, truncate_time: float | None = None) -> torch.Tensor:
        assert self._motion_lengths is not None
        phase = torch.rand(motion_ids.shape, device=self._device)
        motion_len = self._motion_lengths[motion_ids]
        if truncate_time is not None:
            motion_len = torch.clamp(motion_len - truncate_time, min=0.0)
        return phase * motion_len

    def get_motion_length(self, motion_ids: torch.Tensor | None = None) -> torch.Tensor:
        assert self._motion_lengths is not None
        if motion_ids is None:
            return self._motion_lengths
        return self._motion_lengths[motion_ids]

    def get_motion_state(self, motion_ids: torch.Tensor, motion_times: torch.Tensor, offset=None) -> dict[str, torch.Tensor]:
        assert self._motion_lengths is not None
        assert self._motion_num_frames is not None
        assert self._motion_dt is not None
        assert self.length_starts is not None
        assert self.joint_pos is not None
        assert self.joint_vel is not None
        assert self.body_pos_w is not None
        assert self.body_quat_w is not None
        assert self.body_lin_vel_w is not None
        assert self.body_ang_vel_w is not None

        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]
        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)
        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]
        blend_pose = blend.unsqueeze(-1)
        blend_body = blend.unsqueeze(-1).unsqueeze(-1)

        joint_pos = (1.0 - blend_pose) * self.joint_pos[f0l] + blend_pose * self.joint_pos[f1l]
        joint_vel = (1.0 - blend_pose) * self.joint_vel[f0l] + blend_pose * self.joint_vel[f1l]
        body_pos_w = (1.0 - blend_body) * self.body_pos_w[f0l] + blend_body * self.body_pos_w[f1l]
        body_quat_w = (1.0 - blend_body) * self.body_quat_w[f0l] + blend_body * self.body_quat_w[f1l]
        body_lin_vel_w = (1.0 - blend_body) * self.body_lin_vel_w[f0l] + blend_body * self.body_lin_vel_w[f1l]
        body_ang_vel_w = (1.0 - blend_body) * self.body_ang_vel_w[f0l] + blend_body * self.body_ang_vel_w[f1l]
        if offset is not None:
            body_pos_w = body_pos_w + offset[:, None, :]

        return {
            "dof_pos": joint_pos[:, 7:],
            "dof_vel": joint_vel[:, 6:],
            "rg_pos_t": body_pos_w,
            "rg_rot_t": body_quat_w,
            "body_vel_t": body_lin_vel_w,
            "body_ang_vel_t": body_ang_vel_w,
        }

    def _calc_frame_blend(
        self, time: torch.Tensor, length: torch.Tensor, num_frames: torch.Tensor, dt: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        safe_length = torch.clamp(length, min=1e-6)
        phase = torch.clamp(time / safe_length, 0.0, 1.0)
        frame_idx0 = (phase * (num_frames - 1)).long()
        frame_idx1 = torch.minimum(frame_idx0 + 1, num_frames - 1)
        safe_dt = torch.clamp(dt, min=1e-6)
        blend = torch.clamp((time - frame_idx0 * safe_dt) / safe_dt, 0.0, 1.0)
        return frame_idx0, frame_idx1, blend

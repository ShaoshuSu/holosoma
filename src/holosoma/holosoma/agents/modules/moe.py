"""Mixture-of-Experts actor routed by terrain index, used as DAgger teacher."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch
from torch import nn

from holosoma.agents.modules.student_teacher_modules import STActor, STActorEncoder


class MoETerrainMLPActor(nn.Module):
    """Terrain-indexed mixture of experts.

    The expert selected per sample is determined by ``route_obs`` (a terrain
    index tensor of shape ``[B, 1]``).  Only experts for terrain types with
    non-zero proportion are instantiated.  Each expert is an ordinary
    :class:`STActor` built from ``module_config_dict``; the MoE wrapper only
    routes.
    """

    is_recurrent = False

    def __init__(
        self,
        obs_dim_dict: dict[str, int],
        module_config_dict: Any,
        num_actions: int,
        init_noise_std: float,
        history_length: dict[str, int],
    ):
        super().__init__()
        self.num_actions = num_actions
        self._init_terrain_config(module_config_dict.layer_config)
        # Strip the "MoE" prefix so each expert is built via the ordinary non-MoE path.
        expert_cfg = replace(module_config_dict, type=module_config_dict.type[len("MoE") :])
        self._init_experts(obs_dim_dict, expert_cfg, num_actions, init_noise_std, history_length)

    def _init_terrain_config(self, layer_config):
        proportions = list(layer_config.terrain_proportions)
        nonzero_indices = [i for i, p in enumerate(proportions) if p > 0]
        self.num_experts = len(nonzero_indices)
        # Stored on the module so .to(device) moves it alongside the experts.
        self.register_buffer(
            "expert_ids", torch.tensor(nonzero_indices, dtype=torch.long), persistent=False
        )

    def _make_expert(self, obs_dim_dict, module_config_dict, num_actions, init_noise_std, history_length):
        return STActor(
            obs_dim_dict=obs_dim_dict,
            module_config_dict=module_config_dict,
            num_actions=num_actions,
            init_noise_std=init_noise_std,
            history_length=history_length,
        )

    def _init_experts(self, obs_dim_dict, module_config_dict, num_actions, init_noise_std, history_length):
        self.experts = nn.ModuleList(
            [
                self._make_expert(obs_dim_dict, module_config_dict, num_actions, init_noise_std, history_length)
                for _ in range(self.num_experts)
            ]
        )

    def act(self, policy_state_dict):
        obs = policy_state_dict["actor_obs"]
        expert_ids = policy_state_dict["route_obs"]

        batch = obs.shape[0]
        actions = torch.zeros(batch, self.num_actions, device=obs.device, dtype=obs.dtype)
        ids_flat = expert_ids.view(-1).long()
        for expert_idx in range(self.num_experts):
            mask = ids_flat == int(self.expert_ids[expert_idx].item())
            if not mask.any():
                continue
            actions[mask] = self.experts[expert_idx].act({"actor_obs": obs[mask]})
        return actions

    def act_inference(self, policy_state_dict):
        return self.act(policy_state_dict)

    def reset(self, dones=None):
        pass

    def get_encoder_output(self) -> torch.Tensor | None:
        return None


class MoETerrainEncoderActor(MoETerrainMLPActor):
    """Terrain-indexed mixture of encoder experts (MLPEncoder / CNNEncoder).

    Each expert wraps an encoder (MLP or CNN) feeding an MLP head; the MoE
    layer routes ``actor_obs`` to the selected expert *and* exposes a gathered
    encoder output for optional encoder-distillation losses.
    """

    def _make_expert(self, obs_dim_dict, module_config_dict, num_actions, init_noise_std, history_length):
        return STActorEncoder(
            obs_dim_dict=obs_dim_dict,
            module_config_dict=module_config_dict,
            num_actions=num_actions,
            init_noise_std=init_noise_std,
            history_length=history_length,
        )

    def _init_experts(self, obs_dim_dict, module_config_dict, num_actions, init_noise_std, history_length):
        super()._init_experts(obs_dim_dict, module_config_dict, num_actions, init_noise_std, history_length)
        # All experts share the same encoder output width (same module_config).
        self.encoder_output_dim = self.experts[0].actor_module.encoder_output_dim
        self._encoder_outputs: torch.Tensor | None = None

    def act(self, policy_state_dict):
        obs = policy_state_dict["actor_obs"]
        expert_ids = policy_state_dict["route_obs"]

        batch = obs.shape[0]
        actions = torch.zeros(batch, self.num_actions, device=obs.device, dtype=obs.dtype)
        enc_out = torch.zeros(batch, self.encoder_output_dim, device=obs.device, dtype=obs.dtype)
        ids_flat = expert_ids.view(-1).long()
        for expert_idx in range(self.num_experts):
            mask = ids_flat == int(self.expert_ids[expert_idx].item())
            if not mask.any():
                continue
            expert = self.experts[expert_idx]
            actions[mask] = expert.act({"actor_obs": obs[mask]})
            enc_out[mask] = expert.get_encoder_output()
        self._encoder_outputs = enc_out
        return actions

    def get_encoder_output(self) -> torch.Tensor | None:
        return self._encoder_outputs

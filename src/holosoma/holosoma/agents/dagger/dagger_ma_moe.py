"""Decoupled DAgger with a terrain-routed MoE teacher."""

from __future__ import annotations

from pathlib import Path

import torch
from loguru import logger
from torch import nn

from holosoma.agents.dagger.dagger_ma import DAggerMultiActor
from holosoma.config_types.algo import DAggerMAMoEConfig
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.utils.file_cache import get_cached_file_path


def mse_domain_fair(
    error: torch.Tensor,
    terrain_idx: torch.Tensor,
    equal_domain_avg: bool = True,
    expert_ids: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Domain-fair MSE loss, grouping samples by terrain index."""
    per_sample = (error**2).mean(dim=1)
    tid = terrain_idx.squeeze(-1).long()
    assert per_sample.shape[0] == tid.shape[0], "Batch size mismatch"

    if expert_ids is not None:
        expert_ids = expert_ids.long().to(tid.device)
        mask = torch.isin(tid, expert_ids)
        per_sample = per_sample[mask]
        tid = tid[mask]
        if per_sample.numel() == 0:
            return per_sample.new_tensor(0.0)

    domain_ids, inverse, counts = torch.unique(tid, sorted=True, return_inverse=True, return_counts=True)
    k = domain_ids.numel()
    sums = torch.zeros(k, device=per_sample.device, dtype=per_sample.dtype)
    sums.scatter_add_(0, inverse, per_sample)
    domain_means = sums / (counts.to(per_sample.dtype) + eps)

    if equal_domain_avg:
        return domain_means.mean()
    ratio = counts.to(per_sample.dtype) / (per_sample.numel() + eps)
    return (domain_means * (ratio**2)).sum()


def compute_per_terrain_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    terrain_idx: torch.Tensor,
    loss_type: str,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[int, float]]:
    """Compute per-terrain loss breakdown plus an overall scalar loss."""
    error = predictions - targets
    tid = terrain_idx.squeeze(-1).long()

    domain_ids, inverse, counts = torch.unique(tid, sorted=True, return_inverse=True, return_counts=True)
    terrain_losses: dict[int, float] = {}
    for terrain_id in domain_ids:
        mask = tid == terrain_id
        terr_pred = predictions[mask]
        terr_tgt = targets[mask]
        terr_err = error[mask]
        if loss_type == "l1_norm":
            tl = terr_err.abs().sum(dim=-1).mean()
        elif loss_type == "l2_norm":
            tl = torch.sqrt((terr_err**2).sum(dim=-1)).mean()
        elif loss_type in ("mse", "mse_domain"):
            tl = (terr_err**2).mean()
        elif loss_type == "mae":
            tl = terr_err.abs().mean()
        elif loss_type == "bce":
            tl = torch.nn.functional.binary_cross_entropy_with_logits(terr_pred, terr_tgt)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        terrain_losses[terrain_id.item()] = tl.item()

    if loss_type == "l1_norm":
        overall = error.abs().sum(dim=-1).mean()
    elif loss_type == "l2_norm":
        overall = torch.sqrt((error**2).sum(dim=-1)).mean()
    elif loss_type == "mse":
        overall = (error**2).mean()
    elif loss_type == "mae":
        overall = error.abs().mean()
    elif loss_type == "bce":
        overall = torch.nn.functional.binary_cross_entropy_with_logits(predictions, targets)
    elif loss_type == "mse_domain":
        per_sample = (error**2).mean(dim=1)
        k = domain_ids.numel()
        sums = torch.zeros(k, device=per_sample.device, dtype=per_sample.dtype)
        sums.scatter_add_(0, inverse, per_sample)
        domain_means = sums / (counts.to(per_sample.dtype) + eps)
        overall = domain_means.mean()
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")

    return overall, terrain_losses


class DAggerMultiActorMoE(DAggerMultiActor):
    """DAgger with a MoE teacher routed by terrain index."""

    config: DAggerMAMoEConfig

    def __init__(
        self,
        env: BaseTask,
        config: DAggerMAMoEConfig,
        log_dir: str,
        device: str = "cpu",
        multi_gpu_cfg: dict | None = None,
    ):
        if not isinstance(config, DAggerMAMoEConfig):
            raise TypeError(f"DAggerMultiActorMoE requires DAggerMAMoEConfig, got {type(config)}")
        super().__init__(env, config, log_dir, device, multi_gpu_cfg)

    # ------------------------------------------------------------------
    # Config / terrain
    # ------------------------------------------------------------------

    def _init_config(self) -> None:
        super()._init_config()
        self.route_obs_name: str = self.config.route_obs_name
        self._init_terrain_config()

    def _init_terrain_config(self) -> None:
        md = self.config.module_dict
        teacher_cfg = getattr(md, "actor_teacher_lower_body")
        layer_cfg = teacher_cfg.layer_config

        if not layer_cfg.terrain_types or not layer_cfg.terrain_proportions:
            terrain_config = self._read_env_terrain_config()
            if terrain_config:
                terrain_types = list(terrain_config.keys())
                terrain_proportions = [float(terrain_config[k]) for k in terrain_types]
                for ma_key in self.keys:
                    self._patch_layer_terrain(f"actor_teacher_{ma_key}", terrain_types, terrain_proportions)
                layer_cfg = teacher_cfg.layer_config

        proportions = list(layer_cfg.terrain_proportions)
        self.terrain_proportions = proportions
        self.terrain_types = list(layer_cfg.terrain_types)
        nonzero_indices = [i for i, p in enumerate(proportions) if p > 0]
        self.num_experts = len(nonzero_indices)
        self.expert_ids = torch.tensor(nonzero_indices, dtype=torch.long)

    def _read_env_terrain_config(self) -> dict[str, float] | None:
        terrain_manager = getattr(self.env, "terrain_manager", None)
        if terrain_manager is None:
            return None
        cfg = getattr(terrain_manager, "cfg", None)
        terrain_term = getattr(cfg, "terrain_term", None) if cfg is not None else None
        terrain_config = getattr(terrain_term, "terrain_config", None) if terrain_term is not None else None
        if not terrain_config:
            return None
        return dict(terrain_config)

    def _patch_layer_terrain(self, module_name: str, types: list[str], proportions: list[float]) -> None:
        module_cfg = getattr(self.config.module_dict, module_name, None)
        if module_cfg is None or not str(module_cfg.type).startswith("MoE"):
            return
        layer_cfg = module_cfg.layer_config
        # Frozen pydantic dataclass; mutate in place via object.__setattr__.
        object.__setattr__(layer_cfg, "terrain_types", list(types))
        object.__setattr__(layer_cfg, "terrain_proportions", list(proportions))

    # ------------------------------------------------------------------
    # Storage / rollout extensions
    # ------------------------------------------------------------------

    def _setup_storage(self) -> None:
        super()._setup_storage()
        for key in self.keys:
            self.storages[key].register("route_obs", shape=(1,), dtype=torch.float)

    def _setup_models_and_optimizer(self) -> None:
        super()._setup_models_and_optimizer()
        # Wandb-only teacher: parent's load_teacher gate (ckpt_teacher is not None)
        # is false, so handle it here and re-sync across ranks.
        if self.config.ckpt_teacher is None and any(self.config.ckpt_teacher_wandb.values()):
            for key in self.keys:
                self.load_teacher(None, key)
                self.actors_teacher[key].eval()
                if self.actors_encoder_teacher[key] is not None:
                    self.actors_encoder_teacher[key].eval()
            if self.is_multi_gpu:
                self._synchronize_model_weights()

    def _teacher_inputs(
        self,
        obs_dict: dict[str, torch.Tensor],
        key: str,
        actor_teacher_obs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {
            "actor_obs": actor_teacher_obs,
            "route_obs": obs_dict[self.route_obs_name],
        }

    def _multi_actor_rollout_step(self, obs_dict: dict[str, torch.Tensor]):
        actions_dict, per_key = super()._multi_actor_rollout_step(obs_dict)
        route_obs = obs_dict[self.route_obs_name]
        for key in self.keys:
            per_key[key]["route_obs"] = route_obs
        return actions_dict, per_key

    def _rollout_step(self, obs_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        with torch.inference_mode():
            for _ in range(self.config.num_steps_per_env):
                actions_dict, per_key_tensors = self._multi_actor_rollout_step(obs_dict)
                actions_full = self._concat_actions(actions_dict)
                obs_dict, rew_total, dones, infos = self.env.step({"actions": actions_full})

                for obs_key in obs_dict:
                    obs_dict[obs_key] = obs_dict[obs_key].to(self.device)
                dones = dones.to(self.device)
                dones_1 = dones.view(-1, 1)

                rewards_ma = infos.get("rewards_ma")

                for key in self.keys:
                    data = per_key_tensors[key]
                    if rewards_ma is not None and key in rewards_ma:
                        r = rewards_ma[key].to(self.device).view(-1, 1)
                    else:
                        r = rew_total.to(self.device).view(-1, 1)

                    if self.use_critic and infos.get("time_outs") is not None and infos["time_outs"].any():
                        timeouts = infos["time_outs"].to(self.device)
                        fo = infos.get("final_observations") or {}
                        final_critic_obs = torch.cat(
                            [fo[k] for k in self.critic_obs_keys[key]], dim=1
                        ).to(self.device)
                        fv = self.critics[key].evaluate({"critic_obs": final_critic_obs}).detach()
                        r = r + self.config.gamma * torch.squeeze(
                            fv * timeouts.unsqueeze(1).to(self.device), 1
                        ).unsqueeze(1)

                    add_kwargs = {
                        "actor_obs": data["actor_obs"],
                        "actor_teacher_obs": data["actor_teacher_obs"],
                        "critic_obs": data["critic_obs"],
                        "actions": data["actions"],
                        "actions_teacher": data["actions_teacher"],
                        "rewards": r,
                        "dones": dones_1,
                        "route_obs": data["route_obs"].view(-1, 1).float(),
                    }
                    if self.use_critic:
                        add_kwargs["values"] = data["values"]
                    if "encoder_output" in data:
                        add_kwargs["encoder_output"] = data["encoder_output"]
                        add_kwargs["encoder_output_teacher"] = data["encoder_output_teacher"]
                    if "actor_encoder_teacher_obs" in data:
                        add_kwargs["actor_encoder_teacher_obs"] = data["actor_encoder_teacher_obs"]
                    self.storages[key].add(**add_kwargs)

                    self.actors[key].reset(dones)
                    self.actors_teacher[key].reset(dones)
                    if self.actors_encoder_teacher[key] is not None:
                        self.actors_encoder_teacher[key].reset(dones)
                    if self.use_critic and self.critics[key] is not None:
                        self.critics[key].reset(dones)

                if self.log_dir is not None and self.is_main_process:
                    self.logging_helper.update_episode_stats(rew_total.to(self.device), dones, infos)

            if self.use_critic:
                for key in self.keys:
                    last_critic_obs = torch.cat([obs_dict[k] for k in self.critic_obs_keys[key]], dim=1)
                    last_values = self.critics[key].evaluate({"critic_obs": last_critic_obs}).detach().to(self.device)
                    returns, advantages = self._compute_returns_and_advantages(
                        last_values,
                        self.storages[key]["values"].to(self.device),
                        self.storages[key]["dones"].to(self.device),
                        self.storages[key]["rewards"].to(self.device),
                    )
                    self.storages[key]["returns"] = returns
                    self.storages[key]["advantages"] = advantages

        return obs_dict

    # ------------------------------------------------------------------
    # Losses / training
    # ------------------------------------------------------------------

    def _init_loss_dict_at_training_step(self) -> dict[str, float]:
        loss_dict = super()._init_loss_dict_at_training_step()
        self._terrain_loss_counts: dict[str, int] = {}
        for key in self.keys:
            for terrain_idx in self.expert_ids:
                tname = self._terrain_name(int(terrain_idx.item()))
                loss_dict[f"BC_{key}_{tname}"] = 0.0
                self._terrain_loss_counts[f"BC_{key}_{tname}"] = 0
        return loss_dict

    def _terrain_name(self, idx: int) -> str:
        if 0 <= idx < len(self.terrain_types):
            return self.terrain_types[idx]
        return str(idx)

    def _training_step(self) -> dict[str, float]:
        loss_dict = self._init_loss_dict_at_training_step()
        for key in self.keys:
            generator = self.storages[key].mini_batch_generator(
                self.config.num_mini_batches, self.config.num_learning_epochs
            )
            for minibatch in generator:
                loss_dict = self._update_algo_step(minibatch, loss_dict, key)

        num_updates = self.config.num_learning_epochs * self.config.num_mini_batches
        for loss_key in list(loss_dict.keys()):
            if loss_key in self._terrain_loss_counts and self._terrain_loss_counts[loss_key] > 0:
                loss_dict[loss_key] /= self._terrain_loss_counts[loss_key]
            else:
                loss_dict[loss_key] /= num_updates

        for key in self.keys:
            self.storages[key].clear()
        return loss_dict

    def _update_dagger(
        self,
        policy_state_dict: dict[str, torch.Tensor],
        loss_dict: dict[str, float],
        key: str,
    ) -> dict[str, float]:
        gt_actions_batch = policy_state_dict["actions_teacher"]
        self.actors[key].act(policy_state_dict)
        mu_batch = self.actors[key].action_mean

        terrain_idx = policy_state_dict["route_obs"]
        loss_type = self.per_actor_loss_types.get(key, "mse")
        bc_loss, terrain_losses = compute_per_terrain_loss(mu_batch, gt_actions_batch, terrain_idx, loss_type)

        for terrain_id, loss_value in terrain_losses.items():
            tname = self._terrain_name(int(terrain_id))
            kname = f"BC_{key}_{tname}"
            if kname in loss_dict:
                loss_dict[kname] += loss_value
                self._terrain_loss_counts[kname] += 1

        actor_loss = self.config.bc_loss_coef * bc_loss

        enc_loss: torch.Tensor | None = None
        if self.use_encoder_loss[key] and "encoder_output" in policy_state_dict:
            enc_loss = self._compute_loss(
                policy_state_dict["encoder_output"],
                policy_state_dict["encoder_output_teacher"],
                actor_key=key,
                policy_state_dict=policy_state_dict,
            )
            actor_loss = actor_loss + self.encoder_loss_coef[key] * enc_loss

        self.actor_optimizers[key].zero_grad()
        actor_loss.backward()
        if self.is_multi_gpu:
            self._reduce_parameters(key, actor_only=True)
        nn.utils.clip_grad_norm_(self.actors[key].parameters(), self.config.max_grad_norm)
        self.actor_optimizers[key].step()

        if self.use_critic and self.critics[key] is not None:
            returns_batch = policy_state_dict["returns"]
            target_values_batch = policy_state_dict["values"]
            value_batch = self.critics[key].evaluate(policy_state_dict)
            if self.config.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.config.clip_param, self.config.clip_param
                )
                v_losses = (value_batch - returns_batch).pow(2)
                v_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(v_losses, v_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()
            critic_loss = self.config.value_loss_coef * value_loss
            self.critic_optimizers[key].zero_grad()
            critic_loss.backward()
            if self.is_multi_gpu:
                self._reduce_parameters(key, critic_only=True)
            nn.utils.clip_grad_norm_(self.critics[key].parameters(), self.config.max_grad_norm)
            self.critic_optimizers[key].step()
            loss_dict[f"Value_{key}"] += value_loss.item()

        loss_dict[f"BC_{key}"] += bc_loss.item()
        if enc_loss is not None:
            loss_dict[f"Encoder_{key}"] += enc_loss.item()
        return loss_dict

    # ------------------------------------------------------------------
    # Teacher loading
    # ------------------------------------------------------------------

    def _resolve_per_terrain_ckpt(self, ckpt_path: str | None, terrain_type: str, wandb_map: dict[str, str]) -> Path:
        """Return a local path to the per-terrain checkpoint.

        Prefers ``wandb_map[terrain_type]`` (downloaded via the file cache) over
        the directory layout ``<ckpt_path>/<terrain_type>/<run_dir>/model_*.pt``.
        """
        uri = wandb_map.get(terrain_type) if wandb_map else None
        if uri:
            cached = get_cached_file_path(uri)
            return Path(cached)
        if not ckpt_path:
            raise FileNotFoundError(
                f"No checkpoint provided for terrain '{terrain_type}'. "
                "Pass ckpt_teacher (a directory) or ckpt_teacher_wandb (per-terrain wandb URIs)."
            )
        terrain_dir = Path(ckpt_path) / terrain_type
        if not terrain_dir.exists():
            raise FileNotFoundError(f"Terrain directory not found: {terrain_dir}")
        subdirs = [d for d in terrain_dir.iterdir() if d.is_dir()]
        if not subdirs:
            raise FileNotFoundError(f"No subdirectories found in {terrain_dir}")
        subdir = subdirs[0]
        model_files = [f for f in subdir.iterdir() if f.suffix == ".pt"]
        if not model_files:
            raise FileNotFoundError(f"No .pt model files found in {subdir}")
        return model_files[0]

    def load_teacher(self, ckpt_path: str | None, key: str) -> dict | None:
        wandb_map = {terrain: uri for terrain, uri in self.config.ckpt_teacher_wandb.items() if uri}
        if ckpt_path is None and not wandb_map:
            return None
        teacher = self.actors_teacher[key]
        if not hasattr(teacher, "experts"):
            if ckpt_path is None:
                raise ValueError(
                    "ckpt_teacher_wandb requires a MoE teacher. Use ckpt_teacher with a non-MoE teacher."
                )
            return super().load_teacher(ckpt_path, key)

        logger.info(f"Loading MoE teacher models for {key} from {ckpt_path or wandb_map}")
        loaded_count = 0
        total_expected = len(self.expert_ids)
        for expert_idx, terrain_idx in enumerate(self.expert_ids):
            terrain_type = self._terrain_name(int(terrain_idx.item()))
            terrain_model_path = self._resolve_per_terrain_ckpt(ckpt_path, terrain_type, wandb_map)
            logger.info(f"  [expert {expert_idx} / {terrain_type}] loading {terrain_model_path}")
            loaded_dict = torch.load(terrain_model_path, map_location=self.device)
            sd = loaded_dict["actor_model_state_dict"]
            if isinstance(sd, dict) and key in sd:
                sd = sd[key]
            teacher.experts[expert_idx].load_state_dict(sd)
            loaded_count += 1
        logger.info(f"MoE teacher models loaded: {loaded_count}/{total_expected}")
        return {"moe_models_loaded": True, "loaded_count": loaded_count, "total_expected": total_expected}

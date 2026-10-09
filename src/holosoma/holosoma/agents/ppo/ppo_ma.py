"""Decoupled PPO: separate actor–critic pairs for lower and upper body (RPL ``ppo_ma`` analogue)."""

from __future__ import annotations

import itertools
import os
from typing import Any, TypedDict, cast

import torch
import torch.nn.functional as F
from loguru import logger
from rich.console import Console
from torch import nn
from torch.distributions import Normal, kl_divergence
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

from holosoma.agents.base_algo.base_algo import BaseAlgo
from holosoma.agents.callbacks.base_callback import RLEvalCallback
from holosoma.agents.modules.augmentation_utils import SymmetryUtils
from holosoma.agents.modules.data_utils import RolloutStorage
from holosoma.agents.modules.logging_utils import LoggingHelper
from holosoma.agents.modules.module_utils import (
    setup_ppo_actor_module,
    setup_ppo_critic_module,
)
from holosoma.config_types.algo import PPOMAConfig
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.utils.helpers import instantiate
from holosoma.utils.inference_helpers import (
    attach_onnx_metadata,
    get_command_ranges_from_env,
    get_control_gains_from_config,
    get_urdf_text_from_robot_config,
)

console = Console()

# Avoid exp overflow in importance ratio r = π_new/π_old (stabilizes backward into actor).
_PPO_LOG_RATIO_CLAMP = 20.0
_KL_SIGMA_EPS = 1e-6


class MAMinibatch(TypedDict, total=False):
    actor_obs: torch.Tensor
    critic_obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    actions_log_prob: torch.Tensor
    action_mean: torch.Tensor
    action_sigma: torch.Tensor


class PPOMultiActorCritic(BaseAlgo):
    """Two policies / two value functions; env step uses concatenated actions and ``extras[\"rewards_ma\"]``."""

    def __init__(
        self,
        env: BaseTask,
        config: PPOMAConfig,
        log_dir,
        device="cpu",
        multi_gpu_cfg: dict | None = None,
    ):
        if not isinstance(config, PPOMAConfig):
            raise TypeError(f"PPOMultiActorCritic requires PPOMAConfig, got {type(config)}")
        super().__init__(env, config, device, multi_gpu_cfg)
        self.config = config
        self.log_dir = log_dir
        self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)
        self.logging_helper = LoggingHelper(
            self.writer,
            self.log_dir,
            device=self.device,
            num_envs=self.env.num_envs,
            num_steps_per_env=self.config.num_steps_per_env,
            num_learning_iterations=self.config.num_learning_iterations,
            is_main_process=self.is_main_process,
            num_gpus=self.gpu_world_size,
            log_raw_episode=self.config.log_raw_episode,
        )

        self.keys: tuple[str, ...] = tuple(self.config.body_keys)
        self._init_config()

        self.current_learning_iteration = 0
        self.eval_callbacks: list[RLEvalCallback] = []
        _ = self.env.reset_all()

    def _init_config(self) -> None:
        self.algo_obs_dim_dict = self.env.observation_manager.get_obs_dims()
        assert self.env.observation_manager is not None

        rc = self.env.robot_config
        self.num_act: dict[str, int] = {
            "lower_body": len(rc.lower_dof_names),
            "upper_body": len(rc.upper_dof_names),
        }

        self.actor_obs_keys: dict[str, list[str]] = {}
        self.critic_obs_keys: dict[str, list[str]] = {}
        md = self.config.module_dict
        for key in self.keys:
            self.actor_obs_keys[key] = list(getattr(md, f"actor_{key}").input_dim)
            self.critic_obs_keys[key] = list(getattr(md, f"critic_{key}").input_dim)

        self.init_noise_std: dict[str, float] = {
            "lower_body": self.config.init_noise_std_lower_body,
            "upper_body": self.config.init_noise_std_upper_body,
        }

        self.actor_learning_rates: dict[str, float] = {k: self.config.actor_learning_rate for k in self.keys}
        self.critic_learning_rates: dict[str, float] = {k: self.config.critic_learning_rate for k in self.keys}
        self.max_actor_learning_rates: dict[str, float] = {
            k: self.config.max_actor_learning_rate or max(self.actor_learning_rates[k], 1e-2) for k in self.keys
        }
        self.min_actor_learning_rates: dict[str, float] = {
            k: self.config.min_actor_learning_rate or min(self.actor_learning_rates[k], 1e-5) for k in self.keys
        }
        self.max_critic_learning_rates: dict[str, float] = {
            k: self.config.max_critic_learning_rate or max(self.critic_learning_rates[k], 1e-2) for k in self.keys
        }
        self.min_critic_learning_rates: dict[str, float] = {
            k: self.config.min_critic_learning_rate or min(self.critic_learning_rates[k], 1e-5) for k in self.keys
        }

        self.use_symmetry = self.config.use_symmetry
        if self.use_symmetry:
            self.symmetry_utils = SymmetryUtils(self.env)

    def _history_for_module(self, obs_keys: list[str]) -> dict[str, int]:
        out: dict[str, int] = {}
        for k in obs_keys:
            out[k] = self.env.observation_manager.cfg.groups[k].history_length
        return out

    def _get_obs_dim(self, obs_keys: list[str]) -> int:
        obs_dim = 0
        for obs_key in obs_keys:
            key_dim = self.algo_obs_dim_dict[obs_key]
            assert isinstance(key_dim, int), f"Observation dimension for {obs_key} is not an integer: {key_dim}"
            obs_dim += key_dim
        return obs_dim

    def setup(self):
        logger.info("Setting up multi-actor PPO")
        self._setup_models_and_optimizer()
        logger.info("Setting up rollout storage")
        self._setup_storage()
        if self.is_multi_gpu and self.has_curricula_enabled():
            logger.info(f"Multi-GPU curriculum synchronization enabled across {self.gpu_world_size} GPUs")

    def _setup_models_and_optimizer(self) -> None:
        self.actors: dict[str, nn.Module] = {}
        self.critics: dict[str, nn.Module] = {}
        md = self.config.module_dict
        for key in self.keys:
            acfg = getattr(md, f"actor_{key}")
            ccfg = getattr(md, f"critic_{key}")
            self.actors[key] = setup_ppo_actor_module(
                obs_dim_dict=self.algo_obs_dim_dict,
                module_config=acfg,
                num_actions=self.num_act[key],
                init_noise_std=self.init_noise_std[key],
                device=self.device,
                history_length=self._history_for_module(list(acfg.input_dim)),
            )
            self.critics[key] = setup_ppo_critic_module(
                obs_dim_dict=self.algo_obs_dim_dict,
                module_config=ccfg,
                device=self.device,
                history_length=self._history_for_module(list(ccfg.input_dim)),
            )

        if self.is_multi_gpu:
            for key in self.keys:
                for param in self.actors[key].parameters():
                    torch.distributed.broadcast(param.data, src=0)
                for param in self.critics[key].parameters():
                    torch.distributed.broadcast(param.data, src=0)
            logger.info(f"Synchronized multi-actor weights across {self.gpu_world_size} GPUs")

        self.actor_optimizers = {
            k: instantiate(
                self.config.actor_optimizer,
                params=self.actors[k].parameters(),
                lr=self.actor_learning_rates[k],
            )
            for k in self.keys
        }
        self.critic_optimizers = {
            k: instantiate(
                self.config.critic_optimizer,
                params=self.critics[k].parameters(),
                lr=self.critic_learning_rates[k],
            )
            for k in self.keys
        }

    def _setup_storage(self) -> None:
        self.storages: dict[str, RolloutStorage] = {}
        for key in self.keys:
            st = RolloutStorage(self.env.num_envs, self.config.num_steps_per_env, device=self.device)
            adim = self._get_obs_dim(self.actor_obs_keys[key])
            cdim = self._get_obs_dim(self.critic_obs_keys[key])
            console.print(f"[PPO-MA] {key} actor_obs dim={adim} critic_obs dim={cdim}")
            st.register("actor_obs", shape=(adim,), dtype=torch.float)
            st.register("critic_obs", shape=(cdim,), dtype=torch.float)
            na = self.num_act[key]
            for name, shape, dtype in [
                ("actions", (na,), torch.float),
                ("rewards", (1,), torch.float),
                ("dones", (1,), torch.bool),
                ("values", (1,), torch.float),
                ("returns", (1,), torch.float),
                ("advantages", (1,), torch.float),
                ("actions_log_prob", (1,), torch.float),
                ("action_mean", (na,), torch.float),
                ("action_sigma", (na,), torch.float),
            ]:
                st.register(name, shape=shape, dtype=dtype)
            self.storages[key] = st

    def _eval_mode(self) -> None:
        for k in self.keys:
            self.actors[k].eval()
            self.critics[k].eval()

    def _train_mode(self) -> None:
        for k in self.keys:
            self.actors[k].train()
            self.critics[k].train()

    def _concat_actions(self, actions_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([actions_dict[k] for k in self.keys], dim=1)

    def learn(self) -> None:
        self._train_mode()
        obs_dict = self.env.reset_all()
        if self.config.init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        for obs_key in obs_dict:
            obs_dict[obs_key] = obs_dict[obs_key].to(self.device)

        for it in range(
            self.current_learning_iteration,
            self.current_learning_iteration + self.config.num_learning_iterations,
        ):
            self.current_learning_iteration = it
            if self.is_multi_gpu:
                self._synchronize_curriculum_metrics()
            with self.logging_helper.record_collection_time():
                obs_dict = self._rollout_step(obs_dict)
            with self.logging_helper.record_learn_time():
                loss_dict = self._training_step()
            if self.is_main_process:
                self._post_epoch_logging(it, loss_dict)
            if it % self.config.save_interval == 0 and self.is_main_process:
                self.save(os.path.join(self.log_dir, f"model_{it:05d}.pt"))
                self.export(onnx_file_path=os.path.join(self.log_dir, f"model_{it:05d}.onnx"))

        if self.is_main_process:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration:05d}.pt"))
            self.export(onnx_file_path=os.path.join(self.log_dir, f"model_{self.current_learning_iteration:05d}.onnx"))

    def _rollout_step(self, obs_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        with torch.inference_mode():
            for _ in range(self.config.num_steps_per_env):
                actions_dict: dict[str, torch.Tensor] = {}
                values_dict: dict[str, torch.Tensor] = {}
                actor_obs_t: dict[str, torch.Tensor] = {}
                critic_obs_t: dict[str, torch.Tensor] = {}
                for key in self.keys:
                    actor_obs = torch.cat([obs_dict[k] for k in self.actor_obs_keys[key]], dim=1)
                    critic_obs = torch.cat([obs_dict[k] for k in self.critic_obs_keys[key]], dim=1)
                    actor_obs_t[key] = actor_obs
                    critic_obs_t[key] = critic_obs
                    actions_dict[key] = self.actors[key].act({"actor_obs": actor_obs})
                    values_dict[key] = self.critics[key].evaluate({"critic_obs": critic_obs}).detach()

                actions_full = self._concat_actions(actions_dict)
                obs_dict, rew_total, dones, infos = self.env.step({"actions": actions_full})

                for obs_key in obs_dict:
                    obs_dict[obs_key] = obs_dict[obs_key].to(self.device)
                dones = dones.to(self.device)

                rewards_ma = infos.get("rewards_ma")
                if rewards_ma is None:
                    raise RuntimeError(
                        "Multi-agent PPO requires reward config with multi_agent_body_keys "
                        "so the env populates extras['rewards_ma']."
                    )

                timeouts = infos["time_outs"].to(self.device) if infos.get("time_outs") is not None else None

                for key in self.keys:
                    r = rewards_ma[key].to(self.device).view(-1, 1)
                    final_r = torch.zeros_like(r)
                    if timeouts is not None and timeouts.any():
                        fo = infos.get("final_observations") or {}
                        final_critic_obs = torch.cat([fo[k] for k in self.critic_obs_keys[key]], dim=1).to(self.device)
                        fv = self.critics[key].evaluate({"critic_obs": final_critic_obs}).detach()
                        final_r += self.config.gamma * torch.squeeze(
                            fv * timeouts.unsqueeze(1).to(self.device), 1
                        ).unsqueeze(1)
                    self.storages[key].add(
                        actor_obs=actor_obs_t[key],
                        critic_obs=critic_obs_t[key],
                        actions=actions_dict[key],
                        values=values_dict[key],
                        actions_log_prob=self.actors[key].get_actions_log_prob(actions_dict[key]).detach().unsqueeze(1),
                        action_mean=self.actors[key].action_mean.detach(),
                        action_sigma=self.actors[key].action_std.detach(),
                        rewards=(r + final_r),
                        dones=dones.view(-1, 1),
                    )
                    self.actors[key].reset(dones)
                    self.critics[key].reset(dones)

                if self.log_dir is not None:
                    self.logging_helper.update_episode_stats(rew_total.to(self.device), dones, infos)

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

    def _compute_returns_and_advantages(
        self, last_values: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, rewards: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        advantage = 0
        returns = torch.zeros_like(values)
        num_steps = returns.shape[0]
        for step in reversed(range(num_steps)):
            if step == num_steps - 1:
                next_values = last_values
            else:
                next_values = values[step + 1]
            next_is_not_terminal = 1.0 - dones[step].float()
            delta = rewards[step] + next_is_not_terminal * self.config.gamma * next_values - values[step]
            advantage = delta + next_is_not_terminal * self.config.gamma * self.config.lam * advantage
            returns[step] = advantage + values[step]
        advantages = returns - values
        if self.is_multi_gpu:
            advantages = self._normalize_advantages_multi_gpu(advantages)
        else:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

    def _normalize_advantages_multi_gpu(self, advantages: torch.Tensor) -> torch.Tensor:
        local_stats = torch.stack([advantages.mean(), (advantages**2).mean()])
        torch.distributed.all_reduce(local_stats, op=torch.distributed.ReduceOp.SUM)
        global_mean = local_stats[0] / self.gpu_world_size
        global_sq_mean = local_stats[1] / self.gpu_world_size
        global_variance = global_sq_mean - global_mean**2
        global_std = torch.sqrt(global_variance + 1e-8)
        return (advantages - global_mean) / global_std

    def _training_step(self) -> dict[str, float]:
        loss_dict: dict[str, float] = {}
        for key in self.keys:
            gen = self.storages[key].mini_batch_generator(self.config.num_mini_batches, self.config.num_learning_epochs)
            for minibatch in gen:
                mb = cast(MAMinibatch, minibatch)
                loss_dict = self._update_algo_step(mb, loss_dict, key)
        nup = self.config.num_learning_epochs * self.config.num_mini_batches
        for k in list(loss_dict.keys()):
            loss_dict[k] /= nup
        for key in self.keys:
            self.storages[key].clear()
        return loss_dict

    def _update_algo_step(self, minibatch: MAMinibatch, loss_dict: dict[str, float], key: str) -> dict[str, float]:
        ppo_loss_dict = self._compute_ppo_loss(minibatch, key)
        self.actor_optimizers[key].zero_grad()
        self.critic_optimizers[key].zero_grad()
        loss_total = ppo_loss_dict["actor_loss"] + ppo_loss_dict["critic_loss"]
        if not torch.isfinite(loss_total):
            logger.warning(
                f"[PPO-MA] Non-finite loss for {key} (actor+critic); skipping optimizer step for this minibatch."
            )
            return loss_dict
        loss_total.backward()
        if self.is_multi_gpu:
            self._reduce_parameters(key)
        nn.utils.clip_grad_norm_(self.actors[key].parameters(), self.config.max_grad_norm)
        nn.utils.clip_grad_norm_(self.critics[key].parameters(), self.config.max_grad_norm)
        actor_bad = any(
            p.grad is not None and not torch.isfinite(p.grad).all() for p in self.actors[key].parameters()
        )
        critic_bad = any(
            p.grad is not None and not torch.isfinite(p.grad).all() for p in self.critics[key].parameters()
        )
        if actor_bad or critic_bad:
            logger.warning(
                f"[PPO-MA] Non-finite gradients for {key}; skipping optimizer step for this minibatch."
            )
            self.actor_optimizers[key].zero_grad()
            self.critic_optimizers[key].zero_grad()
            return loss_dict
        self.actor_optimizers[key].step()
        self.critic_optimizers[key].step()

        suffix = f"_{key}"
        loss_dict[f"Value{suffix}"] = loss_dict.get(f"Value{suffix}", 0.0) + ppo_loss_dict["value_loss"].item()
        loss_dict[f"Surrogate{suffix}"] = loss_dict.get(f"Surrogate{suffix}", 0.0) + ppo_loss_dict["surrogate_loss"].item()
        loss_dict[f"Entropy{suffix}"] = loss_dict.get(f"Entropy{suffix}", 0.0) + ppo_loss_dict["entropy_loss"].item()
        loss_dict[f"KL{suffix}"] = loss_dict.get(f"KL{suffix}", 0.0) + ppo_loss_dict["kl_mean"].item()
        for ek, ev in ppo_loss_dict.items():
            if ek in ("actor_loss", "critic_loss", "value_loss", "surrogate_loss", "entropy_loss", "kl_mean"):
                continue
            lv = ev.item() if torch.is_tensor(ev) else float(ev)
            loss_dict[f"{ek}{suffix}"] = loss_dict.get(f"{ek}{suffix}", 0.0) + lv
        return loss_dict

    def _compute_ppo_loss(self, minibatch: MAMinibatch, key: str) -> dict[str, Any]:
        actions_batch = minibatch["actions"]
        target_values_batch = minibatch["values"]
        advantages_batch = minibatch["advantages"]
        returns_batch = minibatch["returns"]
        old_actions_log_prob_batch = minibatch["actions_log_prob"]
        old_mu_batch = minibatch["action_mean"]
        old_sigma_batch = minibatch["action_sigma"]

        original_batch_size = actions_batch.shape[0]
        obs_list_a = self.actor_obs_keys[key]
        obs_list_c = self.critic_obs_keys[key]

        if self.use_symmetry:
            actor_obs = self.symmetry_utils.augment_observations(
                obs=minibatch["actor_obs"], env=self.env, obs_list=obs_list_a
            )
            critic_obs = self.symmetry_utils.augment_observations(
                obs=minibatch["critic_obs"], env=self.env, obs_list=obs_list_c
            )
            actions_batch = self.symmetry_utils.augment_actions(actions=actions_batch, body_key=key)
            num_aug = int(actor_obs.shape[0] / original_batch_size)
            old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
            target_values_batch = target_values_batch.repeat(num_aug, 1)
            advantages_batch = advantages_batch.repeat(num_aug, 1)
            returns_batch = returns_batch.repeat(num_aug, 1)
        else:
            actor_obs = minibatch["actor_obs"]
            critic_obs = minibatch["critic_obs"]

        actor = self.actors[key]
        critic = self.critics[key]
        actor.act({"actor_obs": actor_obs})
        value_batch = critic.evaluate({"critic_obs": critic_obs})
        actions_log_prob_batch = actor.get_actions_log_prob(actions_batch)
        mu_batch = actor.action_mean[:original_batch_size]
        sigma_batch = actor.action_std[:original_batch_size]
        entropy_batch = actor.entropy[:original_batch_size]

        kl_mean = torch.tensor(0.0, device=self.device)
        if self.config.desired_kl is not None and self.config.schedule == "adaptive":
            kl_mean = self._compute_kl_div(old_mu_batch, old_sigma_batch, mu_batch, sigma_batch)
            self._update_learning_rate(kl_mean, key)

        log_ratio = actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
        log_ratio = torch.clamp(log_ratio, -_PPO_LOG_RATIO_CLAMP, _PPO_LOG_RATIO_CLAMP)
        ratio = torch.exp(log_ratio)
        surrogate = -torch.squeeze(advantages_batch) * ratio
        surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
            ratio, 1.0 - self.config.clip_param, 1.0 + self.config.clip_param
        )
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
            -self.config.clip_param, self.config.clip_param
        )
        value_losses = (value_batch - returns_batch).pow(2)
        value_losses_clipped = (value_clipped - returns_batch).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()

        symmetry_actor_loss = torch.tensor(0.0, device=self.device)
        symmetry_critic_loss = torch.tensor(0.0, device=self.device)
        if self.use_symmetry and (self.config.symmetry_actor_coef > 0.0 or self.config.symmetry_critic_coef > 0.0):
            mean_actions_batch = actor.act_inference({"actor_obs": actor_obs.detach().clone()})
            m0, m1 = mean_actions_batch[:original_batch_size], mean_actions_batch[original_batch_size:]
            sym_a = self.symmetry_utils.augment_actions(actions=m0, body_key=key)[original_batch_size:]
            symmetry_actor_loss = F.mse_loss(m1, sym_a)
            symmetry_critic_loss = F.mse_loss(
                value_batch[:original_batch_size],
                value_batch[original_batch_size:],
            )

        entropy_loss = entropy_batch.mean()
        actor_loss = (
            surrogate_loss
            - self.config.entropy_coef * entropy_loss
            + self.config.symmetry_actor_coef * symmetry_actor_loss
        )
        critic_loss = self.config.value_loss_coef * value_loss + self.config.symmetry_critic_coef * symmetry_critic_loss

        return {
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "symmetry_actor_loss": symmetry_actor_loss,
            "symmetry_critic_loss": symmetry_critic_loss,
            "value_loss": value_loss,
            "surrogate_loss": surrogate_loss,
            "entropy_loss": entropy_loss,
            "kl_mean": kl_mean,
        }

    def _compute_kl_div(self, old_mu_batch, old_sigma_batch, mu_batch, sigma_batch) -> torch.Tensor:
        with torch.inference_mode():
            old_sigma_batch = torch.clamp(old_sigma_batch, min=_KL_SIGMA_EPS)
            sigma_batch = torch.clamp(sigma_batch, min=_KL_SIGMA_EPS)
            old_dist = Normal(old_mu_batch, old_sigma_batch)
            new_dist = Normal(mu_batch, sigma_batch)
            kl = kl_divergence(old_dist, new_dist).sum(-1)
            kl_mean = torch.mean(kl)
            if self.is_multi_gpu:
                torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                kl_mean /= self.gpu_world_size
        return kl_mean

    def _update_learning_rate(self, kl_mean: torch.Tensor, key: str) -> None:
        if kl_mean > self.config.desired_kl * 2.0:
            self.actor_learning_rates[key] = max(
                self.min_actor_learning_rates[key], self.actor_learning_rates[key] / 1.5
            )
            self.critic_learning_rates[key] = max(
                self.min_critic_learning_rates[key], self.critic_learning_rates[key] / 1.5
            )
        elif kl_mean < self.config.desired_kl / 2.0 and kl_mean > 0.0:
            self.actor_learning_rates[key] = min(
                self.max_actor_learning_rates[key], self.actor_learning_rates[key] * 1.5
            )
            self.critic_learning_rates[key] = min(
                self.max_critic_learning_rates[key], self.critic_learning_rates[key] * 1.5
            )

        # Broadcast rank-0 LR to all ranks so each body-part's adaptive schedule stays
        # identical across workers even under non-deterministic reductions.
        if self.is_multi_gpu:
            lr_tensor = torch.tensor(
                [self.actor_learning_rates[key], self.critic_learning_rates[key]],
                device=self.device,
                dtype=torch.float,
            )
            torch.distributed.broadcast(lr_tensor, src=0)
            self.actor_learning_rates[key] = float(lr_tensor[0].item())
            self.critic_learning_rates[key] = float(lr_tensor[1].item())

        for pg in self.actor_optimizers[key].param_groups:
            pg["lr"] = self.actor_learning_rates[key]
        for pg in self.critic_optimizers[key].param_groups:
            pg["lr"] = self.critic_learning_rates[key]

    def _reduce_parameters(self, key: str) -> None:
        grads = [
            param.grad.view(-1)
            for model in [self.actors[key], self.critics[key]]
            for param in model.parameters()
            if param.grad is not None
        ]
        if not grads:
            return
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        offset = 0
        for model in [self.actors[key], self.critics[key]]:
            for param in model.parameters():
                if param.grad is not None:
                    n = param.numel()
                    param.grad.data.copy_(all_grads[offset : offset + n].view_as(param.grad))
                    offset += n

    def load(self, ckpt_path: str | None) -> dict | None:
        if ckpt_path is None:
            return None
        logger.info(f"Loading checkpoint from {ckpt_path}")
        loaded = torch.load(ckpt_path, map_location=self.device)
        for key in self.keys:
            self.actors[key].load_state_dict(loaded["actor_model_state_dict"][key])
            self.critics[key].load_state_dict(loaded["critic_model_state_dict"][key])
        if self.config.load_optimizer:
            for key in self.keys:
                self.actor_optimizers[key].load_state_dict(loaded["actor_optimizer_state_dict"][key])
                self.critic_optimizers[key].load_state_dict(loaded["critic_optimizer_state_dict"][key])
                self.actor_learning_rates[key] = loaded["actor_optimizer_state_dict"][key]["param_groups"][0]["lr"]
                self.critic_learning_rates[key] = loaded["critic_optimizer_state_dict"][key]["param_groups"][0]["lr"]
        self.current_learning_iteration = loaded["iter"]
        self._restore_env_state(loaded.get("env_state"))
        return loaded.get("infos")

    def save(self, path: str, infos=None) -> None:
        d: dict[str, Any] = {
            "actor_model_state_dict": {k: self.actors[k].state_dict() for k in self.keys},
            "critic_model_state_dict": {k: self.critics[k].state_dict() for k in self.keys},
            "actor_optimizer_state_dict": {k: self.actor_optimizers[k].state_dict() for k in self.keys},
            "critic_optimizer_state_dict": {k: self.critic_optimizers[k].state_dict() for k in self.keys},
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        d.update(self._checkpoint_metadata(iteration=self.current_learning_iteration))
        env_state = self._collect_env_state()
        if env_state:
            d["env_state"] = env_state
        self.logging_helper.save_checkpoint_artifact(d, path)

    def export(self, onnx_file_path: str) -> None:
        was_training = self.actors[self.keys[0]].training
        self._eval_mode()
        os.makedirs(os.path.dirname(onnx_file_path), exist_ok=True)
        kp_list, kd_list = get_control_gains_from_config(self.env.robot_config)
        cmd_ranges = get_command_ranges_from_env(self.env)
        urdf_file_path, urdf_str = get_urdf_text_from_robot_config(self.env.robot_config)
        shared_meta = {
            "dof_names": self.env.robot_config.dof_names,
            "kp": kp_list,
            "kd": kd_list,
            "command_ranges": cmd_ranges,
            "robot_urdf": urdf_str,
            "robot_urdf_path": urdf_file_path,
            "policy": "ppo_ma",
            "body_keys": list(self.keys),
        }
        shared_meta.update(self._checkpoint_metadata(iteration=self.current_learning_iteration))

        class _DecoupledActorOnnx(nn.Module):
            def __init__(self, actor_modules: dict[str, nn.Module], keys: tuple[str, ...]):
                super().__init__()
                self.keys = keys
                self.actor_modules = nn.ModuleDict({k: actor_modules[k] for k in keys})

            def forward(self, *actor_obs_list: torch.Tensor) -> torch.Tensor:
                actions = []
                for i, key in enumerate(self.keys):
                    actions.append(self.actor_modules[key].act_inference({"actor_obs": actor_obs_list[i]}))
                return torch.cat(actions, dim=1)

        example_input_list = []
        input_names = []
        for key in self.keys:
            adim = self._get_obs_dim(self.actor_obs_keys[key])
            example_input_list.append(torch.zeros(1, adim, device=self.device))
            input_names.append(f"actor_obs_{key}")

        wrapper = _DecoupledActorOnnx(self.actors, self.keys)
        torch.onnx.export(
            wrapper,
            tuple(example_input_list),
            onnx_file_path,
            verbose=False,
            input_names=input_names,
            output_names=["action"],
            opset_version=13,
            dynamo=False,
        )
        attach_onnx_metadata(onnx_path=onnx_file_path, metadata=shared_meta)
        self.logging_helper.save_to_wandb(onnx_file_path)

        if was_training:
            self._train_mode()

    def _post_epoch_logging(self, it: int, loss_dict: dict[str, float]) -> None:
        extra = {
            "Policy": {f"mean_noise_std_{k}": self.actors[k].std.mean().item() for k in self.keys},
        }
        for k in self.keys:
            loss_dict[f"actor_learning_rate_{k}"] = self.actor_learning_rates[k]
            loss_dict[f"critic_learning_rate_{k}"] = self.critic_learning_rates[k]
        self.logging_helper.post_epoch_logging(it=it, loss_dict=loss_dict, extra_log_dicts=extra)

    def get_inference_policy(self, device: str | None = None):
        for k in self.keys:
            self.actors[k].eval()
            if device is not None:
                self.actors[k].to(device)

        def policy(actor_obs_by_key: dict[str, torch.Tensor]) -> torch.Tensor:
            acts = []
            for k in self.keys:
                acts.append(self.actors[k].act_inference({"actor_obs": actor_obs_by_key[k]}))
            return torch.cat(acts, dim=1)

        return policy

    @torch.no_grad()
    def evaluate_policy(self, max_eval_steps: int | None = None) -> None:
        self._create_eval_callbacks()
        self._pre_evaluate_policy()
        actor_state: dict[str, Any] = {"done_indices": [], "stop": False}
        self.eval_policy = self.get_inference_policy()
        obs_dict = self.env.reset_all()
        actor_state.update(
            {
                "obs": obs_dict,
                "actions": torch.zeros(self.env.num_envs, sum(self.num_act[k] for k in self.keys), device=self.device),
            }
        )
        actor_state = self._pre_eval_env_step(actor_state)
        for step in itertools.islice(itertools.count(), max_eval_steps):
            actor_state["step"] = step
            actor_state = self._pre_eval_env_step(actor_state)
            actor_state = self.env_step(actor_state)
            actor_state = self._post_eval_env_step(actor_state)
        self._post_evaluate_policy()

    def env_step(self, actor_state: dict[str, Any]) -> dict[str, Any]:
        obs_dict, rewards, dones, extras = self.env.step({"actions": actor_state["actions"]})
        actor_state.update({"obs": obs_dict, "rewards": rewards, "dones": dones, "extras": extras})
        return actor_state

    def _create_eval_callbacks(self) -> None:
        if self.config.eval_callbacks is not None:
            for cb in self.config.eval_callbacks:
                self.eval_callbacks.append(instantiate(self.config.eval_callbacks[cb], training_loop=self))

    def _pre_evaluate_policy(self, reset_env: bool = True) -> None:
        self._eval_mode()
        self.env.set_is_evaluating()
        if reset_env:
            _ = self.env.reset_all()
        for c in self.eval_callbacks:
            c.on_pre_evaluate_policy()

    def _post_evaluate_policy(self) -> None:
        for c in self.eval_callbacks:
            c.on_post_evaluate_policy()

    def _pre_eval_env_step(self, actor_state: dict[str, Any]) -> dict[str, Any]:
        obs = actor_state["obs"]
        obs_by_key = {k: torch.cat([obs[g] for g in self.actor_obs_keys[k]], dim=1) for k in self.keys}
        actions = self.eval_policy(obs_by_key)
        actor_state.update({"actions": actions})
        for c in self.eval_callbacks:
            actor_state = c.on_pre_eval_env_step(actor_state)
        return actor_state

    def _post_eval_env_step(self, actor_state: dict[str, Any]) -> dict[str, Any]:
        for c in self.eval_callbacks:
            actor_state = c.on_post_eval_env_step(actor_state)
        return actor_state

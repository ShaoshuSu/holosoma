"""Decoupled DAgger: student/teacher actor pairs per body part.

Ported from FAR-FALCON-Terrains ``humanoidverse/agents/decouple/dagger_ma.py``
and adapted to Holosoma's config types, observation manager, and rollout
storage APIs.  U-Net heightmap reconstruction is intentionally omitted.
"""

from __future__ import annotations

import itertools
import os
from typing import Any

import torch
from loguru import logger
from rich.console import Console
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

from holosoma.agents.base_algo.base_algo import BaseAlgo
from holosoma.agents.callbacks.base_callback import RLEvalCallback
from holosoma.agents.modules.data_utils import RolloutStorage
from holosoma.agents.modules.logging_utils import LoggingHelper
from holosoma.agents.modules.module_utils import (
    setup_ppo_critic_module,
    setup_student_teacher_actor_module,
)
from holosoma.config_types.algo import DAggerMAConfig
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.utils.helpers import instantiate
from holosoma.utils.inference_helpers import (
    attach_onnx_metadata,
    get_command_ranges_from_env,
    get_control_gains_from_config,
    get_urdf_text_from_robot_config,
)

console = Console()

_SUPPORTED_LOSS_TYPES = ("l1_norm", "l2_norm", "mse", "mae", "bce", "mse_domain")


class DAggerMultiActor(BaseAlgo):
    """Two student actors (lower / upper body) distilled from two teachers."""

    config: DAggerMAConfig

    def __init__(
        self,
        env: BaseTask,
        config: DAggerMAConfig,
        log_dir: str,
        device: str = "cpu",
        multi_gpu_cfg: dict | None = None,
    ):
        if not isinstance(config, DAggerMAConfig):
            raise TypeError(f"DAggerMultiActor requires DAggerMAConfig, got {type(config)}")
        super().__init__(env, config, device, multi_gpu_cfg)
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
        self._init_obs_keys()

        self.current_learning_iteration = 0
        self.eval_callbacks: list[RLEvalCallback] = []
        _ = self.env.reset_all()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _init_config(self) -> None:
        self.algo_obs_dim_dict = self.env.observation_manager.get_obs_dims()
        assert self.env.observation_manager is not None

        rc = self.env.robot_config
        self.num_act: dict[str, int] = {
            "lower_body": len(rc.lower_dof_names),
            "upper_body": len(rc.upper_dof_names),
        }

        self.init_noise_std: dict[str, float] = {
            "lower_body": self.config.init_noise_std_lower_body,
            "upper_body": self.config.init_noise_std_upper_body,
        }

        # Per-actor loss type
        self.per_actor_loss_types: dict[str, str] = {
            "lower_body": self.config.loss_types.lower_body,
            "upper_body": self.config.loss_types.upper_body,
        }
        for key, lt in self.per_actor_loss_types.items():
            if lt not in _SUPPORTED_LOSS_TYPES:
                raise ValueError(
                    f"Unsupported loss type for {key}: {lt}. Supported: {_SUPPORTED_LOSS_TYPES}"
                )

        self.use_encoder_loss: dict[str, bool] = {
            "lower_body": self.config.use_encoder_loss.lower_body,
            "upper_body": self.config.use_encoder_loss.upper_body,
        }
        self.encoder_loss_coef: dict[str, float] = {
            "lower_body": self.config.encoder_loss_coef.lower_body,
            "upper_body": self.config.encoder_loss_coef.upper_body,
        }

        self.use_critic = self.config.use_critic

        self.actor_learning_rates: dict[str, float] = {k: self.config.actor_learning_rate for k in self.keys}
        self.critic_learning_rates: dict[str, float | None] = {
            k: (self.config.critic_learning_rate if self.use_critic else None) for k in self.keys
        }

    def _history_for_module(self, obs_keys: list[str]) -> dict[str, int]:
        out: dict[str, int] = {}
        for k in obs_keys:
            out[k] = self.env.observation_manager.cfg.groups[k].history_length
        return out

    def _init_obs_keys(self) -> None:
        md = self.config.module_dict
        self.actor_obs_keys: dict[str, list[str]] = {}
        self.actor_teacher_obs_keys: dict[str, list[str]] = {}
        self.critic_obs_keys: dict[str, list[str]] = {}
        self.actor_encoder_teacher_obs_keys: dict[str, list[str] | None] = {}

        for key in self.keys:
            self.actor_obs_keys[key] = list(getattr(md, f"actor_{key}").input_dim)
            self.actor_teacher_obs_keys[key] = list(getattr(md, f"actor_teacher_{key}").input_dim)
            critic_cfg = getattr(md, f"critic_{key}", None)
            # Default critic obs = teacher obs (teacher and critic share privileged obs).
            if critic_cfg is not None:
                self.critic_obs_keys[key] = list(critic_cfg.input_dim)
            else:
                self.critic_obs_keys[key] = list(self.actor_teacher_obs_keys[key])

            enc_cfg = getattr(md, f"actor_encoder_teacher_{key}", None)
            if self.use_encoder_loss[key] and enc_cfg is not None:
                self.actor_encoder_teacher_obs_keys[key] = list(enc_cfg.input_dim)
            else:
                self.actor_encoder_teacher_obs_keys[key] = None

    def _get_obs_dim(self, obs_keys: list[str]) -> int:
        obs_dim = 0
        for obs_key in obs_keys:
            key_dim = self.algo_obs_dim_dict[obs_key]
            assert isinstance(key_dim, int), f"Observation dimension for {obs_key} is not an integer: {key_dim}"
            obs_dim += key_dim
        return obs_dim

    def setup(self):
        logger.info("Setting up decoupled DAgger")
        self._setup_models_and_optimizer()
        logger.info("Setting up rollout storage")
        self._setup_storage()

    def _setup_models_and_optimizer(self) -> None:
        md = self.config.module_dict
        self.actors: dict[str, nn.Module] = {}
        self.actors_teacher: dict[str, nn.Module] = {}
        self.actors_encoder_teacher: dict[str, nn.Module | None] = {}
        self.critics: dict[str, nn.Module | None] = {}

        for key in self.keys:
            actor_cfg = getattr(md, f"actor_{key}")
            teacher_cfg = getattr(md, f"actor_teacher_{key}")

            self.actors[key] = setup_student_teacher_actor_module(
                obs_dim_dict=self.algo_obs_dim_dict,
                module_config=actor_cfg,
                num_actions=self.num_act[key],
                init_noise_std=self.init_noise_std[key],
                device=self.device,
                history_length=self._history_for_module(list(actor_cfg.input_dim)),
            )
            self.actors_teacher[key] = setup_student_teacher_actor_module(
                obs_dim_dict=self.algo_obs_dim_dict,
                module_config=teacher_cfg,
                num_actions=self.num_act[key],
                init_noise_std=self.init_noise_std[key],
                device=self.device,
                history_length=self._history_for_module(list(teacher_cfg.input_dim)),
            )

            enc_cfg = getattr(md, f"actor_encoder_teacher_{key}", None)
            if self.use_encoder_loss[key] and enc_cfg is not None:
                self.actors_encoder_teacher[key] = setup_student_teacher_actor_module(
                    obs_dim_dict=self.algo_obs_dim_dict,
                    module_config=enc_cfg,
                    num_actions=self.num_act[key],
                    init_noise_std=self.init_noise_std[key],
                    device=self.device,
                    history_length=self._history_for_module(list(enc_cfg.input_dim)),
                )
            else:
                self.actors_encoder_teacher[key] = None

            if self.use_critic:
                critic_cfg = getattr(md, f"critic_{key}")
                self.critics[key] = setup_ppo_critic_module(
                    obs_dim_dict=self.algo_obs_dim_dict,
                    module_config=critic_cfg,
                    device=self.device,
                    history_length=self._history_for_module(list(critic_cfg.input_dim)),
                )
            else:
                self.critics[key] = None

        # Load teacher policies (skipped during evaluation).
        if self.config.ckpt_teacher is not None:
            for key in self.keys:
                self.load_teacher(self.config.ckpt_teacher, key)
                if self.config.ckpt_actor_module is not None:
                    self.load_actor_module(self.config.ckpt_actor_module, key)
                if self.use_encoder_loss[key] and self.config.ckpt_encoder_teacher is not None:
                    self.load_encoder_teacher(self.config.ckpt_encoder_teacher, key)
                self.actors_teacher[key].eval()
                if self.actors_encoder_teacher[key] is not None:
                    self.actors_encoder_teacher[key].eval()

        if self.is_multi_gpu:
            self._synchronize_model_weights()

        self.actor_optimizers = {
            k: instantiate(
                self.config.actor_optimizer,
                params=self.actors[k].parameters(),
                lr=self.actor_learning_rates[k],
            )
            for k in self.keys
        }
        self.critic_optimizers: dict[str, Any] = {}
        for k in self.keys:
            if self.use_critic and self.critics[k] is not None:
                self.critic_optimizers[k] = instantiate(
                    self.config.critic_optimizer,
                    params=self.critics[k].parameters(),
                    lr=self.critic_learning_rates[k],
                )
            else:
                self.critic_optimizers[k] = None

        if self.config.use_scheduler:
            self._setup_schedulers()
        if self.config.use_warmup:
            self._setup_warmup()

    def _setup_schedulers(self) -> None:
        schedulers: dict[str, Any] = {}
        critic_schedulers: dict[str, Any] = {}
        total_iters = self.config.num_scheduler_iterations or self.config.num_learning_iterations

        defaults = {
            "StepLR": {"step_size": 1000, "gamma": 0.9},
            "ExponentialLR": {"gamma": 0.9},
            "CosineAnnealingLR": {"T_max": total_iters, "eta_min": 1e-6},
            "ReduceLROnPlateau": {
                "mode": "min",
                "factor": 0.5,
                "patience": 10,
                "threshold": 1e-4,
                "min_lr": 1e-6,
            },
            "OneCycleLR": {
                "max_lr": max(self.actor_learning_rates.values()),
                "total_steps": total_iters,
            },
        }
        scheduler_type = self.config.scheduler_type
        params = {**defaults.get(scheduler_type, {}), **dict(self.config.scheduler_params)}

        scheduler_cls_map = {
            "StepLR": lr_scheduler.StepLR,
            "ExponentialLR": lr_scheduler.ExponentialLR,
            "CosineAnnealingLR": lr_scheduler.CosineAnnealingLR,
            "ReduceLROnPlateau": lr_scheduler.ReduceLROnPlateau,
            "OneCycleLR": lr_scheduler.OneCycleLR,
        }
        if scheduler_type not in scheduler_cls_map:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        cls = scheduler_cls_map[scheduler_type]

        for key in self.keys:
            schedulers[key] = cls(self.actor_optimizers[key], **params)
            if self.use_critic and self.critic_optimizers[key] is not None:
                critic_schedulers[key] = cls(self.critic_optimizers[key], **params)

        self.schedulers = schedulers
        self.critic_schedulers = critic_schedulers

    def _setup_warmup(self) -> None:
        p = self.config.warmup_params
        self.warmup_schedulers = {
            k: lr_scheduler.LinearLR(
                self.actor_optimizers[k],
                start_factor=p.warmup_start_factor,
                end_factor=p.warmup_end_factor,
                total_iters=p.warmup_steps,
            )
            for k in self.keys
        }
        self.critic_warmup_schedulers = {}
        if self.use_critic:
            for k in self.keys:
                if self.critic_optimizers[k] is None:
                    continue
                self.critic_warmup_schedulers[k] = lr_scheduler.LinearLR(
                    self.critic_optimizers[k],
                    start_factor=p.warmup_start_factor,
                    end_factor=p.warmup_end_factor,
                    total_iters=p.warmup_steps,
                )

    def _synchronize_model_weights(self) -> None:
        for key in self.keys:
            for param in self.actors[key].parameters():
                torch.distributed.broadcast(param.data, src=0)
            for param in self.actors_teacher[key].parameters():
                torch.distributed.broadcast(param.data, src=0)
            if self.actors_encoder_teacher[key] is not None:
                for param in self.actors_encoder_teacher[key].parameters():
                    torch.distributed.broadcast(param.data, src=0)
            if self.use_critic and self.critics[key] is not None:
                for param in self.critics[key].parameters():
                    torch.distributed.broadcast(param.data, src=0)
        logger.info(f"Synchronized multi-actor DAgger weights across {self.gpu_world_size} GPUs")

    def _setup_storage(self) -> None:
        self.storages: dict[str, RolloutStorage] = {}
        for key in self.keys:
            st = RolloutStorage(self.env.num_envs, self.config.num_steps_per_env, device=self.device)

            a_dim = self._get_obs_dim(self.actor_obs_keys[key])
            t_dim = self._get_obs_dim(self.actor_teacher_obs_keys[key])
            c_dim = self._get_obs_dim(self.critic_obs_keys[key])
            console.print(f"[DAgger-MA] {key} actor_obs={a_dim} actor_teacher_obs={t_dim} critic_obs={c_dim}")

            st.register("actor_obs", shape=(a_dim,), dtype=torch.float)
            st.register("actor_teacher_obs", shape=(t_dim,), dtype=torch.float)
            st.register("critic_obs", shape=(c_dim,), dtype=torch.float)

            enc_keys = self.actor_encoder_teacher_obs_keys[key]
            if self.use_encoder_loss[key] and enc_keys is not None:
                e_dim = self._get_obs_dim(enc_keys)
                st.register("actor_encoder_teacher_obs", shape=(e_dim,), dtype=torch.float)

            na = self.num_act[key]
            for name, shape, dtype in [
                ("actions", (na,), torch.float),
                ("actions_teacher", (na,), torch.float),
                ("rewards", (1,), torch.float),
                ("dones", (1,), torch.bool),
            ]:
                st.register(name, shape=shape, dtype=dtype)

            if self.use_encoder_loss[key]:
                # Use the student's encoder dim as the shared width.
                actor_module = getattr(self.actors[key], "actor_module", None)
                enc_dim = int(getattr(actor_module, "encoder_output_dim", 0) or 0)
                if enc_dim > 0:
                    st.register("encoder_output", shape=(enc_dim,), dtype=torch.float)
                    st.register("encoder_output_teacher", shape=(enc_dim,), dtype=torch.float)

            if self.use_critic:
                for name, shape, dtype in [
                    ("values", (1,), torch.float),
                    ("returns", (1,), torch.float),
                    ("advantages", (1,), torch.float),
                ]:
                    st.register(name, shape=shape, dtype=dtype)

            self.storages[key] = st

    # ------------------------------------------------------------------
    # Mode helpers
    # ------------------------------------------------------------------

    def _eval_mode(self) -> None:
        for actor in self.actors.values():
            actor.eval()
        if self.use_critic:
            for critic in self.critics.values():
                if critic is not None:
                    critic.eval()

    def _train_mode(self) -> None:
        for actor in self.actors.values():
            actor.train()
        if self.use_critic:
            for critic in self.critics.values():
                if critic is not None:
                    critic.train()

    # ------------------------------------------------------------------
    # Load / save
    # ------------------------------------------------------------------

    def load(self, ckpt_path: str | None) -> dict | None:
        if ckpt_path is None:
            return None
        logger.info(f"Loading checkpoint from {ckpt_path}")
        loaded = torch.load(ckpt_path, map_location=self.device)
        for key in self.keys:
            self.actors[key].load_state_dict(loaded["actor_model_state_dict"][key])
            if self.use_critic and "critic_model_state_dict" in loaded and self.critics[key] is not None:
                self.critics[key].load_state_dict(loaded["critic_model_state_dict"][key])
        if self.config.load_optimizer:
            for key in self.keys:
                self.actor_optimizers[key].load_state_dict(loaded["actor_optimizer_state_dict"][key])
                self.actor_learning_rates[key] = loaded["actor_optimizer_state_dict"][key]["param_groups"][0]["lr"]
                if self.use_critic and self.critic_optimizers[key] is not None and \
                        "critic_optimizer_state_dict" in loaded:
                    self.critic_optimizers[key].load_state_dict(loaded["critic_optimizer_state_dict"][key])
                    self.critic_learning_rates[key] = loaded["critic_optimizer_state_dict"][key]["param_groups"][0]["lr"]
        self.current_learning_iteration = loaded["iter"]
        self._restore_env_state(loaded.get("env_state"))
        return loaded.get("infos")

    def load_teacher(self, ckpt_path: str, key: str) -> dict | None:
        logger.info(f"Loading teacher checkpoint for {key} from {ckpt_path}")
        loaded = torch.load(ckpt_path, map_location=self.device)
        sd = loaded["actor_model_state_dict"]
        if isinstance(sd, dict) and key in sd:
            self.actors_teacher[key].load_state_dict(sd[key])
        else:
            self.actors_teacher[key].load_state_dict(sd)
        return loaded.get("infos")

    def load_actor_module(self, ckpt_path: str, key: str) -> dict | None:
        logger.info(f"Loading actor module checkpoint for {key} from {ckpt_path}")
        loaded = torch.load(ckpt_path, map_location=self.device)
        src = loaded["actor_model_state_dict"]
        if isinstance(src, dict) and key in src:
            src = src[key]
        # Only pull actor_module.module.* weights, as in the source repo.
        filtered = {k: v for k, v in src.items() if k.startswith("actor_module.module.")}
        self.actors[key].load_state_dict(filtered, strict=False)
        return loaded.get("infos")

    def load_encoder_teacher(self, ckpt_path: str, key: str) -> dict | None:
        logger.info(f"Loading encoder teacher checkpoint for {key} from {ckpt_path}")
        loaded = torch.load(ckpt_path, map_location=self.device)
        sd = loaded["actor_model_state_dict"]
        if isinstance(sd, dict) and key in sd:
            self.actors_encoder_teacher[key].load_state_dict(sd[key])
        else:
            self.actors_encoder_teacher[key].load_state_dict(sd)
        return loaded.get("infos")

    def save(self, path: str, infos=None) -> None:
        d: dict[str, Any] = {
            "actor_model_state_dict": {k: self.actors[k].state_dict() for k in self.keys},
            "actor_optimizer_state_dict": {k: self.actor_optimizers[k].state_dict() for k in self.keys},
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        if self.use_critic:
            d["critic_model_state_dict"] = {
                k: self.critics[k].state_dict() for k in self.keys if self.critics[k] is not None
            }
            d["critic_optimizer_state_dict"] = {
                k: self.critic_optimizers[k].state_dict() for k in self.keys if self.critic_optimizers[k] is not None
            }
        d.update(self._checkpoint_metadata(iteration=self.current_learning_iteration))
        env_state = self._collect_env_state()
        if env_state:
            d["env_state"] = env_state
        self.logging_helper.save_checkpoint_artifact(d, path)

    # ------------------------------------------------------------------
    # Learning loop
    # ------------------------------------------------------------------

    def learn(self) -> None:
        self._train_mode()
        obs_dict = self.env.reset_all()
        if self.config.init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        for obs_key in obs_dict:
            obs_dict[obs_key] = obs_dict[obs_key].to(self.device)

        start = self.current_learning_iteration
        for it in range(start, start + self.config.num_learning_iterations):
            self.current_learning_iteration = it
            if self.is_multi_gpu:
                self._synchronize_curriculum_metrics()

            with self.logging_helper.record_collection_time():
                obs_dict = self._rollout_step(obs_dict)

            with self.logging_helper.record_learn_time():
                loss_dict = self._training_step()

            self._update_learning_rate(it)

            if self.is_main_process:
                self._post_epoch_logging(it, loss_dict)

            if it % self.config.save_interval == 0 and self.is_main_process:
                self.save(os.path.join(self.log_dir, f"model_{it:05d}.pt"))
                self.export(onnx_file_path=os.path.join(self.log_dir, f"model_{it:05d}.onnx"))

        if self.is_main_process:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration:05d}.pt"))
            self.export(onnx_file_path=os.path.join(self.log_dir, f"model_{self.current_learning_iteration:05d}.onnx"))

    def _concat_actions(self, actions_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([actions_dict[k] for k in self.keys], dim=1)

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
                    # Reward for this body part (fall back to total if no MA rewards provided).
                    if rewards_ma is not None and key in rewards_ma:
                        r = rewards_ma[key].to(self.device).view(-1, 1)
                    else:
                        r = rew_total.to(self.device).view(-1, 1)

                    if self.use_critic and infos.get("time_outs") is not None and infos["time_outs"].any():
                        timeouts = infos["time_outs"].to(self.device)
                        fo = infos.get("final_observations") or {}
                        final_critic_obs = torch.cat([fo[k] for k in self.critic_obs_keys[key]], dim=1).to(self.device)
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

    def _multi_actor_rollout_step(
        self, obs_dict: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, torch.Tensor]]]:
        actions_dict: dict[str, torch.Tensor] = {}
        per_key: dict[str, dict[str, torch.Tensor]] = {}
        for key in self.keys:
            actor_obs = torch.cat([obs_dict[k] for k in self.actor_obs_keys[key]], dim=1)
            actor_teacher_obs = torch.cat([obs_dict[k] for k in self.actor_teacher_obs_keys[key]], dim=1)
            critic_obs = torch.cat([obs_dict[k] for k in self.critic_obs_keys[key]], dim=1)

            actions = self.actors[key].act({"actor_obs": actor_obs})
            teacher_inputs = self._teacher_inputs(obs_dict, key, actor_teacher_obs)
            actions_teacher = self.actors_teacher[key].act(teacher_inputs)

            data: dict[str, torch.Tensor] = {
                "actor_obs": actor_obs,
                "actor_teacher_obs": actor_teacher_obs,
                "critic_obs": critic_obs,
                "actions": actions,
                "actions_teacher": actions_teacher,
            }

            if self.use_critic and self.critics[key] is not None:
                data["values"] = self.critics[key].evaluate({"critic_obs": critic_obs}).detach()

            if self.use_encoder_loss[key] and self.actors[key].get_encoder_output() is not None:
                enc_out = self.actors[key].get_encoder_output()
                data["encoder_output"] = enc_out
                enc_keys = self.actor_encoder_teacher_obs_keys[key]
                if self.actors_encoder_teacher[key] is not None and enc_keys is not None:
                    enc_obs = torch.cat([obs_dict[k] for k in enc_keys], dim=1)
                    self.actors_encoder_teacher[key].act({"actor_obs": enc_obs})
                    data["encoder_output_teacher"] = self.actors_encoder_teacher[key].get_encoder_output()
                    data["actor_encoder_teacher_obs"] = enc_obs
                else:
                    data["encoder_output_teacher"] = self.actors_teacher[key].get_encoder_output()

            actions_dict[key] = actions
            per_key[key] = data

        return actions_dict, per_key

    def _teacher_inputs(
        self,
        obs_dict: dict[str, torch.Tensor],
        key: str,
        actor_teacher_obs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Extension point for subclasses that need extra teacher inputs (e.g. MoE routing)."""
        return {"actor_obs": actor_teacher_obs}

    def _compute_returns_and_advantages(
        self, last_values: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, rewards: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        advantage = 0
        returns = torch.zeros_like(values)
        num_steps = returns.shape[0]
        for step in reversed(range(num_steps)):
            next_values = last_values if step == num_steps - 1 else values[step + 1]
            next_is_not_terminal = 1.0 - dones[step].float()
            delta = rewards[step] + next_is_not_terminal * self.config.gamma * next_values - values[step]
            advantage = delta + next_is_not_terminal * self.config.gamma * self.config.lam * advantage
            returns[step] = advantage + values[step]
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

    # ------------------------------------------------------------------
    # Losses & updates
    # ------------------------------------------------------------------

    def _compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        actor_key: str | None = None,
        policy_state_dict: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        loss_type = self.per_actor_loss_types.get(actor_key, "mse") if actor_key else "mse"
        error = predictions - targets
        if loss_type == "l1_norm":
            return error.abs().sum(dim=-1).mean()
        if loss_type == "l2_norm":
            return torch.sqrt((error**2).sum(dim=-1)).mean()
        if loss_type == "mse":
            return (error**2).mean()
        if loss_type == "mae":
            return error.abs().mean()
        if loss_type == "bce":
            return torch.nn.functional.binary_cross_entropy_with_logits(predictions, targets)
        raise ValueError(f"Unsupported loss type: {loss_type}")

    def _init_loss_dict_at_training_step(self) -> dict[str, float]:
        loss_dict: dict[str, float] = {}
        for key in self.keys:
            loss_dict[f"BC_{key}"] = 0.0
            if self.use_encoder_loss[key]:
                loss_dict[f"Encoder_{key}"] = 0.0
            if self.use_critic:
                loss_dict[f"Value_{key}"] = 0.0
        return loss_dict

    def _training_step(self) -> dict[str, float]:
        loss_dict = self._init_loss_dict_at_training_step()
        for key in self.keys:
            generator = self.storages[key].mini_batch_generator(
                self.config.num_mini_batches, self.config.num_learning_epochs
            )
            for minibatch in generator:
                loss_dict = self._update_algo_step(minibatch, loss_dict, key)
        num_updates = self.config.num_learning_epochs * self.config.num_mini_batches
        for k in list(loss_dict.keys()):
            loss_dict[k] /= num_updates
        for key in self.keys:
            self.storages[key].clear()
        return loss_dict

    def _update_algo_step(
        self,
        minibatch: dict[str, torch.Tensor],
        loss_dict: dict[str, float],
        key: str,
    ) -> dict[str, float]:
        return self._update_dagger(minibatch, loss_dict, key)

    def _update_dagger(
        self,
        policy_state_dict: dict[str, torch.Tensor],
        loss_dict: dict[str, float],
        key: str,
    ) -> dict[str, float]:
        gt_actions_batch = policy_state_dict["actions_teacher"]
        self.actors[key].act(policy_state_dict)
        mu_batch = self.actors[key].action_mean

        bc_loss = self._compute_loss(mu_batch, gt_actions_batch, actor_key=key, policy_state_dict=policy_state_dict)
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

    def _update_learning_rate(self, iteration: int) -> None:
        if not hasattr(self, "schedulers"):
            return
        p = self.config.warmup_params
        total_sched = self.config.num_scheduler_iterations
        for key in self.keys:
            if self.config.use_warmup and iteration < p.warmup_steps:
                self.warmup_schedulers[key].step()
                if self.use_critic and self.critic_warmup_schedulers.get(key) is not None:
                    self.critic_warmup_schedulers[key].step()
                continue
            if total_sched is not None and iteration >= total_sched:
                continue
            if self.config.scheduler_type == "ReduceLROnPlateau":
                if hasattr(self, "_last_bc_losses") and key in self._last_bc_losses:
                    self.schedulers[key].step(self._last_bc_losses[key])
                    if self.use_critic and self.critic_schedulers.get(key) is not None:
                        self.critic_schedulers[key].step(self._last_bc_losses[key])
            else:
                self.schedulers[key].step()
                if self.use_critic and self.critic_schedulers.get(key) is not None:
                    self.critic_schedulers[key].step()
            self.actor_learning_rates[key] = self.actor_optimizers[key].param_groups[0]["lr"]
            if self.use_critic and self.critic_optimizers[key] is not None:
                self.critic_learning_rates[key] = self.critic_optimizers[key].param_groups[0]["lr"]

    def _reduce_parameters(self, key: str, actor_only: bool = False, critic_only: bool = False) -> None:
        grads: list[torch.Tensor] = []
        if not critic_only:
            grads += [p.grad.view(-1) for p in self.actors[key].parameters() if p.grad is not None]
        if self.use_critic and not actor_only and self.critics[key] is not None:
            grads += [p.grad.view(-1) for p in self.critics[key].parameters() if p.grad is not None]
        if not grads:
            return
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        offset = 0
        if not critic_only:
            for p in self.actors[key].parameters():
                if p.grad is None:
                    continue
                n = p.numel()
                p.grad.data.copy_(all_grads[offset : offset + n].view_as(p.grad))
                offset += n
        if self.use_critic and not actor_only and self.critics[key] is not None:
            for p in self.critics[key].parameters():
                if p.grad is None:
                    continue
                n = p.numel()
                p.grad.data.copy_(all_grads[offset : offset + n].view_as(p.grad))
                offset += n

    # ------------------------------------------------------------------
    # Logging / ONNX export
    # ------------------------------------------------------------------

    def _post_epoch_logging(self, it: int, loss_dict: dict[str, float]) -> None:
        extra = {"Policy": {}}
        for key in self.keys:
            loss_dict[f"actor_learning_rate_{key}"] = self.actor_learning_rates[key]
            if self.use_critic and self.critic_learning_rates.get(key) is not None:
                loss_dict[f"critic_learning_rate_{key}"] = self.critic_learning_rates[key]
        self.logging_helper.post_epoch_logging(it=it, loss_dict=loss_dict, extra_log_dicts=extra)

    @property
    def actor_onnx_wrapper(self):
        keys = self.keys
        actors = self.actors

        class _DecoupledActorOnnx(nn.Module):
            def __init__(self):
                super().__init__()
                self.actors = nn.ModuleDict({k: actors[k] for k in keys})
                self.keys = keys

            def forward(self, *actor_obs_list: torch.Tensor) -> torch.Tensor:
                out = []
                for i, key in enumerate(self.keys):
                    out.append(self.actors[key].act_inference({"actor_obs": actor_obs_list[i]}))
                return torch.cat(out, dim=1)

        return _DecoupledActorOnnx()

    def export(self, onnx_file_path: str) -> None:
        was_training = self.actors[self.keys[0]].training
        self._eval_mode()
        os.makedirs(os.path.dirname(onnx_file_path), exist_ok=True)

        kp_list, kd_list = get_control_gains_from_config(self.env.robot_config)
        cmd_ranges = get_command_ranges_from_env(self.env)
        urdf_file_path, urdf_str = get_urdf_text_from_robot_config(self.env.robot_config)
        meta = {
            "dof_names": self.env.robot_config.dof_names,
            "kp": kp_list,
            "kd": kd_list,
            "command_ranges": cmd_ranges,
            "robot_urdf": urdf_str,
            "robot_urdf_path": urdf_file_path,
            "policy": "dagger_ma",
            "body_keys": list(self.keys),
        }
        meta.update(self._checkpoint_metadata(iteration=self.current_learning_iteration))

        example_inputs = []
        input_names = []
        for key in self.keys:
            adim = self._get_obs_dim(self.actor_obs_keys[key])
            example_inputs.append(torch.zeros(1, adim, device=self.device))
            input_names.append(f"actor_obs_{key}")

        wrapper = self.actor_onnx_wrapper
        torch.onnx.export(
            wrapper,
            tuple(example_inputs),
            onnx_file_path,
            verbose=False,
            input_names=input_names,
            output_names=["action"],
            opset_version=14,
            dynamo=False,
        )
        attach_onnx_metadata(onnx_path=onnx_file_path, metadata=meta)
        self.logging_helper.save_to_wandb(onnx_file_path)

        if was_training:
            self._train_mode()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

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
                "actions": torch.zeros(
                    self.env.num_envs,
                    sum(self.num_act[k] for k in self.keys),
                    device=self.device,
                ),
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

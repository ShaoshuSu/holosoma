"""Single-actor DAgger.

Ported from FAR-FALCON-Terrains ``humanoidverse/agents/dagger/dagger.py`` and
adapted to Holosoma's config types, observation manager, and rollout storage
APIs.  U-Net heightmap reconstruction is intentionally omitted.
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
from holosoma.agents.modules.module_utils import setup_student_teacher_actor_module
from holosoma.config_types.algo import DAggerConfig
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.utils.helpers import instantiate
from holosoma.utils.inference_helpers import (
    attach_onnx_metadata,
    export_policy_as_onnx,
    get_command_ranges_from_env,
    get_control_gains_from_config,
    get_urdf_text_from_robot_config,
)

console = Console()

_SUPPORTED_LOSS_TYPES = ("l1_norm", "l2_norm", "mse", "mae", "bce")


class DAgger(BaseAlgo):
    """One student distilled from one teacher via behaviour cloning."""

    config: DAggerConfig

    def __init__(
        self,
        env: BaseTask,
        config: DAggerConfig,
        log_dir: str,
        device: str = "cpu",
        multi_gpu_cfg: dict | None = None,
    ):
        if not isinstance(config, DAggerConfig):
            raise TypeError(f"DAgger requires DAggerConfig, got {type(config)}")
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

        self.num_act = self.env.robot_config.actions_dim

        if self.config.loss_type not in _SUPPORTED_LOSS_TYPES:
            raise ValueError(
                f"Unsupported loss type: {self.config.loss_type}. Supported: {_SUPPORTED_LOSS_TYPES}"
            )
        self.loss_type = self.config.loss_type

        self.use_encoder_loss = self.config.use_encoder_loss
        self.encoder_loss_coef = self.config.encoder_loss_coef

        self.actor_learning_rate = self.config.actor_learning_rate

    def _init_obs_keys(self) -> None:
        md = self.config.module_dict
        self.actor_obs_keys: list[str] = list(md.actor.input_dim)
        self.actor_teacher_obs_keys: list[str] = list(md.actor_teacher.input_dim)
        if self.use_encoder_loss and md.actor_encoder_teacher is not None:
            self.actor_encoder_teacher_obs_keys: list[str] | None = list(md.actor_encoder_teacher.input_dim)
        else:
            self.actor_encoder_teacher_obs_keys = None

    def _history_for_module(self, obs_keys: list[str]) -> dict[str, int]:
        return {k: self.env.observation_manager.cfg.groups[k].history_length for k in obs_keys}

    def _get_obs_dim(self, obs_keys: list[str]) -> int:
        obs_dim = 0
        for obs_key in obs_keys:
            key_dim = self.algo_obs_dim_dict[obs_key]
            assert isinstance(key_dim, int), f"Observation dimension for {obs_key} is not an integer: {key_dim}"
            obs_dim += key_dim
        return obs_dim

    def _get_zero_input(self) -> torch.Tensor:
        return torch.zeros(1, self._get_obs_dim(self.actor_obs_keys), device=self.device)

    def setup(self) -> None:
        logger.info("Setting up DAgger")
        self._setup_models_and_optimizer()
        logger.info("Setting up rollout storage")
        self._setup_storage()

    def _setup_models_and_optimizer(self) -> None:
        md = self.config.module_dict
        self.actor = setup_student_teacher_actor_module(
            obs_dim_dict=self.algo_obs_dim_dict,
            module_config=md.actor,
            num_actions=self.num_act,
            init_noise_std=self.config.init_noise_std,
            device=self.device,
            history_length=self._history_for_module(list(md.actor.input_dim)),
        )
        self.actor_teacher = setup_student_teacher_actor_module(
            obs_dim_dict=self.algo_obs_dim_dict,
            module_config=md.actor_teacher,
            num_actions=self.num_act,
            init_noise_std=self.config.init_noise_std,
            device=self.device,
            history_length=self._history_for_module(list(md.actor_teacher.input_dim)),
        )

        self.actor_encoder_teacher: nn.Module | None = None
        if self.use_encoder_loss and md.actor_encoder_teacher is not None:
            self.actor_encoder_teacher = setup_student_teacher_actor_module(
                obs_dim_dict=self.algo_obs_dim_dict,
                module_config=md.actor_encoder_teacher,
                num_actions=self.num_act,
                init_noise_std=self.config.init_noise_std,
                device=self.device,
                history_length=self._history_for_module(list(md.actor_encoder_teacher.input_dim)),
            )

        if self.config.ckpt_teacher is not None:
            self.load_teacher(self.config.ckpt_teacher)
            if self.use_encoder_loss and self.config.ckpt_encoder_teacher is not None:
                self.load_encoder_teacher(self.config.ckpt_encoder_teacher)
        self.actor_teacher.eval()
        if self.actor_encoder_teacher is not None:
            self.actor_encoder_teacher.eval()

        if self.is_multi_gpu:
            self._synchronize_model_weights()

        self.actor_optimizer = instantiate(
            self.config.actor_optimizer,
            params=self.actor.parameters(),
            lr=self.actor_learning_rate,
        )

        if self.config.use_scheduler:
            self._setup_scheduler()
        if self.config.use_warmup:
            self._setup_warmup()

    def _setup_scheduler(self) -> None:
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
            "OneCycleLR": {"max_lr": self.actor_learning_rate, "total_steps": total_iters},
        }
        scheduler_cls_map = {
            "StepLR": lr_scheduler.StepLR,
            "ExponentialLR": lr_scheduler.ExponentialLR,
            "CosineAnnealingLR": lr_scheduler.CosineAnnealingLR,
            "ReduceLROnPlateau": lr_scheduler.ReduceLROnPlateau,
            "OneCycleLR": lr_scheduler.OneCycleLR,
        }
        stype = self.config.scheduler_type
        if stype not in scheduler_cls_map:
            raise ValueError(f"Unsupported scheduler type: {stype}")
        params = {**defaults.get(stype, {}), **dict(self.config.scheduler_params)}
        self.scheduler = scheduler_cls_map[stype](self.actor_optimizer, **params)

    def _setup_warmup(self) -> None:
        p = self.config.warmup_params
        self.warmup_scheduler = lr_scheduler.LinearLR(
            self.actor_optimizer,
            start_factor=p.warmup_start_factor,
            end_factor=p.warmup_end_factor,
            total_iters=p.warmup_steps,
        )

    def _synchronize_model_weights(self) -> None:
        for param in self.actor.parameters():
            torch.distributed.broadcast(param.data, src=0)
        for param in self.actor_teacher.parameters():
            torch.distributed.broadcast(param.data, src=0)
        if self.actor_encoder_teacher is not None:
            for param in self.actor_encoder_teacher.parameters():
                torch.distributed.broadcast(param.data, src=0)
        logger.info(f"Synchronized DAgger weights across {self.gpu_world_size} GPUs")

    def _setup_storage(self) -> None:
        self.storage = RolloutStorage(self.env.num_envs, self.config.num_steps_per_env, device=self.device)

        a_dim = self._get_obs_dim(self.actor_obs_keys)
        t_dim = self._get_obs_dim(self.actor_teacher_obs_keys)
        console.print(f"[DAgger] actor_obs={a_dim} actor_teacher_obs={t_dim}")

        self.storage.register("actor_obs", shape=(a_dim,), dtype=torch.float)
        self.storage.register("actor_teacher_obs", shape=(t_dim,), dtype=torch.float)

        if self.use_encoder_loss and self.actor_encoder_teacher_obs_keys is not None:
            e_dim = self._get_obs_dim(self.actor_encoder_teacher_obs_keys)
            self.storage.register("actor_encoder_teacher_obs", shape=(e_dim,), dtype=torch.float)

        for name, shape, dtype in [
            ("actions", (self.num_act,), torch.float),
            ("actions_teacher", (self.num_act,), torch.float),
            ("rewards", (1,), torch.float),
            ("dones", (1,), torch.bool),
        ]:
            self.storage.register(name, shape=shape, dtype=dtype)

        if self.use_encoder_loss:
            actor_module = getattr(self.actor, "actor_module", None)
            enc_dim = int(getattr(actor_module, "encoder_output_dim", 0) or 0)
            if enc_dim > 0:
                self.storage.register("encoder_output", shape=(enc_dim,), dtype=torch.float)
                self.storage.register("encoder_output_teacher", shape=(enc_dim,), dtype=torch.float)

    # ------------------------------------------------------------------
    # Mode helpers
    # ------------------------------------------------------------------

    def _eval_mode(self) -> None:
        self.actor.eval()

    def _train_mode(self) -> None:
        self.actor.train()

    # ------------------------------------------------------------------
    # Load / save
    # ------------------------------------------------------------------

    def load(self, ckpt_path: str | None) -> dict | None:
        if ckpt_path is None:
            return None
        logger.info(f"Loading checkpoint from {ckpt_path}")
        loaded = torch.load(ckpt_path, map_location=self.device)
        self.actor.load_state_dict(loaded["actor_model_state_dict"])
        if self.config.load_optimizer:
            self.actor_optimizer.load_state_dict(loaded["actor_optimizer_state_dict"])
            self.actor_learning_rate = loaded["actor_optimizer_state_dict"]["param_groups"][0]["lr"]
        self.current_learning_iteration = loaded["iter"]
        self._restore_env_state(loaded.get("env_state"))
        return loaded.get("infos")

    def load_teacher(self, ckpt_path: str) -> dict | None:
        logger.info(f"Loading teacher checkpoint from {ckpt_path}")
        loaded = torch.load(ckpt_path, map_location=self.device)
        self.actor_teacher.load_state_dict(loaded["actor_model_state_dict"])
        return loaded.get("infos")

    def load_encoder_teacher(self, ckpt_path: str) -> dict | None:
        logger.info(f"Loading encoder teacher checkpoint from {ckpt_path}")
        loaded = torch.load(ckpt_path, map_location=self.device)
        self.actor_encoder_teacher.load_state_dict(loaded["actor_model_state_dict"])
        return loaded.get("infos")

    def save(self, path: str, infos=None) -> None:
        d: dict[str, Any] = {
            "actor_model_state_dict": self.actor.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
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

    def _rollout_step(self, obs_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        with torch.inference_mode():
            for _ in range(self.config.num_steps_per_env):
                actor_obs = torch.cat([obs_dict[k] for k in self.actor_obs_keys], dim=1)
                actor_teacher_obs = torch.cat([obs_dict[k] for k in self.actor_teacher_obs_keys], dim=1)

                actions = self.actor.act({"actor_obs": actor_obs})
                actions_teacher = self.actor_teacher.act({"actor_obs": actor_teacher_obs})

                add_kwargs: dict[str, torch.Tensor] = {
                    "actor_obs": actor_obs,
                    "actor_teacher_obs": actor_teacher_obs,
                    "actions": actions,
                    "actions_teacher": actions_teacher,
                }

                if self.use_encoder_loss:
                    enc_student = self.actor.get_encoder_output()
                    if enc_student is not None:
                        add_kwargs["encoder_output"] = enc_student
                        if self.actor_encoder_teacher is not None and self.actor_encoder_teacher_obs_keys is not None:
                            enc_obs = torch.cat(
                                [obs_dict[k] for k in self.actor_encoder_teacher_obs_keys], dim=1
                            )
                            self.actor_encoder_teacher.act({"actor_obs": enc_obs})
                            add_kwargs["encoder_output_teacher"] = self.actor_encoder_teacher.get_encoder_output()
                            add_kwargs["actor_encoder_teacher_obs"] = enc_obs
                        else:
                            add_kwargs["encoder_output_teacher"] = self.actor_teacher.get_encoder_output()

                obs_dict, rewards, dones, infos = self.env.step({"actions": actions})

                for obs_key in obs_dict:
                    obs_dict[obs_key] = obs_dict[obs_key].to(self.device)
                rewards = rewards.to(self.device)
                dones = dones.to(self.device)

                add_kwargs["rewards"] = rewards.view(-1, 1)
                add_kwargs["dones"] = dones.view(-1, 1)
                self.storage.add(**add_kwargs)

                self.actor.reset(dones)
                self.actor_teacher.reset(dones)
                if self.actor_encoder_teacher is not None:
                    self.actor_encoder_teacher.reset(dones)

                if self.log_dir is not None and self.is_main_process:
                    self.logging_helper.update_episode_stats(rewards, dones, infos)

        return obs_dict

    # ------------------------------------------------------------------
    # Losses / updates
    # ------------------------------------------------------------------

    def _compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        error = predictions - targets
        if self.loss_type == "l1_norm":
            return error.abs().sum(dim=-1).mean()
        if self.loss_type == "l2_norm":
            return torch.sqrt((error**2).sum(dim=-1)).mean()
        if self.loss_type == "mse":
            return (error**2).mean()
        if self.loss_type == "mae":
            return error.abs().mean()
        if self.loss_type == "bce":
            return torch.nn.functional.binary_cross_entropy_with_logits(predictions, targets)
        raise ValueError(f"Unsupported loss type: {self.loss_type}")

    def _training_step(self) -> dict[str, float]:
        loss_dict: dict[str, float] = {"BC": 0.0}
        if self.use_encoder_loss:
            loss_dict["Encoder"] = 0.0

        generator = self.storage.mini_batch_generator(self.config.num_mini_batches, self.config.num_learning_epochs)
        for minibatch in generator:
            gt_actions = minibatch["actions_teacher"]
            self.actor.act(minibatch)
            mu = self.actor.action_mean

            bc_loss = self._compute_loss(mu, gt_actions)
            loss = self.config.bc_loss_coef * bc_loss

            enc_loss: torch.Tensor | None = None
            if self.use_encoder_loss and "encoder_output" in minibatch:
                enc_loss = self._compute_loss(
                    minibatch["encoder_output"], minibatch["encoder_output_teacher"]
                )
                loss = loss + self.encoder_loss_coef * enc_loss

            self.actor_optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self._reduce_parameters()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
            self.actor_optimizer.step()

            loss_dict["BC"] += bc_loss.item()
            if enc_loss is not None:
                loss_dict["Encoder"] += enc_loss.item()

        num_updates = self.config.num_learning_epochs * self.config.num_mini_batches
        for k in list(loss_dict.keys()):
            loss_dict[k] /= num_updates

        if self.config.use_scheduler and self.config.scheduler_type == "ReduceLROnPlateau":
            self._last_bc_loss = loss_dict["BC"]

        self.storage.clear()
        return loss_dict

    def _update_learning_rate(self, iteration: int) -> None:
        if not hasattr(self, "scheduler"):
            return
        p = self.config.warmup_params
        total_sched = self.config.num_scheduler_iterations
        if self.config.use_warmup and iteration < p.warmup_steps:
            self.warmup_scheduler.step()
        elif total_sched is not None and iteration >= total_sched:
            pass  # freeze LR past the schedule window
        elif self.config.scheduler_type == "ReduceLROnPlateau":
            if hasattr(self, "_last_bc_loss"):
                self.scheduler.step(self._last_bc_loss)
        else:
            self.scheduler.step()
        self.actor_learning_rate = self.actor_optimizer.param_groups[0]["lr"]

    def _reduce_parameters(self) -> None:
        grads = [p.grad.view(-1) for p in self.actor.parameters() if p.grad is not None]
        if not grads:
            return
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        offset = 0
        for p in self.actor.parameters():
            if p.grad is None:
                continue
            n = p.numel()
            p.grad.data.copy_(all_grads[offset : offset + n].view_as(p.grad))
            offset += n

    # ------------------------------------------------------------------
    # Logging / ONNX export
    # ------------------------------------------------------------------

    def _post_epoch_logging(self, it: int, loss_dict: dict[str, float]) -> None:
        loss_dict["actor_learning_rate"] = self.actor_learning_rate
        self.logging_helper.post_epoch_logging(it=it, loss_dict=loss_dict, extra_log_dicts={"Policy": {}})

    @property
    def actor_onnx_wrapper(self):
        actor = self.actor

        class ActorWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.actor = actor

            def forward(self, actor_obs: torch.Tensor) -> torch.Tensor:
                return self.actor.act_inference({"actor_obs": actor_obs})

        return ActorWrapper()

    def export(self, onnx_file_path: str) -> None:
        was_training = self.actor.training
        self._eval_mode()
        os.makedirs(os.path.dirname(onnx_file_path), exist_ok=True)

        export_policy_as_onnx(
            wrapper=self.actor_onnx_wrapper,
            onnx_file_path=onnx_file_path,
            example_obs_dict={"actor_obs": self._get_zero_input()},
        )

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
            "policy": "dagger",
        }
        meta.update(self._checkpoint_metadata(iteration=self.current_learning_iteration))
        attach_onnx_metadata(onnx_path=onnx_file_path, metadata=meta)
        self.logging_helper.save_to_wandb(onnx_file_path)

        if was_training:
            self._train_mode()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def get_inference_policy(self, device: str | None = None):
        self.actor.eval()
        if device is not None:
            self.actor.to(device)
        return self.actor.act_inference

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
                "actions": torch.zeros(self.env.num_envs, self.num_act, device=self.device),
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
        actor_obs = torch.cat([actor_state["obs"][k] for k in self.actor_obs_keys], dim=1)
        actions = self.eval_policy({"actor_obs": actor_obs})
        actor_state.update({"actions": actions})
        for c in self.eval_callbacks:
            actor_state = c.on_pre_eval_env_step(actor_state)
        return actor_state

    def _post_eval_env_step(self, actor_state: dict[str, Any]) -> dict[str, Any]:
        for c in self.eval_callbacks:
            actor_state = c.on_post_eval_env_step(actor_state)
        return actor_state

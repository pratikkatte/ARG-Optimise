import math
import os
import tempfile
import hashlib

import numpy as np
import torch

try:
    from .models import ARGModel
    from .lr_control import LearningRateController
    from .env import (
        CoalescenceChoice,
        FixedAttachmentChoice,
        RecombinationChoice,
    )
except ImportError:  # Support the repository's script-style entry points.
    from models import ARGModel
    from lr_control import LearningRateController
    from env import (
        CoalescenceChoice,
        FixedAttachmentChoice,
        RecombinationChoice,
    )
from dataclasses import replace

LOSS_FN = {
    'MSE': torch.nn.MSELoss(),
    'HUBER': torch.nn.HuberLoss(delta=1.0),
}
LOSS_MODES = {"tb", "subtb", "fl_subtb"}
CHECKPOINT_FORMAT_VERSION = 2

class TBGFlowNetGenerator(torch.nn.Module):
    def __init__(
        self,
        env,
        init_z_sample_count,
        cfg=None,
        device=None,
        verbose=True,
        arg_model_lr=0.001,
        z_lr=0.001,
        grad_clip=10.0,
        model_kwargs=None,
        policy_lr=None,
        breakpoint_policy_lr=None,
        time_policy_lr=None,
        log_z_lr=None,
        initialize_z_from_prior=True,
        loss_mode="tb",
        subtb_lambda=0.9,
        subtb_max_span=None,
        terminal_loss_weight=1.0,
        residual_scale=1.0,
        subtb_lambda_initial=None,
        subtb_lambda_final=None,
        subtb_max_span_schedule=None,
        breakpoint_gradient_clip_norm=None,
        time_head_gradient_clip_norm=None,
        time_head_warmup_epochs=0,
        model_diagnostics=True,
        model_diagnostics_update_norm_every=1,
        flow_debug=False,
        flow_debug_max_records=16,
        probability_checks=False,
        lr_scheduler_config=None,
        total_training_steps=None,
    ):
        super().__init__()
        print(f"verbose: {verbose}")
        self.env = env
        self.verbose = verbose
        self.device = torch.device(device) if device is not None else torch.device(env.device)
        self.env.device = self.device
        if hasattr(self.env, "seq_arrays"):
            self.env.seq_arrays = torch.nn.Parameter(
                self.env.seq_arrays.detach().to(self.device),
                requires_grad=False,
            )
        if hasattr(self.env, "block_seq_arrays"):
            self.env.block_seq_arrays = torch.nn.Parameter(
                self.env.block_seq_arrays.detach().to(self.device),
                requires_grad=False,
            )
        for attr in (
            "variant_position_tensor",
            "variant_boundary_tensor",
            "variant_prev_gap_tensor",
            "variant_next_gap_tensor",
        ):
            if hasattr(self.env, attr):
                value = getattr(self.env, attr)
                setattr(
                    self.env,
                    attr,
                    torch.nn.Parameter(value.detach().to(self.device), requires_grad=False),
                )
        self.init_z_sample_count = int(init_z_sample_count)

        ## Policy model
        if policy_lr is not None:
            arg_model_lr = policy_lr
        if log_z_lr is not None:
            z_lr = log_z_lr
        self.arg_model_lr = float(arg_model_lr)
        self.breakpoint_policy_lr = (
            self.arg_model_lr
            if breakpoint_policy_lr is None
            else float(breakpoint_policy_lr)
        )
        if self.breakpoint_policy_lr <= 0.0:
            raise ValueError("breakpoint_policy_lr must be positive")
        self.time_policy_lr = (
            self.arg_model_lr
            if time_policy_lr is None
            else float(time_policy_lr)
        )
        if self.time_policy_lr <= 0.0:
            raise ValueError("time_policy_lr must be positive")
        self.z_lr = float(z_lr)
        self.loss_mode = str(loss_mode).lower()
        if self.loss_mode not in LOSS_MODES:
            raise ValueError(f"loss_mode must be one of {sorted(LOSS_MODES)}, got {loss_mode!r}")
        self.subtb_lambda = float(subtb_lambda)
        if self.subtb_lambda <= 0.0:
            raise ValueError(f"subtb_lambda must be positive, got {subtb_lambda!r}")
        self.subtb_max_span = (
            None if subtb_max_span is None else int(subtb_max_span)
        )
        if self.subtb_max_span is not None and self.subtb_max_span <= 0:
            raise ValueError(
                f"subtb_max_span must be positive when provided, got {subtb_max_span!r}"
            )
        self.terminal_loss_weight = float(terminal_loss_weight)
        if not math.isfinite(self.terminal_loss_weight) or self.terminal_loss_weight < 0.0:
            raise ValueError("terminal_loss_weight must be finite and nonnegative")
        self.residual_scale = float(residual_scale)
        if not math.isfinite(self.residual_scale) or self.residual_scale <= 0.0:
            raise ValueError("residual_scale must be finite and positive")
        self.subtb_lambda_initial = (
            None if subtb_lambda_initial is None else float(subtb_lambda_initial)
        )
        self.subtb_lambda_final = (
            None if subtb_lambda_final is None else float(subtb_lambda_final)
        )
        if (self.subtb_lambda_initial is None) != (self.subtb_lambda_final is None):
            raise ValueError(
                "subtb_lambda_initial and subtb_lambda_final must be provided together"
            )
        if self.subtb_lambda_initial is not None and (
            self.subtb_lambda_initial <= 0.0 or self.subtb_lambda_final <= 0.0
        ):
            raise ValueError("SubTB curriculum lambdas must be positive")
        self.subtb_max_span_schedule = self._normalize_span_schedule(
            subtb_max_span_schedule
        )
        self.active_subtb_lambda = self.subtb_lambda
        self.active_subtb_max_span = self.subtb_max_span
        self.breakpoint_gradient_clip_norm = (
            None
            if breakpoint_gradient_clip_norm is None
            else float(breakpoint_gradient_clip_norm)
        )
        if (
            self.breakpoint_gradient_clip_norm is not None
            and (
                not math.isfinite(self.breakpoint_gradient_clip_norm)
                or self.breakpoint_gradient_clip_norm <= 0.0
            )
        ):
            raise ValueError(
                "breakpoint_gradient_clip_norm must be finite and positive"
            )
        self.time_head_gradient_clip_norm = (
            None
            if time_head_gradient_clip_norm is None
            else float(time_head_gradient_clip_norm)
        )
        if (
            self.time_head_gradient_clip_norm is not None
            and (
                not math.isfinite(self.time_head_gradient_clip_norm)
                or self.time_head_gradient_clip_norm <= 0.0
            )
        ):
            raise ValueError(
                "time_head_gradient_clip_norm must be finite and positive"
            )
        self.time_head_warmup_epochs = int(time_head_warmup_epochs)
        if self.time_head_warmup_epochs < 0:
            raise ValueError("time_head_warmup_epochs must be nonnegative")
        self.model_diagnostics = bool(model_diagnostics)
        self.model_diagnostics_update_norm_every = int(
            model_diagnostics_update_norm_every
        )
        if self.model_diagnostics_update_norm_every <= 0:
            raise ValueError(
                "model_diagnostics_update_norm_every must be positive"
            )
        self.current_epoch = 0
        self.flow_debug = bool(flow_debug)
        self.flow_debug_max_records = int(flow_debug_max_records)
        if self.flow_debug_max_records < 0:
            raise ValueError("flow_debug_max_records must be nonnegative")
        self.probability_checks = bool(probability_checks)
        self.model_kwargs = dict(model_kwargs or {})
        self.arg_model = ARGModel(env, **self.model_kwargs).to(self.device)
        self.time_model = self.arg_model.time_scorer
        self.breakpoint_model = self.arg_model.breakpoint_scorer

        ## Z partition
        self.max_reward_seen = float("-inf")
        if initialize_z_from_prior and self.init_z_sample_count > 0:
            log_rewards = env.sample_log_rewards(self.init_z_sample_count, verbose=verbose)
            self.max_reward_seen = float(np.max(log_rewards))
            init_Z = self.max_reward_seen
        else:
            self.max_reward_seen = 0.0
            init_Z = 0.0
        self._Z = torch.nn.Parameter(  # in log
                torch.ones(256, device=self.device) * init_Z / 256, requires_grad=True
                )
        
        self.arg_model_params = list(self.arg_model.parameters())
        self.policy_params = self.arg_model_params

        self.time_params = list(self.time_model.parameters())
        self.breakpoint_params = list(self.breakpoint_model.parameters())
        time_parameter_ids = {id(parameter) for parameter in self.time_params}
        breakpoint_parameter_ids = {
            id(parameter) for parameter in self.breakpoint_params
        }
        self.structural_policy_params = [
            parameter
            for parameter in self.arg_model_params
            if id(parameter) not in time_parameter_ids
            and id(parameter) not in breakpoint_parameter_ids
        ]
        # Retain the compatibility attribute, but give it the corrected
        # structural-only meaning now that breakpoint parameters are separate.
        self.non_time_policy_params = self.structural_policy_params
        self.model_parameter_groups = {
            "structural": self.structural_policy_params,
            "breakpoint": self.breakpoint_params,
            "time": self.time_params,
        }
        params = [
            {
                'params': self.structural_policy_params,
                'lr': self.arg_model_lr,
            },
            {
                'params': self.breakpoint_params,
                'lr': self.breakpoint_policy_lr,
            },
            {
                'params': self.time_params,
                'lr': self.time_policy_lr,
            },
        ]
        optimizer_group_names = [
            "structural_policy",
            "breakpoint_policy",
            "time_policy",
        ]
        if self.loss_mode == "tb":
            params.append({'params': [self._Z], 'lr': self.z_lr})
            optimizer_group_names.append("log_z")

        # gradient clipping exclude the Z part
        self.gradient_clipping_params = list(self.arg_model.parameters())
        self.grad_clip = float(grad_clip)

        self.opt = torch.optim.Adam(
            params,
            weight_decay=0.0,
            betas=(0.9, 0.999),
            amsgrad=True,
        )

        self.scheduler = None
        self.optimizer_steps = 0
        if lr_scheduler_config is not None:
            if total_training_steps is None:
                raise ValueError(
                    "total_training_steps is required when lr_scheduler_config is provided"
                )
            self.scheduler = LearningRateController(
                self.opt,
                group_names=optimizer_group_names,
                total_training_steps=int(total_training_steps),
                config=lr_scheduler_config,
            )

        self.loss_fn = LOSS_FN['MSE']

        self.grad_norm = lambda model: math.sqrt(sum(
            [p.grad.norm().item() ** 2 for p in self.gradient_clipping_params if p.grad is not None]))
        self.param_norm = lambda model: math.sqrt(sum([p.norm().item() ** 2 for p in self.gradient_clipping_params]))

        # Retained for checkpoint/runtime compatibility; AMP is not otherwise
        # used by this trainer yet.
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.device.type == "cuda",
        )

        self.loss = 0

        self.loss = torch.tensor(0.0, device=self.device)
        self.accumulated_batches = 0
        self.log_z_target_sum = 0.0
        self.log_z_target_count = 0
        self.last_log_z_target = float(self.compute_log_Z().detach().cpu().item())
        self.last_time_subtb_diagnostics = {}
        self.last_balance_diagnostics = {}
        self.last_transition_decomposition = []
        self.last_forward_log_components = {}
        self.last_forward_policy_diagnostics = []
        self._last_balance_details = None
        self._accumulated_balance_records = []
        self._accumulated_internal_loss = 0.0
        self._accumulated_terminal_loss = 0.0
        self.set_training_epoch(0)

    @staticmethod
    def _normalize_span_schedule(schedule):
        if schedule in (None, ()):
            return ()
        normalized = []
        saw_open_end = False
        for index, row in enumerate(schedule):
            if not isinstance(row, dict):
                raise ValueError("each subtb_max_span_schedule row must be a mapping")
            until = row.get("until_epoch")
            if until is not None:
                until = int(until)
                if until <= 0:
                    raise ValueError("schedule until_epoch values must be positive")
                if saw_open_end:
                    raise ValueError("open-ended SubTB schedule row must be last")
            else:
                saw_open_end = True
            value = int(row["value"])
            if value <= 0:
                raise ValueError("scheduled SubTB spans must be positive")
            if normalized and until is not None:
                previous = normalized[-1]["until_epoch"]
                if previous is None or until <= previous:
                    raise ValueError("schedule until_epoch values must increase")
            normalized.append({"until_epoch": until, "value": value})
        return tuple(normalized)

    def set_training_epoch(self, epoch, total_epochs=None):
        """Activate the configured SubTB curriculum and time-head warm-up."""

        self.current_epoch = int(epoch)
        if self.subtb_lambda_initial is None:
            self.active_subtb_lambda = self.subtb_lambda
        else:
            denominator = max(int(total_epochs or 1) - 1, 1)
            progress = min(max(self.current_epoch / denominator, 0.0), 1.0)
            self.active_subtb_lambda = (
                self.subtb_lambda_initial
                + progress * (self.subtb_lambda_final - self.subtb_lambda_initial)
            )
        self.active_subtb_max_span = self.subtb_max_span
        for row in self.subtb_max_span_schedule:
            until = row["until_epoch"]
            if until is None or self.current_epoch < int(until):
                self.active_subtb_max_span = int(row["value"])
                break
        time_trainable = self.current_epoch >= self.time_head_warmup_epochs
        for parameter in self.time_params:
            parameter.requires_grad_(time_trainable)
        return {
            "subtb_active_lambda": float(self.active_subtb_lambda),
            "subtb_active_max_span": self.active_subtb_max_span,
            "time_head_warmup_active": not time_trainable,
        }


    def _encode_states(self, states):
        return self.arg_model._encode_states(states)


    def save(self, path, metadata=None, training_state=None):
        """Atomically save a portable training/inference checkpoint.

        Model metadata is kept separate from mutable training state so
        inference only needs to inspect ``metadata`` and the model weights.
        ``opt_state_dict`` remains the canonical optimizer key for backward
        compatibility with existing checkpoints.
        """
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = {
            "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
            "generator_state_dict": self.state_dict(),
            "opt_state_dict": self.opt.state_dict(),
            "metadata": dict(metadata or {}),
            "training_state": dict(training_state or {}),
        }
        if self.scheduler is not None:
            payload["scheduler_state_dict"] = self.scheduler.state_dict()
        if self.scaler is not None:
            payload["scaler_state_dict"] = self.scaler.state_dict()

        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=directory,
                prefix=f".{os.path.basename(path)}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = handle.name
            torch.save(payload, temporary_path)
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None and os.path.exists(temporary_path):
                os.unlink(temporary_path)

    def load(self, path, load_optimizer=True, map_location=None):
        if map_location is None:
            map_location = self.device
        checkpoint = (
            path
            if isinstance(path, dict)
            else self._torch_load(path, map_location=map_location)
        )
        metadata = dict(checkpoint.get("metadata") or {})
        if (
            any(
                key in metadata
                for key in (
                    "time_bin_scheme",
                    "time_bins",
                    "time_delta_bin_width",
                )
            )
            or str(metadata.get("model_version", "")).endswith("-v1")
        ):
            raise ValueError(
                "Fixed-bin v1 checkpoints are incompatible with "
                "continuous-time v2 and must be retrained."
            )
        state_dict = checkpoint.get("generator_state_dict", checkpoint)
        load_result = self.load_state_dict(state_dict, strict=False)
        allowed_missing = [
            key for key in load_result.missing_keys
            if (
                key.startswith("arg_model.flow_head.")
                or key.startswith("arg_model.local_")
            )
        ]
        unexpected = list(load_result.unexpected_keys)
        non_flow_missing = [
            key for key in load_result.missing_keys
            if key not in allowed_missing
        ]
        if non_flow_missing or unexpected:
            raise RuntimeError(
                "Checkpoint state_dict is incompatible with this generator: "
                f"missing={non_flow_missing}, unexpected={unexpected}"
            )
        self.to(self.device)
        self.last_log_z_target = float(self.compute_log_Z().detach().cpu().item())

        if load_optimizer and "opt_state_dict" in checkpoint:
            self.opt.load_state_dict(checkpoint["opt_state_dict"])
            self._move_optimizer_state_to_device()
        if (
            load_optimizer
            and self.scheduler is not None
            and "scheduler_state_dict" in checkpoint
        ):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.optimizer_steps = int(self.scheduler.optimizer_steps)
        if load_optimizer and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        return checkpoint.get("metadata", {})

    def _move_optimizer_state_to_device(self):
        for state in self.opt.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(self.device)

    def _torch_load(self, path, map_location=None):
        try:
            return torch.load(path, map_location=map_location, weights_only=True)
        except TypeError:
            return torch.load(path, map_location=map_location)

    @staticmethod
    def _grad_norm(params):
        return math.sqrt(sum(
            p.grad.detach().norm().item() ** 2 for p in params if p.grad is not None
        ))

    @staticmethod
    def _param_norm(params):
        return math.sqrt(sum(p.detach().norm().item() ** 2 for p in params))

    @staticmethod
    def _structural_action_diagnostics(logits, candidates, selected_index):
        """Summarize the categorical policy over valid structural actions."""

        with torch.no_grad():
            valid_logits = logits[:len(candidates)]
            log_probabilities = torch.log_softmax(valid_logits, dim=0)
            probabilities = torch.exp(log_probabilities)
            entropy = -(probabilities * log_probabilities).sum()
            support_size = int(len(candidates))
            normalized_entropy = (
                entropy / math.log(support_size)
                if support_size > 1
                else entropy.new_tensor(0.0)
            ).clamp(0.0, 1.0)
            coal_indices = [
                index
                for index, candidate in enumerate(candidates)
                if isinstance(candidate, CoalescenceChoice)
            ]
            recomb_indices = [
                index
                for index, candidate in enumerate(candidates)
                if isinstance(candidate, RecombinationChoice)
            ]
            return {
                "valid_coalescence_actions": len(coal_indices),
                "valid_recombination_actions": len(recomb_indices),
                "coalescence_probability_mass": float(
                    probabilities[coal_indices].sum().detach().cpu().item()
                ) if coal_indices else 0.0,
                "recombination_probability_mass": float(
                    probabilities[recomb_indices].sum().detach().cpu().item()
                ) if recomb_indices else 0.0,
                "structural_action_support_size": support_size,
                "structural_action_entropy": float(
                    entropy.detach().cpu().item()
                ),
                "structural_action_normalized_entropy": float(
                    normalized_entropy.detach().cpu().item()
                ),
                "selected_atomic_action_probability": float(
                    probabilities[int(selected_index)].detach().cpu().item()
                ),
                "structural_action_max_probability": float(
                    probabilities.max().detach().cpu().item()
                ),
            }

    @staticmethod
    def _selected_split_diagnostics(
        row,
        record,
        selected_index,
        selected_action=None,
    ):
        output = {
            key: value
            for key, value in dict(row or {}).items()
            if key != "recombination_split_atomic_adjustments"
        }
        adjustments = None if row is None else row.get(
            "recombination_split_atomic_adjustments"
        )
        if adjustments is not None:
            output["recombination_split_selected_atomic_logit_adjustment"] = float(
                adjustments[int(selected_index)].detach().cpu().item()
            )
        if record is not None:
            output["recombination_split_selected_lineage_score"] = float(
                record.lineage_score.detach().cpu().item()
            )
        if bool(output.get("local_cwr_event_gate_enabled", False)):
            if isinstance(selected_action, RecombinationChoice):
                selected_event = "recombination"
                selected_probability = output[
                    "local_cwr_policy_recombination_probability"
                ]
            elif isinstance(selected_action, CoalescenceChoice):
                selected_event = "coalescence"
                selected_probability = output[
                    "local_cwr_policy_coalescence_probability"
                ]
            else:
                raise ValueError(
                    "local CwR event diagnostics require a generated action"
                )
            output["local_cwr_selected_event"] = selected_event
            output["local_cwr_selected_event_probability"] = float(
                selected_probability
            )
        return output

    @staticmethod
    def _selected_breakpoint_split_diagnostics(record, breakpoint):
        if record is None:
            return {}
        return {
            "recombination_split_selected_breakpoint_score": float(
                record.selected_score(int(breakpoint)).detach().cpu().item()
            )
        }

    @staticmethod
    def _model_group_health(name, params, stage):
        params = list(params)
        gradients = [
            parameter.grad.detach()
            for parameter in params
            if parameter.grad is not None
        ]
        parameter_count = sum(parameter.numel() for parameter in params)
        trainable_count = sum(
            parameter.numel() for parameter in params if parameter.requires_grad
        )
        gradient_count = sum(gradient.numel() for gradient in gradients)
        parameter_finite_count = sum(
            int(torch.isfinite(parameter.detach()).sum().cpu().item())
            for parameter in params
        )
        finite_count = sum(
            int(torch.isfinite(gradient).sum().detach().cpu().item())
            for gradient in gradients
        )
        zero_count = sum(
            int((gradient == 0).sum().detach().cpu().item())
            for gradient in gradients
        )
        parameter_norm = TBGFlowNetGenerator._param_norm(params)
        gradient_norm = TBGFlowNetGenerator._grad_norm(params)
        finite_rate = finite_count / max(gradient_count, 1)
        zero_rate = zero_count / max(gradient_count, 1)
        parameter_finite_rate = parameter_finite_count / max(parameter_count, 1)
        prefix = f"models/{name}"
        result = {
            f"{prefix}/parameter_count": int(parameter_count),
            f"{prefix}/trainable_parameter_count": int(trainable_count),
            f"{prefix}/gradient_element_count": int(gradient_count),
            f"{prefix}/gradient_present": bool(gradient_count),
            f"{prefix}/gradient_finite_rate": float(finite_rate),
            f"{prefix}/gradient_nonfinite_detected": bool(
                gradient_count and finite_count != gradient_count
            ),
            f"{prefix}/gradient_zero_rate": float(zero_rate),
            f"{prefix}/parameter_finite_rate": float(parameter_finite_rate),
            f"{prefix}/parameter_nonfinite_detected": bool(
                parameter_finite_count != parameter_count
            ),
            f"{prefix}/param_norm": float(parameter_norm),
            f"{prefix}/grad_norm_{stage}": float(gradient_norm),
        }
        if stage == "before_clip":
            result[f"{prefix}/grad_to_param_ratio"] = float(
                gradient_norm / max(parameter_norm, 1e-12)
            )
        return result

    def _snapshot_model_parameters(self):
        return {
            name: [parameter.detach().clone() for parameter in parameters]
            for name, parameters in self.model_parameter_groups.items()
        }

    def _model_update_metrics(self, snapshots, parameter_norms):
        result = {}
        for name, parameters in self.model_parameter_groups.items():
            before = snapshots[name]
            squared_update = sum(
                float(
                    (parameter.detach() - old_value)
                    .float()
                    .square()
                    .sum()
                    .cpu()
                    .item()
                )
                for parameter, old_value in zip(parameters, before)
            )
            update_norm = math.sqrt(squared_update)
            parameter_norm = float(parameter_norms[name])
            prefix = f"models/{name}"
            result[f"{prefix}/update_norm"] = float(update_norm)
            result[f"{prefix}/relative_update_norm"] = float(
                update_norm / max(parameter_norm, 1e-12)
            )
            result[f"{prefix}/update_finite"] = bool(
                math.isfinite(update_norm)
            )
            result[f"{prefix}/update_applied"] = bool(
                math.isfinite(update_norm) and update_norm > 0.0
            )
        return result

    def grad_norm(self):
        return self._grad_norm(self.gradient_clipping_params)
    
    def param_norm(self):
        return self._param_norm(self.gradient_clipping_params)

    def policy_grad_norm(self):
        return self._grad_norm(self.policy_params)

    def policy_param_norm(self):
        return self._param_norm(self.policy_params)

    def log_z_grad(self):
        if self._Z.grad is None:
            return 0.0
        return float(self._Z.grad.detach().cpu().reshape(-1)[0].item())

    def log_z_grad_norm(self):
        return self._grad_norm([self._Z])

    def compute_log_Z(self, scale_key=None):
        return self._Z.sum()

    def compute_log_state_flows(self, states):
        if not states:
            return torch.empty(0, dtype=self._model_dtype(), device=self.device)

        log_flows = [None] * len(states)
        nonterminal_states = []
        nonterminal_indices = []
        for idx, state in enumerate(states):
            if state.is_done:
                if state.log_reward is None:
                    raise ValueError("Terminal ARGState is missing log_reward")
                log_flows[idx] = torch.tensor(
                    float(state.log_reward),
                    dtype=self._model_dtype(),
                    device=self.device,
                )
            else:
                nonterminal_indices.append(idx)
                nonterminal_states.append(state)

        if nonterminal_states:
            _, summary_reps, _, _ = self._encode_states(nonterminal_states)
            nonterminal_log_flows = self.arg_model.compute_log_state_flows(summary_reps)
            for idx, log_flow in zip(nonterminal_indices, nonterminal_log_flows):
                if self.loss_mode == "fl_subtb":
                    log_flow = log_flow + log_flow.new_tensor(
                        float(getattr(states[idx], "partial_log_reward", 0.0))
                    )
                log_flows[idx] = log_flow

        return torch.stack(log_flows)

    def compute_root_log_flow(self):
        return self.compute_log_state_flows([self.env.get_initial_state()])[0]

    def _model_dtype(self):
        return next(self.arg_model.parameters()).dtype

    def _sample_continuous_times(
        self,
        selected_action_features,
        states,
        selected_actions,
        rates,
        max_deltas,
        random_spec,
    ):
        context_features = self.time_model.context_features(
            rates,
            max_deltas,
            device=self.device,
            dtype=selected_action_features.dtype,
        )
        biological_context, context_diagnostics = (
            self.arg_model.build_time_context(
                states,
                selected_actions,
                max_deltas,
                dtype=selected_action_features.dtype,
            )
        )
        mixture_logits = self.time_model(
            torch.cat(
                [
                    selected_action_features,
                    context_features,
                    biological_context,
                ],
                dim=-1,
            )
        )
        time_quantiles = self.time_model.sample(
            mixture_logits,
            random_spec,
        )
        delta_times = []
        generated_masses = []
        for quantile, rate, max_delta, state in zip(
            time_quantiles.detach().cpu().tolist(),
            rates,
            max_deltas,
            states,
        ):
            delta_time = self.env.time_env.quantile_to_delta(
                float(quantile),
                float(rate),
                max_delta=max_delta,
            )
            boundary_time = (
                None
                if max_delta is None
                else self.env.next_fixed_ancestor_time(state)
            )
            delta_time = self.env.time_env.event_time_after_delta(
                delta_time,
                state.current_time,
                boundary_time,
            ) - float(state.current_time)
            delta_times.append(delta_time)
            generated_masses.append(
                self.env.time_env.generated_probability(
                    float(rate),
                    max_delta=max_delta,
                )
            )
        log_time_pf = self.time_model.log_time_density(
            mixture_logits,
            time_quantiles,
            delta_times,
            rates,
            generated_masses,
            random_spec=random_spec,
        )
        mixture_diagnostics = self.time_model.mixture_diagnostics(
            mixture_logits,
            random_spec=random_spec,
        )
        return (
            time_quantiles,
            delta_times,
            log_time_pf,
            context_diagnostics,
            mixture_diagnostics,
        )

    def forward(self, input_dict):
        if bool(input_dict.get("local_mode", False)):
            return self._forward_local(input_dict)

        states = input_dict.get("states")

        random_spec = input_dict.get("random_spec")
        

        event = input_dict.get("event")
        event_probs = [
            float(event[idx]["probability"])
            for idx in range(len(states))
        ]
        log_event_pf = torch.log(
            torch.tensor(event_probs, dtype=torch.float32, device=self.device)
        )

        all_actions = input_dict.get("input_actions")

        lineage_reps, summary_reps, lineage_seq_features, batch_active_lineage_counts = self._encode_states(states)
        # input_dict = self._move_input_to_device(input_dict)
        ret = self.arg_model(all_actions, lineage_reps, summary_reps, lineage_seq_features, batch_active_lineage_counts, random_spec)

        (
            log_action_pf,
            selected_action_indices,
            choosen_actions,
            choosen_action_features,
            chosen_split_records,
        ) = ret
        if self.probability_checks:
            self._assert_masked_categorical_probabilities(
                self.arg_model.last_action_probability_logits,
                self.arg_model.last_action_valid_mask,
                "global atomic action",
            )

        log_p_breakpoints = []
        policy_diagnostics = []
        action_probability_logits = self.arg_model.last_action_probability_logits
        for idx, candidates in enumerate(all_actions):
            diagnostics = {
                "selected_gate": "generated",
                "selected_gate_probability": float(event_probs[idx]),
            }
            diagnostics.update(
                self._structural_action_diagnostics(
                    action_probability_logits[idx],
                    candidates,
                    selected_action_indices[idx],
                )
            )
            diagnostics.update(
                self._selected_split_diagnostics(
                    self.arg_model.last_action_split_diagnostics[idx],
                    chosen_split_records[idx],
                    selected_action_indices[idx],
                )
            )
            policy_diagnostics.append(diagnostics)
        for idx, chosen_action in enumerate(choosen_actions):
            if isinstance(chosen_action, RecombinationChoice):
                lineage_idx = int(chosen_action.active_lineage_i)
                if getattr(self.env, "input_mode", "dense") == "vcf":
                    lineage_feature = states[idx].active_lineages[lineage_idx]
                else:
                    lineage_feature = lineage_seq_features[idx, lineage_idx]
                breakpoint, log_p_breakpoint = self.breakpoint_model(
                    chosen_action,
                    lineage_feature,
                    int(self.env.sequence_length),
                    int(self.env.num_blocks),
                    action_context=choosen_action_features[idx],
                    random_spec=random_spec,
                    logit_bias=(
                        None
                        if chosen_split_records[idx] is None
                        else chosen_split_records[idx].breakpoint_bias(
                            self.arg_model.recombination_split_bias_config[
                                "breakpoint_weight"
                            ]
                        )
                    ),
                )
                choosen_actions[idx] = replace(chosen_action, breakpoint=breakpoint)
                log_p_breakpoints.append(log_p_breakpoint)
                policy_diagnostics[idx].update(
                    dict(
                        getattr(
                            self.breakpoint_model,
                            "last_sample_diagnostics",
                            {},
                        )
                    )
                )
                policy_diagnostics[idx].update(
                    self._selected_breakpoint_split_diagnostics(
                        chosen_split_records[idx],
                        breakpoint,
                    )
                )
            else:
                log_p_breakpoints.append(torch.tensor(0.0, device=self.device))

        log_breakpoint_pf = torch.stack(log_p_breakpoints)

        selected_action_features = torch.stack(
            choosen_action_features,
            dim=0,
        )
        rates = [
            self.env._total_event_rate(state.rates)
            for state in states
        ]
        (
            time_quantiles,
            delta_times,
            log_time_pf,
            time_context_diagnostics,
            time_mixture_diagnostics,
        ) = (
            self._sample_continuous_times(
                selected_action_features,
                states,
                choosen_actions,
                rates,
                [None] * len(states),
                random_spec,
            )
        )

        for batch_idx, action in enumerate(choosen_actions):
            time_quantile = float(
                time_quantiles[batch_idx].detach().cpu().item()
            )
            choosen_actions[batch_idx] = replace(
                action,
                time_quantile=time_quantile,
                delta_time=float(delta_times[batch_idx]),
                waiting_rate=float(rates[batch_idx]),
                fixed_horizon=None,
                time_log_density=float(
                    log_time_pf[batch_idx].detach().cpu().item()
                ),
                time_policy_entropy=float(
                    time_mixture_diagnostics["entropy"][batch_idx]
                    .detach()
                    .cpu()
                    .item()
                ),
                time_effective_components=float(
                    time_mixture_diagnostics["effective_components"][batch_idx]
                    .detach()
                    .cpu()
                    .item()
                ),
                time_context_diagnostics=dict(
                    time_context_diagnostics[batch_idx]
                ),
            )

        total_log_pf = log_event_pf + log_action_pf + log_breakpoint_pf + log_time_pf

        self.last_forward_log_components = {
            "gate": log_event_pf.detach(),
            "atomic_action": log_action_pf.detach(),
            "breakpoint": log_breakpoint_pf.detach(),
            "time": log_time_pf.detach(),
            "total": total_log_pf.detach(),
        }
        self.last_forward_policy_diagnostics = policy_diagnostics

        log_probs = torch.exp(total_log_pf)
        
        return total_log_pf, log_probs, choosen_actions

    def _local_gate_log_probabilities(
        self,
        summary_reps,
        rollout,
        random_spec=None,
    ):
        gate_logits = self.arg_model.compute_local_gate_logits(summary_reps)
        prior_gate_weight = float(
            getattr(self.arg_model, "local_prior_gate_logit_bias", 0.0)
        )
        if prior_gate_weight != 0.0:
            epsilon = torch.finfo(gate_logits.dtype).tiny
            prior_gate_rows = []
            for decision in rollout:
                generated_mass = (
                    float(decision["generated_prior_mass"])
                    if bool(decision["can_generate"])
                    else 0.0
                )
                fixed_mass = (
                    float(decision["survival_prior_mass"])
                    if bool(decision["can_attach_fixed"])
                    else 0.0
                )
                prior_gate_rows.append(
                    [
                        math.log(max(generated_mass, float(epsilon))),
                        math.log(max(fixed_mass, float(epsilon))),
                    ]
                )
            gate_logits = gate_logits + prior_gate_weight * torch.as_tensor(
                prior_gate_rows,
                dtype=gate_logits.dtype,
                device=self.device,
            )
        gate_mask = torch.tensor(
            [
                [
                    bool(decision["can_generate"]),
                    bool(decision["can_attach_fixed"]),
                ]
                for decision in rollout
            ],
            dtype=torch.bool,
            device=self.device,
        )
        if not bool(gate_mask.any(dim=1).all().detach().cpu().item()):
            raise RuntimeError("local transition gate contains an all-invalid row")
        probability_logits = gate_logits.masked_fill(~gate_mask, float("-inf"))
        if random_spec is not None and "T" in random_spec:
            probability_logits = probability_logits / float(random_spec["T"])
        return torch.log_softmax(probability_logits, dim=1), gate_mask

    def _forward_local(self, input_dict):
        states = input_dict["states"]
        random_spec = input_dict.get("random_spec")
        rollout = input_dict["rollout"]
        all_actions = input_dict["input_actions"]

        (
            lineage_reps,
            summary_reps,
            lineage_seq_features,
            batch_active_lineage_counts,
        ) = self._encode_states(states)

        gate_log_probabilities, gate_mask = self._local_gate_log_probabilities(
            summary_reps,
            rollout,
            random_spec=random_spec,
        )
        gate_actions = torch.distributions.Categorical(
            logits=gate_log_probabilities
        ).sample()
        batch_indices = torch.arange(
            len(states),
            dtype=torch.long,
            device=self.device,
        )
        selected_gate_log_pf = gate_log_probabilities[
            batch_indices,
            gate_actions,
        ]

        chosen_actions = [None] * len(states)
        total_log_pf = selected_gate_log_pf.clone()
        component_action = torch.zeros_like(selected_gate_log_pf)
        component_breakpoint = torch.zeros_like(selected_gate_log_pf)
        component_time = torch.zeros_like(selected_gate_log_pf)
        policy_diagnostics = [
            {
                "gate_generated_probability": float(
                    torch.exp(gate_log_probabilities[index, 0]).detach().cpu().item()
                )
                if bool(gate_mask[index, 0])
                else 0.0,
                "gate_fixed_probability": float(
                    torch.exp(gate_log_probabilities[index, 1]).detach().cpu().item()
                )
                if bool(gate_mask[index, 1])
                else 0.0,
                "selected_gate_probability": float(
                    torch.exp(selected_gate_log_pf[index]).detach().cpu().item()
                ),
                "selected_gate": (
                    "generated" if int(gate_actions[index].detach().cpu().item()) == 0
                    else "fixed_attachment"
                ),
                "valid_coalescence_actions": 0,
                "valid_recombination_actions": 0,
                "coalescence_probability_mass": 0.0,
                "recombination_probability_mass": 0.0,
                "selected_atomic_action_probability": 1.0,
            }
            for index in range(len(states))
        ]
        generated_indices = [
            index
            for index, gate_action in enumerate(
                gate_actions.detach().cpu().tolist()
            )
            if int(gate_action) == 0
        ]
        fixed_indices = [
            index
            for index, gate_action in enumerate(
                gate_actions.detach().cpu().tolist()
            )
            if int(gate_action) == 1
        ]

        for index in fixed_indices:
            event_time = rollout[index]["next_fixed_time"]
            if event_time is None:
                raise RuntimeError(
                    "local gate selected a fixed attachment without a boundary"
                )
            horizon = float(rollout[index]["max_delta"])
            waiting_rate = float(rollout[index]["total_rate"])
            chosen_actions[index] = FixedAttachmentChoice(
                event_time=float(event_time),
                waiting_rate=waiting_rate,
                fixed_horizon=horizon,
                survival_log_probability=-waiting_rate * horizon,
            )

        if generated_indices:
            generated_actions = [
                all_actions[index] for index in generated_indices
            ]
            if any(not actions for actions in generated_actions):
                raise RuntimeError(
                    "local gate selected generation without legal actions"
                )
            index_tensor = torch.tensor(
                generated_indices,
                dtype=torch.long,
                device=self.device,
            )
            generated_lineage_reps = lineage_reps.index_select(
                0,
                index_tensor,
            )
            generated_summary_reps = summary_reps.index_select(
                0,
                index_tensor,
            )
            generated_counts = batch_active_lineage_counts.index_select(
                0,
                index_tensor,
            )
            if torch.is_tensor(lineage_seq_features):
                generated_lineage_features = (
                    lineage_seq_features.index_select(0, index_tensor)
                )
            else:
                generated_lineage_features = [
                    lineage_seq_features[index]
                    for index in generated_indices
                ]
            (
                log_action_pf,
                _selected_action_indices,
                generated_chosen_actions,
                generated_action_features,
                generated_split_records,
            ) = self.arg_model(
                generated_actions,
                generated_lineage_reps,
                generated_summary_reps,
                generated_lineage_features,
                generated_counts,
                random_spec,
                event_rates=[
                    {
                        "lambda_coal": rollout[index]["lambda_coal"],
                        "lambda_recomb": rollout[index]["lambda_recomb"],
                    }
                    for index in generated_indices
                ],
            )
            probability_logits = self.arg_model.last_action_probability_logits
            if self.probability_checks:
                self._assert_masked_categorical_probabilities(
                    probability_logits,
                    self.arg_model.last_action_valid_mask,
                    "local atomic action",
                )
            for local_index, candidates in enumerate(generated_actions):
                state_index = generated_indices[local_index]
                policy_diagnostics[state_index].update(
                    self._structural_action_diagnostics(
                        probability_logits[local_index],
                        candidates,
                        _selected_action_indices[local_index],
                    )
                )
                policy_diagnostics[state_index].update(
                    self._selected_split_diagnostics(
                        self.arg_model.last_action_split_diagnostics[local_index],
                        generated_split_records[local_index],
                        _selected_action_indices[local_index],
                        generated_chosen_actions[local_index],
                    )
                )

            log_breakpoint_pf = []
            for local_index, action in enumerate(
                generated_chosen_actions
            ):
                state_index = generated_indices[local_index]
                state = states[state_index]
                if isinstance(action, RecombinationChoice):
                    lineage_index = int(action.active_lineage_i)
                    lineage = state.active_lineages[lineage_index]
                    valid_breakpoints = self.env.valid_breakpoints(
                        state,
                        action,
                    )
                    breakpoint, log_probability = self.breakpoint_model(
                        valid_breakpoints,
                        lineage,
                        int(self.env.sequence_length),
                        max(len(state.block_boundaries or ()) - 1, 1),
                        action_context=generated_action_features[
                            local_index
                        ],
                        random_spec=random_spec,
                        state=state,
                        logit_bias=self.arg_model.recombination_breakpoint_logit_bias(
                            generated_split_records[local_index],
                            valid_breakpoints,
                        ),
                    )
                    action = replace(action, breakpoint=breakpoint)
                    generated_chosen_actions[local_index] = action
                    log_breakpoint_pf.append(log_probability)
                    policy_diagnostics[state_index].update(
                        dict(
                            getattr(
                                self.breakpoint_model,
                                "last_sample_diagnostics",
                                {},
                            )
                        )
                    )
                    policy_diagnostics[state_index].update(
                        self._selected_breakpoint_split_diagnostics(
                            generated_split_records[local_index],
                            breakpoint,
                        )
                    )
                else:
                    log_breakpoint_pf.append(
                        log_action_pf.new_tensor(0.0)
                    )
            log_breakpoint_pf = torch.stack(log_breakpoint_pf)

            selected_features = torch.stack(
                generated_action_features,
                dim=0,
            )
            rates = [
                float(rollout[index]["total_rate"])
                for index in generated_indices
            ]
            max_deltas = [
                rollout[index]["max_delta"]
                for index in generated_indices
            ]
            (
                time_quantiles,
                delta_times,
                log_time_pf,
                time_context_diagnostics,
                time_mixture_diagnostics,
            ) = (
                self._sample_continuous_times(
                    selected_features,
                    [states[index] for index in generated_indices],
                    generated_chosen_actions,
                    rates,
                    max_deltas,
                    random_spec,
                )
            )

            for local_index, action in enumerate(
                generated_chosen_actions
            ):
                state_index = generated_indices[local_index]
                time_quantile = float(
                    time_quantiles[local_index].detach().cpu().item()
                )
                diagnostics = dict(time_context_diagnostics[local_index])
                sampled_delta = float(delta_times[local_index])
                lower_distance = sampled_delta
                upper_distance = (
                    None
                    if max_deltas[local_index] is None
                    else max(float(max_deltas[local_index]) - sampled_delta, 0.0)
                )
                diagnostics.update(
                    {
                        "sampled_quantile": time_quantile,
                        "sampled_delta_time": sampled_delta,
                        "sampled_event_time": float(states[state_index].current_time)
                        + sampled_delta,
                        "distance_to_lower_bound": lower_distance,
                        "distance_to_upper_bound": upper_distance,
                        "normalized_sample_location": time_quantile,
                    }
                )
                chosen_actions[state_index] = replace(
                    action,
                    time_quantile=time_quantile,
                    delta_time=sampled_delta,
                    waiting_rate=float(rates[local_index]),
                    fixed_horizon=(
                        None
                        if max_deltas[local_index] is None
                        else float(max_deltas[local_index])
                    ),
                    time_log_density=float(
                        log_time_pf[local_index]
                        .detach()
                        .cpu()
                        .item()
                    ),
                    time_policy_entropy=float(
                        time_mixture_diagnostics["entropy"][local_index]
                        .detach()
                        .cpu()
                        .item()
                    ),
                    time_effective_components=float(
                        time_mixture_diagnostics["effective_components"][local_index]
                        .detach()
                        .cpu()
                        .item()
                    ),
                    time_context_diagnostics=diagnostics,
                )
                total_log_pf[state_index] = (
                    total_log_pf[state_index]
                    + log_action_pf[local_index]
                    + log_breakpoint_pf[local_index]
                    + log_time_pf[local_index]
                )
                component_action[state_index] = log_action_pf[local_index]
                component_breakpoint[state_index] = log_breakpoint_pf[local_index]
                component_time[state_index] = log_time_pf[local_index]

        if any(action is None for action in chosen_actions):
            raise RuntimeError("local policy failed to choose every action")
        self.last_forward_log_components = {
            "gate": selected_gate_log_pf.detach(),
            "atomic_action": component_action.detach(),
            "breakpoint": component_breakpoint.detach(),
            "time": component_time.detach(),
            "total": total_log_pf.detach(),
        }
        self.last_forward_policy_diagnostics = policy_diagnostics
        if self.probability_checks:
            self._assert_forward_probability_components(
                gate_mask,
                gate_log_probabilities,
                total_log_pf,
            )
        return total_log_pf, torch.exp(total_log_pf), chosen_actions

    @staticmethod
    def _assert_masked_categorical_probabilities(logits, valid_mask, label):
        if logits.shape != valid_mask.shape:
            raise RuntimeError(f"{label} logits and mask shapes do not match")
        if not bool(valid_mask.any(dim=1).all().detach().cpu().item()):
            raise RuntimeError(f"{label} contains an all-invalid row")
        valid_logits = logits.masked_select(valid_mask)
        if not bool(torch.isfinite(valid_logits).all().detach().cpu().item()):
            raise RuntimeError(f"{label} has a non-finite valid logit")
        probabilities = torch.softmax(logits, dim=1)
        if not torch.allclose(
            probabilities.sum(dim=1),
            torch.ones_like(probabilities[:, 0]),
            rtol=1e-5,
            atol=1e-6,
        ):
            raise RuntimeError(f"{label} probabilities do not sum to one")
        if bool(
            (probabilities.masked_select(~valid_mask) != 0)
            .any()
            .detach()
            .cpu()
            .item()
        ):
            raise RuntimeError(f"invalid {label} has nonzero probability")

    @staticmethod
    def _assert_forward_probability_components(
        gate_mask,
        gate_log_probabilities,
        total_log_pf,
    ):
        probabilities = torch.exp(gate_log_probabilities)
        if not torch.allclose(
            probabilities.sum(dim=1),
            torch.ones_like(probabilities[:, 0]),
            rtol=1e-5,
            atol=1e-6,
        ):
            raise RuntimeError("valid local gate probabilities do not sum to one")
        if bool((probabilities.masked_select(~gate_mask) != 0).any().detach().cpu().item()):
            raise RuntimeError("invalid local gate action has nonzero probability")
        if not bool(torch.isfinite(total_log_pf).all().detach().cpu().item()):
            raise RuntimeError("sampled forward transition has non-finite log probability")

    @staticmethod
    def _matching_action_index(candidates, action):
        coal = CoalescenceChoice.from_action(action)
        if coal is not None:
            requested = {
                int(coal.active_lineage_i),
                int(coal.active_lineage_j),
            }
            for index, candidate in enumerate(candidates):
                if isinstance(candidate, CoalescenceChoice) and {
                    int(candidate.active_lineage_i),
                    int(candidate.active_lineage_j),
                } == requested:
                    return index, coal
            raise ValueError("recorded coalescence is outside the current action support")
        recomb = RecombinationChoice.from_action(action)
        if recomb is not None:
            for index, candidate in enumerate(candidates):
                if (
                    isinstance(candidate, RecombinationChoice)
                    and int(candidate.active_lineage_i)
                    == int(recomb.active_lineage_i)
                ):
                    return index, recomb
            raise ValueError("recorded recombination is outside the current action support")
        raise ValueError(f"unsupported recorded generated action: {action!r}")

    def score_local_transitions(self, states, actions, random_spec=None):
        """Score fixed local transitions without resampling any component.

        This is the canonical evaluation/replay path.  It uses the same gate,
        candidate-action, breakpoint, and transformed continuous-time densities
        as sampling, so every returned total has a reconstructible decomposition.
        """

        if not bool(getattr(self.env, "is_local", False)):
            raise ValueError("score_local_transitions currently requires a local environment")
        if len(states) != len(actions) or not states:
            raise ValueError("states and actions must have equal nonzero lengths")
        inputs = self.env.prepare_state_rollout_inputs(
            states,
            random_spec=random_spec,
        )
        candidates_by_state = inputs["input_actions"]
        rollout = inputs["rollout"]
        (
            lineage_reps,
            summary_reps,
            lineage_seq_features,
            _active_counts,
        ) = self._encode_states(states)
        gate_log_probs, gate_mask = self._local_gate_log_probabilities(
            summary_reps,
            rollout,
            random_spec=random_spec,
        )
        gate = gate_log_probs.new_zeros(len(states))
        atomic = gate.new_zeros(len(states))
        breakpoint = gate.new_zeros(len(states))
        time = gate.new_zeros(len(states))
        generated_rows = []
        diagnostics = []
        for row, action in enumerate(actions):
            event_type = (
                action.get("event_type")
                if isinstance(action, dict)
                else "fixed_attachment"
                if isinstance(action, FixedAttachmentChoice)
                else "coal"
                if isinstance(action, CoalescenceChoice)
                else "recomb"
                if isinstance(action, RecombinationChoice)
                else None
            )
            gate_index = 1 if event_type == "fixed_attachment" else 0
            if not bool(gate_mask[row, gate_index].detach().cpu().item()):
                raise ValueError(
                    f"recorded {event_type!r} transition is outside gate support"
                )
            gate[row] = gate_log_probs[row, gate_index]
            diagnostics.append(
                {
                    "selected_gate": (
                        "fixed_attachment" if gate_index == 1 else "generated"
                    ),
                    "gate_generated_probability": float(
                        torch.exp(gate_log_probs[row, 0]).detach().cpu().item()
                    ) if bool(gate_mask[row, 0]) else 0.0,
                    "gate_fixed_probability": float(
                        torch.exp(gate_log_probs[row, 1]).detach().cpu().item()
                    ) if bool(gate_mask[row, 1]) else 0.0,
                    "valid_coalescence_actions": 0,
                    "valid_recombination_actions": 0,
                    "coalescence_probability_mass": 0.0,
                    "recombination_probability_mass": 0.0,
                    "selected_atomic_action_probability": 1.0,
                }
            )
            if gate_index == 0:
                generated_rows.append(row)

        if generated_rows:
            selected_candidates = [candidates_by_state[row] for row in generated_rows]
            index_tensor = torch.as_tensor(
                generated_rows,
                dtype=torch.long,
                device=self.device,
            )
            generated_lineages = lineage_reps.index_select(0, index_tensor)
            generated_summaries = summary_reps.index_select(0, index_tensor)
            generated_contexts = [states[row] for row in generated_rows]
            candidate_scoring = self.arg_model.score_action_candidates(
                selected_candidates,
                generated_lineages,
                generated_summaries,
                state_contexts=generated_contexts,
                event_rates=[
                    {
                        "lambda_coal": rollout[row]["lambda_coal"],
                        "lambda_recomb": rollout[row]["lambda_recomb"],
                    }
                    for row in generated_rows
                ],
                random_spec=random_spec,
            )
            probability_logits = candidate_scoring.probability_logits
            action_features = candidate_scoring.action_features
            split_records_by_row = candidate_scoring.split_records
            split_diagnostics_by_row = candidate_scoring.diagnostics
            action_log_probs = torch.log_softmax(probability_logits, dim=1)
            for local_row, state_row in enumerate(generated_rows):
                candidates = selected_candidates[local_row]
                selected_index, parsed_action = self._matching_action_index(
                    candidates,
                    actions[state_row],
                )
                atomic[state_row] = action_log_probs[local_row, selected_index]
                diagnostics[state_row].update(
                    self._structural_action_diagnostics(
                        probability_logits[local_row],
                        candidates,
                        selected_index,
                    )
                )
                split_record = split_records_by_row[local_row][selected_index]
                diagnostics[state_row].update(
                    self._selected_split_diagnostics(
                        split_diagnostics_by_row[local_row],
                        split_record,
                        selected_index,
                        parsed_action,
                    )
                )
                feature = action_features[local_row, selected_index]
                state = states[state_row]
                if isinstance(parsed_action, RecombinationChoice):
                    if parsed_action.breakpoint is None:
                        raise ValueError("recorded recombination has no breakpoint")
                    valid_breakpoints = self.env.valid_breakpoints(
                        state,
                        parsed_action,
                    )
                    bp_logits = self.breakpoint_model.valid_breakpoint_logits(
                        valid_breakpoints,
                        state.active_lineages[int(parsed_action.active_lineage_i)],
                        int(self.env.sequence_length),
                        max(len(state.block_boundaries or ()) - 1, 1),
                        feature,
                        state=state,
                        logit_bias=self.arg_model.recombination_breakpoint_logit_bias(
                            split_record,
                            valid_breakpoints,
                        ),
                    )
                    if random_spec is not None and "T" in random_spec:
                        bp_logits = bp_logits / float(random_spec["T"])
                    valid_breakpoints = [int(value) for value in valid_breakpoints]
                    try:
                        bp_index = valid_breakpoints.index(int(parsed_action.breakpoint))
                    except ValueError as error:
                        raise ValueError(
                            "recorded breakpoint is outside the current support"
                        ) from error
                    bp_log_probabilities = torch.log_softmax(bp_logits, dim=0)
                    breakpoint[state_row] = bp_log_probabilities[bp_index]
                    with torch.no_grad():
                        bp_probabilities = torch.exp(bp_log_probabilities)
                        bp_entropy = -(
                            bp_probabilities * bp_log_probabilities
                        ).sum()
                        support_size = int(bp_logits.numel())
                        normalized_entropy = (
                            bp_entropy / math.log(support_size)
                            if support_size > 1
                            else bp_entropy.new_tensor(0.0)
                        ).clamp(0.0, 1.0)
                        diagnostics[state_row].update(
                            {
                                "breakpoint_support_size": support_size,
                                "breakpoint_entropy": float(
                                    bp_entropy.detach().cpu().item()
                                ),
                                "breakpoint_normalized_entropy": float(
                                    normalized_entropy.detach().cpu().item()
                                ),
                                "breakpoint_selected_probability": float(
                                    bp_probabilities[bp_index]
                                    .detach().cpu().item()
                                ),
                                "breakpoint_max_probability": float(
                                    bp_probabilities.max().detach().cpu().item()
                                ),
                            }
                        )
                    diagnostics[state_row].update(
                        self._selected_breakpoint_split_diagnostics(
                            split_record,
                            parsed_action.breakpoint,
                        )
                    )

                rate = float(rollout[state_row]["total_rate"])
                max_delta = rollout[state_row]["max_delta"]
                delta = parsed_action.delta_time
                quantile = parsed_action.time_quantile
                if delta is None and quantile is None:
                    raise ValueError("recorded generated action is missing its event time")
                if delta is None:
                    delta = self.env.time_env.quantile_to_delta(
                        float(quantile),
                        rate,
                        max_delta=max_delta,
                    )
                if quantile is None:
                    quantile = self.env.time_env.delta_to_quantile(
                        float(delta),
                        rate,
                        max_delta=max_delta,
                    )
                context_features = self.time_model.context_features(
                    [rate],
                    [max_delta],
                    device=self.device,
                    dtype=feature.dtype,
                )
                biological_context, _ = self.arg_model.build_time_context(
                    [state],
                    [parsed_action],
                    [max_delta],
                    dtype=feature.dtype,
                )
                mixture_logits = self.time_model(
                    torch.cat(
                        [feature[None, :], context_features, biological_context],
                        dim=1,
                    )
                )
                generated_mass = self.env.time_env.generated_probability(
                    rate,
                    max_delta=max_delta,
                )
                time[state_row] = self.time_model.log_time_density(
                    mixture_logits,
                    [float(quantile)],
                    [float(delta)],
                    [rate],
                    [generated_mass],
                    random_spec=random_spec,
                )[0].to(time)

        total = gate + atomic + breakpoint + time
        if not bool(torch.isfinite(total).all().detach().cpu().item()):
            raise RuntimeError("rescored local transition has non-finite log P_F")
        return {
            "gate": gate,
            "atomic_action": atomic,
            "breakpoint": breakpoint,
            "time": time,
            "total": total,
            "policy_diagnostics": diagnostics,
        }


    def update_model(self):
        used_learning_rates = [
            float(group["lr"]) for group in self.opt.param_groups
        ]
        record_update_norm = bool(
            self.model_diagnostics
            and self.optimizer_steps % self.model_diagnostics_update_norm_every == 0
        )
        model_snapshots = (
            self._snapshot_model_parameters() if record_update_norm else None
        )
        model_health_before = {}
        model_parameter_norms = {}
        if self.model_diagnostics:
            for name, parameters in self.model_parameter_groups.items():
                group_health = self._model_group_health(
                    name,
                    parameters,
                    "before_clip",
                )
                model_health_before.update(group_health)
                model_parameter_norms[name] = group_health[
                    f"models/{name}/param_norm"
                ]
        gradient_norm_before = self._grad_norm(self.gradient_clipping_params)
        time_gradient_norm_before = self._grad_norm(self.time_params)
        breakpoint_gradient_norm_before = self._grad_norm(self.breakpoint_params)
        structural_gradient_norm_before = self._grad_norm(
            self.structural_policy_params
        )
        info = {'grad_norm': gradient_norm_before,
                'gradient_norm_before_clip': gradient_norm_before,
                'time_head_grad_norm': time_gradient_norm_before,
                'time_head_gradient_norm_before_clip': time_gradient_norm_before,
                'breakpoint_grad_norm': breakpoint_gradient_norm_before,
                'breakpoint_gradient_norm_before_clip': breakpoint_gradient_norm_before,
                'structural_gradient_norm_before_clip': structural_gradient_norm_before,
                # 'z_grad_norm': self._Z.grad.norm().item(),
                'param_norm': self._param_norm(self.gradient_clipping_params),
                'loss': self.loss.detach().cpu().numpy().tolist()}
        info.update(model_health_before)
        info.update(self.last_time_subtb_diagnostics)
        if self._accumulated_balance_records:
            accumulated_details = {
                "loss": self.loss,
                "internal_loss": self.loss.new_tensor(
                    self._accumulated_internal_loss
                ),
                "terminal_loss": self.loss.new_tensor(
                    self._accumulated_terminal_loss
                ),
                "terminal_loss_weight": self.terminal_loss_weight,
                "residual_scale": self.residual_scale,
                "records": self._accumulated_balance_records,
            }
            info.update(self._balance_metrics(accumulated_details))
        else:
            info.update(self.last_balance_diagnostics)
        info.update(
            {
                "subtb_active_lambda": float(self.active_subtb_lambda),
                "subtb_active_max_span": (
                    0
                    if self.active_subtb_max_span is None
                    else int(self.active_subtb_max_span)
                ),
                "time_head_warmup_active": (
                    self.current_epoch < self.time_head_warmup_epochs
                ),
            }
        )

        torch.nn.utils.clip_grad_norm_(
            self.structural_policy_params,
            self.grad_clip,
        )
        breakpoint_clip = (
            self.grad_clip
            if self.breakpoint_gradient_clip_norm is None
            else self.breakpoint_gradient_clip_norm
        )
        torch.nn.utils.clip_grad_norm_(self.breakpoint_params, breakpoint_clip)
        time_clip = (
            self.grad_clip
            if self.time_head_gradient_clip_norm is None
            else self.time_head_gradient_clip_norm
        )
        torch.nn.utils.clip_grad_norm_(self.time_params, time_clip)
        info["gradient_norm_after_clip"] = self._grad_norm(
            self.gradient_clipping_params
        )
        info["time_head_gradient_norm_after_clip"] = self._grad_norm(
            self.time_params
        )
        info["breakpoint_gradient_norm_after_clip"] = self._grad_norm(
            self.breakpoint_params
        )
        info["structural_gradient_norm_after_clip"] = self._grad_norm(
            self.structural_policy_params
        )
        if self.model_diagnostics:
            clip_norms = {
                "structural": float(self.grad_clip),
                "breakpoint": float(breakpoint_clip),
                "time": float(time_clip),
            }
            for name, parameters in self.model_parameter_groups.items():
                info.update(
                    self._model_group_health(name, parameters, "after_clip")
                )
                before_norm = float(
                    info[f"models/{name}/grad_norm_before_clip"]
                )
                after_norm = float(
                    info[f"models/{name}/grad_norm_after_clip"]
                )
                info[f"models/{name}/clip_norm"] = clip_norms[name]
                info[f"models/{name}/gradient_clipped"] = bool(
                    math.isfinite(before_norm)
                    and before_norm > clip_norms[name]
                )
                info[f"models/{name}/clip_scale"] = float(
                    1.0
                    if before_norm <= 1e-12
                    else after_norm / before_norm
                )
        self.opt.step()
        if model_snapshots is not None:
            info.update(
                self._model_update_metrics(
                    model_snapshots,
                    model_parameter_norms,
                )
            )
        self.optimizer_steps += 1
        if self.scheduler is not None:
            self.scheduler.step_update()
        self.opt.zero_grad()
        self.loss = 0
        self._accumulated_balance_records = []
        self._accumulated_internal_loss = 0.0
        self._accumulated_terminal_loss = 0.0

        info.update(self.learning_rate_metrics())
        learning_rate_group_names = (
            self.scheduler.group_names
            if self.scheduler is not None
            else tuple(
                f"group_{index}" for index in range(len(self.opt.param_groups))
            )
        )
        info.update(
            {
                f"lr/used/{name}": value
                for name, value in zip(
                    learning_rate_group_names,
                    used_learning_rates,
                )
            }
        )
        for index, name in enumerate(("structural", "breakpoint", "time")):
            info[f"models/{name}/lr_used"] = float(
                used_learning_rates[index]
            )
            info[f"models/{name}/lr_next"] = float(
                self.opt.param_groups[index]["lr"]
            )
        info["models/diagnostics_enabled"] = bool(self.model_diagnostics)
        info["models/update_norm_recorded"] = bool(
            model_snapshots is not None
        )

        return info

    def learning_rate_metrics(self):
        if self.scheduler is not None:
            return self.scheduler.metrics()
        return {
            "lr/scheduler_type": "constant",
            "lr/optimizer_step": int(self.optimizer_steps),
            **{
                f"lr/group_{index}": float(group["lr"])
                for index, group in enumerate(self.opt.param_groups)
            },
        }

    def step_lr_scheduler(self, metrics):
        if self.scheduler is None:
            return self.learning_rate_metrics()
        return self.scheduler.step_metric(metrics)

    def lr_scheduler_metadata(self):
        if self.scheduler is None:
            return {"type": "constant", "configured": False}
        return {**self.scheduler.metadata(), "configured": True}

    def _record_log_z_targets(self, targets):
        finite_targets = targets[torch.isfinite(targets)]
        if finite_targets.numel() == 0:
            return
        self.log_z_target_sum += float(finite_targets.sum().detach().cpu().item())
        self.log_z_target_count += int(finite_targets.numel())
        self.last_log_z_target = (
            self.log_z_target_sum / max(self.log_z_target_count, 1)
        )

    def count_backward_parents(self, arg_state):
        if bool(getattr(self.env, "is_local", False)):
            return self.env.backward_parent_count(arg_state)
        return len(self._enumerate_inverse_arg_actions(arg_state))

    def _is_initial_arg_state(self, state):
        initial_ids = set(range(self.env.num_sequences))
        if set(state.all_nodes) != initial_ids:
            return False
        if {lineage.node_id for lineage in state.active_lineages} != initial_ids:
            return False

        for node_id in initial_ids:
            lineage = state.all_nodes[node_id]
            if lineage.children or lineage.parents:
                return False
            if lineage.material_segments.segments != ((0, self.env.num_blocks),):
                return False
        return True

    def _enumerate_inverse_arg_actions(self, state):
        inverse_actions = []

        # Use one loop to collect both coal and recomb candidates efficiently
        # Prepare coal candidates in a single pass with a list comprehension
        coal_candidates = [
            (active_idx, lineage)
            for active_idx, lineage in enumerate(state.active_lineages)
            if (
                lineage.event_type == "coal"
                and len(lineage.children) == 2
                and self._is_latest_time_event(state, lineage.node_id)
                and lineage.children[0] in state.all_nodes
                and lineage.children[1] in state.all_nodes
                and lineage.node_id in state.all_nodes[lineage.children[0]].parents
                and lineage.node_id in state.all_nodes[lineage.children[1]].parents
            )
        ]
        for active_idx, lineage in coal_candidates:
            child_i, child_j = lineage.children
            inverse_actions.append(
                {
                    "event_type": "coal",
                    "active_idx": active_idx,
                    "parent_id": lineage.node_id,
                    "child_ids": (child_i, child_j),
                }
            )

        # Prepare recomb_by_event using a single pass with a dictionary
        recomb_by_event = {}
        for active_idx, lineage in enumerate(state.active_lineages):
            if (
                lineage.event_type == "recomb"
                and len(lineage.children) == 1
                and lineage.breakpoint is not None
                and lineage.recombination_side in ("left", "right")
            ):
                key = (lineage.children[0], lineage.breakpoint)
                recomb_by_event.setdefault(key, {})[lineage.recombination_side] = (active_idx, lineage.node_id)

        # We can iterate efficiently over recomb_by_event rather than collecting in a list
        for (child_id, breakpoint), sides in recomb_by_event.items():
            if "left" not in sides or "right" not in sides or child_id not in state.all_nodes:
                continue
            left_idx, left_id = sides["left"]
            right_idx, right_id = sides["right"]
            child = state.all_nodes[child_id]
            left_parent = state.all_nodes[left_id]
            right_parent = state.all_nodes[right_id]

            # Fast short-circuit checks, in a single conditional
            if (
                not self._is_latest_time_event(state, left_id, right_id)
                or set(child.parents) != {left_id, right_id}
                or left_parent.material_segments.intersection_count(right_parent.material_segments) > 0
                or left_parent.material_segments.union(right_parent.material_segments) != child.material_segments
            ):
                continue

            inverse_actions.append(
                {
                    "event_type": "recomb",
                    "active_indices": (left_idx, right_idx),
                    "parent_ids": (left_id, right_id),
                    "child_id": child_id,
                    "breakpoint": breakpoint,
                }
            )

        return inverse_actions

    def _is_latest_time_event(self, state, *node_ids):
        current_time = float(state.current_time)
        return all(
            math.isclose(
                float(state.all_nodes[node_id].time),
                current_time,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            for node_id in node_ids
        )

    def _max_node_time(self, state):
        if not state.all_nodes:
            return 0.0
        return max(float(lineage.time) for lineage in state.all_nodes.values())

    def _apply_inverse_arg_action(self, state, inverse_action):
        if inverse_action["event_type"] == "coal":
            return self._apply_inverse_coalescence(state, inverse_action)
        if inverse_action["event_type"] == "recomb":
            return self._apply_inverse_recombination(state, inverse_action)
        raise ValueError(f"Unknown inverse ARG action: {inverse_action}")

    def _apply_inverse_coalescence(self, state, inverse_action):
        parent_state = state.clone()
        parent_id = inverse_action["parent_id"]
        child_ids = inverse_action["child_ids"]

        remaining_lineages = [
            lineage for lineage in parent_state.active_lineages if lineage.node_id != parent_id
        ]
        parent_state.all_nodes.pop(parent_id)
        parent_state.active_lineages = []
        for child_id in child_ids:
            child = parent_state.all_nodes[child_id]
            child.parents = [node_id for node_id in child.parents if node_id != parent_id]
            parent_state.active_lineages.append(child)
        parent_state.active_lineages.extend(remaining_lineages)
        parent_state.total_active_blocks = None

        active_idx_by_id = self._active_index_by_node_id(parent_state)
        forward_action = {
            "event_type": "coal",
            "active_lineage_i": active_idx_by_id[child_ids[0]],
            "active_lineage_j": active_idx_by_id[child_ids[1]],
        }
        parent_state.current_time = self._max_node_time(parent_state)
        delta_t = float(state.current_time) - float(parent_state.current_time)
        rates = self.env.enumerate_prior_options(parent_state).rates
        forward_action["time_quantile"] = (
            self.env.time_env.delta_to_quantile(
                delta_t,
                self.env._total_event_rate(rates),
            )
        )
        forward_action["delta_time"] = delta_t
        self._finalize_backward_parent_state(parent_state, state, forward_action)
        return parent_state, forward_action

    def _apply_inverse_recombination(self, state, inverse_action):
        parent_state = state.clone()
        left_id, right_id = inverse_action["parent_ids"]
        child_id = inverse_action["child_id"]

        remaining_lineages = [
            lineage for lineage in parent_state.active_lineages if lineage.node_id not in (left_id, right_id)
        ]
        parent_state.all_nodes.pop(left_id)
        parent_state.all_nodes.pop(right_id)

        child = parent_state.all_nodes[child_id]
        child.parents = []
        parent_state.active_lineages = [child] + remaining_lineages
        parent_state.total_active_blocks = None

        active_idx_by_id = self._active_index_by_node_id(parent_state)
        forward_action = {
            "event_type": "recomb",
            "active_lineage_i": active_idx_by_id[child_id],
            "breakpoint": inverse_action["breakpoint"],
        }
        parent_state.current_time = self._max_node_time(parent_state)
        delta_t = float(state.current_time) - float(parent_state.current_time)
        prior_options = self.env.enumerate_prior_options(parent_state)
        rates = prior_options.rates
        total_rate = self.env._total_event_rate(rates)
        reconstructed_choice = next(
            (
                choice
                for choice in prior_options.recomb_choices
                if int(choice.active_lineage_i)
                == int(forward_action["active_lineage_i"])
            ),
            None,
        )
        if reconstructed_choice is None:
            raise RuntimeError(
                "backward recombination has no matching forward candidate"
            )
        forward_action.update(
            {
                "material_count": int(
                    reconstructed_choice.material_count
                ),
                "span_start": int(reconstructed_choice.span_start),
                "span_end": int(reconstructed_choice.span_end),
            }
        )
        forward_action["time_quantile"] = (
            self.env.time_env.delta_to_quantile(delta_t, total_rate)
        )
        forward_action["delta_time"] = delta_t
        self._finalize_backward_parent_state(parent_state, state, forward_action)
        return parent_state, forward_action

    def _finalize_backward_parent_state(self, parent_state, child_state, forward_action):
        parent_state.max_node_idx = max(parent_state.all_nodes) if parent_state.all_nodes else -1
        parent_state.log_reward = None
        parent_state.action_options = None
        parent_state.rates = None
        parent_state.prior_options = None
        parent_state.total_active_blocks = sum(
            lineage.material_segments.count
            for lineage in parent_state.active_lineages
        )
        parent_state.is_done = self.env.is_terminal(parent_state)

        reconstructed_action = (
            CoalescenceChoice.from_action(forward_action)
            or RecombinationChoice.from_action(forward_action)
        )
        if reconstructed_action is None:
            raise RuntimeError(
                "backward reconstruction produced an invalid forward action"
            )
        log_prior = self.env.compute_cwr_event_log_prior(
            parent_state,
            reconstructed_action,
        )
        if math.isfinite(log_prior):
            parent_state.accumulated_log_prior = child_state.accumulated_log_prior - log_prior
        parent_state.action_options = None
        parent_state.rates = None
        parent_state.prior_options = None

    def _active_index_by_node_id(self, state):
        return {lineage.node_id: idx for idx, lineage in enumerate(state.active_lineages)}

    def get_loss_from_rollout_outputs(self, rollout_outputs):
        if self.loss_mode in {"subtb", "fl_subtb"}:
            return self.compute_subtb_loss_from_rollout_outputs(rollout_outputs)
        return self.compute_tb_loss_from_rollout_outputs(rollout_outputs)

    def compute_tb_loss_from_rollout_outputs(self, rollout_outputs):
        log_paths_pf = rollout_outputs['log_paths_pf']
        log_paths_pb = rollout_outputs['log_paths_pb']
        terminal_mask = rollout_outputs.get("terminal_mask")
        if terminal_mask is not None and not bool(terminal_mask.all().detach().cpu().item()):
            raise ValueError(
                "Trajectory balance requires terminal rollout outputs; "
                "use subtb/fl_subtb for capped partial-to-partial rollouts."
            )
        log_rewards = torch.as_tensor(
            rollout_outputs['log_rewards'],
            dtype=log_paths_pf.dtype,
            device=log_paths_pf.device,
        )
        if not bool(torch.isfinite(log_rewards).all().detach().cpu().item()):
            raise ValueError(
                "Trajectory balance requires finite terminal log rewards for every trajectory."
            )

        log_pf = log_paths_pf.sum(-1)
        log_pb = log_paths_pb.sum(-1)

        
        log_z = self.compute_log_Z(None).reshape(-1).to(log_paths_pf)

        forward_value = log_z + log_pf
        backward_value = log_rewards + log_pb

        loss = self.loss_fn(forward_value, backward_value)

        return loss

    def compute_subtb_loss_from_rollout_outputs(self, rollout_outputs):
        if "trajectory_states" not in rollout_outputs:
            raise ValueError("SubTB loss requires rollout_outputs['trajectory_states']")
        if "trajectory_lengths" not in rollout_outputs:
            raise ValueError("SubTB loss requires rollout_outputs['trajectory_lengths']")

        state_paths = rollout_outputs["trajectory_states"]
        flat_states = [state for path in state_paths for state in path]
        flat_log_flows = self.compute_log_state_flows(flat_states)

        log_flows_by_traj = []
        cursor = 0
        for path in state_paths:
            next_cursor = cursor + len(path)
            log_flows_by_traj.append(flat_log_flows[cursor:next_cursor])
            cursor = next_cursor

        loss, details = self._subtb_loss_from_log_flows(
            log_flows_by_traj,
            rollout_outputs["log_paths_pf"],
            rollout_outputs["log_paths_pb"],
            rollout_outputs["trajectory_lengths"],
            self.active_subtb_lambda,
            self.active_subtb_max_span,
            terminal_mask=rollout_outputs.get("terminal_mask"),
            terminal_loss_weight=self.terminal_loss_weight,
            residual_scale=self.residual_scale,
            trajectory_actions=rollout_outputs.get("trajectory_actions"),
            return_details=True,
        )
        self.last_balance_diagnostics = self._balance_metrics(details)
        self._last_balance_details = details
        self.last_balance_diagnostics.update(
            {
                "flow/subtb_active_lambda": float(self.active_subtb_lambda),
                "flow/subtb_active_max_span": (
                    0
                    if self.active_subtb_max_span is None
                    else int(self.active_subtb_max_span)
                ),
                "flow/terminal_loss_weight": float(self.terminal_loss_weight),
                "flow/residual_scale": float(self.residual_scale),
            }
        )
        self.last_transition_decomposition = self._transition_decomposition(
            rollout_outputs,
            log_flows_by_traj,
            details,
        )
        if self.flow_debug and self.flow_debug_max_records:
            for record in self.last_transition_decomposition[
                : self.flow_debug_max_records
            ]:
                print("FLOW_DEBUG", record)
        self.last_time_subtb_diagnostics = self._time_subtb_diagnostics(
            log_flows_by_traj,
            rollout_outputs["log_paths_pf"],
            rollout_outputs["log_paths_pb"],
            rollout_outputs["trajectory_lengths"],
            rollout_outputs.get("trajectory_actions", ()),
            self.active_subtb_lambda,
            self.active_subtb_max_span,
        )
        return loss

    @staticmethod
    def _float_metric(value):
        if torch.is_tensor(value):
            return float(value.detach().cpu().item())
        return float(value)

    @classmethod
    def _balance_metrics(cls, details):
        records = details["records"]
        residual_scale = float(details.get("residual_scale", 1.0))

        def values(predicate=lambda _record: True):
            selected = [
                record["residual"].detach().to(torch.float64)
                for record in records
                if predicate(record)
            ]
            if not selected:
                return None
            return torch.stack(selected)

        def mse(predicate):
            tensor = values(predicate)
            return 0.0 if tensor is None else float(tensor.square().mean().cpu().item())

        all_values = values()
        metrics = {
            "flow/loss_total_weighted": cls._float_metric(details["loss"]),
            "flow/loss_internal_scaled": cls._float_metric(details["internal_loss"]),
            "flow/loss_terminal_scaled_unweighted": cls._float_metric(
                details["terminal_loss"]
            ),
            "flow/loss_terminal_scaled_weighted": cls._float_metric(
                details["terminal_loss"] * details["terminal_loss_weight"]
            ),
            "flow/internal_one_step_residual_mse": mse(
                lambda row: not row["terminal"] and row["span"] == 1
            ),
            "flow/internal_multi_step_subtb_residual_mse": mse(
                lambda row: not row["terminal"] and row["span"] > 1
            ),
            "flow/terminal_one_step_residual_mse": mse(
                lambda row: row["terminal"] and row["span"] == 1
            ),
            "flow/terminal_multi_step_residual_mse": mse(
                lambda row: row["terminal"] and row["span"] > 1
            ),
            "flow/unweighted_total_residual_mse": (
                0.0
                if all_values is None
                else float(all_values.square().mean().cpu().item())
            ),
            "flow/residual_count": len(records),
            "flow/terminal_residual_count": sum(row["terminal"] for row in records),
            "flow/internal_residual_count": sum(not row["terminal"] for row in records),
            "flow/residual_scale": residual_scale,
        }
        if all_values is None:
            for key in (
                "mean", "abs_mean", "rmse", "p50", "p90", "p95", "p99",
                "length_normalized_rmse",
            ):
                metrics[f"flow/residual_{key}"] = 0.0
        else:
            metrics.update(
                {
                    "flow/residual_mean": float(all_values.mean().cpu().item()),
                    "flow/residual_abs_mean": float(all_values.abs().mean().cpu().item()),
                    "flow/residual_rmse": float(
                        all_values.square().mean().sqrt().cpu().item()
                    ),
                    "flow/residual_p50": float(all_values.abs().quantile(0.50).cpu().item()),
                    "flow/residual_p90": float(all_values.abs().quantile(0.90).cpu().item()),
                    "flow/residual_p95": float(all_values.abs().quantile(0.95).cpu().item()),
                    "flow/residual_p99": float(all_values.abs().quantile(0.99).cpu().item()),
                    "flow/scaled_residual_rmse": float(
                        all_values.square().mean().sqrt().div(residual_scale).cpu().item()
                    ),
                    "flow/scaled_residual_abs_mean": float(
                        all_values.abs().mean().div(residual_scale).cpu().item()
                    ),
                }
            )
            all_weights = torch.stack(
                [row["weight"].detach().to(torch.float64) for row in records]
            )
            denominator = all_weights.sum().clamp_min(
                torch.finfo(all_weights.dtype).tiny
            )
            internal_mask = torch.as_tensor(
                [not row["terminal"] for row in records],
                dtype=torch.bool,
                device=all_values.device,
            )
            terminal_mask = ~internal_mask
            metrics["flow/loss_internal_raw_unscaled"] = float(
                (
                    (all_weights[internal_mask] * all_values[internal_mask].square()).sum()
                    / denominator
                ).cpu().item()
            )
            metrics["flow/loss_terminal_raw_unscaled"] = float(
                (
                    (all_weights[terminal_mask] * all_values[terminal_mask].square()).sum()
                    / denominator
                ).cpu().item()
            )
            normalized = torch.stack(
                [
                    row["residual"].detach().to(torch.float64)
                    / max(int(row["trajectory_length"]), 1)
                    for row in records
                ]
            )
            metrics["flow/residual_length_normalized_rmse"] = float(
                normalized.square().mean().sqrt().cpu().item()
            )
        for action_type in ("coal", "recomb", "fixed_attachment", "terminal"):
            action_values = values(
                lambda row, requested=action_type: row["action_type"] == requested
            )
            prefix = f"flow/action/{action_type}"
            metrics[f"{prefix}_count"] = (
                0 if action_values is None else int(action_values.numel())
            )
            metrics[f"{prefix}_residual_mse"] = (
                0.0
                if action_values is None
                else float(action_values.square().mean().cpu().item())
            )
            metrics[f"{prefix}_residual_abs_mean"] = (
                0.0
                if action_values is None
                else float(action_values.abs().mean().cpu().item())
            )
        spans = sorted({int(row["span"]) for row in records})
        for span in spans:
            span_values = values(lambda row, requested=span: row["span"] == requested)
            metrics[f"flow/span/{span}_count"] = int(span_values.numel())
            metrics[f"flow/span/{span}_residual_mse"] = float(
                span_values.square().mean().cpu().item()
            )
        return metrics

    @staticmethod
    def _state_debug_id(state):
        identity = (
            state.structural_identity()
            if hasattr(state, "structural_identity")
            else repr(state)
        )
        return hashlib.sha256(repr(identity).encode("utf-8")).hexdigest()[:16]

    def _transition_decomposition(
        self,
        rollout_outputs,
        log_flows_by_traj,
        details,
    ):
        one_step = {
            (row["trajectory_index"], row["start"]): row
            for row in details["records"]
            if row["span"] == 1
        }
        recorded_components = rollout_outputs.get("trajectory_log_components", ())
        decompositions = []
        for traj_idx, path in enumerate(rollout_outputs["trajectory_states"]):
            actions = rollout_outputs.get("trajectory_actions", ())[traj_idx]
            length = int(rollout_outputs["trajectory_lengths"][traj_idx].item())
            for step in range(length):
                residual_row = one_step[(traj_idx, step)]
                components = (
                    recorded_components[traj_idx][step]
                    if traj_idx < len(recorded_components)
                    and step < len(recorded_components[traj_idx])
                    else {}
                )
                action_type = actions[step].get("event_type", "unknown")
                destination = path[step + 1]
                terminal = bool(destination.is_done)
                source_flow = log_flows_by_traj[traj_idx][step]
                destination_flow = log_flows_by_traj[traj_idx][step + 1]
                row = {
                    "state_id": self._state_debug_id(path[step]),
                    "destination_state_id": self._state_debug_id(destination),
                    "trajectory_index": traj_idx,
                    "step": step,
                    "action_type": action_type,
                    "log_flow_source": self._float_metric(source_flow),
                    "log_flow_source_partial_reward": float(
                        getattr(path[step], "partial_log_reward", 0.0)
                    ),
                    "log_pf_gate": float(components.get("gate", 0.0)),
                    "log_pf_action": float(components.get("atomic_action", 0.0)),
                    "log_pf_time": float(components.get("time", 0.0)),
                    "log_pf_breakpoint": float(components.get("breakpoint", 0.0)),
                    "log_pb_parent": float(
                        rollout_outputs["log_paths_pb"][traj_idx, step]
                        .detach().cpu().item()
                    ),
                    "log_pb_action": 0.0,
                    "log_pb_time": 0.0,
                    "log_flow_destination_or_reward": self._float_metric(destination_flow),
                    "log_flow_destination_partial_reward": float(
                        getattr(destination, "partial_log_reward", 0.0)
                    ),
                    "destination_is_terminal": terminal,
                    "residual": self._float_metric(residual_row["residual"]),
                }
                reconstructed_pf = (
                    row["log_pf_gate"]
                    + row["log_pf_action"]
                    + row["log_pf_time"]
                    + row["log_pf_breakpoint"]
                )
                row["log_pf_total"] = float(
                    rollout_outputs["log_paths_pf"][traj_idx, step]
                    .detach().cpu().item()
                )
                row["log_flow_source_learned_potential"] = (
                    row["log_flow_source"]
                    - row["log_flow_source_partial_reward"]
                )
                row["log_flow_destination_learned_potential"] = (
                    None
                    if terminal
                    else row["log_flow_destination_or_reward"]
                    - row["log_flow_destination_partial_reward"]
                )
                row["log_pf_reconstruction_error"] = reconstructed_pf - row["log_pf_total"]
                row["log_pb_total"] = row["log_pb_parent"]
                row["log_pb_reconstruction_error"] = (
                    row["log_pb_parent"]
                    + row["log_pb_action"]
                    + row["log_pb_time"]
                    - row["log_pb_total"]
                )
                decompositions.append(row)
        return decompositions

    @staticmethod
    def _time_subtb_diagnostics(
        log_flows_by_traj,
        log_paths_pf,
        log_paths_pb,
        trajectory_lengths,
        trajectory_actions,
        subtb_lambda,
        subtb_max_span,
    ):
        """Summarize SubTB residuals spanning generated time actions."""

        lengths = (
            trajectory_lengths.detach().cpu().tolist()
            if torch.is_tensor(trajectory_lengths)
            else list(trajectory_lengths)
        )
        residuals = []
        weights = []
        one_step = []
        terminal_reaching = []
        for traj_idx, length_value in enumerate(lengths):
            length = int(length_value)
            actions = (
                trajectory_actions[traj_idx]
                if traj_idx < len(trajectory_actions)
                else ()
            )
            time_mask = [
                action.get("time_quantile") is not None
                for action in actions[:length]
            ]
            if not any(time_mask):
                continue
            zero = log_paths_pf.new_zeros(1)
            pf_prefix = torch.cat(
                [zero, torch.cumsum(log_paths_pf[traj_idx, :length], dim=0)]
            )
            pb_prefix = torch.cat(
                [zero, torch.cumsum(log_paths_pb[traj_idx, :length], dim=0)]
            )
            flows = log_flows_by_traj[traj_idx]
            for start in range(length):
                max_end = length + 1
                if subtb_max_span is not None:
                    max_end = min(
                        max_end,
                        start + int(subtb_max_span) + 1,
                    )
                for end in range(start + 1, max_end):
                    if not any(time_mask[start:end]):
                        continue
                    residual = (
                        flows[start]
                        + pf_prefix[end]
                        - pf_prefix[start]
                        - flows[end]
                        - pb_prefix[end]
                        + pb_prefix[start]
                    ).detach()
                    residuals.append(residual)
                    weights.append(float(subtb_lambda) ** (end - start))
                    if end - start == 1:
                        one_step.append(residual)
                    if end == length:
                        terminal_reaching.append(residual)
        if not residuals:
            return {
                "time_subtb_count": 0,
                "time_subtb_residual_mean": 0.0,
                "time_subtb_residual_variance": 0.0,
                "time_subtb_squared_residual_mean": 0.0,
                "time_subtb_one_step_squared_residual_mean": 0.0,
                "time_subtb_terminal_squared_residual_mean": 0.0,
            }
        values = torch.stack(residuals).to(torch.float64)
        weight_tensor = torch.as_tensor(
            weights,
            dtype=torch.float64,
            device=values.device,
        )
        weight_total = weight_tensor.sum().clamp_min(
            torch.finfo(weight_tensor.dtype).tiny
        )
        weighted_mean = torch.sum(values * weight_tensor) / weight_total

        def mean_square(items):
            if not items:
                return 0.0
            return float(
                torch.stack(items)
                .to(torch.float64)
                .square()
                .mean()
                .cpu()
                .item()
            )

        return {
            "time_subtb_count": int(values.numel()),
            "time_subtb_residual_mean": float(weighted_mean.cpu().item()),
            "time_subtb_residual_variance": float(
                (
                    torch.sum(
                        weight_tensor * (values - weighted_mean).square()
                    )
                    / weight_total
                )
                .cpu()
                .item()
            ),
            "time_subtb_squared_residual_mean": float(
                (torch.sum(weight_tensor * values.square()) / weight_total)
                .cpu()
                .item()
            ),
            "time_subtb_one_step_squared_residual_mean": mean_square(one_step),
            "time_subtb_terminal_squared_residual_mean": mean_square(
                terminal_reaching
            ),
        }

    @staticmethod
    def _subtb_loss_from_log_flows(
        log_flows_by_traj,
        log_paths_pf,
        log_paths_pb,
        trajectory_lengths,
        subtb_lambda,
        subtb_max_span=None,
        terminal_mask=None,
        terminal_loss_weight=1.0,
        residual_scale=1.0,
        trajectory_actions=None,
        return_details=False,
    ):
        if subtb_max_span is not None:
            subtb_max_span = int(subtb_max_span)
            if subtb_max_span <= 0:
                raise ValueError("subtb_max_span must be positive when provided")
        if torch.is_tensor(trajectory_lengths):
            lengths = trajectory_lengths.detach().cpu().tolist()
        else:
            lengths = list(trajectory_lengths)

        terminal_loss_weight = float(terminal_loss_weight)
        residual_scale = float(residual_scale)
        if not math.isfinite(terminal_loss_weight) or terminal_loss_weight < 0.0:
            raise ValueError("terminal_loss_weight must be finite and nonnegative")
        if not math.isfinite(residual_scale) or residual_scale <= 0.0:
            raise ValueError("residual_scale must be finite and positive")
        if terminal_mask is None:
            terminal_flags = [False] * len(lengths)
        elif torch.is_tensor(terminal_mask):
            terminal_flags = [bool(value) for value in terminal_mask.detach().cpu().tolist()]
        else:
            terminal_flags = [bool(value) for value in terminal_mask]
        if len(terminal_flags) != len(lengths):
            raise ValueError("terminal_mask length must match trajectory_lengths")

        internal_weighted_sum = log_paths_pf.new_tensor(0.0)
        terminal_weighted_sum = log_paths_pf.new_tensor(0.0)
        weight_sum = log_paths_pf.new_tensor(0.0)
        records = []
        for traj_idx, length in enumerate(lengths):
            length = int(length)
            if length <= 0:
                continue
            log_flows = log_flows_by_traj[traj_idx]
            if int(log_flows.numel()) != length + 1:
                raise ValueError(
                    "Each SubTB state path must have exactly trajectory length + 1 "
                    f"log flows, got {int(log_flows.numel())} for length {length}"
                )

            zero = log_paths_pf.new_zeros(1)
            pf_prefix = torch.cat([
                zero,
                torch.cumsum(log_paths_pf[traj_idx, :length], dim=0),
            ])
            pb_prefix = torch.cat([
                zero,
                torch.cumsum(log_paths_pb[traj_idx, :length], dim=0),
            ])

            for start in range(length):
                max_end = length + 1
                if subtb_max_span is not None:
                    max_end = min(max_end, start + int(subtb_max_span) + 1)
                for end in range(start + 1, max_end):
                    span = end - start
                    log_pf = pf_prefix[end] - pf_prefix[start]
                    log_pb = pb_prefix[end] - pb_prefix[start]
                    residual = log_flows[start] + log_pf - log_flows[end] - log_pb
                    weight = log_paths_pf.new_tensor(float(subtb_lambda) ** span)
                    is_terminal = bool(terminal_flags[traj_idx] and end == length)
                    scaled_square = (residual / residual_scale).pow(2)
                    if is_terminal:
                        terminal_weighted_sum = (
                            terminal_weighted_sum + weight * scaled_square
                        )
                    else:
                        internal_weighted_sum = (
                            internal_weighted_sum + weight * scaled_square
                        )
                    weight_sum = weight_sum + weight
                    action_type = "multi_step"
                    if span == 1 and trajectory_actions is not None:
                        action_type = str(
                            trajectory_actions[traj_idx][start].get(
                                "event_type", "unknown"
                            )
                        )
                    if is_terminal:
                        action_type = "terminal"
                    records.append(
                        {
                            "trajectory_index": traj_idx,
                            "trajectory_length": length,
                            "start": start,
                            "end": end,
                            "span": span,
                            "terminal": is_terminal,
                            "action_type": action_type,
                            "residual": residual,
                            "weight": weight,
                        }
                    )

        if float(weight_sum.detach().cpu().item()) == 0.0:
            loss = internal_weighted_sum + terminal_weighted_sum
            internal_loss = internal_weighted_sum
            terminal_loss = terminal_weighted_sum
        else:
            # Both components use the historical all-subtrajectory denominator.
            # Therefore lambda_T=1 and residual_scale=1 reproduce the previous
            # objective exactly, while lambda_T>1 increases only terminal
            # supervision and never double-counts a terminal term.
            internal_loss = internal_weighted_sum / weight_sum
            terminal_loss = terminal_weighted_sum / weight_sum
            loss = internal_loss + terminal_loss_weight * terminal_loss
        if not return_details:
            return loss
        return loss, {
            "loss": loss,
            "internal_loss": internal_loss,
            "terminal_loss": terminal_loss,
            "terminal_loss_weight": terminal_loss_weight,
            "residual_scale": residual_scale,
            "weight_sum": weight_sum,
            "records": records,
        }
        
    
    def accumulate_loss(self, rollout_outputs, factor=1.0):
        loss = self.get_loss_from_rollout_outputs(rollout_outputs)
        if self._last_balance_details is not None:
            self._accumulated_balance_records.extend(
                {
                    **record,
                    "residual": record["residual"].detach(),
                    "weight": record["weight"].detach(),
                }
                for record in self._last_balance_details["records"]
            )
            self._accumulated_internal_loss += self._float_metric(
                self._last_balance_details["internal_loss"]
            ) / float(factor)
            self._accumulated_terminal_loss += self._float_metric(
                self._last_balance_details["terminal_loss"]
            ) / float(factor)
        loss = (loss / factor)
        loss.backward()
        self.loss = self.loss + loss.detach()

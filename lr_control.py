"""Optimizer-step learning-rate control for GFlowNet training."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence


LR_SCHEDULER_TYPES = {"constant", "cosine", "step", "plateau"}
LR_PLATEAU_MODES = {"min", "max"}
LR_PLATEAU_THRESHOLD_MODES = {"rel", "abs"}

DEFAULT_LR_SCHEDULER_CONFIG = {
    "type": "cosine",
    "warmup_steps": None,
    "warmup_fraction": 0.05,
    "warmup_start_ratio": 0.1,
    "min_lr_ratio": 0.1,
    "step_size": 100,
    "step_gamma": 0.5,
    "plateau_metric": "auto",
    "plateau_mode": "min",
    "plateau_factor": 0.5,
    "plateau_patience": 5,
    "plateau_threshold": 1e-3,
    "plateau_threshold_mode": "rel",
    "plateau_cooldown": 0,
}

AUTO_PLATEAU_METRICS = (
    "flow_eval/fixed_bank_subtb_mse",
    "flow_eval/fixed_bank_terminal_mse",
    "eval_fl_subtb_mse",
    "eval_subtb_mse",
    "eval_tb_mse",
    "eval_local_loss_mean",
)


def normalize_lr_scheduler_config(config=None):
    """Return a validated scheduler config without mutating the caller's data."""

    normalized = copy.deepcopy(DEFAULT_LR_SCHEDULER_CONFIG)
    if config is not None:
        if not isinstance(config, Mapping):
            raise ValueError("training.lr_scheduler must be a mapping")
        unknown = sorted(set(config) - set(normalized))
        if unknown:
            raise ValueError(
                "unknown training.lr_scheduler field(s): " + ", ".join(unknown)
            )
        normalized.update(copy.deepcopy(dict(config)))

    scheduler_type = str(normalized["type"]).lower()
    if scheduler_type not in LR_SCHEDULER_TYPES:
        raise ValueError(
            "training.lr_scheduler.type must be one of "
            + ", ".join(repr(value) for value in sorted(LR_SCHEDULER_TYPES))
        )
    normalized["type"] = scheduler_type

    warmup_steps = normalized["warmup_steps"]
    if warmup_steps is not None:
        if isinstance(warmup_steps, bool):
            raise ValueError("training.lr_scheduler.warmup_steps must be nonnegative or null")
        warmup_steps = int(warmup_steps)
        if warmup_steps < 0:
            raise ValueError("training.lr_scheduler.warmup_steps must be nonnegative or null")
    normalized["warmup_steps"] = warmup_steps

    warmup_fraction = float(normalized["warmup_fraction"])
    if not math.isfinite(warmup_fraction) or not 0.0 <= warmup_fraction < 1.0:
        raise ValueError(
            "training.lr_scheduler.warmup_fraction must be finite and in [0, 1)"
        )
    normalized["warmup_fraction"] = warmup_fraction

    for name in ("warmup_start_ratio", "min_lr_ratio"):
        value = float(normalized[name])
        if not math.isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError(
                f"training.lr_scheduler.{name} must be finite and in (0, 1]"
            )
        normalized[name] = value

    step_size = normalized["step_size"]
    if isinstance(step_size, bool):
        raise ValueError("training.lr_scheduler.step_size must be positive")
    step_size = int(step_size)
    if step_size <= 0:
        raise ValueError("training.lr_scheduler.step_size must be positive")
    normalized["step_size"] = step_size

    step_gamma = float(normalized["step_gamma"])
    if not math.isfinite(step_gamma) or not 0.0 < step_gamma < 1.0:
        raise ValueError("training.lr_scheduler.step_gamma must be in (0, 1)")
    normalized["step_gamma"] = step_gamma

    plateau_metric = str(normalized["plateau_metric"]).strip()
    if not plateau_metric:
        raise ValueError("training.lr_scheduler.plateau_metric must not be empty")
    normalized["plateau_metric"] = plateau_metric

    plateau_mode = str(normalized["plateau_mode"]).lower()
    if plateau_mode not in LR_PLATEAU_MODES:
        raise ValueError("training.lr_scheduler.plateau_mode must be 'min' or 'max'")
    normalized["plateau_mode"] = plateau_mode

    plateau_factor = float(normalized["plateau_factor"])
    if not math.isfinite(plateau_factor) or not 0.0 < plateau_factor < 1.0:
        raise ValueError("training.lr_scheduler.plateau_factor must be in (0, 1)")
    normalized["plateau_factor"] = plateau_factor

    for name in ("plateau_patience", "plateau_cooldown"):
        value = normalized[name]
        if isinstance(value, bool):
            raise ValueError(f"training.lr_scheduler.{name} must be nonnegative")
        value = int(value)
        if value < 0:
            raise ValueError(f"training.lr_scheduler.{name} must be nonnegative")
        normalized[name] = value

    plateau_threshold = float(normalized["plateau_threshold"])
    if not math.isfinite(plateau_threshold) or plateau_threshold < 0.0:
        raise ValueError(
            "training.lr_scheduler.plateau_threshold must be finite and nonnegative"
        )
    normalized["plateau_threshold"] = plateau_threshold

    threshold_mode = str(normalized["plateau_threshold_mode"]).lower()
    if threshold_mode not in LR_PLATEAU_THRESHOLD_MODES:
        raise ValueError(
            "training.lr_scheduler.plateau_threshold_mode must be 'rel' or 'abs'"
        )
    normalized["plateau_threshold_mode"] = threshold_mode
    return normalized


def resolve_warmup_steps(config, total_training_steps):
    total_training_steps = int(total_training_steps)
    if total_training_steps <= 0:
        raise ValueError("total_training_steps must be positive")
    if config["warmup_steps"] is not None:
        warmup_steps = int(config["warmup_steps"])
    else:
        fraction = float(config["warmup_fraction"])
        warmup_steps = int(total_training_steps * fraction)
        if fraction > 0.0 and total_training_steps > 1:
            warmup_steps = max(warmup_steps, 1)
    if warmup_steps > total_training_steps:
        raise ValueError(
            "training.lr_scheduler.warmup_steps cannot exceed training.epochs"
        )
    return warmup_steps


class LearningRateController:
    """Small, serializable LR controller driven by optimizer updates.

    Warmup is independent of the post-warmup policy. Cosine and step schedules
    advance after every optimizer update. Plateau schedules advance only when a
    configured evaluation metric is present.
    """

    def __init__(
        self,
        optimizer,
        *,
        group_names: Sequence[str],
        total_training_steps: int,
        config=None,
    ):
        self.optimizer = optimizer
        self.config = normalize_lr_scheduler_config(config)
        self.total_training_steps = int(total_training_steps)
        if self.total_training_steps <= 0:
            raise ValueError("total_training_steps must be positive")
        self.group_names = tuple(str(name) for name in group_names)
        if len(self.group_names) != len(self.optimizer.param_groups):
            raise ValueError("one LR group name is required per optimizer parameter group")
        if len(set(self.group_names)) != len(self.group_names):
            raise ValueError("LR group names must be unique")
        self.base_lrs = tuple(float(group["lr"]) for group in optimizer.param_groups)
        if any(not math.isfinite(value) or value <= 0.0 for value in self.base_lrs):
            raise ValueError("optimizer learning rates must be finite and positive")

        self.warmup_steps = resolve_warmup_steps(
            self.config,
            self.total_training_steps,
        )
        self.optimizer_steps = 0
        self.lr_factor = self._update_factor(0)
        self.best_metric = None
        self.bad_evaluations = 0
        self.cooldown_remaining = 0
        self.reductions = 0
        self.last_metric_name = None
        self.last_metric_value = None
        self._apply_factor(self.lr_factor)

    @property
    def scheduler_type(self):
        return self.config["type"]

    def _warmup_factor(self, step):
        if self.warmup_steps <= 0 or step >= self.warmup_steps:
            return 1.0
        start = float(self.config["warmup_start_ratio"])
        progress = float(step) / float(self.warmup_steps)
        return start + (1.0 - start) * progress

    def _update_factor(self, step):
        step = int(step)
        if step < self.warmup_steps:
            return self._warmup_factor(step)
        scheduler_type = self.scheduler_type
        minimum = float(self.config["min_lr_ratio"])
        if scheduler_type in {"constant", "plateau"}:
            return 1.0
        post_warmup_steps = step - self.warmup_steps
        if scheduler_type == "step":
            exponent = post_warmup_steps // int(self.config["step_size"])
            return max(float(self.config["step_gamma"]) ** exponent, minimum)
        decay_steps = max(self.total_training_steps - self.warmup_steps - 1, 1)
        progress = min(max(post_warmup_steps / decay_steps, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum + (1.0 - minimum) * cosine

    def _apply_factor(self, factor):
        self.lr_factor = float(factor)
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * self.lr_factor

    def step_update(self):
        self.optimizer_steps += 1
        if self.scheduler_type == "plateau" and self.optimizer_steps >= self.warmup_steps:
            # Preserve reductions already made by metric steps.
            if self.optimizer_steps == self.warmup_steps:
                self._apply_factor(max(self.lr_factor, 1.0))
            return
        self._apply_factor(self._update_factor(self.optimizer_steps))

    def _is_improvement(self, value):
        if self.best_metric is None:
            return True
        threshold = float(self.config["plateau_threshold"])
        if self.config["plateau_threshold_mode"] == "rel":
            if self.config["plateau_mode"] == "min":
                boundary = self.best_metric * (1.0 - threshold)
                return value < boundary
            boundary = self.best_metric * (1.0 + threshold)
            return value > boundary
        if self.config["plateau_mode"] == "min":
            return value < self.best_metric - threshold
        return value > self.best_metric + threshold

    def _resolve_metric(self, metrics):
        configured = self.config["plateau_metric"]
        candidates = AUTO_PLATEAU_METRICS if configured == "auto" else (configured,)
        for name in candidates:
            if name not in metrics:
                continue
            try:
                value = float(metrics[name])
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                return name, value
        return None, None

    def step_metric(self, metrics):
        """Advance a plateau schedule when its evaluation metric is available."""

        if self.scheduler_type != "plateau":
            return self.metrics()
        name, value = self._resolve_metric(metrics)
        if name is None:
            result = self.metrics()
            result["lr/plateau_metric_available"] = False
            return result
        self.last_metric_name = name
        self.last_metric_value = value
        if self.optimizer_steps < self.warmup_steps:
            result = self.metrics()
            result["lr/plateau_metric_available"] = True
            result["lr/plateau_warmup_ignored"] = True
            return result

        if self._is_improvement(value):
            self.best_metric = value
            self.bad_evaluations = 0
        elif self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            self.bad_evaluations = 0
        else:
            self.bad_evaluations += 1
            if self.bad_evaluations > int(self.config["plateau_patience"]):
                new_factor = max(
                    self.lr_factor * float(self.config["plateau_factor"]),
                    float(self.config["min_lr_ratio"]),
                )
                if new_factor < self.lr_factor:
                    self._apply_factor(new_factor)
                    self.reductions += 1
                    self.cooldown_remaining = int(self.config["plateau_cooldown"])
                self.bad_evaluations = 0
        result = self.metrics()
        result["lr/plateau_metric_available"] = True
        return result

    def metrics(self):
        result = {
            "lr/scheduler_type": self.scheduler_type,
            "lr/optimizer_step": int(self.optimizer_steps),
            "lr/factor": float(self.lr_factor),
            "lr/warmup_steps": int(self.warmup_steps),
        }
        for name, group in zip(self.group_names, self.optimizer.param_groups):
            result[f"lr/{name}"] = float(group["lr"])
        if self.scheduler_type == "plateau":
            result.update(
                {
                    "lr/plateau_bad_evaluations": int(self.bad_evaluations),
                    "lr/plateau_reductions": int(self.reductions),
                    "lr/plateau_best": self.best_metric,
                    "lr/plateau_metric_name": self.last_metric_name,
                    "lr/plateau_metric_value": self.last_metric_value,
                }
            )
        return result

    def metadata(self):
        return {
            **copy.deepcopy(self.config),
            "total_training_steps": int(self.total_training_steps),
            "resolved_warmup_steps": int(self.warmup_steps),
            "parameter_groups": [
                {"name": name, "base_lr": float(base_lr)}
                for name, base_lr in zip(self.group_names, self.base_lrs)
            ],
        }

    def state_dict(self):
        return {
            "version": 1,
            "config": copy.deepcopy(self.config),
            "total_training_steps": int(self.total_training_steps),
            "warmup_steps": int(self.warmup_steps),
            "group_names": list(self.group_names),
            "base_lrs": list(self.base_lrs),
            "optimizer_steps": int(self.optimizer_steps),
            "lr_factor": float(self.lr_factor),
            "best_metric": self.best_metric,
            "bad_evaluations": int(self.bad_evaluations),
            "cooldown_remaining": int(self.cooldown_remaining),
            "reductions": int(self.reductions),
            "last_metric_name": self.last_metric_name,
            "last_metric_value": self.last_metric_value,
        }

    def load_state_dict(self, state):
        if int(state.get("version", 0)) != 1:
            raise ValueError("unsupported learning-rate scheduler checkpoint version")
        checkpoint_config = normalize_lr_scheduler_config(state.get("config"))
        if checkpoint_config != self.config:
            raise ValueError(
                "checkpoint learning-rate scheduler configuration does not match "
                "the current training configuration"
            )
        if int(state.get("total_training_steps", -1)) != self.total_training_steps:
            raise ValueError("checkpoint total training steps do not match")
        if int(state.get("warmup_steps", -1)) != self.warmup_steps:
            raise ValueError("checkpoint resolved warmup steps do not match")
        if tuple(state.get("group_names", ())) != self.group_names:
            raise ValueError("checkpoint learning-rate parameter groups do not match")
        checkpoint_base_lrs = tuple(float(value) for value in state.get("base_lrs", ()))
        if checkpoint_base_lrs != self.base_lrs:
            raise ValueError("checkpoint base learning rates do not match")
        self.optimizer_steps = int(state["optimizer_steps"])
        if not 0 <= self.optimizer_steps <= self.total_training_steps:
            raise ValueError("checkpoint optimizer step is outside the configured schedule")
        self.best_metric = state.get("best_metric")
        self.bad_evaluations = int(state.get("bad_evaluations", 0))
        self.cooldown_remaining = int(state.get("cooldown_remaining", 0))
        self.reductions = int(state.get("reductions", 0))
        self.last_metric_name = state.get("last_metric_name")
        self.last_metric_value = state.get("last_metric_value")
        factor = float(state["lr_factor"])
        if not math.isfinite(factor) or factor <= 0.0 or factor > 1.0:
            raise ValueError("checkpoint learning-rate factor is invalid")
        self._apply_factor(factor)

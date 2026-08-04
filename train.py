import argparse
import copy
import json
import math
import numbers
import os
import pickle
import random
import re

import numpy as np
import torch
import yaml

try:
    import wandb
except ImportError:
    wandb = None

try:
    from .env import SimpleARGEnvironment, action_as_dict
    from .rollout_worker_arg import RolloutWorker
    from .tb_gfn import TBGFlowNetGenerator
    from .lr_control import (
        DEFAULT_LR_SCHEDULER_CONFIG,
        LR_PLATEAU_MODES,
        LR_PLATEAU_THRESHOLD_MODES,
        LR_SCHEDULER_TYPES,
        normalize_lr_scheduler_config,
        resolve_warmup_steps,
    )
    from .time_env import DEFAULT_TIME_BASIS_COMPONENTS
    from .time_context import TIME_CONTEXT_MODES
    from .recombination_split_bias import (
        DEFAULT_RECOMBINATION_SPLIT_BIAS_CONFIG,
        normalize_recombination_split_bias_config,
    )
    from .cwr_event_gate import (
        DEFAULT_LOCAL_CWR_EVENT_GATE_CONFIG,
        normalize_local_cwr_event_gate_config,
    )
    from .hybrid_replay_config import (
        DEFAULT_HYBRID_REPLAY_CONFIG,
        normalize_hybrid_replay_config,
    )
    from .utils import (
        VCF_PARSER_VERSION,
        is_vcf_path,
        load_sequences,
        load_vcf_variants,
        validate_local_refinement_span,
    )
except ImportError:  # Support the repository's script-style entry points.
    from env import SimpleARGEnvironment, action_as_dict
    from rollout_worker_arg import RolloutWorker
    from tb_gfn import TBGFlowNetGenerator
    from lr_control import (
        DEFAULT_LR_SCHEDULER_CONFIG,
        LR_PLATEAU_MODES,
        LR_PLATEAU_THRESHOLD_MODES,
        LR_SCHEDULER_TYPES,
        normalize_lr_scheduler_config,
        resolve_warmup_steps,
    )
    from time_env import DEFAULT_TIME_BASIS_COMPONENTS
    from time_context import TIME_CONTEXT_MODES
    from recombination_split_bias import (
        DEFAULT_RECOMBINATION_SPLIT_BIAS_CONFIG,
        normalize_recombination_split_bias_config,
    )
    from cwr_event_gate import (
        DEFAULT_LOCAL_CWR_EVENT_GATE_CONFIG,
        normalize_local_cwr_event_gate_config,
    )
    from hybrid_replay_config import (
        DEFAULT_HYBRID_REPLAY_CONFIG,
        normalize_hybrid_replay_config,
    )
    from utils import (
        VCF_PARSER_VERSION,
        is_vcf_path,
        load_sequences,
        load_vcf_variants,
        validate_local_refinement_span,
    )


DEFAULT_NE = 10000
DEFAULT_R_PER_BP = 2e-8
DEFAULT_MU_PER_BP = 2e-8
DEFAULT_INIT_Z_SAMPLE_COUNT = 16
DEFAULT_POLICY_LR = 1e-3
DEFAULT_LOG_Z_LR = 1e-3
DEFAULT_GRAD_CLIP = 10.0
DEFAULT_GRAD_ACCUM_STEPS = 1
DEFAULT_EVAL_EPISODES = 8
DEFAULT_EVAL_EVERY = 10
DEFAULT_LOSS = "tb"
DEFAULT_SUBTB_LAMBDA = 0.9
DEFAULT_SUBTB_MAX_SPAN = None
DEFAULT_PARTIAL_SEGMENT_MAX_STEPS = 16
DEFAULT_PARTIAL_START_MODE = "initial"
DEFAULT_PARTIAL_BOUNDARY_FRACTION = 0.5
PARTIAL_START_MODES = ("initial", "terminal_prefix_mixture")
DEFAULT_TERMINAL_BACKTRACK_LENGTHS = (5, 10, 25)
DEFAULT_EMBEDDING_SIZE = 32
DEFAULT_HIDDEN_SIZE = 64
DEFAULT_DROPOUT = 0.0
DEFAULT_BREAKPOINT_HIDDEN_DIM = 128
DEFAULT_BREAKPOINT_DROPOUT = 0.1
DEFAULT_TRANSFORMER_DEPTH = 6
DEFAULT_TRANSFORMER_HEADS = 4
DEFAULT_TRANSFORMER_MLP_RATIO = 2.0
DEFAULT_ATTENTION_DROPOUT = 0.0
DEFAULT_TIME_HIDDEN_SIZE = 256
DEFAULT_TIME_LAYERS = 3
DEFAULT_TIME_DROPOUT = 0.0
DEFAULT_TIME_CONTEXT_MODE = "baseline"
DEFAULT_BREAKPOINT_GAP_HIDDEN_SIZE = 256
DEFAULT_BREAKPOINT_GAP_LAYERS = 3
DEFAULT_BREAKPOINT_GAP_DROPOUT = 0.0
DEFAULT_BREAKPOINT_USE_POSITION_FEATURES = True
DEFAULT_LOCAL_COALESCENCE_SIMILARITY_BIAS = 0.0
DEFAULT_LOCAL_PRIOR_ACTION_LOGIT_BIAS = 0.0
DEFAULT_LOCAL_PRIOR_GATE_LOGIT_BIAS = 0.0
MODEL_VERSION = "cwr-event-continuous-time-v2"


DEFAULT_CONFIG = {
    "dataset_path": None,
    "output_path": None,
    "device": "auto",
    "refinement": {
        "enabled": False,
        "checkpoint": None,
        "arg_path": None,
        "requests": [],
        "terminal_requires_exhausted_fixed_schedule": False,
    },
    "training": {
        "epochs": None,
        "batch_size": 10,
        "seed": 7,
        "init_z_sample_count": DEFAULT_INIT_Z_SAMPLE_COUNT,
        "verbose": False,
        "wandb": True,
        "policy_lr": DEFAULT_POLICY_LR,
        "breakpoint_policy_lr": None,
        "time_policy_lr": None,
        "log_z_lr": DEFAULT_LOG_Z_LR,
        "lr_scheduler": copy.deepcopy(DEFAULT_LR_SCHEDULER_CONFIG),
        "loss": DEFAULT_LOSS,
        "subtb_lambda": DEFAULT_SUBTB_LAMBDA,
        "subtb_max_span": DEFAULT_SUBTB_MAX_SPAN,
        "subtb_lambda_initial": None,
        "subtb_lambda_final": None,
        "subtb_max_span_schedule": [],
        "terminal_loss_weight": 1.0,
        "residual_scale": 1.0,
        "grad_clip": DEFAULT_GRAD_CLIP,
        "breakpoint_gradient_clip_norm": None,
        "time_head_gradient_clip_norm": None,
        "time_head_warmup_epochs": 0,
        "model_diagnostics": True,
        "model_diagnostics_update_norm_every": 1,
        "grad_accum_steps": DEFAULT_GRAD_ACCUM_STEPS,
        "min_terminal_trajectories_per_batch": 0,
        "flow_debug": False,
        "flow_debug_max_records": 16,
        "probability_checks": False,
        "fixed_eval_bank_path": None,
        "fixed_eval_bank_episodes": 0,
        "fixed_eval_bank_seed": 100007,
        "fixed_eval_bank_source": "baseline",
        "fixed_eval_bank_create_if_missing": False,
        "eval_episodes": DEFAULT_EVAL_EPISODES,
        "eval_every": DEFAULT_EVAL_EVERY,
        "partial_segment_max_steps": DEFAULT_PARTIAL_SEGMENT_MAX_STEPS,
        "partial_start_mode": DEFAULT_PARTIAL_START_MODE,
        "partial_boundary_fraction": DEFAULT_PARTIAL_BOUNDARY_FRACTION,
        "trajectory_training_mode": "complete",
        "complete_trajectory_max_steps": None,
        "selection_margin": 1e-6,
        "min_valid_splice_rate": 0.90,
        "min_unique_topology_rate": 0.25,
        "hybrid_replay": copy.deepcopy(DEFAULT_HYBRID_REPLAY_CONFIG),
    },
    "environment": {
        "bp_per_blocks": 1,
        "effective_population_size": DEFAULT_NE,
        "mutation_rate": DEFAULT_MU_PER_BP,
        "recombination_rate": DEFAULT_R_PER_BP,
    },
    "reward": {
        "constant": 30000,
    },
    "model": {
        "embedding_size": DEFAULT_EMBEDDING_SIZE,
        "hidden_size": DEFAULT_HIDDEN_SIZE,
        "dropout": DEFAULT_DROPOUT,
        "breakpoint_hidden_dim": DEFAULT_BREAKPOINT_HIDDEN_DIM,
        "breakpoint_dropout": DEFAULT_BREAKPOINT_DROPOUT,
        "transformer_depth": DEFAULT_TRANSFORMER_DEPTH,
        "transformer_heads": DEFAULT_TRANSFORMER_HEADS,
        "transformer_mlp_ratio": DEFAULT_TRANSFORMER_MLP_RATIO,
        "attention_dropout": DEFAULT_ATTENTION_DROPOUT,
        "time_basis_components": DEFAULT_TIME_BASIS_COMPONENTS,
        "time_context_mode": DEFAULT_TIME_CONTEXT_MODE,
        "local_coalescence_similarity_bias": DEFAULT_LOCAL_COALESCENCE_SIMILARITY_BIAS,
        "local_prior_action_logit_bias": DEFAULT_LOCAL_PRIOR_ACTION_LOGIT_BIAS,
        "local_prior_gate_logit_bias": DEFAULT_LOCAL_PRIOR_GATE_LOGIT_BIAS,
        "recombination_split_bias": copy.deepcopy(
            DEFAULT_RECOMBINATION_SPLIT_BIAS_CONFIG
        ),
        "local_cwr_event_gate": copy.deepcopy(
            DEFAULT_LOCAL_CWR_EVENT_GATE_CONFIG
        ),
    },
}


CLI_CONFIG_PATHS = {
    "dataset_path": ("dataset_path",),
    "output_path": ("output_path",),
    "device": ("device",),
    "checkpoint": ("refinement", "checkpoint"),
    "local_refinement_arg": ("refinement", "arg_path"),
    "bad_region_top_k": ("refinement", "bad_region_top_k"),
    "bad_region_blocks": ("refinement", "bad_region_blocks"),
    "bad_region_bp": ("refinement", "bad_region_bp"),
    "refine_strategy": ("refinement", "strategy"),
    "terminal_backtrack_lengths": ("refinement", "terminal_backtrack_lengths"),
    "epochs": ("training", "epochs"),
    "batch_size": ("training", "batch_size"),
    "seed": ("training", "seed"),
    "init_z_sample_count": ("training", "init_z_sample_count"),
    "verbose": ("training", "verbose"),
    "wandb": ("training", "wandb"),
    "policy_lr": ("training", "policy_lr"),
    "breakpoint_policy_lr": ("training", "breakpoint_policy_lr"),
    "time_policy_lr": ("training", "time_policy_lr"),
    "log_z_lr": ("training", "log_z_lr"),
    "lr_scheduler_type": ("training", "lr_scheduler", "type"),
    "lr_warmup_steps": ("training", "lr_scheduler", "warmup_steps"),
    "lr_warmup_fraction": ("training", "lr_scheduler", "warmup_fraction"),
    "lr_warmup_start_ratio": (
        "training", "lr_scheduler", "warmup_start_ratio"
    ),
    "lr_min_ratio": ("training", "lr_scheduler", "min_lr_ratio"),
    "lr_step_size": ("training", "lr_scheduler", "step_size"),
    "lr_step_gamma": ("training", "lr_scheduler", "step_gamma"),
    "lr_plateau_metric": ("training", "lr_scheduler", "plateau_metric"),
    "lr_plateau_mode": ("training", "lr_scheduler", "plateau_mode"),
    "lr_plateau_factor": ("training", "lr_scheduler", "plateau_factor"),
    "lr_plateau_patience": (
        "training", "lr_scheduler", "plateau_patience"
    ),
    "lr_plateau_threshold": (
        "training", "lr_scheduler", "plateau_threshold"
    ),
    "lr_plateau_threshold_mode": (
        "training", "lr_scheduler", "plateau_threshold_mode"
    ),
    "lr_plateau_cooldown": (
        "training", "lr_scheduler", "plateau_cooldown"
    ),
    "loss": ("training", "loss"),
    "subtb_lambda": ("training", "subtb_lambda"),
    "subtb_max_span": ("training", "subtb_max_span"),
    "terminal_loss_weight": ("training", "terminal_loss_weight"),
    "residual_scale": ("training", "residual_scale"),
    "grad_clip": ("training", "grad_clip"),
    "breakpoint_gradient_clip_norm": (
        "training", "breakpoint_gradient_clip_norm"
    ),
    "time_head_gradient_clip_norm": ("training", "time_head_gradient_clip_norm"),
    "time_head_warmup_epochs": ("training", "time_head_warmup_epochs"),
    "model_diagnostics": ("training", "model_diagnostics"),
    "model_diagnostics_update_norm_every": (
        "training", "model_diagnostics_update_norm_every"
    ),
    "min_terminal_trajectories_per_batch": (
        "training", "min_terminal_trajectories_per_batch"
    ),
    "grad_accum_steps": ("training", "grad_accum_steps"),
    "eval_episodes": ("training", "eval_episodes"),
    "eval_every": ("training", "eval_every"),
    "partial_segment_max_steps": ("training", "partial_segment_max_steps"),
    "partial_start_mode": ("training", "partial_start_mode"),
    "partial_boundary_fraction": ("training", "partial_boundary_fraction"),
    "trajectory_training_mode": ("training", "trajectory_training_mode"),
    "complete_trajectory_max_steps": (
        "training", "complete_trajectory_max_steps"
    ),
    "selection_margin": ("training", "selection_margin"),
    "min_valid_splice_rate": ("training", "min_valid_splice_rate"),
    "min_unique_topology_rate": ("training", "min_unique_topology_rate"),
    "bp_per_blocks": ("environment", "bp_per_blocks"),
    "effective_population_size": ("environment", "effective_population_size"),
    "mutation_rate": ("environment", "mutation_rate"),
    "recombination_rate": ("environment", "recombination_rate"),
    "reward_constant": ("reward", "constant"),
    "embedding_size": ("model", "embedding_size"),
    "hidden_size": ("model", "hidden_size"),
    "dropout": ("model", "dropout"),
    "breakpoint_hidden_dim": ("model", "breakpoint_hidden_dim"),
    "breakpoint_dropout": ("model", "breakpoint_dropout"),
    "transformer_depth": ("model", "transformer_depth"),
    "transformer_heads": ("model", "transformer_heads"),
    "transformer_mlp_ratio": ("model", "transformer_mlp_ratio"),
    "attention_dropout": ("model", "attention_dropout"),
    "time_basis_components": ("model", "time_basis_components"),
    "time_context_mode": ("model", "time_context_mode"),
    "local_coalescence_similarity_bias": (
        "model",
        "local_coalescence_similarity_bias",
    ),
    "local_prior_action_logit_bias": (
        "model",
        "local_prior_action_logit_bias",
    ),
    "local_prior_gate_logit_bias": (
        "model",
        "local_prior_gate_logit_bias",
    ),
}

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_train_config(config_path=None):
    config = copy.deepcopy(DEFAULT_CONFIG)
    if config_path is None:
        return config
    with open(config_path, "r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError("YAML training config must contain a mapping at the top level")
    return _deep_merge(config, loaded)


def _deep_merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def apply_cli_overrides(config, args):
    merged = copy.deepcopy(config)
    for arg_name, path in CLI_CONFIG_PATHS.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            _set_nested(merged, path, value)
    return merged


def _set_nested(config, path, value):
    cursor = config
    for key in path[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[path[-1]] = value


def parse_positive_int_list(value, field_name, allow_empty=True):
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return []
        raw_items = [item.strip() for item in text.split(",")]
    elif isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    else:
        raw_items = [value]

    parsed = []
    for item in raw_items:
        if item is None or str(item).strip() == "":
            continue
        try:
            parsed_value = int(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must contain only positive integers") from exc
        if parsed_value <= 0:
            raise ValueError(f"{field_name} must contain only positive integers")
        parsed.append(parsed_value)
    if not parsed and not allow_empty:
        raise ValueError(f"{field_name} must contain at least one positive integer")
    return parsed


def parse_optional_positive_int(value, field_name):
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    try:
        parsed_value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer or null") from exc
    if parsed_value <= 0:
        raise ValueError(f"{field_name} must be a positive integer or null")
    return parsed_value


def validate_train_config(config):
    missing = []
    if not config.get("dataset_path"):
        missing.append("dataset_path")
    if not config.get("output_path"):
        missing.append("output_path")
    if config.get("training", {}).get("epochs") is None:
        missing.append("training.epochs")
    if missing:
        raise ValueError(
            "Missing required training config value(s): "
            + ", ".join(missing)
            + ". Provide them in YAML or via CLI flags."
        )
    training = config.get("training", {})
    environment = config.get("environment", {})
    try:
        epochs = int(training["epochs"])
    except (TypeError, ValueError) as exc:
        raise ValueError("training.epochs must be a positive integer") from exc
    if epochs <= 0:
        raise ValueError("training.epochs must be a positive integer")
    training["epochs"] = epochs
    for field_name in ("policy_lr", "log_z_lr"):
        try:
            value = float(training[field_name])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"training.{field_name} must be positive") from exc
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"training.{field_name} must be finite and positive")
        training[field_name] = value
    for field_name in ("breakpoint_policy_lr", "time_policy_lr"):
        value = training.get(field_name)
        if value is not None:
            try:
                value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"training.{field_name} must be positive or null"
                ) from exc
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"training.{field_name} must be finite and positive or null"
                )
        training[field_name] = value
    training["lr_scheduler"] = normalize_lr_scheduler_config(
        training.get("lr_scheduler")
    )
    resolve_warmup_steps(training["lr_scheduler"], epochs)
    legacy_time_fields = sorted(
        key
        for key in ("time_bins", "time_delta_bin_width", "time_bin_scheme")
        if key in environment
    )
    if legacy_time_fields:
        raise ValueError(
            "Fixed-width time bins were removed in continuous-time v2. "
            "Delete "
            + ", ".join(
                f"environment.{key}" for key in legacy_time_fields
            )
            + "; the CWR waiting-time law now determines event times."
        )
    model = config.get("model", {})
    time_basis_components = int(
        model.get(
            "time_basis_components",
            DEFAULT_TIME_BASIS_COMPONENTS,
        )
    )
    if time_basis_components < 2:
        raise ValueError("model.time_basis_components must be at least 2")
    model["time_basis_components"] = time_basis_components
    time_context_mode = str(
        model.get("time_context_mode", DEFAULT_TIME_CONTEXT_MODE)
    ).lower()
    if time_context_mode not in TIME_CONTEXT_MODES:
        raise ValueError(
            "model.time_context_mode must be one of "
            + ", ".join(repr(value) for value in TIME_CONTEXT_MODES)
        )
    model["time_context_mode"] = time_context_mode
    for field_name in (
        "local_coalescence_similarity_bias",
        "local_prior_action_logit_bias",
        "local_prior_gate_logit_bias",
    ):
        try:
            model[field_name] = float(
                model.get(field_name, DEFAULT_CONFIG["model"][field_name])
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"model.{field_name} must be a number") from exc
    model["recombination_split_bias"] = (
        normalize_recombination_split_bias_config(
            model.get("recombination_split_bias")
        )
    )
    model["local_cwr_event_gate"] = normalize_local_cwr_event_gate_config(
        model.get("local_cwr_event_gate")
    )
    if model["recombination_split_bias"]["enabled"]:
        refinement_config = config.get("refinement", {})
        if not bool(
            refinement_config.get("enabled")
            or refinement_config.get("arg_path")
            or refinement_config.get("requests")
        ):
            raise ValueError(
                "model.recombination_split_bias can be enabled only for local "
                "VCF ARG refinement"
            )
        if not is_vcf_path(config.get("dataset_path")):
            raise ValueError(
                "model.recombination_split_bias requires a VCF dataset"
            )
    if model["local_cwr_event_gate"]["enabled"]:
        refinement_config = config.get("refinement", {})
        if not bool(
            refinement_config.get("enabled")
            or refinement_config.get("arg_path")
            or refinement_config.get("requests")
        ):
            raise ValueError(
                "model.local_cwr_event_gate can be enabled only for local VCF "
                "ARG refinement"
            )
        if not is_vcf_path(config.get("dataset_path")):
            raise ValueError(
                "model.local_cwr_event_gate requires a VCF dataset"
            )
    loss = str(training.get("loss", DEFAULT_LOSS)).lower()
    if loss not in {"tb", "subtb", "fl_subtb"}:
        raise ValueError("training.loss must be one of 'tb', 'subtb', or 'fl_subtb'")
    for field_name in ("batch_size", "grad_accum_steps"):
        try:
            parsed_value = int(training.get(field_name))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"training.{field_name} must be a positive integer") from exc
        if parsed_value <= 0:
            raise ValueError(f"training.{field_name} must be a positive integer")
        training[field_name] = parsed_value
    for field_name in ("eval_episodes", "eval_every"):
        try:
            parsed_value = int(training.get(field_name))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"training.{field_name} must be a non-negative integer") from exc
        if parsed_value < 0:
            raise ValueError(f"training.{field_name} must be a non-negative integer")
        training[field_name] = parsed_value
    subtb_lambda = float(training.get("subtb_lambda", DEFAULT_SUBTB_LAMBDA))
    if subtb_lambda <= 0.0:
        raise ValueError("training.subtb_lambda must be positive")
    training["subtb_max_span"] = parse_optional_positive_int(
        training.get("subtb_max_span", DEFAULT_SUBTB_MAX_SPAN),
        "training.subtb_max_span",
    )
    for field_name, strictly_positive in (
        ("terminal_loss_weight", False),
        ("residual_scale", True),
    ):
        try:
            value = float(training.get(field_name, DEFAULT_CONFIG["training"][field_name]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"training.{field_name} must be a number") from exc
        if not math.isfinite(value) or (value <= 0.0 if strictly_positive else value < 0.0):
            qualifier = "positive" if strictly_positive else "nonnegative"
            raise ValueError(f"training.{field_name} must be finite and {qualifier}")
        training[field_name] = value
    for field_name in ("subtb_lambda_initial", "subtb_lambda_final"):
        value = training.get(field_name)
        if value is not None:
            value = float(value)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"training.{field_name} must be positive or null")
        training[field_name] = value
    if (training["subtb_lambda_initial"] is None) != (
        training["subtb_lambda_final"] is None
    ):
        raise ValueError(
            "training.subtb_lambda_initial and subtb_lambda_final must be set together"
        )
    schedule = training.get("subtb_max_span_schedule") or []
    if not isinstance(schedule, list):
        raise ValueError("training.subtb_max_span_schedule must be a list")
    # The generator performs the full ordering/open-ended validation.
    training["subtb_max_span_schedule"] = [dict(row) for row in schedule]
    for field_name in (
        "breakpoint_gradient_clip_norm",
        "time_head_gradient_clip_norm",
    ):
        training[field_name] = (
            None
            if training.get(field_name) is None
            else float(training[field_name])
        )
        if training[field_name] is not None and (
            not math.isfinite(training[field_name])
            or training[field_name] <= 0.0
        ):
            raise ValueError(
                f"training.{field_name} must be finite and positive or null"
            )
    for field_name in (
        "time_head_warmup_epochs",
        "min_terminal_trajectories_per_batch",
        "flow_debug_max_records",
    ):
        value = int(training.get(field_name, DEFAULT_CONFIG["training"][field_name]))
        if value < 0:
            raise ValueError(f"training.{field_name} must be nonnegative")
        training[field_name] = value
    update_norm_every = int(
        training.get("model_diagnostics_update_norm_every", 1)
    )
    if update_norm_every <= 0:
        raise ValueError(
            "training.model_diagnostics_update_norm_every must be positive"
        )
    training["model_diagnostics_update_norm_every"] = update_norm_every
    for field_name in (
        "flow_debug",
        "probability_checks",
        "model_diagnostics",
    ):
        if not isinstance(training.get(field_name), bool):
            raise ValueError(f"training.{field_name} must be true or false")
    for field_name in ("fixed_eval_bank_episodes", "fixed_eval_bank_seed"):
        value = int(training.get(field_name, DEFAULT_CONFIG["training"][field_name]))
        if field_name == "fixed_eval_bank_episodes" and value < 0:
            raise ValueError("training.fixed_eval_bank_episodes must be nonnegative")
        training[field_name] = value
    if not isinstance(training.get("fixed_eval_bank_create_if_missing"), bool):
        raise ValueError(
            "training.fixed_eval_bank_create_if_missing must be true or false"
        )
    training["fixed_eval_bank_source"] = str(
        training.get("fixed_eval_bank_source", "baseline")
    )
    partial_segment_max_steps = int(
        training.get("partial_segment_max_steps", DEFAULT_PARTIAL_SEGMENT_MAX_STEPS)
    )
    if partial_segment_max_steps <= 0:
        raise ValueError("training.partial_segment_max_steps must be positive")
    training["partial_segment_max_steps"] = partial_segment_max_steps
    partial_start_mode = str(
        training.get("partial_start_mode", DEFAULT_PARTIAL_START_MODE)
    ).lower()
    if partial_start_mode not in PARTIAL_START_MODES:
        raise ValueError(
            "training.partial_start_mode must be one of "
            + ", ".join(repr(value) for value in PARTIAL_START_MODES)
        )
    training["partial_start_mode"] = partial_start_mode
    try:
        partial_boundary_fraction = float(
            training.get(
                "partial_boundary_fraction",
                DEFAULT_PARTIAL_BOUNDARY_FRACTION,
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "training.partial_boundary_fraction must be a number between 0 and 1"
        ) from exc
    if not 0.0 <= partial_boundary_fraction <= 1.0:
        raise ValueError(
            "training.partial_boundary_fraction must be between 0 and 1"
        )
    training["partial_boundary_fraction"] = partial_boundary_fraction
    trajectory_training_mode = str(
        training.get("trajectory_training_mode", "complete")
    ).lower()
    if trajectory_training_mode != "complete":
        raise ValueError(
            "partial trajectory training is temporarily disabled; "
            "training.trajectory_training_mode must be 'complete'"
        )
    training["trajectory_training_mode"] = trajectory_training_mode
    training["complete_trajectory_max_steps"] = parse_optional_positive_int(
        training.get("complete_trajectory_max_steps"),
        "training.complete_trajectory_max_steps",
    )
    training["hybrid_replay"] = normalize_hybrid_replay_config(
        training.get("hybrid_replay")
    )
    if training["hybrid_replay"]["enabled"] and not refinement_enabled(config):
        raise ValueError(
            "training.hybrid_replay can be enabled only for local refinement"
        )
    for field_name, default in (
        ("selection_margin", 1e-6),
        ("min_valid_splice_rate", 0.90),
        ("min_unique_topology_rate", 0.25),
    ):
        try:
            value = float(training.get(field_name, default))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"training.{field_name} must be a number") from exc
        if field_name == "selection_margin":
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("training.selection_margin must be finite and nonnegative")
        elif not 0.0 <= value <= 1.0:
            raise ValueError(f"training.{field_name} must be between 0 and 1")
        training[field_name] = value
    refinement = config.get("refinement", {})
    legacy_fields = {
        "bad_region_top_k",
        "bad_region_blocks",
        "bad_region_bp",
        "strategy",
        "terminal_backtrack_lengths",
    }
    configured_legacy_fields = sorted(
        key
        for key in legacy_fields
        if key in refinement
    )
    if configured_legacy_fields:
        raise ValueError(
            "The automatic/backtracked refinement configuration was removed. "
            "Replace "
            + ", ".join(
                f"refinement.{key}" for key in configured_legacy_fields
            )
            + " with explicit refinement.requests entries containing "
            "genomic_range and exactly one of cut_time or cut_event_index."
        )
    if refinement_enabled(config):
        terminal_requires_fixed = refinement.get(
            "terminal_requires_exhausted_fixed_schedule",
            False,
        )
        if not isinstance(terminal_requires_fixed, bool):
            raise ValueError(
                "refinement.terminal_requires_exhausted_fixed_schedule must be "
                "true or false"
            )
        refinement[
            "terminal_requires_exhausted_fixed_schedule"
        ] = terminal_requires_fixed
        if not refinement.get("arg_path"):
            raise ValueError(
                "Missing required local refinement config value: "
                "refinement.arg_path."
            )
        if loss != "fl_subtb":
            raise ValueError(
                "Integrated local refinement requires training.loss: fl_subtb"
            )
        requests = refinement.get("requests")
        if not isinstance(requests, list) or not requests:
            raise ValueError(
                "refinement.requests must be a non-empty list of explicit "
                "interval/time requests"
            )
        normalized_requests = []
        seen_ids = set()
        for index, request in enumerate(requests):
            if not isinstance(request, dict):
                raise ValueError(
                    f"refinement.requests[{index}] must be a mapping"
                )
            request_id = str(
                request.get("id") or f"region_{index + 1:06d}"
            )
            if (
                request_id in {".", ".."}
                or re.fullmatch(r"[A-Za-z0-9_.-]+", request_id) is None
            ):
                raise ValueError(
                    f"refinement.requests[{index}].id may contain only "
                    "letters, numbers, '.', '_' and '-'"
                )
            if request_id in seen_ids:
                raise ValueError(
                    f"duplicate refinement request id {request_id!r}"
                )
            seen_ids.add(request_id)
            genomic_range = request.get("genomic_range")
            if (
                not isinstance(genomic_range, (list, tuple))
                or len(genomic_range) != 2
            ):
                raise ValueError(
                    f"refinement.requests[{index}].genomic_range must be "
                    "a two-value half-open range"
                )
            left, right = (float(value) for value in genomic_range)
            if (
                not math.isfinite(left)
                or not math.isfinite(right)
                or left < 0.0
                or not left < right
            ):
                raise ValueError(
                    f"refinement.requests[{index}].genomic_range must satisfy "
                    "0 <= left < right with finite coordinates"
                )
            validate_local_refinement_span(
                (left, right),
                field_name=f"refinement.requests[{index}].genomic_range",
            )
            supplied = int(request.get("cut_time") is not None) + int(
                request.get("cut_event_index") is not None
            )
            if supplied != 1:
                raise ValueError(
                    f"refinement.requests[{index}] must provide exactly one "
                    "of cut_time or cut_event_index"
                )
            normalized = {
                "id": request_id,
                "genomic_range": [left, right],
            }
            if request.get("cut_time") is not None:
                cut_time = float(request["cut_time"])
                if not math.isfinite(cut_time):
                    raise ValueError(
                        f"refinement.requests[{index}].cut_time must be finite"
                    )
                normalized["cut_time"] = cut_time
            else:
                cut_event_index = request["cut_event_index"]
                if (
                    isinstance(cut_event_index, bool)
                    or not isinstance(cut_event_index, numbers.Integral)
                ):
                    raise ValueError(
                        f"refinement.requests[{index}].cut_event_index must be "
                        "an integer"
                    )
                normalized["cut_event_index"] = int(cut_event_index)
            normalized_requests.append(normalized)
        refinement["requests"] = normalized_requests


def config_to_train_kwargs(config):
    training = config["training"]
    environment = config["environment"]
    reward = config["reward"]
    model = config["model"]
    device = config.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return {
        "dataset_path": config["dataset_path"],
        "output_path": config["output_path"],
        "device": device,
        "bp_per_blocks": environment["bp_per_blocks"],
        "batch_size": training["batch_size"],
        "epochs_num": training["epochs"],
        "seed": training["seed"],
        "init_z_sample_count": training["init_z_sample_count"],
        "use_wandb": training["wandb"],
        "effective_population_size": environment["effective_population_size"],
        "mutation_rate": environment["mutation_rate"],
        "recombination_rate": environment["recombination_rate"],
        "policy_lr": training["policy_lr"],
        "breakpoint_policy_lr": training.get("breakpoint_policy_lr"),
        "time_policy_lr": training.get("time_policy_lr"),
        "log_z_lr": training["log_z_lr"],
        "lr_scheduler_config": copy.deepcopy(training["lr_scheduler"]),
        "loss_mode": str(training["loss"]).lower(),
        "subtb_lambda": training["subtb_lambda"],
        "subtb_max_span": training["subtb_max_span"],
        "subtb_lambda_initial": training["subtb_lambda_initial"],
        "subtb_lambda_final": training["subtb_lambda_final"],
        "subtb_max_span_schedule": training["subtb_max_span_schedule"],
        "terminal_loss_weight": training["terminal_loss_weight"],
        "residual_scale": training["residual_scale"],
        "grad_clip": training["grad_clip"],
        "breakpoint_gradient_clip_norm": training[
            "breakpoint_gradient_clip_norm"
        ],
        "time_head_gradient_clip_norm": training["time_head_gradient_clip_norm"],
        "time_head_warmup_epochs": training["time_head_warmup_epochs"],
        "model_diagnostics": training["model_diagnostics"],
        "model_diagnostics_update_norm_every": training[
            "model_diagnostics_update_norm_every"
        ],
        "grad_accum_steps": training["grad_accum_steps"],
        "min_terminal_trajectories_per_batch": training[
            "min_terminal_trajectories_per_batch"
        ],
        "flow_debug": training["flow_debug"],
        "flow_debug_max_records": training["flow_debug_max_records"],
        "probability_checks": training["probability_checks"],
        "fixed_eval_bank_path": training["fixed_eval_bank_path"],
        "fixed_eval_bank_episodes": training["fixed_eval_bank_episodes"],
        "fixed_eval_bank_seed": training["fixed_eval_bank_seed"],
        "fixed_eval_bank_source": training["fixed_eval_bank_source"],
        "fixed_eval_bank_create_if_missing": training[
            "fixed_eval_bank_create_if_missing"
        ],
        "eval_episodes": training["eval_episodes"],
        "eval_every": training["eval_every"],
        "partial_segment_max_steps": training["partial_segment_max_steps"],
        "partial_start_mode": training["partial_start_mode"],
        "partial_boundary_fraction": training["partial_boundary_fraction"],
        "trajectory_training_mode": training["trajectory_training_mode"],
        "complete_trajectory_max_steps": training["complete_trajectory_max_steps"],
        "hybrid_replay_config": copy.deepcopy(training["hybrid_replay"]),
        "selection_margin": training["selection_margin"],
        "min_valid_splice_rate": training["min_valid_splice_rate"],
        "min_unique_topology_rate": training["min_unique_topology_rate"],
        "reward_C": reward["constant"],
        "embedding_size": model["embedding_size"],
        "hidden_size": model["hidden_size"],
        "dropout": model["dropout"],
        "breakpoint_hidden_dim": model["breakpoint_hidden_dim"],
        "breakpoint_dropout": model["breakpoint_dropout"],
        "transformer_depth": model["transformer_depth"],
        "transformer_heads": model["transformer_heads"],
        "transformer_mlp_ratio": model["transformer_mlp_ratio"],
        "attention_dropout": model["attention_dropout"],
        "time_basis_components": model["time_basis_components"],
        "time_context_mode": model.get(
            "time_context_mode",
            DEFAULT_TIME_CONTEXT_MODE,
        ),
        "local_coalescence_similarity_bias": model.get(
            "local_coalescence_similarity_bias",
            DEFAULT_LOCAL_COALESCENCE_SIMILARITY_BIAS,
        ),
        "local_prior_action_logit_bias": model.get(
            "local_prior_action_logit_bias",
            DEFAULT_LOCAL_PRIOR_ACTION_LOGIT_BIAS,
        ),
        "local_prior_gate_logit_bias": model.get(
            "local_prior_gate_logit_bias",
            DEFAULT_LOCAL_PRIOR_GATE_LOGIT_BIAS,
        ),
        "recombination_split_bias": copy.deepcopy(
            model.get(
                "recombination_split_bias",
                DEFAULT_RECOMBINATION_SPLIT_BIAS_CONFIG,
            )
        ),
        "local_cwr_event_gate": copy.deepcopy(
            model.get(
                "local_cwr_event_gate",
                DEFAULT_LOCAL_CWR_EVENT_GATE_CONFIG,
            )
        ),
        "verbose": training["verbose"],
    }


def refinement_enabled(config):
    refinement = config.get("refinement", {})
    return bool(
        refinement.get("enabled")
        or refinement.get("arg_path")
        or refinement.get("requests")
    )


def config_to_refinement_kwargs(config):
    refinement = config.get("refinement", {})
    return {
        "checkpoint": refinement.get("checkpoint"),
        "local_refinement_arg": refinement.get("arg_path"),
        "requests": list(refinement.get("requests") or []),
        "terminal_requires_exhausted_fixed_schedule": bool(
            refinement.get(
                "terminal_requires_exhausted_fixed_schedule",
                False,
            )
        ),
    }


def save_resolved_config(config, output_path):
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "config.yaml"), "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

def train_epoch(
    epoch_id,
    rollout_worker,
    generator,
    batch_size=1,
    grad_accum_steps=1,
    start_state_sampler=None,
    rollout_logger=None,
):
    batch_size = int(batch_size)
    grad_accum_steps = int(grad_accum_steps)
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if grad_accum_steps <= 0:
        raise ValueError("grad_accum_steps must be a positive integer")
    rollout_metrics = {}

    for accum_idx in range(grad_accum_steps):
        start_states = None
        action_filter = None
        max_steps = None
        rollout_mode = "terminal"
        rollout_spec = None
        if start_state_sampler is not None:
            rollout_spec = start_state_sampler(batch_size)
            if isinstance(rollout_spec, dict):
                start_states = rollout_spec.get("start_states")
                action_filter = rollout_spec.get("action_filter")
                max_steps = rollout_spec.get("max_steps")
                rollout_mode = str(
                    rollout_spec.get(
                        "rollout_mode",
                        "segment" if max_steps is not None else "terminal",
                    )
                )
            else:
                start_states, action_filter = rollout_spec
        if rollout_logger is not None:
            rollout_logger(
                epoch_id=epoch_id,
                accum_step=accum_idx + 1,
                grad_accum_steps=grad_accum_steps,
                batch_size=batch_size,
                rollout_mode=rollout_mode,
                max_steps=max_steps,
                rollout_spec=rollout_spec,
            )
        ret, trajectories = rollout_worker.rollout(
            generator,
            episodes=batch_size,
            start_states=start_states,
            action_filter=action_filter,
            max_steps=max_steps,
        )
        _record_rollout_metrics(rollout_metrics, rollout_mode, ret)
        generator.accumulate_loss(
            ret,
            factor=grad_accum_steps,
        )

    info = generator.update_model()
    info.update(_summarize_rollout_metrics(rollout_metrics))
    return info


def _record_rollout_metrics(metrics, rollout_mode, rollout_outputs):
    mode = str(rollout_mode)
    entry = metrics.setdefault(
        mode,
        {
            "batches": 0,
            "trajectories": 0,
            "length_sum": 0.0,
            "terminal_sum": 0.0,
            "truncated_sum": 0.0,
            "time_count": 0,
            "time_quantile_sum": 0.0,
            "time_delta_sum": 0.0,
            "time_near_boundary_sum": 0.0,
            "time_finite_density_sum": 0.0,
            "time_entropy_sum": 0.0,
            "time_entropy_count": 0,
            "time_effective_components_sum": 0.0,
            "time_effective_components_count": 0,
            "fixed_attachment_sum": 0,
            "coalescence_sum": 0,
            "recombination_sum": 0,
            "structural_decision_count": 0,
            "structural_action_support_size_sum": 0,
            "structural_action_entropy_sum": 0.0,
            "structural_action_normalized_entropy_sum": 0.0,
            "structural_selected_probability_sum": 0.0,
            "structural_max_probability_sum": 0.0,
            "coalescence_probability_mass_sum": 0.0,
            "recombination_probability_mass_sum": 0.0,
            "breakpoint_decision_count": 0,
            "breakpoint_support_size_sum": 0,
            "breakpoint_entropy_sum": 0.0,
            "breakpoint_normalized_entropy_sum": 0.0,
            "breakpoint_selected_probability_sum": 0.0,
            "breakpoint_max_probability_sum": 0.0,
        },
    )
    lengths = rollout_outputs["trajectory_lengths"].detach().float().cpu()
    terminal_mask = rollout_outputs.get("terminal_mask")
    truncated_mask = rollout_outputs.get("truncated_mask")
    entry["batches"] += 1
    entry["trajectories"] += int(lengths.numel())
    entry["length_sum"] += float(lengths.sum().item())
    if terminal_mask is not None:
        entry["terminal_sum"] += float(terminal_mask.detach().float().cpu().sum().item())
    if truncated_mask is not None:
        entry["truncated_sum"] += float(truncated_mask.detach().float().cpu().sum().item())
    quantiles = rollout_outputs["time_quantiles"].detach().cpu()
    deltas = rollout_outputs["time_delta_times"].detach().cpu()
    densities = rollout_outputs["time_log_densities"].detach().cpu()
    entry["time_count"] += int(quantiles.numel())
    entry["time_quantile_sum"] += float(quantiles.sum().item())
    entry["time_delta_sum"] += float(deltas.sum().item())
    entry["time_near_boundary_sum"] += float(
        (quantiles >= 0.99).sum().item()
    )
    entry["time_finite_density_sum"] += float(
        torch.isfinite(densities).sum().item()
    )
    entropies = rollout_outputs["time_policy_entropies"].detach().cpu()
    finite_entropies = entropies[torch.isfinite(entropies)]
    entry["time_entropy_sum"] += float(finite_entropies.sum().item())
    entry["time_entropy_count"] += int(finite_entropies.numel())
    effective_components = rollout_outputs[
        "time_effective_components"
    ].detach().cpu()
    entry["time_effective_components_sum"] += float(
        effective_components[torch.isfinite(effective_components)].sum().item()
    )
    entry["time_effective_components_count"] += int(
        torch.isfinite(effective_components).sum().item()
    )
    entry["fixed_attachment_sum"] += int(
        rollout_outputs["fixed_attachment_count"]
    )
    entry["coalescence_sum"] += int(
        rollout_outputs["coalescence_counts"].sum().item()
    )
    entry["recombination_sum"] += int(
        rollout_outputs["recombination_counts"].sum().item()
    )
    policy_rows = [
        row
        for trajectory in rollout_outputs.get("trajectory_policy_diagnostics", ())
        for row in trajectory
    ]
    structural_rows = [
        row for row in policy_rows if row.get("selected_gate") == "generated"
    ]
    entry["structural_decision_count"] += len(structural_rows)
    for row in structural_rows:
        entry["structural_action_support_size_sum"] += int(
            row.get("structural_action_support_size", 0)
        )
        entry["structural_action_entropy_sum"] += float(
            row.get("structural_action_entropy", 0.0)
        )
        entry["structural_action_normalized_entropy_sum"] += float(
            row.get("structural_action_normalized_entropy", 0.0)
        )
        entry["structural_selected_probability_sum"] += float(
            row.get("selected_atomic_action_probability", 0.0)
        )
        entry["structural_max_probability_sum"] += float(
            row.get("structural_action_max_probability", 0.0)
        )
        entry["coalescence_probability_mass_sum"] += float(
            row.get("coalescence_probability_mass", 0.0)
        )
        entry["recombination_probability_mass_sum"] += float(
            row.get("recombination_probability_mass", 0.0)
        )
    breakpoint_rows = [
        row
        for row in policy_rows
        if int(row.get("breakpoint_support_size", 0)) > 0
    ]
    entry["breakpoint_decision_count"] += len(breakpoint_rows)
    for row in breakpoint_rows:
        entry["breakpoint_support_size_sum"] += int(
            row.get("breakpoint_support_size", 0)
        )
        entry["breakpoint_entropy_sum"] += float(
            row.get("breakpoint_entropy", 0.0)
        )
        entry["breakpoint_normalized_entropy_sum"] += float(
            row.get("breakpoint_normalized_entropy", 0.0)
        )
        entry["breakpoint_selected_probability_sum"] += float(
            row.get("breakpoint_selected_probability", 0.0)
        )
        entry["breakpoint_max_probability_sum"] += float(
            row.get("breakpoint_max_probability", 0.0)
        )


def _summarize_rollout_metrics(metrics):
    summary = {}
    for mode, entry in metrics.items():
        trajectories = max(int(entry["trajectories"]), 1)
        prefix = f"train_{mode}"
        summary[f"{prefix}_batches"] = int(entry["batches"])
        summary[f"{prefix}_trajectory_length_mean"] = float(
            entry["length_sum"] / trajectories
        )
        summary[f"{prefix}_terminal_rate"] = float(entry["terminal_sum"] / trajectories)
        summary[f"{prefix}_truncated_rate"] = float(entry["truncated_sum"] / trajectories)
        time_count = max(int(entry["time_count"]), 1)
        summary[f"{prefix}_time_quantile_mean"] = float(
            entry["time_quantile_sum"] / time_count
        )
        summary[f"{prefix}_time_delta_mean"] = float(
            entry["time_delta_sum"] / time_count
        )
        summary[f"{prefix}_time_near_boundary_rate"] = float(
            entry["time_near_boundary_sum"] / time_count
        )
        summary[f"{prefix}_time_finite_density_rate"] = float(
            entry["time_finite_density_sum"] / time_count
        )
        summary[f"{prefix}_fixed_attachment_mean"] = float(
            entry["fixed_attachment_sum"] / trajectories
        )
        time_prefix = f"models/time/behavior/{mode}"
        summary[f"{time_prefix}/sample_count"] = int(entry["time_count"])
        summary[f"{time_prefix}/quantile_mean"] = summary[
            f"{prefix}_time_quantile_mean"
        ]
        summary[f"{time_prefix}/near_boundary_rate"] = summary[
            f"{prefix}_time_near_boundary_rate"
        ]
        summary[f"{time_prefix}/finite_density_rate"] = summary[
            f"{prefix}_time_finite_density_rate"
        ]
        entropy_count = max(int(entry["time_entropy_count"]), 1)
        summary[f"{time_prefix}/entropy_mean"] = float(
            entry["time_entropy_sum"] / entropy_count
        )
        effective_component_count = max(
            int(entry["time_effective_components_count"]),
            1,
        )
        summary[f"{time_prefix}/effective_components_mean"] = float(
            entry["time_effective_components_sum"] / effective_component_count
        )

        structural_prefix = f"models/structural/behavior/{mode}"
        structural_count = int(entry["structural_decision_count"])
        summary[f"{structural_prefix}/decision_count"] = structural_count
        if structural_count:
            for metric_name, sum_name in (
                ("support_size_mean", "structural_action_support_size_sum"),
                ("entropy_mean", "structural_action_entropy_sum"),
                (
                    "normalized_entropy_mean",
                    "structural_action_normalized_entropy_sum",
                ),
                ("selected_probability_mean", "structural_selected_probability_sum"),
                ("max_probability_mean", "structural_max_probability_sum"),
                ("coalescence_probability_mass_mean", "coalescence_probability_mass_sum"),
                ("recombination_probability_mass_mean", "recombination_probability_mass_sum"),
            ):
                summary[f"{structural_prefix}/{metric_name}"] = float(
                    entry[sum_name] / structural_count
                )
        summary[f"{structural_prefix}/recombination_coalescence_ratio"] = float(
            entry["recombination_sum"] / max(entry["coalescence_sum"], 1)
        )

        breakpoint_prefix = f"models/breakpoint/behavior/{mode}"
        breakpoint_count = int(entry["breakpoint_decision_count"])
        summary[f"{breakpoint_prefix}/decision_count"] = breakpoint_count
        if breakpoint_count:
            for metric_name, sum_name in (
                ("support_size_mean", "breakpoint_support_size_sum"),
                ("entropy_mean", "breakpoint_entropy_sum"),
                ("normalized_entropy_mean", "breakpoint_normalized_entropy_sum"),
                ("selected_probability_mean", "breakpoint_selected_probability_sum"),
                ("max_probability_mean", "breakpoint_max_probability_sum"),
            ):
                summary[f"{breakpoint_prefix}/{metric_name}"] = float(
                    entry[sum_name] / breakpoint_count
                )
    return summary


def evaluate_generator(rollout_worker, generator, episodes, seed):
    episodes = int(episodes)
    if episodes <= 0:
        return {}

    env = rollout_worker.env
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    env_rng_state = env.rng.getstate() if hasattr(env.rng, "getstate") else None
    was_training = generator.training

    try:
        generator.eval()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(env.rng, "seed"):
            env.rng.seed(seed)

        with torch.no_grad():
            outputs, trajectories = rollout_worker.rollout(generator, episodes=episodes)
            log_pf = outputs["log_paths_pf"].sum(-1)
            log_pb = outputs["log_paths_pb"].sum(-1)
            log_rewards = outputs["log_rewards"]
            initial_state = env.get_initial_state()
            initial_event_probs = env.compute_event_probabilities(initial_state)
            if generator.loss_mode == "fl_subtb":
                corrections = torch.tensor(
                    [
                        float(getattr(path[-1], "terminal_partial_correction", 0.0))
                        for path in outputs["trajectory_states"]
                    ],
                    dtype=log_pf.dtype,
                    device=log_pf.device,
                )
                loss_metrics = {
                    "eval_fl_subtb_mse": float(
                        generator.compute_subtb_loss_from_rollout_outputs(outputs)
                        .detach()
                        .cpu()
                        .item()
                    ),
                    "eval_log_f0": float(
                        generator.compute_root_log_flow().detach().cpu().item()
                    ),
                    "eval_terminal_partial_correction_mean": float(
                        corrections.mean().detach().cpu().item()
                    ),
                    "eval_terminal_partial_correction_abs_mean": float(
                        corrections.abs().mean().detach().cpu().item()
                    ),
                }
            elif generator.loss_mode == "subtb":
                loss_metrics = {
                    "eval_subtb_mse": float(
                        generator.compute_subtb_loss_from_rollout_outputs(outputs)
                        .detach()
                        .cpu()
                        .item()
                    ),
                    "eval_log_f0": float(
                        generator.compute_root_log_flow().detach().cpu().item()
                    ),
                }
            else:
                residuals = generator.compute_log_Z().detach().to(log_pf) + log_pf - (
                    log_rewards + log_pb
                )
                loss_metrics = {
                    "eval_tb_mse": float(residuals.pow(2).mean().detach().cpu().item()),
                    "eval_residual_mean": float(residuals.mean().detach().cpu().item()),
                    "eval_residual_std": float(
                        residuals.std(unbiased=False).detach().cpu().item()
                    ),
                }

        lengths = torch.tensor([len(traj) for traj in trajectories], dtype=torch.float32)
        coal_counts = torch.tensor(
            [
                sum(
                    1
                    for action in traj.actions
                    if action_as_dict(action).get("event_type") == "coal"
                )
                for traj in trajectories
            ],
            dtype=torch.float32,
        )
        recomb_counts = torch.tensor(
            [
                sum(
                    1
                    for action in traj.actions
                    if action_as_dict(action).get("event_type") == "recomb"
                )
                for traj in trajectories
            ],
            dtype=torch.float32,
        )
        return {
            **loss_metrics,
            "eval_log_pf_mean": float(log_pf.mean().detach().cpu().item()),
            "eval_log_pb_mean": float(log_pb.mean().detach().cpu().item()),
            "eval_log_reward_mean": float(log_rewards.mean().detach().cpu().item()),
            "eval_trajectory_length_mean": float(lengths.mean().item()),
            "eval_coalescence_count_mean": float(coal_counts.mean().item()),
            "eval_recombination_count_mean": float(recomb_counts.mean().item()),
            "eval_initial_coalescence_prob": float(initial_event_probs.get("coal", 0.0)),
            "eval_initial_recombination_prob": float(initial_event_probs.get("recomb", 0.0)),
        }
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        if env_rng_state is not None:
            env.rng.setstate(env_rng_state)
        generator.train(was_training)

def train(
    dataset_path,
    output_path,
    device,
    bp_per_blocks=1,
    batch_size=1,
    epochs_num=10,
    seed=7,
    init_z_sample_count=DEFAULT_INIT_Z_SAMPLE_COUNT,
    use_wandb=True,
    effective_population_size=DEFAULT_NE,
    mutation_rate=DEFAULT_MU_PER_BP,
    recombination_rate=DEFAULT_R_PER_BP,
    policy_lr=DEFAULT_POLICY_LR,
    breakpoint_policy_lr=None,
    time_policy_lr=None,
    log_z_lr=DEFAULT_LOG_Z_LR,
    lr_scheduler_config=None,
    loss_mode=DEFAULT_LOSS,
    subtb_lambda=DEFAULT_SUBTB_LAMBDA,
    subtb_max_span=DEFAULT_SUBTB_MAX_SPAN,
    subtb_lambda_initial=None,
    subtb_lambda_final=None,
    subtb_max_span_schedule=None,
    terminal_loss_weight=1.0,
    residual_scale=1.0,
    grad_clip=DEFAULT_GRAD_CLIP,
    breakpoint_gradient_clip_norm=None,
    time_head_gradient_clip_norm=None,
    time_head_warmup_epochs=0,
    model_diagnostics=True,
    model_diagnostics_update_norm_every=1,
    grad_accum_steps=DEFAULT_GRAD_ACCUM_STEPS,
    min_terminal_trajectories_per_batch=0,
    flow_debug=False,
    flow_debug_max_records=16,
    probability_checks=False,
    fixed_eval_bank_path=None,
    fixed_eval_bank_episodes=0,
    fixed_eval_bank_seed=100007,
    fixed_eval_bank_source="baseline",
    fixed_eval_bank_create_if_missing=False,
    eval_episodes=DEFAULT_EVAL_EPISODES,
    eval_every=DEFAULT_EVAL_EVERY,
    partial_segment_max_steps=DEFAULT_PARTIAL_SEGMENT_MAX_STEPS,
    partial_start_mode=DEFAULT_PARTIAL_START_MODE,
    partial_boundary_fraction=DEFAULT_PARTIAL_BOUNDARY_FRACTION,
    trajectory_training_mode="complete",
    complete_trajectory_max_steps=None,
    hybrid_replay_config=None,
    selection_margin=1e-6,
    min_valid_splice_rate=0.90,
    min_unique_topology_rate=0.25,
    reward_C=30000,
    embedding_size=DEFAULT_EMBEDDING_SIZE,
    hidden_size=DEFAULT_HIDDEN_SIZE,
    dropout=DEFAULT_DROPOUT,
    breakpoint_hidden_dim=DEFAULT_BREAKPOINT_HIDDEN_DIM,
    breakpoint_dropout=DEFAULT_BREAKPOINT_DROPOUT,
    transformer_depth=DEFAULT_TRANSFORMER_DEPTH,
    transformer_heads=DEFAULT_TRANSFORMER_HEADS,
    transformer_mlp_ratio=DEFAULT_TRANSFORMER_MLP_RATIO,
    attention_dropout=DEFAULT_ATTENTION_DROPOUT,
    time_basis_components=DEFAULT_TIME_BASIS_COMPONENTS,
    local_coalescence_similarity_bias=DEFAULT_LOCAL_COALESCENCE_SIMILARITY_BIAS,
    local_prior_action_logit_bias=DEFAULT_LOCAL_PRIOR_ACTION_LOGIT_BIAS,
    local_prior_gate_logit_bias=DEFAULT_LOCAL_PRIOR_GATE_LOGIT_BIAS,
    recombination_split_bias=None,
    local_cwr_event_gate=None,
    verbose=True,
    
):
    seed_everything(seed)
    device = torch.device(device)
    loss_mode = str(loss_mode).lower()
    batch_size = int(batch_size)
    grad_accum_steps = int(grad_accum_steps)
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if grad_accum_steps <= 0:
        raise ValueError("grad_accum_steps must be a positive integer")

    variant_data = None
    if is_vcf_path(dataset_path):
        input_mode = "vcf"
        variant_data = load_vcf_variants(dataset_path)
        sequences = None
        sequence_length = int(variant_data.sequence_length)
        num_sequences = int(variant_data.num_haplotypes)
    else:
        input_mode = "dense"
        sequences = load_sequences(dataset_path)
        sequence_length = len(sequences[0])
        num_sequences = len(sequences)

    env = SimpleARGEnvironment(
        sequence_length=sequence_length,
        num_sequences=num_sequences,
        bp_per_blocks = bp_per_blocks,
        sequences=sequences,
        variant_data=variant_data,
        device=device,
        recombination_rate=recombination_rate,
        population_size=effective_population_size,
        mutation_rate=mutation_rate,
        reward_C=reward_C,
    )
    requested_recombination_split_bias = (
        normalize_recombination_split_bias_config(recombination_split_bias)
    )
    requested_local_cwr_event_gate = normalize_local_cwr_event_gate_config(
        local_cwr_event_gate
    )
    model_kwargs = {
        "embedding_size": int(embedding_size),
        "hidden_size": int(hidden_size),
        "dropout": float(dropout),
        "breakpoint_hidden_dim": int(breakpoint_hidden_dim),
        "breakpoint_dropout": float(breakpoint_dropout),
        "transformer_depth": int(transformer_depth),
        "transformer_heads": int(transformer_heads),
        "transformer_mlp_ratio": float(transformer_mlp_ratio),
        "attention_dropout": float(attention_dropout),
        "local_coalescence_similarity_bias": float(local_coalescence_similarity_bias),
        "local_prior_action_logit_bias": float(local_prior_action_logit_bias),
        "local_prior_gate_logit_bias": float(local_prior_gate_logit_bias),
        "recombination_split_bias": requested_recombination_split_bias,
        "local_cwr_event_gate": requested_local_cwr_event_gate,
        "time_hidden_size": int(DEFAULT_TIME_HIDDEN_SIZE),
        "time_layers": int(DEFAULT_TIME_LAYERS),
        "time_dropout": float(DEFAULT_TIME_DROPOUT),
        "time_basis_components": int(time_basis_components),
        "breakpoint_gap_hidden_size": int(DEFAULT_BREAKPOINT_GAP_HIDDEN_SIZE),
        "breakpoint_gap_layers": int(DEFAULT_BREAKPOINT_GAP_LAYERS),
        "breakpoint_gap_dropout": float(DEFAULT_BREAKPOINT_GAP_DROPOUT),
        "breakpoint_use_position_features": bool(DEFAULT_BREAKPOINT_USE_POSITION_FEATURES),
    }

    generator = TBGFlowNetGenerator(
        env,
        init_z_sample_count=init_z_sample_count,
        device=device,
        verbose=verbose,
        policy_lr=policy_lr,
        breakpoint_policy_lr=breakpoint_policy_lr,
        time_policy_lr=time_policy_lr,
        log_z_lr=log_z_lr,
        grad_clip=grad_clip,
        model_kwargs=model_kwargs,
        loss_mode=loss_mode,
        subtb_lambda=subtb_lambda,
        subtb_max_span=subtb_max_span,
        subtb_lambda_initial=subtb_lambda_initial,
        subtb_lambda_final=subtb_lambda_final,
        subtb_max_span_schedule=subtb_max_span_schedule,
        terminal_loss_weight=terminal_loss_weight,
        residual_scale=residual_scale,
        breakpoint_gradient_clip_norm=breakpoint_gradient_clip_norm,
        time_head_gradient_clip_norm=time_head_gradient_clip_norm,
        time_head_warmup_epochs=time_head_warmup_epochs,
        model_diagnostics=model_diagnostics,
        model_diagnostics_update_norm_every=(
            model_diagnostics_update_norm_every
        ),
        flow_debug=flow_debug,
        flow_debug_max_records=flow_debug_max_records,
        probability_checks=probability_checks,
        lr_scheduler_config=lr_scheduler_config,
        total_training_steps=int(epochs_num),
    )
    print(f"Generator device: {generator.device}")

    rollout_worker = RolloutWorker(env)
    print(f"Training on device: {generator.device}")

    os.makedirs(output_path, exist_ok=True)
    checkpoints_path = os.path.join(output_path, "checkpoints")
    os.makedirs(checkpoints_path, exist_ok=True)
    best_checkpoint_path = os.path.join(checkpoints_path, "best.pt")
    last_checkpoint_path = os.path.join(checkpoints_path, "last.pt")

    history = []
    best_loss = float("inf")
    best_metric_name = None
    wandb_run = None
    
    print(f"use_wandb: {use_wandb}")
    if use_wandb:
        wandb_run = wandb.init()
        wandb_run.define_metric("lr/optimizer_step")
        wandb_run.define_metric(
            "models/*",
            step_metric="lr/optimizer_step",
        )
        wandb_run.define_metric(
            "lr/*",
            step_metric="lr/optimizer_step",
        )
        wandb.config.update({
            "device": str(generator.device),
            "input_mode": input_mode,
            **env.time_metadata,
            "effective_population_size": float(effective_population_size),
            "mutation_rate": float(mutation_rate),
            "recombination_rate": float(recombination_rate),
            "policy_lr": float(policy_lr),
            "breakpoint_policy_lr": (
                float(policy_lr)
                if breakpoint_policy_lr is None
                else float(breakpoint_policy_lr)
            ),
            "time_policy_lr": float(generator.time_policy_lr),
            "log_z_lr": float(log_z_lr),
            "lr_scheduler": generator.lr_scheduler_metadata(),
            "loss": loss_mode,
            "subtb_lambda": float(subtb_lambda),
            "subtb_max_span": subtb_max_span,
            "subtb_lambda_initial": subtb_lambda_initial,
            "subtb_lambda_final": subtb_lambda_final,
            "subtb_max_span_schedule": list(subtb_max_span_schedule or ()),
            "terminal_loss_weight": float(terminal_loss_weight),
            "residual_scale": float(residual_scale),
            "grad_clip": float(grad_clip),
            "breakpoint_gradient_clip_norm": breakpoint_gradient_clip_norm,
            "time_head_gradient_clip_norm": time_head_gradient_clip_norm,
            "time_head_warmup_epochs": int(time_head_warmup_epochs),
            "model_diagnostics": bool(model_diagnostics),
            "model_diagnostics_update_norm_every": int(
                model_diagnostics_update_norm_every
            ),
            "min_terminal_trajectories_per_batch": int(
                min_terminal_trajectories_per_batch
            ),
            "grad_accum_steps": int(grad_accum_steps),
            "eval_episodes": int(eval_episodes),
            "eval_every": int(eval_every),
            "bp_per_blocks": int(bp_per_blocks),
            **model_kwargs,
            "model_version": MODEL_VERSION,
        })

    try:
        for epoch in range(epochs_num):
            generator.set_training_epoch(epoch, total_epochs=epochs_num)
            info = train_epoch(
                epoch,
                rollout_worker,
                generator,
                batch_size=batch_size,
                grad_accum_steps=grad_accum_steps,
            )
            if generator.loss_mode in {"subtb", "fl_subtb"}:
                with torch.no_grad():
                    log_z = generator.compute_root_log_flow().detach().cpu().reshape(-1)[0].item()
            else:
                log_z = generator.compute_log_Z().detach().cpu().reshape(-1)[0].item()
            if info is None:
                continue

            info = dict(info)
            info["epoch"] = epoch
            info["log_z"] = log_z
            if generator.loss_mode in {"subtb", "fl_subtb"}:
                info["log_f0"] = log_z
            should_eval = int(eval_episodes) > 0 and (
                epoch == 0
                or int(eval_every) <= 1
                or (epoch + 1) % int(eval_every) == 0
            )

            ## TODO: Check this implementation of evaluation
            if should_eval:
                info.update(
                    evaluate_generator(
                        rollout_worker,
                        generator,
                        eval_episodes,
                        seed + 100000 + epoch,
                    )
                )
            info.update(generator.step_lr_scheduler(info))
            history.append(info)
            loss = float(info["loss"])

            if wandb_run is not None:
                wandb.log(info, step=epoch + 1)

            eval_metric_names = (
                "eval_tb_mse",
                "eval_subtb_mse",
                "eval_fl_subtb_mse",
            )
            selection_metric_name = next(
                (name for name in eval_metric_names if name in info),
                None,
            )
            if selection_metric_name is None and int(eval_episodes) <= 0:
                selection_metric_name = "loss"
            selection_value = (
                None
                if selection_metric_name is None
                else float(info[selection_metric_name])
            )
            is_best = (
                selection_value is not None
                and math.isfinite(selection_value)
                and selection_value < best_loss
            )
            if is_best:
                best_loss = selection_value
                best_metric_name = selection_metric_name

            metadata = build_checkpoint_metadata(
                epoch=epoch,
                best_loss=best_loss,
                log_z=log_z,
                sequences=sequences,
                variant_data=variant_data,
                dataset_path=dataset_path,
                input_mode=input_mode,
                sequence_length=sequence_length,
                bp_per_blocks=bp_per_blocks,
                time_metadata=env.time_metadata,
                reward_C=reward_C,
                rho=env.rho,
                effective_population_size=effective_population_size,
                mutation_rate=mutation_rate,
                recombination_rate=recombination_rate,
                policy_lr=policy_lr,
                log_z_lr=log_z_lr,
                loss_mode=loss_mode,
                subtb_lambda=subtb_lambda,
                subtb_max_span=subtb_max_span,
                grad_clip=grad_clip,
                grad_accum_steps=grad_accum_steps,
                eval_episodes=eval_episodes,
                eval_every=eval_every,
                model_kwargs=model_kwargs,
                seed=seed,
                init_z_sample_count=init_z_sample_count,
                model_version=MODEL_VERSION,
            )
            metadata.update(
                {
                    "lr_scheduler": generator.lr_scheduler_metadata(),
                    "model_analysis": {
                        "breakpoint_policy_lr": float(
                            generator.breakpoint_policy_lr
                        ),
                        "breakpoint_gradient_clip_norm": (
                            generator.breakpoint_gradient_clip_norm
                        ),
                        "model_diagnostics": bool(model_diagnostics),
                        "model_diagnostics_update_norm_every": int(
                            model_diagnostics_update_norm_every
                        ),
                        "parameter_groups": {
                            "structural": {
                                "base_lr": float(generator.arg_model_lr),
                                "gradient_clip_norm": float(generator.grad_clip),
                            },
                            "breakpoint": {
                                "base_lr": float(generator.breakpoint_policy_lr),
                                "gradient_clip_norm": float(
                                    generator.grad_clip
                                    if generator.breakpoint_gradient_clip_norm is None
                                    else generator.breakpoint_gradient_clip_norm
                                ),
                            },
                            "time": {
                                "base_lr": float(generator.time_policy_lr),
                                "gradient_clip_norm": float(
                                    generator.grad_clip
                                    if generator.time_head_gradient_clip_norm is None
                                    else generator.time_head_gradient_clip_norm
                                ),
                            },
                        },
                    },
                    "selection_metric": best_metric_name,
                    "selection_value": (
                        None if not math.isfinite(best_loss) else float(best_loss)
                    ),
                }
            )
            training_state = {
                "epoch": int(epoch),
                "epoch_number": int(epoch) + 1,
                "best_metric": best_metric_name,
                "best_metric_value": (
                    None if not math.isfinite(best_loss) else float(best_loss)
                ),
                "optimizer_step": int(info["lr/optimizer_step"]),
            }
            if is_best:
                generator.save(
                    best_checkpoint_path,
                    metadata={**metadata, "checkpoint_kind": "best"},
                    training_state=training_state,
                )
                info["best_checkpoint_path"] = best_checkpoint_path
            if epoch == int(epochs_num) - 1:
                generator.save(
                    last_checkpoint_path,
                    metadata={**metadata, "checkpoint_kind": "last"},
                    training_state=training_state,
                )
                info["last_checkpoint_path"] = last_checkpoint_path

            eval_text = ""
            if "eval_tb_mse" in info:
                eval_text = (
                    f" eval_tb_mse={info['eval_tb_mse']:.4f}"
                    f" eval_residual_mean={info['eval_residual_mean']:.4f}"
                )
            elif "eval_subtb_mse" in info:
                eval_text = f" eval_subtb_mse={info['eval_subtb_mse']:.4f}"
            elif "eval_fl_subtb_mse" in info:
                eval_text = f" eval_fl_subtb_mse={info['eval_fl_subtb_mse']:.4f}"
            log_label = "logF0" if generator.loss_mode in {"subtb", "fl_subtb"} else "logZ"
            print(f"Epoch {epoch + 1} loss={loss:.4f} {log_label}={log_z:.4f}{eval_text}")

        with open(os.path.join(output_path, "training_history.pkl"), "wb") as handle:
            pickle.dump(history, handle)
    finally:
        if wandb_run is not None:
            wandb.finish()
    return history


def _train_local_refinement_legacy(
    dataset_path,
    output_path,
    device,
    local_refinement_arg,
    checkpoint=None,
    bad_region_top_k=None,
    bad_region_blocks=None,
    bad_region_bp=None,
    refine_strategy="before_last_coalescence",
    terminal_backtrack_lengths=None,
    bp_per_blocks=1,
    batch_size=1,
    epochs_num=10,
    seed=7,
    init_z_sample_count=DEFAULT_INIT_Z_SAMPLE_COUNT,
    use_wandb=True,
    effective_population_size=DEFAULT_NE,
    mutation_rate=DEFAULT_MU_PER_BP,
    recombination_rate=DEFAULT_R_PER_BP,
    policy_lr=DEFAULT_POLICY_LR,
    log_z_lr=DEFAULT_LOG_Z_LR,
    loss_mode=DEFAULT_LOSS,
    subtb_lambda=DEFAULT_SUBTB_LAMBDA,
    subtb_max_span=DEFAULT_SUBTB_MAX_SPAN,
    grad_clip=DEFAULT_GRAD_CLIP,
    grad_accum_steps=DEFAULT_GRAD_ACCUM_STEPS,
    eval_episodes=DEFAULT_EVAL_EPISODES,
    eval_every=DEFAULT_EVAL_EVERY,
    partial_segment_max_steps=DEFAULT_PARTIAL_SEGMENT_MAX_STEPS,
    reward_C=30000,
    embedding_size=DEFAULT_EMBEDDING_SIZE,
    hidden_size=DEFAULT_HIDDEN_SIZE,
    dropout=DEFAULT_DROPOUT,
    breakpoint_hidden_dim=DEFAULT_BREAKPOINT_HIDDEN_DIM,
    breakpoint_dropout=DEFAULT_BREAKPOINT_DROPOUT,
    transformer_depth=DEFAULT_TRANSFORMER_DEPTH,
    transformer_heads=DEFAULT_TRANSFORMER_HEADS,
    transformer_mlp_ratio=DEFAULT_TRANSFORMER_MLP_RATIO,
    attention_dropout=DEFAULT_ATTENTION_DROPOUT,
    time_basis_components=DEFAULT_TIME_BASIS_COMPONENTS,
    local_coalescence_similarity_bias=DEFAULT_LOCAL_COALESCENCE_SIMILARITY_BIAS,
    local_prior_action_logit_bias=DEFAULT_LOCAL_PRIOR_ACTION_LOGIT_BIAS,
    local_prior_gate_logit_bias=DEFAULT_LOCAL_PRIOR_GATE_LOGIT_BIAS,
    recombination_split_bias=None,
    local_cwr_event_gate=None,
    verbose=True,
):
    from refinement import (
        build_refinement_context_sets,
        build_refinement_source,
        parse_block_groups,
        parse_bp_intervals,
    )

    if not is_vcf_path(dataset_path):
        raise ValueError("local ARG refinement currently requires a VCF dataset")
    seed_everything(seed)
    device = torch.device(device)
    partial_segment_max_steps = int(partial_segment_max_steps)
    if partial_segment_max_steps <= 0:
        raise ValueError("partial_segment_max_steps must be positive")
    terminal_backtrack_lengths = parse_positive_int_list(
        (
            terminal_backtrack_lengths
            if terminal_backtrack_lengths is not None
            else DEFAULT_TERMINAL_BACKTRACK_LENGTHS
        ),
        "terminal_backtrack_lengths",
    )
    variant_data = load_vcf_variants(dataset_path)
    env = SimpleARGEnvironment(
        sequence_length=int(variant_data.sequence_length),
        num_sequences=int(variant_data.num_haplotypes),
        bp_per_blocks=bp_per_blocks,
        variant_data=variant_data,
        device=device,
        recombination_rate=recombination_rate,
        population_size=effective_population_size,
        mutation_rate=mutation_rate,
        reward_C=reward_C,
    )
    source_env = SimpleARGEnvironment(
        sequence_length=int(variant_data.sequence_length),
        num_sequences=int(variant_data.num_haplotypes),
        bp_per_blocks=bp_per_blocks,
        variant_data=variant_data,
        device="cpu",
        recombination_rate=recombination_rate,
        population_size=effective_population_size,
        mutation_rate=mutation_rate,
        reward_C=reward_C,
    )
    requested_recombination_split_bias = (
        normalize_recombination_split_bias_config(recombination_split_bias)
    )
    requested_local_cwr_event_gate = normalize_local_cwr_event_gate_config(
        local_cwr_event_gate
    )
    model_kwargs = {
        "embedding_size": int(embedding_size),
        "hidden_size": int(hidden_size),
        "dropout": float(dropout),
        "breakpoint_hidden_dim": int(breakpoint_hidden_dim),
        "breakpoint_dropout": float(breakpoint_dropout),
        "transformer_depth": int(transformer_depth),
        "transformer_heads": int(transformer_heads),
        "transformer_mlp_ratio": float(transformer_mlp_ratio),
        "attention_dropout": float(attention_dropout),
        "local_coalescence_similarity_bias": float(local_coalescence_similarity_bias),
        "local_prior_action_logit_bias": float(local_prior_action_logit_bias),
        "local_prior_gate_logit_bias": float(local_prior_gate_logit_bias),
        "recombination_split_bias": requested_recombination_split_bias,
        "local_cwr_event_gate": requested_local_cwr_event_gate,
        "time_hidden_size": int(DEFAULT_TIME_HIDDEN_SIZE),
        "time_layers": int(DEFAULT_TIME_LAYERS),
        "time_dropout": float(DEFAULT_TIME_DROPOUT),
        "time_basis_components": int(time_basis_components),
        "breakpoint_gap_hidden_size": int(DEFAULT_BREAKPOINT_GAP_HIDDEN_SIZE),
        "breakpoint_gap_layers": int(DEFAULT_BREAKPOINT_GAP_LAYERS),
        "breakpoint_gap_dropout": float(DEFAULT_BREAKPOINT_GAP_DROPOUT),
        "breakpoint_use_position_features": bool(DEFAULT_BREAKPOINT_USE_POSITION_FEATURES),
    }

    checkpoint_data = None
    checkpoint_metadata = {}
    if checkpoint:
        checkpoint_data = load_checkpoint(checkpoint, map_location="cpu")
        checkpoint_metadata = checkpoint_data.get("metadata", {})
        validate_checkpoint_metadata_for_env(checkpoint_metadata, env)
        if checkpoint_metadata.get("model"):
            model_kwargs = dict(checkpoint_metadata["model"])
            model_kwargs["recombination_split_bias"] = (
                requested_recombination_split_bias
            )
            model_kwargs["local_cwr_event_gate"] = requested_local_cwr_event_gate

    source = build_refinement_source(
        source_env,
        local_refinement_arg,
        dataset_path,
        population_size=effective_population_size,
        mutation_rate=mutation_rate,
    )
    segment_contexts, terminal_contexts, diagnostic_rows = build_refinement_context_sets(
        source,
        top_k=bad_region_top_k,
        block_groups=parse_block_groups(bad_region_blocks),
        bp_intervals=parse_bp_intervals(bad_region_bp),
        strategy=refine_strategy,
        terminal_backtrack_lengths=terminal_backtrack_lengths,
    )
    if not segment_contexts:
        raise ValueError("no local refinement segment contexts were selected")
    if not terminal_contexts:
        raise ValueError("no local refinement terminal contexts were selected")
    all_contexts = list(segment_contexts) + list(terminal_contexts)
    print_local_refinement_startup_summary(
        segment_contexts,
        terminal_contexts,
        diagnostic_rows,
        partial_segment_max_steps=partial_segment_max_steps,
        terminal_backtrack_lengths=terminal_backtrack_lengths,
        subtb_max_span=subtb_max_span,
    )

    os.makedirs(output_path, exist_ok=True)
    checkpoints_path = os.path.join(output_path, "checkpoints")
    os.makedirs(checkpoints_path, exist_ok=True)
    best_checkpoint_path = os.path.join(checkpoints_path, "best.pt")
    save_refinement_context_manifest(
        output_path,
        checkpoint=checkpoint,
        local_refinement_arg=local_refinement_arg,
        dataset_path=dataset_path,
        strategy=refine_strategy,
        segment_contexts=segment_contexts,
        terminal_contexts=terminal_contexts,
        diagnostic_rows=diagnostic_rows,
        partial_segment_max_steps=partial_segment_max_steps,
        terminal_backtrack_lengths=terminal_backtrack_lengths,
    )

    generator = TBGFlowNetGenerator(
        env,
        init_z_sample_count=0,
        device=device,
        verbose=verbose,
        policy_lr=policy_lr,
        log_z_lr=log_z_lr,
        grad_clip=grad_clip,
        model_kwargs=model_kwargs,
        initialize_z_from_prior=False,
        loss_mode="fl_subtb",
        subtb_lambda=subtb_lambda,
        subtb_max_span=subtb_max_span,
    )
    if checkpoint_data is not None:
        generator.load(checkpoint_data, load_optimizer=False, map_location=generator.device)
        print(f"Loaded global policy checkpoint for local refinement: {checkpoint}")
    else:
        print("No checkpoint supplied; training local refinement policy from scratch.")

    rollout_worker = RolloutWorker(env, verbose=verbose)
    rollout_cursor = {"mode": 0, "segment": 0, "terminal": 0}

    def start_state_sampler(current_batch_size):
        rollout_mode = "segment" if rollout_cursor["mode"] % 2 == 0 else "terminal"
        rollout_cursor["mode"] += 1
        contexts_for_mode = (
            segment_contexts if rollout_mode == "segment" else terminal_contexts
        )
        context = contexts_for_mode[
            rollout_cursor[rollout_mode] % len(contexts_for_mode)
        ]
        rollout_cursor[rollout_mode] += 1
        return {
            "start_states": [context.partial_state for _ in range(int(current_batch_size))],
            "action_filter": context.action_filter(),
            "max_steps": (
                partial_segment_max_steps if rollout_mode == "segment" else None
            ),
            "rollout_mode": rollout_mode,
            "context": context,
        }

    history = []
    best_loss = float("inf")
    wandb_run = None
    if use_wandb:
        if wandb is None:
            raise ImportError("wandb is not installed but training.wandb is true")
        wandb_run = wandb.init()
        wandb.config.update({
            "training_mode": "local_refinement",
            "device": str(generator.device),
            "input_mode": "vcf",
            "checkpoint": os.path.abspath(checkpoint) if checkpoint else None,
            "local_refinement_arg": os.path.abspath(local_refinement_arg),
            "bad_region_top_k": bad_region_top_k,
            "refine_strategy": refine_strategy,
            "partial_segment_max_steps": int(partial_segment_max_steps),
            "terminal_backtrack_lengths": list(terminal_backtrack_lengths),
            "local_segment_contexts": len(segment_contexts),
            "local_terminal_contexts": len(terminal_contexts),
            **env.time_metadata,
            "effective_population_size": float(effective_population_size),
            "mutation_rate": float(mutation_rate),
            "recombination_rate": float(recombination_rate),
            "policy_lr": float(policy_lr),
            "loss": "fl_subtb",
            "subtb_lambda": float(subtb_lambda),
            "subtb_max_span": subtb_max_span,
            "grad_clip": float(grad_clip),
            "grad_accum_steps": int(grad_accum_steps),
            **model_kwargs,
            "model_version": MODEL_VERSION,
        })

    try:
        for epoch in range(int(epochs_num)):
            info = train_epoch(
                epoch,
                rollout_worker,
                generator,
                batch_size=batch_size,
                grad_accum_steps=grad_accum_steps,
                start_state_sampler=start_state_sampler,
                rollout_logger=print_local_rollout_training_spec,
            )
            if info is None:
                continue
            info = dict(info)
            info["epoch"] = epoch
            with torch.no_grad():
                all_start_flows = generator.compute_log_state_flows(
                    [context.partial_state for context in all_contexts]
                )
                segment_start_flows = generator.compute_log_state_flows(
                    [context.partial_state for context in segment_contexts]
                )
                terminal_start_flows = generator.compute_log_state_flows(
                    [context.partial_state for context in terminal_contexts]
                )
                log_f_start = all_start_flows.mean().detach().cpu().reshape(-1)[0].item()
            info["log_f_start_mean"] = log_f_start
            info["log_f_segment_start_mean"] = (
                segment_start_flows.mean().detach().cpu().reshape(-1)[0].item()
            )
            info["log_f_terminal_start_mean"] = (
                terminal_start_flows.mean().detach().cpu().reshape(-1)[0].item()
            )

            should_eval = int(eval_episodes) > 0 and (
                epoch == 0
                or int(eval_every) <= 1
                or (epoch + 1) % int(eval_every) == 0
            )
            if should_eval:
                info.update(
                    evaluate_local_refinement(
                        rollout_worker,
                        generator,
                        segment_contexts,
                        terminal_contexts,
                        episodes=eval_episodes,
                        seed=seed + 100000 + epoch,
                        partial_segment_max_steps=partial_segment_max_steps,
                    )
                )

            history.append(info)
            loss = float(info["loss"])
            if wandb_run is not None:
                wandb.log(info, step=epoch + 1)

            if loss < best_loss:
                best_loss = loss
                metadata = build_checkpoint_metadata(
                    epoch=epoch,
                    best_loss=best_loss,
                    log_z=log_f_start,
                    sequences=None,
                    variant_data=variant_data,
                    dataset_path=dataset_path,
                    input_mode="vcf",
                    sequence_length=int(variant_data.sequence_length),
                    bp_per_blocks=bp_per_blocks,
                    time_metadata=env.time_metadata,
                    reward_C=reward_C,
                    rho=env.rho,
                    effective_population_size=effective_population_size,
                    mutation_rate=mutation_rate,
                    recombination_rate=recombination_rate,
                    policy_lr=policy_lr,
                    log_z_lr=log_z_lr,
                    loss_mode="fl_subtb",
                    subtb_lambda=subtb_lambda,
                    subtb_max_span=subtb_max_span,
                    grad_clip=grad_clip,
                    grad_accum_steps=grad_accum_steps,
                    eval_episodes=eval_episodes,
                    eval_every=eval_every,
                    model_kwargs=model_kwargs,
                    seed=seed,
                    init_z_sample_count=init_z_sample_count,
                    model_version=MODEL_VERSION,
                )
                metadata.update(
                    {
                        "training_mode": "local_refinement",
                        "source_checkpoint": (
                            os.path.abspath(checkpoint) if checkpoint else None
                        ),
                        "local_refinement_arg": os.path.abspath(local_refinement_arg),
                        "refine_strategy": str(refine_strategy),
                        "partial_segment_max_steps": int(partial_segment_max_steps),
                        "terminal_backtrack_lengths": list(terminal_backtrack_lengths),
                        "subtb_max_span": subtb_max_span,
                        "refinement_contexts": [
                            context.to_manifest_record() for context in segment_contexts
                        ],
                        "segment_refinement_contexts": [
                            context.to_manifest_record() for context in segment_contexts
                        ],
                        "terminal_refinement_contexts": [
                            context.to_manifest_record() for context in terminal_contexts
                        ],
                    }
                )
                generator.save(best_checkpoint_path, metadata=metadata)
                info["best_checkpoint_path"] = best_checkpoint_path

            eval_text = ""
            if "eval_local_loss_mean" in info:
                eval_text = f" eval_local_loss={info['eval_local_loss_mean']:.4f}"
            print(
                f"Epoch {epoch + 1} local_loss={loss:.4f} "
                f"logFstart={log_f_start:.4f}{eval_text}"
            )

        with open(os.path.join(output_path, "training_history.pkl"), "wb") as handle:
            pickle.dump(history, handle)
    finally:
        if wandb_run is not None:
            wandb.finish()
    return history


def evaluate_local_refinement(
    rollout_worker,
    generator,
    segment_contexts,
    terminal_contexts,
    episodes,
    seed,
    partial_segment_max_steps=DEFAULT_PARTIAL_SEGMENT_MAX_STEPS,
):
    episodes = int(episodes)
    if episodes <= 0:
        return {}

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    env_rng_state = rollout_worker.env.rng.getstate() if hasattr(rollout_worker.env.rng, "getstate") else None

    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(rollout_worker.env.rng, "seed"):
            rollout_worker.env.rng.seed(seed)

        metrics = {}
        with torch.no_grad():
            for idx in range(episodes):
                rollout_mode = "segment" if idx % 2 == 0 else "terminal"
                contexts_for_mode = (
                    segment_contexts if rollout_mode == "segment" else terminal_contexts
                )
                context = contexts_for_mode[(idx // 2) % len(contexts_for_mode)]
                outputs, trajectories = rollout_worker.rollout(
                    generator,
                    episodes=1,
                    start_states=[context.partial_state],
                    action_filter=context.action_filter(),
                    max_steps=(
                        int(partial_segment_max_steps)
                        if rollout_mode == "segment"
                        else None
                    ),
                )
                loss = generator.compute_subtb_loss_from_rollout_outputs(outputs)
                _record_eval_metrics(metrics, rollout_mode, outputs, loss)
        return _summarize_eval_metrics(metrics)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        if env_rng_state is not None:
            rollout_worker.env.rng.setstate(env_rng_state)


def _record_eval_metrics(metrics, rollout_mode, rollout_outputs, loss):
    mode = str(rollout_mode)
    entry = metrics.setdefault(
        mode,
        {
            "count": 0,
            "loss_sum": 0.0,
            "reward_sum": 0.0,
            "reward_count": 0,
            "length_sum": 0.0,
            "terminal_sum": 0.0,
            "truncated_sum": 0.0,
        },
    )
    entry["count"] += 1
    entry["loss_sum"] += float(loss.detach().cpu().item())
    length = float(rollout_outputs["trajectory_lengths"][0].detach().cpu().item())
    entry["length_sum"] += length
    terminal = bool(rollout_outputs["terminal_mask"][0].detach().cpu().item())
    truncated = bool(rollout_outputs["truncated_mask"][0].detach().cpu().item())
    entry["terminal_sum"] += float(terminal)
    entry["truncated_sum"] += float(truncated)
    reward = rollout_outputs["log_rewards"][0]
    if bool(torch.isfinite(reward).detach().cpu().item()):
        entry["reward_sum"] += float(reward.detach().cpu().item())
        entry["reward_count"] += 1


def _summarize_eval_metrics(metrics):
    summary = {}
    loss_values = []
    length_values = []
    reward_values = []
    for mode, entry in metrics.items():
        count = max(int(entry["count"]), 1)
        prefix = f"eval_{mode}"
        loss_mean = float(entry["loss_sum"] / count)
        length_mean = float(entry["length_sum"] / count)
        summary[f"{prefix}_loss_mean"] = loss_mean
        summary[f"{prefix}_trajectory_length_mean"] = length_mean
        summary[f"{prefix}_terminal_rate"] = float(entry["terminal_sum"] / count)
        summary[f"{prefix}_truncated_rate"] = float(entry["truncated_sum"] / count)
        loss_values.append(loss_mean)
        length_values.append(length_mean)
        if entry["reward_count"] > 0:
            reward_mean = float(entry["reward_sum"] / int(entry["reward_count"]))
            summary[f"{prefix}_log_reward_mean"] = reward_mean
            reward_values.append(reward_mean)
    summary["eval_local_loss_mean"] = float(np.mean(loss_values)) if loss_values else 0.0
    summary["eval_trajectory_length_mean"] = (
        float(np.mean(length_values)) if length_values else 0.0
    )
    if reward_values:
        summary["eval_log_reward_mean"] = float(np.mean(reward_values))
    return summary


def print_local_refinement_startup_summary(
    segment_contexts,
    terminal_contexts,
    diagnostic_rows,
    partial_segment_max_steps,
    terminal_backtrack_lengths,
    subtb_max_span,
):
    print("Detected local refinement bad regions:")
    selected_regions = {}
    for context in list(segment_contexts) + list(terminal_contexts):
        selected_regions[int(context.region.index)] = context.region
    for region in sorted(selected_regions.values(), key=lambda item: int(item.index)):
        print(
            "  "
            + _format_region_summary(region)
        )
    if diagnostic_rows:
        top_rows = diagnostic_rows[: min(len(diagnostic_rows), 5)]
        print("Top bad-region diagnostic blocks:")
        for row in top_rows:
            print(
                "  "
                f"block={int(row['block'])} "
                f"bp=[{float(row['left_bp']):.1f},{float(row['right_bp']):.1f}) "
                f"score={float(row['bad_region_score']):.4f}"
            )
    print(
        "Local refinement rollout mix: "
        f"partial->partial max_steps={int(partial_segment_max_steps)}, "
        "partial->terminal full completion, "
        f"terminal_backtrack_lengths={list(terminal_backtrack_lengths)}, "
        f"subtb_max_span={subtb_max_span}"
    )
    print("Segment rollout starts:")
    for context in segment_contexts:
        print("  " + _format_context_summary(context))
    print("Terminal rollout starts:")
    for context in terminal_contexts:
        print("  " + _format_context_summary(context))


def print_local_rollout_training_spec(
    epoch_id,
    accum_step,
    grad_accum_steps,
    batch_size,
    rollout_mode,
    max_steps,
    rollout_spec,
):
    if not isinstance(rollout_spec, dict):
        return
    context = rollout_spec.get("context")
    if context is None:
        return
    direction = (
        "partial->partial"
        if str(rollout_mode) == "segment"
        else "partial->terminal"
    )
    max_steps_text = "terminal" if max_steps is None else str(int(max_steps))
    print(
        f"Epoch {int(epoch_id) + 1} rollout {int(accum_step)}/{int(grad_accum_steps)}: "
        f"training {direction} "
        f"episodes={int(batch_size)} "
        f"max_steps={max_steps_text} "
        + _format_context_summary(context)
    )


def _format_region_summary(region):
    return (
        f"region={int(region.index)} "
        f"blocks={_format_block_tuple(region.blocks)} "
        f"bp=[{float(region.left_bp):.1f},{float(region.right_bp):.1f}) "
        f"score_max={float(region.max_bad_region_score):.4f} "
        f"score_sum={float(region.sum_bad_region_score):.4f} "
        f"variants={list(region.variant_positions)}"
    )


def _format_context_summary(context):
    return (
        f"mode={context.rollout_mode} "
        f"{_format_region_summary(context.region)} "
        f"backtrack_step={int(context.backtrack_step)} "
        f"strategy_step={context.strategy_backtrack_step} "
        f"offset={int(context.backtrack_offset)} "
        f"partial_active_lineages={len(context.partial_state.active_lineages)} "
        f"partial_total_active_blocks={int(context.partial_state.total_active_blocks)} "
        f"effective_blocks={len(context.effective_blocks)}"
    )


def _format_block_tuple(blocks):
    blocks = tuple(int(block) for block in blocks)
    if len(blocks) <= 8:
        return str(list(blocks))
    return (
        "["
        + ", ".join(str(block) for block in blocks[:4])
        + ", ..., "
        + ", ".join(str(block) for block in blocks[-3:])
        + "]"
    )


def load_checkpoint(path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def validate_checkpoint_metadata_for_env(metadata, env):
    if not metadata:
        return
    expected = {
        "num_sequences": int(env.num_sequences),
        "sequence_length": int(env.sequence_length),
        "num_blocks": int(env.num_blocks),
    }
    mismatches = []
    for key, expected_value in expected.items():
        if key in metadata and int(metadata[key]) != expected_value:
            mismatches.append(f"{key}: checkpoint={metadata[key]} env={expected_value}")
    if mismatches:
        raise ValueError(
            "checkpoint metadata does not match local refinement environment: "
            + "; ".join(mismatches)
        )


def save_refinement_context_manifest(
    output_path,
    checkpoint,
    local_refinement_arg,
    dataset_path,
    strategy,
    segment_contexts,
    terminal_contexts,
    diagnostic_rows,
    partial_segment_max_steps,
    terminal_backtrack_lengths,
):
    manifest = {
        "training_mode": "local_refinement",
        "checkpoint": os.path.abspath(checkpoint) if checkpoint else None,
        "local_refinement_arg": os.path.abspath(local_refinement_arg),
        "dataset_path": os.path.abspath(dataset_path),
        "refine_strategy": str(strategy),
        "partial_segment_max_steps": int(partial_segment_max_steps),
        "terminal_backtrack_lengths": list(terminal_backtrack_lengths),
        "contexts": [context.to_manifest_record() for context in segment_contexts],
        "segment_contexts": [
            context.to_manifest_record() for context in segment_contexts
        ],
        "terminal_contexts": [
            context.to_manifest_record() for context in terminal_contexts
        ],
        "diagnostics": diagnostic_rows,
    }
    with open(
        os.path.join(output_path, "refinement_context_manifest.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(manifest, handle, indent=2)


def build_checkpoint_metadata(
    epoch,
    best_loss,
    log_z,
    sequences,
    variant_data,
    dataset_path,
    input_mode,
    sequence_length,
    bp_per_blocks,
    time_metadata,
    reward_C,
    rho,
    effective_population_size,
    mutation_rate,
    recombination_rate,
    policy_lr,
    log_z_lr,
    loss_mode,
    subtb_lambda,
    subtb_max_span,
    grad_clip,
    grad_accum_steps,
    eval_episodes,
    eval_every,
    model_kwargs,
    seed,
    init_z_sample_count,
    model_version,
):
    num_blocks = int(
        variant_data.num_variants
        if input_mode == "vcf"
        else sequence_length // bp_per_blocks
    )
    metadata = {
        "epoch": int(epoch),
        "best_loss": float(best_loss),
        "log_z": float(log_z),
        "input_mode": str(input_mode),
        "dataset_path": os.path.abspath(str(dataset_path)),
        "num_sequences": int(
            variant_data.num_haplotypes
            if input_mode == "vcf"
            else len(sequences)
        ),
        "sequence_length": int(sequence_length),
        "num_blocks": int(num_blocks),
        "bp_per_blocks": int(bp_per_blocks),
        "rho": float(rho),
        "time": dict(time_metadata),
        **dict(time_metadata),
        "time_basis_components": int(
            model_kwargs["time_basis_components"]
        ),
        "reward_C": float(reward_C),
        "effective_population_size": float(effective_population_size),
        "mutation_rate": float(mutation_rate),
        "recombination_rate": float(recombination_rate),
        "policy_lr": float(policy_lr),
        "log_z_lr": float(log_z_lr),
        "loss": str(loss_mode),
        "subtb_lambda": float(subtb_lambda),
        "subtb_max_span": (
            None if subtb_max_span is None else int(subtb_max_span)
        ),
        "grad_clip": float(grad_clip),
        "grad_accum_steps": int(grad_accum_steps),
        "eval_episodes": int(eval_episodes),
        "eval_every": int(eval_every),
        "model": dict(model_kwargs),
        "seed": int(seed),
        "init_z_sample_count": int(init_z_sample_count),
        "model_version": str(model_version),
    }
    metadata["time"]["time_basis_components"] = int(
        model_kwargs["time_basis_components"]
    )
    if input_mode == "vcf":
        metadata.update({
            "num_variants": int(variant_data.num_variants),
            "sample_ids": list(variant_data.sample_ids),
            "haplotype_ids": list(variant_data.haplotype_ids),
            "vcf_parser_version": VCF_PARSER_VERSION,
        })
        return metadata

    metadata["sequences"] = list(sequences)
    return {
        **metadata,
    }


def train_local_refinement(*args, **kwargs):
    """Dispatch explicit local requests to the production refinement workflow."""

    try:
        from .refinement.training import train_local_refinement as train_local
    except ImportError:
        from refinement.training import train_local_refinement as train_local

    return train_local(*args, **kwargs)


def main():
    parser = argparse.ArgumentParser(description="Train the simplified ARG GFlowNet demo.")
    parser.add_argument("--config", "-c", help="Path to YAML training config.")
    parser.add_argument("--checkpoint", help="Optional global checkpoint to fine-tune for local refinement.")
    parser.add_argument(
        "--local-refinement-arg",
        help="Existing .trees ARG to backtrack for local bad-region refinement.",
    )
    parser.add_argument(
        "--bad-region-top-k",
        type=int,
        help="Deprecated; use explicit refinement.requests in YAML.",
    )
    parser.add_argument(
        "--bad-region-blocks",
        help="Deprecated; use explicit refinement.requests in YAML.",
    )
    parser.add_argument(
        "--bad-region-bp",
        help="Deprecated; use explicit refinement.requests in YAML.",
    )
    parser.add_argument(
        "--refine-strategy",
        default=None,
        choices=["before_last_touch", "before_first_touch", "before_last_coalescence"],
        help="Deprecated; use explicit refinement.requests in YAML.",
    )
    parser.add_argument(
        "--terminal-backtrack-lengths",
        help="Deprecated; use explicit refinement.requests in YAML.",
    )
    parser.add_argument("--output-path")
    parser.add_argument("--dataset-path")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--bp-per-blocks",
        type=int,
        help="Number of bp per block",
    )
    parser.add_argument("--init-z-sample-count", type=int)
    parser.add_argument("--verbose", action="store_true", default=None)
    parser.add_argument("--effective-population-size", type=float)
    parser.add_argument("--mutation-rate", type=float)
    parser.add_argument("--recombination-rate", type=float)
    parser.add_argument("--policy-lr", type=float)
    parser.add_argument("--breakpoint-policy-lr", type=float)
    parser.add_argument("--time-policy-lr", type=float)
    parser.add_argument("--log-z-lr", type=float)
    parser.add_argument(
        "--lr-scheduler",
        dest="lr_scheduler_type",
        choices=sorted(LR_SCHEDULER_TYPES),
    )
    parser.add_argument("--lr-warmup-steps", type=int)
    parser.add_argument("--lr-warmup-fraction", type=float)
    parser.add_argument("--lr-warmup-start-ratio", type=float)
    parser.add_argument("--lr-min-ratio", type=float)
    parser.add_argument("--lr-step-size", type=int)
    parser.add_argument("--lr-step-gamma", type=float)
    parser.add_argument("--lr-plateau-metric")
    parser.add_argument(
        "--lr-plateau-mode",
        choices=sorted(LR_PLATEAU_MODES),
    )
    parser.add_argument("--lr-plateau-factor", type=float)
    parser.add_argument("--lr-plateau-patience", type=int)
    parser.add_argument("--lr-plateau-threshold", type=float)
    parser.add_argument(
        "--lr-plateau-threshold-mode",
        choices=sorted(LR_PLATEAU_THRESHOLD_MODES),
    )
    parser.add_argument("--lr-plateau-cooldown", type=int)
    parser.add_argument("--loss", choices=["tb", "subtb", "fl_subtb"])
    parser.add_argument("--subtb-lambda", type=float)
    parser.add_argument("--subtb-max-span", type=int)
    parser.add_argument("--terminal-loss-weight", type=float)
    parser.add_argument("--residual-scale", type=float)
    parser.add_argument("--grad-clip", type=float)
    parser.add_argument("--breakpoint-gradient-clip-norm", type=float)
    parser.add_argument("--time-head-gradient-clip-norm", type=float)
    parser.add_argument("--time-head-warmup-epochs", type=int)
    parser.add_argument(
        "--model-diagnostics",
        dest="model_diagnostics",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--no-model-diagnostics",
        dest="model_diagnostics",
        action="store_false",
    )
    parser.add_argument("--model-diagnostics-update-norm-every", type=int)
    parser.add_argument("--min-terminal-trajectories-per-batch", type=int)
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument("--eval-episodes", type=int)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--partial-segment-max-steps", type=int)
    parser.add_argument(
        "--partial-start-mode",
        choices=list(PARTIAL_START_MODES),
    )
    parser.add_argument("--partial-boundary-fraction", type=float)
    parser.add_argument(
        "--trajectory-training-mode",
        choices=["complete"],
    )
    parser.add_argument("--complete-trajectory-max-steps", type=int)
    parser.add_argument("--selection-margin", type=float)
    parser.add_argument("--min-valid-splice-rate", type=float)
    parser.add_argument("--min-unique-topology-rate", type=float)
    parser.add_argument("--time-basis-components", type=int)
    parser.add_argument("--reward-constant", type=float)
    parser.add_argument("--embedding-size", type=int)
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--breakpoint-hidden-dim", type=int)
    parser.add_argument("--breakpoint-dropout", type=float)
    parser.add_argument("--transformer-depth", type=int)
    parser.add_argument("--transformer-heads", type=int)
    parser.add_argument("--transformer-mlp-ratio", type=float)
    parser.add_argument("--attention-dropout", type=float)
    parser.add_argument("--wandb", dest="wandb", action="store_true", default=None)
    parser.add_argument("--no-wandb", dest="wandb", action="store_false")
    args = parser.parse_args()

    config = load_train_config(args.config)
    config = apply_cli_overrides(config, args)
    validate_train_config(config)
    save_resolved_config(config, config["output_path"])
    train_kwargs = config_to_train_kwargs(config)

    print(f"Selected device: {train_kwargs['device']}")
    if refinement_enabled(config):
        train_local_refinement(
            **train_kwargs,
            **config_to_refinement_kwargs(config),
        )
    else:
        train(**train_kwargs)


if __name__ == "__main__":
    main()

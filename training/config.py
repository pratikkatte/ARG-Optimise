"""Typed YAML configuration for training."""

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml

from arg_environment.time import DEFAULT_TIME_BINS, DEFAULT_TIME_DELTA_BIN_WIDTH
from utils.device import resolve_device


@dataclass
class DataConfig:
    dataset_path: str
    output_path: str
    bp_per_blocks: int = 1


@dataclass
class RuntimeConfig:
    device: str = "auto"
    seed: int = 7
    verbose: bool = True


@dataclass
class TrainingOptions:
    epochs: int = 10
    batch_size: int = 10
    init_z_sample_count: int = 16
    grad_accum_steps: int = 1
    eval_episodes: int = 8
    eval_every: int = 10


@dataclass
class OptimizerConfig:
    policy_lr: float = 1e-3
    log_z_lr: float = 1e-3
    grad_clip: float = 10.0


@dataclass
class EnvironmentConfig:
    effective_population_size: float = 10_000
    mutation_rate: float = 2e-8
    recombination_rate: float = 2e-8
    time_bins: int = DEFAULT_TIME_BINS
    time_delta_bin_width: float = DEFAULT_TIME_DELTA_BIN_WIDTH


@dataclass
class ModelConfig:
    embedding_size: int = 32
    hidden_size: int = 64
    dropout: float = 0.0
    event_hidden_size: int = 64
    event_dropout: float = 0.0
    event_prior_weight: float = 0.1
    breakpoint_hidden_dim: int = 128
    breakpoint_dropout: float = 0.1
    transformer_depth: int = 6
    transformer_heads: int = 4
    transformer_mlp_ratio: float = 2.0
    attention_dropout: float = 0.0
    time_hidden_size: int = 256
    time_layers: int = 3
    time_dropout: float = 0.0
    breakpoint_gap_hidden_size: int = 256
    breakpoint_gap_layers: int = 3
    breakpoint_gap_dropout: float = 0.0
    breakpoint_use_position_features: bool = True


@dataclass
class LoggingConfig:
    wandb: bool = False
    project: str | None = None
    entity: str | None = None
    run_name: str | None = None


@dataclass
class TrainConfig:
    data: DataConfig
    runtime: RuntimeConfig
    training: TrainingOptions
    optimizer: OptimizerConfig
    environment: EnvironmentConfig
    model: ModelConfig
    logging: LoggingConfig

    @classmethod
    def load(cls, path):
        config_path = Path(path)
        with config_path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, dict):
            raise ValueError("training config must be a YAML mapping")

        section_types = {
            "data": DataConfig,
            "runtime": RuntimeConfig,
            "training": TrainingOptions,
            "optimizer": OptimizerConfig,
            "environment": EnvironmentConfig,
            "model": ModelConfig,
            "logging": LoggingConfig,
        }
        unknown = set(raw) - set(section_types)
        if unknown:
            raise ValueError(f"unknown config sections: {', '.join(sorted(unknown))}")

        sections = {}
        for name, section_type in section_types.items():
            values = raw.get(name, {})
            if not isinstance(values, dict):
                raise ValueError(f"config section {name!r} must be a mapping")
            allowed = {field.name for field in fields(section_type)}
            extra = set(values) - allowed
            if extra:
                raise ValueError(f"unknown {name} settings: {', '.join(sorted(extra))}")
            try:
                sections[name] = section_type(**values)
            except TypeError as exc:
                raise ValueError(f"invalid {name} section: {exc}") from exc
        config = cls(**sections)
        config.validate()
        return config

    def validate(self):
        positive = {
            "data.bp_per_blocks": self.data.bp_per_blocks,
            "training.epochs": self.training.epochs,
            "training.batch_size": self.training.batch_size,
            "training.init_z_sample_count": self.training.init_z_sample_count,
            "training.grad_accum_steps": self.training.grad_accum_steps,
            "environment.effective_population_size": self.environment.effective_population_size,
            "environment.time_bins": self.environment.time_bins,
            "environment.time_delta_bin_width": self.environment.time_delta_bin_width,
            "optimizer.policy_lr": self.optimizer.policy_lr,
            "optimizer.log_z_lr": self.optimizer.log_z_lr,
            "optimizer.grad_clip": self.optimizer.grad_clip,
            "model.embedding_size": self.model.embedding_size,
            "model.transformer_depth": self.model.transformer_depth,
            "model.transformer_heads": self.model.transformer_heads,
            "model.hidden_size": self.model.hidden_size,
            "model.event_hidden_size": self.model.event_hidden_size,
            "model.breakpoint_hidden_dim": self.model.breakpoint_hidden_dim,
            "model.time_hidden_size": self.model.time_hidden_size,
            "model.breakpoint_gap_hidden_size": self.model.breakpoint_gap_hidden_size,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"settings must be positive: {', '.join(invalid)}")
        if self.training.eval_episodes < 0 or self.training.eval_every < 0:
            raise ValueError("evaluation settings cannot be negative")
        if self.environment.mutation_rate < 0 or self.environment.recombination_rate < 0:
            raise ValueError("environment rates cannot be negative")
        if self.model.time_layers < 0 or self.model.breakpoint_gap_layers < 0:
            raise ValueError("model layer counts cannot be negative")
        probabilities = (
            self.model.dropout, self.model.event_dropout,
            self.model.breakpoint_dropout, self.model.attention_dropout,
            self.model.time_dropout, self.model.breakpoint_gap_dropout,
        )
        if any(not 0 <= value < 1 for value in probabilities):
            raise ValueError("dropout settings must be in [0, 1)")
        if not 0 <= self.model.event_prior_weight <= 1:
            raise ValueError("model.event_prior_weight must be in [0, 1]")
        if self.model.embedding_size % self.model.transformer_heads:
            raise ValueError("model.embedding_size must be divisible by model.transformer_heads")
        if self.runtime.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("runtime.device must be auto, cpu, or cuda")

    @property
    def device(self):
        return resolve_device(self.runtime.device)

    def as_dict(self):
        return asdict(self)


# Kept as imports for inference and small external scripts.
DEFAULT_NE = EnvironmentConfig.effective_population_size
DEFAULT_MU_PER_BP = EnvironmentConfig.mutation_rate
DEFAULT_LOG_Z_LR = OptimizerConfig.log_z_lr
MODEL_VERSION = "pytorch-transformer-yaml-v7"

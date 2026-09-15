"""Learning-rate and policy-temperature schedules for joint training."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class LearningRateConfig:
    """Configure a constant or warm-up cosine learning-rate schedule."""
    schedule: str = 'constant'
    total_steps: int = 0
    warmup_steps: int = 0
    warmup_start_factor: float = 0.1
    min_factor: float = 0.1

    @classmethod
    def from_namespace(cls, args, total_steps=None):
        """Build a configuration from parsed command-line arguments."""
        return cls(args.lr_schedule,
                   args.lr_schedule_steps or (args.epochs if total_steps is None else total_steps),
                   args.lr_warmup_steps, args.lr_warmup_start_factor, args.lr_min_factor)

    def validate(self):
        """Reject invalid schedule names, step counts, and scale factors."""
        if self.schedule not in ('constant', 'cosine'):
            raise ValueError('lr_schedule must be constant or cosine')
        for name in ('total_steps', 'warmup_steps'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(name + ' must be a nonnegative integer')
        for name in ('warmup_start_factor', 'min_factor'):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(name + ' must be finite and in (0, 1]')
        if self.schedule == 'constant' and self.warmup_steps:
            raise ValueError('Learning-rate warm-up requires lr_schedule=cosine')
        if self.schedule == 'cosine' and self.total_steps <= self.warmup_steps:
            raise ValueError('Cosine schedule needs total_steps > warmup_steps')

    def factor(self, completed_updates):
        """Return the learning-rate multiplier after the given update count."""
        if self.schedule == 'constant':
            return 1.0
        if completed_updates < self.warmup_steps:
            return self.warmup_start_factor + (1 - self.warmup_start_factor) * completed_updates / self.warmup_steps
        progress = min(1.0, (completed_updates - self.warmup_steps) / (self.total_steps - self.warmup_steps))
        return self.min_factor + (1 - self.min_factor) * (1 + math.cos(math.pi * progress)) / 2


class WarmupCosineScheduler:
    """Apply one warm-up cosine multiplier to all optimizer groups."""
    def __init__(self, optimizer, config, completed_updates=0, base_lrs=None, restoring=False):
        """Initialize and apply the schedule at a validated update position."""
        config.validate()
        if config.schedule != 'cosine':
            raise ValueError('WarmupCosineScheduler requires a cosine configuration')
        if isinstance(completed_updates, bool) or not isinstance(completed_updates, int) or completed_updates < 0:
            raise ValueError('completed_updates must be a nonnegative integer')
        self.optimizer, self.config, self.completed_updates = optimizer, config, completed_updates
        self.base_lrs = list(base_lrs if base_lrs is not None else
                             [group['lr'] for group in optimizer.param_groups])
        if len(self.base_lrs) != len(optimizer.param_groups) or any(
                not math.isfinite(lr) or lr <= 0 for lr in self.base_lrs):
            raise ValueError('Each optimizer parameter group needs a positive finite base learning rate')
        if restoring and any(not math.isclose(group['lr'], expected, rel_tol=1e-12, abs_tol=0)
                             for group, expected in zip(optimizer.param_groups, self.get_last_lr())):
            raise ValueError('Saved optimizer learning rates disagree with the scheduler state')
        self._apply()

    def get_last_lr(self):
        """Return the current rate for every optimizer group."""
        factor = self.config.factor(self.completed_updates)
        return [lr * factor for lr in self.base_lrs]

    def _apply(self):
        """Write the scheduled rates to the optimizer groups."""
        for group, lr in zip(self.optimizer.param_groups, self.get_last_lr()):
            group['lr'] = lr

    def step(self):
        """Advance by one completed optimizer update."""
        self.completed_updates += 1
        self._apply()

    def state_dict(self):
        """Serialize schedule configuration and progress."""
        return dict(schema_version=1, config=asdict(self.config), base_lrs=list(self.base_lrs),
                    completed_updates=self.completed_updates)

    @classmethod
    def from_state_dict(cls, optimizer, state):
        """Restore a scheduler and validate the optimizer's saved rates."""
        if state['schema_version'] != 1:
            raise ValueError('Unsupported learning-rate scheduler checkpoint version')
        return cls(optimizer, LearningRateConfig(**state['config']), state['completed_updates'],
                   state['base_lrs'], restoring=True)


@dataclass(frozen=True)
class PolicyTemperatureConfig:
    """Configure optional linear annealing of discrete policy temperature."""
    schedule: str = 'constant'
    start: float = 1.0
    anneal_steps: int = 0

    @classmethod
    def from_namespace(cls, args):
        """Build a configuration from parsed command-line arguments."""
        return cls(args.policy_temperature_schedule, args.policy_temperature_start,
                   args.policy_temperature_anneal_steps)

    def validate(self):
        """Reject unsupported schedules and invalid temperatures or durations."""
        if self.schedule not in ('constant', 'linear'):
            raise ValueError('Unsupported policy temperature schedule')
        if not math.isfinite(self.start) or self.start < 1:
            raise ValueError('Policy temperature start must be finite and >= 1')
        if isinstance(self.anneal_steps, bool) or not isinstance(self.anneal_steps, int) or self.anneal_steps < 0:
            raise ValueError('Temperature anneal steps must be a nonnegative integer')
        if self.schedule == 'linear' and self.anneal_steps < 1:
            raise ValueError('Linear temperature annealing needs positive steps')
        if self.schedule == 'constant' and (self.start != 1 or self.anneal_steps):
            raise ValueError('Disabled temperature schedule must use start=1 and steps=0')

    def temperature(self, completed_updates):
        """Return the discrete sampling temperature at an update position."""
        self.validate()
        if isinstance(completed_updates, bool) or not isinstance(completed_updates, int) or completed_updates < 0:
            raise ValueError('Completed updates must be a nonnegative integer')
        return (1.0 if self.schedule == 'constant' else
                1 + (self.start - 1) * max(0., 1 - completed_updates / self.anneal_steps))

    def random_spec(self, completed_updates):
        """Return rollout overrides, or ``None`` for exact default sampling."""
        temperature = self.temperature(completed_updates)
        return None if temperature == 1 else {'T': temperature, 'time_T': 1.0}

    def state_dict(self, completed_updates):
        """Serialize configuration and update position."""
        self.temperature(completed_updates)
        return dict(schema_version=1, config=asdict(self), completed_updates=completed_updates,
                    time_temperature=1.0, reward_temperature=1.0)

    def validate_resume(self, state, completed_updates):
        """Ensure resumed state matches the requested schedule."""
        if state is None:
            if self.schedule != 'constant':
                raise ValueError('Cannot enable temperature annealing on a resumed legacy run; use shared initialization')
        elif state != self.state_dict(completed_updates):
            raise ValueError('Temperature schedule or update position differs from checkpoint')

    def validate_training(self, loss_type, event_policy, exploration_fraction, replay_fraction, warmup_steps=0):
        """Validate compatibility with other training variants."""
        self.validate()
        if self.schedule != 'constant' and (loss_type != 'subtb' or event_policy != 'cwr_residual'
                or exploration_fraction or replay_fraction or warmup_steps):
            raise ValueError('Temperature variant requires joint SubTB/cwr_residual without prior/replay or flow warm-up')

"""Learning-rate schedules indexed by completed joint optimizer updates."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class LearningRateConfig:
    schedule: str = 'constant'
    total_steps: int = 0
    warmup_steps: int = 0
    warmup_start_factor: float = 0.1
    min_factor: float = 0.1

    @classmethod
    def from_namespace(cls, args, total_steps=None):
        return cls(args.lr_schedule,
                   args.lr_schedule_steps or (args.epochs if total_steps is None else total_steps),
                   args.lr_warmup_steps, args.lr_warmup_start_factor, args.lr_min_factor)

    def validate(self):
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
        if self.schedule == 'constant':
            return 1.0
        if completed_updates < self.warmup_steps:
            return self.warmup_start_factor + (1 - self.warmup_start_factor) * completed_updates / self.warmup_steps
        progress = min(1.0, (completed_updates - self.warmup_steps) / (self.total_steps - self.warmup_steps))
        return self.min_factor + (1 - self.min_factor) * (1 + math.cos(math.pi * progress)) / 2


class WarmupCosineScheduler:
    """Use one multiplier for all parameter groups and hold the final LR floor."""
    def __init__(self, optimizer, config, completed_updates=0, base_lrs=None, restoring=False):
        config.validate()
        if config.schedule != 'cosine':
            raise ValueError('WarmupCosineScheduler requires a cosine configuration')
        if isinstance(completed_updates, bool) or not isinstance(completed_updates, int) or completed_updates < 0:
            raise ValueError('completed_updates must be a nonnegative integer')
        self.optimizer = optimizer
        self.config = config
        self.completed_updates = completed_updates
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
        factor = self.config.factor(self.completed_updates)
        return [lr * factor for lr in self.base_lrs]

    def _apply(self):
        for group, lr in zip(self.optimizer.param_groups, self.get_last_lr()):
            group['lr'] = lr

    def step(self):
        self.completed_updates += 1
        self._apply()

    def state_dict(self):
        return dict(schema_version=1, config=asdict(self.config), base_lrs=list(self.base_lrs),
                    completed_updates=self.completed_updates)

    @classmethod
    def from_state_dict(cls, optimizer, state):
        if state['schema_version'] != 1:
            raise ValueError('Unsupported learning-rate scheduler checkpoint version')
        return cls(optimizer, LearningRateConfig(**state['config']), state['completed_updates'],
                   state['base_lrs'], restoring=True)

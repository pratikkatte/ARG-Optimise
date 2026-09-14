"""Training-only discrete exploration, indexed by completed joint updates."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class PolicyTemperatureConfig:
    schedule: str = 'constant'
    start: float = 1.0
    anneal_steps: int = 0

    @classmethod
    def from_namespace(cls, args):
        return cls(args.policy_temperature_schedule, args.policy_temperature_start,
                   args.policy_temperature_anneal_steps)

    def validate(self):
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
        self.validate()
        if isinstance(completed_updates, bool) or not isinstance(completed_updates, int) or completed_updates < 0:
            raise ValueError('Completed updates must be a nonnegative integer')
        return (1.0 if self.schedule == 'constant' else
                1 + (self.start - 1) * max(0., 1 - completed_updates / self.anneal_steps))

    def random_spec(self, completed_updates):
        temperature = self.temperature(completed_updates)
        # None preserves the ordinary sampling code and RNG stream exactly.
        return None if temperature == 1 else {'T': temperature, 'time_T': 1.0}

    def state_dict(self, completed_updates):
        self.temperature(completed_updates)
        return dict(schema_version=1, config=asdict(self), completed_updates=completed_updates,
                    time_temperature=1.0, reward_temperature=1.0)

    def validate_resume(self, state, completed_updates):
        if state is None:
            if self.schedule != 'constant':
                raise ValueError('Cannot enable temperature annealing on a resumed legacy run; use shared initialization')
        elif state != self.state_dict(completed_updates):
            raise ValueError('Temperature schedule or update position differs from checkpoint')

    def validate_training(self, loss_type, event_policy, exploration_fraction, replay_fraction, warmup_steps=0):
        self.validate()
        if self.schedule != 'constant' and (loss_type != 'subtb' or event_policy != 'cwr_residual'
                or exploration_fraction or replay_fraction or warmup_steps):
            raise ValueError('Temperature variant requires joint SubTB/cwr_residual without prior/replay or flow warm-up')

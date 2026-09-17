"""Wall-clock progress reporting without touching tensors or sampling RNGs."""
import json
import math
from pathlib import Path
import time
import warnings


class ProgressReporter:
    """Print phase boundaries and throttle event-level updates to a time interval.

    This is deliberately synchronous: an event count advances only after the
    corresponding work returns. It is not a background liveness watchdog.
    """

    def __init__(self, path, *, verbose=True, every_seconds=15., clock=None):
        if not math.isfinite(every_seconds) or every_seconds <= 0:
            raise ValueError('progress_every_seconds must be positive and finite')
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.verbose, self.every_seconds = verbose, float(every_seconds)
        self.clock = clock or time.monotonic
        self.started = self.phase_started = self.clock()
        self.last_emitted = -math.inf
        self.phase = 'startup'
        self.fields = {}
        self.summary = None

    def due(self):
        return self.clock()-self.last_emitted >= self.every_seconds

    def begin(self, phase, **fields):
        self.phase, self.fields = phase, dict(fields)
        self.phase_started = self.clock()
        self.update(force=True)

    def update(self, *, force=False, **fields):
        self.fields.update(fields)
        now = self.clock()
        if not force and now-self.last_emitted < self.every_seconds:
            return
        record = dict(phase=self.phase, elapsed_seconds=round(now-self.started, 3),
                      phase_elapsed_seconds=round(now-self.phase_started, 3), **self.fields)
        with self.path.open('a') as handle:
            handle.write(json.dumps(record, allow_nan=False)+'\n')
        if self.verbose:
            details = ' '.join(f'{key}={value}' for key, value in self.fields.items())
            print(f'[progress {record["elapsed_seconds"]:.1f}s] {self.phase} {details}', flush=True)
        if self.summary is not None:
            try:
                # Update status without advancing W&B's optimizer-step history.
                self.summary.update({'progress': record})
            except Exception as exc:
                warnings.warn(f'W&B progress summary unavailable ({type(exc).__name__}); '
                              'console and progress.jsonl remain active', RuntimeWarning)
                self.summary = None
        self.last_emitted = now

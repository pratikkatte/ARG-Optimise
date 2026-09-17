"""Explicit boundary between the implemented environment and deferred neural migration."""


def require_neural_migration():
    raise NotImplementedError(
        'Infinite-sites Phase 1 implements the environment and diagnostic sampler only. '
        'Neural policy/training/checkpoint inference migration is pending; JC69 workflows '
        'are retired. Use validation/scripts/sample_infinite_sites.py for diagnostic ARGs.'
    )

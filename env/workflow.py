"""Legacy workflow guard retained for callers outside this repository."""


def require_neural_migration():
    raise ValueError('This legacy JC69 workflow is retired; use the infinite-sites train.py or infer.py entrypoint')

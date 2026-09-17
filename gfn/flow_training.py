"""The shared encoder is trained jointly; frozen-feature warm-up is retired."""


def warmup_flow(generator, steps, episodes=32, seed=100019):
    if steps != 0:
        raise ValueError('Cached-feature flow warm-up is incompatible with the shared trainable encoder')
    return {}


def migrate_independent_flow_checkpoint(checkpoint):
    raise ValueError('Legacy flow checkpoints cannot migrate to infinite-sites shared encoding')

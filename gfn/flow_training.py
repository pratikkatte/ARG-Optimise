"""Policy and flow head train jointly from the first update; no head-only prefit."""


def warmup_flow(generator, steps, episodes=32, seed=100019):
    if steps != 0:
        raise ValueError('Head-only flow prefit is unsupported in shared and frozen modes; train policy and flow head jointly')
    return {}


def migrate_independent_flow_checkpoint(checkpoint):
    raise ValueError('Legacy flow checkpoints cannot migrate to infinite-sites shared encoding')

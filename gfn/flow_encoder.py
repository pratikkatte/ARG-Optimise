"""Retired fixed encoder entrypoint; use the generator's shared state_encoder."""


class FrozenFlowEncoder:
    def __init__(self, *args, **kwargs):
        raise ValueError('FrozenFlowEncoder is retired; policy and flow share one trainable state_encoder')

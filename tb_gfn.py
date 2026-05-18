import torch
try:
    from .models import ARGModel
except ImportError:
    from models import ARGModel


class TBGFlowNetGenerator(torch.nn.Module):
    def __init__(self, env, cfg=None):
        super().__init__()
        self.cfg = cfg
        self.env = env

        self.arg_model = ARGModel(env, cfg)

    def forward(self, input_dict):
        ret = self.arg_model(input_dict)
        return ret

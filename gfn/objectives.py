"""Validation shared by objective selection and loss reporting."""
import math


def validate_objective(loss_type, subtb_lambda, flow_lr):
    if loss_type not in ("tb", "subtb"):
        raise ValueError("loss_type must be tb or subtb")
    if not math.isfinite(subtb_lambda) or subtb_lambda < 0:
        raise ValueError("subtb_lambda must be finite and >= 0")
    if not math.isfinite(flow_lr) or flow_lr <= 0:
        raise ValueError("flow_lr must be finite and > 0")



def resolve_log_loss(log_loss, loss_type):
    """Default to the optimized loss; an additional TB metric is opt-in."""
    if loss_type not in ('tb', 'subtb'):
        raise ValueError('loss_type must be tb or subtb')
    names = (loss_type,) if log_loss is None else ((log_loss,) if isinstance(log_loss, str) else log_loss)
    if not isinstance(names, (tuple, list)) or not names or any(name not in ('tb', 'subtb') for name in names):
        raise ValueError('log_loss must be a nonempty list containing tb and/or subtb')
    if len(set(names)) != len(names):
        raise ValueError('log_loss must not contain duplicates')
    if loss_type not in names:
        raise ValueError('log_loss must include the optimized loss_type')
    if loss_type == 'tb' and 'subtb' in names:
        raise ValueError('Logging SubTB requires loss_type=subtb and its trained flow head')
    return tuple(names)

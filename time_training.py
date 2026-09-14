"""Explicit, prediction-preserving migration of the learned waiting-time head."""
from copy import deepcopy

import torch

from time_env import checkpoint_time_policy


def migrate_gamma_time_checkpoint(checkpoint, target):
    """Add a zero log-shape head, preserving every existing weight/Adam moment.

    The supplied target is the matching Gamma generator. Optimizer parameter
    indices are remapped because the two new shape parameters precede the event
    head. Their Adam state starts empty; all pre-existing parameters retain their
    own step counts and moments. No checkpoint is silently upgraded by load().
    """
    metadata = checkpoint.get('metadata', {})
    if checkpoint_time_policy(metadata) != 'cwr_exponential':
        raise ValueError('Gamma migration requires continuous CwR timing')
    saved_model = metadata.get('model', {})
    if saved_model.get('continuous_time_head', 'exponential') != 'exponential':
        raise ValueError('Gamma migration requires an exponential source checkpoint')
    if target.arg_model.continuous_time_head != 'gamma':
        raise ValueError('Gamma migration requires a Gamma target generator')
    if any(target.model_kwargs.get(k) != v for k, v in saved_model.items()
           if k != 'continuous_time_head'):
        raise ValueError('Gamma migration cannot change other policy model settings')
    if metadata.get('loss_type', 'tb') != target.loss_type or (
            target.loss_type == 'subtb' and metadata.get('flow_head_version') != target.flow_head_version):
        raise ValueError('Gamma migration cannot change the objective or flow architecture')
    if target.neural_source_flow:
        raise ValueError('Legacy Gamma migration supports scalar-source checkpoints only; version 5 starts fresh')
    converted = deepcopy(checkpoint)
    saved = converted['generator_state_dict']
    target_state = target.state_dict()
    expected = {prefix+'.shape_layer.'+part
                for prefix in ('arg_model.time_scorer', 'time_model') for part in ('weight', 'bias')}
    if set(target_state)-set(saved) != expected or set(saved)-set(target_state):
        raise ValueError('Unexpected state-dictionary difference during Gamma migration')
    if any(saved[k].shape != target_state[k].shape for k in saved):
        raise ValueError('Gamma migration cannot change existing parameter shapes')
    device = saved['_Z'].device
    for name in expected:
        value = target_state[name]
        if torch.count_nonzero(value):
            raise ValueError('Gamma target log-shape head must be zero initialized')
        saved[name] = value.detach().to(device).clone()

    new_names = {'arg_model.time_scorer.shape_layer.weight', 'arg_model.time_scorer.shape_layer.bias'}
    if 'opt_state_dict' in converted:
        old_opt = converted['opt_state_dict']
        target_opt = target.opt.state_dict()
        if len(old_opt['param_groups']) != len(target_opt['param_groups']):
            raise ValueError('Incompatible optimizer groups for Gamma migration')
        names = {id(p): name for name, p in target.named_parameters()}
        migrated = dict(state={}, param_groups=[])
        seen = set()
        for old_group, new_group, live_group in zip(old_opt['param_groups'], target_opt['param_groups'], target.opt.param_groups):
            group_names = [names[id(p)] for p in live_group['params']]
            existing = [(name, index) for name, index in zip(group_names, new_group['params']) if name not in new_names]
            if len(existing) != len(old_group['params']):
                raise ValueError('Incompatible optimizer parameter order for Gamma migration')
            for old_index, (name, new_index) in zip(old_group['params'], existing):
                seen.add(old_index)
                if old_index in old_opt['state']:
                    entry = old_opt['state'][old_index]
                    for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
                        if key in entry and entry[key].shape != saved[name].shape:
                            raise ValueError('Optimizer moments do not match the expected parameter order')
                    migrated['state'][new_index] = entry
            migrated['param_groups'].append({**old_group, 'params': list(new_group['params'])})
        if set(old_opt['state'])-seen:
            raise ValueError('Unexpected optimizer state outside declared parameter groups')
        converted['opt_state_dict'] = migrated
    converted['metadata']['model'] = {**saved_model, 'continuous_time_head': 'gamma'}
    converted['metadata']['time_head_migration'] = dict(
        from_head='exponential', to_head='gamma', initial_log_shape=0.,
        preserved_existing_optimizer_state=True, new_parameters=sorted(new_names))
    return converted

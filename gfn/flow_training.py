"""Fixed-policy fitting of the intermediate flow head on complete trajectories."""
from contextlib import contextmanager
import random
from copy import deepcopy

import numpy as np
import torch
from gfn.subtb import geometric_subtb_loss


def migrate_independent_flow_checkpoint(checkpoint):
    """Explicitly convert a version-3 checkpoint to version 4 without changing F.

    Freeze the old normalizer value into a separately stored baseline buffer.
    The trainable parameters and Adam moments keep their names and ordering.
    Automatic loading across these versions remains forbidden.
    """
    metadata = checkpoint.get('metadata', {})
    if metadata.get('loss_type') != 'subtb' or metadata.get('flow_head_version') != 3:
        raise ValueError('Independent-flow migration requires a version-3 SubTB checkpoint')
    converted = deepcopy(checkpoint)
    state = converted['generator_state_dict']
    if state['_Z'].numel() != 1:
        raise ValueError('Independent-flow migration requires a scalar logZ')
    state['flow_baseline_log_z'] = state['_Z'].detach().clone().reshape(())
    converted['metadata'].update(
        flow_head_version=4,
        flow_baseline_log_z=state['flow_baseline_log_z'].item(),
        flow_migrated_from_version=3,
    )
    return converted


@contextmanager
def preserve_sampling(generator, seed):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    env_state = generator.env.rng.getstate()
    modes = [(module, module.training) for module in generator.modules()]
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            generator.env.rng.seed(seed)
            generator.eval()
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        generator.env.rng.setstate(env_state)
        for module, mode in modes:
            module.training = mode


class FrozenFlowBatch:
    """Exact SubTB on fixed paths, with detached encoder features and policy scores."""

    def __init__(self, generator, outputs, features):
        self.features = torch.cat(features).detach()
        lengths = outputs['lengths'].tolist()
        b, t = outputs['log_paths_pf'].shape
        # The hook receives time-major active states; map them back to paths.
        mapping = torch.full(
            (b, t + 1), len(self.features), dtype=torch.long, device=self.features.device
        )
        cursor = 0
        for step in range(t):
            for row, n in enumerate(lengths):
                if step < n:
                    if step > 0 or generator.neural_source_flow:
                        mapping[row, step] = cursor
                    cursor += 1
        baseline = outputs['state_flows'].detach().clone()
        with torch.no_grad():
            old = generator.flow_head(self.features).squeeze(-1).double() * generator.flow_output_scale
            old = torch.cat((old, old.new_zeros(1)))
            baseline -= old[mapping]
        self.mapping, self.baseline = mapping, baseline
        self.log_pf = outputs['log_paths_pf'].detach()
        self.log_pb = outputs['log_paths_pb'].detach()
        self.lengths = outputs['lengths']
        self.log_rewards = outputs['log_rewards'].detach()

    def loss(self, generator):
        values = generator.flow_head(self.features).squeeze(-1).double() * generator.flow_output_scale
        values = torch.cat((values, values.new_zeros(1)))
        flows = self.baseline + values[self.mapping]
        return geometric_subtb_loss(self.log_pf, self.log_pb, flows, self.lengths,
                                    self.log_rewards, generator.subtb_lambda)

    @classmethod
    @torch.no_grad()
    def sample(cls, generator, episodes=32, seed=100019):
        from gfn.rollout import RolloutWorker

        features = []
        with preserve_sampling(generator, seed):
            hook = generator.flow_head.register_forward_pre_hook(
                lambda module, args: features.append(args[0].detach())
            )
            try:
                outputs, _ = RolloutWorker(generator.env).rollout(
                    generator, episodes=episodes, collect_flows=True
                )
            finally:
                hook.remove()
        if not features:
            return None
        return cls(generator, outputs, features)


def warmup_flow(generator, steps, episodes=32, seed=100019):
    """Fit only the flow head; preserve policy weights, log Z, and sampling RNG."""
    if steps < 0 or episodes < 1:
        raise ValueError('Flow warm-up needs nonnegative steps and positive episodes')
    if not steps or generator.loss_type != 'subtb' or generator.flow_head_version < 2:
        return {}
    batch = FrozenFlowBatch.sample(generator, episodes, seed)
    if batch is None:
        return {}
    initial = float(batch.loss(generator).detach())
    for _ in range(steps):
        generator.opt.zero_grad(set_to_none=True)
        loss = batch.loss(generator)
        if not bool(torch.isfinite(loss)):
            raise ValueError('Non-finite flow warm-up loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.flow_params, generator.grad_clip)
        generator.opt.step()
    generator.opt.zero_grad(set_to_none=True)
    final = float(batch.loss(generator).detach())
    if not np.isfinite(final):
        raise ValueError('Non-finite flow warm-up loss')
    return dict(flow_warmup_initial_loss=initial, flow_warmup_final_loss=final,
                flow_warmup_steps=steps, flow_warmup_episodes=episodes)

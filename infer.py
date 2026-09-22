"""Sample and independently verify ARGs from a self-contained neural checkpoint."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
from gfn.rollout import RolloutWorker, RolloutFailure
from training.checkpoints import (load_checkpoint, generator_from_checkpoint, seed_everything,
                                  environment_from_metadata, validate_metadata)
from utils import action_as_dict


class TerminalValidationError(AssertionError):
    """A reproducible independent-likelihood discrepancy, never silently skipped."""
    def __init__(self, message, env, state, reference):
        super().__init__(message)
        def number(value):
            return float(value) if math.isfinite(value) else str(value)
        self.details = dict(log_likelihood=number(state.partial_log_likelihood),
            reference_log_likelihood=number(reference.log_likelihood),
            exposure=number(state.exposure*2*env.population_size), reference_exposure=number(reference.exposure),
            compatible_branch_lengths=[number(x) for x in state.completed_site_lengths*2*env.population_size],
            reference_branch_lengths=[number(x) for x in reference.compatible_branch_lengths],
            current_time=state.current_time, actions=[action_as_dict(a) for a in state.actions])


def resolve_device(device='auto'):
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if device in (None,'auto') else device
    if str(device) == 'mps':
        raise ValueError('MPS does not support the required float64 scoring; use CPU or CUDA')
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable')
    return torch.device(device)


def validate_terminal(env, state):
    if not state.is_done or not math.isfinite(state.log_reward):
        raise ValueError('Inference requires a completed ARG with positive finite likelihood')
    reference = env.evaluate_terminal(state)
    try:
        if not math.isfinite(reference.log_likelihood) or abs(reference.log_likelihood-state.partial_log_likelihood) > 1e-9:
            raise AssertionError('Independent terminal likelihood disagrees')
        np.testing.assert_allclose(reference.exposure, state.exposure*2*env.population_size, rtol=1e-12, atol=1e-8)
        np.testing.assert_allclose(reference.compatible_branch_lengths,
                                   state.completed_site_lengths*2*env.population_size, rtol=1e-12, atol=1e-8)
    except AssertionError as exc:
        raise TerminalValidationError(str(exc),env,state,reference) from exc
    return reference


@torch.no_grad()
def collect_samples(generator, num_args=16, batch_size=2, seed=100007, max_events=10000):
    if num_args < 1 or batch_size < 1:
        raise ValueError('Sample and batch counts must be positive')
    seed_everything(seed); generator.env.rng.seed(seed)
    generator.eval()
    worker = RolloutWorker(generator.env, max_events=max_events)
    records, trees, paths = [], [], []
    for start in range(0, num_args, batch_size):
        try:
            outputs, trajectories = worker.rollout(generator, min(batch_size,num_args-start), return_states=True)
        except RolloutFailure as exc:
            exc.histories = [[action_as_dict(a) for a in p.actions] for p in paths]+exc.histories
            exc.completed_records = records
            raise
        for i, (state, path) in enumerate(zip(outputs['states'],trajectories)):
            reference = validate_terminal(generator.env, state)
            pf = float(outputs['log_paths_pf'][i].sum())
            records.append(dict(index=len(records), status='complete', source='policy', temperature=1.,
                log_likelihood=state.partial_log_likelihood, independent_log_likelihood=reference.log_likelihood,
                log_prior=state.accumulated_log_prior, log_policy_density=pf, log_backward_probability=0.,
                log_reward=state.log_reward, log_importance_weight=state.log_reward-pf,
                event_count=len(path), recombinations=sum(a.event_type=='recomb' for a in path.actions),
                actions=[action_as_dict(a) for a in path.actions]))
            trees.append(generator.env.save_to_tree_sequence(state)); paths.append(path)
    return records, trees, paths


def sample_summary(records):
    weights = np.asarray([r['log_importance_weight'] for r in records])
    weights = np.exp(weights-weights.max()); weights /= weights.sum()
    return dict(num_completed=len(records), num_failed=0,
                importance_ess=float(1/(weights@weights)), importance_ess_fraction=float(1/(weights@weights)/len(weights)),
                max_normalized_importance_weight=float(weights.max()),
                max_likelihood_error=max(abs(r['log_likelihood']-r['independent_log_likelihood']) for r in records))


def run_inference(checkpoint, output_dir='inferred_args', num_args=16, batch_size=2, seed=100007,
                  device='auto', temperature=None, max_events=10000, cpu_threads=1, verbose=False):
    if temperature is not None and temperature != 1:
        raise ValueError('Infinite-sites inference currently requires temperature 1')
    torch.set_num_threads(cpu_threads)
    data = load_checkpoint(checkpoint)
    generator = generator_from_checkpoint(data, resolve_device(device), optimizer=False)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Inference requires an empty output directory')
    output.mkdir(parents=True, exist_ok=True)
    try:
        records, trees, _ = collect_samples(generator, num_args, batch_size, seed, max_events)
    except RolloutFailure as exc:
        (output/'failure.json').write_text(json.dumps(dict(error=str(exc), histories=exc.histories, completed_samples=getattr(exc,'completed_records',[])), indent=2))
        raise
    for row, ts in zip(records, trees):
        row['trees_file'] = f'arg_{row["index"]:04d}.trees'
        ts.dump(output/row['trees_file'])
    manifest = dict(schema_version=2, model_version=data['metadata']['model_version'], mutation_model='infinite_sites',
                    sampling_distribution='learned_policy', posterior_calibration_established=False,
                    environment_fingerprint=generator.env.dataset_fingerprint,
                    haplotype_ids=list(generator.env.snp_data.haplotype_ids), num_variants=generator.env.num_variants,
                    sequence_length=generator.env.sequence_length, seed=seed,
                    summary=sample_summary(records), samples=records)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False))
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--num-args', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--seed', type=int, default=100007)
    parser.add_argument('--max-events', type=int, default=10000)
    parser.add_argument('--device', default='auto')
    args=vars(parser.parse_args(argv))
    print(json.dumps(run_inference(**args)['summary'], indent=2))


if __name__ == '__main__':
    main()

"""Sample compatible diagnostic ARGs and independently verify every score.

These are proposal samples, not posterior samples. No ground-truth ancestry is read.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from env.env import SimpleARGEnvironment
from env.snp_data import load_snp_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replicate-dir', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--samples', type=int, default=1)
    parser.add_argument('--max-events', type=int, default=10000)
    parser.add_argument('--output-dir', type=Path, help='New directory for manifest and ancestry files')
    args = parser.parse_args(argv)
    if args.samples < 1:
        parser.error('--samples must be positive')
    data = load_snp_dataset(args.replicate_dir)
    params = json.loads((args.replicate_dir / 'metadata.json').read_text())['parameters']
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for seed in range(args.seed, args.seed + args.samples):
        env = SimpleARGEnvironment(snp_data=data, population_size=params['population_size'],
                                  mutation_rate=params['mutation_rate'],
                                  recombination_rate=params['recombination_rate'], seed=seed)
        state, trajectory = env.sample_compatible_trajectory(args.max_events)
        independent = env.evaluate_terminal(state)
        np.testing.assert_allclose(state.partial_log_likelihood, independent.log_likelihood,
                                   rtol=0, atol=1e-9)
        np.testing.assert_allclose(state.exposure * (2 * env.population_size), independent.exposure,
                                   rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(state.completed_site_lengths * (2 * env.population_size),
                                   independent.compatible_branch_lengths, rtol=1e-12, atol=1e-10)
        if independent.zero_likelihood:
            raise AssertionError('compatible proposal produced zero likelihood')
        rows.append(dict(seed=seed, events=len(trajectory), log_likelihood=state.partial_log_likelihood,
                         independent_log_likelihood=independent.log_likelihood,
                         log_prior=state.accumulated_log_prior,
                         log_proposal=float(sum(trajectory.log_proposals)),
                         log_reward=state.log_reward, exposure_generations_bp=independent.exposure))
        if args.output_dir is not None:
            env.save_to_tree_sequence(state, args.output_dir / f'arg_seed{seed}.trees')
    report = dict(sampler='prior discrete actions conditioned on compatibility; original prior waits',
                  posterior_samples=False, num_haplotypes=data.num_haplotypes,
                  num_variants=data.num_variants, sequence_length=data.sequence_length, results=rows)
    text = json.dumps(report, indent=2, allow_nan=False)
    if args.output_dir is not None:
        (args.output_dir / 'manifest.json').write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()

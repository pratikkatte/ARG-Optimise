"""Export validated policy ARGs from each final checkpoint, with provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import tskit

from env.snp_data import load_snp_dataset
from infer import collect_samples, resolve_device, sample_summary
from training.checkpoints import load_checkpoint, generator_from_checkpoint


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-args', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=20260924)
    parser.add_argument('--checkpoint', type=Path,
                        help='Sample only this checkpoint; dataset defaults to its parent directory.')
    parser.add_argument('--dataset', choices=('r1', 'r2', 'r4'),
                        help='Dataset for --checkpoint, verified against checkpoint observations.')
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT / 'paper/outputs/argflow')
    args = parser.parse_args()
    if args.num_args < 1 or args.batch_size < 1:
        parser.error('sample and batch counts must be positive')
    if args.dataset is not None and args.checkpoint is None:
        parser.error('--dataset requires --checkpoint')
    torch.set_num_threads(1)
    device = resolve_device(args.device)
    print(f'Device: {device} ({torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"})', flush=True)
    if args.checkpoint is not None:
        checkpoints = [args.checkpoint.resolve()]
        if not checkpoints[0].is_file():
            raise ValueError('Checkpoint must exist')
        if args.dataset is None and checkpoints[0].parent.name not in ('r1', 'r2', 'r4'):
            raise ValueError('Specify --dataset or use a directory named r1, r2, or r4')
    else:
        checkpoints = sorted((ROOT / 'paper/checkpoints').glob('*/*.pt'))
        checkpoints = [p for p in checkpoints if p.parent.name in ('r1', 'r2', 'r4')]
        if {p.parent.name for p in checkpoints} != {'r1', 'r2', 'r4'}:
            raise ValueError('Missing dataset checkpoints')
    # Check all destinations before starting any sampling.
    for checkpoint in checkpoints:
        output = args.output_dir / (args.dataset or checkpoint.parent.name) / checkpoint.stem
        if output.exists() and any(output.iterdir()):
            raise ValueError(f'Output must be empty: {output}')
    results = []
    for checkpoint_index, checkpoint in enumerate(checkpoints):
        started = time.monotonic()
        dataset = args.dataset or checkpoint.parent.name
        output = args.output_dir / dataset / checkpoint.stem
        data = load_checkpoint(checkpoint)
        generator = generator_from_checkpoint(data, device, optimizer=False)
        observed = load_snp_dataset(ROOT / 'paper/datasets' / dataset / 'rep0')
        np.testing.assert_array_equal(observed.genotypes, generator.env.snp_data.genotypes)
        np.testing.assert_array_equal(observed.positions, generator.env.snp_data.positions)
        assert observed.haplotype_ids == generator.env.snp_data.haplotype_ids
        assert observed.sequence_length == generator.env.sequence_length
        output.mkdir(parents=True, exist_ok=True)
        manifest = dict(
            schema_version=2, dataset=dataset,
            checkpoint=str(checkpoint.relative_to(ROOT) if checkpoint.is_relative_to(ROOT) else checkpoint),
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            model_version=data['metadata']['model_version'], mutation_model='infinite_sites',
            sampling_distribution='learned_policy', posterior_calibration_established=False,
            temperature=1.0, importance_resampled=False,
            environment_fingerprint=generator.env.dataset_fingerprint,
            haplotype_ids=list(observed.haplotype_ids), num_variants=observed.num_variants,
            sequence_length=observed.sequence_length, time_units='generations',
            requested_samples=args.num_args, batch_size=args.batch_size,
            base_seed=args.seed + checkpoint_index * 100000,
            seed_protocol='Each batch uses base_seed + zero-based batch index.',
            device=str(device), torch_version=torch.__version__,
            batches=[], samples=[], status='running')
        write_json(output / 'manifest.json', manifest)
        print(f'Starting {dataset}/{checkpoint.stem}', flush=True)
        try:
            for batch_index, start in enumerate(range(0, args.num_args, args.batch_size)):
                count = min(args.batch_size, args.num_args - start)
                seed = manifest['base_seed'] + batch_index
                records, trees, paths = collect_samples(generator, count, count, seed, 10000)
                for offset, (record, ts) in enumerate(zip(records, trees)):
                    index = start + offset
                    record.update(index=index, batch_index=batch_index, seed=seed,
                                  trees_file=f'arg_{index:04d}.trees')
                    destination = output / record['trees_file']
                    ts.dump(destination)
                    saved = tskit.load(destination)
                    assert saved.num_samples == observed.num_haplotypes
                    assert saved.sequence_length == observed.sequence_length
                    assert saved.time_units == 'generations'
                    assert all(tree.num_roots == 1 for tree in saved.trees())
                    np.testing.assert_array_equal(saved.samples(), np.arange(observed.num_haplotypes))
                    # Confirm observed derived sets remain clades after serialization.
                    for site, position in enumerate(observed.positions):
                        tree = saved.at(position)
                        derived = set(np.flatnonzero(observed.genotypes[:, site]))
                        assert any(set(tree.samples(node)) == derived for node in tree.nodes())
                    record['trees_sha256'] = hashlib.sha256(destination.read_bytes()).hexdigest()
                manifest['samples'].extend(records)
                manifest['batches'].append(dict(index=batch_index, start=start, count=count, seed=seed))
                manifest['summary'] = sample_summary(manifest['samples'])
                manifest['elapsed_seconds'] = time.monotonic() - started
                write_json(output / 'manifest.json', manifest)
                print(f'{dataset}/{checkpoint.stem}: {len(manifest["samples"])}/{args.num_args} '
                      f'validated and saved, {manifest["elapsed_seconds"]:.1f}s', flush=True)
                del records, trees, paths
            assert len(list(output.glob('*.trees'))) == args.num_args
            manifest['status'] = 'complete'
            write_json(output / 'manifest.json', manifest)
        except Exception as exc:
            manifest['status'] = 'failed'
            manifest['error'] = str(exc)
            write_json(output / 'manifest.json', manifest)
            failure = dict(error=str(exc))
            if hasattr(exc, 'histories'):
                failure['histories'] = exc.histories
            if hasattr(exc, 'details'):
                failure['details'] = exc.details
            write_json(output / 'failure.json', failure)
            raise
        results.append(dict(dataset=dataset, checkpoint=manifest['checkpoint'],
                            output=str(output), summary=manifest['summary'],
                            elapsed_seconds=manifest['elapsed_seconds']))
        write_json(args.output_dir / 'sampling_summary.json', results)
        del generator, data
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()

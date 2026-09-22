"""Summarize completed pilot evaluations without treating loss as convergence."""
import argparse
from collections import defaultdict
import datetime
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


METRICS = {
    'eval_ess': 'ESS (256 fresh draws per repeat)',
    'eval_log_weight_std': 'Log importance weight standard deviation',
    'eval_density_prior_relative_slope': 'Prior-relative density slope (target = 1)',
}


def read_rows(path):
    if not path.exists():
        return []
    rows = []
    lines = path.read_text().splitlines()
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # A running writer may not have finished its last line yet.
            if index != len(lines) - 1:
                raise
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    root, output = Path(args.run), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    summary = dict(updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   experiment=str(root.resolve()), datasets={})
    fig, axes = plt.subplots(3, 4, figsize=(16, 10), squeeze=False)
    for row, ratio in enumerate(('r1', 'r2', 'r4')):
        train = read_rows(root / ratio / 'training.jsonl')
        evaluations = read_rows(root / ratio / 'evaluation.jsonl')
        grouped = defaultdict(list)
        for evaluation in evaluations:
            grouped[evaluation['step']].append(evaluation)
        points = []
        for step, repeats in sorted(grouped.items()):
            point = dict(step=step, repeats=len(repeats),
                         samples_per_repeat=sorted({r['eval_episodes'] for r in repeats}))
            for metric in METRICS:
                values = [r[metric] for r in repeats]
                point[metric] = dict(mean=float(np.mean(values)),
                                     minimum=float(min(values)), maximum=float(max(values)))
            # TB MSE = variance(log R - log q) + (log Z - mean(log R - log q))**2.
            # This separates a slow source-flow offset from density calibration.
            source_errors = [max(0., r['eval_tb_loss'] - r['eval_log_weight_std']**2)**.5
                             for r in repeats]
            point['source_flow_absolute_offset'] = dict(mean=float(np.mean(source_errors)),
                minimum=float(min(source_errors)), maximum=float(max(source_errors)))
            points.append(point)
        entry = dict(completed_training_updates=max((r['step'] for r in train), default=0),
                     evaluations=points, convergence_verified=False)
        status_path = root / ratio / 'run_status.json'
        if status_path.exists():
            entry['run_status'] = json.loads(status_path.read_text())
        summary['datasets'][ratio] = entry
        ax = axes[row, 0]
        for key, label in (('subtb_loss', 'SubTB'), ('tb_loss', 'Full trajectory balance')):
            if train:
                ax.plot([r['step'] for r in train], [r[key] for r in train], alpha=.65, label=label)
        ax.set_title(ratio + ': training objectives')
        ax.set_yscale('log')
        if train:
            ax.legend(fontsize=8)
        for column, (metric, label) in enumerate(METRICS.items(), start=1):
            ax = axes[row, column]
            for point in points:
                value = point[metric]
                ax.errorbar(point['step'], value['mean'],
                    yerr=[[value['mean'] - value['minimum']], [value['maximum'] - value['mean']]],
                    fmt='o', color='tab:blue' if point['repeats'] == 3 else 'tab:orange', capsize=4)
            if metric.endswith('_slope'):
                ax.axhline(1, color='gray', linestyle='--')
            ax.set_title(ratio + ': ' + label, fontsize=9)
        for ax in axes[row]:
            ax.set_xlabel('Completed updates')
            ax.grid(alpha=.2)
    fig.suptitle('Pilot evaluations: bars show repeat range; orange points have fewer than 3 repeats')
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(output / 'pilot_progress.png', dpi=160)
    plt.close(fig)
    (output / 'pilot_progress.json').write_text(json.dumps(summary, indent=2) + '\n')
    lines = ['# Live pilot progress', '', 'Updated: ' + summary['updated_at_utc'], '',
             'These diagnostics do not establish convergence. Each complete evaluation uses three independent repeats. '
             'Ranges below are observed repeat ranges, not confidence intervals.', '',
             '| Dataset | Training updates | Evaluation step | Repeats | ESS | Log-weight SD | Prior-relative slope |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for ratio, entry in summary['datasets'].items():
        for point in entry['evaluations']:
            values = [f"{point[k]['mean']:.3f} [{point[k]['minimum']:.3f}, {point[k]['maximum']:.3f}]"
                      for k in METRICS]
            lines.append(f"| {ratio} | {entry['completed_training_updates']} | {point['step']} | {point['repeats']} | "
                         + ' | '.join(values) + ' |')
    lines += ['', '![Pilot diagnostics](pilot_progress.png)', '']
    (output / 'PILOT_PROGRESS.md').write_text('\n'.join(lines))
    print('\n'.join(lines[:-2]))


if __name__ == '__main__':
    main()

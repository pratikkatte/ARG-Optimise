"""Download the training measurements used by Appendix E.1 from W&B."""
import argparse
import json
from pathlib import Path
import wandb

ROOT = Path(__file__).resolve().parents[2]
KEEP = {
    'step', 'train_update', '_step', '_timestamp', 'loss', 'subtb_loss',
    'tb_loss', 'policy_lr', 'flow_lr', 'eval_checkpoint_step',
    'eval_log_weight_std', 'eval_density_raw_rmse',
    'eval_density_raw_slope', 'eval_density_raw_pearson',
    'eval_density_prior_relative_slope', 'eval_density_prior_relative_pearson',
    'eval_episodes', 'eval_async_failed', 'eval_subtb_loss', 'eval_tb_loss',
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_ids', nargs='+')
    args = parser.parse_args()
    cfg = json.loads(Path(__file__).with_name('appendix_e1.json').read_text())
    out = Path(cfg['output']).parent / 'training'
    out.mkdir(parents=True, exist_ok=True)
    api = wandb.Api(timeout=45)
    for rid in args.run_ids:
        run = api.run('pratikkatte/ARG-Optimise/' + rid)
        rows = [{k: v for k, v in row.items() if k in KEEP}
                for row in run.scan_history(page_size=1000)]
        data = dict(run_id=rid, name=run.name, url=run.url, state=run.state,
                    config={k:v for k,v in run.config.items() if any(
                        t in k for t in ('resume', 'seed', 'dataset', 'eval', 'batch'))},
                    rows=rows)
        (out / (rid + '.json')).write_text(json.dumps(data, indent=2) + '\n')
        print(rid, run.name, len(rows), 'rows downloaded', flush=True)

if __name__ == '__main__':
    main()

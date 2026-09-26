# Paper scripts

Run commands from the repository root. Outputs go to `paper/outputs/`.

- `run_arginfer.sh`, `run_singer.sh`: run the ARGInfer and SINGER baselines for one dataset.
- [Dataset simulation](simulate_infinite_sites.md): `simulate_infinite_sites.py` and its tests.
- [Manuscript table and figures](paper_datasets/README.md): the `paper_datasets/`
  folder includes the evaluation runner, figures, configuration, helpers, and tests.
- `evaluate_gfn_manuscript.py`: evaluate saved GFN draws from a manifest.
- `compare_r1_arg_summaries_256.py`: reproduce the specific R1 comparison.
- `plot_paper_posterior_agreement.py`: posterior agreement plots and shared clade helpers.
- `gfn_autocorrelation_ess.py`: autocorrelation diagnostics for ordered GFN draws.

```bash
python paper/scripts/paper_datasets/evaluate.py --help
python paper/scripts/paper_datasets/figure_2.py --help
python paper/scripts/paper_datasets/figure_3.py --help
python -m pytest paper/scripts/paper_datasets -q
```

Shared validation, inference, checkpoint sampling, and input-preparation tools
remain in [validation/scripts](../../validation/scripts/README.md).
The old `validation/scripts/evaluate_gfn_manuscript.py` entry point forwards
to the implementation here so saved report scripts can still import it.
Other moved manuscript commands should use their new paths above.

The correctness experiment remains in [paper/poc](../poc/README.md), and
Appendix E.1 scripts remain in `paper/validation/`.

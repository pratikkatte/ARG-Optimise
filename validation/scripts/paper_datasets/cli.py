"""Shared directory overrides for the paper evaluation commands."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[3]


def load_config(parser, argv=None):
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.yaml'))
    for method in ('argflow', 'singer', 'arginfer'):
        parser.add_argument(f'--{method}-dir', type=Path,
                            help=f'{method} output root containing r1, r2, and r4; paths are relative to the repository root.')
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text())
    roots = [args.argflow_dir, args.singer_dir, args.arginfer_dir]
    if any(p is not None for p in roots):
        if not all(p is not None for p in roots):
            parser.error('Supply --argflow-dir, --singer-dir, and --arginfer-dir together')
        argflow, singer, arginfer = [p if p.is_absolute() else ROOT / p for p in roots]
        for dataset, settings in config['datasets'].items():
            if not settings.get('enabled', True):
                continue
            manifests = sorted((argflow / dataset).glob('*/manifest.json'))
            if len(manifests) != 1:
                parser.error(f'Expected exactly one ARGFlow sample manifest under {argflow / dataset}, found {len(manifests)}')
            sample_dir = manifests[0].parent
            inputs = arginfer / 'inputs' / dataset
            settings['sources'] = [
                dict(method='ARGFlows', format='argflow', directory=str(sample_dir), expected_samples=1800),
                dict(method='SINGER', format='singer', directory=str(singer / dataset / 'trees'),
                     input_dir=str(inputs), expected_samples=1000, burnin_samples=0,
                     unknown_time_units='generations'),
                dict(method='ARGInfer', format='arginfer', directory=str(arginfer / dataset),
                     input_dir=str(inputs), expected_samples=1800, burnin_samples=0, expected_thin=1000),
            ]
            for figure in ('figure2', 'figure3'):
                config[figure]['main_argflows_checkpoints'][dataset] = sample_dir.name
        config['output_dir'] = 'paper/outputs/evaluation'
        config['figure2']['output_dir'] = 'paper/outputs/figure_2'
        config['figure3']['output_dir'] = 'paper/outputs/figure_3'
    return args, config

"""Compatibility entry point for the canonical paper infinite-sites simulator."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.scripts.simulate_infinite_sites import (
    DEFAULTS, load_config, write_vcf_and_position_map, simulate,
    main as _main,
)

DEFAULT_CONFIG = ROOT / 'validation' / 'config' / 'config.yaml'


def main(argv=None):
    return _main(argv, default_config=DEFAULT_CONFIG)


if __name__ == '__main__':
    main()

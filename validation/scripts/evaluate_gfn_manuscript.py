"""Compatibility entry point; implementation moved to paper/scripts."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if __name__ == '__main__':
    import runpy
    runpy.run_module('paper.scripts.evaluate_gfn_manuscript', run_name='__main__')
else:
    from paper.scripts import evaluate_gfn_manuscript as _implementation
    sys.modules[__name__] = _implementation

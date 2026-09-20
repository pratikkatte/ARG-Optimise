"""Atomic JSON reports, with lossless gzip storage for large evaluation details."""
import gzip
import json
from pathlib import Path
import tempfile


def write_json(path, value):
    """Stream a report to a temporary file and publish only a complete write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name+'.',
                                     suffix='.tmp', delete=False) as handle:
        temporary = Path(handle.name)
    try:
        compressed = path.suffix == '.gz'
        opener = gzip.open if compressed else open
        with opener(temporary, 'wt', encoding='utf-8',
                    **({'compresslevel': 6} if compressed else {})) as handle:
            json.dump(value, handle, allow_nan=False, default=lambda x: x.tolist(),
                      **({'separators': (',', ':')} if compressed else {'indent': 2}))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def open_json(path):
    """Open JSON text, also resolving an old .json path to its .json.gz archive."""
    path = Path(path)
    if not path.exists() and path.suffix == '.json':
        path = path.with_suffix('.json.gz')
    if path.suffix == '.gz':
        return gzip.open(path, 'rt', encoding='utf-8')
    return path.open('rt', encoding='utf-8')

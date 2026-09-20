"""Losslessly compress completed training reports; default to a read-only preview."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile


REPORT_NAME = re.compile(r'step_\d+_repeat_\d+\.json\Z')
CHUNK_SIZE = 4 * 1024 * 1024


def _signature(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _digest(stream):
    digest = hashlib.sha256()
    while chunk := stream.read(CHUNK_SIZE):
        digest.update(chunk)
    return digest.digest()


def _write_compressed(source, raw):
    digest = hashlib.sha256()
    with source.open('rb') as handle, gzip.GzipFile(
            filename='', mode='wb', fileobj=raw, compresslevel=6, mtime=0) as compressed:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
            compressed.write(chunk)
    raw.flush()
    os.fsync(raw.fileno())
    return digest.digest()


def bootstrap_report(path, staging_dir):
    """Make room using a verified archive on another filesystem, retaining recovery data on failure."""
    source = Path(path)
    if source.is_symlink() or not REPORT_NAME.fullmatch(source.name):
        raise ValueError(f'Not a completed evaluation report: {source}')
    before = _signature(source)
    original_stat = source.stat()
    target = source.with_suffix('.json.gz')
    if target.exists() or target.is_symlink():
        raise ValueError(f'Archive already exists; use normal verified conversion: {target}')
    staging_dir = Path(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=staging_dir, prefix=source.stem+'.',
                                     suffix='.json.gz', delete=False) as raw:
        staged = Path(raw.name)
        expected = _write_compressed(source, raw)
    with gzip.open(staged, 'rb') as handle:
        if _digest(handle) != expected:
            raise ValueError(f'Staged archive verification failed; original retained: {source}')
    if _signature(source) != before:
        raise ValueError(f'Report changed during compression; original retained: {source}')
    # Leave a durable recovery record before freeing space on the full filesystem.
    manifest = staged.with_suffix(staged.suffix+'.source.json')
    with manifest.open('x', encoding='utf-8') as handle:
        json.dump(dict(source=str(source.resolve()), target=str(target.resolve()),
                       archive=str(staged.resolve()), sha256=expected.hex()), handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary = None
    try:
        source.unlink()
        with tempfile.NamedTemporaryFile(dir=source.parent, prefix=target.name+'.',
                                         suffix='.tmp', delete=False) as raw:
            temporary = Path(raw.name)
            with staged.open('rb') as handle:
                shutil.copyfileobj(handle, raw, CHUNK_SIZE)
            raw.flush()
            os.fsync(raw.fileno())
        with gzip.open(temporary, 'rb') as handle:
            if _digest(handle) != expected:
                raise ValueError('Copied archive verification failed')
        os.chmod(temporary, original_stat.st_mode & 0o777)
        os.utime(temporary, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
        os.link(temporary, target)
    except Exception as exc:
        raise RuntimeError(f'Verified report retained at {staged}; recovery paths and SHA-256: {manifest}') from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    compressed_size = target.stat().st_size
    staged.unlink()
    manifest.unlink()
    return before[2], compressed_size


def compress_report(path):
    """Remove an original only after verifying its gzip copy byte for byte by SHA-256."""
    source = Path(path)
    if source.is_symlink() or not REPORT_NAME.fullmatch(source.name):
        raise ValueError(f'Not a completed evaluation report: {source}')
    before = _signature(source)
    target = source.with_suffix('.json.gz')
    temporary = None
    try:
        if target.exists() or target.is_symlink():
            if target.is_symlink():
                raise ValueError(f'Refusing a symlink archive: {target}')
            # Recover an interrupted conversion that published gzip before unlinking JSON.
            target_before = _signature(target)
            with source.open('rb') as handle:
                expected = _digest(handle)
            archive = target
        else:
            with tempfile.NamedTemporaryFile(dir=source.parent, prefix=target.name+'.',
                                             suffix='.tmp', delete=False) as raw:
                temporary = Path(raw.name)
                expected = _write_compressed(source, raw)
            archive = temporary
        with gzip.open(archive, 'rb') as handle:
            if _digest(handle) != expected:
                raise ValueError(f'Compressed content differs from original: {source}')
        if _signature(source) != before:
            raise ValueError(f'Report changed during compression; original retained: {source}')
        if temporary is not None:
            # Preserve permissions and timestamps; never overwrite another archive.
            original_stat = source.stat()
            os.chmod(temporary, original_stat.st_mode & 0o777)
            os.utime(temporary, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
            os.link(temporary, target)
        elif _signature(target) != target_before:
            raise ValueError(f'Archive changed during verification; original retained: {target}')
        compressed_size = target.stat().st_size
        source.unlink()
        return before[2], compressed_size
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, help='Run directory or parent containing multiple runs')
    parser.add_argument('--apply', action='store_true',
                        help='Replace completed JSON reports with verified gzip copies')
    parser.add_argument('--staging-dir', '--bootstrap-dir', dest='staging_dir', type=Path,
                        help='Stage each archive on another filesystem when the run filesystem is full')
    args = parser.parse_args(argv)
    if not args.root.is_dir():
        parser.error('root must be an existing directory')
    reports = sorted(p for p in args.root.rglob('step_*_repeat_*.json')
                     if p.parent.name == 'evaluation' and REPORT_NAME.fullmatch(p.name)
                     and p.is_file() and not p.is_symlink())
    total = sum(p.stat().st_size for p in reports)
    print(f'{len(reports)} completed evaluation reports: {total/1e9:.3f} GB', flush=True)
    if not args.apply:
        print('Preview only. Use --apply to compress and verify each report before removing its original.')
        return
    original_bytes = compressed_bytes = 0
    for index, report in enumerate(reports, 1):
        if args.staging_dir is not None:
            old, new = bootstrap_report(report, args.staging_dir)
        else:
            old, new = compress_report(report)
        original_bytes += old
        compressed_bytes += new
        if index % 10 == 0 or index == len(reports):
            print(f'Verified {index}/{len(reports)}; recovered '
                  f'{(original_bytes-compressed_bytes)/1e9:.3f} GB', flush=True)
    print(f'Completed: {original_bytes/1e9:.3f} GB -> {compressed_bytes/1e9:.3f} GB. '
          'Checkpoints, metric logs and unfinished temporary reports were not processed.', flush=True)


if __name__ == '__main__':
    main()

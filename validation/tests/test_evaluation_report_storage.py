"""Protect evaluation data during compressed writes and legacy conversion."""
import gzip
import json
from unittest.mock import patch

import numpy as np
import pytest

from training.reporting import open_json, write_json
from validation.scripts.compress_evaluation_reports import bootstrap_report, compress_report


def test_streamed_report_preserves_arrays_and_legacy_read_path(tmp_path):
    path = tmp_path/'step_000050_repeat_00.json.gz'
    value = dict(metrics={'loss': 1.25}, details={'array': np.arange(24).reshape(4, 6)})
    write_json(path, value)
    with open_json(path.with_suffix('')) as handle:
        restored = json.load(handle)
    assert restored['metrics'] == value['metrics']
    np.testing.assert_array_equal(restored['details']['array'], value['details']['array'])
    legacy = tmp_path/'legacy.json'
    write_json(legacy, restored)
    with open_json(legacy) as handle:
        assert json.load(handle) == restored


@pytest.mark.parametrize('suffix', ['.json', '.json.gz'])
def test_failed_report_write_preserves_previous_report(tmp_path, suffix):
    path = tmp_path/('report'+suffix)
    write_json(path, {'loss': 1.0})
    original = path.read_bytes()
    with pytest.raises(ValueError):
        write_json(path, {'loss': float('nan')})
    assert path.read_bytes() == original
    assert not list(tmp_path.glob('*.tmp'))


def test_legacy_conversion_is_byte_exact(tmp_path):
    path = tmp_path/'step_000050_repeat_00.json'
    original = json.dumps({'details': ['\u03bb=2', 1.2345678901234567]*100}, indent=2).encode()
    path.write_bytes(original)
    before, after = compress_report(path)
    archive = path.with_suffix('.json.gz')
    assert before == len(original) and after == archive.stat().st_size
    assert not path.exists() and not list(tmp_path.glob('*.tmp'))
    with gzip.open(archive, 'rb') as handle:
        assert handle.read() == original


@pytest.mark.parametrize('matches', [True, False])
def test_existing_archive_must_match_before_original_is_removed(tmp_path, matches):
    path = tmp_path/'step_000050_repeat_00.json'
    path.write_bytes(b'{"loss": 1.0}')
    archive = path.with_suffix('.json.gz')
    with gzip.open(archive, 'wb') as handle:
        handle.write(path.read_bytes() if matches else b'{"loss": 2.0}')
    original_archive = archive.read_bytes()
    if matches:
        compress_report(path)
        assert not path.exists()
    else:
        with pytest.raises(ValueError, match='differs'):
            compress_report(path)
        assert path.read_bytes() == b'{"loss": 1.0}'
    assert archive.read_bytes() == original_archive


def test_verification_failure_keeps_original_and_removes_temporary(tmp_path):
    path = tmp_path/'step_000050_repeat_00.json'
    path.write_bytes(b'{"loss": 1.0}')
    with patch('validation.scripts.compress_evaluation_reports.gzip.open', side_effect=OSError('read failed')):
        with pytest.raises(OSError, match='read failed'):
            compress_report(path)
    assert path.read_bytes() == b'{"loss": 1.0}'
    assert list(tmp_path.iterdir()) == [path]


def test_changed_report_is_not_removed(tmp_path):
    from validation.scripts.compress_evaluation_reports import _digest
    path = tmp_path/'step_000050_repeat_00.json'
    path.write_bytes(b'{"loss": 1.0}')
    def concurrent_change(handle):
        digest = _digest(handle)
        path.write_bytes(b'{"loss": 2.0}')
        return digest
    with patch('validation.scripts.compress_evaluation_reports._digest', side_effect=concurrent_change):
        with pytest.raises(ValueError, match='changed during compression'):
            compress_report(path)
    assert path.read_bytes() == b'{"loss": 2.0}'
    assert list(tmp_path.iterdir()) == [path]


def test_bootstrap_restores_archive_to_original_directory(tmp_path):
    path = tmp_path/'step_000050_repeat_00.json'
    path.write_bytes(b'{"loss": 1.0}')
    staging = tmp_path/'staging'
    bootstrap_report(path, staging)
    assert not path.exists() and list(staging.iterdir()) == []
    with open_json(path) as handle:
        assert json.load(handle) == {'loss': 1.0}


def test_bootstrap_retains_verified_recovery_copy_if_copy_back_fails(tmp_path):
    path = tmp_path/'step_000050_repeat_00.json'
    original = b'{"loss": 1.0}'
    path.write_bytes(original)
    staging = tmp_path/'staging'
    with patch('validation.scripts.compress_evaluation_reports.shutil.copyfileobj',
               side_effect=OSError('disk quota exceeded')):
        with pytest.raises(RuntimeError, match='Verified report retained'):
            bootstrap_report(path, staging)
    manifest = json.loads(next(staging.glob('*.source.json')).read_text())
    assert manifest['source'] == str(path)
    with gzip.open(manifest['archive'], 'rb') as handle:
        assert handle.read() == original
    assert not path.with_suffix('.json.gz').exists()
    assert not list(tmp_path.glob('*.tmp'))

import datetime as dt
import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/environment/configured_progress_status_archive.py'
spec = importlib.util.spec_from_file_location('progress_archive', SCRIPT)
archive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive)


def test_preserves_observation_and_archive_times_and_bytes_hash(tmp_path):
    path = tmp_path / 'status.json'
    path.write_text(json.dumps({'observed_at_utc': '2026-11-07T00:03:20Z', 'lanes': []}))
    result = archive.capture({'lane': str(path)}, now=dt.datetime(2026, 11, 7, 0, 8, tzinfo=dt.timezone.utc))
    assert result['archived_at_utc'] == '2026-11-07T00:08:00+00:00'
    assert result['sources']['lane']['observed_at_utc'] == '2026-11-07T00:03:20Z'
    assert result['sources']['lane']['document']['lanes'] == []
    assert len(result['sources']['lane']['sha256']) == 64


def test_missing_and_partial_are_unavailable_no_fabricated_document(tmp_path):
    partial = tmp_path / 'partial.json'
    partial.write_text('{"observed_at_utc":')
    result = archive.capture({'missing': str(tmp_path / 'missing'), 'partial': str(partial)})
    for name in ('missing', 'partial'):
        assert result['sources'][name]['state'] == 'unavailable'
        assert 'document' not in result['sources'][name]
        assert 'sha256' not in result['sources'][name]

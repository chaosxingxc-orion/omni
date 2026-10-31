import hashlib
import importlib.util
import json
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('historical_receipts', ROOT / 'scripts/analysis/restore_historical_receipt_addresses.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def test_verify_every_byte_and_reject_tamper_before_restore(tmp_path):
    d = tmp_path / 'old'; d.mkdir()
    (d / 'session-receipt.json').write_bytes(b'receipt')
    (d / 'session.json').write_bytes(b'marker')
    doc = {'records': {'old': {'files': {n: hashlib.sha256((d/n).read_bytes()).hexdigest()
        for n in ('session.json','session-receipt.json')}}}}
    (tmp_path/'manifest.json').write_text(json.dumps(doc))
    assert m.verified_receipts(tmp_path) == [('old', b'receipt')]
    (d/'session.json').write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'): m.verified_receipts(tmp_path)


def test_escaping_session_refused(tmp_path):
    (tmp_path/'manifest.json').write_text(json.dumps({'records': {'../other': {'files': {}}}}))
    with pytest.raises(ValueError, match='unsafe'): m.verified_receipts(tmp_path)

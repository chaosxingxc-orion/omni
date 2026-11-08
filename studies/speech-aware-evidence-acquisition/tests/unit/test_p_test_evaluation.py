import importlib.util
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import pytest

path = Path(__file__).resolve().parents[2] / 'scripts/analysis/p_test_evaluation.py'
spec = importlib.util.spec_from_file_location('p_test_evaluation', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_denominators_and_failures():
    roster = {'items': [{'sample_id': x} for x in 'abcd']}
    rows = [
        {'sample_id': 'a', 'gold': 'X', 'supply': ['X', 'Y'], 'output_indices': [1, 2], 'status': 'ok'},
        {'sample_id': 'b', 'gold': 'X', 'supply': ['Y'], 'output_indices': [1], 'status': 'ok'},
        {'sample_id': 'c', 'gold': 'X', 'supply': ['X'], 'output_indices': [], 'status': 'failed'},
        {'sample_id': 'd', 'gold': 'X', 'supply': ['X', 'Y', 'Z', 'W'], 'output_indices': [2, 3, 4], 'status': 'ok'},
    ]
    result = m.evaluate(roster, rows, [])
    assert result['supply'] == {'correct': 3, 'total': 4}
    assert result['p_top3'] == {'correct': 1, 'total': 4}
    assert result['conditional_retention'] == {'correct': 1, 'total': 3}
    assert result['valid_output']['valid'] == 3
    assert result['confirmed_exceptions'] == 0 and not result['acceptance']
    exc = [{'sample_id': 'd', 'status': 'confirmed', 'missing_information': 'missing', 'checks': 'checked', 'rationale': 'evidence', 'downstream_evidence': 'unknown', 'downstream_recovered': None}]
    result = m.evaluate(roster, rows, exc)
    assert result['nonexception_top3'] == {'correct': 1, 'total': 3}
    assert result['downstream_recovery']['assessed'] == 0
    with pytest.raises(ValueError):
        m.evaluate(roster, rows[:-1], [])
    with pytest.raises(ValueError):
        m.evaluate(roster, rows, [{'sample_id': 'd', 'status': 'pending'}])


def archive_row(sid='a'):
    return {'kind': 'row', 'sample_id': sid, 'complete': True, 'output_rows': [{
        'sample_id': sid, 'text': '[1, 2, 3]', 'node_outcome': {'ranking_result': {
            'status': 'ok', 'input': [{'index': i, 'intent_id': str(i)} for i in range(1, 5)],
            'indices': [1, 2, 3], 'candidate_ids': ['1', '2', '3']}}}]}


def test_archive_mapping_and_invariants():
    roster = {'sample_ids': ['a']}
    gold = [{'sample_id': 'a', 'gold_intent': '1'}]
    header = {'kind': 'header'}
    row = archive_row()
    encode = lambda *items: '\n'.join(json.dumps(x) for x in items)
    mapped = m.map_checkpoint(roster, encode(header, row), gold)
    assert m.evaluate(roster, mapped, [])['acceptance']
    row['output_rows'][0]['text'] = '[2, 1, 3]'
    assert m.evaluate(roster, m.map_checkpoint(roster, encode(header, row), gold), [])['valid_output']['valid'] == 0
    with pytest.raises(ValueError):
        m.map_checkpoint(roster, encode(header), gold)
    with pytest.raises(ValueError):
        m.map_checkpoint(roster, encode(header, row, row), gold)
    with pytest.raises(ValueError):
        m.map_checkpoint(roster, encode(header, row), gold + gold)
    row['output_rows'][0]['node_outcome']['ranking_result']['input'][0]['index'] = 2
    with pytest.raises(ValueError):
        m.map_checkpoint(roster, encode(header, row), gold)


def test_all_exceptions_do_not_imply_acceptance():
    roster = {'sample_ids': ['a']}
    rows = m.map_checkpoint(roster, '\n'.join(map(json.dumps, [{'kind': 'header'}, archive_row()])), [{'sample_id': 'a', 'gold_intent': '1'}])
    exc = [{'sample_id': 'a', 'status': 'confirmed', 'missing_information': 'x', 'checks': 'x', 'rationale': 'x', 'downstream_evidence': 'x'}]
    result = m.evaluate(roster, rows, exc)
    assert result['nonexception_top3']['total'] == 0 and result['acceptance'] is False


def test_cli_writes_parseable_json(tmp_path):
    def write(name, obj):
        p = tmp_path / name
        p.write_text(json.dumps(obj))
        return p
    roster = write('roster.json', {'sample_ids': ['a']})
    gold = write('gold.json', [{'sample_id': 'a', 'gold_intent': '1'}])
    exc = write('exceptions.json', [])
    ckpt = tmp_path / 'checkpoint.jsonl'
    ckpt.write_text('\n'.join(map(json.dumps, [{'kind': 'header'}, archive_row()])) + '\n')
    out, rows = tmp_path / 'result.json', tmp_path / 'rows.json'
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    subprocess.run([sys.executable, str(path), '--roster', str(roster), '--roster-sha256', sha(roster),
                    '--expected-count', '1', '--checkpoint', str(ckpt), '--checkpoint-sha256', sha(ckpt),
                    '--gold', str(gold), '--gold-sha256', sha(gold), '--exceptions', str(exc),
                    '--normalized-out', str(rows), '--out', str(out)], check=True)
    with out.open() as f:
        assert json.load(f)['acceptance'] is True
    with rows.open() as f:
        assert len(json.load(f)) == 1

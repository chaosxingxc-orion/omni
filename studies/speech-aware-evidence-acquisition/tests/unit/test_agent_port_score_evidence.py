"""A branch nomination cannot stand in for an Agent's final delivered result."""
import json
from pathlib import Path
import sys

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts/analysis'))
import agent_port_score_evidence as read
from test_direct_port_score_evidence import fixture


@pytest.mark.parametrize('case', ['valid', 'first-visit', 'scoring', 'wrong-id', 'duplicate', 'tamper'])
def test_registered_final_surface_required(tmp_path, case):
    recipe = {'schema': read.SCHEMA, 'read_run_id': 'READ', 'quality_read': True,
              'new_scoring_jobs': 0, 'score_surface': 'sealed-final-delivery', 'experiments': ['EXP']}
    if case == 'first-visit': recipe['score_surface'] = 'first-solver-level2'
    if case == 'scoring': recipe['new_scoring_jobs'] = 1
    if case == 'wrong-id': recipe['read_run_id'] = 'FOREIGN'
    path = tmp_path / 'docs/recipe.json'; path.parent.mkdir(); path.write_text(json.dumps(recipe))
    line = f'| today | READ | docs/recipe.json | root countersigned; Plan SHA256 {read.shared.sha(path)} |\n'
    (tmp_path / 'docs/exposure-ledger.md').write_text(
        '| date | run/experiment id | artifacts | notes |\n| --- | --- | --- | --- |\n'
        + line * (2 if case == 'duplicate' else 1))
    if case == 'tamper': path.write_text(path.read_text() + ' ')
    if case == 'valid':
        assert read.require_read(tmp_path, path, 'READ', 'EXP')['score_surface'] == 'sealed-final-delivery'
    else:
        with pytest.raises(ValueError): read.require_read(tmp_path, path, 'READ', 'EXP')


def test_delivered_reader_uses_same_complete_score_custody(tmp_path):
    bundle, receipts, ids, _, _ = fixture(tmp_path)
    # The shared finalizer receipt is the authority; derived column files are not inputs.
    (tmp_path / 'arm-lock-slurp-slu-official-train-scores.json').write_text('{"not":"the result"}')
    result = read.shared.verify_saved_score(bundle, receipts, tmp_path, ids)
    assert result['aggregate']['slu_f1'] == 0.5 and result['new_scoring_jobs'] == 0
    assert not any('arm-lock' in path for path in result['inputs'])

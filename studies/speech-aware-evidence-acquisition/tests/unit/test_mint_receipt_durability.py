"""Receipt publication refuses a racing writer without truncating its decision."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts/flights/direct'))
import thinking_system_mint_receipt as mint
from speech_aware_evidence_acquisition.core import atomic_file


def setup_draft(tmp_path):
    family = SimpleNamespace(PACKAGE='package', DECISION_ID='decision', DECISION_RELPATH='receipt.json', eligible_run_ids=lambda: ['run'])
    (tmp_path/'repo/package').mkdir(parents=True)
    (tmp_path/'repo/package/root-admission-draft-v1.json').write_text(json.dumps({'decision_id':'decision','eligible_run_ids':['run']}))
    return family, str(tmp_path/'repo'), str(tmp_path/'data')


def test_mint_publishes_complete_receipt_and_refuses_overwrite(tmp_path):
    family, repo, data = setup_draft(tmp_path)
    mint.mint(repo, data, 'Codex', family=family)
    p=Path(data)/'receipt.json'
    before=p.read_bytes()
    assert json.loads(before)['status']=='decided'
    with pytest.raises(SystemExit, match='refusing to overwrite'):
        mint.mint(repo, data, 'Codex', family=family)
    assert p.read_bytes()==before


def test_racing_mint_cannot_replace_first_decision(tmp_path, monkeypatch):
    family, repo, data = setup_draft(tmp_path)
    original=atomic_file.publish_new
    def race(path, value):
        original(path,b'first decision')
        original(path,value)
    monkeypatch.setattr(atomic_file,'publish_new',race)
    with pytest.raises(SystemExit, match='refusing to overwrite'):
        mint.mint(repo,data,'Codex',family=family)
    assert (Path(data)/'receipt.json').read_bytes()==b'first decision'

import json
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
sys.path.insert(0,str(Path(__file__).parent))
import fold_a_c2_observer as m


def test_missing_container_does_not_declare_failure_or_restart(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'D',tmp_path)
    monkeypatch.setattr(m.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=1,stdout=''))
    monkeypatch.setattr(m,'hardware',lambda _: {'cards':{},'errors':['unavailable']})
    sleeps=[]
    def sleep(seconds):
        sleeps.append(seconds);raise KeyboardInterrupt
    monkeypatch.setattr(m.time,'sleep',sleep)
    out=tmp_path/'out'
    with pytest.raises(KeyboardInterrupt):m.observe(tmp_path/'tree',tmp_path/'run',out,tmp_path/'exit','owned')
    doc=json.loads((out/'status.json').read_text())
    assert doc['state']=='unconfirmed_process_state' and doc['total_rows']==1080
    assert doc['estimated_remaining_seconds'] is None and sleeps==[600]
    assert json.loads((out/'intent.json').read_text())['can_restart_jobs'] is False


def test_explicit_failed_exit_is_terminal(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'D',tmp_path)
    monkeypatch.setattr(m.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=1,stdout=''))
    monkeypatch.setattr(m,'hardware',lambda _: {'cards':{},'errors':[]})
    monkeypatch.setattr(m.time,'sleep',lambda _:pytest.fail('terminal must not sleep'))
    exit_path=tmp_path/'exit';exit_path.write_text('1\n');out=tmp_path/'out'
    m.observe(tmp_path/'tree',tmp_path/'run',out,exit_path,'owned')
    doc=json.loads((out/'status.json').read_text())
    assert doc['state']=='failed' and doc['campaign_exit']=='1'

import importlib.util
import json
from pathlib import Path
from unittest.mock import patch
import pytest
spec=importlib.util.spec_from_file_location('queue500',Path(__file__).with_name('real500_successor_queue.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def setup(tmp):
    (tmp/'locks').mkdir();(tmp/'locks/generation.guard').touch()
    spec={'run_id':'run','attempt_id':'attempt'}
    d=tmp/'runs/run/attempt';d.mkdir(parents=True)
    (d/'run.generation-manifest.json').write_text(json.dumps({'run_id':'run','status':'COMPLETED'}))
    (d/'score-jobs.jsonl').write_text('{}\n')
    root=tmp/'queue';root.mkdir()
    return spec,d,root,tmp/'run.exit'


def test_generation_seal_does_not_wait_for_cpu_scoring(tmp_path):
    spec,d,root,terminal=setup(tmp_path)
    with patch.object(m,'D',tmp_path):p=m.wait_seal(spec,terminal,root)
    assert p==d/'run.generation-manifest.json'
    assert json.loads((root/'predecessor-seal.json').read_bytes())['cpu_scoring_may_continue']
    assert not terminal.exists()


def test_failed_generation_never_advances(tmp_path):
    spec,d,root,terminal=setup(tmp_path);terminal.write_text('1\n')
    with patch.object(m,'D',tmp_path),pytest.raises(ValueError,match='predecessor failed'):m.wait_seal(spec,terminal,root)
    assert not (root/'predecessor-seal.json').exists()


def test_live_lease_never_advances(tmp_path):
    spec,d,root,terminal=setup(tmp_path);(tmp_path/'locks/flight.lock').touch()
    with patch.object(m,'D',tmp_path),patch.object(m.time,'sleep',side_effect=RuntimeError('waiting')),pytest.raises(RuntimeError,match='waiting'):m.wait_seal(spec,terminal,root)
    assert not (root/'predecessor-seal.json').exists()

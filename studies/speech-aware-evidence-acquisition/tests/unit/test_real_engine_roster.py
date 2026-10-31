"""Real workload selection is stable and independent of runtime completion."""
import importlib.util
import json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('real_roster',ROOT/'scripts/analysis/prepare_real_engine_roster.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
SOURCE=ROOT/'configs/dataset/slurp-speech-native-evo-fold-b-1080.json'

def test_unique_stable_subset_and_parent_custody():
    receipt,view=m.prepare(SOURCE);again,_=m.prepare(SOURCE)
    parent=json.loads(SOURCE.read_bytes());by_id={r['sample_id']:r for r in parent['items']}
    assert receipt==again and len(view['items'])==500
    assert len({r['sample_id'] for r in view['items']})==500
    assert all(r==by_id[r['sample_id']] for r in view['items'])
    assert view['split_identity_hash']!=parent['split_identity_hash']
    assert view['runtime_view_sha256']==m.digest({k:v for k,v in view.items() if k!='runtime_view_sha256'})
    other,_=m.prepare(SOURCE,seed='another-predeclared-seed')
    assert other['items']!=receipt['items']

def test_tampered_parent_is_refused(tmp_path):
    doc=json.loads(SOURCE.read_bytes());doc['items'][0]['audio_seconds']+=1
    p=tmp_path/'source.json';p.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='hash'):m.prepare(p)

import copy,json,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/flights/nx'))
import r01c_cpu_overlap as m


def test_derivation_preserves_encoder_and_changes_only_operational_fields(tmp_path,monkeypatch):
    encoder={'workers':4,'threads_per_worker':16,'first_core':128,'runtime':'cpu-float32','attention_geometry':'causal','files':{'weights':'hash'}}
    inv=tmp_path/'encoder.json';inv.write_text(json.dumps(encoder))
    base={'encoder':{'path':str(inv),'sha256':m.sha(inv)},'sources':{'path':'source','sha256':'a'*64},'monitor_interval_seconds':10,'query_prompt':'query: ','dimension':2048}
    entry={'execution_plan':{'run_id':'old','planned_model_calls':625,'planned_speech_audio_seconds':2727,'planned_gpu_hours':2},'ledger_row_cells':['x']*15}
    monkeypatch.setattr(m.draft,'derive',lambda *a:{m.draft.PROTOCOL:json.dumps(base),m.draft.REGISTRATION:json.dumps({'plans':[entry]})})
    monkeypatch.setattr(m,'sha',lambda p:'a'*64)
    new,protocol,doc=m.derive(tmp_path,tmp_path,tmp_path/'lock')
    assert {**new,'first_core':128}==encoder
    assert new['first_core']==192 and protocol['monitor_interval_seconds']==600
    assert protocol['cpu_execution']['cpuset']=='192-255'
    assert protocol['query_prompt']==base['query_prompt'] and protocol['dimension']==2048
    assert len(protocol['cpu_execution']['serving_profiles'])==4
    assert doc['plans'][0]['execution_plan']['planned_model_calls']==625
    assert protocol['cache_relpath']==f'prepared/{m.RUN}/span-cache'
    assert doc['downstream_adopted'] is False

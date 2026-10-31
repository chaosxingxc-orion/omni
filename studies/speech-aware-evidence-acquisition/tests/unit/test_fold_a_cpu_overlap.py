"""Prospective CPU overlap registration keeps encoder method and own custody."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'scripts/flights/nx'))
import fold_a_cpu_overlap as c
from test_fold_a_encoding_admission import inputs


def commit(repo):
    subprocess.run(['git','add','.'],cwd=repo,check=True)
    subprocess.run(['git','commit','-qm','fixture'],cwd=repo,check=True)


def init(repo):
    subprocess.run(['git','init','-q'],cwd=repo,check=True)
    for key,value in [('user.name','Test'),('user.email','test@example.invalid')]:
        subprocess.run(['git','config',key,value],cwd=repo,check=True)
    commit(repo)


def test_derived_pool_preserves_all_model_and_math_settings(inputs):
    repo,data,lock=inputs
    encoder,protocol,doc=c.derive(*inputs)
    original=c.read(repo/c.prior.draft.FOLD_A_FRAGMENT)
    old_encoder=c.read(Path(original['encoder']['path'].replace('${SAEA_DATA_ROOT}',str(data))))
    for key in set(encoder)-{'workers','first_core'}:assert encoder[key]==old_encoder[key]
    assert encoder['workers']==2 and encoder['threads_per_worker']==16 and encoder['first_core']==32
    moved={'encoder','cpu_workers','output_relpath','query_rows_relpath','cache_relpath','cpu_isolation','contact_policy'}
    for key in set(original)-moved:assert protocol[key]==original[key]
    assert doc['plans'][0]['execution_plan']['planned_model_calls']==1080
    assert doc['plans'][0]['execution_plan']['planned_speech_audio_seconds']==3085


def test_registration_passes_real_tool_plan_gate_without_contacts(inputs):
    repo,data,lock=inputs;init(repo)
    result=c.register(*inputs)
    assert len(result['files'])==3 and result['encoder_contacts']==0
    commit(repo)
    cfg=c.runtime_config(*inputs)
    assert cfg['plan']['run_id']==c.RUN and cfg['plan']['planned_gpu_hours']==2
    assert not (data/'attempts').exists()
    with pytest.raises(ValueError,match='already exists'):c.register(*inputs)


@pytest.mark.parametrize('fault',['inventory','protocol','registration','spent'])
def test_config_rejects_drift_or_spent_identity(inputs,fault):
    repo,data,lock=inputs;init(repo);c.register(*inputs);commit(repo)
    path={'inventory':data/c.INVENTORY,'protocol':repo/c.PROTOCOL,'registration':repo/c.REGISTRATION}.get(fault)
    if path:
        d=c.read(path);d['tampered']=True;path.write_bytes(c.raw(d))
        if fault!='inventory':commit(repo)
    else:(data/'attempts'/c.RUN).mkdir(parents=True)
    with pytest.raises(ValueError):c.runtime_config(*inputs)

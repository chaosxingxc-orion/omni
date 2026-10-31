"""Isolated CPU custody coexists with generation without changing its lease."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from speech_aware_evidence_acquisition.core import speech_preparation_runtime as runtime
from speech_aware_evidence_acquisition.core.flight.lease import GenerationLease, GenerationLeaseError
from test_cpu_encoder_pool import _inventory
from test_speech_preparation_runtime import job, Backend, pin


def protocol(run='TEST-ENCODE'):
    return {'cpu_isolation':{'schema':'saea-cpu-preparation-isolation-v1','cpu_ids':[32,33], 'generation_overlap':True},
        'output_relpath':f'prepared/{run}/index.json','query_rows_relpath':f'prepared/{run}/queries.jsonl',
        'cache_relpath':f'prepared/{run}/span-cache'}


@pytest.mark.parametrize('fault',[None,'schema','affinity','gpu','backend','cache','declaration','modality'])
def test_only_exact_cpu_protocol_can_select_separate_custody(tmp_path,monkeypatch,fault):
    enc=_inventory(workers=2,threads_per_worker=1,first_core=32);p=protocol()
    monkeypatch.setattr(runtime.os,'sched_getaffinity',lambda _: {32,33} if fault!='affinity' else {32,33,34})
    backend=runtime._cpu_speech_backend;kind='speech'
    if fault=='schema':enc['schema']='other'
    if fault=='backend':backend=runtime.NemotronSpeechEncoder
    if fault=='cache':p['cache_relpath']='prepared/shared-cache'
    if fault=='declaration':p['cpu_isolation']['generation_overlap']=False
    if fault=='modality':kind='text'
    original=Path.glob
    if fault=='gpu':monkeypatch.setattr(Path,'glob',lambda self,pattern:iter([Path('/dev/davinci0')]) if self==Path('/dev') else original(self,pattern))
    args=(p,enc,SimpleNamespace(run_id='TEST-ENCODE'),tmp_path,backend,kind)
    if fault:
        with pytest.raises(runtime.SpeechPreparationError):runtime.preparation_lease_root(*args)
    else:
        assert runtime.preparation_lease_root(*args)==tmp_path/'coordination/cpu-preparation-custody'
    assert runtime.preparation_lease_root({},enc,args[2],tmp_path,backend,kind)==tmp_path


@pytest.mark.parametrize('fail',[False,True])
def test_real_preparation_preserves_live_generation_lease(job,monkeypatch,fail):
    root,data,config,r=job
    p=r.pinned_json(config['protocol']);p.update(protocol())
    encoder=Path(p['encoder']['path']);encoder.write_text(json.dumps(_inventory(first_core=32,workers=2,threads_per_worker=1)))
    p['encoder']=pin(encoder)
    pp=Path(config['protocol']['path']);pp.write_text(json.dumps(p))
    config['protocol']=pin(pp);config['plan']['protocol_hash']=pin(pp)['sha256']
    monkeypatch.setattr(r.os,'sched_getaffinity',lambda _:{32,33})
    monkeypatch.setattr(r,'_verify_encoder_files',lambda *a:None)
    isolated=data/'coordination/cpu-preparation-custody'
    with GenerationLease.acquire(data,owner='generation',run_id='A0',attempt_id='a',session_id='serving',study_head='a'*40):
        before=(data/'locks/flight.lock').read_bytes()
        class Checking(Backend):
            def __call__(self,source):
                assert (data/'locks/flight.lock').read_bytes()==before
                assert (isolated/'locks/flight.lock').is_file()
                with pytest.raises(GenerationLeaseError):
                    GenerationLease.acquire(isolated,owner='other',run_id='other',attempt_id='a',session_id='s',study_head='b'*40)
                if fail:raise RuntimeError('worker failure')
                return super().__call__(source)
        backend=Checking()
        monkeypatch.setattr(r,'_cpu_speech_backend',lambda *a:backend)
        if fail:
            with pytest.raises(RuntimeError,match='worker failure'):r.run_preparation(config,source_root=root,data_root=data)
            assert (isolated/'locks/flight.lock').is_file()
        else:
            result=r.run_preparation(config,source_root=root,data_root=data)
            assert result['status']=='SEALED' and result['cpu_isolation']==p['cpu_isolation']
            assert not (isolated/'locks/flight.lock').exists()
        assert backend.closed
        assert (data/'locks/flight.lock').read_bytes()==before

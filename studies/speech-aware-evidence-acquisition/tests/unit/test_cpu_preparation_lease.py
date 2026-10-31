"""CPU preparation can overlap generation but cannot overlap its own partition."""
import hashlib
from pathlib import Path
from types import SimpleNamespace
import pytest
from speech_aware_evidence_acquisition.core import cpu_preparation_lease as m
from speech_aware_evidence_acquisition.core.flight.lease import GenerationLease, GenerationLeaseError


@pytest.fixture
def partition(tmp_path, monkeypatch):
    profiles={}
    for topology,seq in [('2xtp4',96),('2xtp4',128),('1xtp8',256),('1xtp8',384)]:
        name=f'qwen3-omni-real500-{topology}-s{seq}-b2048-v1'
        p=tmp_path/'configs/deployment'/(name+'.json');p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}')
        profiles[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    monkeypatch.setattr(m,'load_deployment_profile',lambda *a:SimpleNamespace(host_cpu_count=256,instances=[SimpleNamespace(cpuset='64-95'),SimpleNamespace(cpuset='128-159')]))
    encoder={'schema':'saea-cpu-encoder-inventory-v1','runtime':'cpu-float32','attention_geometry':'causal',
        'first_core':192,'workers':4,'threads_per_worker':16,'python_executable':'/usr/bin/python3',
        'packages':{'torch':'2.10.0+cpu','transformers':'5.14.1'},'files':{},'model_dir_relpath':'model'}
    protocol={'monitor_interval_seconds':600,'cpu_execution':{'schema':'saea-isolated-cpu-preparation-v1',
        'cpuset':'192-255','driver_cpuset':'32-63','serving_profiles':profiles}}
    devices=tmp_path/'dev';devices.mkdir()
    return tmp_path,encoder,protocol,devices


def test_cpu_partition_checks_affinity_profiles_and_device_isolation(partition):
    root,encoder,protocol,devices=partition
    assert len(m.validate_cpu_partition(encoder,protocol,root,affinity=range(192,256),device_root=devices))==64
    with pytest.raises(ValueError,match='affinity'):
        m.validate_cpu_partition(encoder,protocol,root,affinity=range(256),device_root=devices)
    (devices/'davinci0').touch()
    with pytest.raises(ValueError,match='NPU devices'):
        m.validate_cpu_partition(encoder,protocol,root,affinity=range(192,256),device_root=devices)


def test_cpu_partition_rejects_overlap_changed_profile_and_non_cpu_runtime(partition):
    root,encoder,protocol,devices=partition
    args=dict(affinity=range(192,256),device_root=devices)
    protocol['cpu_execution']['driver_cpuset']='192-223'
    with pytest.raises(ValueError,match='overlaps'):
        m.validate_cpu_partition(encoder,protocol,root,**args)
    protocol['cpu_execution']['driver_cpuset']='32-63'
    name=next(iter(protocol['cpu_execution']['serving_profiles']))
    protocol['cpu_execution']['serving_profiles'][name]='0'*64
    with pytest.raises(ValueError,match='hash changed'):
        m.validate_cpu_partition(encoder,protocol,root,**args)
    encoder['packages']['torch']='2.10.0'
    with pytest.raises(ValueError,match='CPU-only'):
        m.validate_cpu_partition(encoder,protocol,root,**args)



def test_cpu_monitor_does_not_query_accelerator(tmp_path,monkeypatch):
    import json,threading,time
    from speech_aware_evidence_acquisition.core import speech_preparation_runtime as runtime
    from speech_aware_evidence_acquisition.core.session import serving,telemetry
    event=threading.Event()
    def forbidden():raise AssertionError('CPU monitor queried accelerator')
    def memory():event.set();return {'available_gb':20}
    monkeypatch.setattr(telemetry,'read_gpu',forbidden)
    monkeypatch.setattr(serving,'read_host_memory',memory)
    path=tmp_path/'monitor.jsonl'
    close=runtime.start_monitor(path,{'cpu_only':True,'started_monotonic':time.monotonic(),'calls':0,'phase':'admitted'},600)
    assert event.wait(2);close()
    assert json.loads(path.read_text().splitlines()[0])['gpu'] is None

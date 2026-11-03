"""Guarded engine lifecycle tests use real profile/lease logic and a Docker double."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from test_configured_terminal_settlement import configured, write, pin as source_pin, seal
from test_deployment import IDLE

spec=importlib.util.spec_from_file_location('engine_transition',Path(__file__).parents[2]/'scripts/environment/configured_engine_transition.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def pin(path):return {'path':str(path),'sha256':m.launch.digest(path)}


def container(profile,model,cid,pid):
    return {'Id':cid,'Name':'/'+profile.instances[0].name,'Image':profile.image,'RestartCount':0,
        'Config':{'Labels':{m.deployment.LABEL_PROFILE:profile.profile_id,
            m.deployment.LABEL_PROFILE_SHA256:profile.sha256,m.deployment.LABEL_INSTANCE:'0'},
            'Entrypoint':[m.deployment.CONTAINER_INTERPRETER],
            'Cmd':m.deployment.instance_container_args(profile,profile.instances[0],model)},
        'HostConfig':{'PidMode':'host','NetworkMode':'host','RestartPolicy':{'Name':'no'},
            'CpusetCpus':profile.instances[0].cpuset,'Devices':[{'PathOnHost':d} for d in
            [*m.deployment.ASCEND_CONTROL_DEVICES,*(f'/dev/davinci{v}' for v in profile.cards)]]},
        'State':{'Status':'running','Running':True,'Pid':pid,'ExitCode':0,
                 'StartedAt':'2026-11-03T10:00:00.123456789Z','FinishedAt':'0001-01-01T00:00:00Z'}}


@pytest.fixture
def engine(configured,monkeypatch):
    f=configured;logical=f.logical;data=f.data;root=f.tmp_path
    # Source configuration loader is already exercised against real mapped custody
    # in test_configured_terminal_settlement; inject its resolved context only.
    source=copy.deepcopy(f.config);source['generation_release']['logical_data']=str(data)
    producer=copy.deepcopy(f.launch_doc)
    for mt in producer['mounts']:
        if mt['target'].startswith(str(logical)):mt['target']=mt['target'].replace(str(logical),str(data),1)
    monkeypatch.setattr(m.settlement,'load',lambda p:(source,producer,f.output/'generation-released.json',data,f.expected,root/'source-settlement'))
    profile={'schema':'saea-deployment-profile-v1','profile_id':'old-v1','host':'test-host',
        'image':'sha256:'+'a'*64,'model_fragment':'old-model','model_relpath':'models/native',
        'bind_host':'127.0.0.1','data_mount':str(data),'tensor_parallel_size':2,'max_num_seqs':48,
        'gpu_memory_utilization':.92,'max_num_batched_tokens':2048,'busy_threshold_mib':4096,
        'host_card_count':8,'host_cpu_count':256,'shm_size':'16g','namespace_policy':'shared-pid-host',
        'instances':[{'name':'saea-old-a','cards':[4,5],'port':8203,'cpuset':'0-31','env':{}}],'basis':'test'}
    new_profile=copy.deepcopy(profile);new_profile.update(profile_id='new-v1',model_fragment='new-model');new_profile['instances'][0]['name']='saea-new-a'
    model={'transport':'vllm','model_lock_key':'native-frozen','model_materialization_root':str(data),
        'served_model_name':'native','server_args':['--max-model-len','49152','--dtype','bfloat16'],
        'runtime_receipt':'receipts/new-runtime.json','decoding_params':{'max_tokens':16384}}
    new_model=copy.deepcopy(model);new_model['server_args'][1]='65536'
    old_p=write(root/'old-v1.json',profile);new_p=write(root/'new-v1.json',new_profile)
    old_m=write(root/'old-model.json',model);new_m=write(root/'new-model.json',new_model)
    native=write(data/'models/native/config.json',{'thinker_config':{'text_config':{'max_position_embeddings':65536,'rope_scaling':{'rope_type':'default'}}}})
    receipt={'container':{'image_id':profile['image']},'protocol':{'server_args':model['server_args']},
             'deployment':{'instances':[{'instance_id':'saea-old-a','host':'127.0.0.1','port':8203,'device_ids':[4,5]}]}}
    write(f.receipt,receipt);source['runtime_receipt']=pin(f.receipt)
    for name in [f.session,'saea-old-a']:
        write(data/'serving'/name/'session.json',{'session_id':name,'pid':1234,'identity':{'runtime_receipt_sha256':pin(f.receipt)['sha256']}})
        write(data/'serving'/name/'session-receipt.json',{'pid':1234})
    conf={'schema':m.SCHEMA,'source_settlement':pin(f.path),'old_container_id':'d'*64,
        'old_profile':pin(old_p),'new_profile':pin(new_p),'old_model':pin(old_m),'new_model':pin(new_m),
        'native_model_config':pin(native),'logical_model_fragment':'historical-model-id',
        'new_session':'resident-new','data_root':str(data),'nic':'eth0','scope':{'cards':[4,5],'port':8203},
        'state_dir':str(root/'engine-transition'),'ready_timeout_seconds':30,'ready_interval_seconds':1}
    path=write(root/'engine-transition.json',conf);old_profile=m.deployment.load_deployment_profile(old_p);target=m.deployment.load_deployment_profile(new_p)
    old_container=container(old_profile,model,'d'*64,1234);new_container=None;calls=[];fault={'mode':None}
    def docker(argv,**kwargs):
        nonlocal new_container
        calls.append(argv)
        if argv[:3]==['docker','container','inspect']:
            value=old_container if argv[-1] in ('d'*64,'saea-old-a') else new_container
            return SimpleNamespace(returncode=0 if value else 1,stdout=json.dumps([value]) if value else '',stderr='' if value else 'No such container')
        if argv[:2]==['docker','logs']:return SimpleNamespace(returncode=0,stdout='preserved startup',stderr='')
        if argv[:2]==['docker','stop']:
            old_container['State'].update(Status='exited',Running=False,Pid=0,FinishedAt='2026-11-03T10:11:00Z')
            if fault['mode']=='after-stop':raise OSError('lost stop reply')
            return SimpleNamespace(returncode=0,stdout='d'*64,stderr='')
        if argv==['npu-smi','info']:return SimpleNamespace(returncode=0,stdout=IDLE,stderr='')
        assert argv[:2]==['docker','run']
        if fault['mode']=='launch-unknown':raise OSError('lost before creation')
        new_container=container(target,new_model,'e'*64,5678)
        if fault['mode']=='after-launch':raise OSError('lost launch reply')
        return SimpleNamespace(returncode=0,stdout='e'*64,stderr='')
    monkeypatch.setattr(m.launch,'run',docker)
    monkeypatch.setattr(m,'source_barrier',lambda c:{'verified':'fixture-source'})
    monkeypatch.setattr(m,'verify_live_old',lambda *args:None)
    monkeypatch.setattr(m.settlement.transition,'quiescence',lambda *args:{'quiet':True})
    monkeypatch.setattr(m,'wait_ready',lambda c,cid:m.inspect(cid))
    def register(c,current):
        if fault['mode']=='registration':raise ValueError('new runtime not ready')
        path=write(c['out']/'runtime-receipt.json',{'observed_pid':current['State']['Pid']})
        obs=write(c['out']/'observations.json',{'pid':current['State']['Pid']})
        return {'runtime_receipt':pin(path),'observations':pin(obs),'registered_files':[]}
    monkeypatch.setattr(m,'finish_registration',register);f.lease.unlink()
    return SimpleNamespace(**locals())


def test_context_only_plan_does_not_stop_or_start(engine):
    e=engine;result=m.execute(e.path,'plan')
    assert result['new_model_calls']==0 and not result['mutates_services']
    assert not any(c[:2] in (['docker','stop'],['docker','run']) for c in e.calls)
    assert result['argv'][-len(m.deployment.instance_container_args(e.target,e.target.instances[0],e.new_model)):] == m.deployment.instance_container_args(e.target,e.target.instances[0],e.new_model)


def test_single_lane_transition_preserves_old_container_and_logical_model_id(engine):
    e=engine;result=m.execute(e.path)
    assert result['new_context']==65536 and result['logical_model_fragment']=='historical-model-id'
    assert result['old_container_preserved'] and result['new_model_calls']==0
    assert [c for c in e.calls if c[:2]==['docker','stop']]==[['docker','stop','--time','60','d'*64]]
    assert len([c for c in e.calls if c[:2]==['docker','run']])==1
    assert not any('rm' in c for c in e.calls)
    e.calls.clear();assert m.execute(e.path)==result and not e.calls


@pytest.mark.parametrize('fault',['foreign-id','wrong-cards','busy','active-lease','unknown-contact','above-native','changed-flag','new-exists','runtime-verify'])
def test_unsafe_transition_never_stops_engine(engine,monkeypatch,fault):
    e=engine
    if fault=='foreign-id':e.old_container['Id']='f'*64
    elif fault=='wrong-cards':e.old_container['HostConfig']['Devices'].append({'PathOnHost':'/dev/davinci6'})
    elif fault=='busy':monkeypatch.setattr(m.settlement.transition,'quiescence',lambda *a:(_ for _ in ()).throw(ValueError('busy')))
    elif fault=='active-lease':write(e.f.lease,{'unknown':True})
    elif fault=='unknown-contact':monkeypatch.setattr(m,'source_barrier',lambda *a:(_ for _ in ()).throw(ValueError('unknown')))
    elif fault in ('above-native','changed-flag'):
        doc=copy.deepcopy(e.new_model)
        if fault=='above-native':doc['server_args'][1]='98304'
        else:doc['server_args']+=['--seed','9']
        write(e.new_m,doc);e.conf['new_model']=pin(e.new_m);write(e.path,e.conf)
    elif fault=='new-exists':
        old=m.inspect;monkeypatch.setattr(m,'inspect',lambda name:e.old_container if name=='saea-new-a' else old(name))
    else:monkeypatch.setattr(m,'verify_live_old',lambda *a:(_ for _ in ()).throw(ValueError('wrong runtime')))
    with pytest.raises(ValueError):m.execute(e.path)
    assert not any(c[:2] in (['docker','stop'],['docker','run']) for c in e.calls)


@pytest.mark.parametrize('fault',['after-stop','after-launch','registration'])
def test_interruption_resumes_exact_engine_without_duplicate_start(engine,fault):
    e=engine;e.fault['mode']=fault
    with pytest.raises((OSError,ValueError)):m.execute(e.path)
    e.fault['mode']=None;assert m.execute(e.path)['new_container_id']=='e'*64
    assert len([c for c in e.calls if c[:2]==['docker','stop']])==1
    assert len([c for c in e.calls if c[:2]==['docker','run']])==1


def test_unknown_launch_without_container_never_repeats(engine):
    e=engine;e.fault['mode']='launch-unknown'
    with pytest.raises(OSError):m.execute(e.path)
    e.fault['mode']=None
    with pytest.raises(ValueError,match='unresolved'):m.execute(e.path)
    assert len([c for c in e.calls if c[:2]==['docker','run']])==1


def test_other_pending_transition_is_not_bypassed(engine):
    e=engine;write(e.f.locks/'engine-transition.pending.json',{'state_dir':'/other','identity':{}})
    with pytest.raises(ValueError,match='another'):m.execute(e.path)
    assert not any(c[:2]==['docker','stop'] for c in e.calls)


def test_source_native_contact_barrier_is_real_and_refuses_unknown(configured,monkeypatch):
    f=configured;seal(f)
    c={'source':f.config,'producer':f.launch_doc,'release':f.output/'generation-released.json',
       'logical':f.logical,'expected':f.expected,'doc':{'source_settlement':pin(f.path)}}
    monkeypatch.setattr(m.launch,'status',lambda *a:{'container_id':'c'*64,'state':'running'})
    assert len(m.source_barrier(c)['contacts'])==1
    f.journal._path('req-1','response').unlink()
    with pytest.raises(ValueError,match='unknown'):m.source_barrier(c)

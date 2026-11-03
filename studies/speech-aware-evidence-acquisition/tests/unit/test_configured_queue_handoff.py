"""Independent queue handoff refuses unsafe states and preserves scientific failure."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from test_configured_terminal_settlement import configured,write,pin,seal,m as settlement

spec=importlib.util.spec_from_file_location('queue_handoff_test',Path(__file__).parents[2]/'scripts/environment/configured_queue_handoff.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

@pytest.fixture
def handoff(configured,monkeypatch):
    f=configured;seal(f);out=f.tmp_path/'handoff';state_dir=f.tmp_path/'queue';config=f.tmp_path/'queue-config.json'
    write(config,{});identity={'config_sha256':settlement.launcher.digest(config)}
    state={'identity':identity,'queue_id':'source','status':'failed','jobs':[{'id':'producer','status':'failed','container_id':'c'*64},{'id':'cleanup','status':'complete'}]}
    state_path=write(state_dir/'state.json',state)
    source={'queue_id':'source','state_dir':str(state_dir),'jobs':[{'id':'producer'},{'id':'cleanup'}]}
    doc={'queue_config':pin(config),'watch_launcher':pin(f.helper)}
    original=SimpleNamespace(identities=lambda path:identity)
    binding={'source':f.config,'producer':f.launch_doc,'logical':f.logical,'expected':f.expected,'release':f.output/'generation-released.json','locks':f.locks}
    bindings={'producer':binding}
    monkeypatch.setattr(m,'settlement',settlement);monkeypatch.setattr(m,'queue',settlement.queue)
    monkeypatch.setattr(m,'load',lambda path:(doc,original,source,f.launch_doc,bindings,out))
    monkeypatch.setattr(m.transition,'source_barrier',lambda c:{'contacts':settlement.source_custody(c['source'],c['producer'],c['logical'],c['expected'],c['release'])['contacts'],'proof':'verified by independent settlement tests'})
    return SimpleNamespace(**locals())

def test_failed_queue_with_terminal_safe_cleanup_can_handoff(handoff):
    h=handoff;result=m.inspect(h.config)
    assert result['ready'] and result['queue_state']=='failed' and not result['scientific_success']
    assert result['new_model_calls']==0 and result['successor_requires_own_lease_and_readiness']

@pytest.mark.parametrize('state',['ready','running','waiting'])
def test_running_queue_waits_without_custody_or_model_calls(handoff,monkeypatch,state):
    h=handoff;h.state['status']=state;write(h.state_path,h.state)
    monkeypatch.setattr(settlement,'producer_terminal',lambda *a:pytest.fail('must wait'))
    assert m.inspect(h.config)=={'ready':False,'reason':'queue-running','state':state}

@pytest.mark.parametrize('fault',['unknown','pending','blocked','identity','container','lease','transition','unknown-contact','guard-replaced'])
def test_unsafe_or_incomplete_queue_never_publishes_handoff(handoff,monkeypatch,fault):
    h=handoff
    if fault=='unknown':h.state['status']='unknown'
    elif fault in ('pending','blocked'):h.state['jobs'][1]['status']=fault
    elif fault=='identity':h.state['identity']={'other':True}
    elif fault=='container':h.state['jobs'][0]['container_id']='d'*64
    elif fault=='lease':write(h.f.lease,{'successor':True})
    elif fault=='transition':write(h.f.locks/'engine-transition.pending',{})
    elif fault=='unknown-contact':h.f.journal._path('req-1','response').unlink()
    else:
        def replace(instances):
            p=h.f.locks/'generation.guard';p.rename(h.f.locks/'old.guard');p.touch();return {}
        monkeypatch.setattr(settlement.transition,'quiescence',replace)
    write(h.state_path,h.state)
    with pytest.raises(ValueError):m.inspect(h.config)
    assert not (h.out/'complete.json').exists()

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

p=Path(__file__).parents[2]/'scripts/analysis/recover_seal_only_finalize.py'
s=importlib.util.spec_from_file_location('seal_finalize_recovery',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)


def evidence():
    cfg={'run_id':'run','attempt_id':'successor','rows':2,'checkpoint_sha256':'a'*64}
    manifest={'status':'COMPLETED','scores':None,'run_id':'run','attempt':{'attempt_id':'successor'},'study_commit':'c'*40,
              'cost':{'calls_used':0},'config':{'config_hash':'d'*64},'protocol_hash':'e'*64,
              'predecessor_attempt':{'attempt_id':'old','checkpoint_sha256':'a'*64,'counts':{'kept':2,'flown':0,'partial':0}}}
    bundle=SimpleNamespace(attempt_state='OPENED',manifest=manifest,outputs_rows=[{},{}],split_ids={'a','b'})
    pre={'config_hash':'d'*64,'protocol_hash':'e'*64,'sample_count':2,'resume':{'seal_only':True,
        'counts':{'roster':2,'kept':2,'flown':0,'partial':0,'unstarted':0},'worst_case_contacts':0,'predecessor_contacts':5}}
    receipts={'stages':{'preflight':pre,'preprocess':{},'serve':{},'run_phases':{'seal_only':True,'resume':{'kept':2,'flown':0}}}}
    gen={'schema':'saea-seal-only-generation-manifest-v1','run_id':'run','attempt_id':'successor','seal_only':True,
         'cost_provenance':{'predecessor_attempt_ids':['old']},'cost':{'calls_used':5}}
    return cfg,bundle,receipts,gen,'finalize_bundle score_bundle delivery config differs from recorded run',{'generation_commit':'c'*40}


def test_exact_seal_only_evidence_passes():
    m.validate_evidence(*evidence())


@pytest.mark.parametrize('fault',['opened','score','roster','partial','flown','checkpoint','cost','failure','config','stages'])
def test_changed_source_or_failure_is_refused(fault):
    args=list(evidence());cfg,bundle,receipts,gen,failed,identity=args
    if fault=='opened':bundle.attempt_state='FINALIZED'
    elif fault=='score':bundle.manifest['scores']={'path':'existing'}
    elif fault=='roster':bundle.outputs_rows=[]
    elif fault=='partial':receipts['stages']['preflight']['resume']['counts']['partial']=1
    elif fault=='flown':bundle.manifest['predecessor_attempt']['counts']['flown']=1
    elif fault=='checkpoint':bundle.manifest['predecessor_attempt']['checkpoint_sha256']='b'*64
    elif fault=='cost':gen['cost']['calls_used']=0
    elif fault=='failure':args[4]='different error'
    elif fault=='config':receipts['stages']['preflight']['config_hash']='bad'
    else:receipts['stages']['seal']={}
    with pytest.raises(ValueError):m.validate_evidence(*args)

@pytest.mark.parametrize('fail', [False, True])
def test_recover_executes_and_preserves_intent(tmp_path, monkeypatch, fail):
    cfg,bundle,receipts,generation,failed,identity=evidence()
    run=tmp_path/'run'; run.mkdir()
    runtime=tmp_path/'runtime.json'; runtime.write_text('{}')
    launch={'argv':['--runtime-config',str(runtime)], 'environment':{'SAEA_RUNTIME_CONFIG_SHA256':m.sha(runtime),'SPEECHRL_DATA_DIR':str(tmp_path)},
            'mounts':[{'source':str(tmp_path),'target':str(tmp_path),'readonly':True}]}
    label=m.hashlib.sha256(json.dumps(launch,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    producer=[{'Id':'producer','State':{'Status':'exited','Running':False},'Config':{'Labels':{'saea.launch.config':label},'Env':['SAEA_RUNTIME_CONFIG_SHA256='+m.sha(runtime),'SPEECHRL_DATA_DIR='+str(tmp_path)]},
               'Mounts':[{'Destination':str(tmp_path),'Source':str(tmp_path),'RW':False,'Type':'bind'}]}]
    seal=[{'Id':'seal','State':{'Status':'exited','Running':False,'ExitCode':1},'Config':{'Image':'image'},'Args':['seal-only',str(runtime)]}]
    def write(path, data):
        path.write_text(data if isinstance(data,str) else json.dumps(data)); return {'path':str(path),'sha256':m.sha(path)}
    inputs={'manifest':write(run/'manifest.json',{}),'generation':write(run/'generation.json',generation),
            'receipts':write(run/'receipts.json',{}),'failed_stderr':write(tmp_path/'stderr',failed),
            'original_launch':write(tmp_path/'launch.json',launch),'producer_container':write(tmp_path/'producer.json',producer),
            'failed_container':write(tmp_path/'seal.json',seal),'runtime':{'path':str(runtime),'sha256':m.sha(runtime)},
            'admission':write(tmp_path/'admission.json',{}),'official_python':write(tmp_path/'python','python')}
    cfg.update(schema='saea-seal-only-finalize-recovery-v1',producer_id='producer',failed_container_id='seal',image='image',
               inputs=inputs,output=str(run/'recovery'),lock=write(tmp_path/'lock.json',{}))
    config=tmp_path/'config.json'; config.write_text(json.dumps(cfg))
    bundle.manifest['config']['fragments']={'baseline':'b','model':'m','dataset':'d','experiment':'e'}
    bundle.manifest['plan']={'carrier_lock_key':'carrier','split_identity_hash':'split'}
    receipts['stages']['preflight']['split_id']='fold'
    baseline=tmp_path/'baseline';baseline.write_text('{}');model=tmp_path/'model';model.write_text('{}')
    monkeypatch.setenv('SAEA_RUNTIME_CONFIG',str(runtime));monkeypatch.setenv('SAEA_RUNTIME_CONFIG_SHA256',m.sha(runtime));monkeypatch.setenv('SPEECHRL_DATA_DIR',str(tmp_path))
    monkeypatch.setattr(m,'assert_execution_environment',lambda:None)
    monkeypatch.setattr(m,'verify_separate_scoring_tool',lambda *args:identity)
    monkeypatch.setattr(m,'read_flight_receipts',lambda path:receipts)
    monkeypatch.setattr(m,'_recorded_delivery_config_hash',lambda *args:'d'*64)
    monkeypatch.setattr(m,'resolve_asset',lambda source,name:baseline if '/baseline/' in name else model)
    calls=[]
    def verify(path):
        if calls: bundle.attempt_state='FINALIZED'
        return bundle
    monkeypatch.setattr(m,'verify_bundle',verify)
    def score(jobs, **kwargs):
        calls.append((jobs,kwargs))
        assert jobs[0].job_id == 'successor'
        assert (run/'recovery'/'intent.json').is_file()
        return {'succeeded':0,'failed':1} if fail else {'succeeded':1,'failed':0}
    monkeypatch.setattr(m,'run_score_batch',score)
    if fail:
        with pytest.raises(ValueError,match='finalize failed'):m.recover(config)
        assert not (run/'recovery'/'recovery-receipt.json').exists()
    else:
        m.recover(config)
        assert json.loads((run/'recovery'/'recovery-receipt.json').read_text())['new_model_calls']==0
    assert len(calls)==1 and (run/'recovery'/'intent.json').exists()
    bundle.attempt_state='OPENED'
    monkeypatch.setattr(m,'verify_bundle',lambda path:bundle)
    with pytest.raises(ValueError,match='fresh adjacent'):m.recover(config)

"""Real Git/config/checkpoint proof of a bounded operational transition."""
import copy
import hashlib
import json
import subprocess
from pathlib import Path

import pytest
from speech_aware_evidence_acquisition.core.config import compose, protocol_hash
from speech_aware_evidence_acquisition.core.flight.checkpoint_compatibility import (
    CheckpointCompatibility, CheckpointCompatibilityError, IMPLEMENTATION, PREFIX, SCHEMA,
)
from speech_aware_evidence_acquisition.core.flight.resume import read_checkpoint_predecessor, ResumeRecoveryError
from speech_aware_evidence_acquisition.core.row_checkpoint import RowCheckpointWriter, sha256_file


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def transition(tmp_path):
    repo, data = tmp_path/'repo', tmp_path/'data'
    repo.mkdir(); data.mkdir()
    fragments = dict(model='m', dataset='d', baseline='b', experiment='e')
    supply = data/'old-supply.json'; write(supply, {'output': {'rows': ['unchanged']}})
    base = {'arm':'test-arm','native_contact_fit':{'run_id':'fit-old','attempt_id':'fit-old-a','contact_cap':20},
            'prepared_phase_supply':{'relpath':supply.name,'sha256':digest(supply)}, 'policy':'phase-direct-only-v1'}
    cell={'run_id':'run','attempt_id':'old','baseline_fragment':'b','readiness':{'state':'ready'}}
    exp={'cells':[cell], 'population':{'ids':['s1','s2','s3']}}
    for kind, obj in zip(fragments, ({'model':'frozen'}, {'dataset':'frozen'}, base, exp)):
        write(repo/'configs'/kind/(fragments[kind]+'.json'), obj)
    for path in IMPLEMENTATION.values():
        p=repo/path; p.parent.mkdir(parents=True,exist_ok=True); p.write_text('# original\n')
    executor=repo/PREFIX/'reproduction/ojw/runner.py'; executor.parent.mkdir(parents=True); executor.write_text('# default pool\n')
    for args in (['init','-q'], ['add','.'], ['-c','user.name=Test','-c','user.email=test@example.invalid','commit','-qm','predecessor']):
        subprocess.run(['git',*args],cwd=repo,check=True,capture_output=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    old=compose(repo,**fragments)
    header={'config_hash':old.config_hash,'protocol_hash':protocol_hash(old.config_hash,'test-arm','a'*64),
            'split_identity_hash':'a'*64,'implementation':{k:digest(repo/p) for k,p in IMPLEMENTATION.items()}}
    writer=RowCheckpointWriter(data/'runs/run/old','run','old',identity=header)
    output={'sample_id':'s1','request_id':'q1','text':'kept'}
    trace=[json.dumps({'record_id':'q1/'+kind,'channel':channel,'payload':{}})
           for kind,channel in (('request','MODEL_REQUEST'),('response','MODEL_RESPONSE'))]
    writer.append_row(sample_id='s1',complete=True,output_rows=[output],trace_lines=trace); writer.close()
    header.update(run_id='run',attempt_id='old')
    decl={'attempt_id':'old','checkpoint_sha256':sha256_file(writer.path)}
    write(data/'attempts/run/old.json',{'state':'ABORTED','run_id':'run','attempt_id':'old'})
    before=digest(executor); executor.write_text('# reviewed explicit pool\n')
    bridge={'schema':SCHEMA,'predecessor_commit':commit,'predecessor_fragments':fragments,
            'successor_attempt_id':'new','change':'contact-executor-width-v1',
            'reviewed_source_changes':{str(executor.relative_to(repo)):{'before':before,'after':digest(executor)}}}
    base['native_contact_fit'].update(run_id='fit-new',attempt_id='fit-new-a')
    write(repo/'configs/baseline/b.json',base)
    cell.update(attempt_id='new',resume_from=decl,checkpoint_compatibility=bridge)
    write(repo/'configs/experiment/e.json',exp)
    def verify():
        new=compose(repo,**fragments)
        return CheckpointCompatibility.from_document(bridge,attempt_id='new').verify(
            repo_root=repo,data_root=data,current_fragments=fragments,header=header,run_id='run',
            predecessor_attempt_id='old',current_protocol_hash=protocol_hash(new.config_hash,'test-arm','a'*64),
            current_split_identity_hash='a'*64,current_implementation={k:digest(repo/p) for k,p in IMPLEMENTATION.items()},
            declaration=decl)
    return repo,data,header,decl,bridge,verify


def test_actual_config_transition_keeps_checkpoint_without_replay(transition):
    repo,data,header,decl,bridge,verify=transition
    verified=verify()
    with pytest.raises(ResumeRecoveryError,match='protocol'):
        read_checkpoint_predecessor(run_id='run',attempt_id='old',attempt_dir=data/'runs/run/old',
            expected_checkpoint_sha256=decl['checkpoint_sha256'],materialize_dir=data/'rejected',protocol_hash='f'*64)
    assert not (data/'rejected').exists()
    kept=read_checkpoint_predecessor(run_id='run',attempt_id='old',attempt_dir=data/'runs/run/old',
        expected_checkpoint_sha256=decl['checkpoint_sha256'],materialize_dir=data/'new/predecessor',
        protocol_hash=verified['protocol_hash'],split_identity_hash='a'*64,implementation=verified['implementation'])
    assert kept.complete_sample_ids == ('s1',)
    assert kept.canonical_rows_for(['s1'])[0]['text']=='kept'


@pytest.mark.parametrize('fault', ['model','roster','baseline','source','split','commit','state','supply','declaration'])
def test_semantic_and_custody_changes_refused(transition,fault):
    repo,data,header,decl,bridge,verify=transition
    if fault in ('model','baseline','roster','declaration'):
        kind={'roster':'experiment','declaration':'experiment'}.get(fault,fault)
        name={'model':'m','baseline':'b','experiment':'e'}[kind]
        p=repo/'configs'/kind/(name+'.json'); obj=json.loads(p.read_bytes())
        if fault=='roster': obj['population']['ids'].append('s3')
        elif fault=='declaration': obj['cells'][0]['resume_from']['checkpoint_sha256']='f'*64
        else: obj['unreviewed_semantics']=True
        write(p,obj)
    elif fault=='source': (repo/IMPLEMENTATION['runner']).write_text('# changed request rendering\n')
    elif fault=='split': header['split_identity_hash']='b'*64
    elif fault=='commit': header['config_hash']='b'*64
    elif fault=='state': write(data/'attempts/run/old.json',{'state':'OPENED','run_id':'run','attempt_id':'old'})
    elif fault=='supply': write(data/'old-supply.json',{'output':'changed'})
    with pytest.raises(CheckpointCompatibilityError): verify()


def test_no_self_referential_successor_protocol_pin(transition):
    _,_,_,_,bridge,verify=transition
    assert 'successor_protocol_hash' not in bridge
    assert verify()['change']=='contact-executor-width-v1'


def test_runtime_keeps_every_completed_row_and_dispatches_only_missing(transition):
    from types import SimpleNamespace
    from speech_aware_evidence_acquisition.core.flight.runtime import FlightRuntime, FlightRuntimeError
    from speech_aware_evidence_acquisition.core.flight.resume import ResumeDeclaration
    repo,data,header,decl,bridge,verify=transition
    runtime=object.__new__(FlightRuntime)
    runtime._repo_root, runtime._data_root=repo,data
    runtime._run_id, runtime._attempt_id='run','new'
    runtime._model,runtime._dataset,runtime._baseline,runtime._experiment='m','d','b','e'
    runtime._explicit_run_dir=None
    cfg=compose(repo,'m','d','b','e')
    runtime._plan=SimpleNamespace(protocol_hash=protocol_hash(cfg.config_hash,'test-arm','a'*64),split_identity_hash='a'*64)
    runtime._resume_declarations=(ResumeDeclaration.from_document(decl),)
    runtime._checkpoint_compatibility=CheckpointCompatibility.from_document(bridge,attempt_id='new')
    runtime.row_checkpoint_implementation=lambda:{k:digest(repo/p) for k,p in IMPLEMENTATION.items()}
    runtime.shows_construction_targets=lambda:False
    runtime.execution_unit_roster=lambda:None
    runtime._result_kind=lambda:'task-answer'
    result=runtime._load_resume_predecessor(cfg.values,('s1','s2','s3'))
    assert result['counts']=={'roster':3,'kept':1,'flown':2,'partial':0,'unstarted':2}
    assert runtime._resume_dispatch_roster()==('s2','s3')
    assert runtime._resume.complete_sample_ids == ('s1',)
    runtime._checkpoint_compatibility=None
    with pytest.raises(FlightRuntimeError,match='protocol'):
        runtime._load_resume_predecessor(cfg.values,('s1','s2'))


@pytest.fixture
def multihop(transition):
    """Two real Git ancestors with distinct protocol/source identities."""
    repo, data, header, first, old_bridge, verify = transition
    from speech_aware_evidence_acquisition.core.flight.checkpoint_compatibility import MULTI_SCHEMA
    fragments = old_bridge['predecessor_fragments']
    exp_path = repo/'configs/experiment/e.json'
    exp = json.loads(exp_path.read_bytes())
    # The first successor's config declares inherited s1 but checkpoints only
    # its own newly completed s2. This is the production runtime contract.
    for args in (['add','.'], ['-c','user.name=Test','-c','user.email=test@example.invalid','commit','-qm','middle']):
        subprocess.run(['git',*args],cwd=repo,check=True,capture_output=True)
    middle_commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    cfg = compose(repo, **fragments)
    identity = {'config_hash':cfg.config_hash,'protocol_hash':protocol_hash(cfg.config_hash,'test-arm','a'*64),
                'split_identity_hash':'a'*64,'implementation':{k:digest(repo/p) for k,p in IMPLEMENTATION.items()}}
    writer=RowCheckpointWriter(data/'runs/run/new','run','new',identity=identity)
    trace=[json.dumps({'record_id':'q1/'+kind,'channel':channel,'payload':{}})
           for kind,channel in (('request','MODEL_REQUEST'),('response','MODEL_RESPONSE'))]
    writer.append_row(sample_id='s2',complete=True,output_rows=[{'sample_id':'s2','request_id':'q1','text':'middle'}],trace_lines=trace)
    writer.close()
    middle={'attempt_id':'new','checkpoint_sha256':sha256_file(writer.path)}
    write(data/'attempts/run/new.json',{'state':'ABORTED','run_id':'run','attempt_id':'new'})
    executor=repo/PREFIX/'reproduction/ojw/runner.py'
    middle_before=digest(executor);executor.write_text('# reviewed next pool\n')
    first_bridge=copy.deepcopy(old_bridge)
    first_bridge['successor_attempt_id']='next'
    first_bridge['reviewed_source_changes'][str(executor.relative_to(repo))]['after']=digest(executor)
    middle_bridge=copy.deepcopy(first_bridge)
    middle_bridge['predecessor_commit']=middle_commit
    middle_bridge['reviewed_source_changes'][str(executor.relative_to(repo))]['before']=middle_before
    bridge={'schema':MULTI_SCHEMA,'successor_attempt_id':'next','bridges':{'old':first_bridge,'new':middle_bridge}}
    cell=exp['cells'][0];cell.update(attempt_id='next',resume_from=[first,middle],checkpoint_compatibility=bridge)
    write(exp_path,exp)
    base_path=repo/'configs/baseline/b.json';base=json.loads(base_path.read_bytes())
    base['native_contact_fit'].update(run_id='fit-next',attempt_id='fit-next-a');write(base_path,base)
    return repo,data,first,middle,bridge


def multi_runtime(world):
    from types import SimpleNamespace
    from speech_aware_evidence_acquisition.core.flight.runtime import FlightRuntime
    from speech_aware_evidence_acquisition.core.flight.resume import resume_declarations_from_document
    repo,data,first,middle,bridge=world
    cfg=compose(repo,'m','d','b','e')
    cell=cfg.values['cells'][0]
    runtime=object.__new__(FlightRuntime)
    runtime._repo_root,runtime._data_root=repo,data
    runtime._run_id,runtime._attempt_id='run','next'
    runtime._model,runtime._dataset,runtime._baseline,runtime._experiment='m','d','b','e'
    runtime._explicit_run_dir=None
    runtime._plan=SimpleNamespace(protocol_hash=protocol_hash(cfg.config_hash,'test-arm','a'*64),split_identity_hash='a'*64)
    runtime._resume_declarations=resume_declarations_from_document(cell['resume_from'])
    runtime._checkpoint_compatibility=CheckpointCompatibility.from_document(cell['checkpoint_compatibility'],attempt_id='next')
    runtime.row_checkpoint_implementation=lambda:{k:digest(repo/p) for k,p in IMPLEMENTATION.items()}
    runtime.shows_construction_targets=lambda:False
    runtime.execution_unit_roster=lambda:None
    runtime._result_kind=lambda:'task-answer'
    return runtime,cfg


def test_multihop_keeps_ancestors_and_only_dispatches_complement(multihop):
    runtime,cfg=multi_runtime(multihop)
    result=runtime._load_resume_predecessor(cfg.values,('s1','s2','s3'))
    assert result['counts']['kept']==2
    assert runtime._resume_dispatch_roster()==('s3',)
    assert set(runtime._resume.complete_sample_ids)=={'s1','s2'}
    assert set(runtime._checkpoint_compatibility_receipt['predecessors'])=={'old','new'}


@pytest.mark.parametrize('fault', ['missing-ancestor','changed-ancestor-pin','missing-bridge','extra-bridge','model','source','middle-open','middle-config'])
def test_multihop_refuses_custody_or_scientific_changes(multihop,fault):
    from speech_aware_evidence_acquisition.core.flight.runtime import FlightRuntimeError
    repo,data,first,middle,bridge=multihop
    p=repo/'configs/experiment/e.json';exp=json.loads(p.read_bytes());cell=exp['cells'][0]
    if fault=='missing-ancestor':
        cell['resume_from']=[middle];cell['checkpoint_compatibility']['bridges'].pop('old')
    elif fault=='changed-ancestor-pin':cell['resume_from'][0]['checkpoint_sha256']='f'*64
    elif fault=='missing-bridge':cell['checkpoint_compatibility']['bridges'].pop('old')
    elif fault=='extra-bridge':cell['checkpoint_compatibility']['bridges']['foreign']=copy.deepcopy(bridge['bridges']['old'])
    elif fault=='model':write(repo/'configs/model/m.json',{'model':'different'})
    elif fault=='source':(repo/IMPLEMENTATION['runner']).write_text('# changed semantics\n')
    elif fault=='middle-open':write(data/'attempts/run/new.json',{'state':'OPENED','run_id':'run','attempt_id':'new'})
    elif fault=='middle-config':cell['checkpoint_compatibility']['bridges']['new']['predecessor_commit']=bridge['bridges']['old']['predecessor_commit']
    write(p,exp)
    runtime,cfg=multi_runtime(multihop)
    with pytest.raises((FlightRuntimeError,CheckpointCompatibilityError)):
        runtime._load_resume_predecessor(cfg.values,('s1','s2','s3'))


def test_v1_does_not_accept_multiple_predecessors(multihop):
    from speech_aware_evidence_acquisition.core.flight.checkpoint_compatibility import validate_bridge_declarations
    from speech_aware_evidence_acquisition.core.flight.resume import resume_declarations_from_document
    _,_,first,middle,bridge=multihop
    parsed=CheckpointCompatibility.from_document(bridge['bridges']['old'],attempt_id='next')
    with pytest.raises(CheckpointCompatibilityError,match='exactly one'):
        validate_bridge_declarations(parsed,resume_declarations_from_document([first,middle]))


def test_multi_bridge_rejects_self_and_recursive_envelope(multihop):
    from speech_aware_evidence_acquisition.core.flight.checkpoint_compatibility import MULTI_SCHEMA
    _,_,_,_,bridge=multihop
    bad=copy.deepcopy(bridge);bad['bridges']['next']=bad['bridges'].pop('old')
    with pytest.raises(CheckpointCompatibilityError,match='itself'):
        CheckpointCompatibility.from_document(bad,attempt_id='next')
    bad=copy.deepcopy(bridge);bad['bridges']['old']={'schema':MULTI_SCHEMA}
    with pytest.raises(CheckpointCompatibilityError,match='v1 schema'):
        CheckpointCompatibility.from_document(bad,attempt_id='next')


def test_cycle_is_rejected_before_checkpoint_materialization(multihop,monkeypatch):
    import speech_aware_evidence_acquisition.core.flight.checkpoint_compatibility as compat
    from speech_aware_evidence_acquisition.core.flight.resume import resume_declarations_from_document
    repo,_,first,middle,bridge=multihop
    by_commit={bridge['bridges']['old']['predecessor_commit']:{'cells':[{'attempt_id':'old','resume_from':middle}]},
               bridge['bridges']['new']['predecessor_commit']:{'cells':[{'attempt_id':'new','resume_from':first}]}}
    monkeypatch.setattr(compat,'git',lambda repo,*args:json.dumps(by_commit[args[1].split(':')[0]]).encode())
    parsed=CheckpointCompatibility.from_document(bridge,attempt_id='next')
    with pytest.raises(CheckpointCompatibilityError,match='cyclic'):
        parsed.validate_ancestry(repo,resume_declarations_from_document([first,middle]))


def test_config_only_bridge_can_explicitly_pin_empty_source_delta(transition):
    _,_,_,_,bridge,_=transition
    bridge=copy.deepcopy(bridge);bridge['reviewed_source_changes']={}
    assert CheckpointCompatibility.from_document(bridge).reviewed_source_changes=={}


def test_campaign_accepts_only_exact_v2_bridge_set(multihop):
    from speech_aware_evidence_acquisition.core.flight.campaign import CampaignCell, CampaignError
    repo,_,_,_,_=multihop
    exp=json.loads((repo/'configs/experiment/e.json').read_bytes());cell=exp['cells'][0]
    cell.update(cell_id='c',model_fragment='m',dataset_fragment='d')
    assert CampaignCell.from_document(cell).attempt_id=='next'
    cell['checkpoint_compatibility']['bridges'].pop('old')
    with pytest.raises(CampaignError,match='exactly match'):
        CampaignCell.from_document(cell)

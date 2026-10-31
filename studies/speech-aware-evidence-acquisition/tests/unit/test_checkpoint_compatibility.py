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
    exp={'cells':[cell], 'population':{'ids':['s1','s2']}}
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
    result=runtime._load_resume_predecessor(cfg.values,('s1','s2'))
    assert result['counts']=={'roster':2,'kept':1,'flown':1,'partial':0,'unstarted':1}
    assert runtime._resume_dispatch_roster()==('s2',)
    assert runtime._resume.complete_sample_ids == ('s1',)
    runtime._checkpoint_compatibility=None
    with pytest.raises(FlightRuntimeError,match='protocol'):
        runtime._load_resume_predecessor(cfg.values,('s1','s2'))

import copy
import json
from pathlib import Path
import pytest
from speech_aware_evidence_acquisition.rails import tfgrpo_consumption as m
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import empty_experience,sha,canonical
from speech_aware_evidence_acquisition.rails import consumption_module as cm
from speech_aware_evidence_acquisition.rails.adapters.slurp_module_hooks import carrier_module_hooks
from speech_aware_evidence_acquisition.rails.phase_material import PhaseMaterialSpec,render_static_instruction
ROOT=Path(__file__).resolve().parents[2]
SOURCE='slurp:train:000001:real'; TARGET='slurp:train:000002:real'


def candidate():
    c=empty_experience('model');c.pop('sha256')
    usage={'calls':1,'prompt_tokens':10,'completion_tokens':5}
    c.update(items={'E1':'When intent evidence conflicts, compare the requested action.'},learned_query_ids=[SOURCE],
             batch_sha256='a'*64,planned_learning_calls=1,
             contact_response_sha256={'one.response.json':'b'*64},learning_usage=usage,
             accounting={'unknown_contacts':0,'attempted_contacts':1,'observed_contacts':1,'known_usage':usage})
    return {**c,'sha256':sha(c)}


@pytest.fixture
def release(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from speech_aware_evidence_acquisition.core.bundle import verify as bundle_verify
    data=tmp_path/'data';repo=tmp_path/'repo';data.mkdir();(repo/'configs/probes').mkdir(parents=True)
    p=data/'candidate.json';p.write_bytes(canonical(candidate()))
    manifest=data/'manifest.json';manifest.write_text('{}')
    proposal={'schema':'saea-tfgrpo-flight-proposal-v1','result_kind':'artifact',
        'candidate':{'root':'data','relpath':'candidate.json','sha256':m.digest(p),
                     'experience_sha256':candidate()['sha256'],'deployment_accepted':False},
        'batch_sha256':candidate()['batch_sha256'],
        'contact_response_sha256':candidate()['contact_response_sha256'],'usage':candidate()['learning_usage']}
    monkeypatch.setattr(bundle_verify,'verify_bundle',lambda path:SimpleNamespace(
        attempt_state='FINALIZED',trace_records={'proposal':{'org_learning_proposal':proposal}}))
    (repo/'configs/probes/base.txt').write_text('Original Solver instruction.')
    kwargs=dict(candidate_path=p,source_manifest=manifest,repo=repo,data=data,base_template='probes/base.txt',template='probes/learned.txt',
                consumer_ids=[TARGET],model_id='model')
    params=m.prepare(**kwargs)
    return repo,data,params


def test_exact_experience_reaches_existing_phase_renderer(release):
    repo,data,params=release
    assert m.verify(params,repo=repo,data=data,model_id='model',consumer_ids=[TARGET])==candidate()['sha256']
    spec=PhaseMaterialSpec(phase='solver',purpose='Solve',instruction_template=params['template']['relpath'])
    rendered=render_static_instruction(spec,repo,{},data)
    assert rendered=='Original Solver instruction.'+m.SEPARATOR+'[E1]. '+candidate()['items']['E1']+'\n'


@pytest.mark.parametrize('fault',['candidate','template','model','roster','source_overlap','purpose','escape'])
def test_changed_or_leaking_release_refused(release,fault):
    repo,data,params=release;model='model';ids=[TARGET]
    if fault=='candidate':(data/'candidate.json').write_text('{}')
    elif fault=='template':(repo/'configs/probes/learned.txt').write_text('Different')
    elif fault=='model':model='another'
    elif fault=='roster':ids=[SOURCE]
    elif fault=='source_overlap':
        params['consumer_ids']=[SOURCE];ids=[SOURCE]
    elif fault=='purpose':params['purpose']='deployment'
    else:params['template']['relpath']='../outside'
    with pytest.raises(ValueError):m.verify(params,repo=repo,data=data,model_id=model,consumer_ids=ids)


def test_compiler_changes_solver_only(release):
    repo,data,params=release
    base=json.loads((ROOT/'configs/baseline/slurp-tfrl-hardset-a0-observation-r1-v1.json').read_text())
    block=copy.deepcopy(base['native_episode']['consumption_module'])
    before=cm.compile_module(block,carrier_module_hooks(repo_root=ROOT),repo_root=ROOT)
    params['base_template']['relpath']=before.values()['phase_material']['solver']['instruction_template']
    block['components'].append({'component_id':m.COMPONENT,'state':'on','admitted_by':'development-evaluation',
        'record':'docs/decisions/2026-10-31-owner-lock-baseline-and-cost-repair.md','params':params})
    after=cm.compile_module(block,carrier_module_hooks(repo_root=ROOT),repo_root=ROOT)
    assert cm.differing_paths(before.values(),after.values())==['phase_material.solver.instruction_template']
    assert before.program_ceiling==after.program_ceiling==2


def test_no_component_is_exact_noop():
    assert m.verify_for_flight({},repo=ROOT,data=ROOT,model_id='model',consumer_ids=[]) is None


@pytest.mark.parametrize('fault',['unfinalized','missing_proposal','changed_candidate','invalid_bundle'])
def test_release_requires_finalized_exact_proposal(release,monkeypatch,fault):
    from speech_aware_evidence_acquisition.core.bundle import verify as bundle_verify
    repo,data,params=release
    verified=bundle_verify.verify_bundle(data/'manifest.json')
    if fault=='unfinalized':verified.attempt_state='OPENED'
    elif fault=='missing_proposal':verified.trace_records={}
    elif fault=='changed_candidate':verified.trace_records['proposal']['org_learning_proposal']['candidate']['sha256']='0'*64
    else:
        def reject(path):raise ValueError('invalid bundle')
        monkeypatch.setattr(bundle_verify,'verify_bundle',reject)
    if fault!='invalid_bundle':monkeypatch.setattr(bundle_verify,'verify_bundle',lambda path:verified)
    with pytest.raises(ValueError):m.verify(params,repo=repo,data=data,model_id='model',consumer_ids=[TARGET])

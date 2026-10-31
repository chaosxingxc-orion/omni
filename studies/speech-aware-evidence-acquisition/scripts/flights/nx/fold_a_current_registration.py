"""Register the actual-A0 Fold A evaluation under independent spend identities.

Registration is model-free. Mint, own supply, preflight and dispatch are separate.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

REPO=Path(__file__).resolve().parents[3]
for directory in (Path(__file__).parent,REPO/'scripts/flights/direct',REPO/'scripts/analysis'):
    sys.path.insert(0,str(directory))
import fold_a_current_evaluation as evaluation
from l338_port_deployment_binding import validate_binding
from direct_port_refresh_registration import accepted_proof
from l338_port_family import provisional_card_hours
from speech_aware_evidence_acquisition.core.config import compose,protocol_hash
from speech_aware_evidence_acquisition.core.atomic_file import publish_new

read,sha,raw=evaluation.read,evaluation.sha,evaluation.raw
PKG=evaluation.PACKAGE
READ=evaluation.RUN+'-DELIVERED-READ'
AUTH='docs/decisions/2026-10-30-owner-full-test-and-first-system-gain.md'


def register(repo,source_repo,data,binding_path,archive,dry_run=False):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    assert_execution_environment()
    repo,source_repo,data=repo.resolve(),source_repo.resolve(),data.resolve()
    archive=evaluation.comparator.inputs.confined(data,str(archive))
    if archive.exists() or archive.is_relative_to(repo):raise ValueError('fresh external archive required')
    for tree in (repo,source_repo):
        if subprocess.check_output(['git','status','--porcelain'],cwd=tree).strip():raise ValueError('clean committed sources required')
    derivation=read(repo/PKG/'derivation.json')
    if (derivation['launchable'] is not False or derivation['model_contacts']!=0
            or subprocess.check_output(['git','rev-parse','HEAD'],cwd=source_repo,text=True).strip()!=derivation['source_commit']):
        raise ValueError('actual A0 derivation source differs')
    for name,digest in derivation['source_files'].items():
        if sha(evaluation.comparator.inputs.confined(source_repo,name))!=digest:raise ValueError('source configuration changed')
    for name,digest in derivation['outputs'].items():
        if sha(evaluation.comparator.inputs.confined(repo,name))!=digest:raise ValueError('derived Fold A configuration changed')
    _,complete_path,_=evaluation.comparator.bound_parents(source_repo,data,derivation['parent_binding'])
    complete=read(complete_path)
    if complete['rows']!=1080 or complete['score_surface']!='sealed-final-delivery':raise ValueError('complete A0 required')
    ledger_path=repo/'docs/exposure-ledger.md';ledger=ledger_path.read_bytes()
    for run in (evaluation.RUN,evaluation.FIT,READ):
        if f'| {run} |' in ledger.decode() or any((data/d/run).exists() for d in ('runs','attempts','prepared','analysis')):
            raise ValueError('Fold A identity already registered or spent')
    if (repo/PKG/'root-admission-draft-v1.json').exists() or (data/evaluation.DECISION_PATH).exists():
        raise ValueError('Fold A admission already exists')
    base_rel=f'configs/baseline/{evaluation.CELL}.json';exp_rel=f'configs/experiment/{evaluation.CELL}.json'
    base,exp=read(repo/base_rel),read(repo/exp_rel);cell,=exp['cells']
    if cell['readiness']['state']!='held' or cell['allowed_execution_stages']!=['check'] or base.get('prepared_phase_supply'):
        raise ValueError('unprepared held evaluation required')
    roster=read(repo/evaluation.comparator.inputs.ROSTER)
    ids=[r['sample_id'] for r in roster['items']]
    if (len(ids)!=1080 or len(set(ids))!=1080 or exp['populations'][0]['ids']!=ids
            or exp['population']['split_identity_hash']!=roster['split_identity_hash']
            or exp['population']['roster_receipt']!=evaluation.comparator.inputs.ROSTER):
        raise ValueError('exact Fold A population required')
    if (cell['run_id'],cell['attempt_id'],base['native_contact_fit']['run_id'],base['native_contact_fit']['attempt_id'])!=(
            evaluation.RUN,'generation-v1',evaluation.FIT,'fit-v1'):raise ValueError('independent spend identity differs')
    seconds=math.ceil(sum(r['audio_seconds'] for r in roster['items'])*34-1e-9)
    if exp['execution_plan']['planned_speech_audio_seconds']!=seconds:raise ValueError('Fold A audio allowance differs')
    binding=validate_binding(read(binding_path),repo,data)
    proof,proof_pins=accepted_proof(repo,data,binding)
    if (exp['service_session']!=binding['session_id']
            or cell['model_fragment']!=Path(binding['model_fragment']['relpath']).stem):
        raise ValueError('accepted serving identity differs')
    hours=provisional_card_hours(1080,lock_rows_per_minute_per_instance=binding['lock_rows_per_minute_per_instance'],tensor_parallel_size=binding['tensor_parallel_size'])
    exp['execution_plan'].update(planned_gpu_hours=hours,planned_model_calls=36720)
    exp['family_budget'].update(gpu_hours_fit_inclusive=hours,repeated_speech_audio_seconds_cap=seconds)
    exp['generation_budget']['accelerator_bound']={'status':'MEASURED-SYNTHETIC-ALLOWANCE','card_hours':hours,
        'limitation':'Measured synthetic serving allowance; real Fold A cost must be reported.'}
    exp['generation_budget']['definition']=f'Full1080-row c17:36720 generation/18360 FIT ceilings;{seconds}audio seconds;{hours}card-hours. Ceilings are not targets.'
    cell['allowed_execution_stages']=['check','prepare-supply','worker','campaign']
    with tempfile.TemporaryDirectory(prefix='fold-a-register-') as temp:
        stage=Path(temp);shutil.copytree(repo/'configs',stage/'configs');(stage/exp_rel).write_bytes(raw(exp))
        cfg=compose(stage,cell['model_fragment'],cell['dataset_fragment'],cell['baseline_fragment'],evaluation.CELL)
    split=roster['split_identity_hash'];protocol=protocol_hash(cfg.config_hash,cfg.values['arm'],split)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip();now=datetime.now(timezone.utc).isoformat()
    identity={'cell':evaluation.CELL,'experiment':evaluation.CELL,'run_id':evaluation.RUN,'attempt_id':'generation-v1',
        'fit_run_id':evaluation.FIT,'fit_attempt_id':'fit-v1','rows':1080,'split_identity_hash':split,
        'composed_config_hash':cfg.config_hash,'protocol_hash':protocol,
        'model_fragment':cell['model_fragment'],'baseline_fragment':cell['baseline_fragment']}
    spec_path=evaluation.comparator.inputs.confined(source_repo,derivation['parent_binding']['source_spec'])
    source_spec=read(spec_path);source_pkg=spec_path.parent
    source_facts=read(source_pkg/'registered-facts-v1.json')
    measurement=copy.deepcopy(source_facts['stage3_measurement'][source_spec['run_id']])
    measurement['read_roster']=roster['roster_id']
    measurement['confirmation_of']=source_spec['run_id']
    bound={'status':'MEASURED-SYNTHETIC-ALLOWANCE','measurement':binding,'proof_inputs':proof_pins,
        'caps':{'arm':{'generation_calls':36720,'fit_endpoint_calls':18360,'speech_audio_seconds':seconds,'card_hours':hours}},
        'limitation':'Operational ceiling, not measured task throughput or benefit.'}
    outputs={exp_rel:raw(exp)}
    def document(name,value):
        rel=PKG+'/'+name;outputs[rel]=raw(value)
        return {'path':rel,'sha256':hashlib.sha256(outputs[rel]).hexdigest()}
    protocol_pin=document('protocol-registered-v1.json',{'schema':'saea-registered-protocol-v1','cells':{'arm':identity},
        'registered_at_utc':now,'accelerator_bound':bound,'proof':proof})
    facts_pin=document('registered-facts-v1.json',{'schema':'saea-registered-facts-v1','identity':{'arm':identity},
        'stage3_measurement':{evaluation.RUN:measurement},'study_commit_at_registration':head,'registered_at_utc':now})
    plans={}
    for kind,run,attempt,calls in [('generation',evaluation.RUN,'generation-v1',36720),('fit',evaluation.FIT,'fit-v1',18360)]:
        plan={'cell':evaluation.CELL,'experiment':evaluation.CELL,'composed_config_hash':cfg.config_hash,
            'expected_session_id':binding['session_id'],'preparation_base_commit':head,'registered_at_utc':now,
            'accelerator_bound':bound,'family_budget':exp['family_budget'],
            'protocol_document':{'relpath':protocol_pin['path'],'sha256':protocol_pin['sha256']},
            'prepared_phase_supply':{'relpath':f'prepared/{evaluation.RUN}/phase-supply-v1.json','sha256':'MINTED_AT_THE_FLIP'},
            'status':'REGISTERED-BEFORE-CONTACT; HELD until mint, own supply and preflight',
            'execution_plan':dict(exp['execution_plan'],schema='saea-execution-plan-v1',study='speech-aware-evidence-acquisition',
                run_id=run,attempt_id=attempt,carrier_lock_key='slurp',protocol_hash=protocol,split_identity_hash=split,planned_model_calls=calls),
            'serving':{'session_id':binding['session_id'],'model_fragment':cell['model_fragment'],'binding':binding},
            'execution_route':'Shared full campaign and complete official score' if kind=='generation' else 'Inline FIT; no generation'}
        plans[kind]=document(kind+'-plan-registered-v1.json',plan)
    recipe=document('analysis-recipe-v1.json',{'schema':'saea-agent-port-delivered-score-read-v1','read_run_id':READ,
        'quality_read':True,'new_scoring_jobs':0,'score_surface':'sealed-final-delivery','experiments':[evaluation.CELL],
        'identity':{k:identity[k] for k in ('run_id','attempt_id','rows','split_identity_hash')},'authority':AUTH,'claims_benefit':False})
    document('root-admission-draft-v1.json',{'schema':'saea-campaign-decision-v1','status':'DRAFT-NOT-MINTED',
        'decision_id':evaluation.DECISION,'eligible_run_ids':[evaluation.RUN,evaluation.FIT],
        'eligible_attempt_ids':['generation-v1','fit-v1'],'authorization_basis':AUTH,'owner':'root','recorded_at_utc':None,
        'decision':'Evaluate the unchanged current A0 on TRAIN Fold A before D0/A0 lock and TFRL.',
        'antecedents':{'protocol':protocol_pin,'registered_facts':facts_pin,'analysis_recipe':recipe,'plans':plans,
                      'registration_base_commit':head,'derivation':{'path':PKG+'/derivation.json','sha256':sha(repo/PKG/'derivation.json')}},
        'budget':{**exp['family_budget'],'accelerator_bound':bound},
        'population':{'count':1080,'split_role':'discovery','split_identity_hash':split,'test_contact_authorized':False},
        'service':{'expected_session_id':binding['session_id'],'no_child_start_stop':True},
        'not_authorized':['test contact','TFRL promotion','automatic replay','cross-fold supply equality assumption']})
    lines=[]
    for kind,run,calls in [('generation',evaluation.RUN,36720),('fit',evaluation.FIT,18360),('read',READ,0)]:
        pin=recipe if kind=='read' else plans[kind]
        fields=[now[:10],run,'model-free-check' if kind=='read' else 'bounded-discovery-probe',
            'Current A0 Fold A '+kind,'none' if kind=='read' else cell['model_fragment'], 'slurp TRAIN Fold A 1080','discovery',split,
            'no',str(calls),'0' if kind=='read' else str(seconds),'0' if kind=='read' else str(hours),
            'Previously exposed TRAIN Fold A; no test contact.',pin['path'],
            f"REGISTERED BEFORE CONTACT; root countersigned under {AUTH}. Plan SHA256 {pin['sha256']}; no automatic replay."]
        lines.append('| '+' | '.join(fields)+' |')
    outputs['docs/exposure-ledger.md']=ledger+('\n## Current A0 Fold A evaluation\n\n'
        '| date | run/experiment id | execution profile | scope | model/tool | speech carrier + split | split role | split identity hash | consumed | calls | speech audio seconds | gpu hours | inherited exposure | artifacts | notes |\n'
        +'| '+' | '.join(['---']*15)+' |\n'+'\n'.join(lines)+'\n').encode()
    outputs[PKG+'-preregistration.md']=('# Current A0 Fold A registration\n\nIndependent 1080-row TRAIN Fold A evaluation under '+AUTH+
        '. Held until mint, own supply and preflight. Complete final delivery score; no benefit claim.\n\n```json\n'+
        json.dumps({'stage3_measurement':{evaluation.RUN:measurement}},indent=2)+'\n```\n').encode()
    if ledger_path.read_bytes()!=ledger:raise ValueError('ledger changed while staging')
    for tree in (repo,source_repo):
        if subprocess.check_output(['git','status','--porcelain'],cwd=tree).strip():raise ValueError('source changed while staging')
    if dry_run:return {'dry_run':True,'model_contacts':0,'files':len(outputs),'audio_seconds':seconds,'card_hours':hours}
    archive.mkdir(parents=True,exist_ok=False)
    publish_new(archive/'intent.json',raw({'source_commit':head,'outputs':{n:hashlib.sha256(b).hexdigest() for n,b in outputs.items()},'model_contacts':0}))
    for n,blob in outputs.items():
        target=repo/n
        if target.exists():publish_new(archive/'before'/n,target.read_bytes())
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(blob)
    publish_new(archive/'complete.json',raw({'model_contacts':0,'held':True,'audio_seconds':seconds,'card_hours':hours}))
    return {'held':True,'model_contacts':0,'audio_seconds':seconds,'card_hours':hours}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('repo-root','source-repo','data-root','binding','archive'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    print(json.dumps(register(a.repo_root,a.source_repo,a.data_root,a.binding,a.archive,a.dry_run),sort_keys=True))


if __name__=='__main__':main()

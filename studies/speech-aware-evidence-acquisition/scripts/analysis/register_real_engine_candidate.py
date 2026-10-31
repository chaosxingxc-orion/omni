"""Prepare an independent real-data engine candidate using shared runtime contracts.

No model or tokenizer contacts. Registration stays held until live serving receipt,
prepared supply, and the owner's bounded admission are bound and checked.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))


def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')


def prepare(repo,data,roster,source,profile_id):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.deployment import load_deployment_profile
    from speech_aware_evidence_acquisition.core.flight.factory import build_flight_runtime
    from speech_aware_evidence_acquisition.core.flight.campaign import load_campaign_cells
    from speech_aware_evidence_acquisition.core.config import compose,protocol_hash
    assert_execution_environment()
    profile=load_deployment_profile(profile_id,repo)
    if profile_id not in {f'qwen3-omni-real500-{topology}-s{seq}-b2048-v1' for topology,seq in [('2xtp4',96),('2xtp4',128),('1xtp8',256),('1xtp8',384)]}:raise ValueError('unapproved topology')
    receipt,view=read(roster/'roster.json'),read(roster/'dataset.json')
    from prepare_real_engine_roster import prepare as derive
    expected,expected_view=derive(repo/'configs/dataset/slurp-speech-native-evo-fold-b-1080.json')
    if receipt!=expected or view!=expected_view:raise ValueError('roster differs from fixed real train selection')
    suffix=f'tp{profile.tensor_parallel_size}-s{profile.max_num_seqs}'
    name=f'slurp-real500-{suffix}-v1';run='SAEA-SLURP-REAL500-'+suffix.upper()+'-V1'
    attempt='attempt-real500-'+suffix+'-v1';session='serving-real500-'+suffix+'-v1'
    pkg=repo/'docs/readiness'/name
    if pkg.exists():raise ValueError('candidate already exists; inspect, do not overwrite')
    for key in ('runs','attempts','prepared'):
        if (data/key/run).exists() or (data/key/(run+'-FIT')).exists():raise ValueError('spend identity already used')
    baseline=copy.deepcopy(read(source/'configs/baseline/slurp-real-engine-500-c2-v1.json'))
    baseline.pop('prepared_phase_supply',None)
    baseline['native_contact_fit'].update(run_id=run+'-FIT',attempt_id=attempt+'-fit',contact_cap=1000)
    model=copy.deepcopy(read(repo/'configs/model'/f'{profile.model_fragment}.json'))
    model['runtime_receipt']=f'receipts/runtime-real500-{suffix}-v1.json'
    model['timeout_seconds']=1800
    # Deployment profile continues to declare the same content flags. The new
    # fragment records only receipt location and a uniform client deadline.
    model_name=f'bf16-thinking-real500-{suffix}-v1'
    exp=copy.deepcopy(read(source/'configs/experiment/slurp-real-engine-500-c2-v1.json'))
    exp['stage3_algorithm']={'id':'none','note':'Engineering topology measurement of the fixed two-contact baseline; no training-free RL operator or benefit claim.','operators_used':[]}
    exp['stage3_measurement']={'measurement_design':'material-source','rows':500,'expected_effect_pp':0,'claims_benefit':False,'primary_metric_key':'slu','tuning_roster':'none','read_roster':'SLURP-REAL-ENGINE-500-V1','note':'Fixed-system engineering data collection. Quality is descriptive, no algorithm superiority claim; topology is the operational variable.'}
    exp['execution_plan'].update(planned_gpu_hours=16.0,planned_model_calls=2000,planned_speech_audio_seconds=math.ceil(4*view['audio_seconds_total']))
    exp.update(experiment_id=run,service_session=session,purpose='Owner-authorized fixed real500 topology measurement; actual perception feeds solver; no synthetic replay')
    cell,=exp['cells'];cell.update(cell_id=name,run_id=run,attempt_id=attempt,model_fragment=model_name,baseline_fragment=name)
    cell['allowed_execution_stages']=['check','prepare-supply','worker','campaign']
    cell['readiness']={'state':'held','reason':'Live serving receipt and admission pending','requirements':[]}
    dataset=cell['dataset_fragment']
    path=repo/'configs/dataset'/f'{dataset}.json'
    if path.exists():
        if read(path)!=view:raise ValueError('existing dataset differs')
    else:write(path,view)
    write(repo/'configs/model'/f'{model_name}.json',model)
    write(repo/'configs/baseline'/f'{name}.json',baseline)
    write(repo/'configs/experiment'/f'{name}.json',exp)
    load_campaign_cells(repo,name)
    lock=repo.parents[1]/'docs/datasets.lock.json'
    runtime=build_flight_runtime(repo,data,model=model_name,dataset=dataset,baseline=name,experiment=name,
        run_id=run,attempt_id=attempt,runtime_root=data/'work/scratch/runtime-root',lock_path=lock)
    supply=data/'prepared'/run/'phase-supply-v1.json'
    runtime.prepare_phase_supply(supply,max_workers=16)
    pin={'relpath':str(supply.relative_to(data)),'sha256':sha(supply)}
    baseline['prepared_phase_supply']=pin
    (repo/'configs/baseline'/f'{name}.json').write_text(json.dumps(baseline,indent=2,sort_keys=True)+'\n')
    cell['readiness']['requirements']=[{'kind':'preparation',**pin}]
    (repo/'configs/experiment'/f'{name}.json').write_text(json.dumps(exp,indent=2,sort_keys=True)+'\n')
    cfg=compose(repo,model_name,dataset,name,name)
    from speech_aware_evidence_acquisition.contracts import ExecutionPlan
    ExecutionPlan(run_id=run,attempt_id=attempt,carrier_lock_key='slurp',split_identity_hash=view['split_identity_hash'],protocol_hash=protocol_hash(cfg.config_hash,cfg.values['arm'],view['split_identity_hash']),**exp['execution_plan']).validate(source_root=repo)
    identity={'cell':name,'experiment':name,'run_id':run,'attempt_id':attempt,'fit_run_id':run+'-FIT','fit_attempt_id':attempt+'-fit','rows':500,'composed_config_hash':cfg.config_hash,'protocol_hash':protocol_hash(cfg.config_hash,cfg.values['arm'],view['split_identity_hash']),'split_identity_hash':view['split_identity_hash']}
    spec={**identity,'model_fragment':model_name,'baseline':name,'service_session':session,'profile':profile_id,'runtime_receipt':model['runtime_receipt'],'decision_id':name+'-owner-admission','decision_relpath':f'coordination/{name}/root-admission-v1.json','authority':'docs/decisions/2026-10-31-owner-real500-topology-measurement.md','sample_source_sha256':sha(roster/'roster.json'),'scientific_readiness':'held','new_model_calls':0}
    write(pkg/'candidate.json',spec)
    protocol={'schema':'saea-real-engine-protocol-v1','cells':{'arm':identity},'profile':profile_id,'profile_sha256':profile.sha256,'inference':'Real audio/context, fresh P output supplied to fresh S; each roster row once per candidate','selection':receipt['selection'],'claims_benefit':False,'client_timeout_seconds':1800,'budget_basis':'16 allocated card-hours is a bounded allowance, not a measured ETA; 2000 contacts reserves fallback above normal 1000','prepared_phase_supply':pin}
    write(pkg/'protocol-registered-v1.json',protocol)
    rows=[]
    for kind,rid,aid in [('generation',run,attempt),('fit',run+'-FIT',attempt+'-fit')]:
        plan={'schema':f'saea-registered-{kind}-plan-v1','plan_kind':kind,'cell':name,'experiment':name,'composed_config_hash':cfg.config_hash,'expected_session_id':session,'execution_plan':{**exp['execution_plan'],'schema':'saea-execution-plan-v1','study':'speech-aware-evidence-acquisition','run_id':rid,'attempt_id':aid,'carrier_lock_key':'slurp','protocol_hash':identity['protocol_hash'],'split_identity_hash':view['split_identity_hash']},'prepared_phase_supply':pin,'protocol_document':{'relpath':str((pkg/'protocol-registered-v1.json').relative_to(repo)),'sha256':sha(pkg/'protocol-registered-v1.json')},'status':'REGISTERED BEFORE CONTACT; held for actual runtime receipt and preflight'}
        if kind=='fit':plan['execution_plan']['planned_model_calls']=1000
        plan_path=pkg/f'{kind}-plan-registered-v1.json';write(plan_path,plan)
        fields=['2026-10-31',rid,'bounded-discovery-probe','Real500 topology '+kind,model_name,'slurp TRAIN Fold B 500','discovery',view['split_identity_hash'],'no',str(plan['execution_plan']['planned_model_calls']),str(math.ceil(exp['execution_plan']['planned_speech_audio_seconds'])),'16.0','Historical costs retained; new distinct attempt',str(plan_path.relative_to(repo)),f"Plan SHA256 {sha(plan_path)}; root countersigned under recorded owner amendment, coordinator Codex; owner-authorized real-data measurement; no synthetic precondition; no test tuning"]
        rows.append('| '+' | '.join(fields)+' |')
    recipe={'schema':'saea-real-engine-analysis-v1','read_run_id':run+'-READ','run_ids':{'generation':run,'fit':run+'-FIT','read':run+'-READ'},'rows':500,'scope':'Full500 completion, wall throughput, per-stage timing, calls and token costs; official full-manifest SLURP scoring after seal only','claims_benefit':False,'test_contact_authorized':False,'gold_in_runtime':False,'metrics':['SLU-F1','intent accuracy','wall_seconds_per_completed_row','latency mean/p50/p95','failures','per-card utilization'],'decision':'Compare complete candidates; no fastest-completed subset ranking; no test tuning'}
    recipe_path=pkg/'analysis-recipe-v1.json';write(recipe_path,recipe)
    rows.append('| '+' | '.join(['2026-10-31',run+'-READ','model-free-check','Offline real500 engineering and full-manifest quality read','no model','slurp TRAIN Fold B 500','discovery',view['split_identity_hash'],'no','0','0','0','Generation/FIT costs retained',str(recipe_path.relative_to(repo)),'root countersigned under recorded owner amendment, coordinator Codex; no runtime gold; read after seal'])+' |')
    ledger=repo/'docs/exposure-ledger.md'
    if any('| '+rid+' |' in ledger.read_text() for rid in (run,run+'-FIT')):raise ValueError('duplicate ledger entry')
    with ledger.open('a') as f:f.write('\n## '+name+'\n\n| date | run/experiment id | execution profile | scope | model/tool | speech carrier + split | split role | split identity hash | consumed | calls | speech audio seconds | gpu hours | inherited exposure | artifacts | notes |\n| '+' | '.join(['---']*15)+' |\n'+'\n'.join(rows)+'\n')
    (repo/'docs/readiness'/f'{name}-preregistration.md').write_text('# Real500 topology measurement\n\nFixed 500 real train rows; actual shared c2 context/agent pipeline, no static response replay. Separate generation and inline FIT accounting. Normal1000 generation contacts, cap2000 including fallback;16 allocated card-hours is an allowance, not an ETA. Preserve unknown in-flight requests and all historical costs; no blind retry. Score only after sealing, outside runtime. Live service registration, owner admission, prepared identity and shared preflight must all pass before dispatch.\n')
    # Reconstruct after pin insertion: validates production identity, not a copy-only check.
    fresh=build_flight_runtime(repo,data,model=model_name,dataset=dataset,baseline=name,experiment=name,run_id=run,attempt_id=attempt,runtime_root=data/'work/scratch/runtime-root',lock_path=lock)
    checks=fresh.verify_prepared_artifacts()
    if not checks or not all(x['identity_ok'] and x.get('seed_ok',True) for x in checks):raise ValueError(f'prepared identity failed: {checks}')
    write(pkg/'preparation-check.json',{'rows':500,'checks':checks,'new_model_calls':0,'readiness':'held'})
    return {'candidate':name,'rows':500,'prepared_identity':True,'new_model_calls':0,'readiness':'held'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('repo','data','roster','source'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--profile',required=True);a=p.parse_args()
    print(json.dumps(prepare(a.repo,a.data,a.roster,a.source,a.profile)))

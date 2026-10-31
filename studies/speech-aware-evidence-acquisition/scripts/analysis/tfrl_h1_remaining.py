"""Derive and register the exact context-repaired H1 complement (310 real rows).

The 22 affected completed rows remain diagnostics; three unchanged rows retain
separate source custody. This flight never injects historical model responses.
"""
import argparse,copy,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/analysis'),str(ROOT/'scripts/flights/nx')]
from register_real_engine_candidate import read,sha,write
import tfrl_current_generation as generation
import tfrl_v2_sampler as sampler
from tfrl_current_seed_cells import discard_synthetic_cost_forecasts
from tfrl_h1_recovery_custody import NAME as PARENT, CHECKPOINT_SHA,partition
NAME='slurp-tfrl-current-seed-i1-h1-context-repair-v1'
RUN='SAEA-SLURP-TFRL-CURRENT-SEED-I1-H1-CONTEXT-REPAIR-V1'
ATTEMPT='generation-v1'


def sources(repo,data,custody_pin):
    from speech_aware_evidence_acquisition.core.row_checkpoint import read_checkpoint
    path=(data/custody_pin['relpath']).resolve()
    if not path.is_relative_to(data.resolve()) or sha(path)!=custody_pin['sha256']:raise ValueError('custody pin differs')
    doc=read(path);roster=read(repo/doc['parent_roster']['relpath'])
    if sha(repo/doc['parent_roster']['relpath'])!=doc['parent_roster']['sha256']:raise ValueError('parent roster differs')
    cp=read_checkpoint(data/doc['source_checkpoint']['relpath'],expected_sha256=CHECKPOINT_SHA)
    if cp.torn_tail_bytes or any(r.keep_defect() for r in cp.rows):raise ValueError('checkpoint invalid')
    parts=partition([r['sample_id'] for r in roster['items']],[r.output_rows[0] for r in cp.rows])
    if parts!=doc['partition'] or (len(parts['reusable_ids']),len(parts['remaining_ids']))!=(3,310):raise ValueError('source partition changed')
    for pin in doc['retained_fit_receipts']:
        if sha(data/pin['relpath'])!=pin['sha256']:raise ValueError('retained FIT changed')
    return doc,parts


def derive_documents(original,experiment,roster,plan):
    base,exp=copy.deepcopy(original),copy.deepcopy(experiment)
    ids=[r['sample_id'] for r in roster['items']]
    if len(ids)!=310 or len(set(ids))!=310:raise ValueError('exact310 required')
    if base['native_episode']['consumption_module']['columns']!=sampler.execution_columns(plan):raise ValueError('K4 method changed')
    base.pop('prepared_phase_supply',None)
    base['native_contact_fit'].update(run_id=RUN+'-FIT',attempt_id='fit-v1',contact_cap=2170)
    cell,=exp['cells'];cell.update(cell_id=NAME,baseline_fragment=NAME,run_id=RUN,attempt_id=ATTEMPT,
        readiness={'state':'held','requirements':[],'reason':'Context repair complement; source-bound registration, supply and admission pending'})
    exp['experiment_id']=RUN
    exp['population'].update(count=310,roster_receipt=f'docs/readiness/{NAME}/remaining-roster.json',split_identity_hash=roster['split_identity_hash'])
    exp['populations'][0]['ids']=ids
    exp['execution_plan'].update(planned_model_calls=4340,planned_speech_audio_seconds=math.ceil(sum(r['audio_seconds'] for r in roster['items'])*14))
    exp['stage3_measurement'].update(rows=310,read_roster=roster['roster_id'],claims_benefit=False)
    return base,exp


def stage(repo,data,custody):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    assert_execution_environment();head=require_clean_study_repo(repo)
    pin={'relpath':str(custody.relative_to(data)),'sha256':sha(custody)};doc,parts=sources(repo,data,pin)
    pkg=repo/'docs/readiness'/NAME
    if pkg.exists() or any((data/k/RUN).exists() for k in ('runs','attempts','prepared')):raise ValueError('fresh recovery identity required')
    roster=read(custody.parent/'remaining-roster.json');raw=(json.dumps(roster,indent=2,sort_keys=True)+'\n').encode()
    import hashlib
    plan=sampler.build_plan(roster,'R01C-H1',1,roster_file_sha256=hashlib.sha256(raw).hexdigest(),execution_run_id=RUN,execution_attempt_id=ATTEMPT)
    discard_synthetic_cost_forecasts(plan)
    base,exp=derive_documents(read(repo/f'configs/baseline/{PARENT}.json'),read(repo/f'configs/experiment/{PARENT}.json'),roster,plan)
    pkg.mkdir(parents=True);(pkg/'remaining-roster.json').write_bytes(raw)
    write(pkg/'sampling-plan.json',plan)
    write(pkg/'source-custody-pin.json',pin)
    write(repo/f'configs/baseline/{NAME}.json',base);write(repo/f'configs/experiment/{NAME}.json',exp)
    write(pkg/'stage.json',{'source_commit':head,'custody':pin,'rows':310,'normal_calls':2170,'new_model_calls':0,'readiness':'held'})
    return {'rows':310,'new_model_calls':0,'requires':['commit','own supply','register','repin','admit','preflight']}


def validate(repo,data,name,spec,exp,view):
    from speech_aware_evidence_acquisition.data.splits import prefixed,split_identity_hash
    from speech_aware_evidence_acquisition.rails import consumption_module as cm
    from speech_aware_evidence_acquisition.rails.adapters.slurp_module_hooks import carrier_module_hooks
    if name!=NAME or spec['run_id']!=RUN or spec['attempt_id']!=ATTEMPT or spec['rows']!=310:raise ValueError('unexpected recovery')
    doc,parts=sources(repo,data,spec['source_custody']);ids=parts['remaining_ids']
    def pinned(key):
        pin=spec[key];p=(repo/pin['relpath']).resolve()
        if not p.is_relative_to(repo.resolve()) or sha(p)!=pin['sha256']:raise ValueError('recovery input pin differs')
        return read(p)
    roster=pinned('roster');plan=pinned('sampling_plan');cell,=exp['cells']
    if ([r['sample_id'] for r in roster['items']]!=ids or exp['populations'][0]['ids']!=ids
        or exp['population']['count']!=310 or roster['split_identity_hash']!=split_identity_hash(prefixed('slurp',ids))
        or spec['split_identity_hash']!=roster['split_identity_hash'] or pinned('dataset')!=view
        or view['count']!=625 or not set(ids)<={r['sample_id'] for r in view['items']}
        or [r['row_id'] for r in plan['row_plan']]!=ids or plan['run_id']!=RUN or plan['attempt_id']!=ATTEMPT
        or plan['roster']['file_sha256']!=spec['roster']['sha256']):raise ValueError('exact source-bound complement differs')
    if (spec['profile']!=generation.PROFILE or spec['service_session']!=generation.SESSION
        or spec['gain_lock']!={'relpath':generation.LOCK,'sha256':generation.LOCK_SHA}
        or sha(repo/generation.LOCK)!=generation.LOCK_SHA):raise ValueError('baseline prerequisite changed')
    base=read(repo/f'configs/baseline/{NAME}.json');old=read(repo/f'configs/baseline/{PARENT}.json')
    for k in ('prepared_phase_supply','native_contact_fit'):
        base=copy.deepcopy(base) if k=='prepared_phase_supply' else base
        base.pop(k,None);old.pop(k,None)
    if base!=old:raise ValueError('recovery method differs')
    actual=read(repo/f'configs/baseline/{NAME}.json');fit=actual['native_contact_fit']
    if (fit!={'run_id':RUN+'-FIT','attempt_id':'fit-v1','contact_cap':2170} and
        any(fit.get(k)!=v for k,v in {'run_id':RUN+'-FIT','attempt_id':'fit-v1','contact_cap':2170}.items())):raise ValueError('FIT budget changed')
    if exp['execution_plan']['planned_model_calls']!=4340:raise ValueError('generation budget changed')
    block=actual['native_episode']['consumption_module'];cm.verify_declared_module(block,actual,repo_root=repo)
    if block['columns']!=sampler.execution_columns(plan) or cm.compile_module(block,carrier_module_hooks(repo_root=repo),repo_root=repo).program_ceiling!=7:raise ValueError('seed-only seven contacts changed')
    return 310


def register(repo,data):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment,ExecutionPlan
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    from speech_aware_evidence_acquisition.core.config import compose,protocol_hash
    assert_execution_environment();require_clean_study_repo(repo)
    pkg=repo/'docs/readiness'/NAME;exp=read(repo/f'configs/experiment/{NAME}.json');c,=exp['cells'];base=read(repo/f'configs/baseline/{NAME}.json')
    ledger=repo/'docs/exposure-ledger.md';text=ledger.read_text()
    if c['readiness']['state']!='held' or (pkg/'candidate.json').exists():raise ValueError('unregistered held recovery required')
    for rid in (RUN,RUN+'-FIT'):
        if '| '+rid+' |' in text or any((data/k/rid).exists() for k in ('runs','attempts')):raise ValueError('existing spend or registration')
    cfg=compose(repo,c['model_fragment'],c['dataset_fragment'],NAME,NAME);split=exp['population']['split_identity_hash'];ph=protocol_hash(cfg.config_hash,cfg.values['arm'],split)
    identity={'cell':NAME,'experiment':NAME,'run_id':RUN,'attempt_id':ATTEMPT,'fit_run_id':RUN+'-FIT','fit_attempt_id':'fit-v1','rows':310,'composed_config_hash':cfg.config_hash,'protocol_hash':ph,'split_identity_hash':split}
    def pin(path):return {'relpath':str(path.relative_to(repo)),'sha256':sha(path)}
    spec={**identity,'workload_kind':'tfrl-context-repair-complement','model_fragment':generation.MODEL,'baseline':NAME,'service_session':generation.SESSION,'profile':generation.PROFILE,'runtime_receipt':generation.RECEIPT,'authority':generation.AUTH,'decision_id':NAME+'-owner-admission','decision_relpath':f'coordination/{NAME}/root-admission-v1.json','gain_lock':{'relpath':generation.LOCK,'sha256':generation.LOCK_SHA},'source_custody':read(pkg/'source-custody-pin.json'),'roster':pin(pkg/'remaining-roster.json'),'sampling_plan':pin(pkg/'sampling-plan.json'),'dataset':pin(repo/f"configs/dataset/{c['dataset_fragment']}.json")}
    validate(repo,data,NAME,spec,exp,read(repo/spec['dataset']['relpath']));write(pkg/'candidate.json',spec)
    pp=pkg/'protocol-registered-v1.json';write(pp,{'schema':'saea-tfrl-context-repair-generation-v1','cells':{'arm':identity},'source_custody':spec['source_custody'],'roster':spec['roster'],'sampling_plan':spec['sampling_plan'],'prepared_phase_supply':base['prepared_phase_supply'],'normal_generation_calls':2170,'fallback_inclusive_cap':4340,'claims_benefit':False})
    entries=[]
    for kind,rid,aid,calls in [('generation',RUN,ATTEMPT,4340),('fit',RUN+'-FIT','fit-v1',2170)]:
        budget={**exp['execution_plan'],'planned_model_calls':calls}
        ExecutionPlan(run_id=rid,attempt_id=aid,carrier_lock_key='slurp',split_identity_hash=split,protocol_hash=ph,**budget).validate(source_root=repo)
        plan={'schema':f'saea-registered-{kind}-plan-v1','plan_kind':kind,'cell':NAME,'experiment':NAME,'composed_config_hash':cfg.config_hash,'expected_session_id':generation.SESSION,'execution_plan':{**budget,'schema':'saea-execution-plan-v1','study':'speech-aware-evidence-acquisition','run_id':rid,'attempt_id':aid,'carrier_lock_key':'slurp','split_identity_hash':split,'protocol_hash':ph},'prepared_phase_supply':base['prepared_phase_supply'],'protocol_document':pin(pp),'status':'Registered before contact; held pending admission and full preflight'}
        path=pkg/f'{kind}-plan-registered-v1.json';write(path,plan)
        entries.append(['2026-10-31',rid,'bounded-discovery-probe','Context-repaired exact H1 complement '+kind,generation.MODEL,'slurp TRAIN construction H1 remaining310','discovery',split,'no',str(calls),str(budget['planned_speech_audio_seconds']),str(budget['planned_gpu_hours']),'All175 original generation calls and FIT costs retained;22 affected rows superseded',str(path.relative_to(repo)),f'Plan SHA256 {sha(path)}; root countersigned under {generation.AUTH}, coordinator Codex; no score-based selection; no test tuning'])
    ledger.write_text(text+'\n## H1 context-repaired complement\n\n| date | run/experiment id | execution profile | scope | model/tool | speech carrier + split | split role | split identity hash | consumed | calls | speech audio seconds | gpu hours | inherited exposure | artifacts | notes |\n| '+' | '.join(['---']*15)+' |\n'+'\n'.join('| '+' | '.join(r)+' |' for r in entries)+'\n')
    (repo/'docs/readiness'/f'{NAME}-preregistration.md').write_text('# H1 context repair\n\nGenerate exactly310 real rows:288 unfinished and22 affected by erroneous context compaction. Preserve all25 old rows and costs; only3 unchanged rows are candidates for source-bound reuse after offline replay. No historical response injection. Keep original K4 seeds, prompts, full313 denominator and frozen selected resident. Shared admission and preflight required.\n')
    return {'rows':310,'registered':True,'new_model_calls':0}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['stage','register'])
    for k in ('repo','data'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--custody',type=Path);a=p.parse_args()
    print(json.dumps(stage(a.repo,a.data,a.custody) if a.mode=='stage' else register(a.repo,a.data)))

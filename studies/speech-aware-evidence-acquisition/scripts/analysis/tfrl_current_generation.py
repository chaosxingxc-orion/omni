"""Stage current K4 generation against the measured resident, then register it.

Stage changes are committed before the shared supply repinner is invoked.
No service deployment, readiness promotion or inference occurs here.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts/analysis'),str(REPO/'scripts/flights'),str(REPO/'scripts/flights/nx')]
from register_real_engine_candidate import read,sha,write
AUTH='docs/decisions/2026-10-31-owner-lock-baseline-and-cost-repair.md'
LOCK='docs/results/2026-10-31-a0-full1080-baseline-lock.json'
LOCK_SHA='6b76ec8fcc232c2e1b33a9c3221bed93a437f180ab8102ad9706082f5b81ae69'
MODEL='bf16-thinking-real500-tp4-s128-v1'
SESSION='serving-a0-remaining353-selected-resident-v1'
PROFILE='qwen3-omni-real500-2xtp4-s128-b2048-v1'
RECEIPT='receipts/runtime-a0-remaining353-selected-resident-v1.json'


def stage_documents(exp,old_model,new_model):
    old=copy.deepcopy(old_model);new=copy.deepcopy(new_model)
    old.pop('runtime_receipt');new.pop('runtime_receipt')
    if old!=new:raise ValueError('resident binding changes scientific decoding')
    doc=copy.deepcopy(exp);cell,=doc['cells']
    if cell['readiness']['state']!='held' or cell['allowed_execution_stages']!=['check']:
        raise ValueError('unspent held successor required')
    cell['model_fragment']=MODEL
    cell['allowed_execution_stages']=['check','prepare-supply','worker','campaign']
    doc['service_session']=SESSION
    doc['stage3_algorithm']={'id':'shared-rollout','operators_used':['rollout','scoring'],
        'note':'Real per-row seed-only K4 construction; downstream registered TFRL credit consumes these complete trajectories.'}
    return doc


def validate_half(repo, name, spec, exp, view, *, data=None, iteration=1):
    """Validate a fixed real construction half without relaxing full-view checks."""
    import math
    from tfrl_v2_sampler import execution_columns
    from speech_aware_evidence_acquisition.rails import consumption_module as cm
    from speech_aware_evidence_acquisition.rails.adapters.slurp_module_hooks import carrier_module_hooks
    if type(iteration) is not int or iteration not in (1,2,3):raise ValueError('iteration1..3 required')
    prefix='seed' if iteration==1 else 'bank'
    half = name.removeprefix(f'slurp-tfrl-current-{prefix}-i{iteration}-').removesuffix('-v1')
    if half not in ('h1','h2') or name!=f'slurp-tfrl-current-{prefix}-i{iteration}-{half}-v1':
        raise ValueError('unexpected real construction half')
    n = 313 if half == 'h1' else 312
    c, = exp['cells']
    pop, = exp['populations']
    summary = exp['population']
    def pinned(key):
        pin = spec[key]; path = (repo / pin['relpath']).resolve()
        if not path.is_relative_to(repo.resolve()) or sha(path) != pin['sha256']:
            raise ValueError('real half source pin differs: ' + key)
        return read(path)
    lock = pinned('gain_lock')
    if spec['gain_lock'] != {'relpath': LOCK, 'sha256': LOCK_SHA} or lock['status'] != 'locked_full_train_fold_b':
        raise ValueError('full A0 gain prerequisite differs')
    roster = pinned('roster'); plan = pinned('sampling_plan')
    if pinned('dataset') != view:
        raise ValueError('real dataset pin differs')
    ids = [r['sample_id'] for r in roster['items']]
    view_ids = [r['sample_id'] for r in view['items']]
    if (len(ids) != n or len(set(ids)) != n or spec['rows'] != n or summary['count'] != n
            or len(view_ids) != 625 or view['count'] != 625 or len(set(view_ids)) != 625
            or not set(ids).issubset(view_ids) or pop['ids'] != ids
            or pop['dataset_fragment'] != c['dataset_fragment']
            or summary['dataset_fragment'] != c['dataset_fragment']
            or summary['roster_receipt'] != spec['roster']['relpath']
            or roster['split_identity_hash'] != summary['split_identity_hash']
            or summary['split_identity_hash'] != spec['split_identity_hash']):
        raise ValueError('exact real half population differs')
    by_id = {r['sample_id']: r for r in view['items']}
    for row in roster['items']:
        source = by_id[row['sample_id']]
        if (not row['sample_id'].startswith('slurp:train:')
                or '/slurp_real/' not in source['media_relpath']
                or row['audio_seconds'] != source['audio_seconds']
                or not source['audio_sha256']):
            raise ValueError('real training audio binding differs')
    run = f'SAEA-SLURP-TFRL-CURRENT-{prefix.upper()}-I{iteration}-{half.upper()}-V1'
    if (spec['run_id'] != run or plan['run_id'] != run or plan['attempt_id'] != c['attempt_id']
            or plan['roster']['file_sha256'] != spec['roster']['sha256']
            or plan['roster']['rows'] != n or plan['roster']['split_identity_hash'] != spec['split_identity_hash']
            or plan['half'] != 'R01C-' + half.upper() or plan['iteration'] != iteration
            or [r['row_id'] for r in plan['row_plan']] != ids):
        raise ValueError('sampling plan identity differs')
    base = read(repo / f'configs/baseline/{c["baseline_fragment"]}.json')
    block = base['native_episode']['consumption_module']
    if base['policy'] != 'phase-native-module-v1-c7' or block['columns'] != execution_columns(plan):
        raise ValueError('actual program differs from seed-only K4 plan')
    cm.verify_declared_module(block, base, repo_root=repo)
    compiled = cm.compile_module(block, carrier_module_hooks(repo_root=repo), repo_root=repo)
    if compiled.program_ceiling != 7:
        raise ValueError('seven-contact program required')
    if iteration>1:
        if data is None or spec.get('iteration')!=iteration:
            raise ValueError('banked workload requires data root and exact iteration')
        from speech_aware_evidence_acquisition.rails import tfrl_v2_consumption as bank
        from speech_aware_evidence_acquisition.rails.tfrl_supply_allocation import KEY,declaration,validate
        decl=bank.declaration_of_values(base)
        if decl is None or decl.iteration!=iteration-1 or decl.consumer!='R01C-'+half.upper() or decl.bank_kind!='half':
            raise ValueError('previous opposite-half bank declaration required')
        loaded=bank.verify_for_flight(base,data,split_identity_hash=spec['split_identity_hash'],consuming_row_ids=ids)
        allocation=validate(block.get(KEY))
        if (allocation!=declaration(block['decoding_frame']['max_supplied_items'])
                or plan['banks']['supply_allocation']!=allocation
                or plan['banks']['supply_bound_items']!=allocation['candidate_max_items']
                or plan['banks']['bank_receipt_sha256']!=loaded.receipt_sha256):
            raise ValueError('planned bank or supply allocation differs')
        from tfrl_current_sources import pinned as data_pin
        from tfrl_current_next_plan import first_construction_decision
        decision=read(data_pin(data,spec['first_construction_decision']))
        gain=decision.get('construction_gain_pp')
        expected=first_construction_decision({'rows':decision.get('rows'),'iteration':1,
            'vote_minus_lock_pp':gain,'threshold_pp':1.0,
            'construction_gate_passed':decision.get('preparation_gate_passed')})
        if (decision.get('schema')!='saea-tfrl-first-construction-decision-v1'
                or expected['decision']!='prepare-held-successor'
                or any(decision.get(k)!=v for k,v in expected.items())
                or decision.get('generation_authorized') is not False):
            raise ValueError('verified first construction pass required')
        # Bind the persisted decision to immutable registered source result pins.
        if len(decision.get('sources',{}))!=2 or set(decision['sources'])!={'h1','h2'}:
            raise ValueError('two first construction source pins required')
        for source in decision['sources'].values():
            data_pin(data,{'relpath':source['complete_relpath'],'sha256':source['complete_sha256']})
        from tfrl_recovered_consumers import POOL
        pooled=read(data_pin(data,{'relpath':f'analysis/{POOL}/complete.json','sha256':decision['pool_sha256']}))
        if pooled.get('sources')!=decision['sources'] or first_construction_decision(pooled)!=expected:
            raise ValueError('first construction decision differs from source-bound pool')
    elif 'tfrl_v2_bank' in block:
        raise ValueError('seed-only workload cannot consume a bank')
    fit = base['native_contact_fit']
    if (exp['execution_plan']['planned_model_calls'] != 14*n or fit['contact_cap'] != 7*n
            or fit['run_id'] != spec['fit_run_id'] or fit['attempt_id'] != spec['fit_attempt_id']
            or exp['execution_plan']['planned_speech_audio_seconds'] < math.ceil(sum(r['audio_seconds'] for r in roster['items'])*14)
            or exp['stage3_algorithm']['id'] != ('shared-rollout' if iteration==1 else 'tfrl-v2')
            or exp['stage3_algorithm']['operators_used'] != (['rollout','scoring'] if iteration==1 else ['rollout','scoring','context_application'])):
        raise ValueError('real half algorithm or budget differs')
    return n


def stage(repo,data,source,binding,out):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    assert_execution_environment();head=require_clean_study_repo(repo)
    if out.exists() or not out.resolve().is_relative_to(data.resolve()/'work/codex'):raise ValueError('fresh owned stage record required')
    if sha(repo/LOCK)!=LOCK_SHA or read(repo/LOCK)['status']!='locked_full_train_fold_b':raise ValueError('verified full A0 gain lock required')
    completed=read(binding)
    for rel,digest in completed['output_hashes'].items():
        if sha(repo/rel)!=digest:raise ValueError('bound preparation changed')
    model=read(source/f'configs/model/{MODEL}.json')
    if model['runtime_receipt']!=RECEIPT:raise ValueError('expected selected resident required')
    pending=[]
    for half in ('h1','h2'):
        name=f'slurp-tfrl-current-seed-i1-{half}-v1';path=repo/f'configs/experiment/{name}.json';exp=read(path);c,=exp['cells']
        if any((data/k/c['run_id']).exists() for k in ('runs','attempts')):raise ValueError('successor already spent')
        pending.append((path,stage_documents(exp,read(repo/f"configs/model/{c['model_fragment']}.json"),model)))
    out.mkdir(parents=True)
    for rel in (f'configs/model/{MODEL}.json','docs/'+RECEIPT):
        p=repo/rel
        if p.exists():
            dest=out/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
        p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((source/rel).read_bytes())
    for p,doc in pending:
        write(out/(p.stem+'.before.json'),read(p));p.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n')
    result={'source_commit':head,'gain_lock_sha256':LOCK_SHA,'binding_sha256':sha(binding),
        'output_hashes':{str(p.relative_to(repo)):sha(p) for p,_ in pending},
        'model_sha256':sha(repo/f'configs/model/{MODEL}.json'),'runtime_receipt_sha256':sha(repo/'docs'/RECEIPT),
        'new_model_calls':0,'ready':False,'remaining':['commit stage','register','shared repin','admit','preflight']}
    write(out/'complete.json',result);return result


def register(repo,data,half,iteration=1,first_decision=None):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment,ExecutionPlan
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    from speech_aware_evidence_acquisition.core.config import compose,protocol_hash
    assert_execution_environment();require_clean_study_repo(repo)
    if half not in ('h1','h2'):raise ValueError('fixed half required')
    if type(iteration) is not int or iteration not in (1,2,3):raise ValueError('iteration1..3 required')
    prefix='seed' if iteration==1 else 'bank'
    if sha(repo/LOCK)!=LOCK_SHA:raise ValueError('gain lock differs')
    name=f'slurp-tfrl-current-{prefix}-i{iteration}-{half}-v1';pkg=repo/'docs/readiness'/name
    exp=read(repo/f'configs/experiment/{name}.json');c,=exp['cells'];base=read(repo/f'configs/baseline/{name}.json')
    run=c['run_id'];fit=base['native_contact_fit'];n=exp['population']['count'];split=exp['population']['split_identity_hash']
    if (c['readiness']['state']!='held' or exp['service_session']!=SESSION or c['model_fragment']!=MODEL
            or exp['stage3_algorithm']['id']!=('shared-rollout' if iteration==1 else 'tfrl-v2') or n!=(313 if half=='h1' else 312)):
        raise ValueError('staged held real half required')
    ledger=repo/'docs/exposure-ledger.md';text=ledger.read_text()
    for rid in (run,fit['run_id']):
        if '| '+rid+' |' in text or any((data/k/rid).exists() for k in ('runs','attempts')):raise ValueError('existing registration or spend')
    if (pkg/'candidate.json').exists():raise ValueError('registration already started')
    cfg=compose(repo,MODEL,c['dataset_fragment'],name,name);ph=protocol_hash(cfg.config_hash,cfg.values['arm'],split)
    identity={'cell':name,'experiment':name,'run_id':run,'attempt_id':c['attempt_id'],'fit_run_id':fit['run_id'],
        'fit_attempt_id':fit['attempt_id'],'rows':n,'composed_config_hash':cfg.config_hash,'protocol_hash':ph,'split_identity_hash':split}
    spec={**identity,'workload_kind':'tfrl-real-seed-half','model_fragment':MODEL,'baseline':name,'service_session':SESSION,
        'profile':PROFILE,'runtime_receipt':RECEIPT,'authority':AUTH,'decision_id':name+'-owner-admission',
        'decision_relpath':f'coordination/{name}/root-admission-v1.json',
        'gain_lock':{'relpath':LOCK,'sha256':LOCK_SHA},
        'dataset':{'relpath':f"configs/dataset/{c['dataset_fragment']}.json",'sha256':sha(repo/f"configs/dataset/{c['dataset_fragment']}.json")},
        'sampling_plan':{'relpath':str((pkg/'sampling-plan.json').relative_to(repo)),'sha256':sha(pkg/'sampling-plan.json')},
        'roster':{'relpath':exp['population']['roster_receipt'],'sha256':sha(repo/exp['population']['roster_receipt'])},
        'generation_program':'One fresh Perception plus four per-row seed-only Solver candidates, aa and lock; 7 normal contacts/row.'}
    view=read(repo/f"configs/dataset/{c['dataset_fragment']}.json")
    if iteration>1:
        if first_decision is None:raise ValueError('verified first decision required')
        spec.update(workload_kind='tfrl-real-banked-half',iteration=iteration,
            first_construction_decision={'relpath':str(first_decision.resolve().relative_to(data.resolve())),
                'sha256':sha(first_decision)},
            generation_program='Fresh P plus four cross-fit banked Solver candidates and unchanged aa/lock;7contacts per row.')
    validate_half(repo,name,spec,exp,view,data=data,iteration=iteration)
    write(pkg/'candidate.json',spec)
    protocol={'schema':'saea-tfrl-current-seed-generation-v1','cells':{'arm':identity},'profile':PROFILE,
        'gain_lock':spec['gain_lock'],'sampling_plan':spec['sampling_plan'],'roster':spec['roster'],
        'prepared_phase_supply':base['prepared_phase_supply'],'normal_generation_calls':7*n,'fallback_inclusive_cap':14*n,
        'budget_basis':'16 allocated card-hours is a conservative cap, not a time forecast. Real trajectory cost is measured.',
        'claims_benefit':False,'remaining':['shared supply repin','existing resident verification','admission','full preflight']}
    if iteration>1:protocol.update(schema='saea-tfrl-current-bank-generation-v1',iteration=iteration,
        first_construction_decision=spec['first_construction_decision'],
        bank_declaration=base['native_episode']['consumption_module']['tfrl_v2_bank'],
        supply_allocation=base['native_episode']['consumption_module']['tfrl_supply_allocation'])
    pp=pkg/'protocol-registered-v1.json';write(pp,protocol)
    rows=[]
    for kind,rid,aid,calls in [('generation',run,c['attempt_id'],14*n),('fit',fit['run_id'],fit['attempt_id'],7*n)]:
        budget={**exp['execution_plan'],'planned_model_calls':calls}
        ExecutionPlan(run_id=rid,attempt_id=aid,carrier_lock_key='slurp',split_identity_hash=split,protocol_hash=ph,**budget).validate(source_root=repo)
        plan={'schema':f'saea-registered-{kind}-plan-v1','plan_kind':kind,'cell':name,'experiment':name,
            'composed_config_hash':cfg.config_hash,'expected_session_id':SESSION,'execution_plan':{**budget,
            'schema':'saea-execution-plan-v1','study':'speech-aware-evidence-acquisition','run_id':rid,'attempt_id':aid,
            'carrier_lock_key':'slurp','split_identity_hash':split,'protocol_hash':ph},
            'prepared_phase_supply':base['prepared_phase_supply'],'protocol_document':{'relpath':str(pp.relative_to(repo)),'sha256':sha(pp)},
            'status':'Registered before contact; held pending existing resident, repin, admission and preflight'}
        path=pkg/f'{kind}-plan-registered-v1.json';write(path,plan)
        rows.append(['2026-10-31',rid,'bounded-discovery-probe','Real K4 half '+kind,MODEL,'slurp TRAIN construction '+half,
            'discovery',split,'no',str(calls),str(budget['planned_speech_audio_seconds']),str(budget['planned_gpu_hours']),
            'A0 baseline and all source costs retained',str(path.relative_to(repo)),f'Plan SHA256 {sha(path)}; root countersigned under {AUTH}, coordinator Codex; first-system gain locked; no test tuning'])
    ledger.write_text(text+'\n## Real K4 generation '+half+'\n\n| date | run/experiment id | execution profile | scope | model/tool | speech carrier + split | split role | split identity hash | consumed | calls | speech audio seconds | gpu hours | inherited exposure | artifacts | notes |\n| '+' | '.join(['---']*15)+' |\n'+'\n'.join('| '+' | '.join(row)+' |' for row in rows)+'\n')
    title='Real seed-only construction half' if iteration==1 else 'Real cross-fit banked construction half'
    detail='Seed-only candidates.' if iteration==1 else 'Previous opposite-half measured bank, explicit candidate allocation, unchanged aa/lock; first construction decision pinned.'
    (repo/'docs/readiness'/f'{name}-preregistration.md').write_text('# '+title+'\n\n'+detail+' Full A0 gain locked. Exact real roster, seven normal contacts and fourteen fallback-inclusive. Own supply, registered FIT, resident identity, admission and preflight required. No test tuning; no superiority claim.\n')
    return {'cell':name,'rows':n,'new_model_calls':0,'readiness':'held'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['stage','register'])
    for key in ('repo','data'):p.add_argument('--'+key,type=Path,required=True)
    for key in ('source','binding','out'):p.add_argument('--'+key,type=Path)
    p.add_argument('--iteration',type=int,default=1);p.add_argument('--first-decision',type=Path)
    p.add_argument('--half',choices=['h1','h2']);a=vars(p.parse_args());mode=a.pop('mode')
    if mode=='stage':
        a.pop('half');a.pop('iteration');a.pop('first_decision');result=stage(**a)
    else:
        for key in ('source','binding','out'):a.pop(key)
        result=register(**a)
    print(json.dumps(result))

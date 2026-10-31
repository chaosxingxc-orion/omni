"""Register and score the full313 H1 analysis from verified3 + repaired310.

This is an analysis derivative, never a fabricated generation manifest. Each
source is rendered separately using the shared arm renderer. Official scores
cover the complete313 population; incomplete scoring calls are not replayed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/analysis'),str(ROOT/'scripts/checks'),str(ROOT/'scripts/flights/nx')]
from tfrl_current_half_result import read,sha,analyze,ARMS,SCORER,audit,s3,votes_from_copies
from tfrl_current_sources import pinned
from tfrl_h1_remaining import NAME,RUN,PARENT,sources
from tfrl_h1_recovery_custody import CHECKPOINT_SHA
from score_a0_full1080 import complete_union,unique_rows
AUTH='docs/decisions/2026-10-31-owner-lock-baseline-and-cost-repair.md'
READ='SAEA-SLURP-TFRL-CURRENT-SEED-I1-H1-CONTEXT-REPAIR-UNION-READ-V1'
REL=f'docs/readiness/{NAME}/analysis-recipe-v1.json'
SCHEMA='saea-tfrl-h1-source-union-recipe-v1'


def method_plan(plan):
    """Exclude only run routing and population/accounting, never decoding."""
    return {k:v for k,v in plan.items() if k not in ('run_id','roster','row_plan','budget')}


def contact_policy(row):
    return {**row,'contacts':[{k:v for k,v in c.items() if k not in
                             ('contact_id','request_identity_sha256')} for c in row['contacts']]}


def validate_plans(parent,new,kept):
    old={r['row_id']:r for r in parent['row_plan']};fresh={r['row_id']:r for r in new['row_plan']}
    if (len(old)!=313 or len(fresh)!=310 or len(old)!=len(parent['row_plan'])
            or len(fresh)!=len(new['row_plan']) or len(kept)!=3 or len(set(kept))!=3
            or set(old)!=set(fresh)|set(kept) or set(fresh)&set(kept)
            or method_plan(parent)!=method_plan(new)
            or any(contact_policy(r)!=contact_policy(old[sid]) for sid,r in fresh.items())):
        raise ValueError('source plans must preserve exact313 partition and sampling policy')


def validate_recipe(repo,path):
    import flight_registration_check as reg
    path=path.resolve();recipe=read(path)
    expected={'schema':SCHEMA,'read_run_id':READ,'run_id':RUN,'attempt_id':'generation-v1',
        'rows':313,'continuation_rows':310,'preserved_rows':3,'scorer':SCORER,
        'arms':list(ARMS),'official_scoring_jobs':7,'new_model_calls':0,'claims_benefit':False}
    if any(recipe.get(k)!=v for k,v in expected.items()):raise ValueError('wrong source-union scope')
    entries=reg.parse_ledger(repo,Path('docs/exposure-ledger.md')).get(READ,[])
    if (not path.is_relative_to(repo.resolve()) or len(entries)!=1
            or not reg.is_countersigned(entries[0]) or entries[0]['artifacts']!=str(path.relative_to(repo.resolve()))
            or sha(path) not in entries[0].get('notes','')):raise ValueError('exact registered union recipe required')
    return recipe


def group_for(cfg):
    roles=cfg.values['native_episode']['roles']
    return s3.parse_solver_arm_group(cfg.values,roles['solver'],roles['perception'],cfg.values['knowledge_stores'])


def preserved(repo,data,recipe):
    from speech_aware_evidence_acquisition.core.row_checkpoint import read_checkpoint
    from speech_aware_evidence_acquisition.core.config import compose,protocol_hash
    custody,parts=sources(repo,data,recipe['custody'])
    replay=read(pinned(data,recipe['request_replay']))
    if any(replay.get(k)!=v for k,v in {'rows':3,'contacts':21,'vllm_wire_equal':True,
            'delivered_output_equal':True,'boundary_equal_after_custody_projection':True,
            'checkpoint_sha256':CHECKPOINT_SHA,'new_model_calls':0,'capacity_measured':False}.items()):
        raise ValueError('exact three-row offline request replay required')
    replay_root=pinned(data,recipe['request_replay']).parent
    if sha(replay_root/'custody-deltas.json')!=replay['custody_deltas_sha256']:
        raise ValueError('replay custody transition changed')
    parent=read(pinned(repo,recipe['parent_plan']));new=read(pinned(repo,recipe['continuation_plan']))
    validate_plans(parent,new,parts['reusable_ids'])
    roster=read(pinned(repo,recipe['parent_roster']));ids=[r['sample_id'] for r in roster['items']]
    if (ids!=[r['row_id'] for r in parent['row_plan']] or parent['roster']['file_sha256']!=recipe['parent_roster']['sha256']
            or roster['split_identity_hash']!=recipe['split_identity_hash']):raise ValueError('full parent roster changed')
    exp=read(repo/f'configs/experiment/{PARENT}.json');cell,=exp['cells']
    cfg=compose(repo,cell['model_fragment'],cell['dataset_fragment'],cell['baseline_fragment'],PARENT)
    if (protocol_hash(cfg.config_hash,cfg.values['arm'],replay['split_identity_hash'])!=replay['protocol_hash']
            or sha(repo/f"configs/model/{cell['model_fragment']}.json")!=replay['model_sha256']):
        raise ValueError('replayed original model or protocol differs')
    cp=read_checkpoint(pinned(data,custody['source_checkpoint']),expected_sha256=CHECKPOINT_SHA)
    if cp.header.get('split_identity_hash')!=replay['split_identity_hash']:raise ValueError('replay split differs')
    if read(data/'attempts'/parent['run_id']/'generation-v1.json')['state']!='ABORTED':
        raise ValueError('original source must remain terminal')
    rows=[];trace={};row_sources=[]
    for row in cp.rows:
        if row.sample_id not in parts['reusable_ids']:continue
        rows.extend(row.output_rows)
        row_sources.append({'sample_id':row.sample_id,'source':'original-checkpoint',
            'checkpoint_sha256':CHECKPOINT_SHA,'trace_sha256':row.trace_sha256,
            'canonical_output_sha256':hashlib.sha256(json.dumps(row.output_rows[0],sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()})
        for line in row.trace_lines:
            entry=json.loads(line);key=entry['record_id']
            if key in trace and trace[key]!=entry['payload']:raise ValueError('conflicting historical trace')
            trace[key]=entry['payload']
    partial={**parent,'row_plan':[r for r in parent['row_plan'] if r['row_id'] in parts['reusable_ids']]}
    # Input equality is checked again at analysis time, independently of preparation.
    from audit_real500_a0_inputs import compare_supply
    oldbase=read(repo/f'configs/baseline/{PARENT}.json')
    newbase=read(repo/f'configs/baseline/{NAME}.json')
    compare_supply(read(pinned(data,oldbase['prepared_phase_supply']))['output'],
                   read(pinned(data,newbase['prepared_phase_supply']))['output'],parts['remaining_ids'])
    check=audit.verify(partial,rows,trace)
    return dict(parent=parent,new=new,roster=roster,parts=parts,custody=custody,rows=rows,trace=trace,
                config=cfg,group=group_for(cfg),request_audit=check,row_sources=row_sources)


def register(repo,data,replay):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    assert_execution_environment();head=require_clean_study_repo(repo)
    pkg=repo/'docs/readiness'/NAME;spec=read(pkg/'candidate.json')
    if (repo/REL).exists() or any((data/k/RUN).exists() for k in ('runs','attempts')):
        raise ValueError('fresh unspent recovery registration required')
    def rp(p):return {'relpath':str(p.relative_to(repo)),'sha256':sha(p)}
    old=repo/'docs/readiness'/PARENT/'sampling-plan.json';plan=read(old)
    recipe={'schema':SCHEMA,'read_run_id':READ,'run_id':RUN,'attempt_id':'generation-v1',
        'rows':313,'continuation_rows':310,'preserved_rows':3,'scorer':SCORER,'arms':list(ARMS),
        'official_scoring_jobs':7,'new_model_calls':0,'claims_benefit':False,
        'parent_plan':rp(old),'continuation_plan':rp(pkg/'sampling-plan.json'),
        'parent_roster':rp(repo/'docs/receipts/rosters/slurp-stage3-construction-r01c-h1.json'),
        'split_identity_hash':plan['roster']['split_identity_hash'],'custody':spec['source_custody'],
        'request_replay':{'relpath':str(replay.relative_to(data)),'sha256':sha(replay)},
        'entrypoint':'scripts/analysis/tfrl_h1_union.py','authority':AUTH,
        'selection':'No score-based selection; retain exactly three context-intact original rows and regenerate all310 others',
        'source_commit':head,'decision_scope':'Construction diagnostics only;625 pool and independent confirmation required'}
    preserved(repo,data,recipe)
    ledger=repo/'docs/exposure-ledger.md';text=ledger.read_text()
    if '| '+READ+' |' in text:raise ValueError('READ already registered')
    s3._immutable_json(repo/REL,recipe)
    row=['2026-10-31',READ,'model-free-check','Source-bound full313 H1; original3 plus repaired310',
         'no model; official CPU scorer','slurp TRAIN construction313','discovery',recipe['split_identity_hash'],
         'no','0','0','0','All original and continuation costs retained',REL,
         f'Recipe SHA256 {sha(repo/REL)}; root countersigned under {AUTH}, coordinator Codex;7 whole-roster official scores']
    ledger.write_text(text+'\n## H1 context-repair source-union read\n\n| date | run/experiment id | execution profile | scope | model/tool | speech carrier + split | split role | split identity hash | consumed | calls | speech audio seconds | gpu hours | inherited exposure | artifacts | notes |\n| '+' | '.join(['---']*15)+' |\n| '+' | '.join(row)+' |\n')
    return {'registered':True,'rows':313,'recipe_sha256':sha(repo/REL),'new_model_calls':0}


def score_column(path,result_path,*,binding,ids,lock,data,verify_only=False):
    """Durable per-column intent, exact source receipt, no ambiguous replay."""
    from speech_aware_evidence_acquisition.reproduction.slurp_official_eval import score_slurp_outputs_official_train
    from speech_aware_evidence_acquisition.core.atomic_file import publish_new
    import fcntl
    expected={**binding,'outputs_sha256':sha(path),'rows':len(ids),'scorer':SCORER}
    receipt=result_path.with_suffix('.receipt.json');intent=result_path.with_suffix('.intent.json')
    with result_path.with_suffix('.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if receipt.exists():
            doc=read(receipt)
            if doc!={**expected,'scores_sha256':sha(result_path)}:raise ValueError('score receipt differs')
            result=read(result_path)
        else:
            if verify_only:raise ValueError('required official score receipt missing')
            if result_path.exists() or intent.exists():raise ValueError('incomplete official score; explicit reconciliation required')
            publish_new(intent,(json.dumps(expected,sort_keys=True)+'\n').encode())
            result=score_slurp_outputs_official_train(path,lock,data)
            from tfrl_current_half_result import score_table
            score_table(result,ids)
            s3._immutable_json(result_path,result)
            s3._immutable_json(receipt,{**expected,'scores_sha256':sha(result_path)})
    from tfrl_current_half_result import score_table
    score_table(result,ids)
    return result


def run(repo,data,out,validate_only=False,verify_only=False):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.bundle import verify_bundle
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    from speech_aware_evidence_acquisition.core.config import compose,effective_request_decoding_params
    assert_execution_environment();require_clean_study_repo(repo)
    recipe=validate_recipe(repo,repo/REL)
    if verify_only:
        completed=read(out/'complete.json')
        if completed.get('recipe_sha256')!=sha(repo/REL):raise ValueError('completed recipe changed')
        for rel,digest in completed['artifacts'].items():pinned(out,{'relpath':rel,'sha256':digest})
    src=preserved(repo,data,recipe)
    if not out.resolve().is_relative_to(data.resolve()/'analysis'):raise ValueError('owned analysis path required')
    oldsource={'kind':'verified-checkpoint-subset','checkpoint':src['custody']['source_checkpoint'],
               'custody':recipe['custody'],'request_replay':recipe['request_replay']}
    kept=src['parts']['reusable_ids'];parent_ids=[r['sample_id'] for r in src['roster']['items']]
    oldrender=s3.render_supply_source_rows(src['rows'],src['trace'],src['group'],dict.fromkeys(kept,''),out/'preserved',source=oldsource)
    if validate_only:
        result={'rows_verified':3,'remaining_rows':310,'new_model_calls':0,'new_scoring_jobs':0,'recipe_sha256':sha(repo/REL)}
        s3._immutable_json(out/'preflight.json',result);return result
    mp=data/'runs'/RUN/'generation-v1'/(RUN+'.run-manifest.json');bundle=verify_bundle(mp)
    if (bundle.attempt_state!='FINALIZED' or bundle.manifest['status']!='COMPLETED'
            or bundle.manifest['run_id']!=RUN or bundle.manifest['attempt']['attempt_id']!='generation-v1'):
        raise ValueError('complete repaired310 source required')
    check=audit.verify(src['new'],bundle.outputs_rows,bundle.trace_records)
    cfg=compose(repo,**bundle.manifest['config']['fragments'])
    if cfg.config_hash!=bundle.manifest['config']['config_hash']:raise ValueError('continuation config differs')
    from tfrl_h1_remaining import validate
    spec=read(repo/'docs/readiness'/NAME/'candidate.json');exp=read(repo/f'configs/experiment/{NAME}.json')
    validate(repo,data,NAME,spec,exp,read(pinned(repo,spec['dataset'])))
    if effective_request_decoding_params(cfg.values)!=effective_request_decoding_params(src['config'].values):
        raise ValueError('effective decoding differs')
    from speech_aware_evidence_acquisition.data.lock import load_lock,umbrella_lock_path
    from speech_aware_evidence_acquisition.core.scorers import _assert_scoring_entry_gate
    lock=load_lock(umbrella_lock_path(repo));_assert_scoring_entry_gate(bundle,lock,data,repo)
    newrender=s3.render_supply_variant_first_visit(bundle,group_for(cfg),dict.fromkeys(src['parts']['remaining_ids'],''),out/'continuation',
        run_id=RUN,attempt_id='generation-v1',roster_sha256=src['new']['roster']['file_sha256'],config_hash=cfg.config_hash)
    if oldrender['arm_declarations']!=newrender['arm_declarations']:raise ValueError('arm declarations changed')
    provenance={'schema':'saea-tfrl-h1-union-provenance-v1','recipe_sha256':sha(repo/REL),
        'original':oldsource,'continuation_manifest':{'relpath':str(mp.relative_to(data)),'sha256':sha(mp)},
        'source_audits':{'original':src['request_audit'],'continuation':check},
        'source_renders':{k:sha(out/k/'render-manifest.json') for k in ('preserved','continuation')},
        'all_original_costs_retained':True,'complete_source_cost':bundle.manifest['cost'],
        'original_checkpoint_cost_lower_bound':src['custody']['original_cost_lower_bound'],
        'unaccounted_inflight_cost_is_zero':False}
    s3._immutable_json(out/'provenance.json',provenance)
    new_source_rows=[{'sample_id':r['sample_id'],'source':'repaired310','manifest_sha256':sha(mp)} for r in bundle.outputs_rows]
    s3._immutable_jsonl(out/'row-sources.jsonl',complete_union(parent_ids,unique_rows(src['row_sources']),src['parts']['remaining_ids'],unique_rows(new_source_rows)))
    merged={'rows':313,'copies':{}}
    for arm in ARMS:
        def copy(render):
            pin=render['copies'][arm];p=Path(pin['path'])
            if sha(p)!=pin['sha256']:raise ValueError('rendered copy changed')
            return unique_rows(list(s3.iter_jsonl(p)))
        rows=complete_union(parent_ids,copy(oldrender),src['parts']['remaining_ids'],copy(newrender))
        p=out/'columns'/f'arm-{arm}-outputs.jsonl';digest=s3._immutable_jsonl(p,rows)
        merged['copies'][arm]={'path':str(p),'sha256':digest,'rows':313}
    p=out/'columns/arm-vote-outputs.jsonl';digest=s3._immutable_jsonl(p,votes_from_copies(merged));merged['copies']['vote']={'path':str(p),'sha256':digest,'rows':313}
    s3._immutable_json(out/'columns/render-manifest.json',merged)
    binding={'schema':'saea-tfrl-source-union-score-receipt-v1','recipe_sha256':sha(repo/REL),
        'provenance_sha256':sha(out/'provenance.json'),'row_sources_sha256':sha(out/'row-sources.jsonl'),
        'render_manifest_sha256':sha(out/'columns/render-manifest.json')}
    results={a:score_column(Path(merged['copies'][a]['path']),out/'columns'/f'arm-{a}-scores.json',
        binding={**binding,'arm':a},ids=parent_ids,lock=lock,data=data,verify_only=verify_only) for a in (*ARMS,'vote')}
    credit,summary=analyze(src['parent'],results);s3._immutable_jsonl(out/'credit.jsonl',credit)
    summary.update(read_run_id=READ,half='R01C-H1',iteration=1,
        **binding,credit_sha256=sha(out/'credit.jsonl'),new_model_calls=0,official_scoring_jobs=7)
    summary['schema']='saea-tfrl-h1-source-union-result-v1'
    summary['artifacts']={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.suffix!='.lock' and p.name!='complete.json'}
    s3._immutable_json(out/'complete.json',summary)
    if verify_only:
        from types import SimpleNamespace
        partial={**src['parent'],'row_plan':[r for r in src['parent']['row_plan'] if r['row_id'] in kept]}
        return {'plan':src['parent'],'roster':src['roster'],'results':results,'credit':credit,'config':cfg,
            'segments':[
                {'namespace':'original-checkpoint','plan':partial,
                 'rows':SimpleNamespace(outputs_rows=src['rows'],trace_records=src['trace'])},
                {'namespace':'repaired310','plan':src['new'],'rows':bundle}],
            'source':{'kind':'verified-source-union','read_run_id':READ,'complete_relpath':str((out/'complete.json').relative_to(data)),
                'complete_sha256':sha(out/'complete.json'),'recipe_sha256':sha(repo/REL),
                'provenance_sha256':sha(out/'provenance.json')}}
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['register','run'])
    for k in ('repo','data'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--replay',type=Path);p.add_argument('--out',type=Path);p.add_argument('--validate-only',action='store_true')
    a=vars(p.parse_args());mode=a.pop('mode')
    if mode=='register':a.pop('out');a.pop('validate_only');result=register(**a)
    else:a.pop('replay');result=run(**a)
    print(json.dumps(result))

"""Derive held seed-only K4 halves from the verified current A0 construction base.

Reuse actual625 inputs and original prompts. Each half has fresh run/FIT/supply
identities; no model calls, registration, readiness flip or bank consumption.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
REPO=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(REPO/'src'),str(Path(__file__).parent)]
import r01c_current_baseline as baseline
import tfrl_v2_sampler as sampler
import tfrl_v2_split as split
import l338_family as family
from speech_aware_evidence_acquisition.rails import consumption_module as cm
from speech_aware_evidence_acquisition.rails.adapters.slurp_module_hooks import carrier_module_hooks
from speech_aware_evidence_acquisition.core.atomic_file import publish_new
read,sha,raw=baseline.read,baseline.sha,baseline.raw


def discard_synthetic_cost_forecasts(plan):
    """Retain exact call/audio counts; real K4 timing has not been measured."""
    allowed = {'rows', 'k', 'contacts_per_row', 'includes_twin_and_lock',
               'calls', 'calls_with_fallback_ceiling', 'audio_seconds'}
    for key, budget in plan['budget'].items():
        plan['budget'][key] = {k: v for k, v in budget.items() if k in allowed}
        plan['budget'][key].update(cost_forecast_available=False,
            reason='Synthetic throughput is inapplicable; measure this real K4 context and request flow.')
    return plan


def derive(repo, original, experiment, roster, half):
    suffix=half.split('-')[1].lower();name=f'slurp-tfrl-current-seed-i1-{suffix}-v1'
    run=f'SAEA-SLURP-TFRL-CURRENT-SEED-I1-{suffix.upper()}-V1'
    plan=sampler.build_plan(roster,half,1,roster_file_sha256=sha(repo/sampler.cm.HALF_ROSTER_RELPATH[half]),
        execution_run_id=run,execution_attempt_id='generation-v1')
    discard_synthetic_cost_forecasts(plan)
    base,exp=copy.deepcopy(original),copy.deepcopy(experiment)
    block=base['native_episode']['consumption_module']
    if block['columns'] or 'tfrl_v2_bank' in block or base['policy']!='phase-native-module-v1-c2':
        raise ValueError('plain actual two-contact R01C method required')
    block['columns']=sampler.execution_columns(plan)
    block['decoding_frame']={'fit_estimate':dict(family.FIT_ESTIMATE),
        'max_supplied_items':family.MAX_SUPPLIED_ITEMS,'template_tokens':family.DELIVERING_TEMPLATE_TOKENS,'twin':True}
    compiled=cm.compile_module(block,carrier_module_hooks(repo_root=repo),repo_root=repo)
    if compiled.program_ceiling!=7:raise ValueError('one P plus K4 and twin/lock required')
    values=compiled.values_on_roster_of(base)
    base['policy']=values['policy']
    expected=copy.deepcopy(base['native_episode']['roles'])
    expected['solver']['decoding_variants']=values['native_episode']['roles']['solver']['decoding_variants']
    if values['native_episode']['roles']!=expected:
        raise ValueError('compilation changed A0 roles beyond Solver trajectory group')
    base['native_episode']['roles']=expected
    base.pop('prepared_phase_supply',None)
    n=len(roster['items'])
    base['native_contact_fit'].update(run_id=run+'-FIT',attempt_id='fit-v1',contact_cap=7*n)
    base['what_this_is']='Held actual A0 plus K4 per-row seed-only Solver trajectories, twin and lock; fresh own supply required.'
    base['preparation_source_receipts'].append({'root':'repository','relpath':sampler.cm.HALF_ROSTER_RELPATH[half],
        'sha256':sha(repo/sampler.cm.HALF_ROSTER_RELPATH[half])})
    cm.verify_declared_module(block,base,repo_root=repo)
    cell,=exp['cells'];cell.update(cell_id=name,baseline_fragment=name,run_id=run,attempt_id='generation-v1',
        allowed_execution_stages=['check'],readiness={'state':'held','requirements':[],
        'reason':'Full A0 gain lock, own supply, selected service and formal registration/admission pending.'})
    exp.update(experiment_id=run,purpose=base['what_this_is'],population={'count':n,
        'dataset_fragment':cell['dataset_fragment'],'roster_receipt':sampler.cm.HALF_ROSTER_RELPATH[half],
        'split_identity_hash':roster['split_identity_hash']},
        populations=[{'dataset_fragment':cell['dataset_fragment'],'ids':[r['sample_id'] for r in roster['items']]}])
    exp['execution_plan'].update(planned_model_calls=n*14,
        planned_speech_audio_seconds=math.ceil(sum(r['audio_seconds'] for r in roster['items'])*14))
    exp['stage3_measurement'].update(rows=n,read_roster=roster['roster_id'],claims_benefit=False,
        note='Construction trajectories only; no superiority claim. Retain twin and lock for credit.')
    return name,base,exp,plan


def write(repo,data,binding,out):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    assert_execution_environment()
    if subprocess.check_output(['git','status','--porcelain'],cwd=repo).strip():raise ValueError('clean implementation required')
    source=read(binding)
    for rel,digest in source['output_hashes'].items():
        if sha(repo/rel)!=digest:raise ValueError('actual R01C baseline binding changed')
    if not all(c['identity_ok'] and c['seed_ok'] for c in source['checks']):raise ValueError('verified own supply required')
    parent,_=split.load_parent(repo/sampler.cm.ROSTER_RELPATH)
    halves={h:read(repo/sampler.cm.HALF_ROSTER_RELPATH[h]) for h in sampler.cm.HALVES}
    receipt=read(repo/sampler.cm.SPLIT_RECEIPT_RELPATH)
    for half,roster in halves.items():
        if sha(repo/sampler.cm.HALF_ROSTER_RELPATH[half])!=receipt['halves'][half]['file_sha256']:
            raise ValueError('fixed half changed')
    ids=[{x['sample_id'] for x in halves[h]['items']} for h in sampler.cm.HALVES]
    if ids[0]&ids[1] or ids[0]|ids[1]!={x['sample_id'] for x in parent['items']}:
        raise ValueError('halves not a disjoint full625 union')
    original=read(repo/f'configs/baseline/{baseline.CELL}.json');exp=read(repo/f'configs/experiment/{baseline.CELL}.json')
    outputs={};summaries=[]
    for half,roster in halves.items():
        name,b,e,p=derive(repo,original,exp,roster,half)
        run=e['cells'][0]['run_id']
        if any((data/k/run).exists() for k in ('attempts','runs','prepared')):raise ValueError('successor already used')
        for kind,doc in [('baseline',b),('experiment',e)]:outputs[f'configs/{kind}/{name}.json']=raw(doc)
        outputs[f'docs/readiness/{name}/sampling-plan.json']=raw(p)
        summaries.append({'cell':name,'rows':len(roster['items']),'normal_calls':7*len(roster['items']),
            'fallback_cap':14*len(roster['items']),'readiness':'held','own_supply':'pending'})
    out=baseline.comparator.confined(data,str(out))
    if out.exists() or not out.is_relative_to(data.resolve()/'work/codex') or any((repo/p).exists() for p in outputs):
        raise ValueError('fresh successor paths required')
    out.mkdir(parents=True)
    for rel,blob in outputs.items():
        target=repo/rel;target.parent.mkdir(parents=True,exist_ok=True);publish_new(target,blob);target.chmod(0o644)
    result={'schema':'saea-tfrl-current-seed-cells-v1','rows':625,'cells':summaries,'new_model_calls':0,
        'source_binding_sha256':sha(binding),'output_hashes':{p:hashlib.sha256(b).hexdigest() for p,b in outputs.items()},
        'generation_admitted':False,'remaining':['full A0 gain lock','own supply','current service','registration','trace/request verification before credit']}
    publish_new(out/'complete.json',raw(result));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('repo','data','binding','out'):p.add_argument('--'+key,type=Path,required=True)
    print(json.dumps(write(**vars(p.parse_args()))))

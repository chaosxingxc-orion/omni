"""Derive held full-H2 shared-P learned/lock evaluation; no model contacts.

A real sealed candidate and measured template FIT must be supplied before this
can emit a flight configuration. Registration, first-phase supply, admission,
execution and official arm scoring are deliberately separate operations.
"""
import copy
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from speech_aware_evidence_acquisition.rails import consumption_module as cm
from speech_aware_evidence_acquisition.rails import tfgrpo_consumption as consumption
from speech_aware_evidence_acquisition.rails.adapters.slurp_module_hooks import carrier_module_hooks
NAME='slurp-tfgrpo-dense-h2-paired-v1'
RUN='SAEA-SLURP-TFGRPO-DENSE-H2-PAIRED-V1'
ROSTER='docs/receipts/rosters/slurp-stage3-construction-r01c-h2.json'
AUTH='docs/decisions/2026-10-31-owner-lock-baseline-and-cost-repair.md'



def cumulative_supply_bound(base):
    """Bound retained P knowledge plus both Solver sources, without changing them.

    This fixed P/S program has no refresh, acquisition or revision. Associated
    values cap delivery at min(k, top_l); other fusers cap at k. Candidate supply
    has its own max_total_values. Include retained prefix even when compacted:
    the arm receipt counts accumulated IDs, not just rendered Solver examples.
    """
    native = base['native_episode']
    if native.get('max_revisions', 0) or native.get('review_loop') != 'off':
        raise ValueError('supply bound requires fixed non-revising P/S program')
    total = 0
    def positive(value):
        if type(value) is not int or value <= 0:
            raise ValueError('finite positive supply bound required')
        return value
    for store in base['knowledge_stores'].values():
        for phase, spec in store.get('phase_supply', {}).items():
            if phase not in {'perception', 'solver'}:
                raise ValueError('unexpected phase in fixed H2 supply')
            count = positive(spec['k'])
            if spec['fuse_policy'] == 'saea-fuse-associated-values-v1':
                count = min(count, positive(spec.get('fuse_coverage', {}).get('top_l', 3)))
            total += count
        for phase, spec in store.get('candidate_supply', {}).items():
            if phase != 'solver':
                raise ValueError('unexpected candidate phase in fixed H2 supply')
            total += positive(spec['max_total_values'])
    return positive(total)


def derive(repo,data,params,*,fit_estimate,base_template_tokens,learned_template_tokens):
    repo=Path(repo);data=Path(data)
    roster=json.loads((repo/ROSTER).read_bytes());ids=[r['sample_id'] for r in roster['items']]
    if len(ids)!=312 or len(set(ids))!=312 or set(params['consumer_ids'])!=set(ids):
        raise ValueError('the exact complete H2 312 roster is required')
    consumption.verify(params,repo=repo,data=data,model_id=params['model_id'],consumer_ids=ids)
    base=json.loads((repo/'configs/baseline/slurp-tfrl-hardset-a0-observation-r1-v1.json').read_bytes())
    exp=json.loads((repo/'configs/experiment/slurp-tfrl-hardset-a0-observation-r1-v1.json').read_bytes())
    block=copy.deepcopy(base['native_episode']['consumption_module'])
    before=cm.compile_module(block,carrier_module_hooks(repo_root=repo),repo_root=repo)
    if before.values()['phase_material']['solver']['instruction_template']!=params['base_template']['relpath']:
        raise ValueError('candidate source Solver base differs')
    paired={**copy.deepcopy(params),'target_arm':'learned'}
    block['components'].append({'component_id':consumption.COMPONENT,'state':'on',
        'admitted_by':'development-evaluation','record':AUTH,'params':paired})
    block['columns']=[{'column_id':'learned','kind':'template','node':'solver',
        'template':{'relpath':'configs/'+params['template']['relpath'],'sha256':params['template']['sha256']},
        'template_tokens':learned_template_tokens}]
    supply=cumulative_supply_bound(base)
    block['template_frame']={'max_supplied_items':supply,'delivering_template_tokens':base_template_tokens,
        'twin':False,'fit_estimate':copy.deepcopy(fit_estimate)}
    compiled=cm.compile_module(block,carrier_module_hooks(repo_root=repo),repo_root=repo)
    if compiled.program_ceiling!=3:raise ValueError('one P and two Solver contacts required')
    base['native_episode']['consumption_module']=block
    base=compiled.apply_to(base)
    base['native_episode']['consumption_module']=block
    base.pop('prepared_phase_supply',None)
    base['serving_slot_tokens']=fit_estimate['slot_tokens']
    base['native_contact_fit'].update(run_id=RUN+'-FIT',attempt_id='fit-v1',contact_cap=936)
    base['what_this_is']='Held full H2 development: one fresh P, learned Solver and unchanged lock Solver; candidate not accepted.'
    cm.verify_declared_module(block,base,repo_root=repo)
    cell,=exp['cells'];cell.update(cell_id=NAME,run_id=RUN,attempt_id='generation-v1',baseline_fragment=NAME,
        allowed_execution_stages=['check'],readiness={'state':'held','requirements':[],
        'reason':'Requires measured template FIT, exact candidate seal, registration, first-phase supply, recovery and admission.'})
    exp.pop('prepared_phase_supply',None)
    exp.update(experiment_id=RUN,purpose=base['what_this_is'],
        population={'count':312,'dataset_fragment':cell['dataset_fragment'],'roster_receipt':ROSTER,
                    'split_identity_hash':roster['split_identity_hash']},
        populations=[{'dataset_fragment':cell['dataset_fragment'],'ids':ids}])
    # Existing raw-vLLM fallback admits at most one additional completion/contact.
    exp['execution_plan'].update(planned_model_calls=1872,
        planned_speech_audio_seconds=math.ceil(6*sum(r['audio_seconds'] for r in roster['items'])))
    exp['stage3_algorithm']={'id':'tfgrpo-semantic-advantage-v1',
        'operators_used':['context_application','rollout','scoring'],
        'note':'Dense partial-reward H1 adaptation from empty parent; short display IDs. Same fresh P and evidence; no H2 feedback in learning. Not isolated reward causality.'}
    exp['stage3_measurement'].update(rows=312,read_roster=roster['roster_id'],claims_benefit=False,
        note='Development paired comparison, not untouched confirmation. Score both learned and lock arms officially.')
    return {'name':NAME,'baseline':base,'experiment':exp,'roster':roster,
        'evaluation':{'rows':312,'normal_generation_calls':936,'fallback_generation_cap':1872,'fit_calls':936,
            'primary_comparison':'learned minus lock, official SLURP aggregate F1',
            'acceptance':'at least +1 percentage point SLU-F1 with no Intent decrease; report paired uncertainty and per-label regressions',
            'candidate_accepted':False,'new_model_calls':0,'new_tokenizer_calls':0}}

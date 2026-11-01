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
NAME='slurp-tfgrpo-fixed-p-h2-paired-v1'
RUN='SAEA-SLURP-TFGRPO-FIXED-P-H2-PAIRED-V1'
ROSTER='docs/receipts/rosters/slurp-stage3-construction-r01c-h2.json'
AUTH='docs/decisions/2026-10-31-owner-lock-baseline-and-cost-repair.md'


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
    supply=base['knowledge_stores']['train-exemplars']['phase_supply']['solver']['k']
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
        'note':'Same fresh P and evidence, only learned Solver instruction differs; no H2 feedback used in learning.'}
    exp['stage3_measurement'].update(rows=312,read_roster=roster['roster_id'],claims_benefit=False,
        note='Development paired comparison, not untouched confirmation. Score both learned and lock arms officially.')
    return {'name':NAME,'baseline':base,'experiment':exp,'roster':roster,
        'evaluation':{'rows':312,'normal_generation_calls':936,'fallback_generation_cap':1872,'fit_calls':936,
            'primary_comparison':'learned minus lock, official SLURP aggregate F1',
            'acceptance':'at least +1 percentage point SLU-F1 with no Intent decrease; report paired uncertainty and per-label regressions',
            'candidate_accepted':False,'new_model_calls':0,'new_tokenizer_calls':0}}

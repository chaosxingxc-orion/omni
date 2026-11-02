"""Derive held full-H2 shared-P learned/lock evaluation; no model contacts.

A real sealed candidate and measured template FIT must be supplied before this
can emit a flight configuration. Registration, first-phase supply, admission,
execution and official arm scoring are deliberately separate operations.
"""
import copy
import json
import math
import re
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



EVALUATION_SCHEMA = 'saea-learning-h2-evaluation-v1'
EVALUATION_METHODS = {
    'ace_incremental_offline_v1': ('ace-grow-refine-v1', 'ACE offline'),
    'reflexion_offline_transfer_v1': ('reflexion-offline-transfer-v1', 'Reflexion offline transfer'),
}


def evaluation_algorithm(config):
    identity = evaluation_identity(config)
    return EVALUATION_METHODS[identity['producer_strategy']]


def verify_producer(params, *, data, evaluation):
    if evaluation is None:
        return
    identity = evaluation_identity(evaluation)
    if params.get('candidate_release') != identity['candidate_release']:
        raise ValueError('evaluation release differs from consumption release')
    from speech_aware_evidence_acquisition.rails.learning_candidate_release import read_pin
    _, release = read_pin(data, params['candidate_release'])
    _, candidate = read_pin(data, params['candidate'])
    if (release.get('strategy') != identity['producer_strategy']
            or candidate.get('strategy') != identity['producer_strategy']
            or release.get('candidate') != params['candidate']):
        raise ValueError('evaluation producer differs from pinned candidate and release')


def evaluation_identity(config=None):
    if config is None:
        return {'name':NAME,'run_id':RUN,'template':'probes/tfgrpo-fixed-p-h1-experience-v1.txt'}
    if (not isinstance(config,dict) or set(config) != {'schema','name','run_id','template','producer_strategy','runtime','candidate_release'}
            or config['schema'] != EVALUATION_SCHEMA or config['producer_strategy'] not in EVALUATION_METHODS
            or not re.fullmatch('[a-z0-9][a-z0-9-]*',str(config['name']))
            or not re.fullmatch('SAEA-[A-Z0-9][A-Z0-9_-]*',str(config['run_id']))
            or config['name']==NAME or config['run_id']==RUN
            or config['template'] != 'probes/'+config['name']+'-experience.txt'):
        raise ValueError('exact independent implemented learning evaluation identity required')
    # Full pin syntax is checked by the shared consumption contract.
    pin=config['candidate_release']
    if not isinstance(pin,dict) or set(pin)!={'relpath','sha256'}:
        raise ValueError('explicit independent candidate release required')
    return config


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


def derive(repo,data,params,*,fit_estimate,base_template_tokens,learned_template_tokens,evaluation=None):
    identity=evaluation_identity(evaluation); name,run=identity['name'],identity['run_id']
    if evaluation is not None and (params.get('candidate_release')!=identity['candidate_release']
            or params['template']['relpath']!=identity['template']):
        raise ValueError('paired candidate release or template differs from evaluation')
    repo=Path(repo);data=Path(data)
    roster=json.loads((repo/ROSTER).read_bytes());ids=[r['sample_id'] for r in roster['items']]
    if len(ids)!=312 or len(set(ids))!=312 or set(params['consumer_ids'])!=set(ids):
        raise ValueError('the exact complete H2 312 roster is required')
    consumption.verify(params,repo=repo,data=data,model_id=params['model_id'],consumer_ids=ids)
    verify_producer(params,data=data,evaluation=evaluation)
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
    base['native_contact_fit'].update(run_id=run+'-FIT',attempt_id='fit-v1',contact_cap=936)
    base['what_this_is']='Held full H2 development: one fresh P, learned Solver and unchanged lock Solver; candidate not accepted.'
    cm.verify_declared_module(block,base,repo_root=repo)
    cell,=exp['cells'];cell.update(cell_id=name,run_id=run,attempt_id='generation-v1',baseline_fragment=name,
        allowed_execution_stages=['check'],readiness={'state':'held','requirements':[],
        'reason':'Requires measured template FIT, exact candidate seal, registration, first-phase supply, recovery and admission.'})
    exp.pop('prepared_phase_supply',None)
    exp.update(experiment_id=run,purpose=base['what_this_is'],
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
    if evaluation is not None:
        algorithm,label=evaluation_algorithm(evaluation)
        exp['stage3_algorithm']={'id':algorithm,'operators_used':['context_application'],
            'note':f'Frozen {label} candidate consumption; feedback is not learned on H2.'}
    return {'name':name,'baseline':base,'experiment':exp,'roster':roster,
        'evaluation':{'rows':312,'normal_generation_calls':936,'fallback_generation_cap':1872,'fit_calls':936,
            'primary_comparison':'learned minus lock, official SLURP aggregate F1',
            'acceptance':'at least +1 percentage point SLU-F1 with no Intent decrease; report paired uncertainty and per-label regressions',
            'candidate_accepted':False,'new_model_calls':0,'new_tokenizer_calls':0}}

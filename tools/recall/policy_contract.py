"""Finite policy catalog, strict validation, deterministic compilation; no experiments."""
import copy, hashlib, json, re

ALGORITHM = 'TFGRPO_inspired_group_policy_feedback_search_exploration_v1'
FAILURE_DIMENSIONS = ['Q15_question', 'Q50_question', 'Q15_union', 'Q50_union',
                      'unique_labels', 'source_availability', 'KB_size', 'CPU_cost',
                      'engineering_validity', 'semantic_uncertainty']
COMMON = dict(maximum_rounds=3, maximum_accepted_actions=3, maximum_trials=600,
              maximum_pool_candidates=96, maximum_additions=1, maximum_deletions=0,
              candidates_per_error=2, maximum_same_label_deletions=0,
              maximum_observed_competitor_deletions=0, maximum_causal_chain_deletions=0,
              dependencies_per_loss=0, proposal_order='similarity_then_index',
              final_joint_gate=True)
SINGLE = dict(COMMON, mode='single_insert_reference')
JOINT = dict(COMMON, mode='joint_swap_reference', candidates_per_error=1,
             maximum_deletions=2, maximum_same_label_deletions=3,
             maximum_observed_competitor_deletions=3, maximum_causal_chain_deletions=2,
             proposal_order='historical_then_similarity')
REFERENCES = [dict(id='R0_single_insert', policy=SINGLE), dict(id='R1_cause_joint', policy=JOINT)]
CATALOG = {
 'single_insert_reference': 'Top2 distinct eligible TRAIN sources per current Q15 miss; no deletion. Frozen historical single-insert action family; execution seed comes from input refs.',
 'joint_swap_reference': 'Historical actual Q15 correct representatives plus nearest eligible TRAIN; same-label or observed regressed-query competitor deletes, depth<=2. Existing S2 action family, execution seed comes from input refs; cached from68 results remain historical only.',
 'alternative_representative': 'Top3..6 eligible distinct-source TRAIN per Q15 miss; keep closest as counterfactual. Test all bounded alternatives and prefer fewer induced protected loss events, then repairs, target cosine, original index. Never globally prefer weak similarity without replay evidence.',
 'compensation_chain': 'Start from one bounded proposed insertion. Along its ACTUAL protected losses add top1..4 eligible same-gold TRAIN dependencies or remove observed wrong keys; legacy at most3 adds2 deletes; optional causal_depth_v1 allows bounded6/2 with measured loss-class admission; canonical state dedup; choose smallest remaining loss-event count before repair gain. Final joint gate only.'
}

def digest(x):
 return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

CAUSAL_DEFAULTS = dict(max_new_states_per_candidate=24, initial_states_per_candidate=4,
 max_new_states_per_layer=2, beam_width=2, max_depth=4,
 action_order=['delete','replace','add'], max_nonmonotone_steps=1,
 max_loss_increase=2, max_extended_primaries=0, extended_min_lost_classes=4)

def validate_policy(p):
 if 'proposal_plugin' in p:
  from candidate_proposals import validate_plugin
  validate_plugin(p['proposal_plugin'])
  if p.get('mode') in ('single_insert_reference','joint_swap_reference'):raise ValueError('Reference plugin forbidden')
  if 'joint_duty' in p['proposal_plugin']:
   if p.get('mode')!='compensation_chain' or p.get('maximum_deletions',0)<max(p['proposal_plugin']['joint_duty']['delete_sizes']):raise ValueError('joint_duty requires compensation family and deletion bounds')
 if p.get('acceptance_mode','strict_gain') not in ('strict_gain','gain_or_shrink'):raise ValueError('Unsupported acceptance_mode')
 if set(p)-{'proposal_plugin','acceptance_mode'} not in (set(COMMON)|{'mode'}, set(COMMON)|{'mode','compensation_budget'}): raise ValueError('Unsupported policy fields')
 causal=False
 if 'compensation_budget' in p:
  b=p['compensation_budget']
  if p['mode']!='compensation_chain' or not isinstance(b,dict): raise ValueError('Unsupported compensation budget')
  if b.get('mode')=='prescreen_fair_v1':
   if set(b)!={'mode','max_new_states_per_candidate'} or type(b['max_new_states_per_candidate']) is not int or not 1<=b['max_new_states_per_candidate']<=600: raise ValueError('Unsupported compensation budget')
  elif b.get('mode')=='causal_depth_v1':
   if not set(b)<=set(CAUSAL_DEFAULTS)|{'mode'}: raise ValueError('Unsupported causal fields')
   b=dict(copy.deepcopy(CAUSAL_DEFAULTS),**b);p=copy.deepcopy(p);p['compensation_budget']=b;causal=True
   limits=dict(max_new_states_per_candidate=(1,600),initial_states_per_candidate=(1,600),
    max_new_states_per_layer=(1,32),beam_width=(1,8),max_depth=(1,7),
    max_nonmonotone_steps=(0,3),max_loss_increase=(0,32),max_extended_primaries=(0,18),extended_min_lost_classes=(1,59))
   for k,(lo,hi) in limits.items():
    if type(b[k]) is not int or not lo<=b[k]<=hi: raise ValueError('Out of bounds: '+k)
   if not isinstance(b['action_order'],list) or len(b['action_order'])!=3 or set(b['action_order'])!={'add','delete','replace'}: raise ValueError('Action permutation')
   if b['initial_states_per_candidate']>b['max_new_states_per_candidate']: raise ValueError('Initial exceeds primary quota')
  else: raise ValueError('Unsupported compensation budget mode')
 if p['mode'] not in CATALOG or p['final_joint_gate'] is not True: raise ValueError('Unsupported mode/gate')
 bounds=dict(maximum_rounds=(1,3),maximum_accepted_actions=(1,3),maximum_trials=(1,600),
             maximum_pool_candidates=(1,96),maximum_additions=(1,6 if causal else 3),maximum_deletions=(0,2),
             candidates_per_error=(1,6),maximum_same_label_deletions=(0,3),
             maximum_observed_competitor_deletions=(0,3),maximum_causal_chain_deletions=(0,2),
             dependencies_per_loss=(0,4))
 for k,(lo,hi) in bounds.items():
  if type(p[k]) is not int or not lo<=p[k]<=hi: raise ValueError('Out of bounds: '+k)
 mode=p['mode']
 if mode=='single_insert_reference' and p!=SINGLE: raise ValueError('Reference policy must remain frozen')
 if mode=='joint_swap_reference' and p!=JOINT: raise ValueError('Reference policy must remain frozen')
 if mode=='alternative_representative':
  if not (3<=p['candidates_per_error']<=6 and p['maximum_additions']==1 and p['maximum_deletions']<=2 and p['dependencies_per_loss']==0 and p['proposal_order']=='least_induced_losses_then_repairs'): raise ValueError('Alternative family mismatch')
 if mode=='compensation_chain':
  if not (p['maximum_additions']>=2 and p['dependencies_per_loss']>=1 and p['proposal_order']=='least_remaining_losses_then_repairs'): raise ValueError('Compensation family mismatch')
 return p

def _compile_legacy_group(x,input_refs=None):
 x=copy.deepcopy(x)
 if x.get('algorithm_id')!=ALGORITHM or x.get('status')!='PROPOSED_NOT_EVALUATED': raise ValueError('Proposal identity/status')
 ss=x['strategies']
 if not 2<=len(ss)<=3 or len({s['id'] for s in ss})!=len(ss): raise ValueError('Group size/IDs')
 for ref in REFERENCES:
  if sum(s['id']==ref['id'] and s['policy']==ref['policy'] for s in ss)!=1: raise ValueError('Missing exact reference')
 for s in ss:
  if not isinstance(s['id'],str) or not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_-]{0,63}',s['id']): raise ValueError('Unsafe strategy ID')
  s['policy']=validate_policy(s['policy'])
  if not set(s['failure_feedback_dimensions'])<=set(FAILURE_DIMENSIONS): raise ValueError('Unknown failure dimension')
  for field in ['root_cause','counterevidence','falsification','expected_new_information']:
   if not isinstance(s[field],str) or not 1<=len(s[field])<=2000: raise ValueError('Missing bounded mechanism field')
  if not s['evidence_ids'] or not set(s['evidence_ids']) <= {'C1','C2','C3','C4'}: raise ValueError('Evidence references')
 if len({s['policy']['mode'] for s in ss})!=len(ss): raise ValueError('Duplicate action families')
 if x['preferred_strategy_id'] not in {s['id'] for s in ss}: raise ValueError('Preferred ID')
 seed_identity=dict(same_seed=None,input_refs_status="UNBOUND_NO_EXECUTION_SEED_CLAIM") if input_refs is None else dict(same_seed=input_refs['seed_size'],input_refs_sha256=digest(input_refs),seed_sha256=input_refs['local_inputs']['seed']['sha256'],query_baseline_sha256=input_refs['local_inputs'].get('query_baseline',input_refs['local_inputs'].get('query68'))['sha256'])
 return dict(schema='executable-group-policies-v1',algorithm_id=ALGORITHM,status='COMPILED_NOT_EVALUATED',
             group_sha256=digest(x),strategies=copy.deepcopy(ss),**seed_identity,
             evaluation=dict(roles={'fit':624,'valid':1079,'FoldA':1080},depths=[15,50],description_depth=5,
              unique_label_caps={'question':{'15':15,'50':50},'union':{'15':20,'50':55}},
              preserve_initial_classes=59,preserve_initial_source_availability=True,per_query_zero_regressions=True),
             feedback_memory_update='consume paired witnesses; historical reference seeds remain explicit, no centered rank without actual same-start evaluation; propose bounded rule delta, do not execute automatically')


def compile_group(x,input_refs=None):
 """Legacy groups remain compatible; explicit singleton discovery uses actual refs."""
 if len(x.get('strategies',[]))!=1:return _compile_legacy_group(x,input_refs=input_refs)
 if set(x)!={'algorithm_id','status','strategies','preferred_strategy_id'}:raise ValueError('Unsupported singleton output fields')
 required={'id','policy','failure_feedback_dimensions','root_cause','counterevidence','falsification','expected_new_information','evidence_ids'}
 if set(x['strategies'][0])!=required:raise ValueError('Unsupported singleton strategy fields')
 if input_refs is None:raise ValueError('Single discovery requires explicit runtime input refs')
 if type(input_refs.get('seed_size')) is not int or input_refs['seed_size']<59:raise ValueError('Explicit actual seed_size required')
 for key in ['seed','query_baseline']:
  entry=input_refs['local_inputs'][key]
  if not isinstance(entry.get('path'),str) or not re.fullmatch('[0-9a-f]{64}',entry.get('sha256','')):raise ValueError('Invalid input ref: '+key)
 ss=x['strategies'];strategy=ss[0]
 if strategy['id'] in {r['id'] for r in REFERENCES}:raise ValueError('Historical reference cannot be singleton discovery')
 if strategy['policy'].get('compensation_budget',{}).get('mode')!='causal_depth_v1':raise ValueError('Single discovery requires causal_depth_v1 budget')
 if strategy['policy'].get('mode')!='compensation_chain' or 'proposal_plugin' not in strategy['policy']:raise ValueError('Single discovery requires compensation_chain and explicit proposal_plugin')
 enriched=copy.deepcopy(x)
 for ref in REFERENCES:
  enriched['strategies'].append(dict(copy.deepcopy(ref),failure_feedback_dimensions=[],root_cause='Historical compiler compatibility only; not a requested execution',counterevidence='Different historical seed; no current comparison',falsification='Not executed',expected_new_information='None',evidence_ids=['C1']))
 compiled=_compile_legacy_group(enriched,input_refs=input_refs)
 compiled.update(group_sha256=digest(x),strategies=[copy.deepcopy(validate_strategy_policy(strategy))],same_start_group=False,historical_reference_seed=68,centered_feedback_supported=False,execution_scope='SINGLE_DISCOVERY_EXPLORATION_NOT_REFERENCE_GROUP')
 return compiled

def validate_strategy_policy(strategy):
 out=copy.deepcopy(strategy);out['policy']=validate_policy(out['policy']);return out

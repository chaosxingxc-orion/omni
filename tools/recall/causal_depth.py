"""Bounded search helper for run_group's existing test/gate/ledger callbacks.

No runtime, IO, scoring, acceptance, or alternative executor lives here.
A generator keeps a small layered beam across primary quota allocations.
"""
from policy_contract import digest

def state_hash(state):
 return digest(sorted(state))

def frontier(single_state,single_eval,*,ids,primary,budget,add_cap,delete_cap,
             losses,repairs,lost_classes,successors,test,structural,audit,expand_loss_free=False):
 """Yield after each measured child; caller owns all global/local charging."""
 first=dict(state=set(single_state),evaluation=single_eval,nonmonotone=0,
            path=[],layer=0,loss=len(losses(single_eval)))
 nodes=[first]; seen={state_hash(ids),state_hash(single_state)};cursor=0
 def rank(n):
  return (n['loss'],-repairs(n['evaluation']),len(n['state']),state_hash(n['state']),n['nonmonotone'])
 for layer in range(1,budget['max_depth']+1):
  choices=[];queues={f:[] for f in budget['action_order']}
  for node in sorted(nodes,key=rank):
   if not node['loss'] and not expand_loss_free:continue
   remaining=add_cap-len(node['state']-ids);bound=len(lost_classes(node['evaluation']))
   if bound>remaining:
    audit.append(dict(primary=primary,layer=layer,parent=state_hash(node['state']),
     reason='PURE_ADD_LOWER_BOUND_EXCEEDS_REMAINING',lost_gold_classes=bound,remaining_additions=remaining,
     scope='source_distinct_add_only_suffix; delete_and_replace_remain_open'))
   families=successors(node['state'],node['evaluation'])
   for family in queues:
    for state,detail in families[family]:queues[family].append((node,set(state),detail))
  new_at_layer=0
  # Conditional pure-add bound redirects priority, never prunes add->delete routes.
  redirect=any(len(lost_classes(n['evaluation']))>add_cap-len(n['state']-ids) for n in nodes)
  while any(queues.values()) and new_at_layer<budget['max_new_states_per_layer']:
   family=None
   if redirect:
    preferred=[f for f in budget['action_order'] if f!='add']
    for _ in range(2):
     f=preferred[cursor%2];cursor+=1
     if queues[f]:family=f;break
   for _ in range(0 if family else 3):
    f=budget['action_order'][cursor%3];cursor+=1
    if queues[f]:family=f;break
   if family is None:break
   node,state,detail=queues[family].pop(0);h=state_hash(state)
   meta=dict(primary=primary,layer=layer,action=family,parent=state_hash(node['state']),
    path=node['path']+[dict(action=family,**detail)],add_cap=add_cap)
   reason=None
   if h in seen:reason='DUPLICATE_PATH_STATE'
   elif not state or len(state-ids)>add_cap or len(ids-state)>delete_cap:reason='ACTION_CAP'
   else:reason=structural(state)
   if reason:
    audit.append(dict(meta,bank_sha256=h,reason=reason));continue
   seen.add(h)
   result=test(state,meta)
   if result is None:
    audit.append(dict(meta,bank_sha256=h,reason='GLOBAL_OR_PRIMARY_BUDGET'));return
   ev,is_new=result;new_at_layer+=int(is_new);loss=len(losses(ev))
   nonmonotone=node['nonmonotone']+int(loss>=node['loss'])
   rejected=[]
   if loss-node['loss']>budget['max_loss_increase']:rejected.append('LOSS_INCREASE_BOUND')
   if nonmonotone>budget['max_nonmonotone_steps']:rejected.append('NONMONOTONE_STEP_BOUND')
   child=dict(state=state,evaluation=ev,nonmonotone=nonmonotone,path=meta['path'],layer=layer,loss=loss)
   if not rejected:choices.append(child)
   audit.append(dict(meta,bank_sha256=h,loss_events=loss,q15_repairs=repairs(ev),new_state=is_new,
    nonmonotone_steps=nonmonotone,reason=rejected or ['FRONTIER_ELIGIBLE']))
   yield child
  nodes=sorted(choices,key=rank)[:budget['beam_width']]
  audit.append(dict(primary=primary,layer=layer,reason='LAYER_COMPLETE',new_states=new_at_layer,
   retained=[dict(bank_sha256=state_hash(n['state']),loss=n['loss'],nonmonotone=n['nonmonotone']) for n in nodes],
   unexpanded_by_family={k:len(v) for k,v in queues.items()}))
  if not nodes:return

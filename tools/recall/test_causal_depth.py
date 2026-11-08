"""Causal graph fixtures exercise the production frontier, not a second runner.

These are unit engineering counterexamples, not sampled data or CPU cost models.
"""
import copy,json,sys,fcntl
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from causal_depth import frontier
from policy_contract import CAUSAL_DEFAULTS,validate_policy

def run(nodes,edges,*,budget=None,add_cap=6,delete_cap=2,global_cap=30,primary_cap=30):
 ids={1,2,3,4};initial=ids|{10};audit=[];measured=[];ledger={};counts={};new=0
 b=dict(copy.deepcopy(CAUSAL_DEFAULTS),**(budget or {}))
 def successors(s,e):
  out={k:[] for k in ['add','delete','replace']}
  for family,child in edges.get(tuple(sorted(s)),[]):out[family].append((set(child),dict(fixture_cause='actual_loss_edge')))
  return out
 def test(s,meta):
  nonlocal new
  key=tuple(sorted(s))
  if key not in ledger:
   if new>=global_cap or new>=primary_cap:return None
   new+=1;ledger[key]=dict(trial=new,search=meta,evaluation=nodes[key]);counts[str(meta['layer'])]=counts.get(str(meta['layer']),0)+1
   fresh=True
  else:fresh=False
  measured.append(dict(state=list(key),meta=meta,loss=nodes[key]['loss']))
  return nodes[key],fresh
 it=frontier(initial,nodes[tuple(sorted(initial))],ids=ids,primary=10,budget=b,add_cap=add_cap,delete_cap=delete_cap,
  losses=lambda e:set(range(e['loss'])),repairs=lambda e:e.get('repairs',1),lost_classes=lambda e:set(range(e.get('classes',1))),
  successors=successors,test=test,structural=lambda s:None,audit=audit)
 children=list(it)
 assert new<=min(global_cap,primary_cap)
 assert all(v<=b['max_new_states_per_layer'] for v in counts.values())
 return children,measured,audit,new

def key(s):return tuple(sorted(s))
base={1,2,3,4};a=base|{10};b=(a-{1});c=b|{11};d=c|{12};e=d|{13}
nodes={key(a):dict(loss=3,classes=2),key(b):dict(loss=2),key(c):dict(loss=3),key(d):dict(loss=1),key(e):dict(loss=0,repairs=4)}
edges={key(a):[('delete',b)],key(b):[('add',c)],key(c):[('add',d)],key(d):[('add',e)]}
r,m,audit,n=run(nodes,edges,budget=dict(max_new_states_per_layer=1,max_depth=4,max_nonmonotone_steps=1,max_loss_increase=1))
assert any(x['loss']==0 and len(x['state']-base)==4 and base-x['state']=={1} for x in r)
assert [x['meta']['action'] for x in m]==['delete','add','add','add']
assert [x['loss'] for x in m]==[2,3,1,0]
nonmonotone=run(nodes,edges,budget=dict(max_nonmonotone_steps=0,max_loss_increase=1))[0];assert not any(x['loss']==0 for x in nonmonotone)
increase=run(nodes,edges,budget=dict(max_nonmonotone_steps=1,max_loss_increase=0))[0];assert not any(x['loss']==0 for x in increase)
shallow=run(nodes,edges,budget=dict(max_depth=2))[0];assert not any(x['loss']==0 for x in shallow)
assert run(nodes,edges,global_cap=2)[3]==2
assert run(nodes,edges,primary_cap=1)[3]==1
# Conditional bound never deletes a route whose later deletion can repair it.
a=base|{10};b=(a|{11})-{1}
r,m,au,n=run({key(a):dict(loss=5,classes=5),key(b):dict(loss=0)}, {key(a):[('replace',b)]},add_cap=3,budget=dict(max_new_states_per_layer=1))
assert any(x['loss']==0 for x in r) and any(x.get('reason')=='PURE_ADD_LOWER_BOUND_EXCEEDS_REMAINING' for x in au)
# Even an add followed by delete remains searchable under that conditional bound.
b=a|{11};c=b-{1}
r,_,_,_=run({key(a):dict(loss=5,classes=5),key(b):dict(loss=4,classes=4),key(c):dict(loss=0)}, {key(a):[('add',b)],key(b):[('delete',c)]},add_cap=3)
assert any(x['loss']==0 for x in r)
# The beam width has an observed effect on a later feasible path.
a=base|{10};b=a|{11};c=a|{12};d=c|{13}
nodes={key(a):dict(loss=4),key(b):dict(loss=1),key(c):dict(loss=2),key(d):dict(loss=0)}
edges={key(a):[('add',b),('add',c)],key(c):[('add',d)]}
assert not any(x['loss']==0 for x in run(nodes,edges,budget=dict(beam_width=1))[0])
assert any(x['loss']==0 for x in run(nodes,edges,budget=dict(beam_width=2))[0])
# Action rotation exposes delete, replace, add across narrow layers.
a=base|{10};b=a-{1};c=(b|{11})-{2};d=c|{12}
nodes={key(a):dict(loss=4),key(b):dict(loss=3),key(c):dict(loss=2),key(d):dict(loss=0)}
edges={key(a):[('delete',b)],key(b):[('replace',c)],key(c):[('add',d)]}
assert [v['meta']['action'] for v in run(nodes,edges,budget=dict(max_new_states_per_layer=1))[1]]==['delete','replace','add']
print(json.dumps(dict(status='PASS',kind='ENGINEERING_CAUSAL_FIXTURES_NOT_SCIENTIFIC_SEARCH',causal_path_loss=[3,2,3,1,0],final_net_additions=4,final_net_deletions=1,atomic_replace=True,conditional_bound_preserves_add_then_delete=True,nonmonotone_bounds=True,layer_primary_global_caps=True,depth_and_beam_effective=True),indent=2))
# Extended maximum is operational: five compensation adds, six net additions.
a=base|{10};nodes={key(a):dict(loss=5,classes=1)};edges={};state=a
for z in range(11,16):
 child=state|{z};nodes[key(child)]=dict(loss=15-z,classes=1);edges[key(state)]=[('add',child)];state=child
assert any(x['loss']==0 and len(x['state']-base)==6 for x in run(nodes,edges,add_cap=6,budget=dict(max_depth=5,max_new_states_per_layer=1))[0])
assert not any(x['loss']==0 for x in run(nodes,edges,add_cap=3,budget=dict(max_depth=5,max_new_states_per_layer=1))[0])
print(json.dumps(dict(extended_six_add_effective=True,old_cap_three_blocks_same_fixture=True)))

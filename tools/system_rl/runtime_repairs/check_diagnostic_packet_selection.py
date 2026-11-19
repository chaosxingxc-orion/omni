"""Exercise actual target selector with frozen packets and admission negatives."""
import argparse,ast,copy,json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--selected-plan');a=p.parse_args()
tree=ast.parse(Path(a.source).read_bytes());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='diagnostic_targets')
def require(ok,message):
    if not ok:raise ValueError(message)
env={'require':require};exec(compile(ast.Module(body=[fn],type_ignores=[]),a.source,'exec'),env)
select=env['diagnostic_targets'];nodes=('P','S1','S2','D')
packets=[dict(stage=n,request_id=n+str(i),wire_sha256=n+str(i)) for n,count in zip(nodes,(1,10,4,1)) for i in range(count)]
cfg=dict(prepare_mode='finite_diagnostic_capture',native=dict(enabled=False),authorization=dict(native_training=False),
         diagnostic_evidence_only=dict(trainable=False,input_mode='frozen_node_inputs'),
         diagnostic_packet_selection=dict(schema='diagnostic-packet-selection-v1',inputs_per_node_per_draw={n:1 for n in nodes}))
selected=select(cfg,packets)
assert [p['request_id'] for p in selected]==[n+'0' for n in nodes]
assert all(select(cfg,packets)==selected for _ in range(4))
assert select({},packets) is packets
checks=['four nodes original first input retained','all four replicates use same input roster','default training unchanged']
for key,value in [('native',dict(enabled=True)),('authorization',dict(native_training=True)),
                  ('prepare_mode','training'),('diagnostic_evidence_only',dict(trainable=True,input_mode='frozen_node_inputs'))]:
    bad=dict(cfg,**{key:value})
    try:select(bad,packets)
    except ValueError:pass
    else:raise AssertionError(key)
    checks.append('reject '+key)
try:select(cfg,[p for p in packets if p['stage']!='D'])
except ValueError:pass
else:raise AssertionError('missing D')
checks.append('missing node fails before generation')
bad=copy.deepcopy(cfg);bad['diagnostic_packet_selection']['inputs_per_node_per_draw']['S2']=0
try:select(bad,packets)
except ValueError:pass
else:raise AssertionError('zero quota')
checks.append('zero quota rejected')
scoped=copy.deepcopy(cfg)
scoped['diagnostic_packet_selection']=dict(schema='diagnostic-packet-selection-v2',
    inputs_per_node_per_draw=dict(P=1,S1=1,S2=0,D=0))
assert [p['request_id'] for p in select(scoped,packets)]==['P0','S10']
checks.append('scoped v2 excludes S2/D before hydration and HTTP')
for changes in ({'P':0,'S1':0,'S2':0,'D':0},{'P':True,'S1':1,'S2':0,'D':0}):
    bad=copy.deepcopy(scoped);bad['diagnostic_packet_selection']['inputs_per_node_per_draw']=changes
    try:select(bad,packets)
    except ValueError:checks.append('invalid scoped quota rejected')
    else:raise AssertionError(changes)
from prompt_capacity import diagnostic_work
from logits_selection import validate
spec=dict(input_mode='frozen_node_inputs',packet_selection=scoped['diagnostic_packet_selection'],
    bounds=dict(groups=2,G=4,S1_folds=10,S2_types_upper=55,
        prompt_tokens_upper_per_call=20480,completion_tokens_upper_per_call=4096,
        prompt_tokens_upper_by_node=dict(P=6144,S1=3072,S2=1024,D=20480)))
assert diagnostic_work(spec)==(16,139264)
assert validate(dict(schema='omni-logits-input-selection-v2',
    inputs_per_node=dict(P=1,S1=1,S2=0,D=0),calls_per_input=4))==8
checks.append('scoped request/token/full-logits accounting agree')
from run_package import write,read,pin
actual=[]
if a.selected_plan:
    plan=read(a.selected_plan)
    for draw in plan['draws']:
        record=draw['record'];ref=record['online']['trajectory_export'];rows=read(ref)
        # Original exported order and identities, without loading model outputs
        # or choosing a branch based on success/reward.
        roster=[dict(stage=s['node'],branch=s['branch'],request_id=s['request_id'],
                     wire_sha256=s['wire_sha256']) for row in rows for s in row['spans']]
        chosen=select(cfg,roster)
        labels=record['offline_supervision']['labels']
        annotated={e['type'] for e in labels['entities']}
        actual.append(dict(reference_id=record['recording_ref_id'],source_ref=ref,
                           before=len(roster)*4,after=len(chosen)*4,selected=chosen,
                           legal_null_retained=any(x['stage']=='S2' and x['branch'] not in annotated for x in chosen)))
    assert actual and any(x['legal_null_retained'] for x in actual)
    write(Path(a.output)/'REAL-ROSTER.json',dict(passed=True,plan_ref=pin(a.selected_plan),records=actual,
          requests_before=sum(x['before'] for x in actual),requests_after=sum(x['after'] for x in actual),
          selector_uses_offline_labels=False,labels_used_for_coverage_audit_only=True,model_calls=0,
          actual_model_controls_verified=False))
write(Path(a.output)/'RESULT.json',dict(passed=True,checks=checks,NPU=False,model_calls=0,
    requests_per_draw_before=16*4,requests_per_draw_after=len(selected)*4,
    actual_model_controls_verified=False))
print(json.dumps(dict(passed=True,checks=len(checks))))

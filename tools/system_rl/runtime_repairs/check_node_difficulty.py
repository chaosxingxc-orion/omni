"""Actual original stratified sampler with counterexamples to system error buckets."""
import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--release',required=True);p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args()
release=Path(a.release);source=Path(a.source);root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,str(release/'tree/parents/jev-system-rl/implementation'))
from run_package import read,write
from storage import environment
environment(root,0)
sys.path.insert(0,str(release/'tree/parents/fresh-cohort-successor'))
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
module=load(source/'node_difficulty.py','node_difficulty')
cohort_path=source/'cohort.py'
if not cohort_path.exists():cohort_path=release/'tree/parents/fresh-cohort-successor/cohort.py'
cohort=load(cohort_path,'candidate_node_cohort');cohort.HERE=release/'tree/parents/fresh-cohort-successor'
records=[];population=[];outcomes=[];scores=[]
for i in range(8):
    src=dict(recording_ref_id=str(i),source_id='text'+str(i),source='fixture',domain='real' if i%2==0 else 'synthetic',split='TRAIN')
    outcome='success' if i<4 else 'error';population.append(src)
    outcomes.append(dict(src,status=outcome,gold_in_actual_pre_P_dynamic_retrieval=True))
    records.append(dict(src,sample_id=str(i),outcome=outcome,system_version='v',rollout_scope='system',online_gold_injected=False,online={},offline_supervision={}))
    value=int(i>=4)  # System errors are P successes, and vice versa.
    part=dict(complete=True,mask=1,coverage_at3=value,exact=value,joint_exact=value,
              conditional_on_supply=dict(complete=True,mask=1,f1=value),no_branch_path=0)
    scores.append(dict(schema='actual-node-statistics-v2',reference_id=str(i),sample_id=str(i),source_id=src['source_id'],
        offline_gold_only=True,annotation_sha256='fixture',gold_evidence={'fixture':True},evidence={'outcome_sha256':'fixture'},
        node_statistics={n:copy.deepcopy(part) for n in module.METRICS}))
pool=dict(schema='sealed-training-pool-v1',sealed=True,version='fixture',system_version='v',
    lineage={k:'fixture' for k in ('raw_dataset','system_release','knowledge','retrieval','prompt','rails','scorer')},
    official_train_scope='entire_official_TRAIN_real_and_synthetic',population_complete=True,official_train_count=8,
    synthetic_fixture=True,population_sha256=cohort.digest(population),cross_project_exclusions=False,
    population=population,outcomes=outcomes,records=records)
score_doc=dict(schema='frozen-node-difficulty-v1',screen_version='fixture',system_version='v',split='TRAIN',offline_gold_only=True,records=scores)
ref=write(root/'SCORES.json',score_doc)
sampling=dict(schema='dataset-sampling-config-v1',split='TRAIN',purpose='screen',scope='system',seed=17,sample_count=10,
    replacement='allow',insufficient='fail',format='separated-json-v1',pool_version='fixture',system_version='v',
    error_success={'error':.8,'success':.2},domain_mix={'real':.5,'synthetic':.5},
    source_balance=dict(key='source_id',mode='proportional',draw_policy='seeded_source_then_recording',without_replacement='renormalize_nonempty_sources'))
checks=[];original=copy.deepcopy(pool)
for node in module.METRICS:
    config=dict(sampling,node_difficulty=dict(node=node,scores_ref=ref))
    draws,stats=cohort.sample_original(pool,config)
    assert stats['node_outcomes']=={'error':8,'success':2}
    assert stats['selected_system_outcomes']=={'success':8,'error':2}
    assert all(d['record']==records[int(d['record']['sample_id'])] for d in draws)
    assert (draws,stats)==cohort.sample_original(pool,config)
    assert all(d['conditional_record_probability']>0 for d in draws)
    checks.append(node+' original sampler uses node errors, preserves system outcomes/probabilities and seeded replay')
# P position training: historic coverage cannot distinguish a top-1/2/3 hit.
position_doc=copy.deepcopy(score_doc)
expected=[]
for i,row in enumerate(position_doc['records']):
    part=row['node_statistics']['P'];rank=(i+int(i>=4))%4
    selected=[['b'],['a'],['b','a'],['b','c','a']][rank]
    part.update(coverage_at3=int(rank!=0),reward_calls=[dict(call_ids=['held'],labels={'intent':'a'},
        online={'retrieved_candidates':['a','b','c']},output={'selected':selected})])
    expected.append('success' if rank==1 else 'error')
position_ref=write(root/'POSITION.json',position_doc)
posspec=module.bind_objective(dict(node='P',scores_ref=position_ref),
                              {'scorer':{'metric_weights':{'P':{'position_reward_v1':1.}}}})
prepared_pos=module.prepare_selection(dict(sampling,node_difficulty=posspec),None,root)
proj,_,_=module.projection(records,pool,prepared_pos['node_difficulty'])
assert [r['outcome'] for r in proj]==expected
draws,stats=cohort.sample_original(pool,prepared_pos)
assert stats['difficulty_metric']=='position_reward_v1' and stats['node_outcomes']=={'error':8,'success':2}
assert all(d['difficulty']['outcome']==expected[int(d['record']['sample_id'])] for d in draws)
checks.append('position reward binds to original 80/20 sampler, ranks 2 and 3 are improvement cases')
for failure in ['missing action','unadmitted','contradictory score']:
    doc=copy.deepcopy(position_doc);part=doc['records'][1]['node_statistics']['P']
    if failure=='missing action':part.pop('reward_calls')
    elif failure=='unadmitted':part['reward_calls'][0]['online']['retrieved_candidates']=['b']
    else:part['position_reward_v1']=0.
    try:module.projection(records,pool,dict(posspec,scores_ref=write(root/(failure+'.json'),doc)))
    except ValueError:checks.append('reject '+failure)
    else:raise AssertionError(failure)
assert pool==original
assert cohort.sample_original(pool,sampling)[1]['outcomes']=={'error':8,'success':2}
checks.append('unconfigured system sampling unchanged and source population immutable')
def reject(name,document,node='P'):
    path=write(root/(str(len(checks))+'.json'),document)
    try:cohort.sample_original(pool,dict(sampling,node_difficulty=dict(node=node,scores_ref=path)))
    except (ValueError,KeyError):checks.append(name)
    else:raise AssertionError(name)
for name,change in [('reject TEST scores',{'split':'TEST'}),('reject changed upstream',{'system_version':'other'}),('reject missing recording',{'records':scores[:-1]}),('reject duplicate recording',{'records':scores+[scores[0]]})]:
    reject(name,dict(score_doc,**change))
bad=copy.deepcopy(score_doc);bad['records'][0]['node_statistics']['P']['complete']=False;reject('missing evidence not relabeled error',bad)
bad=copy.deepcopy(score_doc);bad['records'][0]['node_statistics']['P']['coverage_at3']=float('inf')
# JSON production writer rejects infinity itself; exercise numeric validator directly.
from unittest.mock import patch
with patch.object(module,'read',return_value=bad):
    try:module.projection(records,pool,dict(node='P',scores_ref=ref))
    except ValueError:checks.append('reject nonfinite metric')
    else:raise AssertionError('nonfinite')
bad=copy.deepcopy(score_doc);bad['records'][0]['node_statistics']['S2']['no_branch_path']=1
no_action=write(root/'NO-ACTION.json',bad)
draws,stats=cohort.sample_original(pool,dict(sampling,node_difficulty=dict(node='S2',scores_ref=no_action)))
assert stats['node_exclusions']=={'no_supplied_S2_action':1} and all(d['record']['sample_id']!='0' for d in draws)
checks.append('S2 zero actual branches separately counted, never sent to rollout')
conditional=copy.deepcopy(score_doc)
conditional['records'][0]['node_statistics']['S2'].update(complete=False,mask=0,reason='unsupplied_value_annotation_missing')
conditional_ref=write(root/'CONDITIONAL.json',conditional)
assert cohort.sample_original(pool,dict(sampling,node_difficulty=dict(node='S2',scores_ref=conditional_ref)))[1]['node_outcomes']=={'error':8,'success':2}
checks.append('S2 uses observed supplied-type score despite unrelated missing upstream annotation')
conditional['records'][0]['node_statistics']['S2']['conditional_on_supply']['mask']=0
reject('S2 missing supplied-type evidence rejected',conditional,node='S2')
next_input=load(source/'next_input.py','candidate_difficulty_next_input')
formal_pool=write(root/'FORMAL-SCREEN.json',dict(pool,official_train_count=119881,synthetic_fixture=False,unknown_count=0))
sampling_ref=write(root/'SAMPLING.json',sampling)
screen=dict(complete=True,population=119881,screen_ref=formal_pool,sampling_config_ref=sampling_ref,system_version='v')
input_config=dict(prepare_mode='bounded_train_probe',sampling=dict(node_difficulty=dict(node='P',scores_ref=ref)))
next_input.bind_screen_sampling(input_config,screen)
assert input_config['sampling']['node_difficulty']==dict(node='P',scores_ref=ref)
checks.append('original formal screen transition preserves explicit node difficulty')
# Run the fixed preparation entry over original node-contribution wire fields.
import hashlib
wire=[]
for row in scores:
    wire.append(dict(sample_id=row['sample_id'],reference_id=row['reference_id'],source_id=row['source_id'],
        node_metric_schema=row['schema'],node_statistics=row['node_statistics'],node_evidence=row['evidence'],
        node_annotation_sha256=row['annotation_sha256'],node_gold_evidence=row['gold_evidence']))
data=b''.join((json.dumps(r)+'\n').encode() for r in wire)
path=root/'CONTRIBUTIONS.jsonl';path.write_bytes(data)
records_ref=dict(path=str(path),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
evaluation=write(root/'EVALUATION.json',dict(split='TRAIN',system={'sha256':'v'}))
report=write(root/'CONTRIBUTION-REPORT.json',dict(schema='offline-node-contributions-v2',report=evaluation,records=records_ref))
screen_ref=write(root/'SCREEN.json',pool)
prepared=module.prepare_scores(screen_ref,[report],root/'PREPARED-SCORES.json')
assert module.prepare_scores(screen_ref,[report],root/'PREPARED-SCORES.json')==prepared
assert cohort.sample_original(pool,dict(sampling,node_difficulty=dict(node='P',scores_ref=prepared)))[1]['node_outcomes']=={'error':8,'success':2}
checks.append('fixed producer consumes original node contribution reports and deterministically feeds original sampler')
configured=dict(sampling,node_difficulty=dict(node='P',contribution_refs=[report]))
ready=module.prepare_selection(configured,screen_ref,root/'automatic')
assert 'scores_ref' in ready['node_difficulty'] and configured['node_difficulty'].get('scores_ref') is None
with patch.object(module,'prepare_scores',side_effect=AssertionError('repeated full join')):
    assert module.prepare_selection(configured,screen_ref,root/'automatic')==ready
checks.append('automatic CPU preparation caches contribution join across restart')
sampled=load(source/'sampled_training.py','candidate_auto_sampled')
from contextlib import contextmanager
import types
@contextmanager
def original_sampler_scope():
    yield types.SimpleNamespace(sample_original=cohort.sample_original)
spec=dict(schema='frozen-sampled-training-v1',sample_count=10,queries_per_cohort=2,
          resample='after_current_dataset_complete',zero_signal='consume_without_optimizer_step')
cfg=dict(execution_mode='formal',sampled_training=spec,max_optimizer_updates=5,root_seed=17)
state=dict(input_ref=write(root/'AUTO-INPUT.json',dict(sampling=configured)),
           screen=dict(screen_ref=screen_ref,system_version='v'))
with patch('fresh_adapter.scope',original_sampler_scope):
    sampled.prepare(cfg,state,root/'automatic-original')
dataset=read(state['sampled_dataset_ref'])
assert dataset['statistics']['node_outcomes']=={'error':8,'success':2}
assert 'scores_ref' in dataset['sampling']['node_difficulty']
assert 'record_ref' in read(dataset['selected_ref'])['draws'][0]
checks.append('original sampled prepare automatically joins scores and emits compact node draws')
for name,ev in [('TEST',dict(split='TEST',system={'sha256':'v'})),('FOREIGN',dict(split='TRAIN',system={'sha256':'other'}))]:
    e=write(root/(name+'-EVALUATION.json'),ev)
    r=write(root/(name+'-REPORT.json'),dict(schema='offline-node-contributions-v2',report=e,records=records_ref))
    try:module.prepare_scores(screen_ref,[r],root/(name+'-SCORES.json'))
    except ValueError:checks.append('producer rejects '+name+' contributions')
    else:raise AssertionError(name)
result=write(root/'RESULT.json',dict(passed=True,checks=checks,fixture=True,model_calls=0,optimizer_updates=0,
    actual_full_train_node_scores=False,production_dispatch=False))
print(json.dumps(result))

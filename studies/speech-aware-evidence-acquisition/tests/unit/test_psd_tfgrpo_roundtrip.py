"""Fixture wire receipts, real release replay, two target-bound PSD rounds."""
import copy
import hashlib
import json
import shutil
from pathlib import Path
import pytest
import intent_slot_workload as workload
import psd_tfgrpo_adapter as node
from test_intent_slot_workload import ROOT, config
from test_tfgrpo_external_reference import material
from test_external_exploration import job, write as expwrite
from test_external_teacher import network, answer
from speech_aware_evidence_acquisition.rails import external_exploration as ext, tfgrpo_external_reference as ref
from speech_aware_evidence_acquisition.rails import tfgrpo_external_release as fused_release, tfgrpo_reference_release as ref_release
from speech_aware_evidence_acquisition.rails.tfgrpo_external_compile import compile_preview as fused_compile
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import sha
from speech_aware_evidence_acquisition.rails.grounded_synthesis_fusion import project_group_evidence
from speech_aware_evidence_acquisition.rails.phase_material import load_phase_material_specs, render_static_instruction
from speech_aware_evidence_acquisition.rails.controllers import default_phase_program


def data_pin(root,name,value):
    path=root/name; path.write_text(json.dumps(value));return {'relpath':name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def round_run(root,repo,data,monkeypatch,kind,target,qid,consumer,selection,parent=None,binding=None):
    root.mkdir(); path,cfg=material(root)
    batch=ext.read(cfg['source_batch']['path']); doc=ext.read(cfg['source']['path'])['document']; old=batch['groups'][0]['query_id']
    group=batch['groups'][0]
    group['query_id']=qid; group['query']=group['query'].replace(old,qid)
    for row in group['rollouts']:
        row['trajectory']=row['trajectory'].replace(old,qid)
        if parent:row['experience_sha256']=parent['sha256']
    if parent:batch['parent']=parent;doc['parent']=parent;doc['rollout_parent_sha256']=parent['sha256'];cfg['parent']=expwrite(root/'parent.json',parent)
    doc['groups']=[{'query_id':qid,'group_sha256':sha(group),'evidence':project_group_evidence(group,evidence_view='full-query-v1')}]
    doc['source_batch_sha256']=sha(batch);doc['offline_stale_policy']=False
    cfg['source_batch']=expwrite(root/'batch.json',batch);cfg['source']=expwrite(root/'source.json',{'document':doc});cfg['query_ids']=[qid]
    if kind=='fused':
        jdir=root/'explore';jdir.mkdir();jobpath,out=job(jdir);j=ext.read(jobpath)
        ctx={'schema':'saea-tfgrpo-external-context-v1','model_id':'frozen-actor','parent':batch['parent'],'reward_mode':doc['reward_mode'],'rollout_policy_sha256':batch['policy_sha256'],'rollout_kind':batch['rollout_kind'],'offline_stale_policy':False,'bounds':{'max_operations':2,'max_items':48,'max_characters':10000},'psd':{'target':target,'source_batch':cfg['source_batch']}}
        if binding:ctx['psd']['parent_binding']=binding
        j.update(source=cfg['source'],query_ids=[qid],context=expwrite(root/'context.json',ctx),profile=ext.pin(ROOT/f'configs/learners/tfrl/tfgrpo-psd-{target}.json'));expwrite(jobpath,j)
        ext.prepare(jobpath)
        item={'id':'rule','target':target+'_experience_library','parent_sha256':batch['parent']['sha256'],'evidence_query_ids':[qid],'relative_feedback':[{'query_id':qid,'higher_trajectory_id':'t1','lower_trajectory_id':'t0','explanation':'fixture contrast'}],'hypothesis':'fixture','operations':[{'operation':'ADD','id':None,'content':'Use observed cues for '+target+' in '+root.name+'.'}],'counterexample':'fixture','validation':'fixture'}
        other=copy.deepcopy(item);other['id']='other';other['operations'][0]['content']='Compare available evidence.'
        _,_,caller,req=ext.load_prepared(out);network(caller,monkeypatch,[answer(json.dumps({'algorithm':'tfgrpo','candidates':[item,other]}))]);caller(req);ext.collect(out)
        spec={'schema':'saea-tfgrpo-external-compile-v1','prepared':ext.pin(out/'prepared.json'),'collected':ext.pin(out/'collected.json'),'candidate_index':0}; candidate=fused_compile(spec);release=fused_release
    else:
        cfg['psd_target']=target;cfg['teacher']['research_prompt']=ref.pin(ROOT/f'configs/prompts/learner/tfrl/tfgrpo-reference-psd-{target}.md')
        if binding:cfg['parent_binding']=binding
        expwrite(path,cfg);ref.prepare(path);out=Path(cfg['output_root'])
        ops=json.dumps([{'operation':'ADD','id':None,'content':'Use observed cues for '+target+' in '+root.name+'.'}])
        network(ref.load(out)[-1],monkeypatch,[answer('low','a'),answer('high','b'),answer('<Experiences>1. Compare evidence.</Experiences>','c'),answer(ops,'d'),answer(ops,'e')]);ref.run(out)
        spec={'schema':'saea-tfgrpo-reference-compile-v1','prepared':ref.pin(out/'prepared.json'),'collected':ref.pin(out/'collected.json')};candidate=ref_release.compile_preview(spec);release=ref_release
    prefix=root.name
    cp=data_pin(data,prefix+'-candidate.json',candidate);sp=data_pin(data,prefix+'-spec.json',spec)
    def rp(rel):return {'relpath':rel,'sha256':hashlib.sha256((repo/rel).read_bytes()).hexdigest()}
    document={'schema':release.SCHEMA,'candidate':cp,'source_manifest':sp,'consumer_roster':rp(fused_release.ROSTER),'learning_dataset':rp(fused_release.DATASET),'consumer_ids':[consumer],'selection_ids':[selection]}
    evidence=({g['query_id']:g['group_sha256'] for g in candidate['relative_evidence']['groups']} if kind=='fused' else candidate['evidence_groups'])
    review={'schema':release.REVIEW_SCHEMA,'decision':'eligible-for-development-evaluation','reviewer':'fixture','procedure':'fixture assessment','bindings':{k:v for k,v in document.items() if k!='schema'},'items':{k:{'content_sha256':sha(v),'decision':'supported','evidence':[{'query_id':qid,'group_sha256':evidence[qid]}],'reason':'fixture'} for k,v in candidate['items'].items()}}
    document['review']=data_pin(data,prefix+'-review.json',review);data_pin(data,prefix+'-release.json',document)
    params=release.prepare(candidate_path=data/(prefix+'-candidate.json'),source_manifest=data/(prefix+'-spec.json'),candidate_release=data/(prefix+'-release.json'),repo=repo,data=data,base_template='probes/base.txt',template='probes/'+prefix+'.txt',consumer_ids=[consumer],model_id='frozen-actor')
    pp=data/prefix;pp.mkdir();params_path=pp/'params.json';params_path.write_text(json.dumps(params))
    binding={'schema':'saea-psd-tfgrpo-parent-v1','repo':str(repo),'data':str(data),'params':ext.pin(params_path)}
    body={'schema':'saea-tfgrpo-experience-v1','model_id':'frozen-actor','items':candidate['items'],'learned_query_ids':candidate['learned_query_ids'],'parent_sha256':candidate['sha256'],'status':'candidate','deployment_accepted':False}
    return candidate,{**body,'sha256':sha(body)},binding,params_path,out


@pytest.mark.parametrize('kind',['fused','reference'])
@pytest.mark.parametrize('target',['slot','decision'])
def test_two_round_target_bound_source_to_actual_node(tmp_path,monkeypatch,kind,target):
    repo=tmp_path/'repo';data=tmp_path/'data';repo.mkdir();data.mkdir()
    for rel in (fused_release.ROSTER,fused_release.DATASET):
        dest=repo/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dest)
    base=repo/'configs/probes/base.txt';base.parent.mkdir(parents=True,exist_ok=True);base.write_text('Fixture base.')
    rows=json.loads((repo/fused_release.DATASET).read_text())['items']; roster=json.loads((repo/fused_release.ROSTER).read_text())['items'];consumer=roster[0]['sample_id'];selection=roster[1]['sample_id']
    excluded={':'.join(x['sample_id'].split(':')[:3]) for x in roster}; chosen=[]
    for row in rows:
        key=':'.join(row['sample_id'].split(':')[:3])
        if key not in excluded and all(row['audio_sha256'] != x['audio_sha256'] and key != ':'.join(x['sample_id'].split(':')[:3]) for x in chosen):chosen.append(row)
        if len(chosen)==3:break
    first,parent,binding,_,_=round_run(tmp_path/'round1',repo,data,monkeypatch,kind,target,chosen[0]['sample_id'],consumer,selection)
    from speech_aware_evidence_acquisition.rails.tfgrpo_psd_parent import verify
    with pytest.raises(ValueError):verify(parent, 'decision' if target=='slot' else 'slot','frozen-actor',binding)
    bad=copy.deepcopy(parent);bad['learned_query_ids']=[]
    with pytest.raises(ValueError):verify(bad,target,'frozen-actor',binding)
    second,_,_,params_path,out=round_run(tmp_path/'round2',repo,data,monkeypatch,kind,target,chosen[1]['sample_id'],consumer,selection,parent,binding)
    assert set(second['learned_query_ids'])=={chosen[0]['sample_id'],chosen[1]['sample_id']}
    assert second['ancestor_usage_identities']==[first.get('usage_identity',first.get('request_sha256'))]
    descriptor={'schema':node.SCHEMA,'target':target,'source':data_pin(data,'final-params.json',json.loads(params_path.read_text())),'template':{'relpath':'configs/probes/node.txt','sha256':''}}
    original_read=workload.read_asset
    monkeypatch.setattr(workload,'read_asset',lambda repo,rel:original_read(ROOT,'configs/model/bf16-thinking-tfgrpo-tp4-ctx48k-v1.json' if rel=='configs/model/frozen-actor.json' else rel))
    view={'split_name':'fixture','items':[chosen[2]]}
    monkeypatch.setattr(workload.NodeWorkload,'material_checked',lambda *a:({'source_roster':{}},{'roster':{'roster_id':'fixture','items':[{'audio_seconds':1}],'split_identity_hash':'a'*64},'view':view,'source_baseline':original_read(ROOT,workload.observation.SOURCE_BASE)}))
    monkeypatch.setattr(workload.NodeWorkload,'runtime_binding',lambda *a:{'model_fragment':'frozen-actor','service_session':'fixture'})
    c=config();c.update(node_arm='enhanced-p-slot-decision',pipeline={'protocol_version':'v2','node_rails':'sdk-handoff-v1','answer_only_fallback':'budgeted-v1'},sample_ids=[chosen[2]['sample_id']],population={'dataset':'slurp-speech-native-evo-fold-a-1080','dataset_sha256':'x','roster':{},'source_baseline':{}})
    base=workload.NodeWorkload(c).derive(repo,data,{})[0];role='slot' if target=='slot' else 'solver';original=ROOT/'configs'/base['phase_material'][role]['instruction_template'];dest=repo/'configs'/base['phase_material'][role]['instruction_template'];dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(original.read_bytes())
    text=node.verify_and_render(repo,data,descriptor,c['sample_ids'],view,'frozen-actor',original.read_text());p=repo/descriptor['template']['relpath'];p.write_text(text);descriptor['template']['sha256']=hashlib.sha256(p.read_bytes()).hexdigest();c['tfgrpo_experience']=descriptor
    result=workload.NodeWorkload(c).derive(repo,data,{})[0];specs=load_phase_material_specs(result,default_phase_program(result['policy']))
    rendered=render_static_instruction(specs[role],repo,{},data)
    assert all(v in rendered for v in second['items'].values()) and len(second['items'])>len(first['items'])
    other_role='solver' if role=='slot' else 'slot'
    other_path=repo/'configs'/base['phase_material'][other_role]['instruction_template'];other_path.parent.mkdir(parents=True,exist_ok=True);other_path.write_bytes((ROOT/'configs'/base['phase_material'][other_role]['instruction_template']).read_bytes())
    bad=copy.deepcopy(c);bad['tfgrpo_experience']['target']='decision' if target=='slot' else 'slot'
    with pytest.raises(ValueError):workload.NodeWorkload(bad).derive(repo,data,{})
    # Source batch tampering propagates through real candidate replay.
    source_path=Path(ext.read(data/'round2-spec.json')['prepared']['path']).parent
    if kind == 'fused':
        manifest=ext.read(source_path/'prepared.json'); source_file=Path(manifest['source']['path'])
    else:
        source_file=Path(ext.read(source_path/'job.json')['source']['path'])
    saved=source_file.read_bytes();source_file.write_bytes(saved+b' ')
    with pytest.raises(ValueError):workload.NodeWorkload(c).derive(repo,data,{})
    source_file.write_bytes(saved)
    # Wire tamper must be detected by the real replay, including during node derivation.
    wire=next(out.glob('**/response/response.bin'));wire.write_bytes(wire.read_bytes()+b' ')
    with pytest.raises(ValueError):workload.NodeWorkload(c).derive(repo,data,{})

"""Immutable prompt mechanisms and genuine same-policy stochastic rollouts."""
import argparse,copy,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'delivery/s-tfrl-rolling-v1'
LOGICAL_OUT=OUT
OUT=Path(os.environ.get('S_ROLLING_CPU_OUTPUT',str(OUT)))
def logical(path):
    return str((LOGICAL_OUT/path.relative_to(OUT)).relative_to(ROOT))
PARENT=ROOT/'delivery/s-single-intent-fewshot-v1/prepared-v5-dynamic'
sys.path.insert(0,str(ROOT/'delivery/s-single-intent-fewshot-v1/code'))
from common import canonical,sha,read,save,lines,require
from speech_aware_evidence_acquisition.core.model.vllm_transport import vllm_request_body

EXPERIENCES={
 'compact-json-v1':dict(mechanism='compact_complete_json_serialization',
  instruction='Serialization: write the complete response as compact JSON on one line, without indentation or newline padding. Include every required field, including evidence_refs and unresolved_fields; immediately close all JSON arrays and objects when their values are complete, then stop. Preserve all audio-supported entities and applicable references. Do not spend output tokens on whitespace.',
  evidence=['delivery/s-single-intent-execution-v1/MAIN-LENGTH-DIAGNOSTIC.private.json'],
  counterevidence='length can also be genuine output/reasoning exhaustion; compactness alone may not resolve other errors',
  falsification='same545 length/whitespace failures do not fall, or semantic/entity/reference regressions outweigh validity repairs'),
 'absent-slot-v1':dict(mechanism='serialize_absent_slot_by_omission',
  instruction='For this supplied candidate, include an entity only if a nonempty spoken span in the current audio fills that slot. When no spoken span fills a slot, omit that entity; when none of the candidate slots have a spoken filler, return entities: []. Do not use an empty string to represent an absent slot. A genuinely spoken value may equal its slot type name; do not suppress it merely because the strings match. Keep supported repeated mentions and all required response fields.',
  evidence=['delivery/s-single-intent-execution-v1/MAIN-EARLY-ERRORS.private.json','delivery/s-ace-teacher-v1/revision-v2/teacher/MAIN-REVIEW.json'],
  counterevidence='parent already permits empty entities; prior combined nonempty/schema protocol intervention regressed 118 to95',
  falsification='empty fillers persist without net repairs, or supported nonempty fillers are newly omitted'),
}

def append_instruction(packet,instruction):
    """Only append the same procedure at the two existing instruction positions."""
    addition='\n\n'+instruction
    packet['wire']['messages'][0]['content']+=addition
    blocks=packet['wire']['messages'][1]['content']
    marker='\n\nCandidate-conditioned pipeline context:'
    require(marker in blocks[-1]['text'],'original context delimiter')
    prefix,context=blocks[-1]['text'].split(marker,1)
    blocks[-1]['text']=prefix+addition+marker+context
    return packet

def prepare(name,seed=None):
    start,cpu=time.monotonic(),time.process_time()
    stochastic=seed is not None
    compiled_path=LOGICAL_OUT/'config'/('teacher-compiled-'+name+'.json')
    compiled=read(compiled_path) if compiled_path.exists() else None
    arm='single_intent_fewshot' if stochastic or compiled and compiled['parent_version'].endswith('/single_intent_fewshot') else 'single_intent_no_examples'
    parentcfg=read(PARENT/arm/'native-UNBOUND.json')
    folder=OUT/('grpo-reserve' if stochastic else 'candidates')/name
    require(not folder.exists(),'fresh immutable output required')
    folder.mkdir(parents=True)
    for f in ['snapshot.projected.private.json','samples.private.jsonl']:
        save(folder/f,lines(PARENT/f) if f.endswith('.jsonl') else read(PARENT/f),jsonl=f.endswith('.jsonl'))
    if not stochastic and name not in EXPERIENCES:
        proposal=read(LOGICAL_OUT/'config'/('teacher-compiled-'+name+'.json'))
        require(proposal['admission']=='ACCEPT_SINGLE_MECHANISM_HYPOTHESIS' and proposal['target']=='instruction_append','bounded reviewed append')
        require(proposal['parent_version']=='prepared-v5-dynamic/'+arm,'one actual parent only')
        EXPERIENCES[name]=proposal['experience']
    experience=None if stochastic else EXPERIENCES[name]
    if experience:save(folder/'experience.json',experience|dict(entry_id=name,operation='ADD',status='PROPOSED_UNVALIDATED',parent_version='prepared-v5-dynamic/'+arm))
    manifest=[]
    for prior in lines(PARENT/arm/'manifest.private.jsonl'):
        raw=(ROOT/prior['payload']).read_bytes();require(sha(raw)==prior['packet_sha256'],'parent packet pin')
        packet=json.loads(raw);before=copy.deepcopy(packet['wire'])
        if stochastic:
            packet['wire'].update(seed=seed,temperature=0.7,top_p=0.9)
            normalized=copy.deepcopy(packet['wire']);normalized.pop('seed')
            parent_normalized=copy.deepcopy(before);parent_normalized.pop('seed');parent_normalized.update(temperature=0.7,top_p=0.9)
            require(normalized==parent_normalized,'different policy within sampling group')
        else:
            append_instruction(packet,experience['instruction'])
        require(packet['wire']['messages'][1]['content'][0]==before['messages'][1]['content'][0],'audio bytes')
        require(packet['wire']['response_format']==before['response_format'],'response schema')
        path=folder/arm/'packets'/Path(prior['payload']).name;save(path,packet)
        row=copy.deepcopy(prior);row.update(payload=logical(path),packet_sha256=sha(path.read_bytes()),
            abstract_wire_sha256=sha(canonical(packet['wire'])),wire_sha256=sha(canonical(vllm_request_body(packet['wire'],packet['wire']['model']))))
        manifest.append(row)
    require(len(manifest)==1635,'1635 fixed branches')
    path=folder/arm/'manifest.private.jsonl';save(path,manifest,jsonl=True)
    cfg=copy.deepcopy(parentcfg);cfg.update(run_id='S-ROLLING-'+name.upper(),attempt_id='v1',
        manifest=logical(path),manifest_sha256=sha(path.read_bytes()),prepared=logical(folder),
        snapshot=logical(folder/'snapshot.projected.private.json'),samples=logical(folder/'samples.private.jsonl'))
    save(folder/arm/'native-UNBOUND.json',cfg)
    if stochastic:
        policy=read(LOGICAL_OUT/'config/grpo-v1.json')
        # Seed is the stochastic realization, not a different strategy.
        policy_id=sha(canonical({k:v for k,v in policy.items() if k!='seeds'}))
        save(folder/'GROUP-MEMBERSHIP.private.jsonl',[dict(sample_id=r['source_sample_id'],branch_id=r['sample_id'],
            candidate_index=r['candidate_index'],policy_id=policy_id,group_id=policy_id+':'+r['sample_id'],
            sample_group_id=policy_id+':'+r['source_sample_id'],seed=seed,group_size=len(policy['seeds']),
            expected_seeds=policy['seeds'],wire_sha256=r['wire_sha256']) for r in manifest],jsonl=True)
        save(folder/'SAMPLING.json',dict(policy_id=policy_id,seed=seed,temperature=0.7,top_p=0.9,
            group_size=len(policy['seeds']),group_member='same545 samev5fewshot policy each seed',
            audio_encodes=0,P_calls=0,model_parameters_frozen=True,group_reward=policy['reward'],
            historical_parent_temperature=0,historical_parent_is_group_member=False))
    save(folder/'CPU-PREPARED.json',dict(status='PREPARED_NEEDS_FIXED_IMAGE_BUDGET',name=name,samples=545,branches=1635,
        parent='prepared-v5-dynamic/'+arm,arm=arm,seed=seed,policy_type='same-policy stochastic reserve' if stochastic else 'single mechanism instruction',
        wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu,model_calls=0,audio_encodes=0,
        inference_completed=0,inference_total=1635,cards=0,instances=0,inference_ETA=None,money=None))
    print(json.dumps(read(folder/'CPU-PREPARED.json')),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--seed',type=int);a=p.parse_args();prepare(a.name,a.seed)

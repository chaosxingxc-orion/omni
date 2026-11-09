"""Project already frozen real TRAIN P-correct partitions; no new sampling or P."""
import argparse,base64,collections,copy,json,os,sys,time,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'delivery/s-tfrl-rolling-v1'
sys.path.insert(0,str(ROOT/'delivery/s-single-intent-fewshot-v1/code'))
from common import read,save,sha,lines,canonical,require,validate_packet,annotations,source_id,load
from speech_aware_evidence_acquisition.rails import candidate_handoff as hb
from speech_aware_evidence_acquisition.rails import intent_slot_pipeline as ip
from speech_aware_evidence_acquisition.rails import knowledge as kh
from speech_aware_evidence_acquisition.core.model.vllm_transport import vllm_request_body

def main(partition):
    start,cpu=time.monotonic(),time.process_time()
    name='train-'+partition.lower()+'-single-v1'
    logical=OUT/'train-supply'/name
    hot=Path('/var/tmp/omni-s-tfrl-rolling-v1/train-supply')/name
    require(not hot.exists() and not logical.exists(),'immutable TRAIN successor')
    source=ROOT/'delivery/s-train-dataset-v1/results'
    frozen=read(source/'FROZEN.json');require(frozen['status']=='FIRST_BATCH_DATASET_FROZEN','existing split freeze')
    rows=lines(source/'splits'/(partition+'.manifest.private.jsonl'))
    expected=frozen[partition] if partition in ('S_fit','S_valid') else frozen['remainder']
    evaluation_only=partition=='S_valid'
    require(len(rows)==expected,'frozen N')
    # Exact old source/quality isolation and fixed split order are reused.
    offline={r['sample_id']:r for r in lines(source/'offline/sample-ledger.private.jsonl')}
    # Retain all frozen IDs and their hold flags. Hold rows can be collected but
    # must never become legal TRAIN knowledge or teacher update examples.
    require(all(r['optimizer_allowed'] for r in rows),'existing optimization flag')
    require(all(offline[r['sample_id']]['P_valid'] and offline[r['sample_id']]['P_Top3_contains_gold']
        and not offline[r['sample_id']]['exclusion_reasons'] and not offline[r['sample_id']]['optimization_exclusion_reasons'] for r in rows),'existing P/source/quality')
    snapshot=kh.KnowledgeSnapshot.from_frozen_json(read(ROOT/'delivery/s-single-intent-fewshot-v1/prepared-v5-dynamic/snapshot.projected.private.json'))
    values={v.value_id:v for v in snapshot.values};bycid=collections.defaultdict(list)
    for v in snapshot.values:bycid[v.value_id.split('::')[0]].append(v)
    save(hot/'snapshot.projected.private.json',snapshot.frozen_json())
    instruction=(ROOT/read(ROOT/'delivery/s-single-intent-fewshot-v1/config/prepare.json')['instruction']).read_text()
    cfgbase=read(ROOT/'delivery/s-single-intent-fewshot-v1/prepared-v5-dynamic/single_intent_no_examples/native-UNBOUND.json')
    pipeline=dict(schema='saea-intent-slot-pipeline-v2',slot_mode='per_candidate',max_candidates=3,
        candidate_outside_recovery=False,duplicate_candidate_policy='group_v2',node_rails='sdk-handoff-v1',answer_only_fallback='budgeted-v1')
    sys.path.insert(0,str(ROOT/'delivery/s-correct-p-data-v1/code'));import materialize_s
    jobs={j['sample_id']:j for j in read(ROOT/'delivery/s-train-p-collection-v1/run-B/inputs/prepared-jobs.private.json')}
    priorbudget={r['sample_id']:r for r in read(ROOT/'delivery/s-train-p-collection-v1/run-B/budget.json')['records']}
    provenance={r['sample_id']:r for r in read(ROOT/'delivery/s-train-p-collection-v1/offline/provenance.private.json')}
    formal=read(OUT/'evidence/train-budget-parent/FIXED-TOKENIZER-VERIFIED.json')
    require(formal['status']=='PASS_FIXED_IMAGE_ACTUAL_TOKENIZER' and formal['n']==7094,'actual P budget receipt')
    require(sha((ROOT/'delivery/s-train-p-collection-v1/run-B/budget.json').read_bytes())==formal['budget_sha256'],'P budget receipt pin')
    metadata={r['slurp_id']:r for r in lines(ROOT/'delivery/recall-card-source-audit-v1/input/train-metadata.private.jsonl')}
    manifest=[];samples=[];budget_records=[];gold=[];proof=[]
    arm='single_intent_no_examples';snap=snapshot.content_hash()
    rel=lambda p:str((logical/p.relative_to(hot)).relative_to(ROOT))
    for r in rows:
        raw=(ROOT/r['payload']).read_bytes();require(sha(raw)==r['packet_sha256'],'original TRAIN packet pin')
        original=json.loads(raw);sid=r['sample_id'];j=jobs[sid];old=priorbudget[sid]
        source_packet=hot/'source-packets'/(sha(sid.encode())+'.private.json')
        source_packet.parent.mkdir(exist_ok=True,parents=True)
        with source_packet.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        require(original['candidate_ids']==r['candidate_ids'],'original real P order')
        audio=original['wire']['messages'][1]['content'][0]
        require(provenance[sid]['original_audio']['audio_sha256']==original['audio_sha256']==j['audio_sha256'],
                'original source audio identity')
        require(sha(base64.b64decode(audio['input_audio']['data'],validate=True))==provenance[sid]['transport_audio_sha256']
                ==j['transport_audio_sha256'],'unchanged transport WAV identity')
        budget_records.append(dict(sample_id=sid,inherited_P_input_upper=old['input_upper'],
            audio_object_sha256=sha(canonical(audio)),source_P_record_sha256=sha(canonical(old))))
        g=offline[sid];record=metadata[source_id(sid)];canonical_intent=record['scenario']+'_'+record['action']
        require(canonical_intent in r['candidate_ids'],'real P contains canonical original intent')
        gold.append(dict(sample_id=sid,gold_intent=canonical_intent,original_intent_field=g['original_intent_OFFLINE_ONLY'],
            reviewed_hold=r['reviewed_hold'],teacher_update_allowed=not r['reviewed_hold'] and not evaluation_only,evaluation_only=evaluation_only,
            gold=[dict(type=e['type'],filler=e['filler'].lower()) for e in annotations(record)]))
        proof.append(dict(sample_id=sid,original_packet=rel(source_packet),original_source_packet=r['payload'],original_packet_sha256=r['packet_sha256'],
            P_response_sha256=r['P_response_sha256'],P_wire_sha256=r['P_wire_sha256'],P_source_proof=g['P_source_proof'],
            source_group=g['source_group'],partition=partition,quality_unchanged=True,reviewed_hold=r['reviewed_hold'],
            teacher_update_allowed=not r['reviewed_hold'] and not evaluation_only,evaluation_only=evaluation_only))
        sample=dict(sample_id=sid,cohort='fit',dataset_partition=partition,candidate_ids=r['candidate_ids'],branches=[],
            reviewed_hold=r['reviewed_hold'],teacher_update_allowed=not r['reviewed_hold'] and not evaluation_only,evaluation_only=evaluation_only)
        for i,cid in enumerate(r['candidate_ids']):
            key=sid+'#S-'+str(i);vs=bycid[cid];allowed=sorted(v.value_id.split('slot-contract:')[1] for v in vs)
            for v in vs:require(sid not in v.source_sample_ids,'query/card source overlap')
            knowledge=[dict(id=v.value_id,snapshot_sha256=snap,text=v.content,
                scope=dict(facets=list(v.associated_facet_ids),sources=list(v.source_sample_ids),provenance=list(v.provenance_refs)),
                annotations={},target_role='slot') for v in vs]
            handoff=hb.build_handoff(sample_id=key,source_role='perception',target_role='slot',
                source_response_sha256=original['handoff']['source_response_sha256'],snapshot_sha256=snap,
                candidate_ids=[cid],judgments=[dict(candidate_id=cid,status='candidate')],knowledge=knowledge)
            parts=[copy.deepcopy(audio)]+[dict(type='text',text=v.content,evidence_id=v.value_id,
                source_uri='knowledge-snapshot://'+snap+'/'+v.value_id,knowledge_role='value') for v in vs]
            inventory=ip.evidence_inventory(sample_id=key,speech_packet_id=sid,speech_sha256=original['audio_sha256'],
                messages=[dict(content=parts)],candidate_ids=[cid])
            context=ip.prompt_context(pipeline,{'intent_slot_context':dict(candidates=[dict(candidate_id=cid,
                sources=[dict(intent_id=cid,evidence_refs=[],unresolved_fields=[])])],evidence_inventory=inventory)},'slot')
            fmt=materialize_s.slot_format(allowed or ['__unreachable_empty_intent__'],[cid],ip)
            if not allowed:
                es=fmt['json_schema']['schema']['properties']['candidates']['items']['properties']['entities']
                es['maxItems']=0;es['items']['properties']['type']={'type':'string'}
            parts.append(dict(type='text',text=instruction+context));wire=copy.deepcopy(original['wire'])
            wire.update(messages=[dict(role='system',content=instruction),dict(role='user',content=parts)],response_format=fmt)
            packet=dict(sample_id=key,source_sample_id=sid,candidate_ids=[cid],candidate_index=i,
                audio_sha256=original['audio_sha256'],handoff=handoff,wire=wire,allowed_types=allowed,arm=arm,examples=[],
                evidence_kind='newly_materialized_not_historical_S_execution')
            validate_packet(packet,selected=[cid],expected_audio=audio,snapshot=snap,values={k:v.content for k,v in values.items()},handoff_module=hb)
            path=hot/arm/'packets'/(sha(key.encode())+'.private.json');save(path,packet)
            manifest.append(dict(cohort='fit',dataset_partition=partition,role='TRAIN_teacher_feedback_only',sample_id=key,
                source_sample_id=sid,candidate_index=i,candidate_ids=[cid],arm=arm,payload=rel(path),packet_sha256=sha(path.read_bytes()),
                reviewed_hold=r['reviewed_hold'],teacher_update_allowed=not r['reviewed_hold'] and not evaluation_only,evaluation_only=evaluation_only,
                abstract_wire_sha256=sha(canonical(wire)),wire_sha256=sha(canonical(vllm_request_body(wire,wire['model']))),
                P_response_sha256=r['P_response_sha256'],P_wire_sha256=r['P_wire_sha256'],allowed_types=allowed,
                original_packet_sha256=r['packet_sha256'],original_audio_sha256=original['audio_sha256'],
                handoff_sha256=handoff['handoff_sha256'],wire_adapter=cfgbase['wire_adapter'],status='input_ready',S_executed=False))
            sample['branches'].append(dict(branch_key=key,candidate_id=cid,candidate_index=i))
        samples.append(sample)
    parent=dict(status='PASS',fixed_image_tokenizer_verified=True,image=formal['image'],records=budget_records,
        original_fixed_image_receipt=formal,semantics='unchanged real P input_upper + unchanged transport audio identity; inherited bound')
    save(hot/'budget-parent.json',parent);save(hot/'samples.private.jsonl',samples,jsonl=True)
    save(hot/'offline/gold.private.jsonl',gold,jsonl=True);save(hot/'source-proof.private.jsonl',proof,jsonl=True)
    path=hot/arm/'manifest.private.jsonl';save(path,manifest,jsonl=True)
    cfg=cfgbase|dict(run_id='S-ROLLING-'+name.upper(),attempt_id='v1',manifest=rel(path),manifest_sha256=sha(path.read_bytes()),
        expected_rows=len(manifest),full_denominator=len(samples),dataset_partition=partition,
        evaluation_only=evaluation_only,teacher_update_allowed=not evaluation_only,
        reference_CPU_only=rel(hot/'offline/gold.private.jsonl'),
        snapshot=rel(hot/'snapshot.projected.private.json'),prepared=rel(hot),samples=rel(hot/'samples.private.jsonl'),
        budget_parent=dict(path=rel(hot/'budget-parent.json'),sha256=sha((hot/'budget-parent.json').read_bytes())))
    save(hot/arm/'native-UNBOUND.json',cfg)
    save(hot/'CPU-PREPARED.json',dict(status='TRAIN_N_INPUTS_PREPARED_NEEDS_FIXED_IMAGE',samples=len(samples),branches=len(manifest),
        fixed_original_partition=partition,P_recollection=0,audio_encodes=0,source_6366_top3_correct=True,
        wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu,model_calls=0))
    pending=logical.with_name(name+'.pending');pending.mkdir(parents=True)
    with tarfile.open(Path('/var/tmp/omni-s-tfrl-rolling-v1')/(name+'.tar'),'w') as tf:
        for p in hot.rglob('*'):
            if p.is_file():tf.add(p,arcname=str(p.relative_to(hot)),recursive=False)
    with (Path('/var/tmp/omni-s-tfrl-rolling-v1')/(name+'.tar')).open('rb') as f:
        import subprocess;subprocess.run(['tar','xf','-','-C',str(pending)],stdin=f,check=True)
    import subprocess;subprocess.run(['sync','-f',str(pending)],check=True);os.rename(pending,logical)
    print(json.dumps(read(logical/'CPU-PREPARED.json')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--partition',choices=['S_fit','S_valid','eligible_remainder'],required=True);a=p.parse_args();main(a.partition)

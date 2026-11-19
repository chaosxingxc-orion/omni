"""Publish already-scored, natural engine actions using the original TRAIN consumer.

This bridge only accepts a bounded initial P cohort. It never submits a request,
computes replacement actions/logprobs, advances a Clock or launches a trainer.
"""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys


def publish(c, parents):
    sys.path.insert(0,str(parents/'jev-system-rl/implementation'))
    from fresh_adapter import scope
    from compare_fresh_group import compare
    from score_fresh_p_actions import original_parser
    with scope() as cohort:
        from cohort_io import read, pin, require, digest, save
        from cohort_capture import normalized, wire_assets, labels_for, audio_spans
        from official import sources, annotation
        from policy import validate
        root=Path(c['prepared'])
        pending=read(pin(root/'PENDING-CAPTURE.json'))
        config=read(pending['input_ref']); plan=read(pending['selected_cohort_ref'])
        consumer=read(pin(root/'CONSUMER.json')); clock=read(consumer['clock_state_ref'])
        learning=read(config['learning_config_ref'])
        current,reference=validate(config['policy'],learning,clock)
        require(current['kind']==reference['kind']=='base' and
                plan['prepare_mode']=='bounded_train_probe' and not plan['synthetic_fixture'] and
                not plan.get('diagnostic_evidence_only') and not any(g['consumed'] for g in clock['groups'].values()),
                'requires a reserved, unused, real base TRAIN cohort')
        spec=Path(c['actor_spec']); h=read(pin(spec)); hot=Path(c['hf_hot']); t=Path(c['result'])
        paired=compare(hot,spec,t); tp=read(pin(t)); done=read(pin(c['exit']))
        features=read(c['features']); rewards=read(c['rewards']); manifest=read(pin(c['requests']))
        require(done['exit_code']==0 and done['result_exists'] and h['fresh_group']['frozen_reference'] and
                paired['passed_requests']==paired['requests'] and not paired['max_abs'] and not paired['max_sequence_abs']
                and all(not row['train_max_abs'] and not row['strict_flips'] for row in paired['rows']), 'exact frozen parity required')
        require(features['manifest']==rewards['requests']==pin(c['requests'])['sha256'] and
                features['behavior']==rewards['behavior']==pin(t)['sha256'] and
                manifest['schema']=='bounded-train-natural-requests-v1' and
                manifest['clock_sha256']==consumer['clock_state_ref']['sha256'] and
                manifest['selected_sha256']==rewards['selected']==pending['selected_cohort_ref']['sha256'] and
                rewards['learning']==config['learning_config_ref']['sha256'] and
                config['native']['model_path']==h['training']['model_path']==tp['model'], 'receipt identity drift')
        rank_refs=[pin(hot/f'rank-{i}/RESULT.json') for i in range(4)]
        scores={a['index']:(a,ref) for ref in rank_refs for a in read(ref)['records']}
        roster={}; G=learning['kernel']['G']
        for i,(req,act,pos) in enumerate(zip(manifest['rows'],tp['fresh_group']['records'],features['rows'],strict=True)):
            g,j=req['group'],req['replicate'];pkt=req['packet']; ident=pkt['fresh_capture_identity'];rid=pkt['request_id']
            a,_=scores[i]; rec=rewards['groups'][g]['observations'][j]
            require((g,j) not in roster and (act['group'],act['replicate'])==(g,j) and
                    rid==pos['request_id']==a['request_id']==act['request_id']==rec['request_id'] and
                    ident['boot_id']==h['boot_id'] and ident['group_id']==plan['group_ids'][g] and
                    ident['episode_id']==plan['group_ids'][g]+':e'+str(j) and ident['run_id']==plan['version'] and
                    pkt['stage']=='P' and not ident['parents'] and req['source_sha256']==act['source_sha256']==digest(pkt)
                    and pkt['wire_sha256']==act['wire_sha256'] and act['seed']==pkt['wire']['seed']
                    and act['text']==pos['decoded'] and a['train_scores']==a['reference_scores']==a['scores']==act['behavior_logprobs']
                    and all(pkt['wire'][k]==v for k,v in consumer['sampling_distribution'].items()), 'action/source mismatch')
            roster[g,j]=i
        require(len({r['packet']['request_id'] for r in manifest['rows']})==len(roster) and
                set(roster)=={(i,j) for i in range(len(plan['draws'])) for j in range(G)}, 'incomplete cohort')
        model=Path(tp['model']); assets=[pin(model/n) for n in ('tokenizer.json','tokenizer_config.json','vocab.json','merges.txt') if (model/n).is_file()]
        tok=save(root/'TOKENIZER.json',dict(schema='retained-tokenizer-v1',assets=assets,vocab_size=features['vocabulary'],pad_token_id=features['pad']))
        proc=save(root/'PROCESSOR.json',dict(schema='retained-audio-processor-v1',assets=assets+[pin(model/'preprocessor_config.json')],feature_layout='audio_feature_time',feature_bins=128,position_layout='retained_multi_axis',position_axes=['rope_axis','tokens'],position_non_token_shape=[3],audio_pad_token_id=features['audio_id']))
        origins={r['recording_ref_id']:r for r in read(consumer['recording_map_ref'])['records']}
        parser=original_parser(c,parents); vec=learning['kernel']['behavior_policy_vector']
        axes=dict(audio_features=['audio','feature','time'],feature_mask=['audio','time'],position_ids=['rope_axis','tokens'],effective_audio_lengths=['audio'])
        from consumer import legacy
        output=root/c['dataset_version']
        with sources(config) as (official,_):
            for d in plan['draws']:
                g=d['draw_index']; gid=d['group_id']; record=d['record']; origin=origins[record['recording_ref_id']]
                ann=annotation(d,official); snap=clock['groups'][gid]['snapshot']; gd=output/'groups'/str(g)
                require(record['split']=='TRAIN' and snap['rollout_scope']=='node' and snap['target_nodes']==['P'], 'not a P TRAIN group')
                episodes=[]; off={}; task=[]; costs=[]; frozen=None
                enc=features['groups'][str(g)]
                for j in range(G):
                    i=roster[g,j]; req=manifest['rows'][i]; pkt=req['packet']; act=tp['fresh_group']['records'][i]
                    hf,rank_ref=scores[i]; pos=features['rows'][i]; ident=pkt['fresh_capture_identity']
                    eid=ident['episode_id']; cid=eid+':c0'; rid=pkt['request_id']
                    prompt=req['prompt_token_ids']; tokens=act['token_ids']
                    reply={k:act[k] for k in ('text','finish_reason','request_id','wire_sha256')}
                    assets=wire_assets(pkt,origin,gd)
                    require(all(asset.get('media_type','audio')=='audio' for asset in assets),'non-audio asset on audio wire')
                    online,out=normalized(pkt,parser.parse_p(pkt,reply),reply,[dict(asset,media_type='audio') for asset in assets])
                    require(ann['gold_intent_id'] in online['retrieved_candidates'] and
                            (frozen is None or frozen==online) and out['terminal']==rewards['groups'][g]['observations'][j]['terminal'],
                            'conditioning, label eligibility or parser mismatch')
                    frozen=online
                    f=dict(prompt_token_ids=prompt,action_token_ids=tokens,behavior_logprobs=act['behavior_logprobs'],
                           action_mask=[1]*len(tokens),behavior_policy_version=vec['P'],adapter_vector=vec,
                           probability_semantics='same_effective_behavior_distribution_and_action_history',
                           sampling_distribution=consumer['sampling_distribution'],processor_manifest=proc,tokenizer_manifest=tok,
                           audio_features=enc['features'],feature_mask=enc['mask'],effective_audio_lengths=list(map(sum,enc['mask'])),
                           attention_mask=[1]*(len(prompt)+len(tokens)),position_ids=pos['positions'],modality_assets=online['audio_assets'],
                           recording_id=origin['capture_recording_id'],audio_token_lengths=audio_spans(prompt,features['audio_id']),
                           prompt_messages=pkt['wire']['messages'],prompt_text=enc['prompt'],action_text=act['text'],decoded_action_text=pos['decoded'],
                           input_boundary='online_only_no_reference',capture_origin='runtime_capture')
                    f['captured_tensor_schema']=dict(processor_manifest_sha256=digest(proc),consumer_contract_version=legacy.CONTRACT,
                        tensors={k:dict(shape=legacy.shape(f[k]),axes=v) for k,v in axes.items()})
                    f['capture_binding']=dict(online_input_sha256=digest(online),output_sha256=digest(out),fields_sha256=digest(f))
                    identity={k:ident[k] for k in ('boot_id','run_id','attempt_id')};identity['request_id']=rid
                    call=dict(call_id=cid,node='P',input_key='P',parents=[],request_identity=identity,observed_terminal=True,online_gold_injected=False,
                              online_input=online,output=out,behavior_policy_version=vec['P'],adapter_vector=vec,probability_semantics=f['probability_semantics'],
                              token_ids=tokens,token_kinds=['generated_eos' if t==features['eos'] else 'generated_action' for t in tokens],
                              action_mask=f['action_mask'],old_logprobs=act['behavior_logprobs'],current_logprobs=hf['scores'],
                              reference_logprobs=hf['reference_scores'],training_fields=f,raw_output=act['text'])
                    call['receipt']=dict(request_identity=identity,episode_id=eid,call_id=cid,payload_sha256=digest(call))
                    common=dict(query_draw_id=d['query_draw_id'],query_id=record['sample_id'],recording_id=origin['capture_recording_id'],sampling_round=plan['version'])
                    episodes.append(dict(common,episode_id=eid,replicate=j,terminal=out['terminal'],physical_requests_drained=True,rollout_scope='node',
                                         request_roster=[identity],expected_call_keys=['P'],calls=[call]))
                    off[eid]=dict(P=dict(calls=[dict(online=online,output=out,labels=labels_for('P',online,ann),call_ids=[cid])]))
                    task.append(dict(group_id=gid,episode_id=eid,call_id=cid,node='P',annotation=ann,packet=pkt['original_packet']))
                    costs.append(dict(request_id=rid,usage=dict(prompt_tokens=len(prompt),completion_tokens=len(tokens),total_tokens=len(prompt)+len(tokens)),
                        journal=dict(transport='direct_vllm_engine',result_ref=pin(t),exit_ref=pin(c['exit']),row=i),
                        fixed_action_hf_receipt=rank_ref,current_forward_tokens=len(prompt)+len(tokens),reference_forward_tokens=len(prompt)+len(tokens),
                        reference_score_source_request_id=rid,currency_cost=None,npu_seconds=None))
                group=dict(common,group_id=gid,split='TRAIN',gold_in_actual_pre_P_dynamic_retrieval=True,fresh_on_policy=True,closed=True,
                           system_version=learning['system_version'],baseline=learning['kernel']['baseline'],behavior_policy_vector=vec,
                           group_size=G,snapshot=snap,frozen_upstream_release=snap['frozen_upstream_release'],frozen_inputs=dict(P=frozen),episodes=episodes,
                           closure=dict(group_id=gid,episode_ids=[e['episode_id'] for e in episodes],snapshot_sha256=snap['snapshot_sha256']))
                save(gd/'GROUP.json',dict(group_id=gid,query_draw_id=d['query_draw_id'],policy=config['policy'],synthetic=False,costs=costs,
                    task_records=task,online=dict(online=dict(audio_assets=[origin['audio_ref']]),behavior_bundle=dict(schema='unified-behavior-bundle-v1',
                    synthetic=False,online_gold_injected=False,global_query_count=1,groups=[group])),offline=dict(supervision=record['offline_supervision'],group_supervision=off)))
        # Padding is a batch-packing setting applied after capture. Keep the old
        # draw and Clock intact; publish the corrected consumer as a new artifact.
        from runtime_builder import model_padding_id
        pad=model_padding_id(model)
        require(pad==features['pad'],'processor and model disagree on padding')
        save(output/'SELECTED.json',plan); save(output/'RESERVED-CLOCK.json',clock)
        derived=dict(consumer,pad_token_id=pad)
        save(output/'CONSUMER.json',derived)
        save(output/'CONSUMER-LINEAGE.json',dict(original=pin(root/'CONSUMER.json'),updated=pin(output/'CONSUMER.json'),
             tokenizer=tok,reason='correct collator padding only; no action, policy or sampling change'))
        return cohort.publish(config,output)


def run(path, child):
    c=json.loads(path.read_bytes());base=Path(c['release']);parents=base/'tree/parents';root=Path(c['prepared'])
    if not c['dataset_version'].isalnum():raise ValueError('invalid local dataset version')
    assembly=json.loads((base/'ASSEMBLY.json').read_bytes())
    logical=Path(json.loads(Path(assembly['delta_manifest_ref']['path']).read_bytes())['logical_root'])
    consumer=json.loads((root/'CONSUMER.json').read_bytes())
    adapter=Path(consumer['sealed_adapter_ref']['path'])
    import hashlib
    expected=Path('training-consumer-adapter/adapter.py')
    if (not adapter.is_relative_to(adapter.parents[1]) or adapter.relative_to(adapter.parents[1])!=expected or
            hashlib.sha256((parents/expected).read_bytes()).hexdigest()!=consumer['sealed_adapter_ref']['sha256']):
        raise ValueError('configured sealed consumer differs in the selected release')
    training_root=adapter.parents[1]
    if child:
        with (root/'COHORT.guard').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            result=publish(c,training_root)
            print(json.dumps({k:result[k] for k in ('selected_groups','complete_episodes','global_episode_denominator','training_ready','gradient_status')}))
        return
    from source.release_completion import require_closed_release
    require_closed_release(base)
    # Only the already-prepared cohort and its isolated CPU index cache may be written.
    cfg=json.loads((root/'INPUT.json').read_bytes())
    source=json.loads(Path(cfg['capture']['official_source_config_ref']['path']).read_bytes())
    cache=Path(source['index_cache_root'])
    local=root/'official-cache';local.mkdir(exist_ok=True)
    mounts=[]
    if not cache.exists():
        mounts=['--tmpfs',str(cache.parent)]
        for entry in sorted(cache.parent.iterdir()):
            mounts.extend(['--ro-bind',str(entry),str(entry)])
    for domain in source['domains']:
        inventory=str(Path(domain['commit']['path']).parent)
        mounts.extend(['--ro-bind',inventory,inventory])
    subprocess.run(['bwrap','--ro-bind','/','/','--ro-bind',str(parents),str(logical),'--ro-bind',str(parents),str(training_root),'--tmpfs','/run',
         '--ro-bind',str(Path(__file__).resolve().parent),'/run/bridge','--ro-bind',str(c['collection_config']),str(c['collection_config']),'--bind',str(root),str(root),*mounts,'--bind',str(local),str(cache),
         '--dev','/dev','--proc','/proc',sys.executable,'-B','/run/bridge/'+Path(__file__).name,'--config',str(path),'--child'],check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--child',action='store_true')
    a=p.parse_args();run(a.config.resolve(),a.child)

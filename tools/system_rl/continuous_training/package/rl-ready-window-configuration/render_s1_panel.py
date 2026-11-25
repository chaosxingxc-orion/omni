"""Consume a sampled panel with the original S1 rails; materialize only selected folds."""
import argparse
import base64
import collections
import fcntl
import importlib.util
import json
import sys
from pathlib import Path

from build import pin,save
from prepare_s1_recordings import checked,needed,read,sha


def render(panel_ref,out):
    from multimodal_jev.local_runtime_storage import check_paths
    check_paths({'panel':out})
    panel=read(checked(panel_ref));pages=read(checked(panel['sampling']['pages']))
    needed(panel['purpose']=='paired_baseline_screen' and not panel['formal_training_set'],'expected screened recording inputs')
    config=pages['configuration'];dataset=read(checked(config['dataset']))
    roots={v['domain']:Path(v['audio_root']) for v in dataset['domains']}
    seal=read(checked(config['knowledge_release']));refs={Path(v['path']).name:v for v in seal['artifacts']}
    knowledge,cache,runtime=(read(checked(refs[n])) for n in ('knowledge.private.json','retrieval.private.json','s1.runtime.json'))
    needed(cache['cache_id']==runtime['s1_consumption']['cache_id'] and cache['knowledge_id']==knowledge['knowledge_id'],'knowledge binding differs')
    here=Path(__file__).resolve().parent;code=here.parents[1]
    sys.path.insert(0,str(code/'s1-source-policy-successor'))
    import policy
    from source_bridge import Resolver,SourceQuery
    from storage import sha as hash_bytes,canonical
    from source.domain_rails import policy_rails
    resolver=Resolver()
    wav_source=resolver.root/resolver.cfg['package_src']/'speech_aware_evidence_acquisition/analysis/slurp_mf7b_vectors.py'
    loader=importlib.util.spec_from_file_location('s1_original_audio_reader',wav_source);mod=importlib.util.module_from_spec(loader);loader.loader.exec_module(mod)
    compiled=dict(cache,queries=[],query_source_bindings={});compiled.pop('cache_id')
    by_page=collections.defaultdict(list)
    for d in panel['draws']:by_page[d['page']].append(d)
    items={};queries={}
    for page_no,draws in by_page.items():
        page=read(checked(pages['pages'][page_no]))
        for d in draws:
            row=page[d['row']];rid=row['reference_id'];proof=resolver.proof(rid)
            needed(rid==d['reference_id'] and rid not in items and proof['decision']=='allowed'
                and all(proof[k]==row[k] for k in ('domain','source_id','source_component','recording_file','source_record_sha256')),
                'S1 draw/source identity differs')
            path=roots[row['domain']]/row['recording_file'];needed(sha(path)==row['audio_sha256'],'changed audio')
            wav=mod.read_audio_wav_bytes(path)
            audio=dict(type='input_audio',input_audio=dict(format='wav',data=base64.b64encode(wav).decode('ascii')))
            sid=proof['sample_id'];needed(sid not in queries,'ambiguous recording identity')
            query=SourceQuery(sid,audio,row['source_component'],row['audio_sha256'],row['domain'],'TRAIN',row['source_id'],rid,row['recording_file'],row['source_record_sha256'])
            queries[sid]=query
            binding=dict(schema='s1-domain-qualified-query-v1',**{k:getattr(query,k) for k in
                ('domain','split','source_id','reference_id','recording_file','source_record_sha256','source_component','audio_sha256')},
                audio_object_sha256=hash_bytes(canonical(audio)))
            compiled['query_source_bindings'][sid]=binding
            compiled['queries'].append(dict(sample_id=sid,**{k:binding[k] for k in ('source_component','audio_sha256','audio_object_sha256')},
                excluded_source_components=[binding['source_component']],neighbors=[dict(source_id=i,score=s) for i,s in row['neighbors']]))
            items[rid]=(sid,row['offline']['present_types'])
    compiled['cache_id']=hash_bytes(canonical(compiled))
    runtime=dict(runtime,s1_consumption=dict(runtime['s1_consumption'],cache_id=compiled['cache_id']))
    rails=policy_rails(policy,resolver,knowledge,compiled,runtime)
    output=[]
    for i,d in enumerate(panel['draws']):
        if i%32==0:check_paths({'panel':out})
        rid=d['reference_id'];sid,types=items[rid]
        packet=rails.before_model_call('S1',queries[sid],d['fold'])
        needed(packet['source_binding']['reference_id']==rid and packet['allowed_types']==pages['folds'][d['fold']]
            and hash_bytes(canonical(packet['wire']))==packet['wire_sha256'],'S1 consumer readback differs')
        output.append(dict(reference_id=rid,fold=d['fold'],packet=save(out/'packets'/(str(i)+'.json'),packet),
            supervision=dict(present_types=[t for t in packet['allowed_types'] if t in types],annotated_types=packet['allowed_types'])))
    return save(out/'INPUTS.json',dict(schema='s1-selected-inputs-v1',panel_ref=panel_ref,cache_id=compiled['cache_id'],
        knowledge_id=knowledge['knowledge_id'],source_policy_id=runtime['source_policy_id'],renderer=pin(wav_source),
        rows=output,rendered=len(output),model_calls=0))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--panel',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();a.output.mkdir(mode=0o700,parents=True,exist_ok=True)
    with (a.output/'render.guard').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(json.dumps(render(pin(a.panel),a.output)))

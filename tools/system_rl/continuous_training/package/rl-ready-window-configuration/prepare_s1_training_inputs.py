"""Prepare official input facts and original wire audio once; attach measured outcomes separately."""
import argparse
import base64
import hashlib
import json
import os
import tempfile
from pathlib import Path
from build import pin,save
from prepare_s1_recordings import read,checked,needed
from source.config_profile import stage_config,bind_environment,digest


def pcm(root,raw):
    key=hashlib.sha256(raw).hexdigest();p=root/(key+'.wav')
    p.parent.mkdir(parents=True,exist_ok=True)
    if not p.exists():
        with tempfile.NamedTemporaryFile(dir=p.parent) as f:
            f.write(raw);f.flush();os.fsync(f.fileno())
            try:os.link(f.name,p)
            except FileExistsError:pass
        fd=os.open(p.parent,os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    ref=pin(p);needed(ref['sha256']==key and ref['bytes']==len(raw),'wrong stored PCM')
    return ref


def prepare(cfg,out):
    from multimodal_jev.local_runtime_storage import check_paths
    space=lambda:check_paths({'s1_training_inputs':out.parent},min_free_fraction=.1,min_free_bytes=20<<30)
    space()
    split=read(checked(cfg['partition']));inputs=read(checked(split['inputs']))
    panel=read(checked(inputs['panel_ref']));pages=read(checked(panel['sampling']['pages']))
    dataset=read(checked(pages['configuration']['dataset']))
    needed(split['schema']=='s1-panel-partition-v1' and dataset['split']=='TRAIN' and not dataset['synthetic']
           and not panel['formal_training_set'] and inputs['schema']=='s1-selected-inputs-v1','invalid TRAIN panel')
    indexed={d['reference_id']:d for d in panel['draws']}; entries={r['reference_id']:r for r in inputs['rows']}
    needed(len(indexed)==len(entries)==len(split['rows']) and indexed.keys()==entries.keys(),'panel membership differs')
    by_domain={d['domain']:(d['metadata']['path'],[json.loads(l) for l in checked(d['metadata']).open()])
               for d in dataset['domains']}
    from functools import lru_cache
    @lru_cache(maxsize=16)
    def page(i):return read(checked(pages['pages'][i]))
    rows=[]
    for i,s in enumerate(split['rows']):
        if not i%32:space()
        rid=s['reference_id'];d=indexed.pop(rid);e=entries.pop(rid)
        dom,line,idx=rid.split(':');line=int(line);idx=int(idx)
        meta,official=by_domain[dom];raw=official[line-1];original=page(d['page'])[d['row']]
        packet=read(checked(s['packet']));binding=packet['source_binding']
        needed(e['packet']==s['packet'] and e['supervision']==s['supervision'] and d['fold']==s['fold']==packet['branch']
               and packet['stage']=='S1' and original['reference_id']==binding['reference_id']==rid
               and original['source_component']==binding['source_component']==s['component']
               and original['domain']==binding['domain']==dom and original['source_record_sha256']==digest(raw)
               and original['recording_file']==raw['recordings'][idx]['file'] and original['source_id']==binding['source_id']==dom+':'+str(raw['slurp_id'])
               and binding['split']=='TRAIN' and packet['s1_consumption']['knowledge_id']==inputs['knowledge_id']
               and s['supervision']['annotated_types']==packet['allowed_types']
               and isinstance(raw['tokens'],list) and isinstance(raw['entities'],list)
               and s['supervision']['present_types']==[t for t in packet['allowed_types'] if t in {a['type'] for a in raw['entities']}],
               'TRAIN recording/fold or official facts differ')
        labels=dict(source=dict(path=meta,line=line),reference_text_source='official_metadata',reference_text=raw.get('sentence'),
                    gold_intent_id=raw['scenario']+'_'+raw['action'],tokens=raw['tokens'],entities=raw['entities'])
        audio,=[part for m in packet['wire']['messages'] if isinstance(m['content'],list)
                for part in m['content'] if part.get('type')=='input_audio']
        media=pcm(out/'pcm',base64.b64decode(audio['input_audio']['data'],validate=True))
        stem=hashlib.sha256(rid.encode()).hexdigest()
        origin=save(out/'origins'/(stem+'.json'),dict(recording_ref_id=rid,source_id=binding['source_id'],source='SLURP',
                domain=dom,split='TRAIN',capture_recording_id=rid,recording_filename=Path(media['path']).name,audio_ref=media,context_audio_refs=[]))
        record=dict(recording_ref_id=rid,sample_id=packet['sample_id'],source_id=binding['source_id'],domain=dom,source='SLURP',
                    split='TRAIN',synthetic_fixture=False,online=dict(node_input_ref=s['packet']),offline_supervision=dict(labels=labels))
        rows.append(dict(reference_id=rid,fold=s['fold'],split=s['split'],source_record_sha256=original['source_record_sha256'],
                         record_ref=save(out/'records'/(stem+'.json'),record),recording_origin_ref=origin))
    return save(out/'INPUTS.json',dict(schema='s1-static-training-inputs-v1',partition=cfg['partition'],rows=rows,
                prepared=len(rows),model_calls=0,formal_training_set=False))


def main():
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('config');cli.add_argument('--profile',required=True)
    a=cli.parse_args();profile=read(a.profile)
    if profile['schema']=='omni-local-profile-v2':profile,_=bind_environment(profile,os.environ)
    cfg=stage_config(read(a.config),profile,'s1-static-training-input-config-v1')
    needed(cfg['schema']=='s1-static-training-input-config-v1','unexpected S1 input config')
    print(json.dumps(prepare(cfg,Path(cfg['output']))))
if __name__=='__main__':main()

"""Selected recording custody prepared on CPU, before a model lease.

Reuse retained PCM bytes and source proofs; no new audio decoding, features,
model output or label inference. Origin references are projected per window.
"""
import base64,copy,hashlib,os
from pathlib import Path
from dependencies import require
from run_package import read,pin,write


def origin(record,root,expand):
    evidence=expand(record['online']['evidence'])
    prepared=evidence['prepare']['output']['prepared']
    proof=read(prepared['source_proof_ref']);rid=record['recording_ref_id']
    require(record['split']==proof['split']=='TRAIN' and
            prepared['reference_id']==proof['reference_id']==rid and
            prepared['sample_id']==proof['sample_id']==record['sample_id'] and
            record['source_id']==proof['source_id'],'selected source/custody mismatch')
    original=prepared['original_audio_ref'];audio=read(original)
    for key in original['content_path']:audio=audio[key]
    raw=base64.b64decode(audio['input_audio']['data'],validate=True)
    digest=hashlib.sha256(raw).hexdigest()
    require(digest==prepared['transport_audio_sha256'],'retained PCM hash differs')
    root=Path(root);path=root/'audio'/(digest+'.wav');path.parent.mkdir(parents=True,exist_ok=True)
    from storage import storage
    # Check mount/free/inodes in O(1); do not rescan the whole run for each row.
    storage.check_paths({'selected_audio':root},min_free_bytes=20*1024**3+len(raw))
    if path.exists():require(path.read_bytes()==raw,'retained PCM content changed')
    else:
        temporary=path.with_suffix('.pending')
        # A partial CPU-only write can be replaced deterministically; the final
        # content-addressed path remains immutable and never contains requests.
        with temporary.open('wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        os.replace(temporary,path)
        fd=os.open(path.parent,os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    row=dict(recording_ref_id=rid,source_id=record['source_id'],source=record['source'],
        domain=record['domain'],split='TRAIN',capture_recording_id=rid,
        recording_filename=path.name,audio_ref=pin(path),context_audio_refs=[])
    return write(root/'origins'/(hashlib.sha256(rid.encode()).hexdigest()+'.json'),row)


def bind_origins(draws,root,expand):
    result=copy.deepcopy(draws);seen={}
    for draw in result:
        record=draw['record'];rid=record['recording_ref_id']
        if rid not in seen:seen[rid]=origin(record,root,expand)
        draw['recording_origin_ref']=seen[rid]
    return result


def window_mapping(draws,output,*,dataset_ref):
    rows={}
    for draw in draws:
        require('recording_origin_ref' in draw,'selected recording origin missing')
        row=read(draw['recording_origin_ref']);rid=row['recording_ref_id']
        previous=rows.setdefault(rid,row);require(previous==row,'selected origin identity drift')
    require(rows,'empty recording window')
    root=Path(output)
    dataset=read(dataset_ref);pool=read(dataset['screen_ref'])
    span=({'screen_population_scope':pool['population_scope'],'screen_population':pool['population']}
          if pool.get('schema')=='node-panel-selection-v1' else {'official_train_count':pool['official_train_count']})
    population=write(root/'POPULATION.json',dict(schema='selected-recording-population-v1',
        dataset_ref=dataset_ref,population_complete=False,**span,
        selected_records=len(rows),population=[{k:r[k] for k in
        ('recording_ref_id','source_id','source','domain','split')} for r in rows.values()]))
    return write(root/'RECORDING-MAP.json',dict(schema='retained-recording-map-v1',
        population_ref=population,records=list(rows.values())))

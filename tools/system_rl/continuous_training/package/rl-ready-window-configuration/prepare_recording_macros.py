"""Seal ordered static recording pages; policy-dependent fields belong to later rollouts."""
import argparse, hashlib, json, os
from pathlib import Path


def read(ref):
    raw=Path(ref['path']).read_bytes()
    if len(raw)!=ref['bytes'] or hashlib.sha256(raw).hexdigest()!=ref['sha256']:raise ValueError('changed static source')
    return json.loads(raw)


def seal(path,obj):
    raw=(json.dumps(obj,ensure_ascii=False,separators=(',',':'))+'\n').encode()
    if path.exists():
        if path.read_bytes()!=raw:raise ValueError('different immutable page')
    else:
        from tempfile import NamedTemporaryFile
        with NamedTemporaryFile(dir=path.parent) as f:
            f.write(raw);f.flush();os.fsync(f.fileno());os.link(f.name,path)
        fd=os.open(path.parent,os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    return dict(path=str(path.resolve()),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))


def build(static_ref,width,output):
    parent=read(static_ref);dataset=read(parent['dataset_ref']);roster=read(dataset['selected_ref'])['draws']
    if type(width)!=int or width<1 or len(roster)!=dataset['count']:raise ValueError('invalid recording macro')
    output.mkdir(parents=True,exist_ok=True);pages=[]
    for start in range(0,len(roster),width):
        rows=[]
        for d in roster[start:start+width]:
            row=read(d['record_ref']);rid=row['recording_ref_id'];entry=parent['rows'][rid]
            if row['online']['node_input_ref']!=entry['packet_ref']:raise ValueError('changed recall packet')
            rows.append(dict(id=rid,entry=entry))
        pages.append(seal(output/f'recordings-{start:06}.json',dict(start=start,rows=rows)))
    return seal(output/'MACROS.json',dict(schema='recording-macros-v1',static_supply_ref=static_ref,
                dataset_ref=parent['dataset_ref'],width=width,count=len(roster),pages=pages,header={k:v for k,v in parent.items() if k!='rows'}))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('config','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();cfg=json.loads(a.config.read_bytes());print(json.dumps(build(cfg['static_supply_ref'],cfg['recordings_per_macro'],a.output)))

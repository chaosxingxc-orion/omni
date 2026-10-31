"""Prepare a label-free, reproducible 500-row real SLURP training workload.

No inference, tokenizer, scoring or registration. Source media identities are
preserved; selection never depends on earlier completion, answer or correctness.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))


def canonical(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()


def prepare(source,seed='20261031-real-engine-500-v1'):
    from speech_aware_evidence_acquisition.data.splits import prefixed,split_identity_hash
    parent=json.loads(source.read_bytes())
    unsigned={k:v for k,v in parent.items() if k!='runtime_view_sha256'}
    if digest(unsigned)!=parent['runtime_view_sha256']:raise ValueError('parent view hash differs')
    rows=parent['items'];ids=[r['sample_id'] for r in rows]
    if len(rows)!=1080 or len(set(ids))!=1080 or any(not s.startswith('slurp:train:') for s in ids):
        raise ValueError('requires unique real Fold B train roster')
    if any('/slurp_real/' not in r['media_relpath'] for r in rows):raise ValueError('requires real speech')
    ranked=sorted(rows,key=lambda r:hashlib.sha256((seed+'\0'+r['sample_id']).encode()).digest())
    selected=sorted(ranked[:500],key=lambda r:r['sample_id'])
    identity=split_identity_hash(prefixed('slurp',[r['sample_id'] for r in selected]))
    def distribution(items):
        values=sorted(r['audio_seconds'] for r in items)
        return {'n':len(items),'mean_audio_s':statistics.mean(values),'p50_audio_s':statistics.median(values),'p95_audio_s':values[int(.95*(len(values)-1))],'max_audio_s':max(values),'total_audio_s':sum(values)}
    receipt={'schema':'saea-real-engine-roster-v1','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'parent_split_identity_hash':parent['split_identity_hash'],'split_identity_hash':identity,'seed':seed,
        'selection':'uniform hash rank of every parent sample; no outcome/label/completion selection',
        'count':500,'items':selected,'distributions':{'parent':distribution(rows),'selected':distribution(selected)},
        'new_model_calls':0,'registered':False,'scope':'real two-contact runtime workload; not static prompt replay or synthetic load',
        'claims_representative_of_full_test':False}
    view={**unsigned,'split_name':'SLURP-REAL-ENGINE-500-V1','seed':20261031,'count':500,
        'split_identity_hash':identity,'source_binding_sha256':digest(receipt),'items':selected,
        'audio_seconds_total':round(sum(r['audio_seconds'] for r in selected),9)}
    view['runtime_view_sha256']=digest(view)
    return receipt,view


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    assert_execution_environment()
    receipt,view=prepare(a.source);a.out.mkdir(parents=True,exist_ok=False)
    for name,value in [('roster.json',receipt),('dataset.json',view)]:
        with (a.out/name).open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    print(json.dumps({'rows':500,'split_identity_hash':receipt['split_identity_hash'],'distributions':receipt['distributions'],'new_model_calls':0,'registered':False}))

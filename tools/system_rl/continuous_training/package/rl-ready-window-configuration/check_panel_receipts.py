"""Exercise collector-native content refs with actual sealed S1 responses on CPU."""
import argparse
import json
import sys
import tempfile
from pathlib import Path

from build import save
from source.panel_receipts import collection_calls
from prepare_s1_recordings import read, checked, needed


def check(seal, collector, output):
    sys.path.insert(0, str(collector.resolve()))
    from content import Content, jsonl
    version = read(checked(read(seal)['version']))
    old = [c for c in collection_calls(version) if c['stage']=='S1' and c['status']=='known']
    sampled = old[:10]
    needed(sampled, 'real closed S1 receipts required')
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        root = Path(temp)
        content = Content(root/'content')
        rows = (dict(c, packet=content.dumps(c['packet']), receipt=content.dumps(c['receipt'])) for c in sampled)
        ref = jsonl(root/'receipts.jsonl', rows)
        new = dict(receipts=ref,content_format='collection-content-v1')
        needed(list(collection_calls(new))==sampled,'native packet or receipt differs')
        for change in [dict(count=len(sampled)+1),dict(bytes=ref['bytes']-1),dict(sha256='0'*64)]:
            try:next(collection_calls(dict(new,receipts=dict(ref,**change))))
            except ValueError:pass
            else:raise AssertionError('modified stream accepted')
        with (root/'receipts.jsonl').open() as stream:
            first = json.loads(stream.readline())
        wrong = jsonl(root/'wrong.jsonl', [dict(first, request_id='another-call')])
        try: list(collection_calls(dict(new, receipts=wrong)))
        except ValueError: pass
        else: raise AssertionError('misbound row accepted')

        refs=[]
        def walk(v):
            if isinstance(v,dict):
                if '$collection_content_v1' in v:refs.append(v['$collection_content_v1'])
                else:
                    for child in v.values():walk(child)
            elif isinstance(v,list):
                for child in v:walk(child)
        walk(json.loads(first['packet']))
        p=Path(refs[0]['path']);p.write_bytes(b'{}')
        try:list(collection_calls(new))
        except ValueError:pass
        else:raise AssertionError('changed content object accepted')
    return save(output,dict(historical_S1=len(old),paired=len(sampled),packet_and_response_equal=True,
            rejected=['row count','stream length','stream digest','content digest','provider request mismatch'],model_calls=0))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('seal',type=Path);p.add_argument('collector',type=Path);p.add_argument('output',type=Path)
    print(check(**vars(p.parse_args())))

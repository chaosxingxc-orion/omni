"""Actual closed collector: large trajectory payload and typed dependency edges."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    spec=importlib.util.spec_from_file_location('closed_trajectory',a.source)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    root=Path(a.output);root.mkdir(parents=True,exist_ok=False);checks=[]
    def put(path,value):
        path.parent.mkdir(parents=True,exist_ok=True);raw=json.dumps(value).encode();path.write_bytes(raw)
        return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
    audio=put(root/'audio.bin',{'fixture':True})
    receipt=put(root/'response.json',{'closed':True})
    missing=dict(path=str(root/'historical-not-a-dependency.json'),sha256='0'*64)
    call=dict(packet=dict(stage='P',retrieval_event_ref=missing),reply={},online=dict(audio_assets=[audio]),
              output={},annotation={},token_ids=[1],behavior_logprobs=[-1.0],prompt_token_ids=[2],
              raw_output='x'*(17*1024**2),usage={},envelope=dict(response_ref=receipt),annotation_ref=missing)
    group=dict(group_id='g',query_draw_id='q',episodes=[dict(episode_id='g:e0',replicate=0,calls=[call],synthetic=False)],policy={})
    path=root/'cohort/groups/0/GENERATED.json';ref=put(path,group)
    put(root/'RETURN.json',dict(generated_ref=ref));put(root/'CLOSED.json',dict(immutable=True))
    files=m.collect(root,64*1024**2)
    retained={x['path'] for x in files.values()}
    assert str(path) in retained and audio['path'] in retained and receipt['path'] in retained
    assert missing['path'] not in retained and path.read_bytes().find(b'historical-not-a-dependency')>=0
    checks.append('large real-shaped trajectory transfers verbatim plus actual receipts/audio, preserves historical citations without recursing')
    def rejected(name,fn):
        try:fn()
        except AssertionError:checks.append(name)
        else:raise AssertionError(name)
    rejected('total configured byte budget retained',lambda:m.collect(root,1024))
    changed=copy.deepcopy(group);changed['episodes'][0]['synthetic']=True
    bad=put(path,changed);put(root/'RETURN.json',dict(generated_ref=bad))
    rejected('malformed large trajectory rejected',lambda:m.collect(root,64*1024**2))
    put(path,group);put(root/'RETURN.json',dict(generated_ref=ref))
    (root/'response.json').write_text('{}')
    rejected('live receipt hash drift rejected',lambda:m.collect(root,64*1024**2))
    put(root/'response.json',{'closed':True})
    (root/'audio.bin').unlink()
    rejected('missing actual audio dependency rejected',lambda:m.collect(root,64*1024**2))
    put(root/'audio.bin',{'fixture':True})
    # The producer inventories every cohort file. Preserve bytes without
    # treating nested historical citations in arbitrary payloads as live refs.
    cohort=root/'cohort'
    control=put(cohort/'INPUT.json',dict(audio_ref=audio))
    metadata=put(cohort/'BUSINESS.json',dict(historical=missing))
    selected=put(cohort/'SELECTED.json',dict(source=missing))
    closed=put(cohort/'GENERATION-CLOSED.json',dict(selected_cohort_ref=selected))
    inv=put(cohort/'GENERATION-INVENTORY.json',dict(closed=True,cohort_root=str(cohort),
        generation_closed_ref=closed,files={'INPUT.json':control,'BUSINESS.json':metadata,
            'SELECTED.json':selected,'GENERATION-CLOSED.json':closed,'groups/0/GENERATED.json':ref}))
    put(root/'RETURN.json',dict(inventory_ref=inv))
    retained={x['path'] for x in m.collect(root,64*1024**2).values()}
    assert metadata['path'] in retained and audio['path'] in retained and missing['path'] not in retained
    checks.append('complete closed inventory transfers all data and only explicit external control dependencies')
    inventory=json.loads(Path(inv['path']).read_bytes());inventory['files']['../escape.json']=receipt
    bad=put(Path(inv['path']),inventory);put(root/'RETURN.json',dict(inventory_ref=bad))
    rejected('inventory traversal rejected',lambda:m.collect(root,64*1024**2))
    huge=put(root/'control.json',dict(data='x'*(17*1024**2)))
    put(root/'RETURN.json',dict(control_ref=huge))
    rejected('ordinary control JSON still capped at 16MiB',lambda:m.collect(root,64*1024**2))
    # Replay the finite selector's actual dependency boundary: population
    # is cited, while outcomes and records were opened by the selector.
    record=put(root/'records.jsonl',{'sample':1});outcome=put(root/'outcomes.jsonl',{'valid':True})
    streams=dict(records=record,outcomes=outcome,population=missing)
    pool=put(root/'pool.json',dict(streams=streams,sealed=True))
    version=put(root/'version.json',dict(training_pool=pool))
    selection=dict(schema='selected-fresh-cohort-v1',prepare_mode='finite_diagnostic_capture',
                   source_custody=[dict(version=version,pool=pool,streams=streams)])
    def select(v):
        sel=put(cohort/'SELECTED.json',v);put(root/'RETURN.json',dict(selected=sel))
    select(selection)
    paths={v['path'] for v in m.collect(root,64*1024**2).values()}
    assert {record['path'],outcome['path'],pool['path'],version['path']}<=paths and missing['path'] not in paths
    checks.append('sealed population citation retained while actually consumed finite inputs are verified')
    forged=copy.deepcopy(selection);forged['source_custody'][0]['streams']['population']['sha256']='1'*64
    select(forged);rejected('reject altered historical population citation',lambda:m.collect(root,64*1024**2))
    select(selection);Path(record['path']).unlink()
    rejected('reject missing consumed finite records',lambda:m.collect(root,64*1024**2))
    report=dict(passed=True,checks=checks,model_calls=0,fixture=True)
    (root/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()

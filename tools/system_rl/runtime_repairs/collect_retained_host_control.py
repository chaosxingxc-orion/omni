"""Collect immutable same-image control results and compare identical HF executions."""
import argparse
import concurrent.futures
import json
import math
import subprocess
import time
from pathlib import Path
from collect_retained_boundaries import read_remote, store
from compare_retained_score_paths import metrics


def collect(record, out):
    host,root=record['host'],Path(record['root'])
    until=time.monotonic()+record['timeout_seconds']
    while time.monotonic()<until:
        try:
            terminal=json.loads(read_remote(host,'cat',str(root/'hot/new-lease/NPU-EXIT.json')))
            break
        except (subprocess.CalledProcessError,subprocess.TimeoutExpired,json.JSONDecodeError):
            time.sleep(30)
    else:raise TimeoutError('reconcile existing lease before reattempting observation')
    if terminal['Running'] or terminal['ExitCode']!=0:raise ValueError(f'{host}: closed unsuccessful phase')
    data=read_remote(host,'cat',str(root/'hot/HF-TEXT-RESULT.json'))
    store(out,data)
    return json.loads(data)


def compare(a,b,score,ordinal,limit):
    for k in ['input_pins','hf_source','token_ids','mode','predictor_position','parent_actor_scores']:
        if a[k]!=b[k]:raise ValueError('different control: '+k)
    if set(a['rows'])!=set(b['rows']) or len(a['rows'])!=98:raise ValueError('incomplete layer rows')
    rows=score['cases']['P']['actions']
    if [x['token'] for x in rows]!=a['token_ids']:raise ValueError('vLLM input does not match')
    arms={'hf_175':a['scores'],'hf_181':b['scores'],'actor_181':a['parent_actor_scores'],
          'prefill_175':[x['vl_prefill'] for x in rows],'decode_175':[x['vl_natural'] for x in rows]}
    for v in arms.values():
        if len(v)!=len(rows) or not all(map(math.isfinite,v)):raise ValueError('incomplete score path')
    differing=[k for k in a['rows'] if a['rows'][k]!=b['rows'][k]]
    return dict(hardware=[a['hardware'],b['hardware']],original_action=ordinal,logprobs={k:v[ordinal-1] for k,v in arms.items()},
        hf_rows_different=differing,compared_rows=98,
        comparisons={x+'_vs_'+y:metrics(arms[x],arms[y],limit) for x,y in [
            ('hf_175','hf_181'),('hf_181','actor_181'),('hf_175','prefill_175'),('hf_175','decode_175')]},
        training_admission=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path)
    cfg=json.loads(p.parse_args().config.read_bytes());out=Path(cfg['out']);out.mkdir(exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(collect,r,out/(r['label']+'.json')) for r in cfg['runs']]
        a,b=[j.result() for j in jobs]
    report=compare(a,b,json.loads(Path(cfg['score_paths']).read_bytes()),cfg['action_ordinal'],cfg['limit'])
    store(out/'COMPARE.json',json.dumps(report,indent=2).encode())

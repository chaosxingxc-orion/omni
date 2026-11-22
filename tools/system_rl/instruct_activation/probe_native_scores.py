"""Compare native sampled scores to archived, exact HF scores only along common histories."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import math
import subprocess
import time
import urllib.request
from pathlib import Path
from io_utils import canonical, doc, pin, require, save
from measure_existing_services import contact, idle, metrics, service_identity


def prepare(paired, hf, output):
    bundle = doc(pin(paired))
    proofs = [pin(f) for f in sorted(hf.glob('rank-*/RESULT.json'))]
    cases = []
    for r in bundle['rows']:
        witnesses = [doc(ref) for ref in proofs if doc(ref)['response_sha'] == r['response']]
        require(witnesses and all(h['passed'] and h['lane'] == r['lane'] and
                h['observed_policy_artifact'] == r['adapter_sha'] and h['tokens'] == len(r['tokens'])
                and all(h[mode+'_max_abs'] == h[mode+'_sequence_abs'] == 0
                        and h[mode+'_top1_flips'] == [] for mode in ('train','eval')) for h in witnesses),
                'reference must have exact HF train/eval evidence')
        w = r['packet']['wire']
        require(r['packet']['wire_sha256'] == hashlib.sha256(canonical(w)).hexdigest()
                and w['logprobs'] and w['return_token_ids'] and w['top_logprobs'] == 0
                and not w.get('vllm_xargs') and not w.get('response_format'), 'not a natural RL wire')
        ref, = [ref for ref in r['source_refs'] if Path(ref['path']).name.startswith('wire-')]
        require(doc(ref) == {k:v for k,v in w.items() if k!='cache_salt'}, 'source mismatch')
        cases.append({k:r[k] for k in ('lane','tokens','logprobs','prompt','raw_top1','adapter_sha','response')}
                     | dict(wire=w))
    return save(output,dict(cases=cases,limits=bundle['limits'],reference=pin(paired),hf=proofs))


def compare(r, body, alias, observe_top1=True):
    c, = body['choices']; ids=c['token_ids']; content=c['logprobs']['content']
    require(body['model']==alias and body['prompt_token_ids']==r['prompt'] and
            c['finish_reason'] in ('stop','length') and len(ids)==len(content)==body['usage']['completion_tokens']
            and len(ids)>0, 'model, prompt IDs or actual action stream mismatch')
    diffs=[];flips=[]
    for i,(t,item) in enumerate(zip(ids,content,strict=True)):
        require(item['token']==f'token_id:{t}' and math.isfinite(item['logprob']) and item['logprob']!=-9999,
                'invalid actual action probability')
        if i>=len(r['tokens']) or i!=len(diffs):continue
        if observe_top1:
            tops=item['top_logprobs']
            # vLLM emits the sampled token and/or top candidates. Do not infer a unique argmax on ties.
            require(tops and all(math.isfinite(k['logprob']) for k in tops),'missing raw top candidates')
            winners = [x['token'] for x in tops if x['logprob']==max(k['logprob'] for k in tops)]
            flips.append({'position':i,'winner_matches':winners==[f"token_id:{r['raw_top1'][i]}"],
                          'tie':len(winners)>1})
        if t==r['tokens'][i]:diffs.append(item['logprob']-r['logprobs'][i])
    return dict(lane=r['lane'],expected=len(r['tokens']),actual=len(ids),common=len(diffs),
                complete=ids==r['tokens'],max_abs=max(map(abs,diffs),default=None),
                sequence_abs=abs(sum(diffs)) if ids==r['tokens'] else None,deltas=diffs,top1=flips)


def run(cfg):
    final=Path(cfg['after']);deadline=time.monotonic()+cfg['wait_seconds']
    while not final.is_file():
        require(time.monotonic()<deadline,'predecessor wait expired');time.sleep(3)
    require(doc(pin(final))['passed'] and doc(pin(final.with_name('DRAINED.json')))['guards_releasable'],
            'unsealed predecessor')
    template=doc(pin(cfg['template'])); s,=[s for s in template['services'] if s['name']==cfg['lane']]
    require(len(template['guards'])==1,'single host guard required')
    root=Path(cfg['output']);packet=doc(cfg['cases'])
    with open(template['guards'][0],'rb') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        service_identity(s);idle(metrics(s['endpoint']))
        cid=json.loads(subprocess.check_output(['docker','inspect',s['container']]))[0]
        require(cid['Id']==doc(cfg['start'])['container_id'] and cid['State']['Running'],'service replaced')
        alias='checkpoint-sha256:'+packet['cases'][0]['adapter_sha']
        overlay=cfg.get('overlay_active',False)
        model=packet['cases'][0]['wire']['model'] if overlay else alias
        if overlay:
            require('stock_adapter_alias' not in s,'mixed transport modes')
            active=json.loads(subprocess.check_output(['docker','exec',s['container'],'cat',
                              '/runtime/state/policy/ACTIVATION.json']))
            require(active['alias']==active['policy_version']==alias and active['weights_ref']['sha256']==alias.split(':')[1]
                    and active['service_identity']['container_name']==s['container'],'overlay identity mismatch')
        else:
            require(s['stock_adapter_alias']==alias,'static adapter identity mismatch')
        with urllib.request.urlopen(s['endpoint']+'/v1/models',timeout=10) as resp:
            registered=json.load(resp)['data']
        require(any(x['id']==alias and x.get('parent')==packet['cases'][0]['wire']['model'] for x in registered),
                'adapter not registered')
        require(not root.exists(),'reconcile preceding probe; no automatic retry')
        save(root/'STARTED.json',dict(cases=cfg['cases'],container=cid['Id'],training_use=False))
        def one(r):
            require(alias=='checkpoint-sha256:'+r['adapter_sha'],'wrong weights')
            response=contact(root/r['lane'],s['endpoint'],dict(r['wire'],model=model,top_logprobs=0 if overlay else 2),
                             cfg['timeout_seconds'])
            body=json.loads(doc(response['receipt_ref'])['raw'])
            return compare(r,body,model,observe_top1=not overlay)
        try:
            with concurrent.futures.ThreadPoolExecutor(cfg['concurrency']) as pool:rows=list(pool.map(one,packet['cases']))
            save(root/'RESULT.json',dict(rows=rows,limits=packet['limits'],training_use=False,
                    scope='natural generation; same prompt and token history only',top_logprobs=0 if overlay else 2))
        finally:
            while True:
                try:idle(metrics(s['endpoint']));break
                except OSError:time.sleep(3)
                except ValueError:time.sleep(3)
            save(root/'DRAINED.json',dict(guards_releasable=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();sp=p.add_subparsers(dest='action',required=True)
    a=sp.add_parser('prepare')
    for key in ('paired','hf','output'):a.add_argument('--'+key,type=Path,required=True)
    a=sp.add_parser('run');a.add_argument('--config',type=Path,required=True)
    a=p.parse_args()
    if a.action=='prepare':prepare(a.paired,a.hf,a.output)
    else:run(doc(pin(a.config)))

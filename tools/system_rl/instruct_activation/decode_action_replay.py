"""Prepare diagnostic chat wires from sealed actions and compare raw decode scores."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from io_utils import canonical, doc, pin, require, save


def choice(ref):
    receipt = doc(ref)
    raw = receipt['raw']
    require(receipt['status'] == 200 and hashlib.sha256(raw.encode()).hexdigest() ==
            receipt['response_sha256'], 'missing successful sealed HTTP result')
    value = json.loads(raw)
    row, = value['choices']
    ids = row['token_ids']
    content = row['logprobs']['content']
    require(ids and len(ids) == len(content) == value['usage']['completion_tokens'],
            'incomplete actions')
    for token, item in zip(ids, content, strict=True):
        require(type(token) is int and token >= 0 and item['token'] == f'token_id:{token}'
                and math.isfinite(item['logprob']) and item['logprob'] != -9999, 'invalid raw action')
    return ids, content


def prepare(cfg):
    wire = doc(cfg['wire'])
    intent = doc(cfg['intent'])
    ids, _ = choice(cfg['response'])
    salt = cfg['source_salt']
    require('cache_salt' not in wire and len(salt)==64 and
            hashlib.sha256(canonical(dict(wire,cache_salt=salt))).hexdigest() == intent['wire_sha256']
            and Path(cfg['intent']['path']).parent==Path(cfg['response']['path']).parent,
            'source conditioning or response identity changed')
    require(wire['n']==1 and wire['return_token_ids'] and wire['logprobs'] and
            wire['top_logprobs']==0 and not wire['stream'] and wire['chat_template_kwargs']=={'enable_thinking':False}
            and 'vllm_xargs' not in wire, 'expected original rollout request')
    return dict(wire,top_logprobs=2,max_tokens=len(ids),vllm_xargs={'decode_action_ids':ids})


def compare(cfg):
    threshold=cfg['max_logprob_delta']
    require(type(threshold) in (int,float) and 0<=threshold<=0.01 and cfg['pairs'], 'invalid tolerance or empty comparison')
    results=[]
    for pair in cfg['pairs']:
        wire=doc(pair['wire'])
        path=wire['vllm_xargs']['decode_action_ids']
        ref_ids,left=choice(pair['baseline'])
        ids,right=choice(pair['candidate'])
        require(ids==ref_ids==path,'observed replay action sequence mismatch')
        wire_hash=hashlib.sha256(canonical(dict(wire,cache_salt=pair['salt']))).hexdigest()
        for name in ('baseline','candidate'):
            require(doc(pin(Path(pair[name]['path']).with_name('INTENT.json')))['wire_sha256']==wire_hash,
                    'comparison input mismatch')
        def top(item):
            candidates=item['top_logprobs']
            # vLLM 0.30 serializes the sampled id first, then the raw top-k.
            # Two entries expose the true maximum even for a forced non-argmax.
            require(len(candidates)==2 and len({v['token'] for v in candidates})==2 and
                    all(v['token'].startswith('token_id:') and math.isfinite(v['logprob'])
                        and v['logprob']!=-9999 for v in candidates), 'missing raw top candidates')
            best=max(candidates,key=lambda v:v['logprob'])
            require(candidates[0]['logprob']!=candidates[1]['logprob'],
                    'ambiguous top-1 tie requires raw engine ranks')
            return best['token']
        scores=[(a['logprob'],b['logprob']) for a,b in zip(left,right,strict=True)]
        peak=max(abs(a-b) for a,b in scores)
        total=abs(math.fsum(a-b for a,b in scores))
        flips=sum(top(a)!=top(b) for a,b in zip(left,right,strict=True))
        results.append(dict(tokens=len(path),max_token_delta=peak,sequence_delta=total,top1_flips=flips,
                            passed=peak<=threshold and total<=threshold and flips==0,input_refs=pair))
    return dict(pairs=results,passed=all(r['passed'] for r in results),training_use=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','compare'))
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    config=doc(pin(args.config))
    result=(prepare if args.action=='prepare' else compare)(config)
    save(args.output,result)
    if result.get('passed') is False:
        raise SystemExit(2)

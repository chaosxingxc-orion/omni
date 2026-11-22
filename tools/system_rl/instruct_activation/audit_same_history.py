"""Compare repeated actual scores for the identical multimodal input, history and action."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from io_utils import digest, doc, pin, require
from probe_native_scores import compare


def audit(root):
    start=doc(pin(root/'STARTED.json')); packet=doc(start['cases'])
    require(doc(pin(root/'DRAINED.json'))['guards_releasable'],'requests not drained')
    recorded=doc(pin(root/'RESULT.json')); original={r['lane']:r for r in recorded['rows']}
    require(len(original)==len(recorded['rows'])==len(packet['cases']),'missing or duplicate case')
    groups=defaultdict(list); reference=defaultdict(list); inputs=set(); n=0
    for case in packet['cases']:
        folder=root/case['lane']; intent=doc(pin(folder/'INTENT.json')); resp=doc(pin(folder/'RESPONSE.json'))
        body=json.loads(resp['raw']); model=body['model']
        require(model in (case['wire']['model'],'checkpoint-sha256:'+case['adapter_sha']),'unexpected response model')
        top=recorded.get('top_logprobs',2); require(top in (0,2),'unknown observation mode')
        wire=dict(case['wire'],model=model,top_logprobs=top)
        require(intent['wire_sha256']==digest(wire) and resp['status']==200 and
                compare(case,body,model,observe_top1=top==2)==original[case['lane']],'request or verified score drift')
        # Sampling seed chooses the action; compare raw probabilities only conditional on that same action.
        ident=(digest({k:v for k,v in wire.items() if k!='seed'}),case['adapter_sha'],digest(body['prompt_token_ids']))
        inputs.add(ident)
        choice,=body['choices']
        for index,(token,logp) in enumerate(zip(choice['token_ids'],choice['logprobs']['content'],strict=True)):
            groups[ident,tuple(choice['token_ids'][:index]),token].append(logp['logprob']);n+=1
        for index,(token,logp) in enumerate(zip(case['tokens'],case['logprobs'],strict=True)):
            reference[ident,tuple(case['tokens'][:index]),token].append(logp)
    def stats(h):
        spans=[max(v)-min(v) for v in h.values() if len(v)>1];tol=packet['limits']['token']
        return dict(common_keys=len(spans),above_tolerance=sum(s>tol for s in spans),
               above_twice_tolerance=sum(s>2*tol for s in spans),max_span=max(spans,default=None))
    return dict(requests=len(original),distinct_multimodal_inputs=len(inputs),actual_actions=n,
                token_tolerance=packet['limits']['token'],engine=stats(groups),sealed_reference=stats(reference),
                result_sha256=pin(root/'RESULT.json')['sha256'])

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('roots',nargs='+',type=Path)
    print(json.dumps({r.name:audit(r) for r in p.parse_args().roots},indent=2))

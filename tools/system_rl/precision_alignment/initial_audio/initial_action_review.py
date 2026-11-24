"""Bind natural responses to the same original prompts, then score exactly those actions in HF eval."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def wire_hash(wire):
    return hashlib.sha256(json.dumps(wire,ensure_ascii=False).encode()).hexdigest()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(inputs, service, output):
    from deploy_resident_graph_lanes import read,save
    d=service/'natural-comparison'
    intent=read(d/'INTENT.json')
    ready=read(service/(intent['lane']+'-READY.json'))
    if (digest(inputs)!=intent['input_sha256'] or ready['activation']!=intent['policy'] or
        digest(service/(intent['lane']+'-READY.json'))!=read(service/'READY.json')['lanes'][intent['lane']]):
        raise ValueError('different deployment or cohort')
    original=read(inputs); rows=sorted(original['rows'],key=lambda r:r['row']); items=[]
    if len(rows)!=intent['count'] or [r['row'] for r in rows]!=list(range(len(rows))):
        raise ValueError('incomplete row roster')
    report=read(d/'RESULT.json')
    for r in rows:
        i=r['row']; path=d/f'{i}-response.json'; raw=read(path)
        if (read(d/f'{i}-intent.json')!=dict(row=i,wire_sha256=wire_hash(r['wire']))
                or raw['prompt_token_ids']!=r['prompt'] or raw['model']!=r['wire']['model']):
            raise ValueError('different served input')
        c,=raw['choices'];probs=[]
        for t,e in zip(c['token_ids'],c['logprobs']['content'],strict=True):
            if e['token']!=f'token_id:{t}' or not math.isfinite(e['logprob']):raise ValueError('invalid response')
            probs.append(e['logprob'])
        if not probs:raise ValueError('no actions')
        items.append(dict(row=i,prompt=r['prompt'],original=r['tokens'],tokens=c['token_ids'],
                          log_probs=probs,response_sha256=digest(path)))
    if sorted(r['row'] for r in report['rows'])!=list(range(len(rows))):raise ValueError('incomplete terminal result')
    save(output,dict(dataset=original['dataset'],policy_version=intent['policy']['version'],input_sha256=digest(inputs),
                     result_sha256=digest(d/'RESULT.json'),rows=items))


def action_fields(fields,items):
    """Use only for no-update evaluation. Preserve all original prompt and audio tensors."""
    import torch
    t=fields['tensors']; plen=t['prompts'].shape[1]
    if len(items)!=len(t['responses']):raise ValueError('actor and service populations differ')
    width=max(len(i['tokens']) for i in items)
    oldmask=t['response_mask'].bool()
    pads=t['responses'][~oldmask].unique().tolist()
    if len(pads)!=1:raise ValueError('ambiguous padding ID')
    response=t['responses'].new_full((len(items),width),pads[0]);mask=t['response_mask'].new_zeros(response.shape)
    pos=t['position_ids'].new_zeros((len(items),t['position_ids'].shape[1],width))
    for n,r in enumerate(items):
        prompt_mask=t['attention_mask'][n,:plen].bool()
        if (r['row']!=n or t['prompts'][n,prompt_mask].tolist()!=r['prompt']
                or t['responses'][n,oldmask[n]].tolist()!=r['original']):
            raise ValueError('action replay row identity differs')
        old=t['position_ids'][n,:,plen:][:,oldmask[n]]
        if not torch.equal(old,old[:,:1]+torch.arange(old.shape[-1])):
            raise ValueError('unsupported action position geometry')
        length=len(r['tokens'])
        pos[n,:,:length]=old[:,:1]+torch.arange(length)
        response[n,:length]=response.new_tensor(r['tokens']);mask[n,:length]=1
    t.update(input_ids=torch.cat((t['input_ids'][:,:plen],response),1),responses=response,response_mask=mask,
             attention_mask=torch.cat((t['attention_mask'][:,:plen],mask),1),
             position_ids=torch.cat((t['position_ids'][:,:,:plen],pos),2))


def review(inputs,ranks,out):
    from deploy_resident_graph_lanes import read,save
    c=read(inputs);rows={};exit=read(ranks/'EXIT.json')
    if exit['exit_code']!=0 or exit['complete_ranks']<=0:raise ValueError('actor incomplete')
    for i in range(exit['complete_ranks']):
        result=read(ranks/f'rank-{i}/PROFILE-EVAL.json')
        if result['action_review_sha256']!=digest(inputs) or result['dataset']!=c['dataset']:raise ValueError('different evaluation')
        for row in result['rows']:
            if row['row'] in rows:raise ValueError('duplicate actor row')
            rows[row['row']]=row['log_probs']
    values=[]
    for item in c['rows']:
        values.extend(abs(a-b) for a,b in zip(rows.pop(item['row']),item['log_probs'],strict=True))
    if rows or not values or not all(map(math.isfinite,values)):raise ValueError('invalid population')
    save(out,dict(rows=len(c['rows']),tokens=len(values),nonzero=sum(v!=0 for v in values),max_abs=max(values),
                  policy=c['policy_version'],action_review_sha256=digest(inputs),formal_updates=0))


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='mode',required=True)
    for name,keys in [('prepare',('inputs','service','output')),('review',('inputs','ranks','out'))]:
        cmd=sub.add_parser(name)
        for key in keys:cmd.add_argument(key,type=Path)
    a=vars(p.parse_args()); mode=a.pop('mode'); (prepare if mode=='prepare' else review)(**a)

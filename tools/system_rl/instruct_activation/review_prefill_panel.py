"""Compare every natural rollout action with restored HF train/eval and isolated raw top-1."""
import math
import sys
import time
from pathlib import Path
from decode_action_replay import choice
from io_utils import digest, doc, pin, require, save


def raw_winner(item):
    entries=item['top_logprobs']
    require(len(entries)>=2 and len({e['token'] for e in entries})==len(entries)
            and all(e['token'].startswith('token_id:') and math.isfinite(e['logprob'])
                    and -9999<e['logprob']<=0 for e in entries), 'missing raw top candidates')
    maximum=max(e['logprob'] for e in entries)
    winners=[int(e['token'][9:]) for e in entries if e['logprob']==maximum]
    return winners[0] if len(winners)==1 else None


def review(c):
    cfg=doc(c['hf'])
    root=Path(cfg['hot'])
    require(doc(pin(root/'EXIT.json'))['exit_code']==0 and doc(pin(root/'EXIT.json'))['complete_ranks']==8,
            'incomplete FSDP8 run')
    source=doc(pin(c['source']))
    pn=pin(root.parent/cfg['updated_http_parity'])
    panel=doc(pn)
    require(doc(pin(root.parent/'PANEL-SOURCE.json'))['source']==pin(c['source'])
            and len(panel['rows'])==len(source['rows']),'training input drift')
    replay_cfg=doc(pin(c['replay']))
    replay_root=Path(replay_cfg['output'])
    require(doc(pin(replay_root/'DRAINED.json'))['guards_releasable'] and
            doc(pin(replay_root/'RESULT.json'))['passed'],'incomplete replay')
    rr,=doc(pin(replay_root/'RESULT.json'))['rounds']
    run=doc(rr)
    entries=sorted(run['requests'],key=lambda r:(r['group'],r['replicate']))
    require(run['input_refs']==[r['replay'] for r in source['rows']] and
            all(r['service']==c['lane'] for r in entries) and
            run['concurrency_per_service']=={c['lane']:panel['width']},'wrong replay panel')
    delta=doc(pin(Path(c['replay']).parent/'REPLAY-DELTA.json'))
    from decode_lineage import snapshot
    require(delta['parent_sha256']==digest(snapshot(Path(c['natural_packet']))),'different arithmetic')
    l=c['limits']
    require(0<=l['token']<=.01 and 0<=l['sequence']<=.01 and l['max_argmax_flips']==0,'limits')
    scored={}
    for rank in range(8):
        f=doc(pin(root/f'rank-{rank}/RESULT.json'))
        require(f['rank']==rank and f['optimizer_updates']==0 and f['panel']==pn['sha256'] and
                f['adapter']==panel['adapter_sha'] and f['checkpoint']==panel['checkpoint_sha'] and
                f['step']==cfg['parity_restore_step'], 'wrong training response')
        for x in f['rows']:
            i=x['index']
            require(i not in scored and i%8==rank,'duplicate or misassigned row')
            scored[i]=x
    require(set(scored)==set(range(len(entries)))==set(range(len(panel['rows']))),'incomplete roster')
    resolved, evidence = {}, None
    if 'argmax_observer' in c:
        from review_replay_argmax import resolve
        resolved,evidence=resolve(c['argmax_observer'],Path(c['replay']).parent,source,panel)
    results=[]
    for i,(natural,r,src) in enumerate(zip(panel['rows'],entries,source['rows'],strict=True)):
        ids,left=choice(src['response'])
        observed,right=choice(r['receipt_ref'])
        x=scored[i]
        replay_wire=dict(doc(src['replay']),cache_salt=digest([replay_cfg['measurement_id'],run['block']]))
        intent=doc(pin(Path(r['receipt_ref']['path']).with_name('INTENT.json')))
        require(ids==observed==natural['tokens']==replay_wire['vllm_xargs']['decode_action_ids']
                and intent['wire_sha256']==digest(replay_wire) and not intent['automatic_retry']
                and (r['group'],r['replicate'])==divmod(i,panel['width']) and
                doc(src['response'])['response_sha256']==x['response']==natural['response'],
                'wrong action, batch or response')
        import json
        response=json.loads(doc(r['receipt_ref'])['raw'])
        require(response['prompt_token_ids']==natural['prompt'] and response['model']==replay_wire['model'],
                'different model or expanded prompt')
        lp=[e['logprob'] for e in left];rp=[e['logprob'] for e in right]
        require(lp==natural['logprobs'],'natural score drift')
        winners=[raw_winner(e) for e in right]
        for pos,item in enumerate(right):
            if (i,pos) in resolved:
                winner,tied=resolved.pop((i,pos))
                tops=item['top_logprobs']
                maximum=max(t['logprob'] for t in tops)
                require(winners[pos] is None and tied=={int(t['token'][9:]) for t in tops if t['logprob']==maximum},
                        'observer did not reproduce original tied candidates')
                winners[pos]=winner
        row=dict(index=i,n=len(ids),replay_changed=sum(a!=b for a,b in zip(lp,rp,strict=True)),
                 ambiguous_top1=winners.count(None),modes={})
        for mode,k,t in [('eval','scores','hf_top1'),('train','train_scores','train_top1')]:
            diffs=[a-b for a,b in zip(x[k],lp,strict=True)]
            require(all(math.isfinite(d) for d in diffs),'non-finite training score')
            flips=sum(a!=b for a,b in zip(x[t],winners,strict=True) if b is not None)
            row['modes'][mode]=dict(max_token_abs=max(map(abs,diffs)),sequence_abs=abs(math.fsum(diffs)),
                                    over_token=sum(abs(d)>l['token'] for d in diffs),top1_flips=flips)
        results.append(row)
    def passed(r):
        return r['replay_changed']==r['ambiguous_top1']==0 and all(
            z['max_token_abs']<=l['token'] and z['sequence_abs']<=l['sequence'] and z['top1_flips']==0
            for z in r['modes'].values())
    require(not resolved,'unconsumed observer evidence')
    return dict(**({'argmax_evidence':evidence} if evidence else {}),requests=len(results),actions=sum(r['n'] for r in results),
                passed_requests=sum(map(passed,results)),passed=all(map(passed,results)),
                limits=l,rows=results,step=cfg['parity_restore_step'],adapter=panel['adapter_sha'],
                runtime=delta['parent_sha256'],checkpoint=panel['checkpoint_sha'],hf_panel=pn['sha256'],
                natural_round=source['round']['sha256'],replay_round=rr['sha256'],
                scope='same-checkpoint natural audio and fixed 8-way batch; train/eval forward; zero optimizer updates')


if __name__=='__main__':
    c=doc(pin(sys.argv[1]))
    if 'wait_seconds' in c:
        terminal=Path(doc(c['hf'])['hot'])/'EXIT.json'
        deadline=time.monotonic()+c['wait_seconds']
        while not terminal.exists():
            require(time.monotonic()<deadline,'FSDP8 scoring not terminal')
            time.sleep(5)
    result=review(c)
    save(c['output'],result)
    if 'aggregate_output' in c:
        rows=result['rows']
        modes={}
        for mode in ('eval','train'):
            scores=[row['modes'][mode] for row in rows]
            modes[mode]=dict(max_token_abs=max(s['max_token_abs'] for s in scores),
                             max_sequence_abs=max(s['sequence_abs'] for s in scores),
                             over_token=sum(s['over_token'] for s in scores),
                             over_sequence=sum(s['sequence_abs']>result['limits']['sequence'] for s in scores),
                             top1_flips=sum(s['top1_flips'] for s in scores))
        evidence=result.get('argmax_evidence',{})
        save(c['aggregate_output'],dict(
            **{k:result[k] for k in ('requests','actions','passed_requests','passed','limits',
               'step','adapter','checkpoint','runtime','natural_round','replay_round','hf_panel','scope')},
            result_sha256=pin(c['output'])['sha256'],modes=modes,
            replay_changed=sum(row['replay_changed'] for row in rows),
            unresolved_top1=sum(row['ambiguous_top1'] for row in rows),
            full_vocab_observations=len(evidence.get('observations',[])),
            observer_round=evidence.get('round_sha256'),optimizer_updates=0,
            production_rollout=False))
    sys.exit(0 if result['passed'] else 2)

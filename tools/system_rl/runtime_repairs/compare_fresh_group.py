"""Score every fresh action, retaining the denominator and tie-aware argmax gate."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from collect_retained_boundaries import store


def compare(hot, spec_path, tp_path):
    def load(p):return json.loads(p.read_bytes())
    s=load(spec_path);tp=load(tp_path)
    if hashlib.sha256(tp_path.read_bytes()).hexdigest()!=s['files'][s['fresh_group']['result']]:
        raise ValueError('wrong TP result')
    done=load(hot/'EXIT.json')
    n=s['training']['world_size']
    if n!=4 or done['exit_code'] or done['failure'] or done['complete_ranks']!=n:
        raise ValueError('HF4 run incomplete')
    if tp['actual_tp']!=4 or tp['fresh_group']['manifest_sha256']!=s['files'][s['fresh_group']['manifest']]:
        raise ValueError('wrong rollout topology or TRAIN request manifest')
    router_layers=s['fresh_group'].get('router_layers')
    if router_layers is not None:
        def check_router(receipt):
            if (not receipt or receipt['arithmetic']!='full_width_bf16_router_grouped_matmul'
                    or set(receipt['layers'])!={str(i) for i in router_layers}
                    or not all(type(v) is int and v>0 for v in receipt['layers'].values())):
                raise ValueError('missing actual router projection on a selected layer')
        tp_ranks=tp['decode_row_receipts']
        if sorted(r['rank'] for r in tp_ranks)!=list(range(n)):
            raise ValueError('missing TP worker')
        for r in tp_ranks:check_router(r.get('router'))
    expected=tp['fresh_group']['records']; rows=[]
    for rank in range(n):
        x=load(hot/f'rank-{rank}'/'RESULT.json')
        if x['rank']!=rank or x['updates']!=0: raise ValueError('rank or model changed')
        if router_layers is not None:check_router(x.get('router'))
        for k in ('gate','down','attention'):
            if set(x[k]['layers'])!={str(i) for i in range(48)} or not all(v>0 for v in x[k]['layers'].values()):
                raise ValueError('missing model layer')
        for r in x['records']:
            i=r['index'];t=expected[i]
            if (i%n!=rank or r['request_id']!=t['request_id'] or r['tokens']!=len(t['token_ids']) or
                not r['tokens'] or len(r['hf_top1'])!=r['tokens']):
                raise ValueError('missing owned actions')
            delta=[a-b for a,b in zip(r['scores'],t['behavior_logprobs'],strict=True)]
            if not all(map(math.isfinite,delta)) or r['delta']!=delta:raise ValueError('invalid scores')
            # Early versions only recorded strict flips. With any such flip,
            # absent full-vocabulary tie evidence cannot authorize a pass.
            strict=r['top1_flips'];bad=r.get('inadmissible_top1',strict)
            m=max(map(abs,delta));seq=abs(math.fsum(delta))
            rows.append(dict(index=i,group=t['group'],replicate=t['replicate'],tokens=len(delta),
                             max_abs=m,sequence_abs=seq,strict_flips=strict,inadmissible_or_unverified=bad,
                             passed=m<=.01 and seq<=.01 and not bad))
    if sorted(r['index'] for r in rows)!=list(range(len(expected))) or not rows:
        raise ValueError('incomplete population')
    return dict(requests=len(rows),tokens=sum(r['tokens'] for r in rows),passed_requests=sum(r['passed'] for r in rows),
                max_abs=max(r['max_abs'] for r in rows),max_sequence_abs=max(r['sequence_abs'] for r in rows),
                rows=sorted(rows,key=lambda r:r['index']),production_acceptance=False)


def compare_waveform_control(hot,spec_path,tp_path,control_path,control_exit):
    """Reuse an HF score only if the entire sampled action and context are unchanged."""
    x=compare(hot,spec_path,tp_path)
    src=json.loads(tp_path.read_bytes());ctrl=json.loads(control_path.read_bytes())
    done=json.loads(control_exit.read_bytes())
    if done['exit_code'] or not done['result_exists'] or ctrl['actual_tp']!=4:
        raise ValueError('incomplete TP4 control')
    if src['fresh_group']['manifest_sha256']!=ctrl['fresh_group']['manifest_sha256']:
        raise ValueError('waveform control changed the source packets')
    rows=[];unmatched=[]
    for row,a,b in zip(x['rows'],src['fresh_group']['records'],ctrl['fresh_group']['records'],strict=True):
        i=row['index']
        if not all(a[k]==b[k] for k in ('request_id','group','replicate','wire_sha256','prompt_tokens','seed')):
            raise ValueError('control changed request identity')
        if a['token_ids']!=b['token_ids']:
            unmatched.append(i);continue
        h=next(r for r in json.loads((hot/f'rank-{i%4}'/'RESULT.json').read_bytes())['records'] if r['index']==i)
        d=[u-v for u,v in zip(h['scores'],b['behavior_logprobs'],strict=True)]
        if not all(map(math.isfinite,d)):raise ValueError('invalid control score')
        flips=[j for j,(u,v) in enumerate(zip(h['hf_top1'],b['raw_top1'],strict=True),1) if u!=v]
        m=max(map(abs,d));seq=abs(math.fsum(d))
        rows.append(dict(index=i,tokens=len(d),max_abs=m,sequence_abs=seq,strict_flips=flips,
                         passed=m<=.01 and seq<=.01 and not flips))
    return dict(total_requests=x['requests'],changed_actions_require_hf=unmatched,
                compared=sum(r['tokens'] for r in rows),strict_passed_requests=sum(r['passed'] for r in rows),
                rows=rows,production_acceptance=False)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('hot','spec','tp','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--control',type=Path);p.add_argument('--control-exit',type=Path)
    a=p.parse_args()
    if bool(a.control)!=bool(a.control_exit):p.error('control and its exit required together')
    result=(compare_waveform_control(a.hot,a.spec,a.tp,a.control,a.control_exit)
            if a.control else compare(a.hot,a.spec,a.tp))
    store(a.output,json.dumps(result,indent=2).encode())

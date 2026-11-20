"""Score the exact observed HTTP actions against the paired restored actor reports."""
import argparse
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_bytes())


def review(packet, hf, checkpoint_sha, geometry):
    rows=packet['rows'];limits=packet['limits'];layers=geometry['layers']
    branches={'qkvo_lora_a':'qkvo','qkv_lora_a':'qkv','o_lora_a':'o'}[geometry['site']]
    expected_sites={f'{i}.{s}' for i in layers for s in branches}
    widths=({'2048'} if 'q' in branches else set())|({'4096'} if 'o' in branches else set())
    exit_=read(hf/'EXIT.json')
    if (exit_['exit_code']!=0 or exit_['complete_ranks']!=8 or exit_['failure'] is not None
            or not 0<len(rows)<=8 or packet['training_use'] is not False):raise ValueError('unclosed or wrong cohort')
    summaries=[dict(lane=r['lane'],tokens=len(r['tokens']),ranks=[],passed=True) for r in rows]
    for rank in range(8):
        i=rank%len(rows);row=rows[i];rec=read(hf/f'rank-{rank}/RESULT.json')
        ops=rec['lora_row'];counts=ops['calls_by_layer_and_branch'];grad=ops['gradient_control']
        if not (rec['rank']==rank and rec['lane']==row['lane'] and rec['tokens']==len(row['tokens'])
                and rec['response_sha']==row['response'] and rec['observed_policy_artifact']==row['adapter_sha']
                and rec['checkpoint_sha']==checkpoint_sha and rec['restored_step']==1 and rec['optimizer_updates']==0
                and ops['algorithm']==geometry['algorithm'] and set(counts)==expected_sites and all(n>0 for n in counts.values())
                and set(grad)==widths and all(set(v)=={'input','fp32_master'} and
                        all(g['elements']>0 and g['differences']==0 for g in v.values()) for v in grad.values())):
            raise ValueError(f'unbound response, checkpoint, or gradient evidence: {rank}')
        s=summaries[i];s['ranks'].append(rank)
        passed=True
        for mode in ('train','eval'):
            vals={k:rec[mode+'_'+k] for k in ('max_abs','sequence_abs','top1_flips')}
            for k in ('max_abs','sequence_abs'):
                if not math.isfinite(vals[k]) or vals[k]<0:raise ValueError('invalid measured difference')
            passed=passed and (vals['max_abs']<=limits['token'] and vals['sequence_abs']<=limits['sequence']
                               and len(vals['top1_flips'])<=limits['max_argmax_flips'])
            if mode in s and s[mode]!=vals:raise ValueError('repeated HF scores differ on identical action')
            s[mode]=vals
        if passed!=rec['passed']:raise ValueError('criterion disagrees with worker')
        s['passed'] &= passed
    return dict(requests=len(rows),actions=sum(s['tokens'] for s in summaries),limits=limits,
                all_passed=all(s['passed'] for s in summaries),sites_per_rank=len(expected_sites),
                gradient_widths=sorted(widths),checkpoint_sha256=checkpoint_sha,optimizer_updates=0,results=summaries)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('packet','hf','config'):p.add_argument('--'+key,required=True,type=Path)
    p.add_argument('--checkpoint-sha',required=True)
    a=p.parse_args()
    print(json.dumps(review(read(a.packet),a.hf,a.checkpoint_sha,read(a.config)['lora_row_geometry']),indent=2))

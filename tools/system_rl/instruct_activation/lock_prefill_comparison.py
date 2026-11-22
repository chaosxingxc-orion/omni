"""Freeze a performance research baseline only from attributable paired RL responses."""
import argparse
import json
import math
from pathlib import Path
from compare_cold_prefill import compare
from decode_lineage import changes, snapshot
from io_utils import digest, doc, pin, require, save


def lock(c):
    p, b = (Path(c[k]).resolve(strict=True) for k in ('parent_packet', 'candidate_packet'))
    inputs = {k:Path(c[k]) for k in ('parent','candidate','gold','parser')}
    result = compare(**inputs, lane=c['lane'])
    for side, packet in [('parent',p),('candidate',b)]:
        source = doc(pin(inputs[side]))
        require(source['measurement_id']==packet.name+'-latency' and
                source['output']==str(packet.parent/(packet.name+'-latency')/'rounds'),
                'study and runtime packet do not match')
        started=doc(pin(source['start_receipts'][c['lane']]))
        cut=doc(pin(packet.parent/(packet.name+'-switch.local.json')))
        lan,=cut['lanes']
        require(source['start_receipts'][c['lane']]==str(Path(cut['output'])/('STARTED-'+str(doc(pin(Path(lan['new_code'])/'config.json'))['port'])+'.json'))
                and Path(lan['new_code'])==packet/('lane-'+c['lane'])
                and len(started['physical_devices'])==4, 'unbound runtime packet')
    base, new = snapshot(p), snapshot(b)
    delta = changes(base, new)
    require(not delta['engine'] and set(delta['files'])=={c['changed_file']}
            and set(delta['rows'])=={c['changed_row']},'unrelated runtime differences')
    for row in result['batches']:
        require(row['saved_s']>0, 'parent faster on a measured batch')
    require(0 < c['min_relative_saving'] < 1 and result['relative_batch_reduction']>=c['min_relative_saving'],
            'insufficient measured improvement')
    score=b.parent/(b.name+'-score')/'RESULT.json'; limits=doc(pin(score))['limits']
    require(doc(pin(score.with_name('DRAINED.json')))['guards_releasable'],'open HF ledger')
    scores=doc(pin(score))['rows'];common=[abs(d) for r in scores for d in r['deltas']]
    full=[r['sequence_abs'] for r in scores if r['complete']]
    require(common and full and all(math.isfinite(d) and d<=limits['token'] for d in common)
            and all(math.isfinite(d) and d<=limits['sequence'] for d in full),'reference score limit exceeded')
    return dict(version=c['version'],scope='P prefill engineering comparison, fixed batch of 8',
                production_rollout=False, runtime_sha256=digest(new),parent_runtime_sha256=digest(base),
                image=new['engine']['image'], policy_sha256=new['engine']['policy_sha256'],
                implementation=delta['files'],row_change=delta['rows'],paired=result,
                hf_diagnostic=dict(result_sha256=pin(score)['sha256'],requests=len(scores),common=len(common),
                                   complete=len(full),max_token_delta=max(common),max_sequence_delta=max(full),
                                   independent_top1_observed=any(r['top1'] for r in scores)),
                limitations='One panel. Retains sampling, no independent top-1 or formal training acceptance.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('config',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args()
    print(json.dumps(save(a.output,lock(doc(pin(a.config))))))

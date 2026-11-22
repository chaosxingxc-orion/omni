"""Freeze a performance research baseline only from attributable paired RL responses."""
import argparse
import json
import math
from pathlib import Path
from compare_cold_prefill import compare
from decode_lineage import changes, snapshot
from io_utils import digest, doc, pin, require, save


def performance_gate(config, paired, cold=None):
    if 'native_target' in config:
        from cold_native_window import compare as compare_native
        from review_native_target import latency_target
        if cold is None:
            cold = compare_native(dict(config, candidate_config=config['candidate']))
        return latency_target(doc(config['native_target']), cold, paired['candidate_round'])
    threshold = config['min_relative_saving']
    require(0 < threshold < 1, 'invalid parent saving threshold')
    return dict(within_target=all(row['saved_s'] > 0 for row in paired['batches'])
                and paired['relative_batch_reduction'] >= threshold)


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
    engine_delta = c.get('expected_engine_delta', {})
    if 'components' in c:
        manifest = doc(c['components'])
        require(Path(c['components']['path']) == b/'COMPONENTS.json' and
                manifest['parent'] == digest(base) and manifest['changes'] == delta and
                not delta['engine'] and len(manifest['components']) == 2 and
                all(not component['engine'] and len(component['files']) == len(component['rows']) == 1
                    for component in manifest['components']) and
                all(len(delta[key]) == 2 and
                    delta[key] == manifest['components'][0][key] | manifest['components'][1][key]
                    for key in ('files','rows')), 'unrelated component change')
    elif c.get('change_scope') == 'scheduler_only':
        require(set(engine_delta) == {'serve_args'} and delta['engine'] == engine_delta
                and not delta['files'] and not delta['rows'], 'unrelated scheduler trial change')
    else:
        require(delta['engine'] == engine_delta and set(delta['files']) == {c['changed_file']}
                and set(delta['rows']) == {c['changed_row']}, 'unrelated runtime differences')
    if 'parent_performance_lock' in c:
        require(digest(base)==doc(pin(c['parent_performance_lock']))['runtime_sha256'], 'unlocked parent')
    gate = performance_gate(c, result)
    require(gate['within_target'], 'configured latency target missed')
    score=b.parent/(b.name+'-score')/'RESULT.json'; limits=doc(pin(score))['limits']
    require(doc(pin(score.with_name('DRAINED.json')))['guards_releasable'],'open HF ledger')
    scores=doc(pin(score))['rows'];common=[abs(d) for r in scores for d in r['deltas']]
    full=[r['sequence_abs'] for r in scores if r['complete']]
    require(common and full and all(math.isfinite(d) and d<=limits['token'] for d in common)
            and all(math.isfinite(d) and d<=limits['sequence'] for d in full),'reference score limit exceeded')
    return dict(version=c['version'],scope='P prefill engineering comparison, fixed batch of 8',
                production_rollout=False, runtime_sha256=digest(new),parent_runtime_sha256=digest(base),
                image=new['engine']['image'], policy_sha256=new['engine']['policy_sha256'],
                **({'native_target':gate} if 'native_target' in c else {}),
                implementation=delta['files'],engine_change=delta['engine'],row_change=delta['rows'],paired=result,
                hf_diagnostic=dict(result_sha256=pin(score)['sha256'],requests=len(scores),common=len(common),
                                   complete=len(full),max_token_delta=max(common),max_sequence_delta=max(full),
                                   independent_top1_observed=any(r['top1'] for r in scores)),
                limitations='One panel. Retains sampling, no independent top-1 or formal training acceptance.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('config',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args()
    print(json.dumps(save(a.output,lock(doc(pin(a.config))))))

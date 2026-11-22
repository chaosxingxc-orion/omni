"""After a closed finite measurement, retain an eligible candidate or try a prepared sibling."""
from pathlib import Path
import sys
import time

from decode_lineage import snapshot
from io_utils import digest, doc, pin, require, save
from lock_prefill_comparison import lock
from report_prefill_candidate import report
from run_isolated_candidate import run


def assess(c):
    spec = doc(c['report'])
    lock_spec = doc(c['lock'])
    root = Path(lock_spec['candidate_packet'])
    parent = Path(lock_spec['parent_packet'])
    score = Path(spec['historical_score'])
    require(Path(spec['candidate_config']) == Path(lock_spec['candidate'])
            and Path(spec['parent_config']) == Path(lock_spec['parent'])
            and score == root.parent/(root.name+'-score')
            and root.parent == parent.parent, 'reports measure different runtimes')
    end = time.monotonic()+c['wait_seconds']
    while not (score/'DRAINED.json').exists():
        require(time.monotonic()<end,'preceding study has not drained')
        time.sleep(5)
    require(doc(pin(score/'DRAINED.json'))['guards_releasable'], 'open score')
    evidence = report(spec)
    summary = save(spec['output'],evidence)
    pair = evidence['paired']
    enough = (all(r['saved_s']>0 for r in pair['batches'])
              and pair['relative_batch_reduction'] >= lock_spec['min_relative_saving'])
    hf = evidence['historical_hf']
    limits = evidence['train']['limits']
    accurate = (all(m['over_token']==m['over_sequence']==0 for m in evidence['train']['modes'].values())
                and hf['common']>0 and hf['complete']>0 and hf['max_action_abs']<=limits['token']
                and hf['max_sequence_abs']<=limits['sequence'])
    if enough and accurate:
        perf = save(c['lock_output'], lock(lock_spec))
        save(c['decision'],dict(action='retain_for_full_precision',report=summary,performance_lock=perf))
        return
    if enough or 'sibling_flow' not in c:
        action = 'retain_for_divergence_analysis' if enough else 'no_promotion'
        save(c['decision'],dict(action=action,report=summary,observed_accuracy_within_limits=accurate))
        return
    flow = doc(c['sibling_flow'])
    cut = doc(pin(flow['cutover']))
    lane, = cut['lanes']
    sibling = Path(lane['new_code']).parent
    derivation=doc(c['sibling_proof'])
    require(Path(lane['old_code']).parent==root and Path(cut['terminal'])==score/'DRAINED.json'
            and sibling==Path(c['sibling_proof']['path']).parent
            and derivation.get('immediate_parent',derivation.get('parent'))==digest(snapshot(parent)), 'unrelated fallback')
    save(c['decision'],dict(action='measure_sibling',report=summary,observed_accuracy_within_limits=accurate))
    run(flow)


if __name__=='__main__':
    assess(doc(pin(sys.argv[1])))

"""Summarize completed paired C8 windows and their overlap with sealed training scores."""
from pathlib import Path
import sys

from batch_window import analyze
from compare_cached_train_forwards import compare as train_overlap
from compare_cold_prefill import compare as paired_windows
from io_utils import doc, pin, require, save
from window_latency_histogram import audit


def report(cfg):
    candidate = Path(cfg['candidate_config'])
    parent, gold, parser = (Path(cfg[key]) for key in ('parent_config','gold','parser'))
    pair = paired_windows(parent,candidate,gold,parser,cfg['lane'])
    windows = analyze(candidate,gold,parser)
    rounds = Path(doc(pin(candidate))['output'])
    hist = audit(rounds)
    ref, = doc(pin(rounds/'RESULT.json'))['rounds']
    metrics = doc(ref)['metrics']
    cache = {lane: {k: m[k] for k in ('mm_hits','mm_queries','prefix_hits','prefix_queries')}
             for lane, m in metrics.items()}
    cold_comparable = all(m['mm_hits'] == 0 and m['mm_queries'] == w['requests']
                          for w in windows for m in [cache[w['lane']]])
    training = train_overlap(cfg)
    require(pair['candidate_round'] == training['candidate_round'], 'training window differs')
    score, drain = (doc(pin(Path(cfg['historical_score'])/file)) for file in ('RESULT.json','DRAINED.json'))
    require(drain['guards_releasable'], 'historical diagnostic outstanding')
    deltas = [abs(d) for r in score['rows'] for d in r['deltas']]
    complete = [r['sequence_abs'] for r in score['rows'] if r['complete']]
    keys = ('lane','valid','requests','mean_batch_s','valid_per_second','round_sha256','token_ranges')
    return dict(paired=pair, cache=cache, both_lanes_audio_cold=cold_comparable, windows=[{k:w[k] for k in keys} for w in windows],
                itl=[dict(lane=v['lane'],n=v['itl']['n'],p50=v['itl']['p50_bounds'],p95=v['itl']['p95_bounds'],
                          over_half=sum(b['count'] for b in v['itl']['bins'] if b['lower']>=.5)) for v in hist],
                train=training, historical_hf=dict(requests=len(score['rows']),common=len(deltas),
                complete=len(complete),max_action_abs=max(deltas,default=None),
                max_sequence_abs=max(complete,default=None),top1_observed=any(r['top1'] for r in score['rows'])))


if __name__ == '__main__':
    cfg = doc(pin(sys.argv[1]))
    save(cfg['output'],report(cfg))

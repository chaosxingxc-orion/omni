"""Review a locked natural C8 performance window against its qualified training forward."""
import json
import math
import sys
from pathlib import Path
from statistics import mean


def latency_target(target, pair, candidate_round):
    if (target['metric'] != 'mean_batch_s' or target['reference'] != 'matched_cold_native'
            or not target['require_full_train_eval_forward'] or not 0 <= target['max_relative_excess'] < 1):
        raise ValueError('unsupported target')
    candidate, native = pair['candidate'], pair['native']
    if not (candidate['round_sha256'] == candidate_round and candidate['requests'] == native['requests']
            and candidate['prefix_hits'] == native['prefix_hits']
            and candidate['prefix_queries'] == native['prefix_queries']):
        raise ValueError('different candidates, population or cache state')
    for window in (candidate, native):
        batches = window['batch_s']
        if not (batches and all(math.isfinite(t) and t > 0 for t in batches)
                and len(batches) * target['width'] == window['requests'] == window['mm_queries']
                and window['mm_hits'] == 0 and math.isclose(mean(batches), window['mean_batch_s'])):
            raise ValueError('invalid cold measurement')
    limit = native['mean_batch_s'] * (1 + target['max_relative_excess'])
    return dict(round=candidate_round, native_round=native['round_sha256'],
                candidate_s=candidate['mean_batch_s'], native_s=native['mean_batch_s'],
                max_batch_s=limit, still_to_save_s=max(0., candidate['mean_batch_s'] - limit),
                relative_excess=candidate['mean_batch_s'] / native['mean_batch_s'] - 1,
                within_target=candidate['mean_batch_s'] <= limit)


def review(target, report, proof):
    pair = report['cold_native_comparison']
    candidate = pair['candidate']
    performance = latency_target(target, pair, report['paired']['candidate_round'])
    if not (proof['passed'] and proof['natural_round'] == performance['round']
            and proof['requests'] == proof['passed_requests'] == candidate['requests']
            and proof['actions'] == report['paired']['output_actions']['candidate_tokens']
            and proof['unresolved_top1'] == proof['replay_changed'] == proof['optimizer_updates'] == 0):
        raise ValueError('incomplete precision proof')
    for mode in ('train','eval'):
        m=proof['modes'][mode]
        if not (m['max_token_abs']<=proof['limits']['token']<=0.01 and
                m['max_sequence_abs']<=proof['limits']['sequence']<=0.01 and m['top1_flips']==0):
            raise ValueError('precision limit exceeded')
    return dict(runtime=proof['runtime'], step=proof['step'], requests=proof['requests'],
                actions=proof['actions'], **performance)


if __name__=='__main__':
    print(json.dumps(review(*(json.loads(Path(p).read_bytes()) for p in sys.argv[1:])),indent=2))

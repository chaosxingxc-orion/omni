"""Review a locked natural C8 performance window against its qualified training forward."""
import json
import math
import sys
from pathlib import Path
from statistics import mean


def review(target, report, proof):
    if (target['metric']!='mean_batch_s' or target['reference']!='matched_cold_native' or
            not target['require_full_train_eval_forward'] or not 0<=target['max_relative_excess']<1):
        raise ValueError('unsupported target')
    pair=report['cold_native_comparison']
    a,b=pair['candidate'],pair['native']
    if not (proof['passed'] and proof['natural_round']==a['round_sha256']==report['paired']['candidate_round']
            and proof['requests']==proof['passed_requests']==a['requests']==b['requests']
            and proof['actions']==report['paired']['output_actions']['candidate_tokens']
            and proof['unresolved_top1']==proof['replay_changed']==proof['optimizer_updates']==0
            and len(a['batch_s'])*target['width']==len(b['batch_s'])*target['width']==a['requests']
            and a['prefix_hits']==b['prefix_hits'] and a['prefix_queries']==b['prefix_queries']
            and all(v['mm_queries']==v['requests'] and v['mm_hits']==0 and
                    math.isclose(mean(v['batch_s']),v['mean_batch_s']) for v in (a,b))):
        raise ValueError('unmatched measurements or incomplete precision proof')
    for mode in ('train','eval'):
        m=proof['modes'][mode]
        if not (m['max_token_abs']<=proof['limits']['token']<=0.01 and
                m['max_sequence_abs']<=proof['limits']['sequence']<=0.01 and m['top1_flips']==0):
            raise ValueError('precision limit exceeded')
    limit=b['mean_batch_s']*(1+target['max_relative_excess'])
    return dict(runtime=proof['runtime'],step=proof['step'],round=a['round_sha256'],native_round=b['round_sha256'],
                requests=a['requests'],actions=proof['actions'],candidate_s=a['mean_batch_s'],native_s=b['mean_batch_s'],
                max_batch_s=limit,still_to_save_s=max(0.,a['mean_batch_s']-limit),
                relative_excess=a['mean_batch_s']/b['mean_batch_s']-1,within_target=a['mean_batch_s']<=limit)


if __name__=='__main__':
    print(json.dumps(review(*(json.loads(Path(p).read_bytes()) for p in sys.argv[1:])),indent=2))

"""Compare completion order and tail latency on the locked, cold native panel."""
import sys
from pathlib import Path
from cold_native_window import compare
from io_utils import doc, pin, require, save


def report(spec):
    pair = compare(spec)
    sides = []
    for side, key, lane_key in [('candidate','candidate_config','lane'),('native','cold_native_config','native_lane')]:
        c = doc(pin(spec[key]))
        ref, = doc(pin(Path(c['output'])/'RESULT.json'))['rounds']
        require(ref['sha256'] == pair[side]['round_sha256'], 'round drift')
        rr=doc(ref)
        groups={}
        for row in rr['requests']:
            if row['service'] != spec[lane_key]: continue
            g=groups.setdefault(row['group'],{})
            i=row['replicate']
            require(i not in g,'duplicate member')
            g[i]=(row['finished_from_round_seconds']-row['batch_release_from_round_seconds'],row['usage']['completion_tokens'])
        sides.append(groups)
    result=[]
    require(sides[0].keys()==sides[1].keys(),'unmatched waves')
    for key in sorted(sides[0]):
        a,b=[s[key] for s in sides]
        require(a.keys()==b.keys() and len(a)>1, 'unmatched members')
        wave={}
        for side,rows in [('candidate',a),('native',b)]:
            order=sorted(rows,key=lambda i:rows[i][0])
            last=order[-1]
            wave[side]=dict(last_member=last,last_tokens=rows[last][1],first_to_last_s=rows[last][0]-rows[order[0]][0],
                            last_only_s=rows[last][0]-rows[order[-2]][0],batch_tokens=sum(x[1] for x in rows.values()))
        result.append(dict(batch=key,same_last_member=wave['candidate']['last_member']==wave['native']['last_member'],
                           equal_output_lengths=sum(a[i][1]==b[i][1] for i in a),**wave))
    return dict(rounds={side:pair[side]['round_sha256'] for side in ('candidate','native')},waves=result,
                comparison='Same distinct requests in the same group. Equal lengths do not imply equal token IDs.')


if __name__=='__main__':
    save(sys.argv[2],report(doc(pin(sys.argv[1]))))

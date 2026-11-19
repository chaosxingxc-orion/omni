"""Read a closed resident parity receipt, identify failing action rows, no model calls."""
import argparse,hashlib,json
from pathlib import Path

def summarize(root):
    def read(ref):
        raw=Path(ref['path']).read_bytes()
        if len(raw)!=ref['bytes'] or hashlib.sha256(raw).hexdigest()!=ref['sha256']:
            raise ValueError('changed parity receipt')
        return json.loads(raw)
    marker=json.loads((root/'CLOSED.json').read_bytes())
    if marker['immutable'] is not True or Path(marker['return_ref']['path'])!=root/'RETURN.json':
        raise ValueError('not a closed command return')
    v=read(marker['return_ref']);raw=read(v['raw_ref'])
    rows=[]
    for a,b,c in zip(raw['records'],v['results'],v['joint_output']['results'],strict=True):
        assert a['node']==b['node']==c['node']
        d=b['P']['delta'];positions=[i for i,x in enumerate(d) if x>c['pairs'][0]['numeric']['token_limit']]
        peak=max(range(len(d)),key=d.__getitem__)
        rows.append(dict(node=a['node'],response_tokens=len(d),repeat_max=b['R']['max_abs'],
            causal_max=b['C'].get('max_abs'),joint_token_limit=c['pairs'][0]['numeric']['token_limit'],
            positions_above_joint_limit=positions,peak_index=peak,peak_logprob_delta=d[peak],
            hf_logprob=a['a1'][peak],rollout_logprob=a['reference'][peak],
            argmax_flips=c['pairs'][0]['flips'],vocabulary=c['pairs'][0]['vocabulary'],
            readback_max_abs=c['pairs'][0]['readback_max_abs'],
            prepared_ref=a['prepared_ref'],behavior_ref=a['behavior_ref']))
    return dict(schema='closed-response-parity-localization-v1',passed=v['passed'],
        updates=v['optimizer_steps'],usable_for_update=v['usable_for_update'],records=rows,
        raw_ref=v['raw_ref'],return_ref=marker['return_ref'],first_internal_divergence_proven=False)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    print(json.dumps(summarize(p.parse_args().root),indent=2))

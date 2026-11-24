"""Seal immutable identities of a previously qualified deployment, without replay."""
import argparse,hashlib,json,os
from pathlib import Path

def seal(qualified, deployment, build, output):
    def pin(p):
        raw=p.read_bytes()
        return dict(path=str(p.resolve()),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
    def verified(ref):
        actual=pin(Path(ref['path']))
        if (actual['sha256'],actual['bytes'])!=(ref['sha256'],ref['bytes']):
            raise ValueError('artifact drift: '+ref['path'])
        return actual
    def read(p):return json.loads(p.read_bytes())
    q=read(qualified);proof=read(Path(verified(q['proof'])['path']))
    if (not q['verdict']['passed'] or proof['checkpoint_ref']!=q['checkpoint_ref']
            or q['checkpoint_ref']['sha256']!=q['verdict']['checkpoint_sha256']):
        raise ValueError('missing qualified numerical result')
    cfg=read(deployment); assembled=read(build)
    refs={k:verified(cfg[k]) for k in ('natural_runner_ref','score_launcher_ref','natural_template_ref')}
    refs.update({k:verified(proof[k]) for k in ('reviewer_ref','profile_ref','precision_ref','paired_ref','activation_ref','cache_ref')})
    refs.update(assembly_ref=verified(assembled['assembly_ref']), build=pin(build), qualification=pin(qualified),
                observed=verified(q['proof']), deployment=pin(deployment))
    record=dict(schema='omni-precision-baseline-v1',purpose='precision_alignment',assets=refs,
        accepted_checkpoint=q['checkpoint_ref'],accepted_verdict=q['verdict'],performance_claim=False)
    from storage import commit
    if output.exists():
        if read(output)!=record:raise ValueError('lock already identifies different baseline')
    else:commit(output,record)
    return pin(output)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('qualified','deployment','build','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();print(json.dumps(seal(a.qualified,a.deployment,a.build,a.output)))

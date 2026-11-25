"""Replay the four CPU publication reads for a closed service switch; no engine RPC."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch


def measure(bundle, marker, reuse=False):
    impl = bundle / 'parents/jev-system-rl/implementation'
    sys.path[:0] = [str(impl), str(bundle/'parents/policy-iteration-successor'),
                    str(bundle/'parents/native-recompute-integration/state-repair/acl-runtime-successor')]
    import torch
    import publication
    import policy_identity
    from service_bootstrap import local_successor
    from service_bootstrap import doc, pin
    change = doc(pin(marker))
    refs = {role: doc(ref)['publication_ref'] for role,ref in change.items()}
    rows, calls = [], []
    cached = {}
    def from_content(path, *, geometry):
        # Private to this CPU replay. Each read uses an immutable bytes snapshot for both hash and parse.
        from safetensors.torch import load
        data = Path(path).read_bytes()
        key = (hashlib.sha256(data).hexdigest(), tuple(sorted(geometry.items())))
        if key not in cached:
            cached[key] = policy_identity.describe(load(data), **geometry)
        return cached[key]
    def identity(path, **kwargs):
        start = time.monotonic()
        result = (from_content if reuse else policy_identity.artifact)(path, **kwargs)
        calls.append(dict(seconds=time.monotonic()-start, identity=result['identity_sha256']))
        return result
    validate = publication.validate_publication
    def observed(ref, **kwargs):
        start = time.monotonic()
        count = len(calls)
        result = validate(ref, **kwargs)
        if len(calls) != count+1: raise ValueError('unexpected artifact call count')
        role, = [r for r,v in refs.items() if v == ref]
        rows.append(dict(role=role, step=result['step'], seconds=time.monotonic()-start,
                         artifact=calls[-1]))
        return result
    with patch.object(publication,'artifact',identity), patch.object(publication,'validate_publication',observed):
        if local_successor(marker.parent,marker,change['previous']) != change['successor']:
            raise ValueError('successor mismatch')
        # Only the publication reads of replace_checkpoint and activate, not an engine switch.
        for _ in range(2): observed(refs['successor'])
    if [r['role'] for r in rows] != ['successor','previous','successor','successor']:
        raise ValueError('unexpected publication chain')
    if reuse and len(cached) != 2: raise ValueError('unexpected artifact identities')
    if reuse:
        old, new = [v for v in cached.values()][1], [v for v in cached.values()][0]
        expected=doc(refs['successor'])['resident_identity']['publication']
        publication.check_export_identity(new,expected,1e-8)
        try: publication.check_export_identity(old,expected,1e-8)
        except ValueError: pass
        else: raise ValueError('stale identity accepted')
    return dict(reuse=reuse, stale_identity_rejected=True if reuse else None, engine_switch=False, threads=torch.get_num_threads(), marker=pin(marker),
                sources={k:hashlib.sha256((impl/(k+'.py')).read_bytes()).hexdigest()
                         for k in ('publication','policy_identity','service_bootstrap','service_overlay')},
                rows=rows, repeated_new_artifact_seconds=sum(r['artifact']['seconds'] for r in rows[2:]))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('bundle','marker'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--reuse',action='store_true')
    print(json.dumps(measure(**vars(p.parse_args()))))

"""Revalidate a bounded P publication and its live TP4 activation receipts."""
import hashlib
import json
import math
from pathlib import Path
from dependencies import ROOT, load, require
from run_package import pin, read
from contextlib import contextmanager
import sys


@contextmanager
def inference_protocol():
    # The training adapter can already have a different 'protocol' loaded.
    # Keep its imports intact and resolve the readback from the sealed peer.
    parent=ROOT/'policy-iteration-successor'
    names=('protocol','io_utils')
    previous={name:sys.modules.pop(name,None) for name in names}
    sys.path.insert(0,str(parent))
    try:
        yield load(parent/'protocol.py','protocol')
    finally:
        sys.path.remove(str(parent))
        for name,value in previous.items():
            sys.modules.pop(name,None)
            if value is not None:sys.modules[name]=value


def replay(publication, ref):
    window=read(ref)
    if window.get('schema')=='individually-verified-tp4-lanes-v1':
        rows=window['lane_proofs']
        require(set(rows)=={'a','b'} and window['checkpoint_ref']==publication['checkpoint_ref'],
                'incomplete TP4 parity roster or checkpoint')
        states={}
        for lane,item in rows.items():
            proof=read(item)
            require(proof['schema']=='omni-p-verified-replayed-tp4-v1' and proof['lane']==lane
                    and proof['profile_ref']==window['profile_ref'],'crossed TP4 parity or profile')
            states.update(replay_same_action(publication,proof)['states'])
        return dict(profile=read(window['profile_ref']),states=states)
    if window.get('schema')=='omni-p-verified-replayed-tp4-v1':
        return replay_same_action(publication, window)
    from parity_window import review_active
    require(window['checkpoint_ref']==publication['checkpoint_ref'],
            'bounded parity checkpoint differs from publication')
    profile=read(window.get('profile_copy_ref',window['profile_ref']))
    source=Path(window['evidence'])
    with inference_protocol():
        report=review_active(window)
    require(report['passed'] and report['step']==publication['step'],'P parity failed')
    states={}
    for lane in window['lanes']:
        act=read(source/lane['name']/'ACTIVATION.json')
        precision=read(source/lane['name']/'PRECISION.json')
        states[lane['name']]=dict(activation=act,precision=precision)
    return dict(profile=profile,states=states)


def replay_same_action(publication, window):
    """Qualify the exact P service and committed checkpoint scored by FSDP8."""
    require(window['checkpoint_ref']==publication['checkpoint_ref']
            and window['lane'] in ('a','b'), 'same-action parity checkpoint or lane differs')
    act=read(window['activation_ref'])
    precision=read(window['precision_ref'])
    cache=read(window['cache_ref'])
    profile=read(window['profile_ref'])
    candidate_ref=window.get('activated_publication_ref',act['publication_ref'])
    candidate=read(candidate_ref)
    def content(ref):
        return ref['sha256'],ref['bytes']
    require(content(candidate_ref)==content(act['publication_ref'])
            and content(candidate['checkpoint_ref'])==content(publication['checkpoint_ref'])
            and candidate['artifact_version']==act['policy_version']==publication['artifact_version']
            and list(map(content,candidate['assets']))==list(map(content,publication['assets']))
            and act['weights_ref'] in candidate['assets']
            and (content(act['weights_ref']) in list(map(content,publication['assets'])))
            and all(act['profile_ref'][key]==window['profile_ref'][key] for key in ('sha256','bytes')),
            'same-action parity does not cover this deployed policy or profile')
    require(cache['kv_cache'] is True and cache['prefix_cache'] is True
            and cache['reset_succeeded'] is True and cache['policy_ref']==act['policy_ref']
            and cache['precision_ref']==precision['specification_ref'],
            'same-action parity requires a cache reset with this policy and precision')
    devices=act['service_identity']['physical_devices']
    require(devices==([0,1,2,3] if window['lane']=='a' else [4,5,6,7])
            and act['service_identity']['source_boot']==act['source_boot']
            and sorted(row['rank'] for row in precision['workers'])==list(range(4)),
            'same-action parity has a different physical service')
    with inference_protocol() as protocol:
        protocol.readback(act['workers'],act['adapter_id'],act['weights_ref'])
    paired=read(window['paired_ref'])
    require(window['response_ref'] in [r for row in paired['rows'] for r in row['source_refs']]
            and read(window['response_ref'])['response_sha256'] in {row['response'] for row in paired['rows']},
            'endpoint response was not included in the same-action observations')
    reviewer=pin(window['reviewer_ref'])
    require(Path(reviewer['path']).name=='updated_http_parity.py','wrong same-action comparator')
    from dependencies import local_path
    review=load(local_path(reviewer['path']),'p_bounded_verified_comparator')
    checkpoint=pin(window['checkpoint_ref'])
    ranks=window['rank_refs']
    verdict=review.review_replayed_top1(paired,act,
        dict(checkpoint,path=str(local_path(checkpoint['path']))),
        read(window['exit_ref']),[read(ref) for ref in ranks],limits=window['limits'])
    require(verdict['passed'] and verdict['step']==publication['step'],'P same-action parity failed')
    return dict(profile=profile,states={window['lane']:dict(activation=act,precision=precision)},
                verdict=verdict)


def endpoint(verified, witness, version):
    """Bind a restarted owner only when its real precision and policy match."""
    w=read(witness)
    old=verified['states'][w['lane']]
    act=read(w['activation_ref']);precision=read(w['precision_ref']);cache=read(w['cache_ref'])
    before=old['activation']
    require(all(act[k]==before[k] for k in ('policy_ref','publication_ref','weights_ref',
            'policy_version','endpoint','fixture','base','source_boot')),
            'new endpoint policy or boot differs from parity')
    def content(ref):return ref['sha256'],ref['bytes']
    require(content(act['profile_ref'])==content(before['profile_ref']),
            'new endpoint sampling profile bytes differ')
    require(act['policy_version']==version and act['fixture'] is False and act['base'] is False
            and all(act['service_identity'][k]==before['service_identity'][k]
                    for k in ('image','physical_devices','source_boot')),
            'new endpoint model/device identity differs')
    prior_precision=old['precision']
    refs={'specification_ref','rows_ref'}
    require({k:v for k,v in precision.items() if k not in refs}==
            {k:v for k,v in prior_precision.items() if k not in refs} and
            all(content(precision[k])==content(prior_precision[k]) for k in refs)
            and cache['kv_cache'] is True and cache['prefix_cache'] is True
            and cache['reset_succeeded'] is True and cache['policy_ref']==act['policy_ref']
            and cache['precision_ref']==precision['specification_ref'],'precision/cache policy differs')
    with inference_protocol() as protocol:
        protocol.readback(act['workers'],act['adapter_id'],act['weights_ref'])
    require(all(x['compared_original_keys']==384 for x in act['workers']),'incomplete adapter')
    response=read(w['response_ref'])
    raw=response['raw'].encode()
    require(response['status']==200 and hashlib.sha256(raw).hexdigest()==response['response_sha256'],
            'endpoint response custody differs')
    body=json.loads(raw);choice,=body['choices'];tokens=choice['token_ids']
    items=choice['logprobs']['content']
    require(body['model']=='qwen3-omni-30b-a3b-instruct' and body['prompt_token_ids']
            and tokens and len(tokens)==len(items)
            and all(item['token']==f'token_id:{t}' and math.isfinite(item['logprob'])
                    and item['logprob']!= -9999 and item['logprob']<=0
                    for item,t in zip(items,tokens,strict=True)),
            'live endpoint lacks exact real token/logprob protocol')
    return act


def validate_routes(publication, witness, profile, *, sampled=False):
    if 'continuous_activation_ref' in publication:
        from continuous_policy import verify
        ev=read(publication['continuous_activation_ref']);w=read(witness)
        require(sampled and w['schema']=='continuous-training-endpoints-v1' and w['fixture'] is False and w['passed'] is True
                and w['model']=='qwen3-omni-30b-a3b-instruct' and w['thinking'] is False
                and w['policy_version']==publication['artifact_version'] and w['lanes']==ev['lanes']
                and w['profile_ref']==ev['profile_ref'] and read(ev['profile_ref'])==profile,
                'continuous witness differs from activated training publication')
        return verify(publication,publication['continuous_activation_ref'])
    verified=replay(publication,publication['bounded_tp4_ref'])
    require(profile==verified['profile'],'sampling profile differs from bounded parity')
    w=read(witness)
    require(w['schema']=='omni-bounded-tp4-endpoints-v1'
            and w['policy_version']==publication['artifact_version']
            and w['fixture'] is False and w['passed'] is True
            and w['model']=='qwen3-omni-30b-a3b-instruct' and w['thinking'] is False,
            'invalid bounded endpoint set')
    rows=w['lanes']
    if sampled:
        require(len(rows) in (1,2) and set(rows)<=set(verified['states']),
                'sampled TRAIN requires one or two individually verified lanes')
    else:
        require(len(rows)==2 and set(rows)==set(verified['states']),'both TP4 lanes required')
    act={name:endpoint(verified,ref,publication['artifact_version']) for name,ref in rows.items()}
    for name,ref in rows.items():
        require(read(ref)['lane']==name,'crossed TP4 receipt')
    return act

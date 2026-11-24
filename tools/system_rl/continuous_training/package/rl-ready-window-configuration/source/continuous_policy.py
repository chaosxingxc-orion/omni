"""Training policy admission: immutable arithmetic lock + live adapter identity.

Reuses original lease, relocation, cache and TP readback. Never runs a numerical
replay or synthesizes business responses. Paths in receipts can be relocated;
their original byte digests are what identify the model and activation.
"""
from pathlib import Path
import re
from dependencies import require
from run_package import pin,read,write


def same(a,b):
    return a['sha256']==b['sha256'] and a['bytes']==b['bytes']


def verify(pub, evidence):
    from bounded_tp4 import inference_protocol
    record=read(evidence)
    require(record['schema']=='continuous-training-activation-v1'
            and same(record['checkpoint_ref'],pub['checkpoint_ref']),'activation belongs to different checkpoint')
    baseline=read(record['precision_lock_ref'])
    require(baseline['purpose']=='precision_alignment' and baseline['accepted_verdict']['passed'] is True,
            'missing numerical baseline')
    anchor=read(baseline['assets']['activation_ref'])
    prec=read(baseline['assets']['precision_ref'])
    prior=read(baseline['accepted_checkpoint'])
    contract=pub['scientific_contract']
    require(pub['resident_identity']['base']==prior['resident_identity']['base'] and
            all(contract[k]==prior['contract'][k] for k in ('lineage','devices','lora_arithmetic','image','training')),
            'trained model or arithmetic differs from qualified contract')
    require(same(record['profile_ref'],baseline['assets']['profile_ref']),'different sampling profile')
    lanes={}
    for key,ref in record['lanes'].items():
        lane=read(ref)
        require(key==lane['lane'] and re.fullmatch(r'[0-9a-f]{64}',lane['adapter_sha256'])
                and lane['adapter_sha256']==pub['assets'][0]['sha256'],'incorrect active lane')
        act=read(lane['activation_ref']);actual_prec=read(lane['precision_ref']);cache=read(lane['cache_ref'])
        service=act['service_identity']
        require(not act['fixture'] and not act['base'] and act['policy_version']==pub['artifact_version']
                and same(act['weights_ref'],pub['assets'][0]) and same(act['profile_ref'],record['profile_ref'])
                and service['image']==anchor['service_identity']['image'] and len(service['physical_devices'])==4
                and len(set(service['physical_devices']))==4 and service['pid']>0 and service['start_ticks']>0
                and service['source_boot']==act['source_boot'],'active policy or process differs')
        require(actual_prec['actual_device_execution'] is True and actual_prec['workers']==prec['workers']
                and same(actual_prec['rows_ref'],prec['rows_ref']) and same(actual_prec['specification_ref'],prec['specification_ref']),
                'TP runtime arithmetic differs from precision baseline')
        require(cache['kv_cache'] is True and cache['prefix_cache'] is True and cache['reset_succeeded'] is True
                and cache['policy_ref']==act['policy_ref'] and same(cache['precision_ref'],actual_prec['specification_ref']),
                'cache under wrong policy/precision')
        with inference_protocol() as protocol: protocol.readback(act['workers'],act['adapter_id'],act['weights_ref'])
        require(all(w['compared_original_keys']==384 for w in act['workers']),'incomplete adapter')
        if lane['activation_depth']:
            sw=read(lane['switch_ref']);before=read(lane['previous_activation_ref'])
            require(sw['old_removed_from_engine'] is True and sw['cache_reset'] is True and
                    before['service_identity']==service and before['source_boot']==act['source_boot'] and
                    sw['previous_policy_ref']==before['policy_ref'] and sw['old_id']==before['adapter_id']
                    and sw['new_id']==act['adapter_id']!=before['adapter_id'] and same(sw['activation_ref'],lane['activation_ref'])
                    and cache['previous_activation_ref']['sha256']==lane['previous_activation_ref']['sha256'],
                    'switch did not preserve service or replace predecessor')
        lanes[key]=act
    require(bool(lanes),'empty training activation')
    return lanes


def control_post(url,expected,timeout):
    import json
    from urllib.parse import urlsplit
    from urllib.request import ProxyHandler,build_opener,Request
    uri=urlsplit(url)
    require(uri.scheme=='http' and uri.hostname=='127.0.0.1' and uri.path in ('','/'),'owner tunnel required')
    req=Request(url.rstrip('/')+'/_omni/activation',data=b'',method='POST')
    with build_opener(ProxyHandler({})).open(req,timeout=timeout) as response:
        require(response.status==200,'control endpoint rejected activation')
        ack=json.load(response)
    require(ack['version']==expected and ack['adapter_id']>0,'service did not load requested version')
    return ack


def activate(config,state,root):
    """Drive only the owner marker and existing service; no calls to generate."""
    import os
    from contextlib import ExitStack
    from formal_rollout import settings,lane_providers,lease_plan,acquire,release,provider_modules,close_tunnel
    from formal_parity import previous_capture,relocate_candidate,publish_inbox,receipt_bytes
    from publication import validate_publication
    spec,primary=settings(config);providers=lane_providers(config,spec,primary)
    policy=read(state['policy_ref']);pub=validate_publication(policy['publication_ref'],verify_parity=False)
    opts=spec['continuous_training']
    require(not os.environ.get('OMNI_TIMESHARED_ROLLOUT_REF') and
            opts['protocol']=='apply-local-marker-v1' and spec['ssh']['target']==primary['host'],
            'continuous rollout requires dedicated resident provider')
    require(pub['step']==state['step'] and pub['checkpoint_ref']==state['resume_ref'],'wrong actor publication')
    first=state['resume_ref']==config['initial_resume_ref']
    capture=None if first else previous_capture(state)
    before={k:read(p['initial_lane_ref'] if first else read(capture['endpoint_witness_ref'])['lanes'][k])
            for k,p in providers.items()}
    active={k:read(v['activation_ref']) for k,v in before.items()}
    root.mkdir(parents=True,exist_ok=True); fence=root/'lease';closed=root/'ACTIVATIONS-CLOSED.json'
    remote,ready,service=lease_plan(config,primary,fence,config['run_id']+'-publish-'+str(state['step']),
        dict(checkpoint_ref=state['resume_ref']))
    if not closed.exists():
        acquire(fence,remote,primary,ready,service)
        if not first:
            inbox=Path(service['policy_overlay']['update_mount']);target=inbox/f"{config['run_id']}-step-{state['step']}"
            require(all(read(p['service_config'])['policy_overlay']['update_mount']==str(inbox) for p in providers.values())
                    and all(a['policy_ref']==active[primary['lane']]['policy_ref'] for a in active.values()),
                    'provider inbox or active policy differs')
            checkout=root/'adapter-input';successor=relocate_candidate(state['policy_ref'],checkout,target)
            prior=active[primary['lane']]['policy_ref']
            if not (root/'MARKER.json').exists():
                write(root/'MARKER.json',publish_inbox(remote,service,checkout,target,prior,successor,state['step']-1))
        lanes={}
        with ExitStack() as stack:
            for name,p in providers.items():
                ready=read(p['service_ready']);work=root/name;work.mkdir(exist_ok=True)
                with provider_modules() as (lease,module,_,_):
                    tunnel=module.tunnel(dict(provider=p,remote_provider_guard=remote),work);stack.callback(close_tunnel,tunnel)
                    ack=control_post(f"http://127.0.0.1:{p['local_port']}",pub['artifact_version'],spec['drain_timeout_seconds'])
                    advertised=Path(ack['activation_ref']['path'])
                    digest=pub['assets'][0]['sha256'];base=Path(p['live_receipt_paths']['activation_ref']).parent
                    expected=base if first else base/'updates'/digest
                    require(advertised==expected/'ACTIVATION.json' and ack['service_identity']==active[name]['service_identity'],
                            'activation reply from wrong path/process')
                    lane=dict(lane=name,activation_depth=0 if first else before[name].get('activation_depth',0)+1,
                              adapter_sha256=digest)
                    files={'activation_ref':'ACTIVATION.json','precision_ref':'PRECISION.json','cache_ref':'CACHE-POLICY.json'}
                    if not first:
                        files['switch_ref']='SWITCHED.json';lane['previous_activation_ref']=before[name]['activation_ref']
                    for role,file in files.items():
                        raw=lease.command(remote,['docker','exec',ready['container_id'],'cat',str(expected/file)],
                                          capture_output=True).stdout
                        lane[role]=receipt_bytes(work/file,raw)
                    actual=read(lane['activation_ref'])
                    require(same(lane['activation_ref'],ack['activation_ref']) and actual['adapter_id']==ack['adapter_id']
                            and actual['policy_ref']==(active[name]['policy_ref'] if first else successor)
                            and actual['service_identity']==ack['service_identity'],'remote activation receipt differs')
                lanes[name]=write(work/'LANE.json',lane)
        evidence=write(root/'ACTIVATIONS.json',dict(schema='continuous-training-activation-v1',
            precision_lock_ref=opts['precision_lock_ref'],checkpoint_ref=pub['checkpoint_ref'],
            profile_ref=read(opts['precision_lock_ref'])['assets']['profile_ref'],lanes=lanes))
        verify(pub,evidence)
        write(closed,dict(terminal_verified=True,unknown_requests=0,activation_ref=evidence))
    terminal=read(closed)
    receipt=root/'RELEASED.json'
    if not receipt.exists():
        write(receipt,release(fence,remote,service,terminal,spec['drain_timeout_seconds']))
    released=read(receipt)
    owner=read(remote['config'])['owner']
    require(released['status']=='released' and released['owner']==owner and
            released['terminal']==dict(terminal,owner=owner),'wrong closed activation lease')
    evidence=terminal['activation_ref']
    require(read(evidence)['precision_lock_ref']==opts['precision_lock_ref'],'precision release changed')
    verify(pub,evidence)
    clean={k:v for k,v in pub.items() if k not in ('bounded_tp4_ref','updated_rcp_ref','runtime_ref')}
    new=write(root/'POLICY-PUBLISHED.json',dict(clean,status='published_for_fresh_sampling',continuous_activation_ref=evidence))
    return write(root/'POLICY.json',dict(policy,publication_ref=new))

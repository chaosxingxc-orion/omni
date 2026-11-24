"""Policy hooks inside DATA's already acquired guard; no second owner/trainer.

The supervisor opens a session before collection. Its terminal watchdog closes
that exact session before releasing the DATA agent. Remote runtime uses the
original API overlay, endpoint journal, and original CID/config restore body.
"""
import json,sys
from pathlib import Path
from dependencies import HERE,require
from run_package import read,pin,write


def validate(ref):
    session=read(ref)
    require(session['schema']=='provider-policy-session-v1','policy session schema')
    policy=read(session['policy_ref']);require(policy['kind'] in ('base','checkpoint'),'policy kind')
    require(session['kind']==('baseline' if policy['kind']=='base' else 'candidate'),'policy/session mismatch')
    if policy['kind']=='base':
        # Reuse closed base inventory, never reread all30B Ceph weight bytes.
        seal=read(policy['asset_manifest_ref']);require(seal['status']=='CLOSED_IMMUTABLE' and seal['assets']==policy['assets'],'base immutable seal differs')
    else:
        for asset in policy['assets']:pin(asset)
    if policy['kind']=='checkpoint':
        from publication import validate_publication
        publication=validate_publication(policy['publication_ref'],verify_checkpoint_shards=False)
        weight=next(a for a in policy['assets'] if Path(a['path']).name=='adapter_model.safetensors')
        require(session['adapter_digest']==weight['sha256'] and publication['artifact_version']==policy['artifact_version'],'actual adapter digest differs')
    provider = session.get('resident_provider')
    if provider:
        require(provider['schema']=='existing-tp4-provider-v2'
                and session['host']==provider['host']
                and session['devices']==provider['devices']
                and session['devices'] in ([0,1,2,3],[4,5,6,7]), 'resident policy scope differs')
        resident_receipts(session)
    else:
        require(session['devices']==[0,1,2,3] and session['host']=='npu','1750-3 policy scope')
    return session


def invoke(request,root,operation):
    """Called by frozen DATA supervisor/watchdog, under the SAME agent lease."""
    ref=request.get('policy_session_ref')
    if ref is None:return None
    session=validate(ref)
    if session.get('resident_provider'):
        return observe_resident(request, root, session, operation)
    from lease_runtime import command
    remote=read(Path(root)/'REMOTE-LEASE.json')['remote']
    # Transfer exact small control refs as one closed payload through the
    # existing DATA command. Client root and remote root need not coincide.
    # Model/checkpoint/source refs already belong to the fixed transport stage.
    payload=dict(request=read(Path(root)/'REQUEST.json'),lease=read(Path(root)/'REMOTE-LEASE.json'),
        session_ref=pin(ref),remote_root=str(Path(remote['hot_root'])/'policy-control'))
    import base64
    files={}
    lease=payload['lease']
    for value in (payload['session_ref'],lease['remote']['config'],lease['remote']['agent']):
        ref=pin(value);files[ref['path']]=dict(ref=ref,data=base64.b64encode(__import__('dependencies').local_path(ref['path']).read_bytes()).decode())
    payload['files']=files
    module=Path(session['runtime_root'])/'provider_session.py'
    code="import sys;sys.path.insert(0,"+repr(str(module.parent))+");from provider_session import dispatch;dispatch("+repr(operation)+")"
    argv=['docker','exec','-i',session['control_container'],'python3','-B','-c',code]
    import time,subprocess
    # DATA transport has a 30s timeout. The actual load runs detached under
    # one durable remote intent; disconnects observe it without resubmission.
    deadline=__import__('datetime').datetime.fromisoformat(request['authority']['expires_at']).timestamp()
    while True:
        require(time.time()<deadline,'policy hook deadline; original remote intent retained')
        try:
            response=command(remote,argv,input=json.dumps(payload),capture_output=True,text=True)
            returned=json.loads(response.stdout.splitlines()[-1])
        except (OSError,subprocess.TimeoutExpired,subprocess.CalledProcessError):
            time.sleep(5);continue
        require(returned['status']!='failed','policy hook failed: '+str(returned.get('error')))
        if returned['status']=='complete':break
        require(returned['status']=='pending','unknown policy hook response')
        time.sleep(5)
    # Transfer all small new activation/witness controls back in the same
    # response, preserving logical refs and hash-bound JSON bytes.
    import base64,hashlib
    from dependencies import register_locations
    mapping={};directory=Path(root)/'provider-return'/operation;directory.mkdir(parents=True,exist_ok=True)
    for index,(logical,item) in enumerate(returned['files'].items()):
        content=base64.b64decode(item['data']);require(hashlib.sha256(content).hexdigest()==item['ref']['sha256'],'policy return custody drift')
        target=directory/str(index)
        if target.exists():require(target.read_bytes()==content,'policy return changed')
        else:
            with target.open('xb') as stream:stream.write(content);stream.flush();__import__('os').fsync(stream.fileno())
        mapping[logical]=str(target)
    register_locations(mapping)
    write(directory/'LOCATION-MAP.json',mapping)
    return returned['result']



def resident_receipts(session):
    """Resolve closed identity evidence for the exact DATA provider."""
    provider = session['resident_provider']
    ready = read(provider['service_ready'])
    service = read(provider['service_config'])
    endpoint = read(session['endpoint_witness_ref'])
    if endpoint.get('schema') == 'resident-tp4-startup-observation-v1':
        require(session['endpoint_witness_ref']==provider['service_ready']
                and endpoint['ready'] is True
                and endpoint['service_config_ref']['sha256']==provider['service_config']['sha256']
                and endpoint['service_config_ref']['bytes']==provider['service_config']['bytes']
                and endpoint['source_boot']==service['host_boot_id']
                and endpoint['devices']==session['devices']
                and endpoint['adapter_sha256']==session['adapter_digest']
                and endpoint['protocol_prompt_tokens']>0 and endpoint['protocol_action_tokens']>0,
                'startup observation does not identify the configured service and adapter')
        names = dict(activation_ref='ACTIVATION.json', precision_ref='PRECISION.json',
                     cache_ref='CACHE-POLICY.json')
        receipts = {key:endpoint['local_receipts'][name] for key,name in names.items()}
        require(all(pin(receipts[key])==receipts[key]
                    and receipts[key]['sha256']==endpoint['remote_receipt_sha256'][name]
                    for key,name in names.items()), 'original provider receipt bytes differ')
        endpoint = receipts
    if 'lanes' in endpoint:
        endpoint = read(endpoint['lanes'][provider['lane']])
    return ready, service, endpoint


def observe_resident(request, root, session, operation):
    """Verify the actual provider while the original DATA agent holds its guard."""
    from lease_runtime import command, probe
    provider = session['resident_provider']
    require(request['provider']==provider, 'request provider differs from the bound resident')
    lease = read(Path(root)/'REMOTE-LEASE.json')
    remote = lease['remote']
    require(remote['host']==provider['host'] and remote['path']==provider['guard_path'],
            'resident policy is outside the active DATA lease')
    ready, service, endpoint = resident_receipts(session)
    held = probe(remote)
    fields = ('pid','boot','start_ticks','owner','guard_inode','guard_device')
    require(all(held[key]==lease['state'][key] for key in fields)
            and held['boot']==service['host_boot_id']
            and held['guard_inode']==ready.get('guard_inode',held['guard_inode']),
            'DATA lease is not holding the verified provider guard')
    refs = provider['live_receipt_paths']
    require(set(refs)=={'activation_ref','precision_ref','cache_ref'}
            and all(Path(p).is_absolute() and '..' not in Path(p).parts for p in refs.values()),
            'invalid resident receipt paths')
    depth = endpoint.get('activation_depth', 0)
    require(type(depth) is int and 0 <= depth <= read(session['policy_ref']).get('step', 0),
            'invalid resident activation depth')
    keys = sorted(refs)
    paths = [str(Path(refs[key]).parent.joinpath(*(['successor'] * depth), Path(refs[key]).name))
             for key in keys]
    if endpoint.get('adapter_sha256') and depth:
        digest=endpoint['adapter_sha256']
        require(len(digest)==64 and all(c in '0123456789abcdef' for c in digest)
                and digest==session['adapter_digest'],'unexpected compact adapter location')
        paths=[str(Path(refs[k]).parent/'updates'/digest/Path(refs[k]).name) for k in keys]
    received = command(remote,['docker','exec',ready['container_id'],'sha256sum','--',*paths],
                       capture_output=True,text=True)
    digests = [line.split()[0] for line in received.stdout.splitlines()]
    require(digests == [pin(endpoint[key])['sha256'] for key in keys],
            'active resident policy, precision or cache evidence changed')
    activation = read(endpoint['activation_ref'])
    cache = read(endpoint['cache_ref'])
    precision = read(endpoint['precision_ref'])
    policy = read(session['policy_ref'])
    require(activation['fixture'] is False and activation['policy_version']==policy.get('artifact_version',policy['version'])
            and activation['service_identity']['physical_devices']==session['devices']
            and activation['service_identity']['image']==service['image']
            and activation['source_boot']==held['boot']
            and cache['kv_cache'] is True and cache['prefix_cache'] is True and cache['reset_succeeded'] is True
            and cache['policy_ref']==activation['policy_ref'] and cache['precision_ref']==precision['specification_ref'],
            'resident policy or cache identity differs')
    # Old per-depth receipts remain readable after a hot switch. Verify that
    # the current depth has not itself started switching to a successor.
    import subprocess
    try:
        command(remote, ['docker','exec',ready['container_id'],'test','!',
                '-e',str(Path(paths[0]).parent/'SWITCH-INTENT.json')],
                capture_output=True,text=True)
    except subprocess.CalledProcessError as error:
        raise ValueError('resident policy already began a subsequent switch') from error
    if session['kind']=='candidate':
        require(activation['weights_ref']['sha256']==session['adapter_digest']
                and {row['rank'] for row in activation['workers']}==set(range(4))
                and len(activation['workers'])==4
                and all(row['compared_original_keys']==384 for row in activation['workers']),
                'resident adapter readback does not match the policy')
    result = dict(session_ref=pin(request['policy_session_ref']), activation_ref=endpoint['activation_ref'])
    if operation=='close':
        active=Path(root)/'PROVIDER-ACTIVE.json'
        if active.exists():
            require(read(active)['activation_ref']==endpoint['activation_ref'], 'session policy changed before close')
        return dict(result, same_resident=True)
    require(operation=='open', 'unknown resident session operation')
    return dict(result, endpoint_witness_ref=session['endpoint_witness_ref'])


def open_session(request,root):
    result=invoke(request,root,'open')
    if result is not None:bind_active(request,root,result)
    return result


def bind_active(request,root,result):
    # New derived business/system pins before any collector start. DATA's
    # original input request remains immutable and is retained in activation.
    root=Path(root)
    from base import checked,save,pin as data_pin,read as data_read
    if (root/'PROVIDER-ACTIVE.json').exists():
        require(data_read(root/'PROVIDER-ACTIVE.json')==result,'active policy session drift');return
    from dispatch_binding import validate as validate_binding
    validate_binding(root/'activation/CONFIG.json')
    # Binding must finish before the original launch reserves any request.
    ledger=Path(data_read(root/'activation/CONFIG.json')['output_root'])/'reservations.sqlite'
    require(not ledger.exists(),'policy binding after Registry initialization requires original-window reconciliation')
    from activate import replace_refs
    cfg=data_read(root/'activation/CONFIG.json')
    if cfg.get('raw_preparation') is None:
        # Official evaluator has a distinct original builder. Derive only its
        # business system, retaining official datasets and full denominators.
        heldout=checked(cfg['heldout_config']);inputs=checked(heldout['inputs'])
        heldout['policy_scientific_system_ref']=validate(request['policy_session_ref'])['system']
        heldout['endpoint_policy_witness_ref']=result['endpoint_witness_ref']
        import copy
        sys.path.insert(0,str(__import__('dependencies').ROOT/'policy-iteration-successor'))
        from context_builder import policy_system
        iteration=read(validate(request['policy_session_ref'])['iteration_request_ref'])
        effective=dict(iteration,endpoint_witness_ref=result['endpoint_witness_ref'])
        for split,spec in inputs['splits'].items():
            system=checked(spec['system']);business=data_pin(system['objects']['business']['kwargs']['config'])
            spec['system'],_=policy_system(effective,spec['system'],business,root/'activation/policy-heldout'/split)
        heldout['inputs']=save(root/'activation/HELDOUT-INPUTS.active-policy.json',inputs)
        cfg['heldout_config']=save(root/'activation/HELDOUT.active-policy.json',heldout)
        save(root/'activation/CONFIG.json',cfg)
        marker=data_read(root/'ACTIVATION.json');marker['config']=data_pin(root/'activation/CONFIG.json')
        marker['artifacts']=[data_pin(ref['path']) for ref in marker['artifacts']]
        marker['provider_active']=result;save(root/'ACTIVATION.json',marker)
        save(root/'PROVIDER-ACTIVE.json',result);return
    raw=checked(cfg['raw_preparation'])
    old=raw['business'];business=checked(old)
    binding=dict(policy_ref=validate(request['policy_session_ref'])['policy_ref'],
        endpoint_witness_ref=result['endpoint_witness_ref'],activation_ref=result['activation_ref'])
    business['policy_iteration']=binding
    new=save(root/'activation/BUSINESS.active-policy.json',business)
    system=replace_refs(checked(raw['system']),old,new);system['policy_iteration']=binding
    system['pins']+= [binding['policy_ref'],binding['endpoint_witness_ref'],binding['activation_ref']]
    system_ref=save(root/'activation/SYSTEM.active-policy.json',system)
    raw=replace_refs(raw,old,new);raw['system']=system_ref
    cfg['raw_preparation']=save(cfg['raw_preparation']['path'],raw)
    publisher=data_read(cfg['publisher_config']);publisher.update(business=new['path'],system=system_ref['path'])
    save(cfg['publisher_config'],publisher)
    save(root/'activation/CONFIG.json',cfg)
    marker=data_read(root/'ACTIVATION.json');marker['config']=data_pin(root/'activation/CONFIG.json')
    marker['artifacts']=[data_pin(ref['path']) for ref in marker['artifacts']]
    marker['provider_active']=binding;save(root/'ACTIVATION.json',marker)
    save(root/'PROVIDER-ACTIVE.json',result)

def close_session(request,root):return invoke(request,root,'close')


def activate(session_ref,request_ref,lease_ref,backend):
    """Executable seam: validate bytes BEFORE stop or any business/model request."""
    session=validate(session_ref);request=read(request_ref);root=Path(request['output_root'])/'policy-session'
    marker=root/'ACTIVE.json'
    if marker.exists():
        result=read(marker);require(result['session_ref']==pin(session_ref),'policy session drift')
        backend.check(result);return result
    backend.guard(lease_ref)
    # Persist admission before ANY stop/load. A lost reply cannot invoke the
    # same mutation twice; its independent watchdog retains restore custody.
    require(not (root/'ACTIVATION-INTENT.json').exists(),'unknown policy activation: reconcile original intent, never reload')
    write(root/'ACTIVATION-INTENT.json',dict(session_ref=pin(session_ref),request_ref=pin(request_ref),lease_ref=pin(lease_ref),automatic_retry=False))
    prepared=backend.prepare(session,request_ref,lease_ref,root)
    # prepare arms independent restoration before original service is stopped.
    backend.drain(prepared)
    activation=backend.load(prepared)
    observed=read(activation)
    require(observed['policy_ref']==session['policy_ref'],'loaded another policy')
    if session['kind']=='candidate':
        require(observed['weights_ref']['sha256']==session['adapter_digest'],'loaded wrong adapter bytes')
        require(len(observed['workers'])==4 and all(x['compared_original_keys']==384 for x in observed['workers']),'384-key device readback required')
    witness=backend.prove(prepared,activation)
    result=dict(session_ref=pin(session_ref),request_ref=pin(request_ref),activation_ref=activation,
        endpoint_witness_ref=witness,prepared_ref=prepared,guard_lease_ref=pin(lease_ref))
    return read(write(marker,result))


def restore(session_ref,request_ref,lease_ref,backend):
    session=validate(session_ref);request=read(request_ref);root=Path(request['output_root'])/'policy-session'
    marker=root/'RESTORED.json'
    if marker.exists():return read(marker)
    backend.guard(lease_ref)
    if not (root/'PREPARED.json').exists():
        receipt=backend.untouched(session,request_ref,root)
        return read(write(marker,dict(session_ref=pin(session_ref),restore_ref=receipt,partial_open='before_prepared_no_model_mutation')))
    prepared=pin(root/'PREPARED.json')
    receipt=backend.restore(prepared)
    value=read(receipt);require(value['config_exact'] and value['container_id']==read(prepared)['original']['Id'],'original CID/config restoration required')
    return read(write(marker,dict(session_ref=pin(session_ref),restore_ref=receipt)))


def same_policy(left,right):
    if left is None or right is None:return left==right
    a,b=read(left),read(right)
    return a['kind']==b['kind'] and a.get('artifact_version',a['version'])==b.get('artifact_version',b['version']) and a['assets']==b['assets']

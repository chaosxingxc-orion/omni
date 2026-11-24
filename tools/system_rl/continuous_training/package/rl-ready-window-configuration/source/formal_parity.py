"""Qualify an updated P policy using the existing TP4 and FSDP8 entries."""

import copy
import json
import os
from pathlib import Path
import time

from dependencies import load, local_path, require
from run_package import pin, read, write


def receipt_bytes(path, data):
    """Keep the service's original bytes, including the serialization."""
    from checkpoint_inventory import syncdir
    import tempfile
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == data, 'receipt changed: '+str(path))
    else:
        with tempfile.NamedTemporaryFile(dir=path.parent) as pending:
            pending.write(data)
            pending.flush()
            os.fsync(pending.fileno())
            os.link(pending.name, path)
        syncdir(path.parent)
    return pin(path)


def content(ref):
    return ref['sha256'], ref['bytes']


def relocate_candidate(policy_ref, checkout, destination):
    """Project the immutable commit and its inference assets into a local mount."""
    policy = read(policy_ref)
    from publication import validate_publication
    pub = validate_publication(policy['publication_ref'], verify_parity=False)
    step = pub['step']

    def copy_ref(ref, relative):
        actual = receipt_bytes(checkout/relative, local_path(pin(ref)['path']).read_bytes())
        return dict(actual, path=str(destination/relative))

    def json_ref(relative, value):
        return dict(write(checkout/relative, value), path=str(destination/relative))

    checkpoint = copy_ref(pub['checkpoint_ref'], Path(f'step-{step}/COMMITTED.json'))
    assets = [copy_ref(ref, Path(f'step-{step}/lora_adapter')/Path(ref['path']).name)
              for ref in pub['assets']]
    # The publication validator authenticates every manifest member in
    # lora_adapter, including metadata outside the public two-asset roster.
    committed = read(pub['checkpoint_ref'])
    parent = Path(pub['checkpoint_ref']['path']).parent
    adapter_files = {name: digest for rank in committed['ranks']
                     for name, digest in rank['files'].items() if name.startswith('lora_adapter/')}
    for name, digest in sorted(adapter_files.items()):
        relative = Path(name)
        require(len(relative.parts) == 2 and relative.name not in ('', '.', '..'),
                'invalid adapter file in checkpoint')
        if any(Path(asset['path']).name == relative.name for asset in assets):
            continue
        ref = pin(parent/relative)
        require(ref['sha256'] == digest, 'adapter metadata changed')
        copy_ref(ref, Path(f'step-{step}')/relative)
    publication = {k: v for k, v in pub.items()
                   if k not in ('bounded_tp4_ref','updated_rcp_ref','runtime_ref','continuous_activation_ref')}
    publication.update(status='candidate_for_updated_parity', checkpoint_ref=checkpoint, assets=assets)
    candidate = json_ref('candidate/POLICY-CANDIDATE.json', publication)
    manifest = json_ref('candidate/ASSETS.json', dict(status='CLOSED_IMMUTABLE',
                                                     assets=assets, publication_ref=candidate))
    return json_ref('candidate/POLICY.json', dict(policy, publication_ref=candidate,
                    asset_manifest_ref=manifest, assets=assets))


def previous_capture(state):
    require(state['consumed_cohorts'] and state.get('capture'), 'prior capture evidence missing')
    latest = state['consumed_cohorts'][-1]
    result = read(latest['result']['result_ref'])
    capture = state['capture']
    release = read(capture['resident_release_ref'])
    terminal = release['terminal']
    require(latest['result']['checkpoint_ref'] == state['resume_ref']
            and result['optimizer_steps'] == 1
            and result['generation_inventory_ref'] == capture['generation_inventory_ref']
            and release['status'] == 'released'
            and terminal['terminal_verified'] is True and terminal['unknown_requests'] == 0
            and terminal['generation_closed_ref'] == capture['generation_closed_ref'],
            'prior actor or capture terminal is not the direct predecessor')
    return capture


def publish_inbox(remote, service, checkout, target, previous, successor, step):
    """Install exact bytes under the existing guard and publish the marker last."""
    from formal_rollout import provider_modules
    import base64
    files = {}
    for path in sorted(checkout.rglob('*')):
        if path.is_file():
            files[str(path.relative_to(checkout))] = dict(
                sha256=pin(path)['sha256'], data=base64.b64encode(path.read_bytes()).decode())
    # Send the large adapter via stdin; no command-line payload or shared write.
    with provider_modules() as (lease, _, _, _):
        held=lease.probe(remote)
        updater=read(remote['config'])['hot_root']
        program = r"""
import base64, hashlib, json, os, sys, tempfile
from pathlib import Path
v = json.load(sys.stdin)
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip() == v['boot']
inbox, target = Path(v['inbox']), Path(v['target'])
assert inbox.resolve() == inbox and inbox.is_dir() and target.parent == inbox
sys.path.insert(0, v['storage_module'])
from local_runtime_storage import check_paths
data = {name: base64.b64decode(row['data'], validate=True) for name,row in v['files'].items()}
check_paths({'inbox':inbox}, min_free_bytes=(20<<30)+sum(map(len,data.values())))
def sync(path):
    fd = os.open(path, os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)
def publish(path, body):
    if path.exists():
        assert path.is_file() and path.read_bytes() == body
        return
    with tempfile.NamedTemporaryFile(dir=path.parent) as f:
        f.write(body); f.flush(); os.fsync(f.fileno()); os.link(f.name,path)
    sync(path.parent)
for name, body in data.items():
    path = target/name
    assert path.resolve() == path and path.is_relative_to(target) and path != target
    assert hashlib.sha256(body).hexdigest() == v['files'][name]['sha256']
    path.parent.mkdir(parents=True, exist_ok=True)
    publish(path, body)
    for parent in path.parents:
        if parent == inbox: break
        sync(parent)
sync(inbox)
marker = inbox/('after-step-'+str(v['step'])+'.json')
publish(marker,(json.dumps(v['change'],sort_keys=True)+'\n').encode())
print(json.dumps({'marker':str(marker),'installed':len(data)}))
"""
        payload = dict(boot=held['boot'], storage_module=updater,
                       inbox=service['policy_overlay']['update_mount'], target=str(target),
                       files=files, step=step, change=dict(previous=previous,successor=successor))
        response=lease.command(remote,['python3','-B','-c',program],input=json.dumps(payload),
                               capture_output=True,text=True)
    return json.loads(response.stdout)


def get_receipts(remote, ready, provider, depth, root, *, switched):
    from formal_rollout import provider_modules
    names = ('ACTIVATION.json', 'CACHE-POLICY.json', 'PRECISION.json')
    if switched:
        names += ('SWITCHED.json',)
    base = Path(provider['live_receipt_paths']['activation_ref']).parent
    directory = base.joinpath(*(['successor']*depth))
    with provider_modules() as (lease, _, _, _):
        result = {}
        for name in names:
            cmd = ['docker','exec',ready['container_id'],'cat',str(directory/name)]
            data = lease.command(remote,cmd,capture_output=True).stdout
            result[name] = receipt_bytes(root/name,data)
    return result


def fixed_contact(measurer, directory, url, wire, timeout):
    terminal = directory/'RESPONSE.json'
    if not terminal.exists():
        measurer.contact(directory,url,wire,timeout)
    ref = pin(terminal)
    require(read(ref)['status'] == 200, 'diagnostic HTTP call failed')
    return ref


def updated_parity(config, state, root, guard):
    from formal_rollout import acquire, idle, lease_plan, provider_modules, release, settings
    spec, provider = settings(config)
    plan = spec['updated_parity']
    measurer = load(local_path(pin(plan['measurement_module_ref'])['path']), 'formal_parity_http')
    reviewer_ref = pin(plan['reviewer_ref'])
    reviewer = load(local_path(reviewer_ref['path']), 'formal_parity_reviewer')
    capture = previous_capture(state)
    return prove_owners(config,state,root,guard,plan,measurer,reviewer_ref,reviewer,capture=capture)


def prove_owners(config,state,root,guard,plan,measurer,reviewer_ref,reviewer,*,capture=None,retained=None):
    """Use one shared guard and one publication, retaining separate natural responses per engine."""
    from contextlib import ExitStack
    from formal_rollout import (settings,lane_providers,close_tunnel,lease_plan,acquire,release,
                                idle,provider_modules)
    from formal_parity_workers import replay_and_score
    from bounded_tp4 import endpoint as verify_owner
    from publication import verify_bounded
    spec,primary=settings(config)
    providers=lane_providers(config,spec,primary)
    initial=capture is None
    step=state['step']
    require(step==read(state['resume_ref'])['step'] and
            0<plan['request_timeout_seconds']<config['deadline_epoch']-time.time(), 'invalid parity window')
    root.mkdir(parents=True,exist_ok=True)
    proof=root/'PARITY.json'
    if proof.exists():return pin(proof)
    prof=pin(plan['profile_ref'])
    packets=[read(ref) for ref in plan['natural_packet_refs']]
    require(0<len(packets)<=8 and all(p['stage']=='P' for p in packets),'real P diagnostic packets required')
    wires=[dict(p['wire'],cache_salt=state['resume_ref']['sha256']) for p in packets]
    if initial:
        kept=read(retained);pub=read(read(state['policy_ref'])['publication_ref'])
        verified=verify_bounded(pub,retained)
        require(kept['schema']=='omni-p-verified-replayed-tp4-v1' and kept['profile_ref']==prof
                and kept['lane'] in providers,'initial proof does not match configured lanes')
        verify_owner(verified,providers[kept['lane']]['initial_lane_ref'],pub['artifact_version'])
        proofs={kept['lane']:retained}
    else:proofs={}
    missing={k:p for k,p in providers.items() if k not in proofs}
    require(missing,'empty parity work')
    binding=dict(checkpoint_ref=state['resume_ref'])
    if not initial:binding['prior_release_ref']=capture['resident_release_ref']
    fence=root/'lease'
    remote,_,service=lease_plan(config,primary,fence,config['run_id']+'-policy-'+str(step),binding)
    before={k:read(p['initial_lane_ref'] if initial else read(capture['endpoint_witness_ref'])['lanes'][k])
            for k,p in providers.items()}
    active={k:read(v['activation_ref']) for k,v in before.items()}
    require(all(v['policy_ref']==active[primary['lane']]['policy_ref'] and
                content(v['profile_ref'])==content(prof) for v in active.values()), 'prior lane policy or profile differs')
    if not initial:
        inbox=Path(service['policy_overlay']['update_mount'])
        require(all(read(p['service_config'])['policy_overlay']['update_mount']==str(inbox)
                    for p in providers.values()),'formal services do not share the update inbox')
        target=inbox/f"{config['run_id']}-step-{step}"
        checkout=root/'activation-input'
        successor=relocate_candidate(state['policy_ref'],checkout,target)
        actual_publication=pin(checkout/'candidate/POLICY-CANDIDATE.json')
    closed_path=root/'NATURAL-DRAINED.json'
    if not closed_path.exists():
        acquire(fence,remote,primary,read(primary['service_ready']),service)
        with ExitStack() as stack:
            urls={}
            for k,p in missing.items():
                work=root/k;work.mkdir(exist_ok=True)
                with provider_modules() as (_,module,_,_):
                    tunnel=module.tunnel(dict(provider=p,remote_provider_guard=remote),work)
                stack.callback(close_tunnel,tunnel)
                urls[k]=f"http://127.0.0.1:{p['local_port']}"
                idle(urls[k],spec['drain_timeout_seconds'])
                earlier=get_receipts(remote,read(p['service_ready']),p,before[k].get('activation_depth',0),work/'prior',switched=False)
                require(all(content(earlier[n])==content(before[k][key]) for n,key in
                    [('ACTIVATION.json','activation_ref'),('PRECISION.json','precision_ref'),('CACHE-POLICY.json','cache_ref')]),
                    'live predecessor receipt differs')
            if not initial and not (root/'SWITCH-PUBLISHED.json').exists():
                write(root/'SWITCH-PUBLISHED.json',publish_inbox(remote,service,checkout,target,
                      active[primary['lane']]['policy_ref'],successor,step-1))
            lanes={}
            for k,p in missing.items():
                work=root/k
                ready=read(p['service_ready'])
                refs=[fixed_contact(measurer,work/'natural'/str(i),urls[k],wire,plan['request_timeout_seconds'])
                      for i,wire in enumerate(wires)]
                depth=before[k].get('activation_depth',0)+int(not initial)
                receipts=get_receipts(remote,ready,p,depth,work/'live',switched=not initial)
                act=read(receipts['ACTIVATION.json'])
                require(act['endpoint']==ready['endpoint'] and act['source_boot']==service['host_boot_id']
                    and act['service_identity']==active[k]['service_identity']
                    and {w['rank'] for w in act['workers']}==set(range(4)) and len(act['workers'])==4 and
                    all(w['compared_original_keys']==384 for w in act['workers']), 'resident owner readback differs')
                if initial:
                    require(content(receipts['ACTIVATION.json'])==content(before[k]['activation_ref']),
                            'initial lane changed during observation')
                else:
                    sw=read(receipts['SWITCHED.json'])
                    require(act['policy_ref']==successor and sw['previous_policy_ref']==active[k]['policy_ref']
                        and content(act['publication_ref'])==content(actual_publication)
                        and sw['old_removed_from_engine'] is True and sw['cache_reset'] is True
                        and sw['old_id']==active[k]['adapter_id'] and sw['new_id']==act['adapter_id']!=sw['old_id']
                        and content(sw['activation_ref'])==content(receipts['ACTIVATION.json']), 'hot update receipt differs')
                idle(urls[k],spec['drain_timeout_seconds'])
                lanes[k]=dict(receipts=receipts,natural_refs=refs,depth=depth)
            write(closed_path,dict(terminal_verified=True,unknown_requests=0,checkpoint_ref=state['resume_ref'],lanes=lanes))
    closed=read(closed_path)
    require(set(closed['lanes'])==set(missing) and closed['checkpoint_ref']==state['resume_ref'],'parity receipt drift')
    write(root/'NATURAL-RELEASED.json',release(fence,remote,service,closed,spec['drain_timeout_seconds']))
    for k,obs in closed['lanes'].items():
        work=root/k
        receipts=obs['receipts']
        # 175 is time-shared with the actor. No simultaneous score or replay on its devices.
        paired,exit_ref,ranks=replay_and_score(config,state,work,guard,plan,packets,wires,
            obs['natural_refs'],read(receipts['ACTIVATION.json']),reviewer,lane=k)
        row=dict(schema='omni-p-verified-replayed-tp4-v1',lane=k,checkpoint_ref=state['resume_ref'],
            profile_ref=prof,reviewer_ref=reviewer_ref,paired_ref=paired,response_ref=read(paired)['rows'][0]['source_refs'][3],
            exit_ref=exit_ref,rank_refs=ranks,limits=plan['limits'],activation_ref=receipts['ACTIVATION.json'],
            cache_ref=receipts['CACHE-POLICY.json'],precision_ref=receipts['PRECISION.json'],activation_depth=obs['depth'])
        if not initial:row['activated_publication_ref']=actual_publication
        proofs[k]=write(work/'PARITY.json',row)
    if len(proofs)==1:return next(iter(proofs.values()))
    return write(proof,dict(schema='individually-verified-tp4-lanes-v1',lane_proofs=proofs,
                            profile_ref=prof,checkpoint_ref=state['resume_ref']))

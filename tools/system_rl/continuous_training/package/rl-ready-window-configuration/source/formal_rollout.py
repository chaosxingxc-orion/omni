"""Connect the original sampled capture to an already running TP4 provider."""
from contextlib import contextmanager, ExitStack
import copy
import json
import os
from pathlib import Path
import sys
import time

from dependencies import ROOT, require
from run_package import pin, read, write


@contextmanager
def provider_modules():
    directory = ROOT/'collection-dispatch-integration/production-pipeline-next-version-v1'
    names = ('base', 'bootstrap', 'provider', 'lease_runtime', 'ready_mode',
             'remote_assets', 'sources', 'frozen_read')
    saved = {name: sys.modules.pop(name, None) for name in names}
    paths = sys.path[:]
    try:
        sys.path.insert(0, str(directory))
        import lease_runtime
        import provider
        import ready_mode
        import base
        original_tunnel = provider.tunnel

        def tunnel(request, root):
            # Match the lease transport without changing an existing lease ref
            # or the provider module used by the sealed DATA program.
            remote = request['remote_provider_guard']
            ssh = remote['ssh']
            executable = ('/usr/bin/ssh' if os.environ.get('SAEA_ENV_SCOPE') == 'container'
                          else ssh.get('executable', 'ssh'))
            bound = dict(remote, ssh=dict(ssh, executable=executable))
            return original_tunnel(dict(request, remote_provider_guard=bound), root)

        provider.tunnel = tunnel
        yield lease_runtime, provider, ready_mode, base
    finally:
        sys.path[:] = paths
        for name, module in saved.items():
            sys.modules.pop(name, None)
            if module is not None:
                sys.modules[name] = module


def lane_providers(config, spec, primary):
    refs = spec.get('provider_refs')
    rows = {primary['lane']: primary} if refs is None else {name: read(ref) for name,ref in refs.items()}
    lanes = spec.get('timeshared_lanes',config['formal_actor']['rollout_lanes'])
    require(lanes and len(lanes) <= 2 and len(lanes) == len(set(lanes))
            and set(rows) == set(lanes) and rows.get(primary['lane']) == primary,
            'resident lanes must include the primary DATA provider exactly once')
    for name,p in rows.items():
        require(p['schema'] == 'existing-tp4-provider-v2' and name == p['lane']
                and name in ('a','b') and p['devices'] == ([0,1,2,3] if name=='a' else [4,5,6,7])
                and p['host'] == primary['host'] and p['guard_path'] == primary['guard_path'],
                'provider differs from the configured shared host guard or physical lane')
    for key in ('local_port','remote_port'):
        require(len({p[key] for p in rows.values()}) == len(rows), 'duplicate resident '+key)
    return {k:rows[k] for k in lanes}


def startup_placement(config):
    # A committed successor does not have a live service until qualify()
    # runs under the iteration lock. Validate its placement, not a stale owner.
    if os.environ.get('OMNI_TIMESHARED_ROLLOUT_REF'):
        from timeshared_rollout import deployment
        plan=deployment(config)
        require(config['formal_rollout']['schema']=='omni-formal-resident-rollout-v1'
                and plan['lane'] in config['formal_actor']['rollout_lanes'],
                'time-sharing deployment is outside the original topology')
    else:
        require(settings(config) is not None,'strict actor requires resident rollout placement')


def settings(config):
    spec = config.get('formal_rollout')
    if spec is None:
        return None
    if os.environ.get('OMNI_TIMESHARED_ROLLOUT_REF'):
        from timeshared_rollout import deployment,binding
        plan=deployment(config);bound=binding(config,plan);lane=plan['lane']
        spec=dict(spec,provider_refs={lane:bound['provider']},timeshared_lanes=[lane],
                  timeshared_proof=bound['proof'],ssh=dict(spec['ssh'],target=plan['host']))
    elif os.environ.get('OMNI_FORMAL_RESUME_REF'):
        from formal_resume import active
        binding = active(config)
        if binding:
            spec = dict(spec, provider_refs=binding['providers'])
    from program_placement import resident_provider
    provider = resident_provider(config)
    require(spec['schema'] == 'omni-formal-resident-rollout-v1'
            and provider and lane_providers(config, spec, provider)
            and 0 < spec['safety_seconds'] <= 5
            and type(spec['episode_concurrency']) is int and spec['episode_concurrency'] > 0
            and type(spec['drain_timeout_seconds']) is int and spec['drain_timeout_seconds'] > 0,
            'formal rollout requires its configured resident TP4 lane and lease limits')
    require(spec['ssh']['target'] == provider['host'], 'formal rollout SSH route differs from provider')
    return spec, provider


def endpoint(config, request, root):
    """Build a witness from the exact active artifact and independently reviewed parity."""
    spec, provider = settings(config)
    providers = lane_providers(config, spec, provider)
    policy = read(request['policy_ref'])
    pub = read(policy['publication_ref'])
    if spec.get('continuous_training'):
        from continuous_policy import verify
        evidence=read(pub['continuous_activation_ref'])
        require(evidence['precision_lock_ref']==spec['continuous_training']['precision_lock_ref']
                and set(verify(pub,pub['continuous_activation_ref']))==set(providers),'unexpected continuous endpoints')
        witness=write(root/'ENDPOINTS.json',dict(schema='continuous-training-endpoints-v1',fixture=False,passed=True,
            policy_version=pub['artifact_version'],model=policy['model'],thinking=False,asset_manifest_ref=policy['asset_manifest_ref'],
            profile_ref=evidence['profile_ref'],lanes=evidence['lanes']))
        return dict(witness_ref=witness)
    from bounded_tp4 import validate_routes
    evidence = read(pub['bounded_tp4_ref'])
    proofs = evidence.get('lane_proofs') or {evidence['lane']:pub['bounded_tp4_ref']}
    require(set(proofs)==set(providers), 'every formal lane requires an individual numeric verdict')
    lanes = {}
    for name,p in providers.items():
        proof=read(proofs[name])
        require(proof['schema']=='omni-p-verified-replayed-tp4-v1' and proof['lane']==name
                and proof['profile_ref']==evidence['profile_ref'], 'crossed formal proof')
        lane={key:proof[key] for key in ('lane','activation_ref','precision_ref','cache_ref','response_ref')}
        lane['activation_depth']=proof.get('activation_depth',0)
        if lane['activation_depth']==0 and p.get('initial_lane_ref'):
            live=read(p['initial_lane_ref']);ready=read(p['service_ready'])
            require(live['lane']==name and live.get('activation_depth',0)==0
                    and all(live[key]==ready['local_receipts'][item] for key,item in
                        [('activation_ref','ACTIVATION.json'),('precision_ref','PRECISION.json'),('cache_ref','CACHE-POLICY.json')])
                    and ready['adapter_sha256']==read(live['activation_ref'])['weights_ref']['sha256'],
                    'restarted provider does not match observed startup receipts')
            lane=live
        lanes[name]=write(root/name/'LANE.json',lane)
    witness = write(root/'ENDPOINTS.json', dict(schema='omni-bounded-tp4-endpoints-v1',
        policy_version=pub['artifact_version'],model=policy['model'],thinking=False,fixture=False,passed=True,
        asset_manifest_ref=policy['asset_manifest_ref'],profile_ref=evidence['profile_ref'],lanes=lanes))
    validate_routes(pub,witness,read(evidence['profile_ref']),sampled=True)
    return dict(witness_ref=witness)


def qualify(config, state, root, guard):
    """Use the initial sealed proof or the configured, isolated update verifier."""
    from publication import promote, verify_bounded
    if config['formal_rollout'].get('continuous_training'):
        from continuous_policy import activate
        require(not os.environ.get('OMNI_TIMESHARED_ROLLOUT_REF'), 'cannot time-share resident training')
        spec,provider=settings(config)
        grant=read(json.loads(os.environ['OMNI_ROLLOUT_HOST_GRANT_REF']))
        require(grant['iteration_ref']==pin(config['_path']) and provider['host'] in grant['hosts'],
                'resident rollout not within current grant')
        promoted=activate(config,state,root)
        source=copy.deepcopy(read(state['input_ref']))
        require(source['policy']['current_ref']==state['policy_ref'],'updated rollout input drift')
        source['policy']['current_ref']=promoted
        source['continuation']['publication_ref']=read(promoted)['publication_ref']
        state['input_ref']=write(root/'QUALIFIED-INPUT.json',source)
        return promoted
    if os.environ.get('OMNI_TIMESHARED_ROLLOUT_REF'):
        from timeshared_rollout import ensure_endpoint
        ensure_endpoint(config,state)
    spec, provider = settings(config)
    grant=os.environ.get('OMNI_ROLLOUT_HOST_GRANT_REF')
    if grant:
        allowed=read(json.loads(grant))
        require(allowed['iteration_ref']==pin(config['_path']) and
                provider['host'] in allowed['hosts'],'rollout provider is outside the current host grant')
    policy_ref = state['policy_ref']
    policy = read(policy_ref)
    pub = read(policy['publication_ref'])
    if pub.get('bounded_tp4_ref'):
        proof = read(pub['bounded_tp4_ref'])
        if ((proof.get('schema') == 'omni-p-verified-replayed-tp4-v1' and list(lane_providers(config,spec,provider))==[proof['lane']])
                or (proof.get('schema')=='individually-verified-tp4-lanes-v1' and set(proof['lane_proofs'])==set(config['formal_actor']['rollout_lanes']))):
            verify_bounded(pub, pub['bounded_tp4_ref'])
            return policy_ref
    root.mkdir(parents=True, exist_ok=True)
    if 'timeshared_proof' in spec:
        ref=pin(spec['timeshared_proof'])
        require(read(ref)['checkpoint_ref']==state['resume_ref'],'time-sharing checkpoint changed')
        checked=verify_bounded(pub,ref)
        from bounded_tp4 import endpoint as verify_owner
        verify_owner(checked,provider['initial_lane_ref'],pub['artifact_version'])
    elif state['resume_ref'] == config['initial_resume_ref']:
        ref = pin(spec['initial_parity_ref'])
        from bounded_tp4 import endpoint as verify_owner
        checked=verify_bounded(pub,ref)
        providers=lane_providers(config,spec,provider)
        if set(checked['states']) < set(providers):
            from formal_parity import prove_owners
            from dependencies import load, local_path
            plan=spec['updated_parity']
            reviewer_ref=pin(plan['reviewer_ref'])
            reviewer=load(local_path(reviewer_ref['path']),'formal_initial_reviewer')
            measurer=load(local_path(pin(plan['measurement_module_ref'])['path']),'formal_initial_http')
            ref=prove_owners(config,state,root/'initial',guard,plan,measurer,reviewer_ref,reviewer,retained=ref)
            checked=verify_bounded(pub,ref)
        require(set(checked['states'])==set(providers),'initial proof missing a configured lane')
        for name,p in providers.items():
            ready=read(p['service_ready']);live=read(p['initial_lane_ref'])
            require(live['activation_ref']==ready['local_receipts']['ACTIVATION.json'],
                    'initial numeric proof lacks current live owner')
            verify_owner(checked,p['initial_lane_ref'],pub['artifact_version'])
    else:
        from formal_parity import updated_parity
        ref = updated_parity(config, state, root, guard)
    promoted = promote(policy_ref, ref, None, root/'promoted', bounded_tp4=True)
    source = copy.deepcopy(read(state['input_ref']))
    require(source['policy']['current_ref'] == policy_ref, 'qualification input policy differs')
    source['policy']['current_ref'] = promoted
    source['continuation']['publication_ref'] = read(promoted)['publication_ref']
    state['input_ref'] = write(root/'QUALIFIED-INPUT.json', source)
    return promoted


def idle(url, timeout):
    from urllib.request import ProxyHandler, build_opener
    end = time.monotonic() + timeout
    opener = build_opener(ProxyHandler({}))
    while time.monotonic() < end:
        with opener.open(url+'/metrics', timeout=min(10, max(1, end-time.monotonic()))) as response:
            rows = response.read().decode().splitlines()
        totals = [[float(row.split()[1]) for row in rows if
                   row.startswith('vllm:num_requests_'+key+'{')] for key in ('running','waiting')]
        require(all(totals), 'resident TP4 metrics do not expose queue state')
        if all(sum(values) == 0 for values in totals):
            return
        time.sleep(min(1, max(0, end-time.monotonic())))
    raise ValueError('resident TP4 did not drain within the configured timeout')


def lease_plan(config, provider, fence, owner, binding):
    spec, _ = settings(config)
    service = read(provider['service_config'])
    ready = read(provider['service_ready'])
    mode = read(config['program_template_ref'])
    (fence/'activation').mkdir(parents=True, exist_ok=True)
    remote_seed = dict(path=provider['guard_path'], host=provider['host'], ssh=spec['ssh'],
                       hot_root=str(Path(provider['remote_hot_parent'])/owner))
    seed = dict(remote_provider_guard=remote_seed, storage=mode['storage'],
                authority=dict(run_id=owner))
    with provider_modules() as (_, _, ready_mode, _):
        remote = ready_mode.guard_config(seed, fence)
    write(fence/'activation/CONFIG.json', dict(authority=seed['authority'], **binding))
    return remote, ready, service


def acquire(fence, remote, provider, ready, service):
    prior = fence/'REMOTE-LEASE.json'
    with provider_modules() as (lease, data_provider, _, _):
        if prior.exists():
            old = read(prior)
            observed = lease.probe(remote)
            require(old['remote'] == remote and all(old['state'][key] == observed[key] for key in
                ('owner','pid','start_ticks','boot','guard_inode','guard_device')),
                'cannot adopt a different resident TP4 lease')
        else:
            physical = data_provider.preflight(dict(provider=provider, remote_provider_guard=remote))
            require(physical['container'] == ready['container_id']
                    and physical['boot'] == service['host_boot_id'], 'resident provider preflight drift')
            intent = fence/'ACQUIRE-INTENT.json'
            require(not intent.exists(), 'reconcile the original unreturned lease acquisition')
            write(intent, dict(remote=remote, activation=pin(fence/'activation/CONFIG.json')))
            lease.acquire(remote, fence)


def release(fence, remote, service, evidence, timeout):
    with provider_modules() as (lease, _, _, _):
        cfg = read(remote['config'])
        def live_state():
            return json.loads(lease.command(remote, ['cat', cfg['lease_state']],
                                            capture_output=True, text=True).stdout)
        state = live_state()
        if state['status'] == 'held':
            observed = lease.probe(remote)
            require(observed['pid'] == read(fence/'REMOTE-LEASE.json')['state']['pid']
                    and observed['boot'] == service['host_boot_id'], 'release owner differs')
            lease.release(remote, evidence)
            end = time.monotonic() + timeout
            while state['status'] == 'held' and time.monotonic() < end:
                time.sleep(1)
                state = live_state()
        require(state['status'] == 'released' and state['owner'] == cfg['owner']
                and state['terminal'] == dict(evidence, owner=cfg['owner']),
                'resident release remains unresolved')
        return state


def close_tunnel(tunnel):
    if tunnel.poll() is None:
        tunnel.terminate()
        tunnel.wait(timeout=15)


def collect_lane(source_ref, cohort, lane):
    import subprocess
    import traceback

    try:
        from capture_stage import bounded
        bounded(source_ref, cohort, 'lane', lane=lane)
    except Exception as exc:
        # Scoped source modules exist only in this worker. Return plain data
        # so unpickling their exception classes cannot hide the actual fault.
        details = []
        cause = exc
        while cause is not None:
            if isinstance(cause, subprocess.CalledProcessError) and cause.stderr:
                value = cause.stderr
                details.append(value.decode(errors='replace') if isinstance(value, bytes) else value)
            cause = cause.__cause__
        return dict(traceback=traceback.format_exc(), subprocess_stderr=details)
    return None


def collect_lanes(source_ref,cohort,lanes,fence):
    # Original capture scopes mutate sys.modules. Fork before opening either
    # CaptureLease; both children inherit the controller's closed path map.
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from multiprocessing import get_context
    with ProcessPoolExecutor(len(lanes),mp_context=get_context('fork')) as workers:
        futures={workers.submit(collect_lane,source_ref,cohort,lane):lane for lane in lanes}
        for task in as_completed(futures):
            lane = futures[task]
            try:
                error = task.result()
                if error is not None:
                    evidence = write(fence/'worker-errors'/str(time.time_ns())/(lane+'.json'), error)
                    if not (fence/'DRAIN').exists():
                        write(fence/'DRAIN', dict(reason='resident lane failed', lane=lane, failure_ref=evidence))
                    raise RuntimeError(f'resident lane {lane} failed; details: '+evidence['path'])
            except BaseException:
                if not (fence/'DRAIN').exists():
                    write(fence/'DRAIN',dict(reason='resident lane failed',lane=lane))
                raise


def rearm_unissued_capture(config, request_ref, request, cohort, fence, lanes):
    """Adopt the same healthy remote lease after a zero-request worker failure."""
    import fcntl
    from iteration import original_controller
    from storage import commit

    lost, drain = (fence/name for name in ('LEASE-LOST.json', 'DRAIN'))
    if not lost.exists() and not drain.exists():
        return
    with ExitStack() as stack:
        guards = [cohort/'COHORT.guard', *(cohort/f'lane-{i}.guard' for i in range(len(lanes))),
                  fence/'lease-admission.guard']
        for path in guards:
            file = stack.enter_context(path.open('ab'))
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        latched = read(drain) if drain.exists() else {}
        require(latched.get('reason') in ('LEASE_LOST', 'resident lane failed'),
                'unrelated capture drain requires reconciliation')
        prior = read(fence/'REMOTE-LEASE.json')
        require(read(fence/'activation/CONFIG.json')['request_ref'] == request_ref,
                'capture lease belongs to a different window')
        usage = original_controller().usage(config)
        preparation_files = {'BUSINESS.json', 'LANE.json', 'assets.sqlite', 'assets.sqlite-shm', 'assets.sqlite-wal'}
        require(usage == request['usage_before'] and usage['unknown'] == 0
                and all(path.is_dir() or path.name in preparation_files
                        for path in (cohort/'groups').rglob('*')), 'capture requests require original reconciliation')
        epoch = fence/'LEASE-EPOCH.json'
        prior_epoch = read(epoch)
        prior_rearm = read(prior_epoch['rearm_ref']) if 'rearm_ref' in prior_epoch else None
        loss = None
        if lost.exists():
            loss = read(lost)
            identity = ('pid', 'boot', 'start_ticks', 'guard_inode', 'guard_device', 'owner')
            require(loss['reason'] == 'LEASE_LOST'
                    and loss['expected'] == {k:prior['state'][k] for k in identity},
                    'capture lease loss belongs to a different owner')
        elif not (prior_rearm and prior_rearm['drain'] == latched):
            # A worker can fail after validating its lease but before creating
            # any HTTP intent. Preserve its failure and the original watermark.
            failure = latched.get('failure_ref')
            if failure is None and latched.get('lane') in lanes:
                older = list((fence/'worker-errors').glob('*/'+latched['lane']+'.json'))
                if len(older) == 1 and older[0].stat().st_mtime_ns <= drain.stat().st_mtime_ns:
                    failure = pin(older[0])
            require(latched['reason'] == 'resident lane failed' and latched.get('lane') in lanes
                    and failure and read(failure).get('traceback'), 'unattributed worker failure')
        if loss is not None or prior_rearm is None or prior_rearm['drain'] != latched:
            evidence = write(fence/'lease-generations'/f'rearmed-{time.time_ns()}.json', dict(
                request_ref=request_ref, usage=usage, prior=prior_epoch, loss=loss,
                drain=latched, remote=prior,
                worker_failure_ref=failure if loss is None and latched['reason']=='resident lane failed'
                and not (prior_rearm and prior_rearm['drain'] == latched) else None))
            commit(epoch, dict(rearm_ref=evidence, state=prior['state']), replace=True)
            if lost.exists():
                lost.unlink()
        else:
            # Resume a crash after archiving the loss but before opening admission.
            require(prior_rearm['request_ref'] == request_ref and prior_rearm['usage'] == usage
                    and prior_rearm['remote'] == prior,
                    'incomplete capture rearm differs from the original drain')
        # No previous observer or request can still act: controller, each lane
        # and admission are exclusive and the remote identity was just probed.
        drain.unlink()
        fd = os.open(fence, os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def capture(config, request_ref, request, root):
    spec, provider = settings(config)
    source = copy.deepcopy(read(request['input_ref']))
    witness = read(request['endpoint_witness_ref'])
    providers = lane_providers(config,spec,provider)
    require(source['prepare_mode'] == 'full_train_screen' and source.get('sampled_window_ref')
            and source['policy']['current_ref'] == request['policy_ref']
            and source['execution_registry'] == config['registry']
            and set(witness['lanes']) == set(providers), 'formal capture requires a fresh sampled window')
    root.mkdir(parents=True, exist_ok=True)
    cohort = root/'cohort'
    cohort.mkdir(exist_ok=True)
    fence = root/'lease'
    owner = config['run_id']+'-capture-'+str(request['sequence'])
    remote, ready, service = lease_plan(config, provider, fence, owner, dict(request_ref=request_ref))
    urls={k:f"http://127.0.0.1:{p['local_port']}" for k,p in providers.items()}
    source.update(output_root=str(cohort), cohort_id='dp8-'+request_ref['sha256'][:20],
        lease_fence=dict(root=str(fence), remote=remote, safety_seconds=spec['safety_seconds']))
    source['native']['attempt_id'] = source['cohort_id']
    if 'official_index_cache_root' in spec:
        source['capture']['official_index_cache_root'] = spec['official_index_cache_root']
    if spec.get('continuous_training'):source['capture']['screened_p_parser_member']=spec['screened_p_parser_member']
    if 'dispatch_unit' in spec:source['capture']['dispatch_unit']=spec['dispatch_unit']
    source['capture'].update(lanes=list(providers),episode_concurrency={k:spec['episode_concurrency'] for k in providers},
        endpoint_policy_ref=request['endpoint_witness_ref'],lane_transports={k:dict(
            local_endpoint=urls[k]+'/v1/chat/completions',remote_endpoint=read(p['service_ready'])['endpoint'])
            for k,p in providers.items()})
    source['policy']['runtime_evidence_ref'] = request['endpoint_witness_ref']
    source_ref = write(root/'CAPTURE.json', source)
    from capture_stage import bounded
    from iteration import original_controller
    from provider_policy import observe_resident
    terminal = root/'DRAINED.json'
    if not terminal.exists():
        prepared=root/'PREPARED.json'
        if not prepared.exists() and not (cohort/'RUN-BOOT.json').exists():
            result=bounded(source_ref, cohort, 'prepare')
            require(result['can_dispatch'] is True,'capture preparation incomplete')
            write(prepared,dict(source=source_ref,input=pin(cohort/'INPUT.json'),
                                selected=pin(cohort/'SELECTED.json')))
        acquire(fence, remote, provider, ready, service)
        with ExitStack() as stack:
            sessions=[]
            for lane,p in providers.items():
                directory=fence/lane
                write(directory/'REMOTE-LEASE.json',read(fence/'REMOTE-LEASE.json'))
                session_ref=write(directory/'SESSION.json',dict(resident_provider=p,devices=p['devices'],
                    kind='candidate',policy_ref=request['policy_ref'],endpoint_witness_ref=request['endpoint_witness_ref'],
                    adapter_digest=read(read(witness['lanes'][lane])['activation_ref'])['weights_ref']['sha256']))
                call=dict(provider=p,policy_session_ref=session_ref)
                session=read(session_ref)
                with provider_modules() as (_, data_provider, _, _):
                    active=observe_resident(call,directory,session,'open')
                    write(directory/'PROVIDER-ACTIVE.json',active)
                    tunnel=data_provider.tunnel(dict(provider=p,remote_provider_guard=remote),directory)
                stack.callback(close_tunnel,tunnel)
                sessions.append((call,directory,session))
            if (fence/'DRAIN').exists() or (fence/'LEASE-LOST.json').exists():
                for url in urls.values():idle(url,spec['drain_timeout_seconds'])
                rearm_unissued_capture(config, request_ref, request, cohort, fence, list(providers))
            if len(providers)==1:
                bounded(source_ref,cohort,'lane',lane=next(iter(providers)))
            else:
                collect_lanes(source_ref,cohort,list(providers),fence)
            result=bounded(source_ref,cohort,'close')
            write(root/'GENERATION-RESULT.json',result)
            for url in urls.values():idle(url,spec['drain_timeout_seconds'])
            usage=original_controller().usage(config)
            require(usage['unknown']==0,'formal capture has unresolved request outcomes')
            with provider_modules():
                for call,directory,session in sessions:
                    write(directory/'PROVIDER-CLOSED.json',observe_resident(call,directory,session,'close'))
            write(terminal,dict(terminal_verified=True,unknown_requests=0,usage=usage,
                  request_ref=request_ref,generation_closed_ref=result['generation_closed_ref'],
                  closed_lanes={c['provider']['lane']:pin(d/'PROVIDER-CLOSED.json') for c,d,_ in sessions}))
    evidence = read(terminal)
    result = read(root/'GENERATION-RESULT.json')
    require(evidence['request_ref'] == request_ref and
            evidence['generation_closed_ref'] == result['generation_closed_ref'],
            'terminal capture evidence differs')
    released = write(root/'RELEASED.json', release(fence, remote, service, evidence,
                                                    spec['drain_timeout_seconds']))
    if 'timeshared_lanes' in spec:
        from timeshared_rollout import stop_after_capture
        stop_after_capture(provider,service,ready,evidence,released,root)
    return dict(result, resident_release_ref=released, charged_tokens_upper=evidence['usage']['cost_units'])


def run(config, request_ref, directory):
    request = read(request_ref)
    root = Path(request['output_root'])
    root.mkdir(parents=True, exist_ok=True)
    if request['stage'] == 'endpoint':
        payload = endpoint(config, request, root)
    else:
        require(request['stage'] == 'capture' and not request['extra'].get('purpose'),
                'formal rollout stage does not consume diagnostic requests')
        payload = capture(config, request_ref, request, root)
    write(Path(directory)/'RESPONSE.json', dict(status='complete', fixture=False,
          request_ref=request_ref, **payload))

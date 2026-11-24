"""Submit and reconcile one strict FSDP8 window through the original launcher."""

import copy
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

from dependencies import require
from run_package import pin, read, write


def settings(config):
    spec = config.get('formal_actor')
    if spec is None:
        return None
    require(spec['schema'] == 'formal-strict-actor-v1'
            and config['execution_mode'] == 'formal'
            and config['formal_rl_enabled'] is True
            and config.get('sampled_training')
            and config.get('initial_resume_ref')
            and not config.get('resident', {}).get('enabled')
            and not config.get('newprocess_replay', {}).get('enabled'),
            'strict actor requires a restored, sampled formal iteration')
    require(type(spec['return_max_bytes']) is int and spec['return_max_bytes'] > 0
            and type(spec['return_timeout_seconds']) is int and 0 < spec['return_timeout_seconds'] <= 86400
            and type(spec['terminal_poll_seconds']) is int and spec['terminal_poll_seconds'] > 0
            and type(spec['submission_timeout_seconds']) is int and spec['submission_timeout_seconds'] > 0
            and re.fullmatch(r'[0-9a-f-]{36}', spec['source_boot']),
            'strict actor return envelope differs')
    launcher = pin(spec['launcher_ref'])
    require(Path(launcher['path']).name == 'stage_bounded_actor_step.py',
            'strict actor requires the retained staging entry')
    return spec


def recovery_run(checkpoint_ref, step):
    marker = Path(checkpoint_ref['path'])
    require(marker.is_absolute() and '..' not in marker.parts
            and marker.name == 'COMMITTED.json'
            and marker.parent.name == f'step-{step}'
            and marker.parents[1].name == 'hot', 'strict checkpoint path differs')
    return marker.parents[2]


def observe_recovered_actor(template, target, intent_ref, directory):
    """Observe the original actor after its failed preflight was reconciled."""
    probe = '''import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); config=root/'TRAIN.local.json'; boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
if not (root/'hot/CONTAINER.json').is_file():
    print('null'); sys.exit(0)
stamp=json.loads((root/'hot/INTENT.json').read_bytes())
preflight=json.loads((root/'CAPTURE-PREFLIGHT.json').read_bytes())
assert stamp['boot_id']==boot and stamp['devices']==list(range(8))
assert stamp['config_sha256']==preflight['train_config_sha256']==hashlib.sha256(config.read_bytes()).hexdigest()
cfg=json.loads(config.read_bytes())
print(json.dumps(dict(boot=boot, config_sha256=stamp['config_sha256'], name=cfg['name'],
  inventory_sha256=preflight['inventory_sha256'], checkpoint_sha256=cfg['restore_checkpoint']['sha256'],
  container=json.loads((root/'hot/CONTAINER.json').read_bytes()))))'''
    raw = subprocess.check_output(['ssh', '-oBatchMode=yes', '-oConnectTimeout=10', template['host'],
                                   shlex.join(['python3', '-B', '-c', probe, str(target)])], timeout=20)
    result = json.loads(raw)
    if result is None:
        return False
    require(result['name'] == template['run'] and result['boot'] == template['expected_boot']
            and result['inventory_sha256'] == template['generation']['generation_inventory_ref']['sha256']
            and result['checkpoint_sha256'] == template['restore_checkpoint_sha256'],
            'recovered actor identity differs from original submission')
    marker = directory/'SUBMIT-RECOVERED.json'
    if marker.exists():
        old = read(marker)
        require(old['intent_ref'] == intent_ref
                and all(old['observation'][k] == result[k] for k in result if k != 'container'),
                'recovered actor configuration changed')
        if old['observation']['container'] != result['container']:
            retries = [read(path) for path in (directory/'SCORING-RECOVERY-SUBMIT.json',
                       directory/'RECORDING-POPULATION-RECOVERY-SUBMIT.json') if path.is_file()]
            require(all(retry['original_intent'] == intent_ref
                        and retry['plan']['config_sha256'] == result['config_sha256'] for retry in retries)
                    and retries and retries[0]['plan']['container_id'] == old['observation']['container']['container_id'],
                    'replacement actor lacks a matching scoring recovery intent')
    else:
        write(marker, dict(intent_ref=intent_ref, observation=result))
    return True


def scoring_timeout_policy(config, spec):
    raw = os.environ.get('OMNI_FORMAL_ACTOR_TIMEOUT_REF')
    if not raw:
        return None
    policy = read(json.loads(raw))
    require(policy['schema'] == 'formal-actor-timeout-policy-v1'
            and policy['iteration_ref'] == pin(config['_path'])
            and policy['boot'] == spec['source_boot']
            and type(policy['timeout_seconds']) is int
            and 0 < policy['timeout_seconds'] <= 86400,
            'scoring timeout policy does not match the original iteration')
    for key, name in (('launcher', 'launch_retained_hf_fsdp.py'),
                      ('recovery', 'recover_training_timeout.py')):
        require(Path(pin(policy[key])['path']).name == name, 'invalid recovery source pin')
    return policy


def submit_scoring_recovery(template, target, intent_ref, directory, policy, *, population_proof=None):
    """Use the original slot and guard, only after the remote watchdog closed."""
    probe = '''import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1])
terminal=json.loads((root/'hot/EXIT.json').read_bytes())
intent=json.loads((root/'hot/INTENT.json').read_bytes())
raw=(root/'TRAIN.local.json').read_bytes(); cfg=json.loads(raw)
print(json.dumps(dict(terminal=terminal,hot_intent=intent,config_sha256=hashlib.sha256(raw).hexdigest(),
    timeout_seconds=cfg['timeout_seconds'],generation_sha256=cfg['generation']['generation_inventory_ref']['sha256'],
    checkpoint_sha256=cfg['restore_checkpoint']['sha256'],
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip())))'''
    ssh = ['ssh', '-oBatchMode=yes', '-oConnectTimeout=10', template['host']]
    observed = json.loads(subprocess.check_output(
        [*ssh, shlex.join(['python3', '-B', '-c', probe, str(target)])], timeout=30))
    terminal = observed['terminal']
    population = population_proof is not None
    if ((population and terminal['failure'] is not None)
            or (not population and terminal['failure'] not in (
                "TimeoutError('bounded same-input forward deadline')",
                "TimeoutError('bounded formal actor deadline')"))):
        return None
    require(observed['boot'] == policy['boot'] == observed['hot_intent']['boot_id']
            and observed['config_sha256'] == observed['hot_intent']['config_sha256']
            and observed['generation_sha256'] == template['generation']['generation_inventory_ref']['sha256']
            and observed['checkpoint_sha256'] == template['restore_checkpoint_sha256']
            and observed['timeout_seconds'] < policy['timeout_seconds']
            and terminal['complete_ranks'] == 0 and terminal['exit_code'] != 0
            and terminal['container_id'], 'terminal scoring timeout does not match original submission')
    plan = dict(schema='formal-actor-scoring-timeout-v1', boot=policy['boot'],
                config_sha256=observed['config_sha256'], container_id=terminal['container_id'],
                generation_sha256=observed['generation_sha256'], checkpoint_sha256=observed['checkpoint_sha256'],
                timeout_seconds=policy['timeout_seconds'])
    import base64
    slug = 'scoring-timeout-recovery'
    marker = directory/'SCORING-RECOVERY-SUBMIT.json'
    if population:
        require(terminal['container_id'] == population_proof['terminal_container'],
                'recording population recovery is for a different terminal container')
        source = pin(population_proof['population_ref'])
        require(source['bytes'] <= 1024**2, 'recording population supplement is too large')
        plan.update(schema='formal-actor-recording-population-recovery-v1', population_ref=source,
                    population_base64=base64.b64encode(Path(source['path']).read_bytes()).decode())
        slug = 'recording-population-recovery'
        marker = directory/'RECORDING-POPULATION-RECOVERY-SUBMIT.json'
    files = {'SCORING-TIMEOUT-PLAN.json': (json.dumps(plan, sort_keys=True)+'\n').encode()}
    for key in ('launcher', 'recovery'):
        ref = pin(policy[key])
        files[Path(ref['path']).name] = Path(ref['path']).read_bytes()
    from hashlib import sha256
    row = dict(plan={k: v for k, v in plan.items() if k != 'population_base64'},
               original_intent=intent_ref, submitted_at=time.time(),
               target=str(target), files={name:sha256(raw).hexdigest() for name,raw in files.items()})
    # This durable intent precedes any remote action. A failed handoff must
    # reconcile the original submission rather than launch a second recovery.
    write(marker, row)
    deploy = '''import base64,json,os,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]); slug=sys.argv[2]
assert slug in ('scoring-timeout-recovery','recording-population-recovery')
plan=root/slug
assert root.resolve()==root and (root/'TRAIN.local.json').is_file()
assert subprocess.check_output(['findmnt','-n','-T',str(root),'-o','FSTYPE'],text=True).strip() in ('ext4','xfs','btrfs')
plan.mkdir(mode=0o700)
for name, data in json.load(sys.stdin).items():
    assert name in ('SCORING-TIMEOUT-PLAN.json','launch_retained_hf_fsdp.py','recover_training_timeout.py')
    with (plan/name).open('xb') as f:
        f.write(base64.b64decode(data,validate=True)); f.flush(); os.fsync(f.fileno())
for directory in (plan,root):
    fd=os.open(directory,os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)
env=dict(os.environ,PYTHONPATH=str(root),PYTHONDONTWRITEBYTECODE='1')
with (plan/'launch.log').open('xb') as log:
    proc=subprocess.Popen(['python3','-B',str(plan/'launch_retained_hf_fsdp.py'),
        '--config',str(root/'TRAIN.local.json'),'--recover-scoring-timeout',str(plan/'SCORING-TIMEOUT-PLAN.json')],
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,cwd=root,env=env,start_new_session=True)
print(proc.pid)'''
    request = json.dumps({k:base64.b64encode(v).decode() for k,v in files.items()})
    pid = subprocess.check_output([*ssh, shlex.join(['python3','-B','-c',deploy,str(target),slug])],
                                  input=request, text=True, timeout=45)
    process_marker = ('RECORDING-POPULATION-RECOVERY-PROCESS.json' if population
                      else 'SCORING-RECOVERY-PROCESS.json')
    write(directory/process_marker, dict(submission_ref=pin(marker), remote_pid=int(pid)))
    return row


def train(config, state, directory, guard):
    """A crash can re-read a closed return but never launch a second job."""
    from iteration import Pending
    from transport import pull_stream, restore_locations, unpack_closed

    spec = settings(config)
    directory.mkdir(parents=True, exist_ok=True)
    template = copy.deepcopy(read(spec['stage_template_ref']))
    require(template['devices'] == list(range(8))
            and not template.get('stage_only') and not template.get('check_only'),
            'formal launch must use all eight devices and execute')
    opts=spec.get('continuous_actor')
    if opts or state['resume_ref'] == config['initial_resume_ref']:
        # The sealed policy replica and the original optimizer checkpoint live
        # at different paths. The retained stager verifies the exact commit hash.
        prior = Path(spec['initial_actor_run'])
        require(prior.is_absolute() and '..' not in prior.parts,
                'configured physical recovery run required')
    else:
        prior = recovery_run(state['resume_ref'], state['step'])
    name = f"{config['run_id']}-actor-{state['cohort_sequence']:06d}"
    require(re.fullmatch(r'[a-z0-9-]+', name), 'invalid strict actor run name')
    target = prior.with_name(name)
    call=None
    if opts:
        from resident_transport import reserve_call
        require(opts['protocol']=='formal-resident-v1' and opts['run_id']==config['run_id'] and
                config['formal_rollout'].get('continuous_training') and not os.environ.get('OMNI_TIMESHARED_ROLLOUT_REF')
                and not os.environ.get('OMNI_FORMAL_MEASURED_BATCH_REF') and template.get('actor_batch')
                and template['generation']['runtime'].get('reference_batch'),'ambiguous or incomplete continuous deployment')
        call=reserve_call(Path(config['output_root'])/'formal-resident-client',directory,opts,lambda:dict(target=str(target)))
        template['continuous_actor']=opts
        template['resident_owner_hot']=str(Path(call['target'])/'hot')
        template['capture_mounts']['resident_input_root']=str(Path(call['target']).with_name(Path(call['target']).name+'-inputs'))
    capture = state['capture']
    generation = template['generation']
    generation.update(pending_capture_ref=pin(capture['pending_capture_ref']),
                      generation_inventory_ref=pin(capture['generation_inventory_ref']))
    inventory = read(generation['generation_inventory_ref'])
    require(inventory['closed'] is True
            and inventory['generation_closed_ref'] == pin(capture['generation_closed_ref'])
            and read(capture['input_ref'])['sampled_window_ref'] == state['sampled_pending_window_ref'],
            'strict actor needs the current sampled window and its closed generation')
    template.update(run=name, source_bundle=str(prior), restore_run=str(prior),
                    prior_step=state['step'], expected_boot=spec['source_boot'],
                    restore_checkpoint_sha256=state['resume_ref']['sha256'], formal_sampled=True,
                    rollout_lanes=spec['rollout_lanes'], zero_result_refs=state.get('zero_result_refs', []))
    if os.environ.get('OMNI_TIMESHARED_ROLLOUT_REF'):
        from timeshared_rollout import deployment
        template['rollout_lanes']=[deployment(config)['lane']]
    measured_ref=os.environ.get('OMNI_FORMAL_MEASURED_BATCH_REF')
    adopted=None
    if measured_ref:
        binding=read(json.loads(measured_ref))
        require(binding['iteration_ref']==pin(config['_path']) and binding['source_boot']==spec['source_boot'],
                'measured batch belongs to another iteration or boot')
        profile=read(binding['selection_ref'])
        require(profile['training_rows']==config['sampled_training']['queries_per_cohort']*
                read(state['learning_ref'])['kernel']['G'],'measured batch does not cover the reserved cohort')
        spec=dict(spec,launcher_ref=pin(binding['launcher_ref']))
        template.update(actor_batch=profile['actor_batch'],runtime_release=read(binding['runtime_ref']),
                        measured_driver_ref=pin(binding['measured_driver_ref']),batch_selection_ref=pin(binding['selection_ref']))
        if state['cohort_sequence']==binding['sequence']:
            adopted=read(binding['staged_request_ref'])
            require(pin(directory/'SUBMIT-INTENT.json')==binding['original_intent_ref'] and
                    state['resume_ref']['sha256']==profile['predecessor_sha256'] and
                    adopted['closed_score_replay']['failed_run']==str(target) and
                    adopted['closed_score_replay']['cohort_sha256']==profile['cohort_sha256'] and
                    all(adopted[k]==template[k] for k in ('devices','restore_run','generation','prior_step',
                        'expected_boot','lineage','restore_checkpoint_sha256','formal_sampled','rollout_lanes',
                        'actor_batch','runtime_release','measured_driver_ref','batch_selection_ref')),
                    'cannot adopt a different generation, policy or training profile')
            target=target.with_name(adopted['run'])
    template['capture_mounts']['formal_input_root'] = str(target.with_name(target.name + '-input'))
    request_path = directory/'TRAIN-REQUEST.json'
    timeout_policy = scoring_timeout_policy(config, spec)
    if timeout_policy:
        prior_intent = directory/'SUBMIT-INTENT.json'
        if prior_intent.exists():
            template['timeout_seconds'] = read(read(prior_intent)['training_ref'])['timeout_seconds']
        elif request_path.exists():
            template['timeout_seconds'] = read(request_path)['timeout_seconds']
        else:
            template['timeout_seconds'] = max(template['timeout_seconds'], timeout_policy['timeout_seconds'])
    capture_root_ref = os.environ.get('OMNI_FORMAL_ACTOR_CAPTURE_ROOT_REF')
    if capture_root_ref and not request_path.exists():
        binding = read(json.loads(capture_root_ref))
        root = Path(binding['capture_root'])
        require(binding['iteration_ref'] == pin(config['_path'])
                and binding['boot'] == spec['source_boot'] and root.parent == target.parent
                and binding['overlays'] == template['capture_mounts']['overlays']
                and binding['tools'] == template['capture_mounts']['tools'],
                'retained actor capture assets differ from the training configuration')
        template['capture_mounts']['root'] = str(root)
    mount_ref = os.environ.get('OMNI_FORMAL_ACTOR_MOUNTS_REF')
    if mount_ref and not adopted:
        mount_ref = json.loads(mount_ref)
        extension = read(mount_ref)
        require(extension['schema'] == 'formal-actor-input-mounts-v1'
                and extension['iteration_ref'] == pin(config['_path']),
                'actor input mount extension belongs to another iteration')
        mounts = template['capture_mounts']
        roots = mounts['logical_roots']
        additions = extension['roots']
        if os.environ.get('OMNI_TIMESHARED_ROLLOUT_REF'):
            from timeshared_rollout import actor_inputs
            additions=dict(additions,**actor_inputs(config,state))
            mount_ref=write(directory/'TIMESHARED-MOUNTS.json',dict(iteration_ref=extension['iteration_ref'],
                            checkpoint=state['resume_ref'],roots=additions))
        require(additions and not roots.keys() & additions.keys(), 'overlapping actor input roles')
        if request_path.exists():
            require(read(request_path) == template, 'prior unsubmitted training request changed')
        mounts['logical_roots'].update(additions)
        mounts['formal_input_roles'].extend(sorted(additions))
        request_path = directory/f"TRAIN-REQUEST-mounts-{mount_ref['sha256']}.json"
        if not request_path.exists():
            require(not any((directory/name).exists() for name in ('SUBMIT-INTENT.json', 'INPUT-INTENT.json',
                    'INPUT-CLOSED.json', 'actor-input.tar')), 'prior actor input transfer or submission needs reconciliation')
    if adopted:
        template=adopted
        request_path=directory/'MEASURED-BATCH-REQUEST.json'
    staged = write(request_path, template)
    host = template['host']
    write(directory/'REQUEST.json', dict(run=dict(host=host, source_boot=spec['source_boot'],
          checkpoint_cache_root=config['checkpoint_cache_root'],
          closed_pull_timeout_seconds=spec['return_timeout_seconds'])))
    if call is not None:return resident_train(config,state,directory,guard,template,staged,call)
    lease = directory/('MEASURED-BATCH-SUBMIT.json' if adopted else 'SUBMIT-INTENT.json')
    returned = directory/'returned'
    if returned.is_dir():
        restore_locations(returned)
    elif (directory/'closed-return.tar').exists():
        unpack_closed(directory/'closed-return.tar', returned)
    else:
        if adopted and not lease.exists():
            # The new actor already owns the original device guard. Adopt only
            # the witnessed container and its config; never submit it again.
            witness=read(binding['worker_witness_ref'])
            raw=subprocess.check_output(['ssh','-oBatchMode=yes',host,shlex.join(
                ['sha256sum',str(target/'TRAIN.local.json'),str(target/'hot/CONTAINER.json')])],text=True)
            require([line.split()[0] for line in raw.splitlines()]==witness['source_digests']
                    and witness['target']==str(target) and witness['boot']==spec['source_boot'],
                    'launched actor identity changed')
            write(lease,dict(training_ref=staged,launcher_ref=spec['launcher_ref'],target=str(target),
                  source_boot=spec['source_boot'],original_intent=pin(directory/'SUBMIT-INTENT.json'),
                  observed_at=witness['observed_at'],adopted_container=witness['container']))
        if not lease.exists():
            launcher = pin(spec['launcher_ref'])
            # Input custody is established before the launcher records submission.
            from formal_actor_inputs import stage
            inputs = stage(template, config, directory)
            write(lease, dict(training_ref=staged, launcher_ref=launcher,
                              input_custody_ref=inputs, target=str(target),
                              source_boot=spec['source_boot'], automatic_retry=False,
                              observed_at=time.time()))
            argv = [sys.executable, '-B', launcher['path'], '--config', staged['path']]
            with (directory/'submit.log').open('ab') as log:
                try:
                    proc = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT,
                                          timeout=spec['submission_timeout_seconds'])
                except subprocess.TimeoutExpired:
                    # Submission may have reached the original host.
                    proc = None
            write(directory/'SUBMIT-EXIT.json', dict(returncode=None if proc is None else proc.returncode))
        intent = read(lease)
        require(intent['training_ref'] == staged and intent['target'] == str(target)
                and intent['source_boot'] == spec['source_boot'], 'strict actor submission identity changed')
        hot = target/'hot'
        argv = ['python3', '-B', str(target/'pull_closed.py'), str(hot/'forward-return'),
                str(spec['return_max_bytes']), '--formal-actor']
        from storage import storage
        exit_marker = directory/'SUBMIT-EXIT.json'
        submit_failed = not adopted and exit_marker.is_file() and read(exit_marker)['returncode'] not in (0, None)
        if submit_failed:
            submit_failed = not observe_recovered_actor(template, target, pin(lease), directory)
        wait_deadline = intent['observed_at'] + template['timeout_seconds'] + spec['submission_timeout_seconds']
        retry_marker = directory/'SCORING-RECOVERY-SUBMIT.json'
        retry = read(retry_marker) if retry_marker.is_file() and not adopted else None
        population_marker = directory/'RECORDING-POPULATION-RECOVERY-SUBMIT.json'
        population_retry = read(population_marker) if population_marker.is_file() and not adopted else None
        if population_retry:
            retry = population_retry
        if retry:
            require(retry['original_intent'] == pin(lease) and retry['target'] == str(target),
                    'scoring recovery belongs to a different actor')
            wait_deadline = retry['submitted_at'] + retry['plan']['timeout_seconds'] + spec['submission_timeout_seconds']
        while True:
            remaining = wait_deadline - time.time()
            storage.check_paths({'formal_actor_return': directory},
                                min_free_bytes=20*1024**3 + spec['return_max_bytes'])
            try:
                # Read-only observation is allowed even after the original deadline.
                observed = pull_stream(directory, argv, guard.fileno(),
                                       time.time()+spec['return_timeout_seconds'])
                code = observed.returncode
            except (subprocess.TimeoutExpired, OSError):
                code = 255
            if code == 0:
                unpack_closed(directory/'closed-return.tar', returned)
                break
            if code == 3:
                if adopted:raise ValueError('measured formal actor exited with failure: '+str(hot/'EXIT.json'))
                if timeout_policy and retry is None:
                    retry = submit_scoring_recovery(template, target, pin(lease), directory, timeout_policy)
                    if retry:
                        wait_deadline = retry['submitted_at'] + retry['plan']['timeout_seconds'] + spec['submission_timeout_seconds']
                        continue
                population_repair = os.environ.get('OMNI_FORMAL_RECORDING_POPULATION_REPAIR_REF')
                if population_repair and timeout_policy and not population_retry:
                    proof = read(json.loads(population_repair))
                    if proof['original_intent'] == pin(lease):
                        native = read(generation['pending_capture_ref'])
                        consumer = read(native['consumer_config_ref'])
                        population_ref = read(consumer['recording_map_ref'])['population_ref']
                        require(proof['schema'] == 'formal-recording-population-repair-v1'
                                and population_ref == proof['population_ref'], 'recording population recovery proof differs')
                        population_retry = submit_scoring_recovery(template, target, pin(lease),
                            directory, timeout_policy, population_proof=proof)
                        if population_retry:
                            retry = population_retry
                            wait_deadline = retry['submitted_at'] + retry['plan']['timeout_seconds'] + spec['submission_timeout_seconds']
                            continue
                # The old terminal may remain visible briefly while the same
                # guard is acquired. A new terminal is not eligible to retry.
                if not retry or time.time() > retry['submitted_at'] + min(spec['submission_timeout_seconds'], 300):
                    raise ValueError('strict actor exited with failure: '+str(hot/'EXIT.json'))
                code = 75
            if submit_failed or remaining <= 0:
                raise Pending('reconcile original strict actor intent: '+str(lease))
            require(code in (2, 75, 255), 'failed to validate strict actor return')
            time.sleep(min(spec['terminal_poll_seconds'], max(0, remaining)))
    wire = read(returned/'WIRE-MANIFEST.json')
    manifest = wire['files']
    closed = read(manifest['files/'+wire['closure']['closed_path'].lstrip('/')])
    result = read(manifest['RETURN.json'])
    accepted = read(result['result_ref'])
    require(Path(wire['closure']['closed_path']) == target/'hot/forward-return/CLOSED.json'
            and closed['schema'] == 'omni-formal-actor-return-v1'
            and closed['source_boot'] == spec['source_boot']
            and accepted['formal_rl_enabled'] is True
            and accepted['sampled_window_ref'] == state['sampled_pending_window_ref']
            and accepted['generation_inventory_ref'] == generation['generation_inventory_ref']
            and accepted['optimizer_steps'] == result['optimizer_steps'],
            'strict actor return does not consume the requested generation')
    return result


def submit_command(config, template, owner, command, directory):
    """Only the first command starts an actor; subsequent commands have no launcher."""
    from dependencies import local_path
    spec=config['formal_actor'];opts=template['continuous_actor']
    source=pin(opts['command_source_ref'])
    script=local_path(source['path']).read_text()
    args=['python3','-B','-c',script,str(owner),opts['run_id'],spec['source_boot']]
    intent=write(directory/'COMMAND.json',command)
    reply=subprocess.run(['ssh','-oBatchMode=yes','-oConnectTimeout=15',template['host'],shlex.join(args)],
                         input=json.dumps(command),capture_output=True,text=True,timeout=spec['submission_timeout_seconds'],check=True)
    ack=json.loads(reply.stdout)
    require(ack['sequence']==command['sequence'] and ack['source_boot']==spec['source_boot']
            and ack['run_id']==opts['run_id'] and re.fullmatch('[0-9a-f]{64}',ack['sha256']),
            'resident command acknowledged by a different owner')
    write(directory/'COMMAND-ACK.json',dict(command_ref=intent,remote=ack))


def resident_train(config,state,directory,guard,template,staged,call):
    from iteration import Pending
    from formal_actor_inputs import stage
    from transport import pull_stream,restore_locations,unpack_closed
    spec=config['formal_actor'];opts=template['continuous_actor'];seq=call['sequence'];owner=Path(call['target'])
    gen=template['generation'];lease=directory/'SUBMIT-INTENT.json';returned=directory/'returned'
    root=Path(config['output_root'])/'formal-resident-client';hot=owner/'hot'
    coord=dict(run_id=opts['run_id'],sequence=seq,checkpoint_sha256=state['resume_ref']['sha256'],
               pending_capture_ref=gen['pending_capture_ref'],generation_inventory_ref=gen['generation_inventory_ref'])
    if not returned.exists() and not (directory/'closed-return.tar').exists():
        if not lease.exists():
            inputs=stage(template,config,directory)
            if seq:
                previous=read(root/'completed'/f'{seq-1:06d}.json')
                require(previous['owner']==str(owner) and previous['source_boot']==spec['source_boot']
                        and previous['checkpoint_sha256']==coord['checkpoint_sha256'],'resident predecessor differs')
                cmd=dict(coord,action='update',previous_close_sha256=previous['closure_sha256'])
            else:
                require(state['resume_ref']==config['initial_resume_ref'] and owner.name==template['run'],
                        'initial resident recovery mismatch')
                cmd=None
            write(lease,dict(training_ref=staged,input_custody_ref=inputs,target=str(owner),sequence=seq,
                source_boot=spec['source_boot'],command=cmd,observed_at=time.time()))
            if not seq:
                # No retry of an unknown process launch. Nonzero command
                # writes are idempotent and never launch or retry optimizer work.
                with (directory/'submit.log').open('ab') as log:
                    try:
                        proc=subprocess.run([sys.executable,'-B',spec['launcher_ref']['path'],'--config',staged['path']],
                            stdout=log,stderr=subprocess.STDOUT,timeout=spec['submission_timeout_seconds'])
                    except subprocess.TimeoutExpired:proc=None
                write(directory/'SUBMIT-EXIT.json',dict(returncode=None if proc is None else proc.returncode))
        intent=read(lease)
        require(intent['training_ref']==staged and intent['target']==str(owner) and intent['sequence']==seq
                and intent['source_boot']==spec['source_boot'],'reconciliation targets different resident command')
        if seq and not (directory/'COMMAND-ACK.json').exists():
            submit_command(config,template,owner,intent['command'],directory)
        command_root=hot/'resident/results'/f'{seq:06d}'
        args=['python3','-B',str(owner/'pull_closed.py'),str(command_root),str(spec['return_max_bytes']),
              '--formal-resident','--owner='+opts['run_id']]
        deadline=min(opts['deadline_epoch'],intent['observed_at']+template['timeout_seconds']+spec['submission_timeout_seconds'])
        while True:
            try:code=pull_stream(directory,args,guard.fileno(),time.time()+spec['return_timeout_seconds']).returncode
            except (OSError,subprocess.TimeoutExpired):code=255
            if code==0:break
            if code==3:raise ValueError('resident owner exited without closing its reserved update')
            if code not in (75,255,2) or time.time()>=deadline:
                raise Pending('reconcile original resident command: '+str(lease))
            time.sleep(min(spec['terminal_poll_seconds'],max(0,deadline-time.time())))
    if returned.exists():restore_locations(returned)
    else:unpack_closed(directory/'closed-return.tar',returned)
    return accept_resident_result(state,gen,spec,opts,root,returned,owner,seq)


def accept_resident_result(state,gen,spec,opts,root,returned,owner,seq):
    wire=read(returned/'WIRE-MANIFEST.json');closure=wire['closure']
    require(Path(closure['closed_path'])==owner/'hot/resident/results'/f'{seq:06d}'/'CLOSED.json',
            'closed command is outside original owner')
    closed=read(wire['files']['files/'+closure['closed_path'].lstrip('/')])
    result=read(closed['return_ref']);accepted=read(result['result_ref']);ack=read(accepted['resident_intent_ref'])
    require(closed['schema']=='formal-resident-command-v1' and closed['immutable'] is True and
            closed['sequence']==seq and closed['run_id']==opts['run_id'] and closed['source_boot']==spec['source_boot']
            and result['result_ref']==closed['result_ref'] and ack['command']['sequence']==seq
            and ack['command']['checkpoint_sha256']==state['resume_ref']['sha256']
            and ack['command']['pending_capture_ref']==gen['pending_capture_ref']
            and ack['command']['generation_inventory_ref']==gen['generation_inventory_ref']==accepted['generation_inventory_ref']
            and accepted['formal_rl_enabled'] is True and accepted['sampled_window_ref']==state['sampled_pending_window_ref']
            and accepted['optimizer_steps']==result['optimizer_steps'] in (0,1),'wrong resident optimizer result')
    checkpoint=result.get('checkpoint_ref',state['resume_ref'])
    require(accepted['clock']['successful_optimizer_steps']==state['step']+result['optimizer_steps']
            and read(checkpoint)['step']==accepted['clock']['successful_optimizer_steps'],
            'resident Clock did not match reported update')
    require(result.get('policy_ref')==accepted.get('policy_ref') and result.get('checkpoint_ref')==accepted.get('checkpoint_ref')
            and result['trajectory_ref']==accepted['trajectory_ref'],'controller and actor disagree on return')
    if result['optimizer_steps']:
        policy=read(result['policy_ref']);publication=read(policy['publication_ref'])
        require(policy['step']==publication['step']==state['step']+1 and publication['checkpoint_ref']==checkpoint
                and publication['assets']==policy['assets'] and policy['version']=='checkpoint-sha256:'+policy['assets'][0]['sha256']
                and all(pin(asset)==asset for asset in policy['assets']),'export is not the committed successor')
    write(root/'completed'/f'{seq:06d}.json',dict(owner=str(owner),sequence=seq,source_boot=spec['source_boot'],
          checkpoint_sha256=checkpoint['sha256'],closure_sha256=closure['closed_sha256']))
    return result


def end_resident(config,guard):
    from iteration import Pending
    spec=config['formal_actor'];opts=spec['continuous_actor']
    root=Path(config['output_root'])/'formal-resident-client'
    if not (root/'OWNER.json').exists():return
    done=root/'END-COMPLETE.json'
    if done.exists():return read(done)
    owner=Path(read(root/'OWNER.json')['target']);template=read(spec['stage_template_ref'])
    template=dict(template,continuous_actor=opts)
    completed=sorted((root/'completed').glob('*.json'))
    require(completed and [int(p.stem) for p in completed]==list(range(len(completed))),
            'reconcile resident commands before end')
    seq=len(completed)
    require(seq<opts['max_commands'] and not (root/'calls'/f'{seq:06d}.json').exists(),'unclosed command before end')
    cmd=dict(action='end',run_id=opts['run_id'],sequence=seq,previous_close_sha256=read(completed[-1])['closure_sha256'])
    folder=root/'end'
    submit_command(config,template,owner,cmd,folder)
    code="""import json,sys
from pathlib import Path
h=Path(sys.argv[1]);p=h/'EXIT.json'
if not p.exists():sys.exit(75)
exit=json.loads(p.read_bytes());end=json.loads((h/'resident/END.json').read_bytes())
assert exit['exit_code']==0 and not exit['failure'] and exit['complete_ranks']==8
assert exit['container_id']==json.loads((h/'CONTAINER.json').read_bytes())['container_id']
print(json.dumps(end))"""
    args=shlex.join(['python3','-B','-c',code,str(owner/'hot')])
    until=time.monotonic()+spec['return_timeout_seconds']
    while time.monotonic()<until:
        try:r=subprocess.run(['ssh','-oBatchMode=yes','-oConnectTimeout=15',template['host'],args],
                            capture_output=True,text=True,timeout=spec['return_timeout_seconds'])
        except subprocess.TimeoutExpired:break
        if r.returncode==0:
            end=json.loads(r.stdout)
            require(end['run_id']==opts['run_id'] and end['sequence']==seq
                    and end['previous']['sha256']==cmd['previous_close_sha256'],'wrong resident end')
            return write(done,end)
        require(r.returncode in (75,255),'resident end could not be verified')
        time.sleep(spec['terminal_poll_seconds'])
    raise Pending('observe original resident end: '+str(folder))

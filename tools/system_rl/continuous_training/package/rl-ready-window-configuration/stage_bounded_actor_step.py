"""Stage a new attempt under the existing physical-device guard; never replace prior outcomes."""
import argparse,hashlib,json,re,shlex,subprocess,tempfile
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
import sys
import tarfile
import time

from launch_retained_hf_fsdp import training_memory_bytes


@contextmanager
def measure(timings, name):
    start = time.monotonic()
    try:
        yield
    finally:
        timings[name] = time.monotonic() - start


def read_predecessor(remote, prior, step, world, command=None):
    base = Path('hot')
    if command is not None:
        seq = command['sequence']
        if type(seq) is not int or seq < 0 or Path(command['owner']) != prior:
            raise ValueError('wrong resident command source')
        base = base/'resident/results'/f'{seq:06d}'
    names = ['TRAIN.local.json', 'hot/EXIT.json', f'{base}/step-{step}/COMMITTED.json',
             *(f'{base}/rank-{rank}/RESULT.json' for rank in range(world))]
    if command is not None:
        names += [str(base/'CLOSED.json'),str(base/'RESULT.json')]
    command = shlex.join(['tar', '-cf', '-', '-C', str(prior), '--', *names])
    raw = remote(command, text=False)
    with tarfile.open(fileobj=BytesIO(raw), mode='r:') as archive:
        roster = archive.getmembers()
        if [m.name for m in roster] != names or any(not m.isfile() for m in roster):
            raise ValueError('unexpected predecessor receipt roster')
        return [archive.extractfile(member).read() for member in roster]


def stage_continuation(c, d, dest, remote, *, snapshots=None, timings):
    """Continue from a terminal run using its unchanged worker and optimizer."""
    q=shlex.quote
    initial=c.get('initial_actor')
    prior=Path(c['source_bundle'] if initial else c['restore_run'])
    if not prior.is_absolute() or prior.parent!=dest.parent or '..' in prior.parts or prior==dest:
        raise ValueError('invalid continuation source')
    if any(k in c for k in ('trace_update','efficiency','updated_http_parity','published_adapter_control')):
        raise ValueError('continuation cannot be combined with a diagnostic')
    step=c['prior_step']
    if initial:
        if (type(step) is not int or step!=0 or not c.get('formal_sampled') or not c.get('continuous_actor')
                or 'restore_run' in c or 'restore_checkpoint_sha256' in c or 'batch_selection_ref' in c):
            raise ValueError('weight initialization cannot resume a checkpoint or migrate a measured batch')
        with measure(timings, 'predecessor_receipts'):
            names=['TRAIN.local.json','hot/EXIT.json']
            chunk=remote(shlex.join(['tar','-cf','-','-C',str(prior),'--',*names]),text=False)
            with tarfile.open(fileobj=BytesIO(chunk)) as archive:
                if archive.getnames()!=names or not all(m.isfile() for m in archive):
                    raise ValueError('invalid source actor receipts')
                old,terminal=(json.load(archive.extractfile(n)) for n in names)
        verify_started=time.monotonic()
        if (terminal['exit_code']!=0 or terminal['failure'] or terminal['complete_ranks']!=len(c['devices'])
                or old['boot_id']!=c['expected_boot'] or old['devices']!=c['devices']
                or any(old[k]!=initial[k] for k in ('image','training','devices'))
                or old['files']['BASE-ASSETS.json']!=initial['base']['asset_manifest_ref']['sha256']):
            raise ValueError('initialization source is not the qualified actor')
        # Retain only production dependencies. The source may also contain
        # actions, diagnostic packets and an unrelated scoring configuration.
        inherited=('BASE-ASSETS.json','dispatch_retained_hf_boundary.py','actor_arithmetic.py',
                   'retained_router_candidate.py','retained_kv_candidate.py','retained_expert_candidate.py',
                   'policy_identity.py','worker_dp8.py','rank_profile.py')
        s={k:old[k] for k in ('image','training','devices','boot_id','existing_guard',
                              'communication','shared_root','source_pins')}
        s.update(files={n:old['files'][n] for n in inherited},name=c['run'],hot=str(dest/'hot'),
                 initial_actor=initial,lineage=c['lineage'],deterministic_backward=True,
                 dataset_sha256=c.get('dataset_sha256'),training_memory_gib=c['training_memory_gib'],
                 timeout_seconds=c['timeout_seconds'],storage_budget_bytes=c['storage_budget_bytes'],
                 continuation=dict(prior_step=step))
        proof_sha=None
    else:
        if type(step) is not int or step<1:raise ValueError('positive prior step required')
        command=c.get('restore_command')
        base=prior/'hot'
        if command is not None:
            base=base/'resident/results'/f"{command['sequence']:06d}"
        checkpoint=base/f'step-{step}'
        with measure(timings, 'predecessor_receipts'):
            old_raw, terminal_raw, proof_raw, *owner_raw = read_predecessor(remote, prior, step, 8, command)
            old = json.loads(old_raw)
            terminal = json.loads(terminal_raw)
        verify_started = time.monotonic()
        if command is not None:
            closed_raw,result_raw=owner_raw[8:]
            owner_raw=owner_raw[:8]
            closed=json.loads(closed_raw);result=json.loads(result_raw)
            def matches(raw, ref, path):
                return ref==dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
            if (hashlib.sha256(closed_raw).hexdigest()!=command['closure_sha256'] or closed['immutable'] is not True
                    or closed['sequence']!=command['sequence'] or closed['source_boot']!=old['boot_id']
                    or command['checkpoint_sha256']!=c['restore_checkpoint_sha256']
                    or not matches(result_raw,closed['result_ref'],base/'RESULT.json')
                    or not matches(proof_raw,result['checkpoint_ref'],checkpoint/'COMMITTED.json')
                    or result['optimizer_steps']!=1 or len(result['rank_refs'])!=8
                    or not all(matches(raw,ref,base/f'rank-{i}/RESULT.json')
                               for i,(raw,ref) in enumerate(zip(owner_raw,result['rank_refs'])))
                    or not terminal.get('container_id') or type(terminal['exit_code']) is not int):
                raise ValueError('resident checkpoint is not the closed predecessor command')
        elif terminal['exit_code']!=0 or terminal.get('failure') or terminal['complete_ranks']!=8:
            raise ValueError('predecessor is not terminal with eight complete ranks')
        proof_sha=hashlib.sha256(proof_raw).hexdigest()
        if proof_sha!=c['restore_checkpoint_sha256']:raise ValueError('checkpoint pin changed')
        proof=json.loads(proof_raw)
        if old.get('deterministic_backward') is not True or old['lineage']!=c['lineage'] or old['devices']!=c['devices']:
            raise ValueError('strict training source differs')
        if c.get('expected_boot') is not None and old['boot_id'] != c['expected_boot']:
            raise ValueError('strict training predecessor boot differs')
        if old['files']['bounded_actor_step.py']!=proof['contract']['sources']['bounded_actor_step.py']:
            raise ValueError('predecessor driver does not match committed source')
        owner_result=[json.loads(raw) for raw in owner_raw]
        if any(r.get('rank')!=i or r.get('step')!=step or not r.get('recoverable_checkpoint')
               or r.get('publication_sha256')!=proof['resident_identity']['publication']['identity_sha256']
               for i,r in enumerate(owner_result)):raise ValueError('terminal rank/checkpoint identity differs')
        s=dict(old,name=c['run'],hot=str(dest/'hot'),dataset_sha256=c.get('dataset_sha256'),
               training_memory_gib=c['training_memory_gib'],
               restore_checkpoint=dict(directory=str(checkpoint),sha256=proof_sha),
               continuation=dict(prior_step=step,predecessor_driver_sha256=proof['contract']['sources']['bounded_actor_step.py']),
               timeout_seconds=c['timeout_seconds'],storage_budget_bytes=c['storage_budget_bytes'])
    if initial and 'publication_norm_atol' in c:
        s['publication_norm_atol'] = c['publication_norm_atol']
    if type(s['timeout_seconds']) is not int or s['timeout_seconds']<=0 or s['storage_budget_bytes']<2*1024**3:
        raise ValueError('invalid per-run duration/storage envelope')
    if not initial:
        s.pop('initial_actor',None)
        s.pop('restore_command',None)
        if command is not None:s['restore_command']=command
    s.pop('trace_update',None)
    s['files']=dict(s['files'])
    s['files'].pop('finalized_generation',None)
    s.pop('generation',None)
    s.pop('capture_mounts',None)
    s.pop('configuration_sha256',None)
    s.pop('formal_sampled',None)
    s.pop('actor_batch',None)
    s.pop('batch_migration',None)
    s.pop('closed_score_replay',None)
    if c.get('formal_sampled') is True:
        if 'generation' not in c:
            raise ValueError('formal continuation requires a sealed generation')
        lanes = c['rollout_lanes']
        if not isinstance(lanes, list) or len(lanes) not in (1, 2) or len(set(lanes)) != len(lanes):
            raise ValueError('one or two distinct formal rollout lanes are required')
        if 'actor_batch' in c:
            batch=c['actor_batch']
            if (set(batch)-{'microbatch_per_card','arithmetic','gradient_accumulation_steps'} or
                ('arithmetic' in batch and batch['arithmetic']!='single_clip_compact_qo_v1') or
                type(batch.get('gradient_accumulation_steps',1)) is not int or batch.get('gradient_accumulation_steps',1)<=0 or
                type(batch.get('microbatch_per_card')) is not int or batch['microbatch_per_card']<=0):
                raise ValueError('invalid per-card actor microbatch')
            s['actor_batch']=dict(batch)
        s['formal_sampled'] = True
        s['continuation']['zero_result_refs'] = c.get('zero_result_refs', [])
        s['continuation']['rollout_lanes'] = lanes
    elif 'zero_result_refs' in c:
        raise ValueError('zero-gradient continuation chain is only valid for formal training')
    overlay={name:d/name for name in ('bounded_actor_step.py','launch_retained_hf_fsdp.py')}
    if s.get('formal_sampled'):
        # Use the deployed training role named by the paired release. Its
        # filesystem layout can differ from the development source checkout.
        # require_release_sources below authenticates the bytes actually staged.
        role = Path(c['runtime_release']['roots']['training'])
        if not role.is_absolute():
            raise ValueError('absolute training release root required')
        overlay.update({
            'formal_actor_receipt.py': role/'formal_actor_receipt.py',
            'pull_closed.py': role/'pull_closed.py',
        })
    if c.get('continuous_actor'):
        s['continuous_actor']=c['continuous_actor']
        if (not c.get('formal_sampled') or not c.get('actor_batch') or (initial or old).get('actor_batch')!=c['actor_batch']
                or s['training_memory_gib']<1024 or not c['generation']['runtime'].get('reference_batch')):
            raise ValueError('continuous actor requires original accepted batch geometry and reference batching')
        overlay.update({name:d/'source'/name for name in ('resident.py','formal_actor_receipt.py','pull_closed.py','continuous_policy.py')})
        if initial:overlay['actor_start.py']=d/'source/actor_start.py'
        overlay['publication.py']=d.parent/'implementation/publication.py'
    if 'actor_batch' in s:
        overlay['events.py']=d/'source/events.py'
        overlay['retained_attention_candidate.py']=d/'retained_attention_candidate.py'
    geometry=c.get('lora_row_geometry',old.get('lora_row_geometry'))
    if geometry is not None or 'lora_row_geometry' in c:
        from source.lora_row_precision import validate
        if not isinstance(geometry,dict) or 'generation' not in c:
            raise ValueError('LoRA arithmetic continuation requires a sealed generation')
        validate(geometry)
        if geometry.get('algorithm')!='grouped_bf16':
            raise ValueError('unsupported training LoRA arithmetic')
        overlay['lora_row_precision.py']=d/'source/lora_row_precision.py'
        s['lora_row_geometry']=geometry
    if 'generation' in c:
        s['generation']=c['generation']
        s['capture_mounts']=c['capture_mounts']
        from capture_runtime import capture_runtime
        runtime=capture_runtime(s)
        if runtime.get('first_wave_forward'):
            for name in ('first_wave_forward.py','microbatch_producer.py','microbatch_fields.py','microbatch_roster.py','policy_row_assignment.py'):
                overlay[name]=d/'source'/name
        overlay.update({name:d/name for name in ('pack_actor_fields.py','retained_hf_fsdp_control.py',
                                                'capture_runtime.py')})
        s['source_pins']=dict(old['source_pins'])
        if not initial:s['continuation']['predecessor_control_sha256']=proof['contract']['sources']['retained_hf_fsdp_control.py']
        s['accepted_fields']='finalized_generation'
        s['files']['finalized_generation']=c['generation']['generation_inventory_ref']['sha256']
    else:
        overlay['accepted.pt']=Path(c['accepted_fields'])
    payloads={name:path.read_bytes() for name,path in overlay.items()}
    if snapshots:
        s['configuration_sha256']={name:hashlib.sha256(raw).hexdigest() for name,raw in snapshots.items()}
    for name,payload in payloads.items():
        s['files'][name]=hashlib.sha256(payload).hexdigest()
    if c.get('batch_selection_ref'):
        ref=c['batch_selection_ref'];raw=Path(ref['path']).read_bytes()
        if len(raw)!=ref['bytes'] or hashlib.sha256(raw).hexdigest()!=ref['sha256']:
            raise ValueError('measured training batch receipt changed')
        measured=json.loads(raw)
        if (not s.get('formal_sampled') or measured['schema']!='actor-batch-measurement-selection-v1'
            or s.get('actor_batch')!=measured['actor_batch'] or s['devices']!=measured['devices']
            or measured['training_rows']!=len(s['devices'])*s['actor_batch']['microbatch_per_card']*s['actor_batch'].get('gradient_accumulation_steps',1)
            or any(s['files'].get(n)!=measured['sources'][n] for n in (
                'events.py','retained_attention_candidate.py','retained_hf_fsdp_control.py',
                'pack_actor_fields.py'))):
            raise ValueError('measured batch does not identify staged actor')
        driver='bounded_actor_step.py'
        if s['files'][driver]!=measured['sources'][driver]:
            import ast
            probe_path=Path(c['measured_driver_ref']['path'])
            tested=probe_path.read_bytes()
            if hashlib.sha256(tested).hexdigest()!=measured['sources'][driver] or len(tested)!=c['measured_driver_ref']['bytes'] or c['measured_driver_ref']['sha256']!=measured['sources'][driver]:
                raise ValueError('measured driver copy changed')
            def compute_tree(source):
                tree=ast.parse(source)
                match=[f for f in tree.body if isinstance(f,ast.FunctionDef) and f.name=='verify_continuation']
                if len(match)!=1:raise ValueError('missing continuation verifier')
                tree.body.remove(match[0])
                return ast.dump(tree)
            if compute_tree(tested)!=compute_tree(payloads[driver]):
                raise ValueError('measured executable changed outside checkpoint admission')
        if proof_sha==measured['predecessor_sha256']:
            s['batch_migration']=dict(prior_sha256=proof_sha, selection_sha256=ref['sha256'],
                predecessor_sources=measured['predecessor_sources'],candidate_sources=dict(measured['sources'],**{driver:s['files'][driver]}))
            if 'closed_score_replay' in c:
                replay=c['closed_score_replay']
                if replay['cohort_sha256']!=measured['cohort_sha256']:
                    raise ValueError('score replay differs from measured cohort')
                s['closed_score_replay']=dict(replay)
        elif old.get('actor_batch')!=s['actor_batch']:
            raise ValueError('new batch geometry requires its measured predecessor')
    if 'generation' in c:
        s['source_pins']['retained_hf_fsdp_control.py']=s['files']['retained_hf_fsdp_control.py']
    from bounded_actor_step import verify_continuation
    sys_path=d.parent/'implementation'
    sys.path.insert(0,str(sys_path))
    try:
        if initial:
            from source.actor_start import check_spec
            check_spec(s)
        else:verify_continuation(s,proof)
    finally:sys.path.remove(str(sys_path))
    if 'generation' in c:
        from verify_runtime_sources import require_release_sources
        if not c.get('runtime_release'):
            raise ValueError('real rollout continuation requires a paired runtime_release configuration')
        effective = dict(s['files'])
        for name, digest in s['source_pins'].items():
            if name in effective and digest != effective[name]:
                raise ValueError('conflicting prepared source pin: '+name)
            effective[name] = digest
        if s.get('continuous_actor'):
            from verify_runtime_sources import continuous_release
            receipt=continuous_release(c['runtime_release'],s['continuous_actor'],effective,s['files']['lora_row_precision.py'])
        else:receipt = require_release_sources(c['runtime_release'], effective)
        if geometry is not None:
            expected=s['files']['lora_row_precision.py']
            for role in ('training','rollout'):
                matches=[r for r in receipt['components'] if r['role']==role and
                         Path(r['component']).name=='lora_row_precision.py']
                if len(matches)!=1 or matches[0]['observed']!=expected:
                    raise ValueError('release does not bind paired LoRA arithmetic: '+role)
        snapshots = dict(snapshots or {}, **{'RUNTIME-SOURCES.json':json.dumps(receipt,indent=2).encode()})
        s.setdefault('configuration_sha256',{})['RUNTIME-SOURCES.json']=hashlib.sha256(snapshots['RUNTIME-SOURCES.json']).hexdigest()
    timings['config_and_source_checks'] = time.monotonic() - verify_started
    if c.get('check_only') is True:
        print(json.dumps(dict(status='continuation_contract_checked',predecessor_step=step,
            checkpoint_sha256=proof_sha,optimizer_updates=0)))
        return
    # Never copy hot state or alter a sealed predecessor. Original launcher
    # still locks the same guard, checks boot/occupancy and mounts state RO.
    links=(set(s['files'])|{'tree','dispatch_retained_hf_boundary.py','retained_hf_fsdp_control.py'})-set(overlay)-{'finalized_generation'}
    payloads['TRAIN.local.json']=json.dumps(s,indent=2).encode()
    with measure(timings, 'stage_bundle'):
        buffer = BytesIO()
        with tarfile.open(fileobj=buffer, mode='w:') as archive:
            for name, payload in {**payloads, **(snapshots or {})}.items():
                if Path(name).name != name or name in ('', '.', '..'):
                    raise ValueError('invalid staged filename')
                member = tarfile.TarInfo(name)
                member.mode = 0o600
                member.size = len(payload)
                archive.addfile(member, BytesIO(payload))
        mkdir = shlex.join(['mkdir', '-m', '700', '--', str(dest)])
        link = shlex.join(['cp', '-al', '--', *(str(prior/n) for n in sorted(links)), str(dest)+'/'])
        unpack = shlex.join(['tar', '--keep-old-files', '-xmf', '-', '-C', str(dest)])
        remote(f'umask 077 && {mkdir} && {link} && {unpack}', text=False, input=buffer.getvalue())
    launcher='python3 -B launch_retained_hf_fsdp.py --config TRAIN.local.json'
    if 'generation' in c:
        with measure(timings, 'capture_preflight'):
            print(remote('cd '+q(str(dest))+' && '+launcher+' --check-capture').strip())
    if not c.get('stage_only'):
        with measure(timings, 'launch_submission'):
            remote('cd '+q(str(dest))+' && (nohup '+launcher+' >launch.log 2>&1 </dev/null & )')
    print(dest)


def stage(c, snapshots=None):
    training_memory_bytes(c)
    host=c['host']; tag=c['run']; q=shlex.quote
    if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_-]*',host) or not re.fullmatch(r'[a-z0-9-]+',tag):raise ValueError('invalid run or alias')
    parent=Path(c['source_bundle']); dest=parent.with_name(tag); d=Path(__file__).resolve().parent
    def remote(cmd, *, text=True, **kwargs):
        return subprocess.check_output(['ssh',host,cmd],text=text,**kwargs)
    if 'prior_step' in c:
        timings = {}
        completed = False
        try:
            with measure(timings, 'total'):
                result = stage_continuation(c,d,dest,remote,snapshots=snapshots,timings=timings)
            completed = True
            return result
        finally:
            timings['other'] = timings['total'] - sum(v for k,v in timings.items() if k!='total')
            # Keep the stdout handoff contract for existing callers.
            print(json.dumps(dict(schema='omni-continuation-submit-timing-v1', completed=completed,
                                  stage_only=bool(c.get('stage_only')), seconds=timings)), file=sys.stderr)
    if 'check_only' in c:raise ValueError('check_only requires a continuation')
    s=json.loads(remote('cat '+q(str(parent/'HF-ACTOR.local.json'))))
    if json.loads(remote('cat '+q(str(parent/'hot/EXIT.json'))))['exit_code']!=0:raise ValueError('incomplete reference run')
    from compose_actor_arithmetic import MODULES
    files={n:d/n for n in ('bounded_actor_step.py','retained_hf_fsdp_control.py','launch_retained_hf_fsdp.py')}
    files['events.py']=d/'source/events.py'
    files['policy_identity.py']=d/'source/policy_identity.py'
    files['accepted.pt']=Path(c['accepted_fields'])
    s.update(accepted_fields='accepted.pt',dataset_sha256=c['dataset_sha256'],name=tag,hot=str(dest/'hot'),
             devices=c['devices'],training_memory_gib=c['training_memory_gib'])
    if c.get('trace_update') is True:
        if 'restore_run' in c:raise ValueError('trace_update requires an isolated initial step')
        s['trace_update']=True
    if c.get('deterministic_backward') is True:
        if 'restore_run' in c:raise ValueError('initial-step precision choice cannot override a restored policy')
        s['deterministic_backward']=True
    restore_sources={}
    if c.get('recoverable'):
        if c['devices']!=list(range(8)):raise ValueError('recoverable first step requires the original world8 checkpoint topology')
        from source.actor_arithmetic import PROFILES
        s['training'].update(world_size=8,actor_arithmetic_profile=PROFILES[8],checkpoint_storage_mode='lora_training_state')
        s.update(storage_budget_bytes=c['storage_budget_bytes'],timeout_seconds=c['timeout_seconds'],
                 lineage=c['lineage'])
        if s['storage_budget_bytes']<2*1024**3:raise ValueError('checkpoint and runtime budget too small')
        files.update({n:d/n for n in MODULES})
        files['actor_arithmetic.py']=d/'source/actor_arithmetic.py'
        if 'restore_run' in c:
            run=Path(c['restore_run'])
            if run.parent!=parent.parent or run==dest:raise ValueError('invalid restore source')
            prior=json.loads(remote('cat '+q(str(run/'hot/EXIT.json'))))
            if prior.get('failure') or prior['exit_code'] or prior['complete_ranks']!=8:raise ValueError('restore source not closed')
            restore_step=c.get('restore_step',1)
            if type(restore_step) is not int or restore_step<1 or (restore_step!=1 and 'updated_http_parity' not in c):
                raise ValueError('explicit later step requires a no-update parity probe')
            proof=run/f'hot/step-{restore_step}/COMMITTED.json'
            raw=subprocess.check_output(['ssh',host,'cat '+q(str(proof))])
            if json.loads(raw)['step']!=restore_step:raise ValueError('wrong diagnostic checkpoint')
            if c.get('restore_checkpoint_sha256') and hashlib.sha256(raw).hexdigest()!=c['restore_checkpoint_sha256']:
                raise ValueError('diagnostic checkpoint pin changed')
            if c.get('expected_boot') and remote('cat /proc/sys/kernel/random/boot_id').strip()!=c['expected_boot']:
                raise ValueError('diagnostic training source boot changed')
            s['restore_checkpoint']=dict(directory=str(proof.parent),sha256=hashlib.sha256(raw).hexdigest())
            if restore_step!=1:
                s['parity_restore_step']=restore_step
                s['diagnostic_commit']=json.loads(raw)['contract']
            source_pins=json.loads(raw)['contract']['sources']
            # Restore the original training arithmetic even when the workspace
            # is preparing a different inference precision candidate.
            for name in (*MODULES,'actor_arithmetic.py','events.py'):
                path=run/name
                if remote('sha256sum -- '+q(str(path))).split()[0]!=source_pins[name]:
                    raise ValueError('checkpoint training arithmetic source differs: '+name)
                restore_sources[name]=path
                del files[name]
    if 'efficiency' in c:
        if not s.get('restore_checkpoint'):
            raise ValueError('efficiency probe must restore a closed P update')
        s['efficiency']=c['efficiency']
        files['actor_efficiency.py']=d/'actor_efficiency.py'
    if 'lora_row_geometry' in c:
        from source.lora_row_precision import validate
        cfg=c['lora_row_geometry']
        validate(cfg)
        if cfg.get('algorithm')!='grouped_bf16' or 'updated_http_parity' not in c or c.get('published_adapter_control'):
            raise ValueError('grouped updated parity control required')
        files['lora_row_precision.py']=d/'source/lora_row_precision.py'
        files['check_lora_row_vjp.py']=d/'check_lora_row_vjp.py'
        s['lora_row_geometry']=cfg
    if 'updated_http_parity' in c:
        if not s.get('restore_checkpoint') or 'efficiency' in c:
            raise ValueError('updated parity requires only the closed checkpoint probe')
        packet=json.loads(Path(c['updated_http_parity']).read_bytes())
        if packet.get('schema') in ('closed-rollout-probability-diagnostic-v1','closed-cohort-width-parity-v1'):
            files['retained_group_parity.py']=d/'retained_group_parity.py'
        if 'limits' not in packet:  # historical packet; use its frozen criterion
            limits_path=Path(c.get('parity_criterion',d.parent/'parity-criterion-v1/CRITERION.v1.json'))
            criterion=json.loads(limits_path.read_bytes())
            if not criterion['result_flip']['required']:raise ValueError('expected no-flip reference criterion')
            s['parity_limits']=dict(token=criterion['output']['token_limit'],sequence=criterion['output']['sequence_limit'],max_argmax_flips=0)
        s['updated_http_parity']='updated-http.json'
        files['updated_http_parity.py']=d/'updated_http_parity.py'
        files[s['updated_http_parity']]=Path(c['updated_http_parity'])
        if c.get('published_adapter_control') is True:
            s['published_adapter_control']=True
            if c.get('projection_family_shadow') is True:
                s['projection_family_shadow']=True
                files['q_projection_trace.py']=d.parent/'implementation/q_projection_trace.py'
            if c.get('trace_layer0') is True or c.get('trace_layer1') is True:
                s['trace_layer0']=c.get('trace_layer0') is True
                s['trace_layer1']=c.get('trace_layer1') is True
                files['retained_group_parity.py']=d/'retained_group_parity.py'
                if c.get('trace_q_operands') in (True,'o','q'):
                    s['trace_q_operands']=c['trace_q_operands']
                    s['operand_layer']=c.get('operand_layer',0)
                    if s['operand_layer'] not in ((0,1) if s['trace_layer1'] else (0,)):
                        raise ValueError('operand layer must be observed')
                    if 'q_replay_plan' in c:s['q_replay_plan']=c['q_replay_plan']
                    files['q_projection_trace.py']=d.parent/'implementation/q_projection_trace.py'
    elif 'published_adapter_control' in c:
        raise ValueError('published adapter control requires closed HTTP parity evidence')
    for n,p in files.items():
        (s['source_pins'] if n in s['source_pins'] else s['files'])[n]=hashlib.sha256(p.read_bytes()).hexdigest()
    # Keep installed implementation, worker and model unchanged; overlaid files are never written through hard links.
    for name,path in restore_sources.items():
        s['files'][name]=source_pins[name]
    link=set(s['files'])-set(files)-set(restore_sources)|{'tree','dispatch_retained_hf_boundary.py'}
    remote('mkdir -m 700 '+q(str(dest))+' && cp -al -- '+' '.join(q(str(parent/n)) for n in sorted(link))+' '+q(str(dest))+'/')
    if restore_sources:
        remote('cp -al -- '+' '.join(q(str(p)) for p in restore_sources.values())+' '+q(str(dest))+'/')
    with tempfile.TemporaryDirectory() as tmp:
        worker=Path('tree/parents/jev-system-rl/implementation/worker_dp8.py')
        old=remote('cat '+q(str(parent/worker)))
        expected=s['source_pins'][worker.name]
        if hashlib.sha256(old.encode()).hexdigest()!=expected:raise ValueError('worker source drift')
        start=old.index('   replaced=[]',old.index('  def export_adapter('))
        end=old.index('   if rank==0:',start)
        new=old[:start]+"   from policy_identity import resident_lora_weights\n   params=resident_lora_weights(self)\n"+old[end:]
        w=Path(tmp)/worker.name;w.write_text(new);remote('unlink '+q(str(dest/worker)))
        subprocess.run(['scp','-q',str(w),host+':'+str(dest/worker)],check=True)
        s['source_pins'][worker.name]=hashlib.sha256(w.read_bytes()).hexdigest()
        if c.get('recoverable'):
            # The reference worker was already overlaid for arithmetic; pin
            # the exact effective worker that will own the saved shards.
            pins=worker.with_name('SOURCE-PINS.json')
            raw=remote('cat '+q(str(parent/pins)))
            manifest=json.loads(raw)
            updated=[r for r in manifest['sources'] if r['path'].endswith('/jev-system-rl/implementation/worker_dp8.py')
                     and '/implementation/hot/' not in r['path']]
            if len(updated)!=1:raise ValueError('expected one current worker source pin')
            updated[0].update(sha256=s['source_pins'][worker.name],bytes=len(w.read_bytes()))
            sealed=Path(tmp)/pins.name;sealed.write_text(json.dumps(manifest,indent=2))
            remote('unlink '+q(str(dest/pins)))
            subprocess.run(['scp','-q',str(sealed),host+':'+str(dest/pins)],check=True)
            s['source_pins'][pins.name]=hashlib.sha256(sealed.read_bytes()).hexdigest()
        p=Path(tmp)/'TRAIN.local.json';p.write_text(json.dumps(s,indent=2))
        for n,f in [*files.items(),(p.name,p)]:
            subprocess.run(['scp','-q',str(f),host+':'+str(dest/n)],check=True)
    remote('cd '+q(str(dest))+' && (nohup python3 -B launch_retained_hf_fsdp.py --config TRAIN.local.json >launch.log 2>&1 </dev/null & )')
    print(dest)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    import os
    p.add_argument('--config',type=Path,default=os.environ.get('OMNI_ACTOR_CONFIG'))
    p.add_argument('--profile',type=Path,help='optional deployment profile for a template')
    p.add_argument('--resolution-dir',type=Path,default=os.environ.get('OMNI_DEPLOYMENT_OUTPUT'))
    a=p.parse_args()
    if not a.config:p.error('provide --config or OMNI_ACTOR_CONFIG')
    raw=a.config.read_bytes()
    config=json.loads(raw)
    if a.profile:
        from source.config_profile import compile_profile
        from run_package import read
        if not a.resolution_dir:a.resolution_dir=a.config.parent/(a.config.name+'.resolution')
        receipt=read(compile_profile(a.config,a.profile,a.resolution_dir))
        profile_raw=Path(receipt['profile_snapshot_ref']['path']).read_bytes()
        config=read(receipt['resolved_ref'])
        if 'prior_step' not in config: p.error('--profile requires a continuation')
        stage(config,{'CAPTURE-TEMPLATE.json':raw,'CAPTURE-PROFILE.local.json':profile_raw,
                      'CAPTURE-RESOLVED.json':json.dumps(config,indent=2).encode()})
    else:
        stage(config)

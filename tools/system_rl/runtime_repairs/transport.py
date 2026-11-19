"""Original client tar/bootstrap/step; fixed target agent, no unknown replay.

This module is inert on import. Only explicit iteration execution contacts hosts.
All archives contain sealed sources or closed results, never active hot snapshots.
"""
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time
from dependencies import HERE,ROOT,JEV,require,load,sha,local_path
from run_package import read as package_read,pin,write


def resolve_placeholder(value):
    """An empty mount placeholder is not a pinned immutable input's bytes.

    Bind only a byte-for-byte authenticated original data file. Nonempty staged
    drift and executable sources remain errors; no pin is refreshed.
    """
    if not isinstance(value,dict) or not {'path','sha256'}<=value.keys():return
    original=Path(value['path']);resolved=local_path(original)
    if (not original.is_absolute() or original.suffix=='.py' or resolved==original or
            not resolved.is_file() or resolved.stat().st_size!=0 or not original.is_file()):return
    import hashlib
    if value['sha256']==hashlib.sha256(b'').hexdigest():return
    h=hashlib.sha256()
    with original.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    require(h.hexdigest()==value['sha256'],'original placeholder input pin drift: '+str(original))
    from dependencies import register_locations
    register_locations({str(original):str(original)})


def read(value):
    """Keep exact pin validation and name the failed nested input."""
    if isinstance(value,dict) and {'path','sha256'}<=value.keys() and not Path(value['path']).is_absolute():
        from dependencies import VIRTUAL_ROOT
        value=dict(value,path=str(VIRTUAL_ROOT.parents[1]/value['path']))
    resolve_placeholder(value)
    try:return package_read(value)
    except ValueError as error:
        path=value.get('path') if isinstance(value,dict) else str(value)
        raise ValueError('transport read failed: '+str(path)+': '+str(error)) from error


def bounded_selector_inputs(document, collect, read_value=read):
    """Stage the audit/stream bytes consumed by bounded_train_probe.draw.

    Historical record.online is replaced by bind_frozen_inputs before capture.
    Its source configs are neither imported nor hydrated by this selector.
    Frozen packet dependencies are collected separately from the actual input.
    """
    require(document.get('read_only') is True and
            document.get('official_full_population_score') is False,
            'bounded selector audit schema differs')
    for batch in document['batches']:
        version_ref=batch['sealed_version'];collect(version_ref,False)
        version=read_value(version_ref)
        if not version.get('training_pool'):continue
        pool_ref=version['training_pool'];collect(pool_ref,False)
        pool=read_value(pool_ref)
        for name in ('records','outcomes'):
            collect(pool['streams'][name],False)


def source_member(path):
    """Map physical sealed and logical mounted names to one source member."""
    from dependencies import VIRTUAL_ROOT
    path=Path(path)
    for base in (ROOT,VIRTUAL_ROOT):
        if path.is_relative_to(base):return 'parents/'+str(path.relative_to(base))
    return None


def client():
    directory=ROOT/'native-recompute-integration/state-repair/configured-dispatch-successor'
    # Original client has generic module names. Run this transport in its own
    # configured CPU process; isolate imports from training runtime/storage.
    import importlib.util
    names=('storage','bundle')
    saved={n:sys.modules.pop(n,None) for n in names};paths=sys.path[:]
    try:
        sys.path.insert(0,str(directory))
        result=load(directory/'client.py','dp8_original_client')
    finally:
        sys.path[:]=paths
        for name,value in saved.items():
            sys.modules.pop(name,None)
            if value is not None:sys.modules[name]=value
    return result


def closed_reader_source():
    """Explicit client recovery reader; numerical/source bundle stays frozen."""
    import os,hashlib
    path=os.environ.get('OMNI_CLOSED_READER_PATH')
    if not path:return (HERE/'pull_closed.py').read_text()
    raw=Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest()==os.environ.get('OMNI_CLOSED_READER_SHA256'),
            'closed recovery reader drift')
    return raw.decode()


def verify_execution_bundle(bundle):
    """The preparation interpreter and dispatched release must be identical.

    A valid archive hash alone authenticates an archive, not its agreement
    with the code currently computing the request/closure. Check before any
    historical dependency walk or accounting snapshot.
    """
    members=bundle['files'];prefix='parents/jev-system-rl/implementation/'
    paths=sorted(HERE.glob('*.py'))+[HERE/'SOURCE-PINS.json',HERE/'runtime.json']
    failures=[]
    for path in paths:
        expected=members.get(prefix+path.name)
        actual=dict(sha256=sha(path),bytes=path.stat().st_size)
        if expected is None or any(expected.get(k)!=v for k,v in actual.items()):
            failures.append(dict(member=prefix+path.name,expected=expected,actual=actual))
    require(not failures,'prepare/dispatch release mismatch: '+json.dumps(failures,sort_keys=True))
    return dict(passed=True,checked=len(paths),scope='execution modules, source pins and runtime identity')


def heldout_iteration_inputs(document):
    """Runtime refs consumed by production_stage and provider_session.

    The full iteration JSON is still authenticated and transferred unchanged.
    This projection controls dependency traversal only, never execution bytes.
    Training trajectories, optimizer inputs and client archival configuration
    are not read by a heldout remote job.
    """
    keys=('program_template_ref','node_evaluation','reward_evaluation',
          'resource_owner_config_ref','registry','authorization_ref',
          'named_rpc_overlay','endpoint_probe_packet_ref','endpoint_probe_query_id',
          'initial_input')
    return {key:document[key] for key in keys if key in document}


def projection_runtime_refs(document):
    """Exact shard inputs read by Executor.shard and encoder_stage.prepare.

    The authenticated index is transferred unchanged. Producer manifests,
    receipts and source inventories are lineage, not inputs opened by either
    runtime consumer. Missing runtime members remain fatal.
    """
    require(document.get('schema')=='full-audio-plain-projection-index-v1',
            'projection runtime schema differs')
    refs=[]
    for row in document['shards']:
        for field in ('tasks','bindings'):
            ref=row[field]
            require(isinstance(ref,dict) and {'path','sha256'}<=ref.keys(),
                    'projection runtime member absent: '+field)
            refs.append(ref)
    return refs


def cache_runtime_inputs(document):
    """Cache.__init__, Session.retrieve and Matrices consume these fields.

    Keep exact manifest bytes; prior_complete/base_plan are producer history,
    not instructions to rerun the producer or import its historical namespace.
    """
    require(document.get('schema')=='historical-audio-cache-sparse-overlay-v1',
            'cache runtime schema differs')
    require(document['current_qualification'] is False,'historical cache qualification differs')
    result={k:document[k] for k in ('entries','keys','descriptions','queries','ranking_identity')}
    if document.get('matrix_hash_receipts'):result['matrix_hash_receipts']=document['matrix_hash_receipts']
    return result


def materializer_runtime_inputs(document):
    """Exact inputs loaded by Materializer.__init__ and render_supply.

    Original Reader authenticates each input; it does not recursively open
    arbitrary references inside authority, acceptance or registry seals.
    """
    require(document.get('schema')=='p-supply-materializer-config-v1',
            'materializer runtime schema differs')
    keys=('authority','acceptance','reading_plan','cards','card_provenance','bank',
          'groups','members','frozen_bank','registry_complete','source_registry_binding',
          'source_registry','historical_bindings','support_text')
    return dict(inputs={k:document['inputs'][k] for k in keys},
                renderer=document['renderer'],storage_checker=document['storage_checker'])


def wait_terminal(config, task, directory, plan):
    """Read-only completion event; the original closed reader remains authority.

    Resident actors stay alive between commands, so their existing command
    receipt path keeps its normal cadence. Disconnects never imply completion
    and never cause a second START.
    """
    seconds=min(900,max(0,config['deadline_epoch']-time.time()))
    if seconds<=0:return
    if task.get('resident',{}).get('enabled') and not task.get('_resident_attach'):
        time.sleep(seconds);return
    run=read(Path(directory)/'REQUEST.json')['run']
    import shlex
    code="""import json,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]);boot=sys.argv[2]
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==boot,'terminal event boot drift'
cid=json.loads((root/'CREATED.json').read_text())['container_id']
if (root/'CONTROLLER-RECOVERY.json').is_file():
 recovery=json.loads((root/'CONTROLLER-RECOVERY.json').read_text())
 assert recovery['kind']=='pre-admission-cpu-auxiliary-mount-recovery' and recovery['original_container_id']==cid
 assert recovery['source_boot']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
 assert recovery['original_failure_sha256']==__import__('hashlib').sha256((root/'FAILED.json').read_bytes()).hexdigest()
 previous=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
 assert previous['Id']==cid and not previous['State']['Running'] and previous['State']['ExitCode']!=0
 cid=recovery['container_id']
if (root/'CONTROLLER-RECOVERY-CHAIN.json').is_file():
 for recovery in json.loads((root/'CONTROLLER-RECOVERY-CHAIN.json').read_text()):
  assert recovery['original_container_id']==cid
  assert recovery['source_boot']==boot
  if recovery.get('original_failure_sha256'):
   assert __import__('hashlib').sha256(Path(recovery['failure_path']).read_bytes()).hexdigest()==recovery['original_failure_sha256']
  previous=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
  assert not previous['State']['Running'] and previous['State']['ExitCode']!=0
  cid=recovery['container_id']
state=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
assert state['Id']==cid,'terminal event container identity drift'
subprocess.run(['docker','wait',cid],check=True,stdout=subprocess.DEVNULL)
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==boot,'terminal event boot changed'
"""
    argv=['python3','-B','-c',code,plan['target'],plan['job']['source_boot']]
    command=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',run['host'],shlex.join(argv)]
    started=time.monotonic()
    try:
        result=subprocess.run(command,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,timeout=seconds)
        if result.returncode==0:return
    except (OSError,subprocess.TimeoutExpired):
        pass
    # Failure is an observation failure, not a failed job or retry permission.
    time.sleep(max(0,seconds-(time.monotonic()-started)))


def generation_runtime_inputs(task, collect, read_value=read):
    """Stage closed rollout bytes once; expand only actual training consumers.

    clone_generation copies every inventory member verbatim. prepare_generation
    uses embedded wire audio and annotations; finalize additionally reads policy,
    consumer/source pins and the recording map. Historical owner/config citations
    are retained as bytes, never recursively executed as a new input graph.
    """
    if not task.get('pending_capture'):
        return set()
    require(task.get('generation_inventory'), 'closed generation inventory required')
    native_ref=task['pending_capture'];inventory_ref=task['generation_inventory']
    native=read_value(native_ref);pending_ref=native['pending_capture_ref']
    pending=read_value(pending_ref);inventory=read_value(inventory_ref)
    root=Path(pending['cohort_root'])
    require(root.is_absolute() and inventory.get('closed') is True and
            inventory.get('cohort_root')==str(root), 'generation runtime source differs')
    files=inventory['files']
    require(isinstance(files,dict) and files, 'closed generation members required')
    for name,ref in files.items():
        rel=Path(name)
        require(name and not rel.is_absolute() and '..' not in rel.parts and
                Path(ref['path'])==root/rel, 'generation runtime member escapes')
    for name,ref in [('NATIVE.pending.disabled.json',native_ref),
                     ('PENDING-CAPTURE.json',pending_ref),('INPUT.json',pending['input_ref'])]:
        require(name in files and files[name]['path']==ref['path'] and
                files[name]['sha256']==ref['sha256'], 'generation control identity differs')
    closed_ref=inventory['generation_closed_ref'];closed=read_value(closed_ref)
    require(files['GENERATION-CLOSED.json']['sha256']==closed_ref['sha256'] and
            files['GENERATION-CLOSED.json']['path']==closed_ref['path'], 'generation closure identity differs')
    # Keep exact closure semantics under original clone_generation validation.
    require(isinstance(closed,dict), 'closed generation record required')
    for ref in [inventory_ref,*files.values()]:collect(ref,False)
    captured=read_value(pending['input_ref'])
    require(captured.get('schema')=='configured-fresh-cohort-v1', 'generation input schema differs')
    for key in ('learning_config_ref','clock_state_ref'):
        collect(captured[key])
    collect(captured['policy'])
    collect(captured['capture']['endpoint_policy_ref'])
    if captured.get('phase_frozen_upstream_ref'):
        collect(captured['phase_frozen_upstream_ref'],False)
    for key in ('learning_config_ref','clock_state_ref','consumer_source_ref',
                'restore_learning_config_ref','restore_consumer_config_ref','restore_checkpoint_ref'):
        if native.get(key):collect(native[key])
    for consumer_ref in (captured['consumer_config_ref'], native['consumer_config_ref']):
        collect(consumer_ref,False)
        consumer=read_value(consumer_ref)
        for key in ('source_pins','sealed_adapter_ref','learning_config_ref','clock_state_ref'):
            if consumer.get(key):collect(consumer[key])
        if consumer.get('selected_cohort_ref'):collect(consumer['selected_cohort_ref'],False)
        if consumer.get('recording_map_ref'):
            collect(consumer['recording_map_ref'],False)
            for row in read_value(consumer['recording_map_ref'])['records']:
                collect(row['audio_ref'],False)
                for ref in row.get('context_audio_refs',[]):collect(ref,False)
    # The selected plan is read as data; its historical upstream packets are
    # already embedded in GENERATED and are not a request to rerun the system.
    collect(native['selected_cohort_ref'],False)
    handled={'pending_capture','generation_inventory'}
    if task.get('rollout_logits_index_ref'):
        index_ref=task['rollout_logits_index_ref'];index=read_value(index_ref)
        require(index.get('schema')=='omni-http-full-logits-index-v1', 'actual logits index required')
        collect(index_ref,False)
        if index.get('config_ref'):collect(index['config_ref'],False)
        for ref in index.get('observation_decisions',{}).values():collect(ref,False)
        for ref in index['by_wire_sha256'].values():
            collect(ref,False);wire=read_value(ref)
            require(wire.get('schema')=='omni-http-full-logits-v1','actual logits wire required')
            collect(wire['activation_ref'],False)
            complete_ref=wire['full_logits_ref'];collect(complete_ref,False)
            complete=read_value(complete_ref)
            require(complete['identity']['activation_ref']==wire['activation_ref'], 'logits activation differs')
            for worker in complete['workers']:
                for request in worker['requests']:
                    collect(request['ref'],False)
                    collect(read_value(request['ref'])['tensor_ref'],False)
        handled.add('rollout_logits_index_ref')
    return handled


def prepare(config,task,directory,host,operation):
    directory=Path(directory);request=directory/'TRANSPORT.json'
    from storage import configure_scope
    configure_scope(config['output_root'],config['client_storage_bytes'],config['run_id']+':client',Path('/proc/sys/kernel/random/boot_id').read_text().strip(),noncheckpoint_limit_bytes=config.get('client_noncheckpoint_limit_bytes'))
    from policy_identity import digest
    identity=digest(dict(run_id=config['run_id'],directory=str(directory.resolve()),host=host,operation=operation,task=task))
    if request.exists():
        prior=read(request);require(prior['task_identity']==identity,'transport logical task drift');return prior
    t=config['transports'][host];stage=Path(t['stage_parent'])/config['run_id']/identity/operation
    target=Path(t['run_parent'])/config['run_id']/identity/operation
    require(target.is_relative_to(Path(t['storage_anchor'])) and stage.is_relative_to(Path(t['storage_anchor'])),'explicit local anchor')
    pack=load(HERE/'stage.py','transport_original_bundle').pack
    from storage import environment
    environment(directory,t.get('client_storage_bytes',t['storage_bytes']))
    if config.get('source_bundle_ref'):
        data=read(config['source_bundle_ref']);bundle=Path(pin(config['source_bundle_ref'])['path']).parent
        require(sha(bundle/'sources.tar')==data['archive']['sha256'],'frozen transport bundle drift')
    else:
        bundle=directory/'source';pack(bundle,HERE/'runtime.json',source_only=True);data=read(bundle/'BUNDLE.json')
    verify_execution_bundle(data)
    files={}
    with tarfile.open(bundle/'sources.tar') as tar:
        for item in tar:files[item.name]=tar.extractfile(item).read()
    files['bootstrap.py']=files['parents/jev-system-rl/implementation/bootstrap.py'] if config.get('source_bundle_ref') else (HERE/'bootstrap.py').read_bytes()
    files['BUNDLE.json']=(bundle/'BUNDLE.json').read_bytes()
    # Original streaming bootstrap installs expanded files directly; retaining
    # a second nested source archive duplicates the exact same bytes.
    mappings=[];seen={};expanded=set();cold_assets=[]
    stage_request=read(task['request_ref']) if task.get('request_ref') else {}
    active_stage=stage_request.get('stage')
    capture_input=read(stage_request['input_ref']) if active_stage=='capture' else {}
    bounded_audit=capture_input.get('bounded_train_probe',{}).get('source_audit_ref')
    active_business_ref=(read(stage_request['input_ref']).get('capture',{}).get('business_config_ref')
                         if active_stage=='capture' else None)
    errors=[];chain=[];provenance_refs=set();content_seen=set();derived_systems=set();template_businesses=set();authority_inputs={}
    def unwrap(value):
        while isinstance(value,dict) and set(value)=={'$collection_content_v1'}:
            ref=value['$collection_content_v1'];collect(ref,False);value=read(ref)
        return value
    def content_refs(value):
        # material.expand recursively opens only content wrappers. It never
        # opens ordinary pins in historical encoder outputs while expanding.
        if isinstance(value,dict):
            if {'binding','consumer_result','entry'}<=value.keys():
                collect(value['consumer_result'],False)
                collect(value['entry']['vector'],False)
            if {'closure','prepared','observed','validation'}<=value.keys():
                # material.expand opens this content wrapper without walking
                # ordinary historical refs. hydrate nevertheless consumes the
                # original deterministic D config and per-query S1 cache.
                prepared_ref=unwrap(value['prepared']);closure_ref=unwrap(value['closure'])
                require(isinstance(prepared_ref,dict) and {'path','sha256'}<=prepared_ref.keys(),'closed source prepared ref schema')
                prepared=read(prepared_ref);base=Path(prepared_ref['path']).parent.parent
                closed=read(closure_ref);collect(closure_ref,False)
                require(closed['schema']=='standard-d-closed-collection-v1' and closed['status']=='CLOSED_IMMUTABLE','retained D closure schema')
                # Original Source.member_doc/read_one consumes the exact sealed
                # member roster, including RECORDS, handoff and raw envelopes.
                # These are immutable leaves; their historic configs stay leaves.
                for member in closed['members']:collect(member,False)
                collect(pin(base/'preparation/s1'/(__import__('hashlib').sha256(prepared['reference_id'].encode()).hexdigest()+'.json')),False)
                binding_ref=pin(base/'bindings'/(__import__('hashlib').sha256(prepared['sample_id'].encode()).hexdigest())/'CONFIG.json')
                collect(binding_ref,False)
                policy_ref=read(binding_ref)['source_policy']['complete'];collect(policy_ref,False)
                for member in read(policy_ref)['artifacts']:
                    if Path(member['path']).name in ('retrieval.private.json','s1.runtime.json','s2-v2.runtime.json'):collect(member,False)
                collect(pin(base/'d-config'/(closure_ref['sha256']+'.json')))
            if set(value)=={'$collection_content_v1'}:
                ref=value['$collection_content_v1'];key=(ref['path'],ref['sha256'])
                if key not in content_seen:
                    content_seen.add(key);collect(ref,False);content_refs(read(ref))
                return
            for child in value.values():content_refs(child)
        elif isinstance(value,list):
            for child in value:content_refs(child)
    def knowledge_refs(value):
        # Original S1KnowledgeConsumer reads JSON-pointer source_refs at init.
        if isinstance(value,dict):
            if {'path','sha256','pointer'}<=value.keys():collect(value,False);return
            for child in value.values():knowledge_refs(child)
        elif isinstance(value,list):
            for child in value:knowledge_refs(child)
    def leaf_refs(value):
        if isinstance(value,dict):
            if {'path','sha256'}<=value.keys():collect(value,False)
            else:
                for child in value.values():leaf_refs(child)
        elif isinstance(value,list):
            for child in value:leaf_refs(child)
    def collect(value,expand=True):
        label=value.get('path') if isinstance(value,dict) else None
        if label:chain.append(label)
        try:visit(value,expand)
        except (ValueError,FileNotFoundError,KeyError,PermissionError) as error:errors.append(dict(chain=list(chain),error=str(error)))
        finally:
            if label:chain.pop()
    def visit(value,expand=True):
        if isinstance(value,dict):
            if {'path','sha256','text'}<=value.keys():
                require(__import__('hashlib').sha256(value['text'].encode()).hexdigest()==value['sha256'],'embedded source hash drift')
                return
            if {'path','sha256'}<=value.keys():
                path=Path(value['path'])
                if not path.is_absolute():
                    from dependencies import VIRTUAL_ROOT
                    path=VIRTUAL_ROOT.parents[1]/path
                    value=dict(value,path=str(path))
                resolve_placeholder(value)
                if str(path)==config['registry']['path']:return # explicit single-writer accounting handoff, never a readonly input
                if str(path) in seen:
                    require(seen[str(path)]==value['sha256'],'conflicting transport pin: '+str(path))
                    if not expand or str(path) in expanded:return
                seen[str(path)]=value['sha256']
                if expand:expanded.add(str(path))
                if len(seen)%200==0:
                    from storage import commit
                    commit(directory/'CLOSURE-PROGRESS.json',dict(count=len(seen),current=str(path),chain=list(chain)),replace=True)
                #181 restores its original authoritative full checkpoint at its
                #local path.175 only receives the committed identity+adapter.
                if host=='npu2' and path.is_relative_to(Path(t['run_parent'])):return
                if any(path.is_relative_to(Path(p)) for p in t['readonly_roots']):return
                asset=next((r for r in t.get('readonly_asset_refs',[]) if r['path']==str(path)),None)
                if str(path) in authority_inputs:
                    # Policy binder explicitly snapshots current authority text.
                    # Old cold-asset pins describe historical text, not this
                    # current input. Transfer exact authenticated bytes below;
                    # never refresh the old asset identity or weaken other pins.
                    require(value==authority_inputs[str(path)],'current authority identity differs')
                    asset=None
                if asset:
                    require(asset['sha256']==value['sha256'],'readonly asset identity drift')
                    remote=asset.get('remote_paths',{}).get(host,str(Path(t['storage_anchor'])/'cold-assets'/asset['sha256']))
                    cold_assets.append(dict(local_ref=asset,remote_path=remote,sha256=asset['sha256']))
                    mappings.append(dict(source=remote,target=str(path),readonly=True,expected_sha256=asset['sha256']))
                    if not expand or path.suffix!='.json':return
                if source_member(path) and path.suffix=='.py':
                    member=source_member(path)
                    if member in files:
                        require(__import__('hashlib').sha256(files[member]).hexdigest()==value['sha256'],'bundled executable hash drift: '+str(path))
                        # Legacy consumers open the exact physical pin path,
                        # not dependencies.local_path. Preserve that alias even
                        # when bytes already live in the sealed source bundle.
                        from dependencies import VIRTUAL_ROOT
                        if not path.is_relative_to(VIRTUAL_ROOT):
                            mappings.append(dict(source=str(stage/member),target=str(path),readonly=True))
                        return
                try:ref=pin(value)
                except ValueError as error:raise ValueError('transport input pin drift: '+str(path)) from error
                member=source_member(path)
                if asset:
                    logical=None # physical cold JSON still exposes runtime refs
                elif member in files and __import__('hashlib').sha256(files[member]).hexdigest()==ref['sha256']:
                    logical=member # already installed by original source bootstrap; no duplicate copy/mount
                    from dependencies import VIRTUAL_ROOT
                    if not path.is_relative_to(VIRTUAL_ROOT):
                        mappings.append(dict(source=str(stage/member),target=str(path),readonly=True))
                elif ref['bytes']>64*1024:
                    # Source-closure verification requires the bytes, not a
                    # recursive history traversal. Reuse large immutable members
                    # via the same content-addressed cold path as explicit assets.
                    remote=str(Path(t['storage_anchor'])/'cold-assets'/ref['sha256'])
                    cold_assets.append(dict(local_ref=ref,remote_path=remote,sha256=ref['sha256']))
                    mappings.append(dict(source=remote,target=str(path),readonly=True,expected_sha256=ref['sha256']))
                    if not expand:return
                    logical=None
                else:
                    require(ref['bytes']<=t['max_file_bytes'],'explicit readonly root needed: '+str(path))
                    logical='inputs/'+str(path).lstrip('/');files[logical]=local_path(path).read_bytes()
                    mappings.append(dict(source=str(stage/logical),target=str(path),readonly=True))
                if not expand:return
                if path.name in ('SOURCE-CLOSURE.json','CAPTURE-CLOSURE.json','FINITE-SOURCES.json','SOURCE-PINS.json','PROGRAM-SOURCEPINS.json','NODE-CLOSURE.json','NODE-SOURCES.json','POLICY-HOOK-SOURCES.json'):
                    manifest=read(ref)
                    if isinstance(manifest,list):
                        for item in manifest:collect(item,False)
                        return
                if path.suffix=='.json':
                    document=read(ref)
                    if isinstance(document,dict) and document.get('schema')=='unstarted-p-reward-revision-v1':
                        # Immutable migration lineage is retained verbatim.
                        # Runtime binds learning/consumer/Clock separately;
                        # predecessor implementation pins cannot be activated.
                        return
                    if bounded_audit and str(path)==bounded_audit['path']:
                        require(ref['sha256']==bounded_audit['sha256'],'bounded audit identity drift')
                        bounded_selector_inputs(document,collect)
                        return
                    if (active_stage=='capture' and
                            str(path)==stage_request.get('input_ref',{}).get('path')):
                        # Preparation consumed this lineage before dispatch.
                        # Capture consumes the resulting runtime fields.
                        document=dict(document)
                        retained=document.pop('retained_source_material_ref',None)
                        if retained:collect(retained,False)
                        # Revision receipt is audit lineage, not executable
                        # runtime input. Old and new scorer pins intentionally
                        # coexist inside it; consumers use the current fields.
                        revision=document.pop('reward_revision_ref',None)
                        if revision:collect(revision,False)
                    if (active_stage=='endpoint' and str(path) in
                            (config['initial_input'],stage_request.get('input_ref',{}).get('path'))):
                        # provider_session.load and phase_process.worker_context
                        # consume this projection. A resumed request need not
                        # use initial_input. Project BEFORE content expansion;
                        # source provenance is not an endpoint runtime input.
                        # The complete original bytes were authenticated above.
                        document={k:v for k,v in document.items() if k in
                                  ('policy','sampling','execution_registry','authorization','budget','resources','capture')}
                        document['capture']={k:v for k,v in document['capture'].items()
                                             if k in ('profile_ref','transport_config_ref')}
                    if active_stage in ('dev','test') and str(path)==config['program_template_ref']['path']:
                        # Same retained fields as the original policy binder.
                        # The raw JSON is still transferred/hash-checked above.
                        from production_stage import policy_template
                        document=policy_template(document)
                    if (active_stage in ('dev','test') and isinstance(document,dict)
                            and document.get('schema')=='automatic-system-release-v1'
                            and str(path) not in derived_systems):
                        # The original production/provider binder derives every
                        # active policy system BEFORE Engine imports its modules.
                        # Reuse that exact binder here; do not traverse stale
                        # executable pins as if the template itself were run.
                        sys.path.insert(0,str(ROOT/'policy-iteration-successor'))
                        from context_builder import policy_system
                        business_path=document['objects']['business']['kwargs']['config']
                        business_ref=pin(business_path)
                        candidates=[r for r in document['pins'] if r['path']==business_path]
                        require(len(candidates)==1 and candidates[0]==business_ref,
                                'policy template business identity differs')
                        collect(business_ref,False)
                        template_businesses.add(business_ref['path'])
                        bound_root=directory/'policy-dependencies'/value['sha256']
                        sp,bp=policy_system(stage_request,ref,business_ref,bound_root)
                        derived_systems.add(sp['path'])
                        collect(sp)
                        # Remote bind still reads the original materializer
                        # template and current authority source bytes.
                        original_business=read(business_ref)
                        materializer=original_business['encoder']['p_materializer_config']
                        collect(materializer,False)
                        for authority in read(materializer)['inputs']['authority']:
                            current=pin(authority['path'])
                            require(current['bytes']<=t['max_file_bytes'],'authority input exceeds inline limit')
                            authority_inputs[current['path']]=current
                            collect(current,False)
                        return
                    if active_stage in ('dev','test') and isinstance(document,dict):
                        if document.get('schema')=='production-input-manifest-v1':
                            # production_stage replaces both before executing.
                            # Bind system first so its original business remains
                            # an authenticated template, not a second runtime.
                            collect(document['system'])
                            collect(document['business'],False)
                            for key,child in document.items():
                                if key not in ('system','business'):collect(child)
                            return
                        if document.get('schema')=='bounded-official-diagnostic-v1':
                            # Original heldout.build reads splits and replaces
                            # selection/budgets. It never runs full_official_config.
                            collect(document['splits'])
                            return
                        if str(path) in template_businesses:
                            return
                    if ((active_stage=='endpoint' or (active_stage=='capture' and active_business_ref)) and
                            str(path)==stage_request.get('system_policy_ref',{}).get('path')):
                        # Endpoint executes the sealed probe; capture executes
                        # its derived business_config_ref.
                        # This frozen system is its scientific identity, not
                        # an instruction to execute the historical deployment.
                        require(document.get('schema')=='automatic-system-release-v1',
                                'endpoint/capture system identity schema differs')
                        return
                    if (active_stage=='capture' and isinstance(document,dict) and
                            document.get('schema')=='retained-recording-map-v1'):
                        # Capture reads records and their audio, not producer lineage.
                        collect(document['population_ref'],False)
                        for row in document['records']:
                            collect(row['audio_ref'],False)
                            for audio in row.get('context_audio_refs',[]):collect(audio,False)
                        return
                    content_refs(document)
                    if str(path) in provenance_refs:return
                    if isinstance(document,dict) and document.get('schema')=='full-audio-plain-projection-index-v1':
                        for dependency in projection_runtime_refs(document):collect(dependency,False)
                        return
                    if active_stage in ('capture','dev','test') and isinstance(document,dict):
                        if document.get('schema')=='historical-audio-cache-sparse-overlay-v1':
                            leaf_refs(cache_runtime_inputs(document))
                            return
                        if document.get('schema')=='p-supply-materializer-config-v1':
                            leaf_refs(materializer_runtime_inputs(document))
                            plan=read(document['inputs']['reading_plan'])
                            for key in ('producer_complete','cache_manifest'):collect(plan[key],False)
                            return
                    if active_stage in ('dev','test') and isinstance(document,dict) and document.get('schema')=='frozen-next-input-contract-v1':
                        # Executor/encoder_stage consume the recipe and exact
                        # historical encoding config; S1/P provenance seals
                        # are authenticated inputs, not executable old runs.
                        # Retain all referenced bytes, without traversing old
                        # COMPLETE -> INPUT-PINS -> raw model journals.
                        # consume.py reads the recipe as values; Executor and
                        # encoder_stage additionally open historical_encoding
                        # and record the original encoder implementation.
                        # P/S1 producer pointers are not opened by these readers.
                        for key in ('historical_encoding','original_encoder'):
                            collect(document[key],False)
                        return
                    if isinstance(document,dict) and document.get('schema')=='policy-http-witness-v1':
                        # protocol.verify replays these exact evidence bytes.
                        # Service identity embeds historical execution refs for
                        # equality checks; it never executes that old task DAG.
                        for key in ('activation_ref','call_ref','response_ref','receipt_ref','event_ref','profile_ref'):
                            collect(document[key],False)
                        collect(document['policy_ref'])
                        return
                    if isinstance(document,dict) and document.get('schema')=='official-train-task-source-v1':
                        # OfficialTrainSources reads the original domain metadata
                        # and reference index, not miner cleanup/review history.
                        collect(document['manifest'],False)
                        collect(document['population'],False)
                        manifest=read(document['manifest'])
                        domains={d['domain']:d for d in manifest['domains']}
                        for binding in document['domains']:
                            collect(domains[binding['domain']]['original'],False)
                            collect(binding['commit'],False)
                            commit=read(binding['commit'])
                            indices=[r for r in commit['artifacts'] if Path(r['path']).name=='references.jsonl']
                            require(len(indices)==1,'one official recording reference index required')
                            collect(indices[0],False)
                        for key in ('index_implementation','index_common'):collect(document[key],False)
                        return
                    if isinstance(document,dict) and isinstance(document.get('artifacts'),list) and any(Path(r['path']).name=='supplies-v1.private.jsonl' for r in document['artifacts']) and document.get('dispatch_allowed') is False:return # typed closed renderer-supply provenance
                    if str(path)==config['endpoint_probe_packet_ref']['path']:return # sealed wire; COMPLETE/read-pins are offline provenance
                    if isinstance(document,dict) and document.get('schema')=='automatic-system-release-v1':
                        # Engine.pin_inputs hashes each pin's bytes; it does not
                        # recursively execute pinned historical business configs.
                        # Hooks imports modules and opens the active business
                        # object's config, which is a plain path in the schema.
                        for ref in document.get('pins',[]):collect(ref,False)
                        collect(document.get('successor_of'),False)
                        collect(document.get('modules',{}))
                        business_path=document.get('objects',{}).get('business',{}).get('kwargs',{}).get('config')
                        if isinstance(business_path,str):
                            candidates=[r for r in document.get('pins',[]) if r.get('path')==business_path]
                            require(len(candidates)==1,'active business config requires one system pin')
                            collect(candidates[0])
                        for key,child in document.items():
                            if key not in ('pins','successor_of','modules'):collect(child)
                        return
                    if isinstance(document,dict) and document.get('schema')=='native-automatic-collection-binding-v2' and active_stage=='endpoint':return
                    if isinstance(document,dict) and document.get('schema')=='native-automatic-collection-binding-v2' and active_stage in ('capture','dev','test'):
                        if active_stage=='capture' and active_business_ref and str(path)!=active_business_ref['path']:
                            return # Historical release bindings authenticate lineage only.
                        # Encoder execution is disabled during fresh capture,
                        # but Executor also owns CPU PCM, shard and knowledge
                        # preparation. Preserve those actual consumer inputs.
                        encoder=document['encoder']
                        collect(encoder['handoff_contract'],False)
                        collect(encoder['projection_index'],False)
                        projection=read(encoder['projection_index'])
                        for dependency in projection_runtime_refs(projection):collect(dependency,False)
                        if encoder.get('task_binding_manifest'):
                            collect(encoder['task_binding_manifest'],False)
                        collect(encoder['s1_supplement'],False)
                        supplement=read(encoder['s1_supplement'])
                        collect(supplement['parent_contract'],False)
                        leaf_refs(supplement['S1'])
                        for name in ('runtime_io.py','s1_supplement.py'):
                            collect(pin(Path(encoder['s1_supplement']['path']).parent/name),False)
                        from dependencies import VIRTUAL_ROOT
                        collect(pin(VIRTUAL_ROOT.parents[1]/'delivery/s1-round03-bound-layout-combination-v1/package/src/speech_aware_evidence_acquisition/analysis/slurp_mf7b_vectors.py'),False)
                        for key in ('p_repair_config','p_materializer_config'):
                            collect(encoder[key],False)
                            cpu=read(encoder[key])
                            leaf_refs({k:v for k,v in cpu['inputs'].items()
                                       if k not in ('historical_events','historical_pins')})
                            leaf_refs(cpu['renderer'])
                            if 'reading_plan' in cpu['inputs']:
                                plan=read(cpu['inputs']['reading_plan'])
                                for item in ('producer_complete','cache_manifest'):collect(plan[item],False)
                        collect(document['retrieval_manifest'],False)
                        retrieval=read(document['retrieval_manifest'])
                        for key in ('keys','descriptions','queries'):collect(retrieval[key],False)
                        leaf_refs(retrieval['ranking_identity'])
                        for key in ('cache_entry','supply_row'):
                            collect(document.get(key),False)
                        document=copy.deepcopy(document)
                        for key in ('encoder','evaluation_bindings','export_template','cache_entry','supply_row','retrieval_manifest'):document.pop(key,None)
                    if isinstance(document,dict) and {'encoder','prepare','close_source'}<=document.keys():
                        collect({k:v for k,v in document.items() if k in ('P','S1','S2','D','prepare','close_source')})
                        return
                    if isinstance(document,dict) and document.get('schema')=='standard-d-source-v1':
                        collect(document['scientific_parent_pin'],False)
                        from dependencies import VIRTUAL_ROOT
                        # io_utils opens this sibling directly under the logical
                        # source tree. Authenticate the sealed bytes, but stage
                        # them at the consumer's path, not only the build path.
                        dependency_ref=pin(ROOT/'standard-d-source-adapter/DEPENDENCIES.json')
                        dependency_ref=dict(dependency_ref,path=str(VIRTUAL_ROOT/'standard-d-source-adapter/DEPENDENCIES.json'))
                        ddependencies=read(dependency_ref)
                        collect(dependency_ref,False)
                        for dependency in ddependencies.values():collect(dependency,False)
                        from dependencies import VIRTUAL_ROOT
                        original=ddependencies['original_config'];original=dict(original,path=str(VIRTUAL_ROOT.parents[1]/original['path'])) if not Path(original['path']).is_absolute() else original
                        original_cfg=read(original)
                        for key in ('knowledge','runtime_policy','output_schema','prompt','legacy_bridge','code_pins'):collect(original_cfg[key],False)
                        collect(document.get('source_bridge_config_pin'),False)
                        return
                    if active_stage in ('capture','dev','test') and isinstance(document,dict):
                        if document.get('schema')=='configured-cpu-collection-v1':
                            # All three stages use business.Session.build_collection:
                            # Collection.__new__ restores the selected proof and
                            # four knowledge/runtime members. It does not call
                            # Collection.__init__ or execute the historical seal's
                            # source-policy/P-materializer audit pipeline.
                            provenance_refs.update(str(Path(document['p_supply'][key]['path'])) for key in ('complete','events_complete','main_acceptance'))
                            collect(document['source_resolver']['config'])
                            policy_seal=read(document['source_policy']['complete'])
                            collect(document['source_policy']['complete'],False)
                            for member in policy_seal['artifacts']:
                                if Path(member['path']).name in ('knowledge.private.json','retrieval.private.json','s1.runtime.json','s2-v2.runtime.json'):
                                    collect(member,False)
                                    if Path(member['path']).name=='knowledge.private.json':knowledge_refs(read(member))
                            collect(document['p_supply']['renderer']['config'])
                            return
                        if document.get('schema')=='full-system-cpu-collection-v1':
                            for key in ('code_pins','knowledge','s1_policy','s2_policy','s1_cache','s2_common','s2_contract','p_common','p_renderer','p_prompter','p_prompt_config','p_seed','audio_reader','handoff_assembler'):collect(document[key],False)
                            knowledge_refs(read(document['knowledge']))
                            return
                        if source_member(path)=='parents/full-train-source-bridge/CONFIG.json':
                            collect(document['collection_config']);collect(document['collection_adapter']);return
                        if source_member(path)=='parents/p-card-field-repair/CONFIG.json':
                            # Renderer reads the actual input tables; preserved
                            # seals describe producer history, not runtime work.
                            leaf_refs(document['inputs'])
                            leaf_refs(document['renderer'])
                            return
                    if active_stage in ('endpoint','dev','test') and str(path)==config['initial_input']:
                        document={k:v for k,v in document.items() if k in ('policy','sampling','execution_registry','authorization','budget','resources','capture')}
                        document['capture']={k:v for k,v in document['capture'].items() if k in ('profile_ref','transport_config_ref')}
                    if str(path)==str(config['_path']):
                        # Independent client source/acceptance/archive config
                        # stays on client; not projected into an affected host.
                        document=copy.deepcopy(document)
                        for key in ('source_bundle_ref','archive'):document.pop(key,None)
                        if active_stage=='capture':
                            document={k:v for k,v in document.items() if k in ('registry','authorization_ref','named_rpc_overlay','resource_owner_config_ref','endpoint_probe_packet_ref','endpoint_probe_query_id','finite_diagnostic_binding','finite_diagnostic_config_ref')}
                        if active_stage=='endpoint':
                            document={k:v for k,v in document.items() if k in ('registry','authorization_ref','named_rpc_overlay','endpoint_probe_packet_ref','endpoint_probe_query_id','resource_owner_config_ref','initial_input')}
                        if active_stage in ('dev','test'):
                            document=heldout_iteration_inputs(document)
                    # Closed model inventory identifies base assets; do not
                    # rescan/copy large model weights or historical archives.
                    if isinstance(document,dict) and (document.get('status')=='CLOSED_IMMUTABLE' or document.get('kind')=='base'):
                        document=copy.deepcopy(document);document.pop('assets',None)
                    # Closed historical provenance authenticates itself; its
                    # old executable pins are history, not this runtime closure.
                    if isinstance(document,dict):
                        schema=document.get('schema')
                        if schema=='finite-capture-binding-v1':
                            document=copy.deepcopy(document);document.pop('inherited_capture_binding_ref',None)
                        if schema=='automatic-dataset-manifest-v1':
                            document=copy.deepcopy(document);document.pop('provenance',None)
                        if schema in ('generation-environment-v1','current-retrieval-binding-v1'):return # environment/history identity, no runtime consumer
                        if schema=='omni-production-row-precision-v1':
                            collect(document['sources'])
                            if document.get('language_rows'):
                                rows=document['language_rows'];binding=rows['binding']
                                collect(binding['module']);collect(binding['dependencies'])
                                if rows.get('moe_binding'):
                                    collect(rows['moe_binding']['module']);collect(rows['moe_binding']['dependencies'])
                                    collect(rows['worker_sources'])
                                if rows.get('validation_ref'):
                                    collect(rows['validation_ref'],False)
                                    # Worker admission replays original raw evidence.
                                    # Project only its typed closed inputs; never recurse
                                    # into historical reports/source trees or treat the
                                    # saved validation boolean as authority.
                                    validation=read(rows['validation_ref'])
                                    for key in ('report_ref','archive_ref','receipt_ref'):
                                        collect(validation[key],False)
                                    for member in validation['verified_members']:
                                        collect(member,False)
                            if document.get('moe'):
                                collect(document['moe']['sources']);collect(document['moe']['module_ref'])
                            return
                        if document.get('read_only') is True and document.get('official_full_population_score') is False:
                            # bounded selector reads batch versions, not audit samples' history.
                            for batch in document['batches']:collect(batch['sealed_version'])
                            return
                        if schema=='automatic-collection-version-v1' and document.get('status')=='SEALED':
                            if document.get('training_pool'):collect(document['training_pool'])
                            return
                        if schema=='sealed-training-pool-v1' and document.get('sealed'):
                            require('streams' in document or 'records' in document,'sealed pool has neither streams nor records: '+str(path))
                            collect(document['streams'] if 'streams' in document else document['records']);return
                        if schema=='standard-d-closed-collection-v1' and document.get('status')=='CLOSED_IMMUTABLE':return
                        if document.get('binding',{}).get('schema')=='bound-recording-input-v1' and 'consumer_result' in document:
                            # Retained encoder result is expanded as data, never re-executed.
                            return
                    if isinstance(document,dict) and document.get('schema')=='sealed-train-source-proof-v1' and active_stage!='endpoint':
                        for key in ('base_reference','reference_isolation','registry_source','source_evidence'):collect(document[key])
                        return
                    if isinstance(document,dict) and document.get('schema') in ('sealed-train-source-proof-v1','machine-collection-validation-v2'):
                        # Prepared-source validation reads this sealed row's fields;
                        # original corpus/index references are provenance only.
                        return
                    if isinstance(document,dict) and document.get('schema')=='configured-ready-request-v1':
                        # material.selected/hydrate_packets read the prepared
                        # bytes and original audio/proof, never execute its old
                        # per-stage factory/config or retrieval provenance.
                        collect(document['source_proof_ref'])
                        collect(document['original_audio_ref'])
                        # Original preparation executor stores exact per-sample
                        # cache/policies alongside business/prepared; no rebuild.
                        cache_path=path.parent.parent/'preparation/s1'/(__import__('hashlib').sha256(document['reference_id'].encode()).hexdigest()+'.json')
                        if cache_path.is_file():collect(pin(cache_path),False)
                        return
                    try:collect(document)
                    except (ValueError,FileNotFoundError) as error:raise ValueError(str(path)+' -> '+str(error)) from error
                elif path.suffix=='.jsonl':
                    for line in (files[logical] if logical else local_path(path).read_bytes()).splitlines():
                        if line.strip():collect(json.loads(line))
            else:
                if {'path','sha256','text'}<=value.keys():
                    require(__import__('hashlib').sha256(value['text'].encode()).hexdigest()==value['sha256'],'embedded source hash drift')
                    return
                if value.get('schema')=='sealed-train-source-proof-v1':
                    if active_stage!='endpoint':
                        for key in ('base_reference','reference_isolation','registry_source','source_evidence'):collect(value[key])
                    return
                if value.get('schema')=='machine-collection-validation-v2':return
                if set(value)=={'$collection_content_v1'}:
                    collect(value['$collection_content_v1'])
                    # material.expand recursively reads retained response/evidence
                    # bodies even when the same file was a sealed manifest leaf.
                    content_ref=value['$collection_content_v1']
                    content=read(content_ref)
                    if isinstance(content,dict) and {'encoder','prepare','close_source'}<=content.keys():
                        # Retained encoder/inventory history is not consumed by
                        # material.selected/hydrate; model replies and closed
                        # source are expanded recursively with original bytes.
                        content={k:v for k,v in content.items() if k in ('P','S1','S2','D','prepare','close_source')}
                    collect(content)
                    return
                if {'closure','prepared','observed','validation'}<=value.keys() and isinstance(value['prepared'],dict) and 'path' in value['prepared']:
                    prepared=read(value['prepared']);base=Path(value['prepared']['path']).parent.parent
                    cache_path=base/'preparation/s1'/(__import__('hashlib').sha256(prepared['reference_id'].encode()).hexdigest()+'.json')
                    collect(pin(cache_path),False)
                    collect(pin(base/'d-config'/(value['closure']['sha256']+'.json')))
                if value.get('schema_version')=='system-rl-trajectory-analysis-v2':
                    for span in value['spans']:
                        collect(span.get('packet_ref') or span['online_request'])
                    return
                for child in value.values():collect(child)
        elif isinstance(value,list):
            for child in value:collect(child)
        elif isinstance(value,str) and value in (config.get('initial_input'),config.get('training_package_template')):collect(pin(value))
    accounting=None
    if host=='npu' and config.get('registry_target_parent'):
        from registry_custody import snapshot,facts
        seed=directory/'accounting-input/SEED.sqlite'
        if seed.exists():
            require(not (directory/'STAGE-INTENT.json').exists() and not (directory/'START-INTENT.json').exists(),'staged seed requires original reconciliation')
            before=facts(config['registry']['path'])
            require(facts(seed)==before and all(r[2] in ('known','closed_unobserved') for r in before['spend']),'partial seed differs from known original ledger')
            accounting=dict(seed_ref=pin(seed),logical_parent=str(Path(config['registry']['path']).parent),basename=Path(config['registry']['path']).name,remote_parent=config['registry_target_parent'],before=before)
        else:accounting=snapshot(config['registry'],directory/'accounting-input',config['registry_target_parent'])
    generation_keys=generation_runtime_inputs(task,collect) if operation=='training' else set()
    if operation=='training':
        # run_package pins these two identities but does not execute their
        # historical deployment graph; actual worker data comes from trajectory.
        for key,value in task.items():
            if key in generation_keys:continue
            if key=='newprocess_replay' and value.get('enabled') and value.get('input_ref'):
                # begin/actual_cohort read the original JSON's sampling lineage,
                # sampling, capture.seed and policy identities. They never run
                # this input's historic source/consumer/deployment program.
                collect(value['input_ref'],False)
                collect(read(value['input_ref'])['policy'])
                for field,child in value.items():
                    if field!='input_ref':collect(child)
            else:collect(value,expand=key not in ('system','dataset'))
    else:collect(task)
    collect(config.get('resource_owner_config_ref'))
    # Training consumes its exact task/package refs. The client iteration JSON
    # is identity evidence, not a request to stage all historical program inputs.
    collect(pin(config['_path']),expand=operation!='training')
    report=dict(errors=errors,passed=not errors,stage=active_stage,mappings=len(mappings))
    previous=directory/'CLOSURE-CHECK.json'
    if previous.exists() and read(previous)!=report:
        write(directory/('CLOSURE-CHECK-history-'+sha(previous)+'.json'),read(previous))
        from storage import commit
        commit(previous,report,replace=True)
    else:write(previous,report)
    require(not errors,'transport closure failed; see CLOSURE-CHECK.json ('+str(len(errors))+' errors)')
    task=copy.deepcopy(task)
    if operation=='training':task.update(stage_root=str(stage),output_root=str(target/'control'))
    readonly_roots=list(t['readonly_roots'])
    if host=='npu2':readonly_roots.append(t['run_parent'])
    cold_assets=list({(r['remote_path'],r['sha256']):r for r in cold_assets}.values())
    unique={}
    for mapping in mappings:
        old=unique.get(mapping['target'])
        require(old is None or old==mapping,'duplicate logical mount with different source: '+mapping['target'])
        unique[mapping['target']]=mapping
    mappings=list(unique.values())
    # Group original closed hot-project input trees to stay within execve's
    # argv limit. Cold file overlays retain individual SHA checks. No guard,
    # active service or unrelated project directory is projected.
    hot_prefix=Path('/home/xingchao/omni-hot-runtime');groups={};remaining=[]
    for mapping in mappings:
        target_path=Path(mapping['target'])
        if operation=='rollout' and target_path.is_relative_to(hot_prefix) and len(target_path.relative_to(hot_prefix).parts)>1:
            project=hot_prefix/target_path.relative_to(hot_prefix).parts[0]
            grouped_source=stage/'inputs'/str(project).lstrip('/')
            if mapping['source']==str(stage/'inputs'/str(target_path).lstrip('/')):
                groups[str(project)]=dict(source=str(grouped_source),target=str(project),readonly=True)
                continue
            if mapping.get('expected_sha256'):
                files.setdefault('inputs/'+str(target_path).lstrip('/'),b'')
        remaining.append(mapping)
    mappings=list(groups.values())+remaining
    job=dict(operation=operation,task=task,task_identity=identity,source_stage=str(stage),output_root=str(target),
        source_boot=t['source_boot'],image=read(HERE/'runtime.json')['cpu_image_id'],mappings=mappings,
        readonly_roots=readonly_roots,cold_assets=cold_assets,storage_bytes=t['storage_bytes'],closed_return_max_bytes=t['closed_return_max_bytes'],storage_scope_identity=config['run_id']+':'+host,storage_scope_root=str(Path(t['run_parent'])/config['run_id']),noncheckpoint_limit_bytes=t.get('noncheckpoint_limit_bytes'),storage_auxiliary_roots=t.get('noncheckpoint_roots',[]),max_scope_tasks=t.get('max_scope_tasks'),max_full_checkpoints=t.get('max_full_checkpoints'))
    files['JOB.json']=(json.dumps(job,sort_keys=True)+'\n').encode()
    files['INPUTS.json']=(json.dumps(dict(run=dict(remote_root=str(target)),budget=dict(storage_budget_bytes=t['storage_bytes'])))+'\n').encode()
    files['STAGE-PINS.json']=json.dumps({name:__import__('hashlib').sha256(raw).hexdigest() for name,raw in files.items()}).encode()
    if accounting:
        source=Path(accounting['seed_ref']['path']);logical='inputs/'+str(source).lstrip('/')
        files[logical]=source.read_bytes();accounting['seed_ref']=dict(accounting['seed_ref'],path=str(stage/logical))
        job['accounting']=accounting
        files['JOB.json']=(json.dumps(job,sort_keys=True)+'\n').encode()
        files['STAGE-PINS.json']=json.dumps({name:__import__('hashlib').sha256(raw).hexdigest() for name,raw in files.items() if name!='STAGE-PINS.json'}).encode()
    for mapping in mappings:
        logical=Path(mapping['target'])
        member=source_member(logical)
        if member:
            if member not in files:files[member]=b''
    files['STAGE-PINS.json']=json.dumps({name:__import__('hashlib').sha256(raw).hexdigest() for name,raw in files.items() if name!='STAGE-PINS.json'}).encode()
    c=client();archive=c.tar_bytes(files)
    write(directory/'STAGE-SIZES.json',dict(archive_bytes=len(archive),limit=t['max_bundle_bytes'],members=[dict(name=n,bytes=len(raw)) for n,raw in sorted(files.items(),key=lambda item:len(item[1]),reverse=True)]))
    require(len(archive)<=t['max_bundle_bytes'],'sealed transport source/input budget')
    from storage import environment
    environment(directory,t.get('client_storage_bytes',t['storage_bytes']))
    # Original bootstrap writes/validates local target storage, stdlib only.
    write(directory/'REQUEST.json',dict(run=dict(host=config.get('host_bindings',{}).get(host,host),remote_stage=str(stage),remote_root=str(target),remote_storage_anchor=t['storage_anchor'],source_boot=t['source_boot'],checkpoint_cache_root=config.get('checkpoint_cache_root',str(Path(config['output_root'])/'checkpoint-inventory')))))
    with (directory/'stage.tar').open('xb') as f:
        f.write(archive);f.flush();__import__('os').fsync(f.fileno())
    return_value=dict(host=host,stage=str(stage),target=str(target),job=job,task_identity=identity,
        stage_tar_ref=pin(directory/'stage.tar'),automatic_retry=False)
    write(request,return_value);return return_value


def remote_agent_source():
    """Explicit CPU lifecycle recovery; worker/model source bundle stays frozen."""
    import os,hashlib
    path=os.environ.get('OMNI_REMOTE_AGENT_PATH')
    if not path:return (HERE/'remote_agent.py').read_text()
    raw=Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest()==os.environ.get('OMNI_REMOTE_AGENT_SHA256'),
            'remote lifecycle recovery source drift')
    return raw.decode()


def start_result(plan,directory,original,fd):
    """Reconcile a known missing projection before deterministic Docker create."""
    result_path=directory/'START-RESULT.json'
    require(result_path.exists(),'START outcome unknown; observe original controller')
    failed=read(result_path)
    if failed.get('returncode')==0:return failed
    receipt=directory/'START-PROJECTION-RECOVERY-COMPLETE.json'
    if receipt.exists():
        saved=read(receipt)
        require(saved['failed_ref']==pin(result_path) and saved['stage']==plan['stage'], 'START recovery identity drift')
        result=read(saved['result_ref']);require(result.get('returncode')==0,'START recovery failed')
        return result
    parents={str(Path(p['source']).parent) for p in plan['job']['task'].get('resident',{}).get('input_projections',[])}
    require(failed.get('returncode')==1 and not failed.get('unknown') and not failed.get('stdout') and
            any('FileNotFoundError: [Errno 2] No such file or directory: '+repr(p) in failed.get('stderr','') for p in parents),
            'START failed outside known pre-create projection boundary')
    # Probe is read-only and repeated just before the repaired remote agent,
    # which repeats the absence assertion before any stage mutation.
    code="""import json,hashlib,sys,subprocess
from pathlib import Path
stage=Path(sys.argv[1]);job=json.loads((stage/'JOB.json').read_text());root=Path(job['output_root'])
assert job['source_boot']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
assert not (root/'CREATE-INTENT.json').exists() and not (root/'CREATED.json').exists()
name='omni-dp8-'+hashlib.sha256(str(root).encode()).hexdigest()[:20]
p=subprocess.run(['docker','inspect',name],capture_output=True,text=True)
assert p.returncode==1 and ('No such object: '+name in p.stderr or 'No such container: '+name in p.stderr)
print(json.dumps(dict(source_boot=job['source_boot'],controller=name,create_absent=True)))
"""
    probe=directory/'START-PROJECTION-PROBE-RESULT.json'
    if not (directory/'START-PROJECTION-PROBE-INTENT.json').exists():
        original.step(directory,'START-PROJECTION-PROBE',['python3','-B','-c',code,plan['stage']],fd)
    require(probe.exists() and read(probe).get('returncode')==0,'START recovery probe unknown')
    observed=json.loads(read(probe)['stdout'])
    require(observed['source_boot']==plan['job']['source_boot'] and observed['create_absent'] is True,
            'START recovery boot/create mismatch')
    name='START-PROJECTION-REPAIR'
    prior=directory/(name+'-RESULT.json')
    if prior.exists() and read(prior).get('returncode')!=0:
        failure=read(prior)
        require(failure.get('returncode')==1 and not failure.get('unknown') and not failure.get('stdout') and
                'ValueError: projection must stay inside admitted local scope' in failure.get('stderr',''),
                'prior START recovery failed outside known projection scope check')
        # This second CPU-only failure preceded CREATE; repaired agent repeats
        # the live same-boot/absent-controller checks immediately before create.
        name='START-AUXILIARY-PROJECTION-REPAIR'
    if not (directory/(name+'-INTENT.json')).exists():
        original.step(directory,name,['python3','-B','-c',remote_agent_source(),plan['stage'],'--recover-precreate'],fd)
    result=read(directory/(name+'-RESULT.json'))
    require(result.get('returncode')==0,'START recovery failed or unknown; observe same controller')
    write(receipt,dict(failed_ref=pin(result_path),probe_ref=pin(probe),
        result_ref=pin(directory/(name+'-RESULT.json')),stage=plan['stage'],model_replayed=False))
    return result


def execute(config,task,directory,host,operation):
    directory=Path(directory)
    if (directory/'returned/RETURN.json').exists():
        restore_locations(directory/'returned')
        if config.get('archive',{}).get('enabled'):
            from checkpoint_archive import trigger
            trigger(config,directory,read(directory/'TRANSPORT.json'))
        result=read(directory/'returned/RETURN.json')
        if result.get('accounting_ref'):
            from registry_custody import accept_closed
            accept_closed(config['registry'],result['accounting_ref'],directory)
        return result
    plan=prepare(config,task,directory,host,operation)
    import fcntl
    with (directory/'transport.guard').open('ab') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);c=client()
        if not (directory/'STAGE-INTENT.json').exists():
            c.step(directory,'STAGE',bootstrap_command(c,directory,plan['job']['image']),lock.fileno(),input=Path(plan['stage_tar_ref']['path']).read_bytes())
        require(stage_result(config,plan,directory,c,lock.fileno())['returncode']==0,'unknown stage: reconcile same identity')
        if not (directory/'START-INTENT.json').exists():
            if plan['job'].get('cold_assets'):
                from cold_assets import ensure
                ensure(config,plan,directory)
            require(stage_result(config,plan,directory,c,lock.fileno())['returncode']==0,'unknown stage: reconcile same identity')
            code=remote_agent_source()
            c.step(directory,'START',['python3','-B','-c',code,plan['stage']],lock.fileno())
        start_result(plan,directory,c,lock.fileno())
        if task.get('program_control_only'):
            from program_bridge import complete_client
            complete_client(config,task['request_ref'],directory,plan,c,lock.fileno())
        # Read-only observation only after an intent; never second create/start.
        while time.time()<config['deadline_epoch']:
            code=closed_reader_source()
            args=['python3','-B','-c',code,plan['target'],str(plan['job']['closed_return_max_bytes'])]
            if task.get('checkpoint_storage_mode')=='lora_training_state' or task.get('newprocess_replay',{}).get('enabled'):
                args += ['--full-checkpoint']
            if task.get('resident',{}).get('enabled') and not task.get('_resident_attach'):
                args += ['--resident-command','--command-root='+str(Path(plan['target'])/'control/package/hot/resident/results/000000')]
            try:result=pull_stream(directory,args,lock.fileno(),config['deadline_epoch'])
            except (OSError,subprocess.TimeoutExpired):result=None
            if result and result.returncode==0:
                unpack_closed(directory/'closed-return.tar',directory/'returned')
                result=read(directory/'returned/RETURN.json')
                if result.get('accounting_ref'):
                    from registry_custody import accept_closed
                    accept_closed(config['registry'],result['accounting_ref'],directory)
                if config.get('archive',{}).get('enabled'):
                    from checkpoint_archive import trigger
                    trigger(config,directory,plan)
                return result
            if result and result.returncode not in (75,255):
                write(directory/'PULL-FAILURE.json',dict(returncode=result.returncode,stderr=result.stderr))
                raise RuntimeError('closed result pull failed; original task retained')
            # Terminal events wake closed-result retrieval; progress stays 900s.
            wait_terminal(config,task,directory,plan)
    raise RuntimeError('overall deadline reached; original task/unknown intent retained')


_PULL_GUARDS={}

def _release_pull(directory):
    guard=_PULL_GUARDS.pop(str(Path(directory).resolve()),None)
    if guard is not None:guard.close()

def pull_stream(directory,argv,fd,deadline):
    import fcntl
    directory=Path(directory);run=read(directory/'REQUEST.json')['run']
    cache=Path(run.get('checkpoint_cache_root',directory/'checkpoint-inventory'))
    cache.mkdir(parents=True,exist_ok=True)
    key=str(directory.resolve());require(key not in _PULL_GUARDS,'same-process pull transaction already active')
    guard=(cache/'transfer.guard').open('ab')
    try:
        while True:
            try:fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                require(time.time()<deadline,'inventory transfer guard deadline')
                time.sleep(min(.1,max(0,deadline-time.time())))
        _PULL_GUARDS[key]=guard
        result=_pull_stream_locked(directory,argv,fd,deadline)
        if result.returncode!=0:_release_pull(directory)
        return result
    except BaseException:
        _release_pull(directory);guard.close();raise


def _pull_stream_locked(directory,argv,fd,deadline):
    """SSH stdout goes straight to durable local disk, never capture_output."""
    import os,shlex
    directory=Path(directory);run=read(directory/'REQUEST.json')['run']
    from checkpoint_inventory import inventory
    cache=Path(run.get('checkpoint_cache_root',directory/'checkpoint-inventory'))
    known=inventory(cache)
    argv=list(argv)+['--inventory-json='+json.dumps(known,sort_keys=True,separators=(',',':'))]
    if run.get('source_boot'):argv+=['--expected-boot='+run['source_boot']]
    write(directory/('PULL-INVENTORY-'+str(time.time_ns())+'.json'),dict(cache_root=str(cache),verified_objects=known,source_boot=run.get('source_boot'),read_only_closed_pull=True))
    command=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',run['host'],shlex.join(argv)]
    target=directory/'closed-return.tar';pending=directory/'closed-return.pending'
    io_timeout=run.get('closed_pull_timeout_seconds',3600)
    require(type(io_timeout) in (int,float) and 0<io_timeout<=86400,'closed pull timeout must be 0..86400 seconds')
    with pending.open('wb') as output,(directory/'pull.stderr').open('wb') as errors:
        result=subprocess.run(command,stdout=output,stderr=errors,pass_fds=(fd,),timeout=min(io_timeout,max(1,deadline-time.time())))
        output.flush();os.fsync(output.fileno())
    if result.returncode==0:
        os.replace(pending,target);parent=os.open(directory,os.O_DIRECTORY);os.fsync(parent);os.close(parent)
    result.stderr=(directory/'pull.stderr').read_text(errors='replace')[-4000:]
    return result


def unpack_closed(data,destination):
    """Original typed transport interface; durable hash inventory, exact graph."""
    from checkpoint_inventory import unpack
    destination=Path(destination)
    request=destination.parent/'REQUEST.json'
    run=read(request)['run'] if request.is_file() else {}
    cache=Path(run.get('checkpoint_cache_root',destination.parent/'checkpoint-inventory'))
    try:return unpack(data,destination,cache,expected_boot=run.get('source_boot'))
    finally:_release_pull(destination.parent)


def training(config,spec,directory):
    if config.get('resident',{}).get('enabled'):
        from resident_transport import training as resident_training
        return resident_training(config,spec,directory)
    return execute(config,spec,directory,'npu2','training')


def rollout(config,request_ref,directory):
    from program_placement import client_owned
    local_program=client_owned(config,read(request_ref)['stage'])
    task=dict(request_ref=request_ref)
    if local_program:task['program_control_only']=True
    result=execute(config,task,Path(directory)/'transport','npu','rollout')
    if local_program:
        original=Path(directory)/'transport/CLIENT-PROGRAM-RESULT.json'
        require(result.get('program_control_closed') is True and
                result['client_result_sha256']==pin(original)['sha256'],'provider/client result differs')
        write(Path(directory)/'RESPONSE.json',read(original))
        return
    require('response_ref' in result,'rollout result response absent')
    # Returned response is already closed by original owner. Copy exact receipt
    # to client mailbox; referenced files remain in closed return location map.
    response=read(result['response_ref'])
    write(Path(directory)/'RESPONSE.json',response)


def restore_locations(destination):
    """Rebuild closed custody after process restart before reading any return."""
    destination=Path(destination).resolve()
    require((destination/'CLOSED-COMMIT.json').is_file(),'unpublished partial closed return; resume original incoming archive')
    commit=read(destination/'CLOSED-COMMIT.json')
    expected_manifest=commit.get('manifest_sha256',commit.get('manifest_ref',{}).get('sha256'))
    require(commit['immutable'] and sha(destination/'CLOSED-MANIFEST.json')==expected_manifest,'closed manifest commit drift')
    manifest=json.loads((destination/'CLOSED-MANIFEST.json').read_text())
    paths=json.loads((destination/'LOCATION-MAP.json').read_text())
    expected={v['path']:str(destination/k) for k,v in manifest.items()}
    require(paths==expected,'closed location mapping drift')
    hashed={}
    for name,ref in manifest.items():
        file=(destination/name).resolve()
        require(file.is_relative_to(destination) and file.is_file(),'closed mapping escaped')
        stat=file.stat();key=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
        if key not in hashed:hashed[key]=sha(file)
        require(stat.st_size==ref['bytes'] and hashed[key]==ref['sha256'],'closed mapped bytes changed')
    if commit.get('wire_sha256'):
        require(sha(destination/'WIRE-MANIFEST.json')==commit['wire_sha256'],'closed wire commit drift')
        wire=json.loads((destination/'WIRE-MANIFEST.json').read_text())
        require(wire['files']==manifest and wire['closure']==commit['closure'],'closed graph/identity drift')
        from checkpoint_inventory import validate_manifest
        validate_manifest(wire)
        closure=commit['closure'];closed=json.loads((destination/next(k for k,v in manifest.items() if v['path']==closure['closed_path'])).read_text())
        require(closed['immutable'] and closed['source_boot']==closure['source_boot'] and closed['return_ref']['sha256']==closure['return_sha256'],'closed source identity drift')
    from dependencies import register_locations
    register_locations(paths)
    return paths


def initialize_anchor_code(anchor, image):
    """Create only the configured project namespace using existing Docker rights."""
    import shlex
    inner="""import os,sys
from pathlib import Path
anchor=Path(sys.argv[1]);base=Path(sys.argv[2]);uid=int(sys.argv[3]);gid=int(sys.argv[4])
assert uid>0 and gid>0 and anchor.is_relative_to(base) and anchor!=base
assert base.resolve()==base and base.is_dir()
current=base
for part in anchor.relative_to(base).parts:
 current=current/part
 assert not current.is_symlink()
 if not current.exists():
  current.mkdir(mode=0o750);os.chown(current,uid,gid)
  fd=os.open(current.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
 assert current.is_dir()
assert anchor.stat().st_uid==uid and anchor.stat().st_gid==gid
fd=os.open(anchor,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
"""
    return ("anchor=Path("+repr(str(anchor))+")\ncapacity(anchor)\n"
        "if not anchor.exists():\n"
        " parent=anchor.parent\n"
        " while not parent.exists():parent=parent.parent\n"
        " require(parent.resolve()==parent and parent!=Path('/'),'unsafe anchor ancestor')\n"
        " argv=['docker','run','--rm','--network','none','--read-only','--log-driver','none',"
        "'--mount','type=bind,src='+str(parent)+',dst='+str(parent),"
        "'--entrypoint','python3',"+repr(image)+",'-B','-c',"+repr(inner)+",str(anchor),str(parent),str(os.getuid()),str(os.getgid())]\n"
        " subprocess.run(argv,check=True)\n"
        "require(anchor.resolve()==anchor and anchor.is_dir() and os.access(anchor,os.W_OK|os.X_OK),'anchor unavailable to execution user')\n"
        "capacity(anchor)\nfsync_dir(anchor)\n")


def stage_result(config,plan,directory,original,fd):
    """Reconcile only a known pre-extraction mkdir denial; never retry unknown."""
    failed=read(directory/'STAGE-RESULT.json')
    if failed.get('returncode')==0:return failed
    receipt=directory/'STAGE-ANCHOR-RECOVERY-COMPLETE.json'
    if receipt.exists():
        saved=read(receipt)
        require(saved['failed_ref']==pin(directory/'STAGE-RESULT.json') and
                saved['stage_tar_ref']==plan['stage_tar_ref'], 'stage recovery identity drift')
        recovered=read(saved['result_ref']);require(recovered['returncode']==0,'recovered stage failed')
        return recovered
    run=read(directory/'REQUEST.json')['run'];anchor=run['remote_storage_anchor']
    require(failed.get('returncode')==1 and not failed.get('unknown') and
            not failed.get('stdout') and 'PermissionError: [Errno 13] Permission denied: '+repr(anchor) in failed.get('stderr','') and
            "anchor.mkdir(parents=True,exist_ok=True)" in read(directory/'STAGE-INTENT.json')['argv'][3] and
            not (directory/'START-INTENT.json').exists(), 'unknown stage: reconcile same identity')
    probe_code="""import json,sys
from pathlib import Path
r=json.loads(sys.argv[1]);assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==r['source_boot']
assert not Path(r['remote_stage']).exists() and not Path(r['remote_root']).exists()
print(json.dumps(dict(stage_absent=True,run_absent=True,source_boot=r['source_boot'])))
"""
    probe=directory/'STAGE-ANCHOR-PROBE-RESULT.json'
    if not (directory/'STAGE-ANCHOR-PROBE-INTENT.json').exists():
        original.step(directory,'STAGE-ANCHOR-PROBE',['python3','-B','-c',probe_code,json.dumps(run)],fd)
    require(probe.exists() and read(probe).get('returncode')==0,'anchor recovery probe unknown')
    observation=json.loads(read(probe)['stdout'])
    require(observation==dict(stage_absent=True,run_absent=True,source_boot=run['source_boot']), 'stage already materialized')
    name='STAGE-ANCHOR-REPAIR'
    if not (directory/(name+'-INTENT.json')).exists():
        original.step(directory,name,bootstrap_command(original,directory,plan['job']['image']),fd,
                      input=Path(plan['stage_tar_ref']['path']).read_bytes())
    recovered=read(directory/(name+'-RESULT.json'))
    require(recovered.get('returncode')==0,'stage repair failed or unknown; preserve original receipts')
    write(receipt,dict(failed_ref=pin(directory/'STAGE-RESULT.json'),probe_ref=pin(probe),
        result_ref=pin(directory/(name+'-RESULT.json')),stage_tar_ref=plan['stage_tar_ref'],model_replayed=False))
    return recovered


def bootstrap_command(original,directory,image):
    # Reuse exact original tar/storage/fsync bootstrap with configured CPU image.
    import types
    scoped=types.FunctionType(original.bootstrap.__code__,dict(original.bootstrap.__globals__,IMAGE=image),
        original.bootstrap.__name__,original.bootstrap.__defaults__,original.bootstrap.__closure__)
    argv=scoped(directory)
    require(argv[:3]==['python3','-B','-c'] and len(argv)==4,'original bootstrap command contract')
    run=read(Path(directory)/'REQUEST.json')['run']
    anchor=Path(run['remote_storage_anchor'])
    require(anchor.is_absolute() and '..' not in anchor.parts and
            len(anchor.parts)>=3 and not anchor.is_relative_to('/mnt/cephfs'),
            'explicit project-local storage anchor required')
    for key in ('remote_stage','remote_root'):
        p=Path(run[key]);require(p.is_relative_to(anchor) and '..' not in p.parts,
            'bootstrap path outside configured anchor')
    old="capacity(subprocess.check_output(['docker','info','--format','{{.DockerRootDir}}'],text=True).strip())"
    require(argv[3].count(old)==1,'original bootstrap host storage seam changed')
    setup=("require(Path('/proc/sys/kernel/random/boot_id').read_text().strip()=="+
           repr(run['source_boot'])+",'bootstrap source boot changed')\n"+
           initialize_anchor_code(anchor,image))
    argv[3]=argv[3].replace(old,setup+old,1)
    # The original root bootstrap owns extracted files but only chowned the
    # final run parent. Newly created scope ancestors must belong to the same
    # explicit uid before cold_assets configures its guard as that user.
    import ast
    tree=ast.parse(argv[3])
    assignments=[n for n in tree.body if isinstance(n,ast.Assign) and
        len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='argv']
    require(len(assignments)==1,'original bootstrap Docker argv assignment')
    node=assignments[0].value;docker=ast.literal_eval(node)
    index=docker.index('-c')+1;inner=docker[index]
    seam='root.parent.mkdir(parents=True,exist_ok=True);os.chown(root.parent,uid,gid);fsync_dir(root.parent)'
    require(inner.count(seam)==1,'original bootstrap run parent ownership seam')
    replacement=('missing=[];parent=root.parent\nwhile not parent.exists():\n'
        ' missing.append(parent);parent=parent.parent\n'
        'for parent in reversed(missing):\n'
        ' parent.mkdir();os.chown(parent,uid,gid);fsync_dir(parent.parent)\n'
        'os.chown(root.parent,uid,gid);fsync_dir(root.parent)')
    docker[index]=inner.replace(seam,replacement,1)
    literal=ast.get_source_segment(argv[3],node)
    require(argv[3].count(literal)==1,'unambiguous bootstrap Docker literal')
    argv[3]=argv[3].replace(literal,repr(docker),1)
    return argv

#!/usr/bin/env python3
"""Change one idle resident's native context from pinned deployment configuration."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
SCHEMA = 'saea-configured-engine-transition-v1'


def sibling(name):
    spec = importlib.util.spec_from_file_location('engine_' + name, Path(__file__).with_name(name + '.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


settlement = sibling('configured_terminal_settlement')
queue, launch, require = settlement.queue, settlement.launcher, settlement.require
from speech_aware_evidence_acquisition.core import deployment


def sha(value):return hashlib.sha256(value).hexdigest()
def read(path):return settlement.read(Path(path))
def pin(value):return queue.pin_path(value)


def publish(path, value):
    """Immutable durable JSON publication; a completed publication is idempotent."""
    if path.exists():
        require(read(path) == value, 'existing transition evidence differs')
        return
    queue.atomic(path, value)


def context(args):
    require(isinstance(args, list) and args.count('--max-model-len') == 1, 'one explicit native context required')
    index = args.index('--max-model-len')
    require(index + 1 < len(args) and re.fullmatch('[1-9][0-9]*', args[index+1]), 'numeric context required')
    return int(args[index+1])


def without_context(args):
    result = list(args); result[result.index('--max-model-len')+1] = 'CONTEXT'
    return result


def inspect(container):
    result = launch.run(['docker', 'container', 'inspect', container])
    if result.returncode:
        require('No such container' in result.stderr or 'No such object' in result.stderr, 'Docker inspect failed')
        return None
    doc, = json.loads(result.stdout)
    return doc


def load(path):
    doc = read(path)
    keys = {'schema', 'source_settlement', 'old_container_id', 'old_profile', 'old_model',
        'new_profile', 'new_model', 'native_model_config', 'logical_model_fragment',
        'new_session', 'data_root', 'nic', 'scope', 'state_dir', 'ready_timeout_seconds', 'ready_interval_seconds'}
    require(set(doc) - {'source_retirement', 'original_profile', 'original_instance_index'} == keys and doc['schema'] == SCHEMA, 'exact engine transition configuration required')
    source, producer, release, logical, expected, _ = settlement.load(pin(doc['source_settlement']))
    paths = {key: pin(doc[key]) for key in ('old_profile','old_model','new_profile','new_model','native_model_config')}
    old, new = (deployment.load_deployment_profile(paths[key], data_root=str(logical)) for key in ('old_profile','new_profile'))
    old_model, new_model = (read(paths[key]) for key in ('old_model','new_model'))
    require(len(old.instances) == len(new.instances) == 1 and old.effective_namespace_policy == new.effective_namespace_policy == 'shared-pid-host',
        'one host-PID resident instance required')
    require(isinstance(doc['scope'], dict) and set(doc['scope']) == {'cards','port'}
        and list(old.instances[0].cards) == list(new.instances[0].cards) == doc['scope']['cards']
        and old.instances[0].port == new.instances[0].port == doc['scope']['port'], 'exact unchanged lane scope required')
    require(re.fullmatch('sha256:[a-f0-9]{64}', new.image) and new.image == old.image, 'same pinned engine image required')
    old_doc, new_doc = read(paths['old_profile']), read(paths['new_profile'])
    for value in (old_doc, new_doc):
        for key in ('profile_id', 'model_fragment', 'basis'):value.pop(key, None)
        value['instances'][0].pop('name')
    require(old_doc == new_doc and old.instances[0].name != new.instances[0].name, 'only profile identity and native context may change')
    original=old; original_index=0
    if 'original_profile' in doc:
        original=deployment.load_deployment_profile(pin(doc['original_profile']),data_root=str(logical))
        original_index=doc.get('original_instance_index')
        require(type(original_index) is int and 0 <= original_index < len(original.instances), 'original instance index required')
        oi=original.instances[original_index]; li=old.instances[0]
        require((oi.name,oi.cards,oi.port,oi.cpuset,oi.env)==(li.name,li.cards,li.port,li.cpuset,li.env), 'original instance and lane differ')
        require(all(getattr(original,k)==getattr(old,k) for k in ('image','model_path','bind_host','tensor_parallel_size','max_num_seqs','gpu_memory_utilization','max_num_batched_tokens','mm_processor_cache_type','namespace_policy')), 'original profile geometry differs')
    else:require('original_instance_index' not in doc,'original profile required')
    require(old_model['model_lock_key'] == new_model['model_lock_key']
        and old_model['model_materialization_root'] == new_model['model_materialization_root'] == str(logical),
        'model weights identity must not change')
    old_args, new_args = old_model['server_args'], new_model['server_args']
    old_protocol, new_protocol = copy.deepcopy(old_model), copy.deepcopy(new_model)
    for value in (old_protocol, new_protocol):
        value['server_args'] = without_context(value['server_args'])
        value.pop('runtime_receipt', None); value.pop('geometry_basis', None)
    require(old_protocol == new_protocol, 'decoding and all non-context model fields must remain frozen')
    require(without_context(old_args) == without_context(new_args) and context(new_args) > context(old_args)
        and old_model['served_model_name'] == new_model['served_model_name'], 'context-only engine argv required')
    native = read(paths['native_model_config']); text = native.get('thinker_config', {}).get('text_config', {})
    maximum = text.get('max_position_embeddings')
    rope = text.get('rope_scaling')
    require(type(maximum) is int and context(new_args) <= maximum
        and (rope is None or isinstance(rope, dict) and rope.get('rope_type', rope.get('type', 'default')) == 'default'),
        'target context exceeds pinned native default-RoPE capability')
    require(doc['data_root'] == str(logical) and new.model_path == old.model_path
        and paths['native_model_config'] == Path(old.model_path)/'config.json', 'unchanged native model/config binding required')
    require(re.fullmatch('[a-f0-9]{64}', doc['old_container_id']), 'full exact old container ID required')
    for key in ('logical_model_fragment','new_session'):
        require(isinstance(doc[key], str) and re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]*', doc[key]), 'confined logical identity required')
    require(doc['new_session'] != expected['service_session'], 'fresh session identity required')
    require(type(doc['ready_timeout_seconds']) is int and 1 <= doc['ready_timeout_seconds'] <= 3600
        and type(doc['ready_interval_seconds']) is int and 1 <= doc['ready_interval_seconds'] <= 60, 'bounded service readiness timing required')
    deployment.validated_nic(doc['nic'])
    out = launch.absolute(doc['state_dir'])
    require(out.absolute() == out.resolve() and len(out.parts) >= 4, 'owned local transition state required')
    locks = settlement.mapped(producer, logical/'locks', writable=True)
    serving = settlement.mapped(producer, logical/'serving')
    require(not out.is_relative_to(locks) and not out.is_relative_to(serving), 'transition output must not overlap live state')
    receipt = read(pin(source['runtime_receipt']))
    require(receipt['container']['image_id'] == old.image and receipt['protocol']['server_args'] == old_args
        and len(receipt['deployment']['instances']) == 1
        and receipt['deployment']['instances'][0]['device_ids'] == doc['scope']['cards']
        and receipt['deployment']['instances'][0]['port'] == doc['scope']['port'], 'old runtime differs from source lane')
    argv = deployment.docker_run_argv(new, new.instances[0], new_model, nic=doc['nic'])
    return {'doc':doc,'source':source,'producer':producer,'release':release,'logical':logical,'expected':expected,
        'old':old,'new':new,'old_model':old_model,'new_model':new_model,'receipt':receipt,
        'out':out,'locks':locks,'serving':serving,'argv':argv,'original':original,'original_index':original_index}


def verify_container(container, profile, model, *, exact_id=None, require_running=True, instance_index=0):
    instance = profile.instances[instance_index]
    require(container is not None and (exact_id is None or container['Id'] == exact_id)
        and container['Name'] == '/' + instance.name and container['Image'] == profile.image, 'engine container identity differs')
    labels = container['Config'].get('Labels') or {}
    require(labels.get(deployment.LABEL_PROFILE) == profile.profile_id
        and labels.get(deployment.LABEL_PROFILE_SHA256) == profile.sha256
        and labels.get(deployment.LABEL_INSTANCE) == str(instance_index), 'engine profile labels differ')
    require(container['Config']['Entrypoint'] == [deployment.CONTAINER_INTERPRETER]
        and container['Config']['Cmd'] == deployment.instance_container_args(profile, instance, model),
        'engine process command differs')
    host = container['HostConfig']
    require(host['PidMode'] == host['NetworkMode'] == 'host' and host['RestartPolicy']['Name'] == 'no'
        and host['CpusetCpus'] == instance.cpuset, 'engine process namespace or CPU placement differs')
    actual_devices = {v['PathOnHost'] for v in host['Devices']}
    expected_devices = {*deployment.ASCEND_CONTROL_DEVICES, *(f'/dev/davinci{v}' for v in instance.cards)}
    require(actual_devices == expected_devices and container.get('RestartCount', 0) == 0, 'engine devices or restart history differs')
    if require_running:
        require(container['State']['Running'] is True and container['State']['Status'] == 'running'
            and type(container['State']['Pid']) is int and container['State']['Pid'] > 0, 'engine is not running')
    return container


def retirement_barrier(c):
    """Validate explicit unknown-retirement custody without converting it to settlement."""
    from speech_aware_evidence_acquisition.contracts import ExecutionPlan
    from speech_aware_evidence_acquisition.core.contact_journal import ContactJournal, canonical, digest
    retirement = read(pin(c['doc']['source_retirement']))
    require(set(retirement)=={'schema','manifest','complete','lease','attempt_before','attempt_aborted','marker','journal_root','config_sha256'}
        and retirement['schema']=='saea-engine-retirement-source-v1', 'exact retirement source required')
    paths={k:pin(retirement[k]) for k in ('manifest','complete','lease','attempt_before','attempt_aborted','marker')}
    manifest,done,lease,before,after,marker=(read(paths[k]) for k in ('manifest','complete','lease','attempt_before','attempt_aborted','marker'))
    expected=c['expected'];registered=read(pin(c['source']['source_plan'])); values=dict(registered['execution_plan'])
    if 'owner_allowance' in registered:values['owner_allowance']=registered['owner_allowance']
    plan=ExecutionPlan(**values).validate(source_root=Path(c['producer']['repo_path']))
    require((plan.run_id,plan.attempt_id)==(expected['run_id'],expected['attempt_id']), 'retirement plan identity differs')
    require(done.get('schema')=='saea-unknown-attempt-retirement-v1' and done.get('status')=='RETIRED_UNKNOWN_RESOURCE_RELEASED'
        and done.get('config_sha256')==retirement['config_sha256'] and done.get('manifest_sha256')==launch.digest(paths['manifest'])
        and done.get('automatic_retry') is False and done.get('scientific_success') is False and done.get('new_model_calls')==0,
        'retirement completion differs')
    require(manifest.get('config_sha256')==retirement['config_sha256'] and manifest.get('run_id')==plan.run_id
        and manifest.get('attempt_id')==plan.attempt_id and manifest.get('unknown_responses_remain_unknown') is True
        and manifest.get('automatic_retry') is False and manifest.get('lease_sha256')==launch.digest(paths['lease']), 'retirement manifest differs')
    require(marker.get('run_id')==plan.run_id and marker.get('attempt_id')==plan.attempt_id
        and marker.get('manifest_sha256')==launch.digest(paths['manifest']) and marker.get('automatic_retry') is False, 'retirement marker differs')
    require(before.get('state')=='OPENED' and after.get('state')=='ABORTED' and before.get('actual')==after.get('actual')==manifest.get('cost')==done.get('actual'), 'retired attempt cost differs')
    require(lease.get('run_id')==plan.run_id and lease.get('attempt_id')==plan.attempt_id and lease.get('session_id')==expected['service_session']
        and lease.get('study_head')==expected['study_commit'] and lease.get('run_ids')==[plan.run_id], 'retired lease identity differs')
    live_attempt=settlement.mapped(c['producer'],c['logical']/'attempts'/plan.run_id/(plan.attempt_id+'.json'))
    require(live_attempt.read_bytes()==paths['attempt_aborted'].read_bytes(), 'live retired attempt differs')
    live_root=settlement.mapped(c['producer'],c['logical']/'contact-journals'/plan.run_id/plan.attempt_id)
    require(live_root==Path(retirement['journal_root']) and paths['marker']==live_root/'retirement.json', 'live retirement marker location differs')
    journal=ContactJournal(c['logical'],plan,c['source']['arm']);journal.root=live_root
    from retire_unknown_attempt import inventory
    snapshot=inventory(journal)
    require(snapshot==manifest['snapshot'] and snapshot['calls']==after['actual']['calls']
        and snapshot['unknown_count']==done['unknown_count']==marker['unknown_count']
        and done['observed_count']==snapshot['calls']-snapshot['unknown_count'], 'retired journal inventory differs')
    archive=paths['manifest'].parent
    for name,expected_sha in snapshot['files'].items():require(launch.digest(archive/name)==expected_sha,'retirement archive journal differs')
    for name,expected_sha in manifest['fit_files'].items():require(launch.digest(archive/'fit'/name)==expected_sha,'retirement archive FIT differs')
    require(launch.digest(archive/'plan.json')==launch.digest(pin(c['source']['source_plan']))
        and launch.digest(archive/'checkpoint.jsonl')==manifest['checkpoint_sha256'], 'retirement archive plan/checkpoint differs')
    return {'mode':'explicit-unknown-retirement','complete_sha256':launch.digest(paths['complete']),
            'manifest_sha256':launch.digest(paths['manifest']),'unknown_count':snapshot['unknown_count'],
            'observed_count':done['observed_count'],'cost':after['actual'],'automatic_retry':False}


def source_barrier(c):
    """Independent branches require release OR exact completed source settlement."""
    d, producer, expected = c['source'], c['producer'], c['expected']
    if 'source_retirement' in c['doc']:
        return retirement_barrier(c)
    if c['release'].exists():
        observed = launch.status(producer, queue.canonical_hash(producer), d['producer_launcher']['sha256'])
        result = read(Path(producer['receipt_dir'])/'launch-result.json')
        intent = read(Path(producer['receipt_dir'])/'launch-intent.json')
        require(intent.get('expected_commit') == expected['study_commit']
            and result.get('returncode') == 0 and result.get('stdout', '').strip() == observed.get('container_id')
            and observed.get('state') in ('running','exited'), 'source release launch custody differs')
        proof = queue.verify_generation_release(d, producer, d['generation_release'])
        mode = 'sealed-generation-release'
    else:
        complete = Path(d['state_dir'])/'complete.json'; proof = read(complete)
        audit_path = complete.with_name('audit.json'); audit = read(audit_path)
        require(proof.get('schema') == settlement.SCHEMA and proof.get('settled') is True
            and proof.get('identity', {}).get('config_sha256') == c['doc']['source_settlement']['sha256']
            and proof.get('audit_sha256') == launch.digest(audit_path)
            and audit.get('identity') == proof['identity'], 'exact completed source settlement required')
        terminal = settlement.producer_terminal(d, producer)
        require(terminal == audit['terminal'], 'settled producer terminal changed')
        if audit.get('mode') == 'stale-lease':
            require(launch.digest(audit['preserved_lease']) == audit['lease_sha256'], 'settled lease evidence changed')
        mode = 'terminal-settlement'
    # Release can precede CPU finalization; only the native contact custody must be complete.
    from speech_aware_evidence_acquisition.contracts import ExecutionPlan
    from speech_aware_evidence_acquisition.core.contact_journal import ContactJournal
    registered = read(pin(d['source_plan'])); values = dict(registered['execution_plan'])
    if 'owner_allowance' in registered:values['owner_allowance'] = registered['owner_allowance']
    plan = ExecutionPlan(**values).validate(source_root=Path(producer['repo_path']))
    require((plan.run_id,plan.attempt_id) == (expected['run_id'],expected['attempt_id']), 'barrier source plan differs')
    journal = ContactJournal(c['logical'], plan, d['arm'])
    journal.root = settlement.mapped(producer, c['logical']/'contact-journals'/plan.run_id/plan.attempt_id)
    attempt = read(settlement.mapped(producer, c['logical']/'attempts'/plan.run_id/(plan.attempt_id+'.json')))
    require(attempt.get('run_id') == plan.run_id and attempt.get('attempt_id') == plan.attempt_id
        and attempt.get('carrier_lock_key') == plan.carrier_lock_key, 'source attempt identity differs')
    files = sorted(journal.root.glob('*')) if journal.root.exists() else []
    intents = [p for p in files if p.name.endswith('.intent.json')]
    require(len(files) == len(intents)*2 and len(intents) == attempt['actual']['calls'], 'unknown or orphan source contacts')
    pins = []
    for p in intents:
        require(p.is_file() and not p.is_symlink(), 'regular native custody required')
        rid = read(p)['request_id']; contact = journal.inspect(rid)
        require(p == journal._path(rid, 'intent') and contact['state'] == 'observed', 'unknown source request')
        response = journal._path(rid,'response');require(not response.is_symlink(), 'redirected response')
        pins.append({'request_id':rid,'intent_sha256':launch.digest(p),'response_sha256':launch.digest(response)})
    return {'mode':mode,'proof':proof,'contacts':pins}


def old_sessions(c):
    receipt = c['receipt']; names = [c['expected']['service_session'], receipt['deployment']['instances'][0]['instance_id']]
    blobs = {}
    for name in names:
        for leaf in ('session.json','session-receipt.json'):
            p = c['serving']/name/leaf; blobs[str(Path(name)/leaf)] = read(p)
        doc = blobs[str(Path(name)/'session.json')]
        require(doc['session_id'] == name and doc['identity']['runtime_receipt_sha256'] == c['source']['runtime_receipt']['sha256'],
                'old resident session custody differs')
    return blobs


def verify_live_old(c, old, sessions):
    from speech_aware_evidence_acquisition.core.session.receipt import SessionReceipt
    from speech_aware_evidence_acquisition.core.session.vllm_identity import verify_vllm_session
    instance = c['receipt']['deployment']['instances'][0]
    for name in [c['expected']['service_session'],instance['instance_id']]:
        saved = SessionReceipt.from_json(sessions[str(Path(name)/'session-receipt.json')])
        require(saved.pid == old['State']['Pid'], 'old resident PID differs from exact engine')
        verify_vllm_session(saved, c['receipt'])


def wait_ready(c, container_id, *, sleep=time.sleep, now=time.monotonic, opener=urllib.request.urlopen):
    deadline = now() + c['doc']['ready_timeout_seconds']
    instance = c['new'].instances[0]; base = c['new'].base_url(instance)
    while True:
        current = verify_container(inspect(container_id), c['new'], c['new_model'], exact_id=container_id)
        try:
            with opener(base+'/health', timeout=5) as response:require(response.status == 200, 'engine not healthy')
            with opener(base+'/v1/models', timeout=5) as response:models = json.loads(response.read())
            require(c['new_model']['served_model_name'] in [m['id'] for m in models.get('data',[])], 'wrong served model')
            return current
        except (OSError, ValueError):
            require(now() < deadline, 'new engine readiness timeout; preserve pending transition')
            sleep(c['doc']['ready_interval_seconds'])


def npu_inventory(c):
    # The coordinator has no device permissions; use one pinned device-read probe.
    argv=['docker','run','--rm','--network','none']
    for device in (*deployment.ASCEND_CONTROL_DEVICES, *(f'/dev/davinci{v}' for v in c['new'].cards)):
        argv += ['--device',device]
    for mount in deployment.ASCEND_READONLY_MOUNTS:
        argv += ['-v',f'{mount}:{mount}:ro']
    argv += ['--entrypoint','/usr/local/sbin/npu-smi',c['new'].image,'info']
    result=launch.run(argv)
    require(result.returncode==0,'NPU inventory unreadable')
    return result.stdout


def finish_registration(c, current):
    """Reverify immutable runtime/weights and observed live argv before publication."""
    from speech_aware_evidence_acquisition.contracts.gate_runtime_vllm import (
        protocol_sha256, validate_vllm_receipt_document, verify_vllm_runtime_files, verify_model_directory)
    from speech_aware_evidence_acquisition.core.session.vllm_identity import verify_vllm_process_identity
    from speech_aware_evidence_acquisition.core.session.vllm_resident import register_deployment
    receipt = copy.deepcopy(c['receipt']); receipt['protocol']['server_args'] = list(c['new_model']['server_args'])
    instance = receipt['deployment']['instances'][0]; instance['instance_id'] = c['new'].instances[0].name
    receipt['protocol_sha256'] = protocol_sha256(receipt)
    validate_vllm_receipt_document(receipt); verify_vllm_runtime_files(receipt)
    verify_model_directory(receipt, c['logical'], str(Path(receipt['model']['model_dir']).relative_to(c['logical'])))
    verify_vllm_process_identity(current['State']['Pid'], receipt, instance, host=instance['host'], port=instance['port'])
    path = c['out']/'runtime-receipt.json'; publish(path, receipt)
    log = launch.run(['docker','logs',current['Id']]); require(log.returncode == 0, 'new startup logs unreadable')
    # Freeze the first startup observation; later log growth must not change session identity.
    observations_path = c['out']/'observations.json'
    if observations_path.exists():observations = read(observations_path)
    else:
        observations = {instance['instance_id']:{'pid':current['State']['Pid'],
            'started_at':current['State']['StartedAt'], 'startup_stdout_sha256':sha((log.stdout+log.stderr).encode())}}
        publish(observations_path, observations)
    require(observations[instance['instance_id']]['pid'] == current['State']['Pid']
        and observations[instance['instance_id']]['started_at'] == current['State']['StartedAt'], 'new engine process changed')
    # The configured logical data root must expose the identical lane serving directory.
    require(os.path.samefile(c['logical']/'serving', c['serving']), 'logical serving mount is not the source lane')
    files = register_deployment(data_root=c['logical'], fragment=c['new_model'],
        model_fragment=c['doc']['logical_model_fragment'], receipt_path=path, instances=observations, session_id=c['doc']['new_session'])
    return {'runtime_receipt':{'path':str(path),'sha256':launch.digest(path)},
        'observations':{'path':str(observations_path),'sha256':launch.digest(observations_path)},
        'registered_files':[{'path':str(p),'sha256':launch.digest(p)} for p in files]}


def execute(path, mode='run'):
    c = load(path); d = c['doc']; out = c['out']
    identity = {'config_sha256':launch.digest(path),'tool_sha256':launch.digest(__file__)}
    if mode == 'plan':
        verify_container(inspect(d['old_container_id']), c['original'], c['old_model'], exact_id=d['old_container_id'], instance_index=c['original_index'])
        require(inspect(c['new'].instances[0].name) is None, 'new engine already exists')
        return {'schema':SCHEMA,'identity':identity,'argv':c['argv'],'new_model_calls':0,'mutates_services':False}
    require(mode in ('run','recover-stop'), 'plan, run or recover-stop required')
    out.mkdir(parents=True, exist_ok=True); settlement.sync(out.parent)
    lease = c['locks']/'flight.lock'; pending = c['locks']/'engine-transition.pending.json'
    with settlement.guard(c['locks']/'generation.guard') as (check_guard, guard_identity):
        require(not os.path.lexists(lease), 'active or unknown lease blocks engine transition')
        complete = out/'complete.json'
        if complete.exists():
            done = read(complete);require(done['identity'] == identity, 'completed engine transition identity changed')
            for value in [done['runtime_receipt'], done['observations'], *done['registered_files']]:pin(value)
            if pending.exists():
                marker=read(pending);require(marker['identity']==identity and marker['state_dir']==str(out),
                    'different transition pending after completion')
                pending.unlink();settlement.sync(pending.parent)
            return done
        intent_path = out/'intent.json'
        if pending.exists():
            marker = read(pending);require(marker['state_dir'] == str(out) and marker['identity'] == identity, 'another or changed transition is pending')
        old = inspect(d['old_container_id']);verify_container(old,c['original'],c['old_model'],exact_id=d['old_container_id'],require_running=not intent_path.exists(),instance_index=c['original_index'])
        proof = source_barrier(c)
        if not intent_path.exists():
            require(inspect(c['new'].instances[0].name) is None, 'unowned new engine')
            sessions = old_sessions(c);verify_live_old(c,old,sessions)
            quiet = settlement.transition.quiescence(c['receipt']['deployment']['instances'])
            logs = launch.run(['docker','logs',d['old_container_id']]);require(logs.returncode == 0,'old engine logs unreadable')
            check_guard();require(not lease.exists(), 'lease appeared before transition')
            intent = {'schema':SCHEMA,'identity':identity,'guard':guard_identity,'source_proof':proof,
                'old_container':old,'old_sessions':sessions,'old_logs':{'stdout':logs.stdout,'stderr':logs.stderr},
                'quiescence':quiet,'new_argv':c['argv'],'new_model_calls':0}
            # Commit the global barrier before any crash can expose a partial transition.
            publish(pending,{'schema':SCHEMA,'state_dir':str(out),'identity':identity})
            publish(intent_path,intent)
        else:
            intent = read(intent_path);require(intent['identity']==identity and intent['guard']==guard_identity
                and intent['new_argv']==c['argv'] and intent['source_proof']==proof, 'transition evidence changed')
        publish(pending,{'schema':SCHEMA,'state_dir':str(out),'identity':identity})
        stopped = out/'old-stopped.json'
        if not stopped.exists():
            stop_intent = out/'stop-intent.json'
            if old['State']['Running']:
                if stop_intent.exists():
                    require(mode == 'recover-stop', 'previous stop outcome unresolved; use explicit recover-stop')
                    require(read(stop_intent)['container_id'] == d['old_container_id'], 'original stop identity differs')
                    verify_live_old(c,old,intent['old_sessions'])
                    settlement.transition.quiescence(c['receipt']['deployment']['instances'])
                    recovery_intent=out/'stop-recovery-intent.json'
                    require(not recovery_intent.exists(), 'explicit stop recovery already attempted; preserve unknown outcome')
                    publish(recovery_intent,{'container_id':d['old_container_id'],'verified_running':True,
                        'original_intent_sha256':launch.digest(stop_intent),'new_model_calls':0})
                    result_path=out/'stop-recovery-result.json'
                else:
                    require(mode == 'run', 'recover-stop requires an existing uncertain stop')
                    publish(stop_intent,{'container_id':d['old_container_id'],'new_model_calls':0})
                    result_path=out/'stop-result.json'
                result = launch.run(['docker','stop','--time','60',d['old_container_id']])
                publish(result_path,{'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
                require(result.returncode == 0, 'old engine stop failed; preserve transition')
                old = inspect(d['old_container_id'])
            require(old['State']['Status']=='exited' and old['State']['Running'] is False and old['State']['Pid']==0, 'old engine not conclusively stopped')
            publish(stopped,old)
        # Preserve old registration and mark stopped only after exact terminal engine evidence.
        stopped_doc = read(stopped)
        require(stopped_doc['Id'] == d['old_container_id'] and old['State']==stopped_doc['State'], 'old engine terminal changed')
        for rel, value in intent['old_sessions'].items():
            require(read(c['serving']/rel) == value, 'old session evidence changed')
        for name in [c['expected']['service_session'], c['receipt']['deployment']['instances'][0]['instance_id']]:
            publish(c['serving']/name/'session-stop.json',{'schema':'saea-serving-session-stop-v1',
                'session_id':name,'transition_state':str(out),'old_container_id':d['old_container_id'],
                'stopped_at':stopped_doc['State']['FinishedAt'],'stale':False,'new_model_calls':0})
        new_name = c['new'].instances[0].name; new = inspect(new_name);launch_intent = out/'launch-intent.json'
        if not launch_intent.exists():
            require(new is None, 'unattributed new engine exists')
            table = npu_inventory(c)
            allocation = deployment.allocate_from_npu_smi(c['new'],table)
            publish(out/'allocation.json',{'allocation':allocation,'npu_smi':table})
            check_guard();require(not lease.exists(),'lease appeared before engine start')
            publish(launch_intent,{'identity':identity,'argv':c['argv'],'automatic_retry':False})
            result = launch.run(c['argv'])
            publish(out/'launch-result.json',{'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
            require(result.returncode == 0,'new engine launch failed; do not automatically repeat')
            new = inspect(new_name)
        else:
            require(read(launch_intent)=={'identity':identity,'argv':c['argv'],'automatic_retry':False},'engine launch intent changed')
            require(new is not None,'unresolved engine launch intent; no automatic retry')
        new = verify_container(new,c['new'],c['new_model'])
        if (out/'launch-result.json').exists():
            result=read(out/'launch-result.json');require(result['returncode']==0 and result['stdout'].strip()==new['Id'],'new engine launch result differs')
        publish(out/'container-binding.json',{'container_id':new['Id'],'started_at':new['State']['StartedAt'],'image':new['Image']})
        new = wait_ready(c,new['Id'])
        registered = finish_registration(c,new)
        check_guard();require(not lease.exists(),'lease appeared before transition completion')
        done={'schema':SCHEMA,'identity':identity,'old_container_id':d['old_container_id'],'new_container_id':new['Id'],
            'old_context':context(c['old_model']['server_args']),'new_context':context(c['new_model']['server_args']),
            'new_session':d['new_session'],'logical_model_fragment':d['logical_model_fragment'],
            'source_identity_preserved':True,'old_container_preserved':True,'new_model_calls':0,
            'successor_requires_new_preflight':True,**registered}
        publish(complete,done);pending.unlink();settlement.sync(pending.parent)
        return done


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('plan','run','recover-stop'))
    parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
    print(json.dumps(execute(args.config,args.mode),sort_keys=True))

#!/usr/bin/env python3
"""Exercise the native custody chain using pinned assets and isolated fixture state.

The source attempt may be live: only already written row and contact bytes are
read. The generated ABORTED/producer observations are explicitly test fixtures.
Run only in a network-disabled container with source and factory data read-only.
"""
import argparse
import json
import math
import os
from dataclasses import asdict, replace
from pathlib import Path

from speech_aware_evidence_acquisition.contracts import ExecutionPlan
from speech_aware_evidence_acquisition.core.contact_journal import ContactJournal, canonical, comparable_request, digest, logical_contact
from speech_aware_evidence_acquisition.core.native_fit_recovery import NativeFitJournal
from speech_aware_evidence_acquisition.core.preparation import _artifact_bytes, digest as object_digest
from speech_aware_evidence_acquisition.core.preparation_fit import boundary_identity
from speech_aware_evidence_acquisition.core.row_checkpoint import RowCheckpointWriter, read_checkpoint
import configured_terminal_custody_package as package
import configured_checkpoint_successor as compiler


def put(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(obj) if not isinstance(obj, bytes) else obj)


def pin(path):
    path = Path(path)
    return {'path': str(path), 'sha256': digest(path.read_bytes())}


def run(config_path):
    c = json.loads(Path(config_path).read_bytes())
    root = Path(c['output']); root.mkdir(exist_ok=False)
    data = root / 'data'; data.mkdir()
    template = json.loads(Path(c['template']['path']).read_bytes())
    assert pin(c['template']['path']) == c['template']
    held = json.loads(Path(c['held']['path']).read_bytes())
    assert pin(c['held']['path']) == c['held']
    cfg = held['config']
    plan_doc = json.loads(Path(cfg['plan']['path']).read_bytes())
    plan = ExecutionPlan(**plan_doc['execution_plan']).validate(source_root=cfg['repo'])
    manifest = json.loads(Path(template['assets_manifest']['path']).read_bytes())
    for p in manifest['files'].values(): assert pin(p['path']) == p
    roster_pin = manifest['files'][c['roster_relpath']]
    roster = json.loads(Path(roster_pin['path']).read_bytes())
    assert len(roster['items']) == c['roster_count']
    ids = {x['sample_id'] for x in roster['items']}
    source = Path(c['source_state'])
    source_journal = ContactJournal(source, plan, 'raw-speech-agent-native-tools', source_root=cfg['repo'])
    source_cp = source / 'runs' / plan.run_id / plan.attempt_id / (plan.run_id + '.rows.ckpt.jsonl')
    # Snapshot through the requested row; verify the entire chain, including
    # every trace, before trusting any field from that row.
    lines = []
    with source_cp.open('rb') as f:
        for line in f:
            assert line.endswith(b'\n'), 'selected row not found before torn tail'
            lines.append(line)
            if json.loads(line).get('sample_id') == c['sample_id']:
                break
        else:
            raise ValueError('selected sample not found in checkpoint')
    prefix = root / 'source-prefix.jsonl'; put(prefix, b''.join(lines))
    original = read_checkpoint(prefix)
    row = original.rows[-1]
    assert row.sample_id == c['sample_id'] and row.sample_id in ids
    assert row.keep_defect() is None
    roles = {visit['role_id'] for output in row.output_rows
             for visit in output.get('native_episode_receipt', {}).get('role_visits', [])}
    assert {'perception', 'slot', 'solver'} <= roles
    source_record = json.loads(lines[-1])
    put(root/'source-provenance.json', {'source_checkpoint': str(source_cp), 'prefix': pin(prefix),
        'sample_id': row.sample_id, 'source_seq': row.seq, 'source_row_sha': source_record['sha'],
        'source_row_bytes_sha256': digest(lines[-1]), 'trace_sha256': row.trace_sha256,
        'role_visits': sorted(roles)})
    trace = [json.loads(line) for line in row.trace_lines]
    requests = [x['payload']['payload'] for x in trace if x['channel'] == 'MODEL_REQUEST' and x['record_id'].endswith('/request')]
    assert len(requests) == 3 and all(p['sample_id'] == row.sample_id for p in requests)
    roster_item = next(item for item in roster['items'] if item['sample_id'] == row.sample_id)
    for payload in requests:
        query = payload['raw_speech_query']
        assert query['media_relpath'] == payload['media_relpath']
        media = (Path(c['factory_data']) / query['media_relpath']).resolve()
        assert media.is_relative_to(Path(c['factory_data']).resolve())
        assert digest(media.read_bytes()) == query['sha256']
        assert roster_item['sample_id'] == payload['sample_id']
    writer = RowCheckpointWriter(data/'runs'/plan.run_id/plan.attempt_id, plan.run_id, plan.attempt_id,
        identity={k:v for k,v in original.header.items() if k not in ('schema','kind','run_id','attempt_id','seq','prev_sha','sha')})
    writer.append_row(sample_id=row.sample_id, complete=row.complete, output_rows=row.output_rows, trace_lines=row.trace_lines, reason=row.reason)
    writer.close(); checkpoint = writer.path
    journal = ContactJournal(data, plan, 'raw-speech-agent-native-tools', source_root=cfg['repo'])
    fit_root = data/'prepared'/plan_doc['execution_plan']['run_id']/'fixture-fit'; fit_root.mkdir(parents=True)
    logical_fit = Path(c['logical_fit_root'])
    audio = []
    session_id = c['session_id']
    for seq, payload in enumerate(requests, 1):
        rid = payload['request_id']; observed = source_journal.inspect(rid)
        assert observed['state'] == 'observed' and observed['intent']['payload'] == payload
        journal.intent(payload); journal.response(rid, digest(canonical(payload)), observed['response_bytes'], observed['response']['latency_seconds'])
        audio.append(payload['audio_seconds'])
        custody = json.loads(NativeFitJournal(source, plan, 'raw-speech-agent-native-tools').path(payload).read_bytes())
        assert comparable_request(custody['payload']) == comparable_request(payload)
        old_artifact = source / 'prepared' / logical_fit.parent.name / logical_fit.name / Path(custody['receipt']['path']).name
        original_artifact = json.loads(old_artifact.read_bytes())
        assert digest(old_artifact.read_bytes()) == custody['receipt']['sha256']
        inputs, output = original_artifact['inputs'], original_artifact['output']
        boundary = boundary_identity((payload,))
        inputs['sources']['boundary_sha256'] = boundary
        inputs['native_contact']['raw_speech_query'] = payload['raw_speech_query']
        output['boundary_sha256'] = boundary
        output['cumulative_contacts'] = seq
        artifact = fit_root / old_artifact.name
        put(artifact, _artifact_bytes(kind='request-fit', inputs=inputs, output=output, contact_class='tokenizer'))
        identity = logical_contact(payload); key = object_digest(identity)
        endpoint = Path(str(artifact)+'.contacts.jsonl')
        fit_plan = inputs['fit_plan']
        put(endpoint, canonical({**identity, 'contact_identity_sha256':key, 'contact':seq, 'method':'POST', 'route':'/tokenize',
            'run_id':fit_plan['run_id'], 'attempt_id':fit_plan['attempt_id'], 'generation_run_id':plan.run_id, 'session_id':session_id})+b'\n')
        receipt = {**identity,'contact_identity_sha256':key,'contact_class':'tokenizer','path':str(artifact),'sha256':pin(artifact)['sha256']}
        NativeFitJournal(data, plan, 'raw-speech-agent-native-tools').record(payload, {k:v for k,v in inputs.items() if k not in ('native_contact','fit_plan')}, receipt, custody['measured_prompt_tokens'], custody['session_identity'])
        cp = NativeFitJournal(data, plan, 'raw-speech-agent-native-tools').path(payload)
        doc = json.loads(cp.read_bytes()); doc['receipt']['path'] = str(logical_fit/artifact.name); put(cp, doc)
    attempt = {'schema':'saea-attempt-v1','state':'ABORTED','run_id':plan.run_id,'attempt_id':plan.attempt_id,
        'carrier_lock_key':plan.carrier_lock_key,'actual':{'calls':len(requests),'audio_seconds':math.fsum(audio)}}
    attempt_path = data/'attempts'/plan.run_id/(plan.attempt_id+'.json'); put(attempt_path,attempt)
    receipt_pin = manifest['files'][c['receipt_relpath']]; receipt_doc = json.loads(Path(receipt_pin['path']).read_bytes())
    session = root/'session.json'; put(session, {'session_id':session_id,'identity':{'runtime_receipt_sha256':receipt_pin['sha256']},
        'deployment_instance_ids':[x['instance_id'] for x in receipt_doc['deployment']['instances']]})
    launch = root/'launch.json'; argv = ['/fixture/python','/fixture/run','--data','/fixture/data']
    put(launch, {'argv':argv,'generation_lease':{'lock_directory':'/fixture/locks'},
        'mounts':[{'source':'/fixture/locks','target':'/fixture/data/locks','readonly':False}],
        'environment':{'SAEA_OBSERVABILITY_RUN_ID':plan.run_id,'SAEA_OBSERVABILITY_ATTEMPT_ID':plan.attempt_id}})
    intent = root/'launch-intent.json'; put(intent, {'config_sha256':pin(launch)['sha256']})
    result = root/'launch-result.json'; put(result, {'returncode':0,'stdout':'fixture-container\n'})
    live_config = root/'producer-config.json'; put(live_config, {'schema':'saea-producer-observation-config-v1','plan':cfg['plan'],
        'runtime_receipt':receipt_pin,'session':pin(session),'session_id':session_id,'launch_config':pin(launch),'launch_intent':pin(intent),'launch_result':pin(result)})
    producer = root/'producer.json'; put(producer, {'schema':'saea-producer-process-observation-v1','plan':cfg['plan'],'runtime_receipt':receipt_pin,
        'container_id':'fixture-container','producer_pid':10,'producer_start_ticks':1,'boot_id':'fixture-boot','guard':{'device':1,'inode':2},
        'run_id':plan.run_id,'attempt_id':plan.attempt_id,'observed_before_exit':0.5,'config':pin(live_config),
        'launch_config':pin(launch),'launch_intent':pin(intent),'launch_result':pin(result),'process_cmdline':argv,'lock_directory':'/fixture/locks',
        'lease_sha256':'a'*64,'lease_identity':{'run_id':plan.run_id,'attempt_id':plan.attempt_id,'pid':10,'session_id':session_id},
        'session_id':session_id,'session':pin(session)})
    observation = root/'natural-observation.json'; put(observation, {'schema':'saea-natural-terminal-observation-v1','mode':'natural-drain',
        'container':{'id':'fixture-container','state':{'Running':False,'Status':'exited','Pid':0}},'boot_id':'fixture-boot','producer_pid':10,
        'session_id':session_id,'guard':{'device':1,'inode':2},'lease_absent_under_guard':True,'producer_start_ticks':1,
        'producer_identity_sha256':pin(producer)['sha256'],'run_id':plan.run_id,'attempt_id':plan.attempt_id,'plan_sha256':cfg['plan']['sha256'],
        'attempt_sha256':pin(attempt_path)['sha256'],'runtime_receipt_sha256':receipt_pin['sha256'],'session_sha256':pin(session)['sha256'],
        'observed_requests':sorted(p['request_id'] for p in requests), 'quiescence':{'queues':{str(x['port']):{'vllm:num_requests_running':0,'vllm:num_requests_waiting':0} for x in receipt_doc['deployment']['instances']},'established_connections':[],'observed_at':1.0},'captured_at':1.0})
    ancestor = {'attempt_id':plan.attempt_id,'plan':cfg['plan'],'checkpoint':pin(checkpoint),'generation_root':str(data),
        'prepared_root':str(fit_root),'logical_prepared_root':str(logical_fit),'roster':roster_pin,'bridge_key':plan.attempt_id,
        'terminal_mode':'natural-drain','reconciliation':pin(observation),'lease':{'absent':True},'producer_identity':pin(producer)}
    # Factory data is supplied through a separate read-only mount by the caller.
    manifest['files'][c['extra_asset_relpath']] = c['extra_asset']
    assert pin(c['extra_asset']['path']) == c['extra_asset']
    manifest_path = root/'source-manifest.json'; put(manifest_path, manifest)
    template['assets_manifest'] = pin(manifest_path)
    template['data'] = c['factory_data']; template['output'] = str(root/'compiled'); template['admission']['workspace'] = str(root/'workspace')
    template_path = root/'compiler-template.json'; put(template_path, template)
    cfg.update(data=c['factory_data'], output=str(root/'package'), resume_from={'attempt_id':plan.attempt_id,'checkpoint_sha256':pin(checkpoint)['sha256']},
        ancestors=[ancestor], evidence=[cfg['plan'],pin(checkpoint),roster_pin,pin(observation),receipt_pin,pin(producer),pin(session),pin(live_config),pin(launch),pin(intent),pin(result)])
    cfg['compiler']['template'] = pin(template_path)
    package.build(cfg)
    compiled = compiler.execute(root/'package'/'compiler-finalize.json')
    assert compiled['status'] == 'HELD' and compiled['activation_authorized'] is False
    assert compiled['new_model_calls'] == compiled['new_fit_calls'] == 0
    validation = json.loads((root/'package'/'validation.json').read_bytes())
    assert validation['ancestors'][0]['checkpoint_native_keepable_rows'] >= 1
    put(root/'result.json', {'fixture_only':True,'registered_roster_count':len(ids),'source_rows':1,'source_contacts':len(requests),
        'source_provenance':pin(root/'source-provenance.json'),'compiler':compiled})
    return compiled


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--config', required=True)
    print(json.dumps(run(parser.parse_args().config), sort_keys=True))

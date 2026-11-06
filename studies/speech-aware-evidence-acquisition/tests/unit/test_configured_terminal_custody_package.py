import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts' / 'environment'))
from checkpoint_terminal_custody import inspect_generation, validate_terminal_custody
from speech_aware_evidence_acquisition.core.contact_journal import ContactJournal, canonical, digest
from test_frozen_core_adapter import _plan, _payload


def test_generation_observed_and_terminal(tmp_path):
    plan = _plan()
    journal = ContactJournal(tmp_path, plan, 'bare-core')
    payload = _payload(request_id='req-1', history=[{'role': 'control', 'contact_id': 'solver:1'}])
    journal.intent(payload)
    journal.response('req-1', digest(canonical(payload)), b'{}', 0.1)
    path = tmp_path / 'attempts' / plan.run_id / (plan.attempt_id + '.json')
    path.parent.mkdir(parents=True)
    attempt = {'schema': 'saea-attempt-v1', 'state': 'ABORTED', 'run_id': plan.run_id,
               'attempt_id': plan.attempt_id, 'carrier_lock_key': plan.carrier_lock_key,
               'actual': {'calls': 1, 'audio_seconds': payload['audio_seconds']}}
    path.write_bytes(canonical(attempt))
    assert inspect_generation(data_root=tmp_path, plan=plan, arm='bare-core')['observed_requests'] == ['req-1']
    attempt['state'] = 'OPENED'
    path.write_bytes(canonical(attempt))
    with pytest.raises(ValueError, match='ABORTED'):
        inspect_generation(data_root=tmp_path, plan=plan, arm='bare-core')
    attempt['state'] = 'ABORTED'
    path.write_bytes(canonical(attempt))
    journal._path('req-1', 'response').unlink()
    with pytest.raises(ValueError, match='unknown'):
        inspect_generation(data_root=tmp_path, plan=plan, arm='bare-core')


def test_fit_join_and_missing_custody(tmp_path):
    from dataclasses import asdict, replace
    from checkpoint_terminal_custody import inspect_fit
    from speech_aware_evidence_acquisition.core.contact_journal import logical_contact
    from speech_aware_evidence_acquisition.core.native_fit_recovery import NativeFitJournal
    from speech_aware_evidence_acquisition.core.preparation import _artifact_bytes, digest as object_digest
    from speech_aware_evidence_acquisition.core.preparation_fit import boundary_identity

    plan = _plan()
    fit_plan = replace(plan, run_id='SAEA-FIT-UNIT-TEST', attempt_id='fit-v1', planned_model_calls=3)
    journal = ContactJournal(tmp_path, plan, 'raw-speech-agent-native-tools')
    payload = _payload(request_id='req-fit', history=[{'role': 'control', 'contact_id': 'solver:1'}])
    payload['raw_speech_query'] = {'media_relpath': 'audio.wav', 'sha256': 'a' * 64}
    identity = logical_contact(payload)
    key = object_digest(identity)
    session = {'model': 'fixture'}
    boundary = boundary_identity((payload,))
    inputs = {'config_sha256': 'b' * 64, 'sources': {'boundary_sha256': boundary, 'required_identity': session}}
    native = {**identity, 'contact_identity_sha256': key, 'capacity_scope': 'actual-variable-trajectory-contact',
              'raw_speech_query': payload['raw_speech_query']}
    artifact_inputs = {**inputs, 'native_contact': native, 'fit_plan': asdict(fit_plan)}
    output = {**identity, 'contact_identity_sha256': key, 'boundary_sha256': boundary,
              'every_request_fits': True, 'rows': [{'measured_prompt_tokens': 42}], 'session_id': 'session',
              'session_identity_sha256': object_digest(session), 'contacts': 1,
              'contact_counts': {'POST /tokenize': 1}, 'cumulative_contacts': 1}
    root = tmp_path / 'prepared' / fit_plan.run_id / fit_plan.attempt_id
    root.mkdir(parents=True)
    artifact = root / (key + '.request-fit.json')
    artifact.write_bytes(_artifact_bytes(kind='request-fit', inputs=artifact_inputs, output=output, contact_class='tokenizer'))
    endpoint = Path(str(artifact) + '.contacts.jsonl')
    endpoint.write_bytes(canonical({**identity, 'contact_identity_sha256': key, 'contact': 1,
        'method': 'POST', 'route': '/tokenize', 'run_id': fit_plan.run_id, 'attempt_id': fit_plan.attempt_id,
        'generation_run_id': plan.run_id, 'session_id': 'session'}) + b'\n')
    kwargs = dict(journal=journal, prepared_root=root, logical_prepared_root=root, fit_plan=fit_plan,
                  session_identity=session, max_contacts=3, roster=[identity['sample_id']])
    assert inspect_fit(**kwargs)['missing_custody'] == 1
    receipt = {**identity, 'contact_identity_sha256': key, 'contact_class': 'tokenizer',
               'path': str(artifact), 'sha256': digest(artifact.read_bytes())}
    NativeFitJournal(tmp_path, plan, journal.identity['arm']).record(payload, inputs, receipt, 42, session)
    result = inspect_fit(**kwargs)
    assert result['endpoint_calls'] == 1 and result['missing_custody'] == 0
    fit_only_payload = _payload(request_id='fit-only',sample_id='5000002',
        history=[{'role':'control','contact_id':'solver:5000002'}])
    fit_only_payload['raw_speech_query'] = {'media_relpath':'audio2.wav','sha256':'c'*64}
    fit_only_identity = logical_contact(fit_only_payload)
    key2 = object_digest(fit_only_identity)
    fit_plan2 = replace(fit_plan, planned_model_calls=6)
    native2 = {**fit_only_identity,'contact_identity_sha256':key2,'capacity_scope':'actual-variable-trajectory-contact',
               'raw_speech_query':fit_only_payload['raw_speech_query']}
    boundary2 = boundary_identity((fit_only_payload,))
    inputs2 = {'config_sha256':'b'*64,'sources':{'boundary_sha256':boundary2,'required_identity':session}}
    output2 = {**output,**fit_only_identity,'contact_identity_sha256':key2,'boundary_sha256':boundary2,'cumulative_contacts':2}
    artifact2 = root/(key2+'.request-fit.json')
    artifact2.write_bytes(_artifact_bytes(kind='request-fit',inputs={**inputs2,'native_contact':native2,'fit_plan':asdict(fit_plan2)},output=output2,contact_class='tokenizer'))
    endpoint2 = Path(str(artifact2)+'.contacts.jsonl')
    endpoint2.write_bytes(canonical({**fit_only_identity,'contact_identity_sha256':key2,'contact':2,
        'method':'POST','route':'/tokenize','run_id':fit_plan2.run_id,'attempt_id':fit_plan2.attempt_id,
        'generation_run_id':plan.run_id,'session_id':'session'})+b'\n')
    # The first artifact must carry the same authenticated FIT plan.
    artifact.write_bytes(_artifact_bytes(kind='request-fit', inputs={**inputs,'native_contact':native,'fit_plan':asdict(fit_plan2)}, output=output, contact_class='tokenizer'))
    custody = next(journal.root.glob('*.fit.json')); custody_doc = json.loads(custody.read_bytes())
    custody_doc['receipt']['sha256'] = digest(artifact.read_bytes()); custody.write_bytes(canonical(custody_doc))
    kwargs.update(fit_plan=fit_plan2, roster=[identity['sample_id'],fit_only_identity['sample_id']])
    result = inspect_fit(**kwargs)
    assert result['endpoint_calls'] == 2 and result['missing_custody'] == 1
    artifact.unlink()
    with pytest.raises(ValueError, match='unknown'):
        inspect_fit(**kwargs)


def test_checkpoint_header_and_digest(tmp_path):
    from checkpoint_terminal_custody import inspect_checkpoint
    from speech_aware_evidence_acquisition.core.row_checkpoint import RowCheckpointWriter
    plan = _plan()
    writer = RowCheckpointWriter(tmp_path, plan.run_id, plan.attempt_id,
        identity={'protocol_hash': plan.protocol_hash, 'split_identity_hash': plan.split_identity_hash})
    writer.close()
    checkpoint_hash = digest(writer.path.read_bytes())
    assert inspect_checkpoint(path=writer.path, sha256=checkpoint_hash, plan=plan).header['run_id'] == plan.run_id
    with pytest.raises(ValueError, match='header'):
        inspect_checkpoint(path=writer.path, sha256=checkpoint_hash, plan=_plan(protocol_hash='c' * 64))
    with pytest.raises(Exception, match='hashes'):
        inspect_checkpoint(path=writer.path, sha256='0' * 64, plan=plan)


def test_inventory_detects_new_file_and_byte_changes(tmp_path):
    from checkpoint_terminal_custody import inventory_snapshot
    path = tmp_path / 'one'
    path.write_bytes(b'original')
    inventory = [{'root': str(tmp_path), 'files': {'one': digest(b'original')}}]
    assert inventory_snapshot(inventory)
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='inventory'):
        inventory_snapshot(inventory)
    path.write_bytes(b'original')
    (tmp_path / 'orphan').write_bytes(b'extra')
    with pytest.raises(ValueError, match='inventory'):
        inventory_snapshot(inventory)


def test_native_settlement_pins_and_observed_set(tmp_path):
    from checkpoint_terminal_custody import inspect_settlement
    attempt = {'state': 'ABORTED', 'run_id': 'run', 'attempt_id': 'old'}
    lock = {'run_id': 'run', 'attempt_id': 'old', 'pid': 10, 'session_id': 'session'}
    lockpath = tmp_path / 'preserved-flight.lock.json'
    lockpath.write_bytes(canonical(lock))
    receipt = {'container': 'container-id', 'state': {'Running': False, 'Status': 'exited', 'Pid': 0},
               'attempt': attempt, 'observed_requests': ['request'],
               'quiescence': {'queues': {'8200': {'vllm:num_requests_running': 0, 'vllm:num_requests_waiting': 0}}, 'established_connections': [], 'observed_at': 1.0},
               'lease_sha256': digest(lockpath.read_bytes()), 'no_model_calls': True}
    recpath = tmp_path / 'reconciliation.json'
    recpath.write_bytes(canonical(receipt))
    pin = lambda p: {'path': str(p), 'sha256': digest(p.read_bytes())}
    assert inspect_settlement(reconciliation=pin(recpath), lease=pin(lockpath), attempt=attempt,
                              observed=['request']) == lock
    with pytest.raises(ValueError, match='settlement'):
        inspect_settlement(reconciliation=pin(recpath), lease=pin(lockpath), attempt=attempt, observed=[])


def test_full_terminal_fixture(tmp_path):
    import subprocess
    import copy
    from dataclasses import asdict, replace
    from _vllm_world import VllmWorld, PROTOCOL_ARGS
    from speech_aware_evidence_acquisition.core.config import _compose_fragments, protocol_hash
    from speech_aware_evidence_acquisition.core.flight.checkpoint_compatibility import IMPLEMENTATION, STUDY_PREFIX
    from speech_aware_evidence_acquisition.core.native_fit_recovery import NativeFitJournal
    from speech_aware_evidence_acquisition.core.preparation import _artifact_bytes, artifact_inputs, digest as object_digest
    from speech_aware_evidence_acquisition.core.preparation_fit import boundary_identity
    from speech_aware_evidence_acquisition.core.row_checkpoint import RowCheckpointWriter
    from speech_aware_evidence_acquisition.core.session.vllm_resident import required_identity
    from speech_aware_evidence_acquisition.data.splits import prefixed, split_identity_hash

    source = Path(__file__).resolve().parents[2]
    world = VllmWorld(tmp_path / 'world')
    repo = tmp_path / 'git'; repo.mkdir()
    study = repo / STUDY_PREFIX
    data = tmp_path / 'data'; data.mkdir()
    sample = '5000001'; sample2 = '5000002'; arm = 'raw-speech-agent-native-tools'
    split = split_identity_hash(prefixed('earnings22-original', [sample, sample2]))
    names = {'model':'m','dataset':'d','baseline':'b','experiment':'e'}
    model = {'transport':'vllm', 'server_args':PROTOCOL_ARGS, 'model_materialization_root':str(world.materialization_root),
             'runtime_receipt':'receipts/runtime-vllm.json'}
    supply_path = data/'supply.json'
    supply_path.write_bytes(canonical({'output': {'items': [sample, sample2]}}))
    supply_pin = {'relpath': 'supply.json', 'sha256': digest(supply_path.read_bytes())}
    base = {'prepared_phase_supply': supply_pin, 'arm':arm, 'policy':'phase-native-intent-slot-batched-v1',
            'native_contact_fit':{'run_id':'SAEA-FIT-UNIT-TEST','attempt_id':'fit-v1','contact_cap':6}}
    source_plan = _plan(split_identity_hash=split)
    exp = {'cells':[{'cell_id':'fixture', 'model_fragment':'m', 'dataset_fragment':'d', 'baseline_fragment':'b', 'run_id':source_plan.run_id,'attempt_id':source_plan.attempt_id}],
           'execution_plan':{k:getattr(source_plan,k) for k in ('execution_profile','split_role','planned_model_calls','planned_speech_audio_seconds','planned_gpu_hours')}}
    docs = {'model':model,'dataset':{'fixture_dataset':'test'},'baseline':base,'experiment':exp}
    def put(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical(value) if not isinstance(value, bytes) else value)
    for kind in names:
        put(study/'configs'/kind/(names[kind]+'.json'), docs[kind])
    policy = 'phase-native-intent-slot-batched-v1'
    put(study/'configs/policy'/(policy+'.json'), (source/'configs/policy'/(policy+'.json')).read_bytes())
    implementation = {'runtime':'core/flight/runtime.py','transport':'core/model/transport.py',
        'preparation_fit':'core/preparation_fit.py','preparation':'core/preparation.py','config':'core/config.py',
        'runtime_view':'rails/adapters/runtime_view.py','core_client':'rails/core_client.py',
        'speech_features':'core/model/speech_features.py'}
    for rel in implementation.values():
        p = 'src/speech_aware_evidence_acquisition/' + rel
        put(study/p, (source/p).read_bytes())
    for rel in IMPLEMENTATION.values():
        if not (study/rel).exists(): put(study/rel, (source/rel).read_bytes())
    receipt_path = study/'docs/receipts/runtime-vllm.json'
    put(receipt_path, world.receipt())
    for args in (['init','-q'], ['add','.'], ['-c','user.name=Test','-c','user.email=test@localhost','commit','-qm','fixture']):
        subprocess.run(['git',*args],cwd=repo,check=True,capture_output=True)
    def git(*args): return subprocess.check_output(['git',*args],cwd=repo)
    commit = git('rev-parse','HEAD').decode().strip()
    obj = tmp_path/'commit'; obj.write_bytes(git('cat-file','commit',commit))
    listing = tmp_path/'listing'; listing.write_bytes(git('ls-tree','-r','--full-tree',commit))
    archive = tmp_path/'archive.tar'; archive.write_bytes(git('archive',commit))
    def pin(path): return {'path':str(path), 'sha256':digest(path.read_bytes())}
    from speech_aware_evidence_acquisition.core.data_root_placeholder import collapse_data_root
    normalized = {k: collapse_data_root(v, str(world.materialization_root)) for k,v in docs.items()}
    composed = _compose_fragments(names, normalized, str(world.materialization_root))
    source_plan = replace(source_plan, protocol_hash=protocol_hash(composed.config_hash, arm, split))
    header = {'config_hash':composed.config_hash,'protocol_hash':source_plan.protocol_hash,'split_identity_hash':split,
              'implementation':{k:digest((study/p).read_bytes()) for k,p in IMPLEMENTATION.items()}}
    writer = RowCheckpointWriter(data/'runs'/source_plan.run_id/source_plan.attempt_id,source_plan.run_id,source_plan.attempt_id,identity=header)
    journal = ContactJournal(data,source_plan,arm)
    contact1 = 'phase-native-intent-slot-batched-v1:perception:0:'+sample
    contact2 = 'phase-native-intent-slot-batched-v1:perception:0:'+sample2
    payload = _payload(request_id='request',history=[{'role':'control','contact_id':contact1}])
    payload['raw_speech_query'] = {'media_relpath':'audio.wav','sha256':'a'*64}
    response_bytes = canonical({'text':'fixture response','usage':{'prompt_tokens':4}})
    response_trace = {'text':'fixture response','response_sha256':digest(response_bytes),
                      'response_text_sha256':digest(b'fixture response'),'usage':{'prompt_tokens':4}}
    journal.intent(payload); journal.response('request',digest(canonical(payload)),response_bytes,0.1)
    payload2 = _payload(request_id='request2', sample_id=sample2, speech_ref='earnings22-original/'+sample2,
                        media_relpath='datasets/earnings21-22/earnings22/media/'+sample2+'.mp3',
                        history=[{'role':'control','contact_id':contact2}])
    payload2['raw_speech_query'] = {'media_relpath':'audio2.wav','sha256':'b'*64}
    journal.intent(payload2); journal.response('request2',digest(canonical(payload2)),response_bytes,0.1)
    for p in (payload, payload2):
        rid = p['request_id']
        trace = [{'channel':'MODEL_REQUEST','record_id':rid+'/request',
                  'payload':{'payload':p,'payload_sha256':digest(canonical(p))}},
                 {'channel':'MODEL_RESPONSE','record_id':rid+'/response',
                  'payload':dict(response_trace)}]
        writer.append_row(sample_id=p['sample_id'], complete=True, output_rows=[{'sample_id':p['sample_id'],
                          'request_id':rid,'request_sha256':digest(canonical(p)),'response_sha256':digest(response_bytes)}],
                          trace_lines=[json.dumps(x) for x in trace])
    writer.close(); checkpoint = writer.path
    attempt = {'schema':'saea-attempt-v1','state':'ABORTED','run_id':source_plan.run_id,'attempt_id':source_plan.attempt_id,
               'carrier_lock_key':source_plan.carrier_lock_key,'actual':{'calls':2,'audio_seconds':200.0}}
    attempt_path = data/'attempts'/source_plan.run_id/(source_plan.attempt_id+'.json'); put(attempt_path,attempt)
    roster_path = tmp_path/'roster.json'
    put(roster_path,{'carrier_lock_key':source_plan.carrier_lock_key,'split_role':source_plan.split_role,
                     'split_identity_hash':split,'items':[{'sample_id':sample},{'sample_id':sample2}]})
    plan_doc = {'execution_plan':asdict(source_plan),'composed_config_hash':composed.config_hash}
    plan_path = tmp_path/'plan.json'; put(plan_path,plan_doc)
    session = required_identity(fragment=model,model_fragment='m',runtime_receipt=world.receipt(),
                                receipt_path=receipt_path,host='127.0.0.1',port=8200)
    boundary = boundary_identity((payload,))
    impl = {k:digest((study/'src/speech_aware_evidence_acquisition'/v).read_bytes()) for k,v in implementation.items()}
    inputs = artifact_inputs(config=composed.values,split_identity_hash=split,sources={'boundary_sha256':boundary,
                   'required_identity':session,'runtime_receipt':world.receipt(),'implementation':impl})
    key = object_digest({'sample_id':sample,'agent_contact_id':contact1})
    native = {'sample_id':sample,'agent_contact_id':contact1,'contact_identity_sha256':key,
              'capacity_scope':'actual-variable-trajectory-contact','raw_speech_query':payload['raw_speech_query']}
    fit_plan = replace(source_plan,run_id=base['native_contact_fit']['run_id'],attempt_id='fit-v1',planned_model_calls=6)
    fit_root = data/'prepared'/fit_plan.run_id/fit_plan.attempt_id; fit_root.mkdir(parents=True)
    artifact = fit_root/(key+'.request-fit.json')
    output = {'sample_id':sample,'agent_contact_id':contact1,'contact_identity_sha256':key,'boundary_sha256':boundary,
              'every_request_fits':True,'rows':[{'measured_prompt_tokens':42}],'session_id':'session',
              'session_identity_sha256':object_digest(session),'contacts':1,'contact_counts':{'POST /tokenize':1},'cumulative_contacts':1}
    artifact.write_bytes(_artifact_bytes(kind='request-fit',inputs={**inputs,'native_contact':native,'fit_plan':asdict(fit_plan)},output=output,contact_class='tokenizer'))
    endpoint = Path(str(artifact)+'.contacts.jsonl')
    put(endpoint,canonical({'sample_id':sample,'agent_contact_id':contact1,'contact_identity_sha256':key,'contact':1,
            'method':'POST','route':'/tokenize','run_id':fit_plan.run_id,'attempt_id':fit_plan.attempt_id,
            'generation_run_id':source_plan.run_id,'session_id':'session'})+b'\n')
    NativeFitJournal(data,source_plan,arm).record(payload,inputs,{'path':str(artifact),'sha256':digest(artifact.read_bytes()),
        'sample_id':sample,'agent_contact_id':contact1,'contact_identity_sha256':key,'contact_class':'tokenizer'},42,session)
    custody_path = next(journal.root.glob('*.fit.json'))
    custody_doc = json.loads(custody_path.read_bytes())
    custody_doc['receipt']['path'] = str(world.materialization_root/'prepared'/fit_plan.run_id/fit_plan.attempt_id/artifact.name)
    custody_path.write_bytes(canonical(custody_doc))
    boundary2 = boundary_identity((payload2,))
    inputs2 = artifact_inputs(config=composed.values,split_identity_hash=split,sources={'boundary_sha256':boundary2,
                   'required_identity':session,'runtime_receipt':world.receipt(),'implementation':impl})
    key2 = object_digest({'sample_id':sample2,'agent_contact_id':contact2})
    native2 = {'sample_id':sample2,'agent_contact_id':contact2,'contact_identity_sha256':key2,
               'capacity_scope':'actual-variable-trajectory-contact','raw_speech_query':payload2['raw_speech_query']}
    artifact2 = fit_root/(key2+'.request-fit.json')
    output2 = {**output,'sample_id':sample2,'agent_contact_id':contact2,'contact_identity_sha256':key2,
               'boundary_sha256':boundary2,'cumulative_contacts':2}
    artifact2.write_bytes(_artifact_bytes(kind='request-fit',inputs={**inputs2,'native_contact':native2,'fit_plan':asdict(fit_plan)},output=output2,contact_class='tokenizer'))
    endpoint2 = Path(str(artifact2)+'.contacts.jsonl')
    put(endpoint2,canonical({'sample_id':sample2,'agent_contact_id':contact2,'contact_identity_sha256':key2,'contact':2,
            'method':'POST','route':'/tokenize','run_id':fit_plan.run_id,'attempt_id':fit_plan.attempt_id,
            'generation_run_id':source_plan.run_id,'session_id':'session'})+b'\n')
    NativeFitJournal(data,source_plan,arm).record(payload2,inputs2,{'path':str(artifact2),'sha256':digest(artifact2.read_bytes()),
        'sample_id':sample2,'agent_contact_id':contact2,'contact_identity_sha256':key2,'contact_class':'tokenizer'},42,session)
    for cp in journal.root.glob('*.fit.json'):
        d = json.loads(cp.read_bytes())
        d['receipt']['path'] = str(world.materialization_root/'prepared'/fit_plan.run_id/fit_plan.attempt_id/Path(d['receipt']['path']).name)
        cp.write_bytes(canonical(d))
    lock_path = tmp_path/'preserved-flight.lock.json'; put(lock_path,{'run_id':source_plan.run_id,'attempt_id':source_plan.attempt_id,'pid':10,'session_id':'session'})
    reconciliation = tmp_path/'reconciliation.json'; put(reconciliation,{'container':'id','state':{'Running':False,'Status':'exited','Pid':0},
        'attempt':attempt,'observed_requests':['request','request2'],'quiescence':{'queues':{str(port):{'vllm:num_requests_running':0,'vllm:num_requests_waiting':0} for port in (8200,8201)},'established_connections':[], 'observed_at':1.0},
        'lease_sha256':digest(lock_path.read_bytes()),'no_model_calls':True})
    dummy = pin(obj)
    bridge = {'schema':'saea-checkpoint-image-release-v1','predecessor_commit':commit,'predecessor_fragments':names,
       'successor_attempt_id':'next','change':'image-release-operational-v1',
       'evidence':{'commit_object':pin(obj),'tree_listing':pin(listing),'source_archive':pin(archive),
          'release_manifest':dummy,'source_review':dummy,'reference_pins':dummy,'startup_data_root':str(world.materialization_root)}}
    roots = [journal.root, attempt_path.parent, checkpoint.parent, fit_root]
    inventory = [{'root':str(root),'files':{p.relative_to(root).as_posix():digest(p.read_bytes()) for p in root.rglob('*') if p.is_file()}} for root in roots]
    ancestor = {'attempt_id':source_plan.attempt_id,'plan':pin(plan_path),'checkpoint':pin(checkpoint),'generation_root':str(data),
        'prepared_root':str(fit_root),'logical_prepared_root':str(world.materialization_root/'prepared'/fit_plan.run_id/fit_plan.attempt_id),'roster':pin(roster_path),'bridge_key':source_plan.attempt_id,
        'reconciliation':pin(reconciliation),'lease':pin(lock_path)}
    index = tmp_path/'index.json'; put(index,{'schema':'saea-checkpoint-terminal-custody-v1','ancestors':[ancestor],'inventory':inventory})
    cfg = {'repo':str(source),'target':{'attempt_id':'next','fit_run_id':'SAEA-FIT-NEXT','fit_attempt_id':'fit-next'},'terminal':{'resume_from':{'attempt_id':source_plan.attempt_id,
           'checkpoint_sha256':digest(checkpoint.read_bytes())},'bridge':bridge,'evidence':[pin(index),pin(plan_path),pin(checkpoint),
           pin(roster_path),pin(reconciliation),pin(lock_path),pin(receipt_path)]}}
    result = validate_terminal_custody(cfg=cfg,candidate={'run_id':source_plan.run_id,'attempt_id':source_plan.attempt_id},plan=plan_doc,data=data)
    assert result['generation_calls'] == 2 and result['fit_endpoint_calls'] == 2
    assert result['ancestors'][0]['checkpoint_complete_rows'] == 2
    assert result['ancestors'][0]['checkpoint_native_keepable_rows'] == 2
    assert result['ancestors'][0]['checkpoint_incomplete_rows'] == {}

    def audit():
        return validate_terminal_custody(cfg=cfg,candidate={'run_id':source_plan.run_id,'attempt_id':source_plan.attempt_id},plan=plan_doc,data=data)
    def reseal():
        for item in inventory:
            root = Path(item['root'])
            item['files'] = {p.relative_to(root).as_posix():digest(p.read_bytes()) for p in root.rglob('*') if p.is_file()}
        put(index,{'schema':'saea-checkpoint-terminal-custody-v1','ancestors':[ancestor],'inventory':inventory})
        cfg['terminal']['evidence'][0] = pin(index)
    response_path = journal._path('request', 'response')
    response_blob = response_path.read_bytes()
    response_path.unlink(); reseal()
    with pytest.raises(ValueError, match='unknown'):
        audit()
    response_path.write_bytes(response_blob); reseal()
    with pytest.raises(ValueError, match='candidate root'):
        validate_terminal_custody(cfg=cfg,candidate={'run_id':source_plan.run_id,'attempt_id':'missing'},plan=plan_doc,data=data)
    original_rec = reconciliation.read_bytes()
    rec = json.loads(original_rec); del rec['quiescence']['queues']['8201']; put(reconciliation,rec)
    ancestor['reconciliation'] = pin(reconciliation); cfg['terminal']['evidence'][4] = pin(reconciliation); reseal()
    with pytest.raises(ValueError, match='quiet endpoints'):
        audit()
    reconciliation.write_bytes(original_rec); ancestor['reconciliation'] = pin(reconciliation); cfg['terminal']['evidence'][4] = pin(reconciliation); reseal()
    bad_plan = copy.deepcopy(plan_doc); bad_plan['execution_plan']['planned_model_calls'] += 1
    put(plan_path,bad_plan); ancestor['plan'] = pin(plan_path); cfg['terminal']['evidence'][1] = pin(plan_path); reseal()
    with pytest.raises(ValueError, match='budget differs'):
        validate_terminal_custody(cfg=cfg,candidate={'run_id':source_plan.run_id,'attempt_id':source_plan.attempt_id},plan=bad_plan,data=data)
    put(plan_path,plan_doc); ancestor['plan'] = pin(plan_path); cfg['terminal']['evidence'][1] = pin(plan_path); reseal()
    custody_path = next(p for p in journal.root.glob('*.fit.json') if json.loads(p.read_bytes())['payload']['request_id'] == 'request')
    custody_blob = custody_path.read_bytes(); custody_path.unlink(); reseal()
    with pytest.raises(ValueError, match='matching successful FIT'):
        audit()
    custody_path.write_bytes(custody_blob); reseal()
    assert audit()['generation_calls'] == 2
    checkpoint_blob = checkpoint.read_bytes()
    checkpoint.unlink()
    bad_writer = RowCheckpointWriter(checkpoint.parent, source_plan.run_id, source_plan.attempt_id, identity=header)
    for p in (payload, payload2):
        rid = p['request_id']
        trace = [{'channel':'MODEL_REQUEST','record_id':rid+'/request',
                  'payload':{'payload':p,'payload_sha256':digest(canonical(p))}},
                 {'channel':'MODEL_RESPONSE','record_id':rid+'/response',
                  'payload':{**response_trace,'text':'tampered' if rid == 'request' else response_trace['text'],
                             'response_text_sha256':digest(b'tampered') if rid == 'request' else response_trace['response_text_sha256']}}]
        bad_writer.append_row(sample_id=p['sample_id'], complete=True, output_rows=[{'sample_id':p['sample_id'],
                              'request_id':rid,'request_sha256':digest(canonical(p)),'response_sha256':digest(response_bytes)}],
                              trace_lines=[json.dumps(x) for x in trace])
    bad_writer.close()
    ancestor['checkpoint'] = pin(checkpoint); cfg['terminal']['evidence'][2] = pin(checkpoint)
    cfg['terminal']['resume_from']['checkpoint_sha256'] = digest(checkpoint.read_bytes()); reseal()
    with pytest.raises(ValueError, match='response text or usage differs'):
        audit()
    checkpoint.write_bytes(checkpoint_blob)
    ancestor['checkpoint'] = pin(checkpoint); cfg['terminal']['evidence'][2] = pin(checkpoint)
    cfg['terminal']['resume_from']['checkpoint_sha256'] = digest(checkpoint_blob); reseal()
    from speech_aware_evidence_acquisition.core.delivery import (native_abstention_records, native_context_abstention_records,
        native_seal_refusal_abstention_records)
    contact = {'agent_contact_id':contact1,'request_id':'request','request_sha256':digest(canonical(payload)),
               'response_sha256':digest(response_bytes),'response_text_sha256':digest(b'fixture response')}
    finite_receipt = {'role_visits':[],'planner_routes':[],
        'interrupted_role':{'role_id':'perception','stage':'perception','execution_receipt':{'contacts':[contact],'final_contact':contact}},
        'terminal_disposition':{'reason':'finite_completion','finish_reason':'length','role_id':'perception',
            'stage':'perception','revision_count':0,'planner_accepted':False,'rejected_contact':contact},
        'abstention_receipt':{'source_contact':contact}}
    seal_receipt = copy.deepcopy(finite_receipt)
    seal_receipt['terminal_disposition'].pop('finish_reason')
    seal_receipt['terminal_disposition']['reason'] = 'seal_refusal'
    seal_receipt['terminal_disposition']['seal_refusal'] = {'error_class':'RailsRunnerError',
        'message':'cited-answer contract refused: intent citations must be unique non-blank ids'}
    context_disposition = {'reason':'context_compaction','role_id':'perception','stage':'perception','revision_count':0,
        'planner_accepted':False,'compaction_pass':1,'dropped_identifiers':[],'tokens_before':120,
        'tokens_after':110,'limit_tokens':100,'source':'dispatch-budget','server_message':None}
    context_receipt = {'role_visits':[],'planner_routes':[],
        'interrupted_role':{'role_id':'perception','stage':'perception'},'terminal_disposition':context_disposition,
        'abstention_receipt':{'source_contact':None,'reason':'context_compaction','source':'dispatch-budget'}}
    for kind, forged in (('finite', False), ('finite', True), ('seal', False), ('seal', True), ('context', False)):
        checkpoint.unlink()
        fallback = RowCheckpointWriter(checkpoint.parent,source_plan.run_id,source_plan.attempt_id,identity=header)
        if kind in ('finite', 'seal'):
            receipt = copy.deepcopy(finite_receipt if kind == 'finite' else seal_receipt)
            if forged:
                for copy_contact in (receipt['interrupted_role']['execution_receipt']['contacts'][0],
                                     receipt['interrupted_role']['execution_receipt']['final_contact'],
                                     receipt['terminal_disposition']['rejected_contact'],
                                     receipt['abstention_receipt']['source_contact']):
                    copy_contact['agent_contact_id'] = 'forged-contact'
            source_contact = receipt['abstention_receipt']['source_contact']
            if kind == 'finite':
                derived = native_abstention_records(sample_id=sample,source_contact=source_contact,finish_reason='length')
            else:
                derived = native_seal_refusal_abstention_records(sample_id=sample,source_contact=source_contact,
                    seal_refusal=receipt['terminal_disposition']['seal_refusal'])
            trace = [{'channel':'MODEL_REQUEST','record_id':'request/request',
                      'payload':{'payload':payload,'payload_sha256':digest(canonical(payload))}},
                     {'channel':'MODEL_RESPONSE','record_id':'request/response','payload':dict(response_trace)}]
        else:
            derived = native_context_abstention_records(sample_id=sample,disposition=context_disposition)
            receipt = context_receipt; trace = []
        derived_id, request_doc, response_doc = derived
        trace.extend([{'channel':'TOOL','record_id':derived_id+'/request','payload':request_doc},
                      {'channel':'TOOL','record_id':derived_id+'/response','payload':response_doc}])
        fallback.append_row(sample_id=sample,complete=True,output_rows=[{'sample_id':sample,'request_id':derived_id,
            'request_sha256':request_doc['payload_sha256'],'response_sha256':response_doc['response_sha256'],
            'native_episode_receipt':receipt}],trace_lines=[json.dumps(x) for x in trace])
        fallback.close()
        ancestor['checkpoint'] = pin(checkpoint); cfg['terminal']['evidence'][2] = pin(checkpoint)
        cfg['terminal']['resume_from']['checkpoint_sha256'] = digest(checkpoint.read_bytes()); reseal()
        if forged:
            with pytest.raises(ValueError, match='checkpoint derived output identity differs') as exc:
                audit()
            assert str(exc.value.__cause__) == 'abstention source differs from observed journal'
        else:
            classified = audit()['ancestors'][0]
            assert classified['checkpoint_complete_rows'] == 1 and classified['checkpoint_native_keepable_rows'] == 1
            assert classified['checkpoint_incomplete_rows'] == {}
    checkpoint.write_bytes(checkpoint_blob)
    ancestor['checkpoint'] = pin(checkpoint); cfg['terminal']['evidence'][2] = pin(checkpoint)
    cfg['terminal']['resume_from']['checkpoint_sha256'] = digest(checkpoint_blob); reseal()
    for fault in ('sample', 'request', 'response_hash'):
        checkpoint.unlink()
        altered = RowCheckpointWriter(checkpoint.parent, source_plan.run_id, source_plan.attempt_id, identity=header)
        for p in (payload, payload2):
            rid = p['request_id']
            out = {'sample_id':p['sample_id'],'request_id':rid,'request_sha256':digest(canonical(p)),
                   'response_sha256':digest(response_bytes)}
            if rid == 'request':
                if fault == 'sample': out['sample_id'] = sample2
                if fault == 'request': out['request_id'] = 'request2'
                if fault == 'response_hash': out['response_sha256'] = '0'*64
            trace = [{'channel':'MODEL_REQUEST','record_id':rid+'/request',
                      'payload':{'payload':p,'payload_sha256':digest(canonical(p))}},
                     {'channel':'MODEL_RESPONSE','record_id':rid+'/response','payload':dict(response_trace)}]
            altered.append_row(sample_id=p['sample_id'], complete=True, output_rows=[out],
                               trace_lines=[json.dumps(x) for x in trace])
        altered.close()
        ancestor['checkpoint'] = pin(checkpoint); cfg['terminal']['evidence'][2] = pin(checkpoint)
        cfg['terminal']['resume_from']['checkpoint_sha256'] = digest(checkpoint.read_bytes()); reseal()
        if fault == 'response_hash':
            with pytest.raises(ValueError, match='canonical output hashes'):
                audit()
        else:
            classified = audit()['ancestors'][0]
            assert classified['checkpoint_complete_rows'] == 2
            assert classified['checkpoint_native_keepable_rows'] == 1
            assert sample in classified['checkpoint_incomplete_rows']
            from speech_aware_evidence_acquisition.core.flight.resume import read_checkpoint_predecessor
            native = read_checkpoint_predecessor(run_id=source_plan.run_id,attempt_id=source_plan.attempt_id,
                attempt_dir=checkpoint.parent,expected_checkpoint_sha256=digest(checkpoint.read_bytes()),
                materialize_dir=tmp_path/('native-'+fault),protocol_hash=source_plan.protocol_hash,
                split_identity_hash=split,implementation=header['implementation'])
            assert native.complete_sample_ids == (sample2,)
            assert native.incomplete[sample] == classified['checkpoint_incomplete_rows'][sample]
    checkpoint.write_bytes(checkpoint_blob)
    ancestor['checkpoint'] = pin(checkpoint); cfg['terminal']['evidence'][2] = pin(checkpoint)
    cfg['terminal']['resume_from']['checkpoint_sha256'] = digest(checkpoint_blob); reseal()

    # The same durable fixture also exercises a normal GenerationLease context
    # exit: no preserved lease is accepted unless the observation was captured
    # under the stable guard and binds the exited producer and receipt.
    natural_path = tmp_path/'natural-observation.json'
    session_path = tmp_path/'session.json'
    put(session_path, {'session_id':'session','identity':{'runtime_receipt_sha256':digest(receipt_path.read_bytes())},
                       'deployment_instance_ids':[i['instance_id'] for i in world.receipt()['deployment']['instances']]})
    producer_path = tmp_path/'producer.json'
    live_launch = tmp_path/'live-launch.json'
    live_cmd = ['/python','/run','--data','/data']
    put(live_launch, {'argv':live_cmd,'generation_lease':{'lock_directory':'/locks'},
        'mounts':[{'source':'/locks','target':'/data/locks','readonly':False}],
        'environment':{'SAEA_OBSERVABILITY_RUN_ID':source_plan.run_id,'SAEA_OBSERVABILITY_ATTEMPT_ID':source_plan.attempt_id}})
    live_intent = tmp_path/'live-intent.json'; put(live_intent, {'config_sha256':digest(live_launch.read_bytes())})
    live_result = tmp_path/'live-result.json'; put(live_result, {'returncode':0,'stdout':'id\n'})
    live_config = tmp_path/'live-config.json'
    put(live_config, {'schema':'saea-producer-observation-config-v1','plan':pin(plan_path),'runtime_receipt':pin(receipt_path),
        'session':pin(session_path),'session_id':'session','launch_config':pin(live_launch),
        'launch_intent':pin(live_intent),'launch_result':pin(live_result)})
    producer = {'schema':'saea-producer-process-observation-v1','plan':pin(plan_path),'runtime_receipt':pin(receipt_path),
                'container_id':'id','producer_pid':10,'producer_start_ticks':1,'boot_id':'boot-fixture',
                'guard':{'device':1,'inode':2},
                'run_id':source_plan.run_id,'attempt_id':source_plan.attempt_id,'observed_before_exit':0.5,
                'config':pin(live_config),'launch_config':pin(live_launch),'launch_intent':pin(live_intent),
                'launch_result':pin(live_result),'process_cmdline':live_cmd,'lock_directory':'/locks',
                'lease_sha256':'a'*64,'lease_identity':{'run_id':source_plan.run_id,'attempt_id':source_plan.attempt_id,'pid':10,'session_id':'session'},
                'session_id':'session','session':pin(session_path)}
    put(producer_path, producer)
    natural = {'schema':'saea-natural-terminal-observation-v1','mode':'natural-drain',
        'container':{'id':'id','state':{'Running':False,'Status':'exited','Pid':0}},
        'boot_id':'boot-fixture','producer_pid':10,'session_id':'session',
        'guard':{'device':1,'inode':2},'lease_absent_under_guard':True,
            'producer_start_ticks':1,'producer_identity_sha256':digest(producer_path.read_bytes()),
        'run_id':source_plan.run_id,'attempt_id':source_plan.attempt_id,
        'plan_sha256':digest(plan_path.read_bytes()),'attempt_sha256':digest(attempt_path.read_bytes()),
        'runtime_receipt_sha256':digest(receipt_path.read_bytes()),'session_sha256':digest(session_path.read_bytes()),
        'observed_requests':['request','request2'],'quiescence':{'queues':{str(port):{'vllm:num_requests_running':0,'vllm:num_requests_waiting':0} for port in (8200,8201)},
            'established_connections':[],'observed_at':1.0},'captured_at':1.0}
    put(natural_path,natural)
    natural_index = {'schema':'saea-checkpoint-terminal-custody-v1','ancestors':[{
        **ancestor,'terminal_mode':'natural-drain','reconciliation':pin(natural_path),'lease':{'absent':True},
        'producer_identity':pin(producer_path)}],
        'inventory':inventory}
    put(index,natural_index)
    natural_cfg = copy.deepcopy(cfg)
    natural_cfg['terminal']['evidence'] = [pin(index),pin(plan_path),pin(checkpoint),pin(roster_path),pin(natural_path),pin(receipt_path),pin(producer_path),pin(session_path),pin(live_config),pin(live_launch),pin(live_intent),pin(live_result)]
    natural_result = validate_terminal_custody(cfg=natural_cfg,
        candidate={'run_id':source_plan.run_id,'attempt_id':source_plan.attempt_id},plan=plan_doc,data=data)
    assert natural_result['ancestors'][0]['lease_session_id'] == 'session'
    import configured_terminal_custody_package as package
    def package_config(tag, cfg0, ancestors0, candidate0, plan0):
        candidate0 = {**candidate0, 'experiment':names['experiment'], 'baseline':names['baseline'],
            'model_fragment':names['model'], 'dataset':{'relpath':'configs/dataset/d.json'},
            'fit_run_id': ('SAEA-FIT-UNIT-TEST-SECOND' if tag == 'multi' else 'SAEA-FIT-UNIT-TEST'),
            'fit_attempt_id': ('fit-v2' if tag == 'multi' else 'fit-v1'),
            'protocol_hash':plan0['execution_plan']['protocol_hash'], 'composed_config_hash':plan0['composed_config_hash'],
            'split_identity_hash':split, 'cell':names['experiment'], 'decision_relpath':'coordination/old.json',
            'scientific_baseline_sha256':digest(canonical(base)), 'service_session':'session',
            'execution':{'admitted_level':1}, 'job_config':{'execution':{'admitted_level':1}}}
        candidate_path = tmp_path/(tag+'-candidate.json'); put(candidate_path, candidate0)
        plan_input = tmp_path/(tag+'-plan-input.json'); put(plan_input, plan0)
        bridge_path = tmp_path/(tag+'-bridge.json'); put(bridge_path, cfg0['terminal']['bridge'])
        candidate_rel = 'docs/readiness/'+names['experiment']+'/candidate.json'
        plan_rel = 'docs/readiness/'+names['experiment']+'/generation-plan-registered-v1.json'
        assets = {candidate_rel:pin(candidate_path), plan_rel:pin(plan_input)}
        active_docs = docs2 if tag == 'multi' else docs
        for kind, name in names.items():
            path = tmp_path/tag/'configs'/kind/(name+'.json'); put(path, active_docs[kind])
            assets['configs/'+kind+'/'+name+'.json'] = pin(path)
        manifest_path = tmp_path/(tag+'-manifest.json'); put(manifest_path, {'schema':'saea-experiment-assets-v1','files':assets})
        template_path = tmp_path/(tag+'-template.json')
        put(template_path, {'schema':'saea-checkpoint-successor-compiler-v1','mode':'finalize',
            'repo':str(source),'data':str(data), 'output':str(tmp_path/(tag+'-compiled')),
            'assets_manifest':pin(manifest_path), 'target':cfg0['target'],
            'candidate':{'relpath':candidate_rel,'sha256':pin(candidate_path)['sha256']},
            'plan':{'relpath':plan_rel,'sha256':pin(plan_input)['sha256']}, 'fragments':names,
            'dispatch':None, 'width':32, 'supply':{'old':supply_pin,'new':supply_pin}, 'terminal':None,
            'admission':{'workspace':str(tmp_path/(tag+'-workspace')),'decision_relpath':'coordination/new.json',
                         'predecessor_decision':{'relpath':'coordination/old.json','sha256':digest(b'fixture decision')}}})
        return {'schema':'saea-terminal-custody-package-config-v1','repo':str(source),'data':str(data),
            'candidate':pin(candidate_path),'plan':pin(plan_input),'target':cfg0['target'],
            'resume_from':cfg0['terminal']['resume_from'],'bridge':pin(bridge_path),'ancestors':copy.deepcopy(ancestors0),
            'evidence':copy.deepcopy(cfg0['terminal']['evidence'][1:]),'output':str(tmp_path/(tag+'-package')),
            'compiler':{'template':pin(template_path),'registration_relpath':'docs/registration.json','lock':pin(lock_path)}}
    natural_package = package_config('natural', natural_cfg, natural_index['ancestors'],
        {'run_id':source_plan.run_id,'attempt_id':source_plan.attempt_id}, plan_doc)
    package.build(natural_package)
    assert json.loads((Path(natural_package['output'])/'validation.json').read_bytes())['ancestors'][0]['lease_session_id'] == 'session'
    if __import__('os').environ.get('SAEA_TEST_NATIVE_COMPILER') == '1':
        import configured_checkpoint_successor as compiler
        compiled = compiler.execute(Path(natural_package['output'])/'compiler-finalize.json')
        assert compiled['status'] == 'HELD' and compiled['activation_authorized'] is False
        assert compiled['new_model_calls'] == compiled['new_fit_calls'] == 0
    bad_template = tmp_path/'incomplete-template.json'
    put(bad_template, {'mode':'finalize','repo':str(source),'data':str(data), 'target':natural_cfg['target'],
                       'candidate':{'sha256':natural_package['candidate']['sha256']},
                       'plan':{'sha256':natural_package['plan']['sha256']}})
    incomplete = copy.deepcopy(natural_package); incomplete['compiler']['template'] = pin(bad_template)
    incomplete['output'] = str(tmp_path/'incomplete-package')
    with pytest.raises(ValueError, match='exact compiler schema'):
        package.build(incomplete)
    assert not Path(incomplete['output']).exists()
    template = json.loads(Path(natural_package['compiler']['template']['path']).read_bytes())
    for field in ('assets_manifest', 'fragments', 'dispatch', 'width', 'supply', 'admission'):
        invalid = copy.deepcopy(template); del invalid[field]
        put(bad_template, invalid); incomplete['compiler']['template'] = pin(bad_template)
        with pytest.raises(ValueError, match='exact compiler schema'):
            package.build(incomplete)
        assert not Path(incomplete['output']).exists()
    invalid = copy.deepcopy(template); invalid['candidate']['relpath'] = 'docs/missing.json'
    put(bad_template, invalid); incomplete['compiler']['template'] = pin(bad_template)
    with pytest.raises(ValueError, match='original source asset differs'):
        package.build(incomplete)
    assert not Path(incomplete['output']).exists()
    package.build(natural_package)
    original_fsync = package.fsync_dir
    synced = []
    def check_parent(path):
        synced.append(path)
        if path == Path(natural_package['output']).parent:
            raise OSError('parent fsync failure')
        return original_fsync(path)
    try:
        package.fsync_dir = check_parent
        with pytest.raises(OSError, match='parent fsync failure'):
            package.build(natural_package)
        assert Path(natural_package['output']).parent in synced
    finally:
        package.fsync_dir = original_fsync


    put(index,{'schema':'saea-checkpoint-terminal-custody-v1','ancestors':[ancestor],'inventory':inventory})
    cfg['terminal']['evidence'][0] = pin(index)
    import copy
    wrong = copy.deepcopy(cfg)
    wrong['terminal']['resume_from']['attempt_id'] = 'other'
    with pytest.raises(ValueError):
        validate_terminal_custody(cfg=wrong,candidate={'run_id':source_plan.run_id,'attempt_id':source_plan.attempt_id},plan=plan_doc,data=data)
    bad_candidate = {'run_id':source_plan.run_id,'attempt_id':source_plan.attempt_id,'protocol_hash':'0'*64}
    with pytest.raises(ValueError, match='candidate'):
        validate_terminal_custody(cfg=cfg,candidate=bad_candidate,plan=plan_doc,data=data)
    extra = fit_root/'orphan.request-fit.json.contacts.jsonl'; extra.write_bytes(b'{}\n')
    with pytest.raises(ValueError, match='inventory'):
        validate_terminal_custody(cfg=cfg,candidate={'run_id':source_plan.run_id,'attempt_id':source_plan.attempt_id},plan=plan_doc,data=data)
    extra.unlink()
    journal._path('request','response').unlink()
    with pytest.raises(ValueError, match='inventory'):
        validate_terminal_custody(cfg=cfg,candidate={'run_id':source_plan.run_id,'attempt_id':source_plan.attempt_id},plan=plan_doc,data=data)

    # A genuine two-hop recovery: older row retained, newer attempt flies only
    # the other sample. Each durable AttemptStore contains only its own spend.
    journal._path('request','response').write_bytes(response_blob)
    from speech_aware_evidence_acquisition.core.attempts import AttemptStore
    from speech_aware_evidence_acquisition.core.flight.resume import read_checkpoint_predecessor, MergedPredecessors, ResumeRecoveryError
    store = AttemptStore(data/'attempts')
    for path in (attempt_path,): path.unlink()
    old_handle = store.open_attempt(source_plan.run_id, source_plan.attempt_id,
        carrier_lock_key=source_plan.carrier_lock_key, reserved_calls=2, reserved_audio_seconds=250)
    old_handle.record_usage(calls=1, audio_seconds=100); old_handle.mark_aborted('fixture interruption')
    old_attempt = json.loads(attempt_path.read_bytes())
    # Keep first contact in the old attempt, and retain a successful FIT-only
    # second contact: its paid tokenizer endpoint was reached before generation.
    journal._path('request2','intent').unlink()
    journal._path('request2','response').unlink(missing_ok=True)
    checkpoint.unlink()
    old_writer = RowCheckpointWriter(checkpoint.parent,source_plan.run_id,source_plan.attempt_id,identity=header)
    def native_trace(p):
        rid = p['request_id']
        return [json.dumps({'channel':'MODEL_REQUEST','record_id':rid+'/request',
                           'payload':{'payload':p,'payload_sha256':digest(canonical(p))}}),
                json.dumps({'channel':'MODEL_RESPONSE','record_id':rid+'/response','payload':dict(response_trace)})]
    old_writer.append_row(sample_id=sample,complete=True,
        output_rows=[{'sample_id':sample,'request_id':'request','request_sha256':digest(canonical(payload)),
                      'response_sha256':digest(response_bytes),'text':'fixture response'}],trace_lines=native_trace(payload))
    old_writer.close()
    first_decl = {'attempt_id':source_plan.attempt_id,'checkpoint_sha256':digest(checkpoint.read_bytes())}
    ancestor['checkpoint'] = pin(checkpoint)
    put(reconciliation, {'container':'id','state':{'Running':False,'Status':'exited','Pid':0},
        'attempt':old_attempt,'observed_requests':['request'],
        'quiescence':{'queues':{str(port):{'vllm:num_requests_running':0,'vllm:num_requests_waiting':0} for port in (8200,8201)},
                      'established_connections':[],'observed_at':1.0},
        'lease_sha256':digest(lock_path.read_bytes()),'no_model_calls':True})
    ancestor['reconciliation'] = pin(reconciliation)

    middle_id = 'attempt-0002'
    exp2 = copy.deepcopy(exp)
    exp2['cells'][0]['attempt_id'] = middle_id
    exp2['cells'][0]['resume_from'] = first_decl
    base2 = copy.deepcopy(base)
    base2['native_contact_fit'].update(run_id='SAEA-FIT-UNIT-TEST-SECOND',attempt_id='fit-v2')
    put(study/'configs/experiment/e.json',exp2); put(study/'configs/baseline/b.json',base2)
    for args in (['add','.'], ['-c','user.name=Test','-c','user.email=test@localhost','commit','-qm','middle']):
        subprocess.run(['git',*args],cwd=repo,check=True,capture_output=True)
    commit2 = git('rev-parse','HEAD').decode().strip()
    obj2=tmp_path/'commit2'; obj2.write_bytes(git('cat-file','commit',commit2))
    listing2=tmp_path/'listing2'; listing2.write_bytes(git('ls-tree','-r','--full-tree',commit2))
    archive2=tmp_path/'archive2.tar'; archive2.write_bytes(git('archive',commit2))
    docs2 = {**docs,'experiment':exp2,'baseline':base2}
    normalized2 = {k:collapse_data_root(v,str(world.materialization_root)) for k,v in docs2.items()}
    composed2 = _compose_fragments(names,normalized2,str(world.materialization_root))
    middle_plan = replace(source_plan,attempt_id=middle_id,protocol_hash=protocol_hash(composed2.config_hash,arm,split))
    middle_header = {'config_hash':composed2.config_hash,'protocol_hash':middle_plan.protocol_hash,
        'split_identity_hash':split,'implementation':header['implementation']}
    middle_payload = copy.deepcopy(payload2); middle_payload['request_id']='middle-request'
    middle_journal = ContactJournal(data,middle_plan,arm)
    middle_journal.intent(middle_payload)
    middle_journal.response('middle-request',digest(canonical(middle_payload)),response_bytes,0.1)
    middle_handle = store.open_attempt(middle_plan.run_id,middle_id,carrier_lock_key=middle_plan.carrier_lock_key,
                                       reserved_calls=2,reserved_audio_seconds=250)
    middle_handle.record_usage(calls=1,audio_seconds=100); middle_handle.mark_aborted('fixture interruption')
    middle_attempt_path = data/'attempts'/middle_plan.run_id/(middle_id+'.json')
    middle_attempt = json.loads(middle_attempt_path.read_bytes())
    middle_writer = RowCheckpointWriter(data/'runs'/middle_plan.run_id/middle_id,middle_plan.run_id,middle_id,identity=middle_header)
    middle_writer.append_row(sample_id=sample2,complete=True,
        output_rows=[{'sample_id':sample2,'request_id':'middle-request','request_sha256':digest(canonical(middle_payload)),
                      'response_sha256':digest(response_bytes),'text':'fixture response'}],trace_lines=native_trace(middle_payload))
    middle_writer.close(); middle_cp = middle_writer.path
    middle_decl = {'attempt_id':middle_id,'checkpoint_sha256':digest(middle_cp.read_bytes())}
    middle_plan_doc = {'execution_plan':asdict(middle_plan),'composed_config_hash':composed2.config_hash}
    middle_plan_path = tmp_path/'middle-plan.json'; put(middle_plan_path,middle_plan_doc)
    middle_fit_plan = replace(middle_plan,run_id=base2['native_contact_fit']['run_id'],attempt_id='fit-v2',planned_model_calls=6)
    middle_fit_root = data/'prepared'/middle_fit_plan.run_id/middle_fit_plan.attempt_id; middle_fit_root.mkdir(parents=True)
    middle_boundary = boundary_identity((middle_payload,))
    middle_inputs = artifact_inputs(config=composed2.values,split_identity_hash=split,sources={'boundary_sha256':middle_boundary,
        'required_identity':session,'runtime_receipt':world.receipt(),'implementation':impl})
    middle_native = {**native2,'raw_speech_query':middle_payload['raw_speech_query']}
    middle_artifact = middle_fit_root/(key2+'.request-fit.json')
    middle_output = {**output2,'boundary_sha256':middle_boundary,'cumulative_contacts':1}
    middle_artifact.write_bytes(_artifact_bytes(kind='request-fit',inputs={**middle_inputs,'native_contact':middle_native,
        'fit_plan':asdict(middle_fit_plan)},output=middle_output,contact_class='tokenizer'))
    middle_endpoint = Path(str(middle_artifact)+'.contacts.jsonl')
    put(middle_endpoint,canonical({'sample_id':sample2,'agent_contact_id':contact2,'contact_identity_sha256':key2,'contact':1,
        'method':'POST','route':'/tokenize','run_id':middle_fit_plan.run_id,'attempt_id':middle_fit_plan.attempt_id,
        'generation_run_id':middle_plan.run_id,'session_id':'session'})+b'\n')
    NativeFitJournal(data,middle_plan,arm).record(middle_payload,middle_inputs,{'path':str(middle_artifact),
        'sha256':digest(middle_artifact.read_bytes()),'sample_id':sample2,'agent_contact_id':contact2,
        'contact_identity_sha256':key2,'contact_class':'tokenizer'},42,session)
    middle_custody = next(middle_journal.root.glob('*.fit.json'))
    middle_custody_doc = json.loads(middle_custody.read_bytes())
    middle_custody_doc['receipt']['path'] = str(world.materialization_root/'prepared'/middle_fit_plan.run_id/middle_fit_plan.attempt_id/middle_artifact.name)
    put(middle_custody,middle_custody_doc)
    middle_lock = tmp_path/'middle-lock.json'; put(middle_lock,{'run_id':middle_plan.run_id,'attempt_id':middle_id,'pid':11,'session_id':'session'})
    middle_rec = tmp_path/'middle-rec.json'; put(middle_rec,{'container':'id2','state':{'Running':False,'Status':'exited','Pid':0},
        'attempt':middle_attempt,'observed_requests':['middle-request'],
        'quiescence':{'queues':{str(port):{'vllm:num_requests_running':0,'vllm:num_requests_waiting':0} for port in (8200,8201)},
                      'established_connections':[],'observed_at':1.0},
        'lease_sha256':digest(middle_lock.read_bytes()),'no_model_calls':True})
    bridge2 = copy.deepcopy(bridge); bridge2['predecessor_commit']=commit2
    bridge2['evidence'].update(commit_object=pin(obj2),tree_listing=pin(listing2),source_archive=pin(archive2))
    middle_ancestor = {'attempt_id':middle_id,'plan':pin(middle_plan_path),'checkpoint':pin(middle_cp),'generation_root':str(data),
        'prepared_root':str(middle_fit_root),'logical_prepared_root':str(world.materialization_root/'prepared'/middle_fit_plan.run_id/middle_fit_plan.attempt_id),
        'roster':pin(roster_path),'bridge_key':middle_id,'reconciliation':pin(middle_rec),'lease':pin(middle_lock)}
    all_roots = [journal.root,attempt_path.parent,checkpoint.parent,fit_root,middle_journal.root,middle_cp.parent,middle_fit_root]
    multi_inventory = [{'root':str(root),'files':{p.relative_to(root).as_posix():digest(p.read_bytes())
                        for p in root.rglob('*') if p.is_file()}} for root in all_roots]
    multi_index = tmp_path/'multi-index.json'
    put(multi_index,{'schema':'saea-checkpoint-terminal-custody-v1','ancestors':[ancestor,middle_ancestor],'inventory':multi_inventory})
    multi_cfg = {'repo':str(source),'target':{'attempt_id':'next','fit_run_id':'SAEA-FIT-NEXT','fit_attempt_id':'fit-next'},'terminal':{'resume_from':[first_decl,middle_decl],
        'bridge':{'schema':'saea-checkpoint-compatibility-v2','successor_attempt_id':'next',
                  'bridges':{source_plan.attempt_id:bridge,middle_id:bridge2}},
        'evidence':[pin(multi_index),pin(plan_path),pin(checkpoint),pin(roster_path),pin(reconciliation),pin(lock_path),
                    pin(receipt_path),pin(middle_plan_path),pin(middle_cp),pin(middle_rec),pin(middle_lock)]}}
    multi_candidate = {'run_id':middle_plan.run_id,'attempt_id':middle_id}
    def multi_audit():
        return validate_terminal_custody(cfg=multi_cfg,candidate=multi_candidate,plan=middle_plan_doc,data=data)
    multi_package = package_config('multi', multi_cfg, [ancestor,middle_ancestor], multi_candidate, middle_plan_doc)
    package.build(multi_package)
    assert json.loads((Path(multi_package['output'])/'validation.json').read_bytes())['generation_calls'] == 2
    package.build(multi_package)
    conflict = copy.deepcopy(multi_package); conflict['compiler']['registration_relpath'] = 'docs/other-registration.json'
    with pytest.raises(ValueError, match='conflicting'): package.build(conflict)
    bad = copy.deepcopy(multi_package); bad['evidence'].append(bad['evidence'][0])
    with pytest.raises(ValueError, match='duplicate'): package.build(bad)
    bad = copy.deepcopy(multi_package); bad['evidence'][0]['sha256'] = '0'*64
    with pytest.raises(ValueError, match='changed'): package.build(bad)
    bad = copy.deepcopy(multi_package); bad['evidence'][0]['path'] = str(tmp_path/'absent')
    with pytest.raises(ValueError, match='missing'): package.build(bad)
    bad = copy.deepcopy(multi_package); bad['output'] = str(tmp_path/'interrupted')
    real_publish = package.publish
    try:
        package.publish = lambda *args: (_ for _ in ()).throw(OSError('interrupt'))
        with pytest.raises(OSError, match='interrupt'): package.build(bad)
        assert not Path(bad['output']).exists()
    finally: package.publish = real_publish
    real_scan = package.scan
    calls = []
    def changing(roots):
        result = real_scan(roots); calls.append(1)
        if len(calls) == 2: result[0]['files']['changed'] = '0'*64
        return result
    try:
        package.scan = changing
        with pytest.raises(ValueError, match='changed'): package.build(bad)
        assert not Path(bad['output']).exists()
    finally: package.scan = real_scan
    link = middle_fit_root/'bad-link'; link.symlink_to(middle_artifact)
    with pytest.raises(ValueError, match='non-regular'): package.build(bad)
    link.unlink()
    bad = copy.deepcopy(multi_package); bad['resume_from'] = []
    with pytest.raises(ResumeRecoveryError): package.build(bad)
    multi = multi_audit()
    assert [x['generation']['actual']['calls'] for x in multi['ancestors']] == [1,1]
    assert multi['generation_calls'] == 2 and multi['generation_audio_seconds'] == 200
    assert multi['same_run_accumulated']['calls'] == 2 and multi['same_run_accumulated']['audio_seconds'] == 200
    assert [x['fit']['endpoint_calls'] for x in multi['ancestors']] == [2,1]
    assert multi['fit_endpoint_calls'] == 3
    assert all(x['status'] == 'success_recoverable' for x in multi['ancestors'][0]['fit']['contacts'])
    assert {x['sample_id'] for x in multi['ancestors'][0]['fit']['contacts']} == {sample,sample2}
    assert multi['ancestors'][0]['generation']['observed_requests'] == ['request']
    predecessors = []
    for p, cp in ((source_plan,checkpoint),(middle_plan,middle_cp)):
        predecessors.append(read_checkpoint_predecessor(run_id=p.run_id,attempt_id=p.attempt_id,attempt_dir=cp.parent,
            expected_checkpoint_sha256=digest(cp.read_bytes()),materialize_dir=tmp_path/'materialized'/p.attempt_id,
            protocol_hash=p.protocol_hash,split_identity_hash=split,implementation=header['implementation']))
    merged = MergedPredecessors(tuple(predecessors))
    assert merged.complete_sample_ids == (sample,sample2)
    assert merged.origin(sample).attempt_id == source_plan.attempt_id and merged.origin(sample2).attempt_id == middle_id
    with pytest.raises(ResumeRecoveryError, match='COMPLETE in two predecessor'):
        MergedPredecessors((predecessors[0], predecessors[0]))
    # Re-sealed semantic fault: keep inventory current while introducing an
    # unknown middle response; rejection must come from the journal join.
    middle_journal._path('middle-request','response').unlink()
    bad = copy.deepcopy(multi_package); bad['output'] = str(tmp_path/'unknown')
    with pytest.raises(ValueError, match='unknown'): package.build(bad)
    assert not Path(bad['output']).exists()
    for item in multi_inventory:
        root=Path(item['root']); item['files']={p.relative_to(root).as_posix():digest(p.read_bytes()) for p in root.rglob('*') if p.is_file()}
    put(multi_index,{'schema':'saea-checkpoint-terminal-custody-v1','ancestors':[ancestor,middle_ancestor],'inventory':multi_inventory})
    multi_cfg['terminal']['evidence'][0]=pin(multi_index)
    with pytest.raises(ValueError,match='unknown'):
        multi_audit()


def test_capture_binds_launcher_and_guard(tmp_path, monkeypatch):
    from dataclasses import asdict
    import capture_natural_terminal_custody as capture_module

    def put(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical(value))
        return {'path': str(path), 'sha256': digest(path.read_bytes())}

    plan = _plan()
    data = tmp_path/'data'; data.mkdir()
    locks = tmp_path/'mapped-locks'; locks.mkdir()
    (locks/'generation.guard').write_bytes(b'')
    instance = {'instance_id':'one', 'host':'127.0.0.1', 'port':8200}
    receipt = {'deployment':{'instances':[instance]}}
    plan_pin = put(tmp_path/'plan.json', {'execution_plan':asdict(plan)})
    runtime_pin = put(tmp_path/'runtime.json', receipt)
    session_pin = put(tmp_path/'session.json', {'session_id':'session',
        'identity':{'runtime_receipt_sha256':runtime_pin['sha256']}, 'deployment_instance_ids':['one']})
    container_id = 'a'*64
    command = ['/python','/run.py','--data','/logical']
    launch = {'argv':command, 'environment':{'SAEA_OBSERVABILITY_RUN_ID':plan.run_id,
        'SAEA_OBSERVABILITY_ATTEMPT_ID':plan.attempt_id}, 'generation_lease':{'lock_directory':str(locks)},
        'mounts':[{'source':str(locks),'target':'/logical/locks','readonly':False}], 'image':'fixture-image'}
    launch_pin = put(tmp_path/'launch.json', launch)
    intent_pin = put(tmp_path/'intent.json', {'schema':'saea-configured-job-launch-v1','config_sha256':launch_pin['sha256'],'runner_sha256':'c'*64})
    result_pin = put(tmp_path/'result.json', {'returncode':0,'stdout':container_id+'\n'})
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    proc = tmp_path/'proc'; pid = 2147483647
    (proc/str(pid)).mkdir(parents=True)
    (proc/'sys/kernel/random').mkdir(parents=True)
    (proc/'sys/kernel/random/boot_id').write_text(boot)
    (proc/str(pid)/'stat').write_text(str(pid)+' (python) '+ ' '.join(['S']+['0']*18+['123']))
    (proc/str(pid)/'cmdline').write_bytes(b'\0'.join(x.encode() for x in command)+b'\0')
    lease_path = locks/'flight.lock'
    lease = {'run_id':plan.run_id,'attempt_id':plan.attempt_id,'pid':pid,'session_id':'session'}
    put(lease_path, lease)
    config = {'schema':'saea-producer-observation-config-v1','plan':plan_pin,'runtime_receipt':runtime_pin,
        'session':session_pin,'session_id':'session','launch_config':launch_pin,'launch_intent':intent_pin,'launch_result':result_pin}
    config_path = tmp_path/'config.json'; config_pin = put(config_path, config)
    row = {'Id':container_id, 'State':{'Running':True,'Status':'running','Pid':pid},
           'Path':command[0], 'Args':command[1:], 'Image':'fixture-image',
           'Config':{'Labels':{'saea.launch.config':launch_pin['sha256'],'saea.launch.runner':'c'*64}},
           'Mounts':[{'Source':str(locks),'Destination':'/logical/locks','RW':True}]}
    monkeypatch.setattr(capture_module.subprocess, 'check_output', lambda _: canonical([row]))
    def observe(name):
        return capture_module.observe_running(config_path=str(config_path),config_sha256=config_pin['sha256'],
                                              output=tmp_path/name,repo=tmp_path,proc_root=proc)
    identity = observe('identity.json'); identity_path = tmp_path/'identity.json'
    assert identity['producer_start_ticks'] == 123
    lease['pid'] = 2; put(lease_path, lease)
    with pytest.raises(ValueError, match='lease identity'): observe('bad-lease.json')
    lease['pid'] = pid; put(lease_path, lease)
    row['Id'] = 'b'*64
    with pytest.raises(ValueError, match='container identity'): observe('bad-live-container.json')
    row['Id'] = container_id
    row['State']['Pid'] = pid - 1
    with pytest.raises(ValueError): observe('bad-live-pid.json')
    row['State']['Pid'] = pid
    (proc/str(pid)/'stat').write_text(str(pid)+' (python) '+ ' '.join(['S']+['0']*18+['0']))
    with pytest.raises(ValueError, match='start'): observe('bad-start.json')
    (proc/str(pid)/'stat').write_text(str(pid)+' (python) '+ ' '.join(['S']+['0']*18+['123']))
    row['Mounts'][0]['Source'] = '/wrong'
    with pytest.raises(ValueError, match='container identity|guard mount'): observe('bad-live-guard.json')
    row['Mounts'][0]['Source'] = str(locks)
    calls = []
    def changed(_):
        calls.append(1)
        if len(calls) == 2:
            (proc/str(pid)/'stat').write_text(str(pid)+' (python) '+ ' '.join(['S']+['0']*18+['124']))
        return canonical([row])
    monkeypatch.setattr(capture_module.subprocess, 'check_output', changed)
    with pytest.raises(ValueError, match='changed'): observe('changed.json')
    monkeypatch.setattr(capture_module.subprocess, 'check_output', lambda _: canonical([row]))
    lease_path.unlink()
    row['State'] = {'Running':False,'Status':'exited','Pid':0}
    attempt = data/'attempts'/plan.run_id/(plan.attempt_id+'.json')
    put(attempt, {'state':'ABORTED'})
    monkeypatch.setattr(capture_module.subprocess, 'check_output', lambda _: canonical([row]))
    monkeypatch.setattr(capture_module, 'inspect_generation', lambda **_: {'observed_requests':[]})
    monkeypatch.setattr(capture_module, 'quiescence', lambda _: {'queues':{'8200':{'vllm:num_requests_running':0,
        'vllm:num_requests_waiting':0}},'established_connections':[],'observed_at':1.0})
    def run(name):
        pin = put(identity_path, identity)
        return capture_module.capture(data=data,identity_path=str(identity_path),identity_sha256=pin['sha256'],
                                     arm='raw-speech-agent-native-tools',output=tmp_path/name,repo=tmp_path)
    identity_pin = {'path':str(identity_path),'sha256':digest(identity_path.read_bytes())}
    assert capture_module.capture(data=data, identity_path=str(identity_path), identity_sha256=identity_pin['sha256'],
        arm='raw-speech-agent-native-tools', output=tmp_path/'observation.json', repo=tmp_path)['guard']['inode'] == (locks/'generation.guard').stat().st_ino
    row['Id'] = 'b'*64
    with pytest.raises(ValueError, match='container command or guard'):
        run('bad-container.json')
    row['Id'] = container_id
    identity['producer_pid'] = 0
    with pytest.raises(ValueError, match='process identity'):
        run('bad-pid.json')
    identity['producer_pid'] = 2147483647
    identity['lock_directory'] = str(tmp_path/'wrong-locks')
    with pytest.raises(ValueError, match='guard mapping'):
        run('bad-guard.json')
    assert not (tmp_path/'bad-container.json').exists()
    assert not (tmp_path/'bad-pid.json').exists()
    assert not (tmp_path/'bad-guard.json').exists()



def test_layered_ancestor_reachability():
    from checkpoint_terminal_custody import validate_ancestry
    validate_ancestry({'new': {'middle', 'old'}, 'middle': {'old'}, 'old': set()}, 'new')
    with pytest.raises(ValueError, match='candidate root'):
        validate_ancestry({'old': set()}, 'missing')
    with pytest.raises(ValueError, match='reachable'):
        validate_ancestry({'new': set(), 'old': set()}, 'new')
    with pytest.raises(ValueError, match='cyclic'):
        validate_ancestry({'new': {'old'}, 'old': {'new'}}, 'new')

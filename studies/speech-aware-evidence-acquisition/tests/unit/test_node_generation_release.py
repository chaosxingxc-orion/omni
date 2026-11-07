"""Exercise the release against a generated and finalized node bundle."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_bundle import bundled_repo, _run_completed, _HUMAN_LEDGER_CELLS
from test_configured_job_queue import queue, internal_generation_fixture, m
from speech_aware_evidence_acquisition.core.bundle import verify_bundle, finalize_bundle
from speech_aware_evidence_acquisition.core.tracing import TraceRecord
from speech_aware_evidence_acquisition.e0.artifacts import write_json_artifact, sha256_file
from test_artifact_generation_release import producer
from speech_aware_evidence_acquisition.core.flight.receipts import FlightReceipts
from speech_aware_evidence_acquisition.core.atomic_file import publish_new


def node_fixture(tmp_path, synthetic_world, bundled_repo):
    run, aid = 'SAEA-PRODUCER', 'generation-v1'
    data = synthetic_world.data_root
    rel = f'runs/{run}/{aid}'
    path, _, _, lock = _run_completed(synthetic_world, bundled_repo, data, run_id=run,
                                      attempt_id=aid, run_dir_name=rel)
    manifest = json.loads(path.read_text())
    outputs = Path(manifest['outputs']['path'])
    trace = Path(manifest['raw_trace']['path'])
    trace_manifest_path = Path(manifest['trace_manifest']['path'])
    rows = [json.loads(line) for line in outputs.read_text().splitlines()]
    traces = [json.loads(line) for line in trace.read_text().splitlines()]
    for row in rows:
        contact = {k:row[k] for k in ('request_id', 'request_sha256', 'response_sha256', 'response_text_sha256')}
        node = {'result_type':'node_outcome', 'stage':'perception', 'sample_id':row['sample_id'],
                'response':row['text'], 'status':'ok', 'node_receipt':{'contacts':[contact], 'final_contact':contact}}
        row.update(result_kind='node', node_outcome=node)
        traces.append({'channel':'USE', 'record_id':row['sample_id']+'/perception-node', 'payload':{'use_node_outcome':node}})
    outputs.write_text(''.join(json.dumps(row, sort_keys=True)+'\n' for row in rows))
    trace.write_text(''.join(json.dumps(row, sort_keys=True)+'\n' for row in traces))
    trace_manifest = json.loads(trace_manifest_path.read_text())
    trace_manifest['records'] = {r['record_id']:TraceRecord(channel=r['channel'], record_id=r['record_id'], payload=r['payload']).content_hash() for r in traces}
    write_json_artifact(trace_manifest_path, trace_manifest)
    manifest.update(result_kind='node', node_result={'schema':'saea-perception-node-v1','stage':'perception'}, scorers=[])
    for key in ('outputs','raw_trace','trace_manifest'):
        manifest[key]['sha256'] = sha256_file(Path(manifest[key]['path']))
    write_json_artifact(path, manifest)
    assert len(verify_bundle(path).outputs_rows) == 3
    result = finalize_bundle(path, lock, data, repo_root=bundled_repo[1], ledger_cells=_HUMAN_LEDGER_CELLS)
    assert result.attempt_state == 'FINALIZED' and result.scores_path is None
    return path, data


def event_for(path, data, expected, original_data):
    manifest = json.loads(path.read_text())
    logical = Path('/logical')
    pins = {}
    for key, file in [('manifest',path), ('outputs',data/Path(manifest['outputs']['path']).relative_to(original_data)),
                      ('attempt',data/Path(manifest['attempt']['state_path']).relative_to(original_data))]:
        pins[key] = {'path':str(logical/file.relative_to(data)), 'sha256':sha256_file(file)}
    # Rewrite only the bundle's path references for a container-style logical root.
    manifest['outputs']['path'] = pins['outputs']['path']
    manifest['attempt']['state_path'] = pins['attempt']['path']
    manifest['study_commit'] = expected['study_commit']
    write_json_artifact(path, manifest)
    pins['manifest']['sha256'] = sha256_file(path)
    seal = {'manifest_path':pins['manifest']['path'], 'manifest_sha256':pins['manifest']['sha256'],
            'outputs_path':pins['outputs']['path'], 'outputs_row_count':expected['rows'],
            'phases_path':None, 'cost':manifest['cost'], 'verified_attempt_state':'FINALIZED',
            'finalize':{'status':'not-applicable','result_kind':'node','attempt_state':'FINALIZED','scores_path':None}}
    return {'schema':'saea-node-generation-released-v1', **expected, 'generation_sealed':True,
            'generation_lease_released':True,'scoring_complete':False,
            'consumer_requires_own_lease_and_readiness':True,'released_at_unix':123.,
            'campaign_seconds_to_release':12.,'artifacts':pins,'seal_receipt':seal}


def test_producer_generated_node_release_dispatches_queue(queue, tmp_path, synthetic_world, bundled_repo, monkeypatch):
    manifest_path, data = node_fixture(tmp_path, synthetic_world, bundled_repo)
    manifest = json.loads(manifest_path.read_text())
    release, _, _ = internal_generation_fixture(queue, tmp_path, learning=False, rows=3)
    path, config, docker = queue
    job = config['jobs'][0]
    expected = job['generation_release']['expected']
    expected['study_commit'] = manifest['study_commit']
    job['generation_release']['logical_data'] = str(data)
    launch_path = Path(job['launch']['path'])
    launch = json.loads(launch_path.read_text())
    launch['expected_commit'] = expected['study_commit']
    def run_command(argv, **kwargs):
        if argv[0] == 'git' and argv[-1] == 'HEAD':
            return SimpleNamespace(returncode=0, stdout=expected['study_commit'], stderr='')
        return docker(argv, **kwargs)
    monkeypatch.setattr(m.launcher, 'run', run_command)
    launch['argv'][launch['argv'].index('/logical')] = str(data)
    launch['environment']['SPEECHRL_DATA_DIR'] = str(data)
    mount = next(x for x in launch['mounts'] if x['target'] == '/logical')
    mount.update(source=str(data), target=str(data))
    launch_path.write_text(json.dumps(launch))
    job['launch']['sha256'] = m.launcher.digest(launch_path)
    docker.containers['producer']['Config']['Labels']['saea.launch.config'] = m.canonical_hash(launch)
    intent_path = Path(launch['receipt_dir'])/'launch-intent.json'
    intent = json.loads(intent_path.read_text())
    intent['config_sha256'] = m.canonical_hash(launch)
    intent_path.write_text(json.dumps(intent))
    release.with_name('started.json').write_text(json.dumps(
        {'study_commit':expected['study_commit'], 'candidate':expected['candidate'], 'rows':3}))
    path.write_text(json.dumps(config))
    seal = {'manifest_path':str(manifest_path), 'manifest_sha256':sha256_file(manifest_path),
            'outputs_path':manifest['outputs']['path'], 'outputs_row_count':3, 'phases_path':None,
            'cost':manifest['cost'], 'verified_attempt_state':'FINALIZED',
            'finalize':{'status':'not-applicable', 'result_kind':'node',
                        'attempt_state':'FINALIZED', 'scores_path':None}}
    receipts = FlightReceipts(expected['run_id'], expected['attempt_id'])
    receipts.record('seal', seal)
    runtime = SimpleNamespace(run_id=expected['run_id'], attempt_id=expected['attempt_id'],
                              receipts=receipts, _data_root=data)
    assert m.execute(path, 'once')['status'] == 'running'
    doc = producer.generation_release_document(runtime, candidate=expected['candidate'],
        spec={**expected, 'extra_candidate_field':'ignored'}, commit=expected['study_commit'], rows=3, elapsed=12.)
    assert set(doc['artifacts']) == {'manifest', 'outputs', 'attempt'}
    assert doc['seal_receipt'] == receipts.require('seal')
    publish_new(release, (json.dumps(doc)+'\n').encode())
    monkeypatch.setattr(m.launcher, 'run', docker)
    result = m.execute(path, 'once')
    assert [j['status'] for j in result['jobs']] == ['scoring', 'skipped', 'running']
    assert docker.launches == ['producer', 'job1'] and docker.containers['producer']['State']['Running']
    assert not docker.waits and not receipts.has('score')


@pytest.mark.parametrize('fault',[None,'phases','bytes','missing','unfinished','run','cost','protocol','config','samples','stage','answer','unknown'])
def test_generated_node_reaches_queue_through_host_mapping(queue, tmp_path, synthetic_world, bundled_repo, fault):
    manifest_path, generated_data = node_fixture(tmp_path, synthetic_world, bundled_repo)
    release, _, data = internal_generation_fixture(queue, tmp_path, learning=False, rows=3)
    path, config, docker = queue
    expected = config['jobs'][0]['generation_release']['expected']
    # Move the verified bundle files under the queue's actual data mount.
    import shutil
    shutil.copytree(generated_data/'runs', data/'runs', dirs_exist_ok=True)
    shutil.copytree(generated_data/'attempts', data/'attempts', dirs_exist_ok=True)
    manifest_path = data/manifest_path.relative_to(generated_data)
    doc = event_for(manifest_path, data, expected, generated_data)
    helper = m.release_helper(config)
    resolver = lambda logical: data/Path(logical).relative_to('/logical')
    assert helper.verify_release_document(doc, data=Path('/logical'), expected=expected, resolve_artifact=resolver) == doc
    if fault == 'phases':
        phase = manifest_path.parent/'SAEA-PRODUCER.phases.json'
        phase.write_text('{}')
        doc['artifacts']['phases'] = {'path':str(Path('/logical')/phase.relative_to(data)), 'sha256':sha256_file(phase)}
        doc['seal_receipt'].update(phases_path=doc['artifacts']['phases']['path'], phases_sha256=sha256_file(phase))
    elif fault == 'bytes': resolver(Path(doc['artifacts']['outputs']['path'])).write_text('changed')
    elif fault == 'missing': doc['artifacts'].pop('attempt')
    elif fault == 'unfinished': doc['seal_receipt']['finalize']['attempt_state'] = 'OPENED'
    elif fault == 'run': doc['run_id'] = 'OTHER'
    elif fault == 'cost': doc['seal_receipt']['cost'] = {}
    elif fault == 'unknown': doc['schema'] = 'unknown'
    elif fault in ('protocol','config','samples','stage','answer'):
        man = json.loads(manifest_path.read_text())
        if fault == 'protocol': man['protocol_hash'] = '0'*64
        if fault == 'config': man['config']['config_hash'] = '0'*64
        if fault == 'samples': man['split']['ids'][0] = 'other'
        if fault == 'stage': man['node_result']['stage'] = 'decision'
        if fault == 'answer':
            output = resolver(Path(doc['artifacts']['outputs']['path']))
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            rows[0]['system_answer'] = 'invented'
            output.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            doc['artifacts']['outputs']['sha256'] = sha256_file(output)
            man['outputs']['sha256'] = sha256_file(output)
        write_json_artifact(manifest_path, man)
        doc['artifacts']['manifest']['sha256'] = sha256_file(manifest_path)
        doc['seal_receipt']['manifest_sha256'] = sha256_file(manifest_path)
    assert m.execute(path,'once')['status'] == 'running'
    release.write_text(json.dumps(doc))
    if fault and fault != 'phases':
        with pytest.raises((ValueError, KeyError)): m.execute(path,'once')
        assert docker.launches == ['producer']
    else:
        result = m.execute(path,'once')
        assert [j['status'] for j in result['jobs']] == ['scoring','skipped','running']
        assert docker.launches == ['producer','job1'] and not docker.waits

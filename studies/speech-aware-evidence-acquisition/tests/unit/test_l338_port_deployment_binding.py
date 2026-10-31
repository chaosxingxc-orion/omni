"""Measured admission rejects mixed deployments and incomplete performance evidence."""
import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'scripts/flights/nx'))
from l338_port_deployment_binding import validate_binding
from speech_aware_evidence_acquisition.core.deployment import load_deployment_profile


@pytest.fixture
def evidence(tmp_path):
    repo, data = tmp_path/'repo', tmp_path/'data'
    profile_name = 'qwen3-omni-thinking-bf16-181-4xtp2-v3'
    profile_path = repo/'configs/deployment'/f'{profile_name}.json'
    profile_path.parent.mkdir(parents=True)
    profile_path.write_bytes((REPO/'configs/deployment'/profile_path.name).read_bytes())
    profile = load_deployment_profile(profile_name, repo, data_root=data)
    def put(root, relpath, value):
        p = root/relpath
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value))
        return {'relpath': relpath, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
    receipt = {'protocol': {'served_model_name': 'model'}, 'deployment': {'instances': [
        {'instance_id': i.name, 'port': i.port, 'device_ids': list(i.cards),
         'max_num_seqs': 32, 'tensor_parallel_size': 2} for i in profile.instances]}}
    receipt_pin = put(repo, 'receipt.json', receipt)
    start_time = '2026-10-30T13:00:00Z'
    identity = {'runtime_receipt_sha256': receipt_pin['sha256'], 'host': '127.0.0.1', 'port': 8200, 'server_argv': ['serve', '--port', '8200']}
    session = {'schema': 'saea-serving-session-v1', 'study': 'speech-aware-evidence-acquisition',
               'session_id': 'coordinator', 'identity': identity,
               'deployment_instance_ids': [i.name for i in profile.instances]}
    binding = {'schema': 'saea-port-deployment-binding-v1', 'profile': profile_name,
               'profile_sha256': profile.sha256, 'admitted_level': 24,
               'runtime_receipt': receipt_pin, 'session_id': 'coordinator',
               'session': put(data, 'session.json', session), 'members': {}}
    observations = []
    for n, i in enumerate(profile.instances):
        member = {'schema': session['schema'], 'study': session['study'], 'session_id': i.name,
                  'identity': {**identity, 'port': i.port, 'server_argv': ['serve', '--port', str(i.port)]}, 'deployment_parent_session_id': 'coordinator',
                  'pid': n+100, 'started_at': start_time}
        binding['members'][i.name] = put(data, i.name+'.json', member)
        observations.append({'Name': '/'+i.name, 'RestartCount': 0,
                             'State': {'Pid': n+100, 'StartedAt': start_time, 'Running': True}})
    binding['recorded_chain'] = put(data, 'chain.json', {'steps': {'deploy-status': {'decision': {
        'status_lines': [i.name+' [started='+start_time+']' for i in profile.instances]}}}})
    common = {'receipt_sha256': receipt_pin['sha256'],
              'profile_sha256': hashlib.sha256(profile_path.read_bytes()).hexdigest(),
              'git_head': 'a'*40, 'driver_sha256': 'b'*64, 'started': '2026-10-30T14:00:00Z'}
    binding['normal_start'] = put(data, 'normal-start.json', {**common,
        'parent_status_sha256': binding['recorded_chain']['sha256']})
    binding['worst_start'] = put(data, 'worst-start.json', {**common,
        'parent_started_sha256': binding['normal_start']['sha256'], 'deployment_observed': observations})
    for kind in ('normal', 'worst'):
        summary = {'concurrency_per_instance': 24, 'requests': 288, 'ok': 288, 'errors': 0,
                   'timeouts': 0, 'aborted': False, 'preemptions_delta': 0, 'wall_s': 300,
                   'latency_s': {'max': 250}, 'e2e_tok_s': {'p10': 16},
                   'waiting_steady_state': {'steady_peak': 6},
                   'metrics_before': [{'preemptions': 0} for _ in range(4)],
                   'metrics_after': [{'preemptions': 0} for _ in range(4)],
                   'gauges': [{'kv_cache_usage_perc': {'max': 0.7}} for _ in range(4)]}
        report = {'synthetic': True, 'dataset_rows': 0, 'generated_at': '2026-10-30T14:10:00Z',
                  'settings': {'model': 'model', 'base_urls': [profile.base_url(i) for i in profile.instances],
                               'thinking_budget': 3072, 'max_tokens': 4096, 'levels': [24],
                               'min_tokens': 4096 if kind == 'worst' else None},
                  'levels': [summary], 'requests': [{'status': 'ok'} for _ in range(288)]}
        report_pin = put(data, kind+'-report.json', report)
        measure = {'summary': {**summary, 'instances': 4}, 'instances': 4,
                   'report_sha256': report_pin['sha256'], 'min_hbm_free_mib': 4096,
                   'poll_kv_peak': [0.7]*4, 'poll_waiting_steady': 6,
                   'names': [i.name for i in profile.instances], 'max_tokens_sent': 4096,
                   'min_tokens_sent': 4096 if kind == 'worst' else None}
        binding[kind+'_report'] = report_pin
        binding[kind] = put(data, kind+'.json', {'measure': measure, 'passes': True, 'reasons': []})
    return binding, repo, data, put


def test_valid_binding_keeps_ceiling_and_client_admission_separate(evidence):
    binding, repo, data, _ = evidence
    result = validate_binding(binding, repo, data)
    assert result['admitted_level'] == 24
    assert result['server_max_num_seqs'] == 32
    assert result['lock_rows_per_minute_per_instance'] == pytest.approx(7.2)


def test_binding_pins_the_selected_model_fragment(evidence):
    binding, repo, data, put = evidence
    profile = load_deployment_profile(binding['profile'], repo, data_root=data)
    path = f'configs/model/{profile.model_fragment}.json'
    document = json.loads((REPO/path).read_bytes())
    binding['model_fragment'] = put(repo, path, document)
    assert validate_binding(binding, repo, data)['model_fragment'] == binding['model_fragment']
    (repo/path).write_text('{}')
    with pytest.raises(ValueError, match='hash mismatch'):
        validate_binding(binding, repo, data)
    binding['model_fragment'] = put(repo, 'configs/model/wrong.json', document)
    with pytest.raises(ValueError, match='differs from deployment'):
        validate_binding(binding, repo, data)


@pytest.mark.parametrize('mutation', ['receipt', 'startup', 'member', 'requests', 'synthetic', 'preemption', 'missing', 'nan', 'failed'])
def test_binding_refuses_mixed_or_incomplete_evidence(evidence, mutation):
    binding, repo, data, put = evidence
    key = 'worst_start' if mutation in ('receipt', 'startup') else 'worst'
    if mutation == 'member':
        key = 'members'
        binding['members'].pop('saea-omni-i3')
    elif mutation in ('requests', 'synthetic', 'preemption', 'missing'):
        key = 'worst_report'
    if key != 'members':
        pin = binding[key]
        doc = json.loads((data/pin['relpath']).read_bytes())
        if mutation == 'receipt': doc['receipt_sha256'] = 'f'*64
        elif mutation == 'startup': doc['deployment_observed'][0]['State']['Pid'] += 1
        elif mutation == 'requests': doc['requests'].pop()
        elif mutation == 'synthetic': doc['synthetic'] = False
        elif mutation == 'preemption': doc['levels'][0]['metrics_after'][1]['preemptions'] = 1
        elif mutation == 'missing': doc['levels'][0]['gauges'].pop()
        elif mutation == 'nan': doc['measure']['min_hbm_free_mib'] = float('nan')
        elif mutation == 'failed': doc['passes'] = False
        binding[key] = put(data, pin['relpath'], doc)
        if key == 'worst_report':
            # Re-sign the summary too: validation must check the raw counters, not trust a positive verdict.
            ev = json.loads((data/binding['worst']['relpath']).read_bytes())
            ev['measure']['summary'] = {**doc['levels'][0], 'instances': 4}
            ev['measure']['report_sha256'] = binding[key]['sha256']
            binding['worst'] = put(data, 'worst.json', ev)
    with pytest.raises(ValueError):
        validate_binding(binding, repo, data)

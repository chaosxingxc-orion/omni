import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location('summary', Path(__file__).with_name('summarize.py'))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(obj) + '\n')


def test_complete_trajectory_and_host_wall_not_sum_of_rates(tmp_path):
    candidates = [{'id': str(i), 'client_concurrency': 32} for i in range(4)]
    for i, c in enumerate(candidates):
        root = tmp_path / 'round-0' / c['id']
        write(root / 'segments/1/metrics.jsonl', {'time': 100, 'metrics': {}})
        for k in range(3):
            write(root / f'requests/{k}.result.json', {'result': {'status': 'ok', 'trajectory': 0, 'contact': k, 'completion_tokens': 10}})
            write(root / f'requests/{k}.intent.json', {})
        write(root / 'complete.json', {'wall_s': 10 * (i + 1), 'resumed': False, 'interrupted': False,
                                      'metrics_before': {}, 'metrics_after': {}})
    result = m.summarize({'rounds': [candidates], 'trajectories': 1}, tmp_path, now=150)
    assert result['host_generation_wall_s'] == 40
    assert result['host_seconds_per_trajectory'] == 10
    assert result['completed_requests'] == 12 and result['completed_trajectories'] == 4
    assert result['all_complete'] and not result['promotion']


def test_partial_counts_failures_unknowns_and_resume_without_rate(tmp_path):
    c = {'id': 'x', 'client_concurrency': 96}
    write(tmp_path / 'segments/1/metrics.jsonl', {'time': 100, 'metrics': {}})
    write(tmp_path / 'segments/2/metrics.jsonl', {'time': 200, 'metrics': {}})
    for k, status in enumerate(['ok', 'ok', 'timeout']):
        write(tmp_path / f'requests/{k}.result.json', {'result': {'status': status, 'trajectory': 0, 'contact': k}})
        write(tmp_path / f'requests/{k}.intent.json', {})
    write(tmp_path / 'requests/unknown.intent.json', {})
    result = m.trial(tmp_path, c, 2, 220)
    assert result['completed_requests'] == 2 and result['completed_trajectories'] == 0
    assert result['errors'] == 1 and result['inflight_or_unresolved_intents'] == 1
    assert result['ETA_s'] is None and result['output_tokens_s'] is None
    assert not result['uninterrupted_comparison_eligible']


def test_interrupted_terminal_is_not_running_and_has_no_eta(tmp_path):
    write(tmp_path / 'segments/1/metrics.jsonl', {'time': 100, 'metrics': {}})
    write(tmp_path / 'segments/1/result.json', {'wall_s': 20, 'interrupted': True, 'fit_wall_s': 5})
    for k in range(3):
        write(tmp_path / f'requests/{k}.result.json', {'result': {'status': 'ok', 'trajectory': 0, 'contact': k}})
    result = m.trial(tmp_path, {'id': 'x', 'client_concurrency': 32}, 2, 1000)
    assert result['state'] == 'stopped'
    assert result['generation_wall_s'] == 20
    assert result['completed_trajectories'] == 1
    assert result['ETA_s'] is None and result['seconds_per_complete_trajectory'] is None

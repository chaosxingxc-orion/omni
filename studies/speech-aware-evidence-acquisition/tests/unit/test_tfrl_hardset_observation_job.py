import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

P = Path(__file__).resolve().parents[2]/'scripts/analysis/tfrl_hardset_observation_job.py'
spec = importlib.util.spec_from_file_location('hardset_next_job', P)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def fixture():
    ids = ['slurp:train:%06d:audio-%d' % (i, i) for i in range(625)]
    items = [{'sample_id': s, 'audio_seconds': 2.0} for s in ids]
    pool = [{'sample_id': s, 'intent': 'music_play', 'slot_types': ['song_name']} for s in ids]
    queue = [{'sample_id': s, 'model_calls_authorized': False} for s in ids[:400]]
    rows = [{'sample_id': s, 'scenario_action_exact': int(i != 0),
             'slot_f1_exact_tp': 1, 'slot_f1_exact_fp': 0, 'slot_f1_exact_fn': 0}
            for i, s in enumerate(ids[:313])]
    scores = {m.SCORER: {'per_sample': rows}, 'cost': {}}
    docs = {
        'experiment': {}, 'pool': pool, 'queue': queue, 'h1': {'items': items[:313]}, 'h2': {'items': items[313:]},
        'scores': scores,
        'dataset': {'items': [{**r, 'audio_sha256': hashlib.sha256(r['sample_id'].encode()).hexdigest(),
                               'media_relpath': 'datasets/slurp_real/'+r['sample_id']+'.wav'} for r in items]},
        'attempt': {'run_id': m.SOURCE_RUN, 'attempt_id': 'generation-v1', 'state': 'FINALIZED',
                    'actual': {'calls': 626}},
        'model': {'decoding_params': {'temperature': 0, 'seed': 20260913, 'max_tokens': 4096}},
        'baseline': {'serving_slot_tokens': 49152, 'prepared_phase_supply': {'old': 'supply'},
                     'native_contact_fit': {},
                     'native_episode': {'roles': {k: {'max_iterations': int(k in ('perception', 'solver'))}
                                                   for k in ('perception', 'solver', 'planner', 'reviewer')},
                                        'consumption_module': {'columns': [], 'review_loop': 'off', 'components': []}}}}
    def record(k, v):
        text = ''.join(json.dumps(r)+'\n' for r in v) if k in ('pool', 'queue') else json.dumps(v)
        return {'text': text, 'sha256': m.digest(text.encode()), 'source_path': '/fixture/'+k}
    result = {k: record(k, v) for k, v in docs.items()}
    result['manifest'] = record('manifest', {'artifacts': {
        'candidate-pool.jsonl': result['pool']['sha256'],
        'difficulty-observation-queue.jsonl': result['queue']['sha256']}})
    result['run_manifest'] = record('run_manifest', {
        'run_id': m.SOURCE_RUN, 'status': 'COMPLETED', 'attempt': {'attempt_id': 'generation-v1'},
        'scores': {'sha256': result['scores']['sha256']}, 'cost': {'calls_used': 626},
        'scorers': [m.SCORER, 'cost']})
    return result


def run(b):
    return m.build(b, expected_manifest_sha256=b['run_manifest']['sha256'],
                   expected_pool_sha256=b['pool']['sha256'])


def mutate(b, name, change):
    d = json.loads(b[name]['text']); change(d)
    b[name]['text'] = json.dumps(d); b[name]['sha256'] = m.digest(b[name]['text'].encode())


def test_only_fixed_h1_errors_one_pass_h2_excluded_prior_cost_retained():
    b = fixture(); original = copy.deepcopy(b); result = run(b)
    p = result['plan']
    assert p['rows'] == 1 and p['normal_generation_calls'] == 2
    assert p['original_queue_h2_excluded'] == 87
    assert p['existing_full_ps_observations_per_h1'] == 1
    assert p['first_observation_missing_h1_rows'] == 0
    assert p['prior_source_cost']['calls_used'] == 626
    assert not p['independent_stochastic_draws_claimed'] and not p['hardness_confirmed']
    assert p['planned_additional_passes'] == 1 and not p['generation_authorized']
    assert 'prepared_phase_supply' not in result['baseline']
    assert b == original


@pytest.mark.parametrize('fault', ['score_hash', 'not_final', 'h2_overlap', 'duplicate', 'decoding', 'knowledge_experience', 'bad_counts'])
def test_refuse_incompatible_or_corrupt_inputs(fault):
    b = fixture()
    if fault == 'score_hash':
        mutate(b, 'scores', lambda d: d[m.SCORER]['per_sample'][0].update(scenario_action_exact=1))
    elif fault == 'not_final':
        mutate(b, 'attempt', lambda d: d.update(state='OPEN'))
    elif fault == 'h2_overlap':
        first = json.loads(b['h1']['text'])['items'][0]
        mutate(b, 'h2', lambda d: d['items'].__setitem__(0, first))
    elif fault == 'duplicate':
        mutate(b, 'h1', lambda d: d['items'].__setitem__(1, d['items'][0]))
    elif fault == 'decoding':
        mutate(b, 'model', lambda d: d['decoding_params'].update(temperature=0.5))
    elif fault == 'knowledge_experience':
        mutate(b, 'baseline', lambda d: d['native_episode']['consumption_module']['components'].append(
            {'component_id': 'tfgrpo-solver-experience', 'state': 'on'}))
    else:
        with pytest.raises(ValueError, match='slot counts'):
            m.task_error({'scenario_action_exact': 1, 'slot_f1_exact_tp': 0,
                          'slot_f1_exact_fp': True, 'slot_f1_exact_fn': 0})
        return
    with pytest.raises(ValueError):
        run(b)


def test_trusted_pin_and_exact_bytes_are_checked():
    b = fixture()
    with pytest.raises(ValueError, match='trusted source'):
        m.build(b, expected_manifest_sha256='0'*64, expected_pool_sha256=b['pool']['sha256'])
    b['queue']['text'] += ' '
    with pytest.raises(ValueError, match='source bytes'):
        run(b)

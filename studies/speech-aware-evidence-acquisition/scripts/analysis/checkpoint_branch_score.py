"""Official scoring of an explicitly registered completed-checkpoint subset.

Never fabricates a finalized parent run. Results describe only the checkpoint's
complete rows, including their completion-selection bias, not the full roster.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT / 'src', ROOT / 'scripts/analysis', ROOT / 'scripts/checks'):
    sys.path.insert(0, str(directory))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(recipe_path, data, out):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.row_checkpoint import read_checkpoint
    from speech_aware_evidence_acquisition.core.base import DELIVERY_NORMALISERS
    from speech_aware_evidence_acquisition.rails import delivery_normalisers
    from speech_aware_evidence_acquisition.rails.native_episode_receipt import validate_native_episode_delivery_receipt
    from speech_aware_evidence_acquisition.rails.supply_variants import parse_perception_sampling_variants
    from speech_aware_evidence_acquisition.reproduction.slurp_official_eval import score_slurp_outputs_official_train
    from speech_aware_evidence_acquisition.data.lock import load_lock, umbrella_lock_path
    from speech_aware_evidence_acquisition.core.atomic_file import publish_new
    from types import SimpleNamespace
    from stage3_rollout_score import _committed_solver_samples
    import flight_registration_check as registration
    assert_execution_environment()
    recipe = json.loads(recipe_path.read_bytes())
    entries = registration.parse_ledger(ROOT, Path('docs/exposure-ledger.md')).get(recipe['read_run_id'], [])
    if (len(entries) != 1 or not registration.is_countersigned(entries[0])
            or entries[0]['artifacts'] != str(recipe_path.relative_to(ROOT))
            or sha(recipe_path) not in entries[0].get('notes', '')):
        raise ValueError('checkpoint READ must be uniquely registered and pinned')
    if (recipe['schema'] != 'saea-checkpoint-branch-read-v1' or recipe['new_model_calls'] != 0
            or recipe['scorer'] != 'slurp-slu-official-train'):
        raise ValueError('unreviewed checkpoint read scope')
    source = (data / recipe['checkpoint']['relpath']).resolve()
    if not source.is_relative_to(data.resolve()): raise ValueError('checkpoint path escapes data root')
    checkpoint = read_checkpoint(source, expected_sha256=recipe['checkpoint']['sha256'])
    if any(row.keep_defect() is not None for row in checkpoint.rows):
        raise ValueError('checkpoint contains incomplete or invalid rows')
    if len(checkpoint.rows) != recipe['rows']:
        raise ValueError('registered checkpoint row count differs')
    config_path = ROOT / recipe['baseline']['relpath']
    if sha(config_path) != recipe['baseline']['sha256']: raise ValueError('baseline pin differs')
    base = json.loads(config_path.read_bytes())
    group = parse_perception_sampling_variants(base['native_episode']['roles']['perception']['sampling_variants'], phase_id='perception')
    names = list(group.branch_ids) + ['l1-support-sample0']
    if names != recipe['columns']: raise ValueError('column inventory differs')
    out.mkdir(parents=True, exist_ok=True)
    publish_new(out / 'intent.json', json.dumps({'recipe_sha256': sha(recipe_path), 'new_model_calls': 0}).encode())
    columns = {name: [] for name in names}
    for row in checkpoint.rows:
        output, = row.output_rows
        episode = output['native_episode_receipt']
        validate_native_episode_delivery_receipt(episode, error_type=ValueError)
        records = {}
        for line in row.trace_lines:
            event = json.loads(line)
            if event['record_id'] in records: raise ValueError('duplicate trace record')
            records[event['record_id']] = event['payload']
        contacts = {c['agent_contact_id']: c for visit in episode['role_visits']
                    for c in visit['execution_receipt']['contacts']}
        normalise = DELIVERY_NORMALISERS.resolve(episode['deliver_normaliser']['normaliser_id'])
        branches = episode['perception_branches']['branches']
        if [b['branch_id'] for b in branches] != list(group.branch_ids): raise ValueError('row branch inventory differs')
        for i, branch in enumerate(branches):
            name = branch['branch_id']
            contact = contacts[branch['solver_contact_id']]
            request = records[contact['request_id'] + '/request']
            response = records[contact['request_id'] + '/response']
            raw = response['text']
            if (request['payload_sha256'] != contact['request_sha256']
                    or hashlib.sha256(raw.encode()).hexdigest() != contact['response_text_sha256']
                    or branch['solver_response_text_sha256'] != contact['response_text_sha256']):
                raise ValueError('branch contact hash mismatch')
            if group.branch_at(i).solver_samples is not None:
                samples = branch['sampling']['solver_samples']['samples']
                own = [(s['sample_index'], contacts[s['contact_id']]) for s in samples]
                # Shared renderer verifies every sample and replays the exact
                # support commit; N1 remains unavailable as in execution.
                bundle = SimpleNamespace(trace_records=records)
                raw, sample0, _ = _committed_solver_samples(row.sample_id, name,
                    group.branch_at(i).solver_samples, own, branch, bundle, None)
                columns[name + '-sample0'].append({'sample_id': row.sample_id, 'text': normalise(sample0).text})
            rendered = normalise(raw).text
            if name == 'lock' and rendered != output['text']:
                raise ValueError('lock re-render differs from original delivered output')
            columns[name].append({'sample_id': row.sample_id, 'text': rendered})
    lock = load_lock(umbrella_lock_path(ROOT))
    summary = {'schema': 'saea-checkpoint-branch-score-v1', 'rows': recipe['rows'],
               'checkpoint_sha256': recipe['checkpoint']['sha256'], 'new_model_calls': 0,
               'selection': 'completed rows only; not random; no full-roster benefit claim', 'columns': {}}
    for name, rows in columns.items():
        path = out / (name + '.outputs.jsonl')
        publish_new(path, ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows).encode())
        result = score_slurp_outputs_official_train(path, lock, data)
        if not result['aggregate']['identity_check']['passed'] or len(result['per_sample']) != recipe['rows']:
            raise ValueError('official scorer identity or row count differs')
        target = out / (name + '.scores.json')
        publish_new(target, json.dumps(result, indent=2, sort_keys=True).encode())
        summary['columns'][name] = {'aggregate': result['aggregate'], 'outputs_sha256': sha(path), 'scores_sha256': sha(target)}
    publish_new(out / 'complete.json', json.dumps(summary, indent=2, sort_keys=True).encode())
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('recipe', 'data-root', 'out'): parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = run(args.recipe, args.data_root, args.out)
    print(json.dumps({'rows': result['rows'], 'columns': list(result['columns']), 'new_model_calls': 0}))

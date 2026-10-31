"""Qualify complete Agent repeats from sealed procedure and output custody.

Re-render in temporary storage without scoring or model contact. Emit hashes and
counts only. A failed qualification requires complete-system repeat flights;
branch noise remains a separate diagnostic and is never silently substituted.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import tempfile

REPO = Path(__file__).resolve().parents[2]
for directory in (REPO / 'src', REPO / 'scripts/analysis', REPO / 'scripts/flights/nx'):
    sys.path.insert(0, str(directory))
import agent_port_column_read as columns
import stage3_rollout_score as renderer
from speech_aware_evidence_acquisition.core.config import compose
from speech_aware_evidence_acquisition.core.atomic_file import publish_new
from speech_aware_evidence_acquisition.contracts import assert_execution_environment

COPIES = ('lock', 'aa', 'lock-rep2')
SCHEMA = 'saea-agent-complete-repeat-custody-v1'


def audit(bundle, values, roster, roster_sha256, stored):
    """Require a verified bundle; re-derive every copy before qualification."""
    native = values['native_episode']
    roles = native['roles']
    group = renderer.parse_perception_arm_group(values, roles['perception'], values.get('knowledge_stores', {}))
    reasons = []
    if (native.get('max_revisions') != 0 or native.get('review_loop') != 'off'
            or any(roles.get(role, {}).get('max_iterations') != count
                   for role, count in [('perception', 1), ('solver', 1), ('planner', 0), ('reviewer', 0)])):
        reasons.append('procedure is not a single Perception-Solver pass with review disabled')
    declared = roles['perception'].get('sampling_variants', {}).get('arms', [])
    by_name = {arm.get('arm_id'): arm for arm in declared}
    expected = {'lock': {'arm_id': 'lock', 'deliver': True},
                'aa': {'arm_id': 'aa', 'replicate_of': 'lock'},
                'lock-rep2': {'arm_id': 'lock-rep2', 'arm_kind': 'replicate', 'replicate_of': 'lock'}}
    if any(by_name.get(name) != expected[name] for name in COPIES):
        reasons.append('the three repeat declarations differ from the registered lock procedure')
    if not set(COPIES).issubset(group.branch_ids):
        raise ValueError('required repeat columns are absent')
    if bundle.manifest.get('predecessor_attempt') or bundle.manifest.get('predecessor_attempts'):
        # Such bundles can be valid, but qualification needs a separately reviewed
        # cross-attempt procedure proof; do not infer it from equal predictions.
        return {'schema': SCHEMA, 'qualified': False, 'reasons': ['cross-attempt repeat custody requires separate review'],
                'rows': len(roster), 'copies': list(COPIES), 'claims_benefit': False}
    extra_visits = 0
    for row in bundle.outputs_rows:
        episode = row.get('native_episode_receipt', {})
        active = Counter(v.get('stage') for v in episode.get('role_visits', [])
                         if v.get('execution_receipt', {}).get('contacts'))
        if active != {'perception': 1, 'solver': 1}:
            extra_visits += 1
    if extra_visits:
        reasons.append('some rows lack exactly one complete Perception and Solver visit')
    stored_manifest = columns.registration.read(stored / 'render-manifest.json')
    with tempfile.TemporaryDirectory(prefix='agent-repeat-custody-') as temporary:
        fresh = renderer.render_perception_branch_columns(bundle, group, roster, Path(temporary),
            run_id=bundle.manifest['run_id'], attempt_id=bundle.manifest_path.parent.name,
            roster_sha256=roster_sha256, config_hash=bundle.manifest['config']['config_hash'])
        if fresh['source'] != stored_manifest['source']:
            raise ValueError('saved column source differs from the verified complete system')
        for name in ('sampling-rows.json', 'first-visit-rows.json'):
            fresh_path, saved_path = Path(temporary) / name, stored / name
            if (fresh_path.exists() != saved_path.exists()
                    or (fresh_path.exists() and fresh_path.read_bytes() != saved_path.read_bytes())):
                raise ValueError('saved diagnostic sidecar differs from source-bound re-render')
        for name, entry in fresh['copies'].items():
            saved = stored_manifest['copies'][name]
            path = columns.delivered.shared.within(saved['path'], stored)
            if (saved['sha256'] != entry['sha256'] or saved['rows'] != len(roster)
                    or columns.registration.sha(path) != entry['sha256']):
                raise ValueError('saved column differs from source-bound re-render')
        final = {row['sample_id']: row.get('text') for row in bundle.outputs_rows}
        lock = {row['sample_id']: row['text'] for row in renderer.iter_jsonl(Path(fresh['copies']['lock']['path']))}
        equality = set(final) == set(lock) == set(roster) and all(
            isinstance(final[sid], str) and final[sid] == lock[sid] for sid in roster)
        if not equality:
            reasons.append('delivering copy does not equal every sealed final output')
        if fresh['at1_column_reproduction_checked_rows'] != len(roster) or fresh['later_solver_visit_count']:
            reasons.append('not every row has a final-delivery reproduction check without later visits')
        if any(fresh['solver_absent_by_branch'][name] for name in COPIES):
            reasons.append('a repeat lacks an authoritative Solver answer')
    return {'schema': SCHEMA, 'qualified': not reasons, 'reasons': reasons, 'rows': len(roster),
            'copies': list(COPIES), 'source': fresh['source'], 'all_final_outputs_equal_lock': equality,
            'reproduced_rows': fresh['at1_column_reproduction_checked_rows'],
            'copy_hashes': {name: fresh['copies'][name]['sha256'] for name in COPIES},
            'repeat_scope': 'complete Perception-Solver-delivery procedure within one flight',
            'limitation': 'Within-flight repeats do not estimate deployment drift between flights.',
            'claims_benefit': False, 'new_scoring_jobs': 0, 'model_contacts': 0}


def build(repo, data):
    spec, _, recipe = columns.require_recipe(repo)
    if recipe.get('repeat_custody') != {
        'reader': 'scripts/analysis/agent_port_repeat_evidence.py', 'copies': list(COPIES),
        'requires_all_final_output_equality': True, 'requires_complete_procedure': True,
        'on_unqualified': 'register complete-system repeats; keep branch noise separate'}:
        raise ValueError('repeat qualification must be declared before the Agent flight')
    final = columns.delivered.build(repo, data, spec['cell'], spec['read_run_id'],
                                    repo / columns.registration.PKG / 'analysis-recipe-v1.json')
    root = data / 'analysis' / columns.registration.BRANCH_READ
    columns.verify_columns(root, data, spec)
    directory = data / 'runs' / spec['run_id'] / spec['attempt_id']
    bundle = columns.delivered.shared.verify_bundle(directory / (spec['run_id'] + '.run-manifest.json'))
    fragments = bundle.manifest['config']['fragments']
    cfg = compose(repo, *(fragments[key] for key in ('model', 'dataset', 'baseline', 'experiment')))
    if cfg.config_hash != bundle.manifest['config']['config_hash']:
        raise ValueError('repeat procedure differs from sealed configuration')
    roster, roster_sha, _ = renderer.load_roster(repo, columns.reader.fam.ROSTER[columns.reader.pf.CELLS['arm']['roster']])
    result = audit(bundle, cfg.values, roster, roster_sha, root / 'columns')
    result['inputs'] = {**final['inputs'], str(repo / columns.registration.BRANCH_RECIPE): columns.registration.sha(repo / columns.registration.BRANCH_RECIPE),
                        str(root / 'columns/render-manifest.json'): columns.registration.sha(root / 'columns/render-manifest.json')}
    for name in ('sampling-rows.json', 'first-visit-rows.json'):
        path = root / 'columns' / name
        if path.exists():
            result['inputs'][str(path)] = columns.registration.sha(path)
    result['implementation'] = {str(path): columns.registration.sha(path)
        for path in (Path(__file__), Path(renderer.__file__))}
    result['read_run_id'] = columns.registration.BRANCH_READ
    return result


def main():
    assert_execution_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=REPO)
    parser.add_argument('--data-root', type=Path, required=True)
    args = parser.parse_args()
    result = build(args.repo_root, args.data_root)
    target = args.data_root / 'analysis' / columns.registration.BRANCH_READ / 'complete-repeat-custody.json'
    publish_new(target, columns.registration.raw(result))
    print(json.dumps({key: result[key] for key in ('qualified', 'rows', 'reasons')}))


if __name__ == '__main__':
    main()

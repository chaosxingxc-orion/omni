"""Run the separately registered complete Agent branch diagnostics without model calls."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
for directory in (REPO / 'src', REPO / 'scripts/flights/nx', REPO / 'scripts/analysis'):
    sys.path.insert(0, str(directory))
import agent_port_full_registration as registration
import agent_port_score_evidence as delivered
import slurp_l338_port_column_read as reader
from speech_aware_evidence_acquisition.core.atomic_file import publish_new
from speech_aware_evidence_acquisition.contracts import assert_execution_environment
from speech_aware_evidence_acquisition.core.scoring_batch import score_complete_derived_copy


def identity(repo):
    spec = registration.read(repo / registration.PKG / 'successor-draft-v1.json')
    smoke = registration.read(repo / 'configs/experiment/slurp-delivery-assets-proof-4-npu-v1.json')['cells'][0]
    return spec, SimpleNamespace(READ_RUN_ID=registration.BRANCH_READ, CELLS={
        'arm': {'run': spec['run_id'], 'attempt': spec['attempt_id'], 'roster': reader.pf.CELLS['arm']['roster']},
        'smoke': {'run': smoke['run_id']}})


def require_recipe(repo):
    spec, family = identity(repo)
    path = repo / registration.BRANCH_RECIPE
    recipe = reader.load_recipe(path, family=family)
    rows = delivered.shared.registration.parse_ledger(repo, Path('docs/exposure-ledger.md')).get(family.READ_RUN_ID, [])
    if (len(rows) != 1 or not delivered.shared.registration.is_countersigned(rows[0])
            or rows[0].get('artifacts') != registration.BRANCH_RECIPE
            or registration.sha(path) not in rows[0].get('notes', '')):
        raise ValueError('branch READ must be uniquely registered and hash-pinned')
    exp = registration.read(repo / 'configs/experiment' / (spec['cell'] + '.json'))
    expected = {'run_id': spec['run_id'], 'attempt_id': spec['attempt_id'], 'rows': 1080,
                'split_identity_hash': exp['population']['split_identity_hash']}
    if (recipe.get('identity') != expected or recipe.get('score_surface') != 'first-solver-level2'
            or recipe.get('official_complete_scoring_jobs') != 6
            or recipe.get('final_delivery_read') != spec['read_run_id']):
        raise ValueError('branch recipe scope differs from the complete Agent')
    return spec, family, recipe


def score_command(repo, data, spec, root, stage):
    return [sys.executable, str(repo / 'scripts/analysis/stage3_rollout_score.py'),
        '--mode', 'perception-branches', '--stage', stage, '--run', spec['run_id'],
        '--attempt', spec['attempt_id'], '--roster', reader.fam.ROSTER[reader.pf.CELLS['arm']['roster']],
        '--analysis-root', str(root / 'columns'), '--scorer-id', reader.hist.SCORER,
        '--repo-root', str(repo), '--data-root', str(data)]


def require_complete_columns(root):
    """Require all completed columns before source-bound cache verification."""
    columns = root / 'columns'
    manifest = registration.read(columns / 'render-manifest.json')
    names = [*reader.pf.ARM_ORDER, *reader.hist.stream_column_names()]
    if (manifest.get('rows') != 1080 or manifest.get('branch_order') != list(reader.pf.ARM_ORDER)
            or manifest.get('stream_order') != reader.hist.stream_column_names()
            or set(manifest.get('copies', {})) != set(names)):
        raise ValueError('complete five branches and sample0 stream required')
    for name in names:
        path = columns / reader.hist.COLUMN_SCORE_NAME.format(arm=name, scorer=reader.hist.SCORER)
        receipt = path.with_name(path.name + '.receipt.json')
        if not path.is_file() or not receipt.is_file():
            raise ValueError('incomplete column score; no automatic scorer replay')
        if registration.read(receipt).get('result_sha256') != registration.sha(path):
            raise ValueError('column result hash differs from receipt')


def verify_columns(root, data, spec):
    require_complete_columns(root)
    columns = root / 'columns'
    manifest_path = columns / 'render-manifest.json'
    manifest = registration.read(manifest_path)
    parent = data / 'runs' / spec['run_id'] / spec['attempt_id'] / (spec['run_id'] + '.run-manifest.json')
    source = manifest['source']
    if (source['run_id'] != spec['run_id'] or source['attempt_id'] != spec['attempt_id']
            or Path(source['manifest_path']).resolve() != parent.resolve()
            or source['manifest_sha256'] != registration.sha(parent)):
        raise ValueError('branch render belongs to another sealed source')
    for name, entry in manifest['copies'].items():
        copy_path = delivered.shared.within(Path(entry['path']), columns)
        score_path = columns / reader.hist.COLUMN_SCORE_NAME.format(arm=name, scorer=reader.hist.SCORER)
        result = score_complete_derived_copy(parent_manifest=parent, outputs_copy=copy_path,
            expected_copy_sha256=entry['sha256'], scorer_id=reader.hist.SCORER,
            result_path=score_path, lock={}, data_root=data, cache_only=True,
            column_binding={'branch_id': name, 'render_manifest_sha256': registration.sha(manifest_path),
                'roster_sha256': source['roster_sha256'], 'visit': 'first-solver-level2'})
        if (len(result['per_sample']) != 1080
                or len({row['sample_id'] for row in result['per_sample']}) != 1080
                or not result['aggregate']['identity_check']['passed']):
            raise ValueError('column score is not a complete official roster')


def run(repo, data, stage, gold_intents=None, n1_json=None):
    assert_execution_environment()
    repo, data = repo.resolve(), data.resolve()
    spec, family, recipe = require_recipe(repo)
    # Verify canonical final delivery before either secondary action.
    final = delivered.build(repo, data, spec['cell'], spec['read_run_id'], repo / registration.PKG / 'analysis-recipe-v1.json')
    root = data / 'analysis' / family.READ_RUN_ID
    root.mkdir(parents=True, exist_ok=True)
    for name in (stage + '.intent.json', stage + '.complete.json', stage + '.failed.json'):
        if (root / name).exists():
            raise ValueError('stage already started; inspect existing state without replay')
    if stage == 'columns' and (root / 'columns').exists():
        raise ValueError('column output already exists; inspect without replay')
    if stage == 'read':
        require_complete_columns(root)
        if any((root / name).exists() for name in ('read.json', 'README.md', reader.pf.VERDICTS_NAME)):
            raise ValueError('branch report already exists; never overwrite')
    inputs = {str(repo / registration.BRANCH_RECIPE): registration.sha(repo / registration.BRANCH_RECIPE)}
    for path in (gold_intents, n1_json):
        if path is not None:
            inputs[str(path)] = registration.sha(path)
    publish_new(root / (stage + '.intent.json'), registration.raw({'stage': stage, 'inputs': inputs,
        'identity': recipe['identity'], 'final_delivery_inputs': final['inputs'],
        'model_contacts': 0, 'new_scoring_jobs': 6 if stage == 'columns' else 0}))
    try:
        if stage == 'columns':
            with (root / (stage + '.log')).open('x') as stream:
                result = subprocess.run(score_command(repo, data, spec, root, 'all'),
                    cwd=repo, stdout=stream, stderr=subprocess.STDOUT,
                    env={**os.environ, 'SAEA_ROLLOUT_TRACE_READ_ID': family.READ_RUN_ID})
            if result.returncode:
                raise ValueError('source-bound complete column scoring failed')
        verify_columns(root, data, spec)
        if stage == 'read':
            args = SimpleNamespace(repo_root=repo, data_root=data, recipe=repo / registration.BRANCH_RECIPE,
                seed=reader.hist.SEED, draws=reader.hist.DRAWS, gold_intents=gold_intents, n1_json=n1_json)
            report, out, meta, verdict_out = reader._read(args, root, family=family)
            report['score_surface'] = 'first-solver-level2'
            report['diagnostic_input_availability'] = {'gold_intents': gold_intents is not None, 'n1': n1_json is not None}
            report['additional_inputs'] = inputs
            report_bytes = registration.raw(report)
            citations = dict(meta['inputs'], read_json={'path': str(out), 'sha256': hashlib.sha256(report_bytes).hexdigest()})
            interval = dict(report['interval'], registered=True)
            verdict = reader.port_verdicts(report['contrast'], report['arm_words'], meta['rules'],
                inputs=citations, interval=interval, recipe_status=recipe['port_extensions']['status'], family=family)
            publish_new(out, report_bytes)
            publish_new(verdict_out, registration.raw(verdict))
            publish_new(root / 'README.md', (reader.hist.render_markdown(report) + '\n' + reader.noise_markdown(report)).encode())
        publish_new(root / (stage + '.complete.json'), registration.raw({'stage': stage, 'rows': 1080,
            'columns': 6, 'model_contacts': 0, 'new_scoring_jobs': 6 if stage == 'columns' else 0,
            'claims_system_benefit': False}))
    except Exception as error:
        publish_new(root / (stage + '.failed.json'), registration.raw({'error': str(error), 'automatic_retry': False}))
        raise
    return {'stage': stage, 'root': str(root), 'rows': 1080, 'columns': 6}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=REPO)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--stage', choices=['columns', 'read'], required=True)
    parser.add_argument('--gold-intents', type=Path)
    parser.add_argument('--n1-json', type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.repo_root, args.data_root, args.stage, args.gold_intents, args.n1_json)))


if __name__ == '__main__':
    main()

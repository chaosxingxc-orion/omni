"""Register and score the current A0's six complete Fold A diagnostic columns.

The executed source checkout is immutable. Registration lives in a separate
analysis checkout; all scoring uses the standard source-bound derived-copy path.
These are within-flight mechanism comparisons, never a Direct/A0 system verdict.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'flights/nx'))
import fold_a_current_registration as reg
import agent_port_score_evidence as delivered
import slurp_l338_port_column_read as reader
from speech_aware_evidence_acquisition.core.atomic_file import publish_new
from speech_aware_evidence_acquisition.core.scoring_batch import score_complete_derived_copy

v = reg.evaluation
READ = v.RUN + '-COLUMNS-READ'
RECIPE = 'docs/readiness/2026-10-31-current-a0-fold-a-columns-read-npu-v1/analysis-recipe-v1.json'
ROSTER = v.comparator.inputs.ROSTER
NAMES = [*reader.pf.ARM_ORDER, *reader.hist.stream_column_names()]


def clean(repo, expected=None):
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo).strip():
        raise ValueError('clean committed checkout required')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    if expected is not None and head != expected:
        raise ValueError('source commit differs')
    return head


def draft(source, data, admitted, completed):
    source, data = source.resolve(), data.resolve()
    admitted, completed = (v.comparator.inputs.confined(data, str(p)) for p in (admitted, completed))
    admission, campaign = reg.read(admitted), reg.read(completed)
    head = clean(source, admission['source_commit'])
    if (campaign['commit'] != head or campaign['run_id'] != v.RUN or campaign['rows'] != 1080
            or campaign['score_surface'] != 'sealed-final-delivery'
            or admission['run_id'] != v.RUN or admission['rows'] != 1080
            or admission['model_contacts'] != 0):
        raise ValueError('complete admitted current A0 Fold A required')
    for name, digest in admission['inputs'].items():
        if reg.sha(Path(name)) != digest:
            raise ValueError('admitted input changed')
    exp = reg.read(source/f'configs/experiment/{v.CELL}.json')
    cell, = exp['cells']
    roster = reg.read(source/ROSTER)
    ids = [r['sample_id'] for r in roster['items']]
    if (len(ids) != 1080 or len(set(ids)) != 1080 or exp['populations'][0]['ids'] != ids
            or exp['population']['split_identity_hash'] != roster['split_identity_hash']
            or (cell['run_id'], cell['attempt_id']) != (v.RUN, 'generation-v1')
            or cell['readiness']['state'] != 'ready'):
        raise ValueError('exact complete Fold A source and roster required')
    paths = [source/ROSTER, admitted, completed, source/f'configs/experiment/{v.CELL}.json',
        source/f'configs/baseline/{v.CELL}.json', source/reg.PKG/'analysis-recipe-v1.json',
        source/'docs/exposure-ledger.md', source/'scripts/analysis/stage3_rollout_score.py']
    implementations = [Path(__file__), Path(reader.__file__), Path(reader.hist.__file__),
        Path(reader.paired_ci.__file__), Path(delivered.__file__)]
    return {'schema': 'saea-current-a0-fold-a-column-recipe-v1', 'read_run_id': READ,
        'authority': reg.AUTH, 'source_repo': str(source), 'source_commit': head,
        'admitted': str(admitted), 'completed': str(completed),
        'identity': {'run_id': v.RUN, 'attempt_id': 'generation-v1', 'rows': 1080,
                     'split_identity_hash': roster['split_identity_hash']},
        'roster': ROSTER, 'columns': NAMES, 'scorer': reader.hist.SCORER,
        'score_surface': 'first-solver-level2', 'new_model_calls': 0, 'official_complete_scoring_jobs': 6,
        'seed': reader.hist.SEED, 'draws': reader.hist.DRAWS,
        'replicate_copies': list(reader.pf.REPLICATE_SET), 'noise_formula': reader.paired_ci.FORMULA,
        'historical_fold_b_parity_applies': False, 'claims_system_gain': False,
        'inputs': {str(p): reg.sha(p) for p in paths},
        'implementation': {str(p): reg.sha(p) for p in implementations}}


def register(repo, source, data, admitted, completed):
    if repo.resolve() == source.resolve():
        raise ValueError('separate analysis checkout required')
    clean(repo)
    recipe = draft(source, data, admitted, completed)
    path, ledger = repo/RECIPE, repo/'docs/exposure-ledger.md'
    root = data/'analysis'/READ
    before = ledger.read_bytes()
    if path.exists() or root.exists() or ('| '+READ+' |').encode() in before:
        raise ValueError('READ already registered; inspect without replay')
    blob = reg.raw(recipe)
    digest = hashlib.sha256(blob).hexdigest()
    root.mkdir(parents=True, exist_ok=False)
    publish_new(root/'registration.intent.json', reg.raw({'recipe_sha256': digest, 'new_model_calls': 0}))
    publish_new(path, blob); path.chmod(0o644)
    fields = ['2026-10-31', READ, 'model-free-check', 'Six complete current A0 Fold A mechanism columns',
        'none', 'slurp TRAIN Fold A 1080', 'discovery', recipe['identity']['split_identity_hash'],
        'no', '0', '0', '0', 'Sealed same-flight columns; six complete CPU scorer jobs.', RECIPE,
        f'REGISTERED BEFORE COLUMN SCORING; root countersigned under {reg.AUTH}. Plan SHA256 {digest}; no system promotion.']
    if ledger.read_bytes() != before:
        raise ValueError('ledger changed before registration')
    with ledger.open('a') as f:
        f.write('\n## Current A0 Fold A diagnostic columns\n\n'
            '| date | run/experiment id | execution profile | scope | model/tool | speech carrier + split | split role | split identity hash | consumed | calls | speech audio seconds | gpu hours | inherited exposure | artifacts | notes |\n'
            +'| '+' | '.join(['---']*15)+' |\n| '+' | '.join(fields)+' |\n')
    publish_new(root/'registration.complete.json', reg.raw({'recipe_sha256': digest, 'new_model_calls': 0}))
    return {'files': [RECIPE, 'docs/exposure-ledger.md'], 'recipe_sha256': digest}


def require_recipe(repo, data):
    clean(repo)
    path = repo/RECIPE; recipe = reg.read(path)
    rows = delivered.shared.registration.parse_ledger(repo, Path('docs/exposure-ledger.md')).get(READ, [])
    if (len(rows) != 1 or not delivered.shared.registration.is_countersigned(rows[0])
            or rows[0].get('artifacts') != RECIPE or reg.sha(path) not in rows[0].get('notes', '')):
        raise ValueError('uniquely registered hash-pinned diagnostic READ required')
    expected = draft(Path(recipe['source_repo']), data, Path(recipe['admitted']), Path(recipe['completed']))
    if recipe != expected:
        raise ValueError('recipe or source inputs changed')
    if reg.read(data/'analysis'/READ/'registration.complete.json')['recipe_sha256'] != reg.sha(path):
        raise ValueError('completed registration differs')
    return recipe


def command(source, data, root):
    return [sys.executable, str(source/'scripts/analysis/stage3_rollout_score.py'),
        '--mode', 'perception-branches', '--stage', 'all', '--run', v.RUN, '--attempt', 'generation-v1',
        '--roster', ROSTER, '--analysis-root', str(root/'columns'), '--scorer-id', reader.hist.SCORER,
        '--repo-root', str(source), '--data-root', str(data)]


def verify_columns(source, data, root):
    directory = root/'columns'; render_path = directory/'render-manifest.json'
    manifest = reg.read(render_path)
    parent = data/'runs'/v.RUN/'generation-v1'/(v.RUN+'.run-manifest.json')
    origin = manifest['source']
    if (manifest.get('rows') != 1080 or manifest.get('branch_order') != list(reader.pf.ARM_ORDER)
            or manifest.get('stream_order') != reader.hist.stream_column_names()
            or set(manifest.get('copies', {})) != set(NAMES)
            or origin['run_id'] != v.RUN or origin['attempt_id'] != 'generation-v1'
            or Path(origin['manifest_path']).resolve() != parent.resolve()
            or origin['manifest_sha256'] != reg.sha(parent) or origin['roster_sha256'] != reg.sha(source/ROSTER)):
        raise ValueError('complete source-bound Fold A column render required')
    ids = {r['sample_id'] for r in reg.read(source/ROSTER)['items']}
    pins = {str(render_path): reg.sha(render_path)}
    for name in NAMES:
        entry = manifest['copies'][name]
        copy = delivered.shared.within(Path(entry['path']), directory)
        score = directory/reader.hist.COLUMN_SCORE_NAME.format(arm=name, scorer=reader.hist.SCORER)
        receipt = score.with_name(score.name+'.receipt.json')
        if not score.is_file() or not receipt.is_file():
            raise ValueError('incomplete column scores; no automatic scorer replay')
        result = score_complete_derived_copy(parent_manifest=parent, outputs_copy=copy,
            expected_copy_sha256=entry['sha256'], scorer_id=reader.hist.SCORER, result_path=score,
            lock={}, data_root=data, cache_only=True, column_binding={'branch_id': name,
                'render_manifest_sha256': reg.sha(render_path), 'roster_sha256': origin['roster_sha256'],
                'visit': 'first-solver-level2'})
        if (len(result['per_sample']) != 1080 or {r['sample_id'] for r in result['per_sample']} != ids
                or not result['aggregate']['identity_check']['passed']):
            raise ValueError('column scores do not cover the exact Fold A roster')
        pins.update({str(p): reg.sha(p) for p in (copy, score, receipt)})
    return pins


def summarize(source, root):
    ids, cluster, scenario = reader.base.roster_maps(source/ROSTER)
    scored = reader.load_port_column_rows(root/'columns')
    if any(set(rows) != set(ids) for rows in scored.values()):
        raise ValueError('column statistics require the exact roster')
    bootstrap = reader.npc.paired_contrasts(scored, ids, cluster, seed=reader.hist.SEED,
        draws=reader.hist.DRAWS, delivering='lock')
    contrast = reader.paired_ci.noise_adjusted_contrast(scored, ids, cluster, bootstrap,
        copies=reader.pf.REPLICATE_SET, delivering='lock', seed=reader.hist.SEED, draws=reader.hist.DRAWS)
    return {'rows': len(ids), 'contrast': contrast, 'contrast_bootstrap_only': bootstrap,
        'per_scenario': reader.npc.per_scenario_table(scored, ids, scenario, 'lock'),
        'read_run_id': READ, 'score_surface': 'first-solver-level2', 'claims_system_gain': False,
        'new_model_calls': 0, 'new_scoring_jobs': 0,
        'limitations': ['Within-flight mechanism and branch-repeat evidence only.',
            'Fold B historical parity bands do not apply to this roster.',
            'No Direct/A0 paired system claim or TFRL promotion.',
            'Gold-intent/N1 diagnostics and full-system repeat qualification remain separate.']}


def run(repo, data, stage):
    recipe = require_recipe(repo, data)
    source = Path(recipe['source_repo']); root = data/'analysis'/READ
    final = delivered.build(source, data, v.CELL, reg.READ, source/reg.PKG/'analysis-recipe-v1.json')
    if stage == 'columns' and (root/'columns').exists():
        raise ValueError('column output already exists; inspect without replay')
    if stage == 'read' and not (root/'columns.complete.json').is_file():
        raise ValueError('complete column stage required')
    for suffix in ('intent.json', 'complete.json', 'failed.json'):
        if (root/(stage+'.'+suffix)).exists():
            raise ValueError('stage already started; inspect without replay')
    publish_new(root/(stage+'.intent.json'), reg.raw({'recipe_sha256': reg.sha(repo/RECIPE),
        'final_delivery_inputs': final['inputs'], 'new_model_calls': 0,
        'new_scoring_jobs': 6 if stage == 'columns' else 0}))
    try:
        if stage == 'columns':
            with (root/'columns.log').open('x') as log:
                subprocess.run(command(source, data, root), cwd=source, check=True, stdout=log,
                    stderr=subprocess.STDOUT, env={**os.environ, 'PYTHONPATH': str(source/'src'),
                                                   'SAEA_ROLLOUT_TRACE_READ_ID': READ})
        pins = verify_columns(source, data, root)
        if stage == 'read':
            report = summarize(source, root)
            report['inputs'] = {**pins, **final['inputs'], str(repo/RECIPE): reg.sha(repo/RECIPE)}
            publish_new(root/'read.json', reg.raw(report))
        require_recipe(repo, data)
        publish_new(root/(stage+'.complete.json'), reg.raw({'rows': 1080, 'columns': 6,
            'inputs': pins, 'new_model_calls': 0, 'claims_system_gain': False}))
    except BaseException as error:
        publish_new(root/(stage+'.failed.json'), reg.raw({'reason': str(error), 'automatic_retry': False}))
        raise
    return {'rows': 1080, 'stage': stage, 'new_model_calls': 0}


def queued(repo, source, data, admitted, completed, queue_state, queue_id):
    from fold_a_current_admit_queued import queue_done
    from fold_a_current_prepare_queued import commit_exact
    queue_done(queue_state, queue_id, 'fold-a-current-campaign')
    result = register(repo, source, data, admitted, completed)
    commit_exact(repo, result['files'], 'chore: register complete current A0 Fold A diagnostics')
    run(repo, data, 'columns')
    return run(repo, data, 'read')


def main():
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    assert_execution_environment()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('register', 'columns', 'read', 'queued'))
    for key in ('repo-root', 'data-root'):
        p.add_argument('--'+key, type=Path, required=True)
    for key in ('source-repo', 'admitted', 'completed', 'queue-state'):
        p.add_argument('--'+key, type=Path)
    p.add_argument('--queue-id')
    a = p.parse_args()
    if a.stage in ('register', 'queued'):
        if any(x is None for x in (a.source_repo, a.admitted, a.completed)):
            p.error('registration requires source-repo, admitted and completed')
        if a.stage == 'queued':
            if a.queue_state is None or a.queue_id is None:
                p.error('queued requires queue-state and queue-id')
            result = queued(a.repo_root, a.source_repo, a.data_root, a.admitted, a.completed, a.queue_state, a.queue_id)
        else:
            result = register(a.repo_root, a.source_repo, a.data_root, a.admitted, a.completed)
    else:
        result = run(a.repo_root, a.data_root, a.stage)
    print(json.dumps(result))


if __name__ == '__main__': main()

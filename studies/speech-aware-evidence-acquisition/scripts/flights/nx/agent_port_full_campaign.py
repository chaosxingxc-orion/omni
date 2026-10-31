"""Run one admitted full Agent campaign and its registered diagnostics, detached.

Every step has an exclusive intent and terminal result. On failure or interruption,
inspect the existing attempt. A registered checkpoint successor uses a fresh
attempt and coordinator directory; an existing attempt is never retried.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
for directory in (REPO / 'src', REPO / 'scripts/analysis', Path(__file__).parent):
    sys.path.insert(0, str(directory))
import agent_port_full_registration as registration
import agent_port_score_evidence as delivered
import agent_port_column_read as columns
import agent_port_fit_evidence as fit
from speech_aware_evidence_acquisition.core.atomic_file import publish_new
from speech_aware_evidence_acquisition.contracts import assert_execution_environment


def require_fresh_generation(repo, data, cell):
    """Allow only pinned ABORTED predecessors, never reuse the successor."""
    from speech_aware_evidence_acquisition.core.flight.campaign import CampaignCell
    from speech_aware_evidence_acquisition.core.flight.resume import resume_declarations_from_document
    from speech_aware_evidence_acquisition.core.row_checkpoint import checkpoint_path
    CampaignCell.from_document(cell)
    run, attempt = cell['run_id'], cell['attempt_id']
    if cell.get('resume_from') is None:
        if any((data / kind / run).exists() for kind in ('runs', 'attempts')):
            raise ValueError('Agent attempt already started; never replay')
        return []
    declarations = resume_declarations_from_document(cell['resume_from'])
    if any(d.mode != 'checkpoint' for d in declarations):
        raise ValueError('full Agent coordinator requires checkpoint predecessors')
    allowed = {d.attempt_id for d in declarations}
    for kind in ('runs', 'attempts'):
        root = data / kind / run
        present = {p.name if kind == 'runs' else p.stem for p in root.iterdir()} if root.exists() else set()
        if present - allowed:
            raise ValueError('undeclared or already started successor attempt; never replay')
    result = []
    for declaration in declarations:
        record = registration.read(data / 'attempts' / run / (declaration.attempt_id + '.json'))
        if (record.get('state') != 'ABORTED' or record.get('run_id') != run
                or record.get('attempt_id') != declaration.attempt_id):
            raise ValueError('checkpoint predecessor must be the named ABORTED attempt')
        path = checkpoint_path(data / 'runs' / run / declaration.attempt_id, run)
        if registration.sha(path) != declaration.checkpoint_sha256:
            raise ValueError('checkpoint predecessor pin differs')
        result.append({'attempt_id':declaration.attempt_id, 'checkpoint_sha256':declaration.checkpoint_sha256,
                       'actual':record['actual']})
    return result


def validate(repo, data):
    spec = registration.read(repo / registration.PKG / 'successor-draft-v1.json')
    exp = registration.read(repo / 'configs/experiment' / (spec['cell'] + '.json'))
    cell, = exp['cells']
    base = registration.read(repo / 'configs/baseline' / (spec['baseline'] + '.json'))
    if (cell['readiness']['state'] != 'ready' or 'campaign' not in cell['allowed_execution_stages']
            or exp['experiment_id'] != spec['run_id'] or base.get('policy') != 'phase-native-module-v1-c17'
            or exp['family_budget']['official_complete_scoring_jobs'] != 1
            or (cell['run_id'], cell['attempt_id']) != (spec['run_id'], spec['attempt_id'])
            or (base['native_contact_fit']['run_id'], base['native_contact_fit']['attempt_id']) != (spec['fit_run_id'], spec['fit_attempt_id'])):
        raise ValueError('ready full c17 campaign identity required')
    ids = exp['populations'][0]['ids']
    if len(ids) != 1080 or len(set(ids)) != 1080 or exp['population']['count'] != 1080:
        raise ValueError('complete unique Fold B roster required')
    predecessors = require_fresh_generation(repo, data, cell)
    if any((data / directory / spec['fit_run_id']).exists() for directory in ('runs', 'attempts')):
        raise ValueError('Agent FIT attempt already started; never replay')
    # A prepared phase-supply is intentionally published under the generation
    # run id during the model-free preparation step.  The FIT run is consumed
    # only when its own attempt directory or a request-fit artifact exists.
    # Treating every ``prepared/<fit_run_id>`` directory as spent rejects a
    # valid, freshly prepared campaign before it can make its first contact.
    fit_root = data / 'prepared' / spec['fit_run_id']
    fit_attempt_root = fit_root / spec['fit_attempt_id']
    fit_started = fit_attempt_root.exists() or any(fit_root.glob('*.request-fit.json'))
    if fit_started:
        raise ValueError('Agent FIT already started; never replay')
    pin = base['prepared_phase_supply']
    supply = delivered.shared.within(data / pin['relpath'], data)
    if registration.sha(supply) != pin['sha256']:
        raise ValueError('prepared Agent supply pin differs')
    if registration.read(supply)['output'] != registration.read(data / spec['supply_reference'])['output']:
        raise ValueError('Agent supply differs from frozen predecessor')
    recipe = repo / registration.PKG / 'analysis-recipe-v1.json'
    delivered.require_read(repo, recipe, spec['read_run_id'], spec['cell'])
    columns.require_recipe(repo)
    draft = registration.read(repo / registration.PKG / 'root-admission-draft-v1.json')
    binding = draft['budget']['accelerator_bound']['measurement']
    registration.validate_binding(binding, repo, data)
    registration.accepted_proof(repo, data, binding)
    if exp['service_session'] != binding['session_id']:
        raise ValueError('Agent and accepted proof use different deployments')
    return {'spec': spec, 'session': exp['service_session'], 'recipe': recipe,
            'supply': supply, 'rows': len(ids), 'predecessors': predecessors}


def commands(repo, data, runtime, lock, plan, gold_intents=None, n1_json=None):
    spec = plan['spec']
    common = ['--repo-root', str(repo), '--data-root', str(data), '--runtime-root', str(runtime), '--lock-path', str(lock),
              '--experiment', spec['cell'], '--cell', spec['cell']]
    cli = [sys.executable, '-m', 'speech_aware_evidence_acquisition.core.flight.cli']
    checks = [('registration', [sys.executable, 'scripts/checks/flight_registration_check.py',
                               '--repo-root', str(repo), '--data-root', str(data), '--cell', spec['cell'], '--json']),
              ('preflight', cli + ['check', '--read-only', *common])]
    campaign = cli + ['campaign', *common, '--owner', 'root-agent-full-fold-b', '--expected-session-id', plan['session']]
    diagnostics = []
    for stage in ('columns', 'read'):
        args = [sys.executable, 'scripts/analysis/agent_port_column_read.py', '--repo-root', str(repo), '--data-root', str(data), '--stage', stage]
        for key, path in [('gold-intents', gold_intents), ('n1-json', n1_json)]:
            if path is not None: args += ['--' + key, str(path)]
        diagnostics.append((stage, args))
    diagnostics.append(('repeat-custody', [sys.executable, 'scripts/analysis/agent_port_repeat_evidence.py',
        '--repo-root', str(repo), '--data-root', str(data)]))
    return checks, campaign, diagnostics


def execute(repo, data, runtime, lock, out, gold_intents=None, n1_json=None):
    assert_execution_environment()
    repo, data, out = repo.resolve(), data.resolve(), out.resolve()
    if out.exists() or not out.is_relative_to(data) or out.is_relative_to(repo):
        raise ValueError('fresh external coordinator directory required')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo).strip():
        raise ValueError('clean committed study required')
    if (data / 'locks/flight.lock').exists():
        raise ValueError('generation custody already held; wait without replacing it')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    plan = validate(repo, data)
    checks, campaign, diagnostics = commands(repo, data, runtime, lock, plan, gold_intents, n1_json)
    spec = plan['spec']
    pins = {str(p): registration.sha(p) for p in [plan['recipe'], plan['supply'], repo / registration.BRANCH_RECIPE]}
    for path in (gold_intents, n1_json):
        if path is not None: pins[str(path)] = registration.sha(path)
    out.mkdir(parents=True, exist_ok=False)
    def publish(name, obj): publish_new(out / name, registration.raw(obj))
    publish('started.json', {'pid': os.getpid(), 'commit': head, 'time': datetime.now(timezone.utc).isoformat(),
                            'inputs': pins, 'generation_run_id': spec['run_id'], 'rows': 1080,
                            'campaign_family': registration.FAMILY.name})
    def run(name, argv, read_id):
        publish(name + '.intent.json', {'argv': argv, 'commit': head, 'read_id': read_id})
        with (out / (name + '.log')).open('x') as stream:
            result = subprocess.run(argv, cwd=repo, stdout=stream, stderr=subprocess.STDOUT,
                                    env={**os.environ, 'SAEA_ROLLOUT_TRACE_READ_ID': read_id})
        publish(name + '.exit.json', {'exit': result.returncode, 'time': datetime.now(timezone.utc).isoformat()})
        if result.returncode:
            raise RuntimeError(name + ' failed; preserve partial effects, never retry coordinator')
    try:
        for name, args in checks: run(name, args, spec['read_run_id'])
        # Runtime preflight revalidates every predecessor and refuses any used
        # attempt. The read-only check may have written its resume declaration.
        if not plan["predecessors"]:
            validate(repo, data)
        elif (data / "attempts" / spec["run_id"] / (spec["attempt_id"] + ".json")).exists():
            raise ValueError("successor attempt was spent during preflight")
        if (subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo).strip()
                or subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip() != head
                or any(registration.sha(Path(p)) != h for p, h in pins.items())):
            raise ValueError('code or pinned input changed during preflight')
        run('campaign', campaign, spec['read_run_id'])
        publish('native-fit-evidence.json', fit.build(repo, data, spec['cell'], spec['read_run_id']))
        final = delivered.build(repo, data, spec['cell'], spec['read_run_id'], plan['recipe'])
        publish('final-delivered-score-evidence.json', final)
        for name, args in diagnostics: run(name, args, registration.BRANCH_READ)
        publish('complete.json', {'commit': head, 'run_id': spec['run_id'], 'rows': final['rows'],
            'aggregate': final['aggregate'], 'score_surface': 'sealed-final-delivery',
            'secondary_complete_columns': 6, 'claims_system_benefit': False,
            'diagnostic_input_availability': {'gold_intents': gold_intents is not None, 'n1': n1_json is not None}})
    except Exception as error:
        publish('failed.json', {'error': str(error), 'automatic_retry': False})
        raise
    return {'out': str(out), 'rows': 1080, 'score_surface': 'sealed-final-delivery', 'secondary_columns': 6}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=REPO)
    for key in ('data-root', 'runtime-root', 'lock-path', 'out'):
        parser.add_argument('--' + key, type=Path, required=True)
    for key in ('gold-intents', 'n1-json'): parser.add_argument('--' + key, type=Path)
    args = parser.parse_args()
    print(json.dumps(execute(args.repo_root, args.data_root, args.runtime_root, args.lock_path, args.out, args.gold_intents, args.n1_json)))


if __name__ == '__main__': main()

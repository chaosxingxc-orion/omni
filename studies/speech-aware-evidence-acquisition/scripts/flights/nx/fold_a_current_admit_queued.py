"""Chain committed Fold A preparation into admission, supply and queued flight.

The preparation worktree remains the sole owner of generated configurations.
This driver runs its committed tools as subprocesses and preserves every stage.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fold_a_current_prepare_queued as preparation
import fold_a_current_registration as registration
from speech_aware_evidence_acquisition.core.atomic_file import publish_new

v = registration.evaluation
read, sha, raw = registration.read, registration.sha, registration.raw


def queue_done(queue, queue_id, job):
    state = read(queue)
    item = state.get('jobs', {}).get(job, {})
    if (state.get('schema') != 'saea-card-queue-state-v1' or state.get('queue_id') != queue_id
            or item.get('status') != 'done' or item.get('exit') != 0):
        raise ValueError('successful upstream job required: '+job)


def changed(repo):
    paths = set()
    for args in (['diff', '--name-only', '-z', 'HEAD'], ['ls-files', '--others', '--exclude-standard', '-z']):
        paths.update(subprocess.check_output(['git', *args], cwd=repo).decode().split('\0'))
    paths.discard('')
    return paths


def commit_package(repo, *, stage, expected=None):
    paths = changed(repo)
    allowed = {f'configs/{kind}/{v.CELL}.json' for kind in ('baseline', 'experiment')}
    allowed.update({'docs/exposure-ledger.md', registration.PKG+'-preregistration.md'})
    allowed.update(str(p.relative_to(repo)) for p in (repo/registration.PKG).glob('*.json'))
    if not paths or not paths <= allowed or (expected is not None and paths != set(expected)):
        raise ValueError('unexpected changes during '+stage)
    if expected is not None:
        for name, digest in expected.items():
            if sha(repo/name) != digest:
                raise ValueError('registered output bytes changed')
    return preparation.commit_exact(repo, paths, 'chore: '+stage+' current A0 Fold A')


def admit(repo, source, data, lock, binding, prepared, queue, queue_id, base_commit, out):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    assert_execution_environment()
    repo, source, data = (p.resolve() for p in (repo, source, data))
    out = v.comparator.inputs.confined(data, str(out))
    if out.exists() or out.is_relative_to(repo) or out.is_relative_to(source):
        raise ValueError('fresh external admission output required')
    queue_done(queue, queue_id, 'fold-a-current-prepare')
    done = read(prepared)
    if done['rows'] != 1080 or done['model_contacts'] != 0 or done['launchable'] is not False:
        raise ValueError('complete held preparation required')
    head = preparation.clean(repo, done['source_commit'])
    subprocess.run(['git', 'merge-base', '--is-ancestor', base_commit, head], cwd=repo, check=True)
    # Preparation may add only its exact configurations and derivation records.
    allowed = {f'configs/{kind}/{cell}.json' for cell in (v.CELL, v.comparator.CELL)
               for kind in ('baseline', 'experiment')}
    allowed.update({v.PACKAGE+'/derivation.json', v.comparator.PKG+'/derivation.json'})
    delta = subprocess.check_output(['git', 'diff', '--name-only', base_commit, head], cwd=repo, text=True).splitlines()
    if set(delta) != allowed:
        raise ValueError('prepared source differs beyond admitted derivation')
    preparation.clean(source)
    if (data/'locks/flight.lock').exists():
        raise ValueError('generation lease still held')
    pins = {str(p): sha(p) for p in (prepared, lock, binding)}
    out.mkdir(parents=True, exist_ok=False)
    def publish(name, value): publish_new(out/name, raw(value))
    publish('intent.json', {'source_commit': head, 'base_commit': base_commit, 'inputs': pins,
        'run_id': v.RUN, 'model_contacts': 0})
    def stage(name, argv):
        preparation.clean(repo, head)
        if any(sha(Path(p)) != digest for p, digest in pins.items()):
            raise ValueError('admission input changed')
        publish(name+'.intent.json', {'argv': argv, 'commit': head})
        with (out/(name+'.log')).open('x') as log:
            result = subprocess.run(argv, cwd=repo, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, 'PYTHONPATH': str(repo/'src')})
        publish(name+'.exit.json', {'exit': result.returncode})
        if result.returncode:
            raise ValueError(name+' failed; preserve evidence without replay')
    def command(tool, *args):
        return [sys.executable, 'scripts/flights/nx/fold_a_current_'+tool+'.py', '--repo-root', str(repo),
                '--data-root', str(data), *map(str, args)]
    try:
        stage('registration', command('registration', '--source-repo', source, '--binding', binding,
                                      '--archive', out/'registration'))
        expected = read(out/'registration/intent.json')['outputs']
        head = commit_package(repo, stage='register', expected=expected)
        publish('registration.commit.json', {'commit': head})
        stage('mint', command('mint'))
        decision = data/v.DECISION_PATH
        pins[str(decision)] = sha(decision)
        stage('supply', command('supply', '--lock-path', lock, '--out', out/'supply'))
        head = commit_package(repo, stage='prepare supply for')
        if any(sha(Path(p)) != digest for p, digest in pins.items()):
            raise ValueError('admission input changed before completion')
        publish('complete.json', {'source_commit': head, 'run_id': v.RUN, 'rows': 1080, 'inputs': pins,
            'model_contacts': 0, 'launchable': False, 'remaining': ['registration check', 'preflight', 'campaign']})
        return {'source_commit': head, 'model_contacts': 0}
    except BaseException as error:
        publish('failed.json', {'reason': str(error), 'automatic_retry': False})
        raise


def dispatch(repo, data, lock, admitted, queue, queue_id, out):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    assert_execution_environment()
    queue_done(queue, queue_id, 'fold-a-current-admit')
    done = read(admitted)
    if done['run_id'] != v.RUN or done['rows'] != 1080 or done['model_contacts'] != 0:
        raise ValueError('complete independent admission required')
    preparation.clean(repo, done['source_commit'])
    if str(lock) not in done['inputs'] or any(sha(Path(p)) != digest for p, digest in done['inputs'].items()):
        raise ValueError('admitted inputs changed')
    command = [sys.executable, 'scripts/flights/nx/fold_a_current_campaign.py', '--repo-root', str(repo),
        '--data-root', str(data), '--lock-path', str(lock), '--out', str(out), '--source-commit', done['source_commit']]
    subprocess.run(command, cwd=repo, check=True,
        env={**os.environ, 'PYTHONPATH': str(repo/'src')})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['admit', 'campaign'])
    for n in ('repo-root', 'data-root', 'lock-path', 'queue-state', 'out'):
        p.add_argument('--'+n, type=Path, required=True)
    for n in ('source-repo', 'binding', 'prepared', 'admitted'):
        p.add_argument('--'+n, type=Path)
    p.add_argument('--queue-id', required=True); p.add_argument('--base-commit')
    a = p.parse_args()
    if a.stage == 'admit':
        if any(x is None for x in (a.source_repo, a.binding, a.prepared, a.base_commit)):
            p.error('admit requires source-repo, binding, prepared and base-commit')
        print(json.dumps(admit(a.repo_root, a.source_repo, a.data_root, a.lock_path, a.binding,
            a.prepared, a.queue_state, a.queue_id, a.base_commit, a.out)))
    else:
        if a.admitted is None: p.error('campaign requires admitted')
        dispatch(a.repo_root, a.data_root, a.lock_path, a.admitted, a.queue_state, a.queue_id, a.out)


if __name__ == '__main__': main()

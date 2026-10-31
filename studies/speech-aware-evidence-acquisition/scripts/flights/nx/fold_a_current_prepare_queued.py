"""Import the verified comparator into an isolated tree and prepare held Fold A.

Run detached after the comparator queue job succeeds. This commits only exact
verified preparation files in its owning worktree; it never registers or flies.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fold_a_current_evaluation as evaluation
from speech_aware_evidence_acquisition.core.atomic_file import publish_new

c = evaluation.comparator


def git(repo, *args):
    return subprocess.check_output(['git', *args], cwd=repo, text=True).strip()


def clean(repo, expected=None):
    if git(repo, 'status', '--porcelain'):
        raise ValueError('clean preparation/source checkout required')
    head = git(repo, 'rev-parse', 'HEAD')
    if expected is not None and head != expected:
        raise ValueError('preparation commit changed')
    return head


def commit_exact(repo, paths, message):
    """Refuse any unrelated tracked, staged or untracked change."""
    actual = set(subprocess.check_output(
        ['git', 'diff', '--name-only', '-z', 'HEAD'], cwd=repo).decode().split('\0'))
    actual.update(subprocess.check_output(
        ['git', 'ls-files', '--others', '--exclude-standard', '-z'], cwd=repo).decode().split('\0'))
    actual.discard('')
    if actual != set(paths):
        raise ValueError('unexpected changes before preparation commit')
    subprocess.run(['git', 'add', '--', *sorted(paths)], cwd=repo, check=True)
    subprocess.run(['git', 'commit', '-m', message, '--', *sorted(paths)], cwd=repo, check=True)
    return clean(repo)


def verified_comparator(repo, source, producer, data, binding, completion, producer_commit):
    clean(source, binding['source_commit'])
    if git(producer, 'rev-parse', 'HEAD') != producer_commit:
        raise ValueError('comparator producer commit changed')
    queue = c.read(c.inputs.confined(data, binding['queue_state']))
    item = queue.get('jobs', {}).get('fold-a-comparator', {})
    if item.get('status') != 'done' or item.get('exit') != 0:
        raise ValueError('successful comparator queue job required')
    spec_path, source_done, retrieval_done = c.bound_parents(source, data, binding)
    record = c.read(completion)
    spec, done, retrieval = map(c.read, (spec_path, source_done, retrieval_done))
    if (record.get('parent_binding') != binding or record['source_commit'] != binding['source_commit']
            or record['source_run'] != binding['source_run']
            or record['agent_completion_sha256'] != c.sha(source_done)
            or record['retrieval_completion_sha256'] != c.sha(retrieval_done)
            or done['rows'] != 1080 or done['score_surface'] != 'sealed-final-delivery'
            or record['allowed_execution_stages'] != ['check'] or record['model_contacts'] != 0
            or record['claims_system_lock'] is not False
            or retrieval['rows'] != 1080 or retrieval['values_unchanged'] is not True):
        raise ValueError('comparator provenance differs')
    for name, digest in retrieval['pins'].items():
        if c.sha(c.inputs.confined(data, name)) != digest:
            raise ValueError('retrieval artifact changed')
    base_name = f"configs/baseline/{spec['baseline']}.json"
    exp_name = f"configs/experiment/{spec['cell']}.json"
    base, exp = c.read(source/base_name), c.read(source/exp_name)
    model = 'configs/model/' + exp['cells'][0]['model_fragment'] + '.json'
    if (record['source_files'] != {n: c.sha(source/n) for n in (base_name, exp_name, model)}
            or c.sha(repo/model) != c.sha(source/model)):
        raise ValueError('actual A0 source/model pins differ')
    rebuilt = c.derive(repo, base, exp, retrieval['pins'])
    outputs = {f'configs/{kind}/{c.CELL}.json': c.raw(value)
               for kind, value in zip(('baseline', 'experiment'), rebuilt)}
    if record['output_hashes'] != {n: hashlib.sha256(b).hexdigest() for n, b in outputs.items()}:
        raise ValueError('comparator does not reproduce actual A0 transform')
    outputs[c.PKG+'/derivation.json'] = c.raw(record)
    for name, blob in outputs.items():
        if c.inputs.confined(producer, name).read_bytes() != blob:
            raise ValueError('produced comparator bytes differ')
        if (repo/name).exists():
            raise ValueError('destination comparator already exists')
    return outputs


def execute(repo, source, producer, data, lock, binding_path, completion, out,
            source_commit, producer_commit):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    assert_execution_environment()
    repo, source, producer, data = [p.resolve() for p in (repo, source, producer, data)]
    if len({repo, source, producer}) != 3:
        raise ValueError('separate owning preparation tree required')
    clean(repo, source_commit)
    completion = c.inputs.confined(data, str(completion))
    out = c.inputs.confined(data, str(out))
    if out.exists() or any(out.is_relative_to(p) for p in (repo, source, producer)):
        raise ValueError('fresh external output required')
    binding = c.read(binding_path)
    outputs = verified_comparator(repo, source, producer, data, binding, completion, producer_commit)
    pins = {str(p): c.sha(p) for p in (binding_path, completion, lock)}
    out.mkdir(parents=True, exist_ok=False)
    publish_new(out/'intent.json', c.raw({'source_commit': source_commit,
        'producer_commit': producer_commit, 'inputs': pins, 'model_contacts': 0}))
    try:
        for name, blob in outputs.items():
            publish_new(repo/name, blob); (repo/name).chmod(0o644)
        imported = commit_exact(repo, outputs, 'chore: import verified current A0 Fold A comparator')
        evaluation.agreement.prepare(repo, data, lock, completion, out/'agreement', imported)
        if any(c.sha(Path(p)) != digest for p, digest in pins.items()):
            raise ValueError('preparation parent changed')
        clean(repo, imported)
        evaluation.write(repo, source, data, binding, completion, out/'agreement/complete.json', out/'evaluation')
        paths = [f'configs/{kind}/{evaluation.CELL}.json' for kind in ('baseline', 'experiment')]
        paths.append(evaluation.PACKAGE+'/derivation.json')
        head = commit_exact(repo, paths, 'chore: derive held current A0 Fold A evaluation')
        if any(c.sha(Path(p)) != digest for p, digest in pins.items()):
            raise ValueError('preparation parent changed before completion')
        result = {'source_commit': head, 'rows': 1080, 'model_contacts': 0,
                  'launchable': False, 'remaining': ['registration', 'mint', 'own supply', 'preflight', 'campaign']}
        publish_new(out/'complete.json', c.raw(result))
        return result
    except BaseException as error:
        publish_new(out/'failed.json', c.raw({'reason': str(error), 'automatic_retry': False}))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('repo-root', 'source-repo', 'producer-repo', 'data-root', 'lock-path',
                 'parent-binding', 'comparator-complete', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--source-commit', required=True)
    p.add_argument('--producer-commit', required=True)
    a = p.parse_args()
    print(json.dumps(execute(a.repo_root, a.source_repo, a.producer_repo, a.data_root,
        a.lock_path, a.parent_binding, a.comparator_complete, a.out, a.source_commit, a.producer_commit)))


if __name__ == '__main__': main()

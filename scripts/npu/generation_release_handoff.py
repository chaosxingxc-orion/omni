"""Verify a sealed generation release; wait on Linux file events, not model polls."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import select
import math


ARTIFACT_RELEASE_SCHEMA = 'saea-artifact-generation-released-v1'


def release_paths(document, *, data, expected):
    """Keep task scoring inputs and finalized proposal artifacts distinct."""
    root = Path(data) / 'runs' / expected['run_id'] / expected['attempt_id']
    if document.get('schema') == 'saea-generation-released-v1':
        return {key: root/name for key, name in {
            'generation_manifest': expected['run_id']+'.generation-manifest.json',
            'score_ledger_cells': 'score-ledger-cells.json', 'score_jobs': 'score-jobs.jsonl'}.items()}
    if document.get('schema') != ARTIFACT_RELEASE_SCHEMA:
        raise ValueError('unsupported generation release schema')
    seal = document.get('seal_receipt', {})
    final = seal.get('finalize', {})
    if (final.get('status') != 'not-applicable' or final.get('result_kind') != 'artifact'
            or final.get('attempt_state') != 'FINALIZED' or final.get('scores_path') is not None
            or seal.get('outputs_row_count') != expected['rows']):
        raise ValueError('artifact release needs finalized scoreless proposal seal')
    paths = {'manifest': root/(expected['run_id']+'.run-manifest.json'),
        'outputs': root/(expected['run_id']+'.outputs.jsonl'),
        'attempt': Path(data)/'attempts'/expected['run_id']/(expected['attempt_id']+'.json')}
    if seal.get('phases_path') is not None:
        paths['phases'] = root/(expected['run_id']+'.phases.json')
    return paths


def verify_release(path, *, data, expected, resolve_artifact=None):
    """Verify original release bytes, optionally resolving container paths to host files.

    The resolver only selects where to read; logical pins and seal paths remain
    unchanged and must still name the exact run/attempt files below data/runs.
    """
    return verify_release_document(json.loads(path.read_bytes()), data=data,
        expected=expected, resolve_artifact=resolve_artifact)


def verify_release_document(document, *, data, expected, resolve_artifact=None):
    """Verify an unpublished event with the same rules as its durable form."""
    schema = document.get('schema')
    logical_paths = release_paths(document, data=data, expected=expected)
    required = {"schema": schema, "generation_sealed": True,
                "generation_lease_released": True, "scoring_complete": False,
                "consumer_requires_own_lease_and_readiness": True, **expected}
    if any(type(document.get(k)) is not type(v) or document[k] != v for k, v in required.items()):
        raise ValueError("generation release identity or state differs")
    if any(type(document.get(key)) not in (int, float) or not math.isfinite(document[key])
           or document[key] < 0 for key in ('released_at_unix', 'campaign_seconds_to_release')):
        raise ValueError("generation release timing differs")
    data = Path(data)
    root = data / "runs" / expected["run_id"] / expected["attempt_id"]
    if resolve_artifact is None:
        root = root.resolve()
    elif not callable(resolve_artifact) or not data.is_absolute() or '..' in root.parts:
        raise ValueError("generation release resolver or logical root differs")
    if not root.is_relative_to((data.resolve() if resolve_artifact is None else data) / "runs"):
        raise ValueError("generation release escapes owned runs")
    if set(document.get("artifacts", {})) != set(logical_paths):
        raise ValueError("generation release needs all sealed inputs")
    seal = document["seal_receipt"]
    physical_paths = {}
    for key, logical in logical_paths.items():
        pin = document["artifacts"][key]
        if not isinstance(pin, dict) or set(pin) != {'path', 'sha256'}:
            raise ValueError('exact sealed artifact pin required')
        target = Path(pin["path"])
        if not target.is_absolute() or '..' in target.parts:
            raise ValueError('canonical sealed artifact path required')
        want = logical.resolve() if resolve_artifact is None else logical
        if resolve_artifact is None and not want.is_relative_to(data.resolve()):
            raise ValueError('generation release artifact escapes data root')
        if ((resolve_artifact is None and (target.is_symlink() or target.resolve() != want))
                or (resolve_artifact is not None and str(target) != pin['path'])
                or (resolve_artifact is not None and target != want)):
            raise ValueError("generation release input path differs")
        if schema != ARTIFACT_RELEASE_SCHEMA or key in ('manifest', 'phases'):
            if seal.get(key + "_path") != str(target) or seal.get(key + "_sha256") != pin["sha256"]:
                raise ValueError("generation release seal differs")
        elif key == 'outputs' and seal.get('outputs_path') != str(target):
            raise ValueError('artifact output seal path differs')
        physical = target if resolve_artifact is None else Path(resolve_artifact(target))
        if resolve_artifact is not None and (not physical.is_absolute() or physical.is_symlink()
                or physical.resolve() != physical or not physical.is_file()):
            raise ValueError("generation release mapped input path differs")
        with physical.open("rb") as handle:
            digestor = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digestor.update(chunk)
            digest = digestor.hexdigest()
        if digest != pin["sha256"]:
            raise ValueError("generation release sealed input changed")
        physical_paths[key] = physical
    if schema == ARTIFACT_RELEASE_SCHEMA:
        manifest = json.loads(physical_paths['manifest'].read_bytes())
        attempt = json.loads(physical_paths['attempt'].read_bytes())
        if (manifest.get('result_kind') != 'artifact' or manifest.get('status') != 'COMPLETED'
                or manifest.get('run_id') != expected['run_id']
                or manifest.get('study_commit') != expected['study_commit']
                or manifest.get('plan', {}).get('run_id') != expected['run_id']
                or manifest.get('attempt', {}).get('run_id') != expected['run_id']
                or manifest.get('attempt', {}).get('attempt_id') != expected['attempt_id']
                or manifest.get('scorers') != [] or manifest.get('scores') is not None
                or manifest.get('plan', {}).get('attempt_id') != expected['attempt_id']
                or manifest.get('attempt', {}).get('state_path') != str(logical_paths['attempt'])
                or manifest.get('outputs') != document['artifacts']['outputs']
                or any(attempt.get(k) != v for k, v in {
                    'run_id': expected['run_id'], 'attempt_id': expected['attempt_id'], 'state': 'FINALIZED'}.items())):
            raise ValueError('artifact release finalized source binding differs')
        with physical_paths['outputs'].open() as handle:
            rows = [json.loads(line) for line in handle]
        if (len(rows) != expected['rows']
                or any(not isinstance(row, dict) or row.get('result_kind') != 'artifact'
                       or not isinstance(row.get('sample_id'), str) or not row['sample_id'] for row in rows)
                or len({row['sample_id'] for row in rows}) != len(rows)):
            raise ValueError('artifact release row coverage differs')
    return document


class DirectoryEvents:
    """Watch the existing queue directory before checking for a new run folder.

    A timeout checks the SAME child every600s; it never restarts a job. Kernel
    notifications wake immediately when the producer publishes the release.
    """
    def __init__(self, directory):
        self.lib = ctypes.CDLL(None, use_errno=True)
        self.fd = self.lib.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
        if self.fd < 0:
            raise OSError(ctypes.get_errno(), "inotify_init1")
        try:
            self.add(directory)
        except BaseException:
            self.close()
            raise

    def add(self, directory):
        # IN_CLOSE_WRITE | IN_MOVED_TO | IN_CREATE
        if self.lib.inotify_add_watch(self.fd, os.fsencode(directory), 0x0188) < 0:
            raise OSError(ctypes.get_errno(), "inotify_add_watch")

    def wait(self, timeout=600):
        if select.select([self.fd], [], [], timeout)[0]:
            while True:
                try:
                    os.read(self.fd, 65536)
                except BlockingIOError:
                    break
            return True
        return False

    def close(self):
        os.close(self.fd)


def wait_release(child, path, *, data, expected, events_factory=DirectoryEvents):
    # The producer creates path.parent; its existing parent is watched first.
    events = events_factory(path.parent.parent)
    watched = False
    try:
        while True:
            if not watched and path.parent.exists():
                events.add(path.parent)
                watched = True
            if path.exists():
                return verify_release(path, data=data, expected=expected)
            code = child.poll()
            if code is not None:
                # Close the race with publication immediately before process exit.
                if path.exists():
                    return verify_release(path, data=data, expected=expected)
                raise RuntimeError("generation exited without release; preserve checkpoints: " + str(code))
            events.wait(600)
    finally:
        events.close()

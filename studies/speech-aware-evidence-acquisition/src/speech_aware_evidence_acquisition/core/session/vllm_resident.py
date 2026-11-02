"""Observe and register externally deployed vLLM servers without spawning or stopping them."""
from __future__ import annotations

import hashlib
import fcntl
import json
from dataclasses import replace
from pathlib import Path
from typing import Mapping

from ...contracts import STUDY_SLUG
from ...contracts.gate_runtime_vllm import (
    instance_argv, instance_for_endpoint, validate_vllm_receipt_document,
    verify_vllm_runtime_files, verify_model_directory,
)
from ..atomic_file import publish_new
from ..config import fragment_hash, runtime_receipt_relpath
from ..data_root_placeholder import collapse_data_root
from . import SessionError
from .receipt import SESSION_RECEIPT_SCHEMA, SessionReceipt
from .serving import build_session_document, find_resident_session, serving_identity, serving_session_dir
from .vllm_identity import (
    assert_extra_argv_matches_instance, interpreter_identity, pinned_model_files,
    verify_vllm_process_identity, verify_vllm_session,
)


def required_identity(*, fragment, model_fragment, runtime_receipt, receipt_path, host, port):
    """The fragment's protocol and the receipt's endpoint form one serving identity."""
    instance = instance_for_endpoint(runtime_receipt, host, port)
    assert_extra_argv_matches_instance(fragment['server_args'], runtime_receipt, instance)
    binary, sha, _ = interpreter_identity(runtime_receipt)
    return serving_identity(
        model_fragment=model_fragment,
        model_fragment_sha256=fragment_hash(collapse_data_root(fragment, fragment.get('model_materialization_root'))),
        runtime_receipt_relpath=runtime_receipt_relpath(fragment),
        runtime_receipt_sha256=hashlib.sha256(Path(receipt_path).read_bytes()).hexdigest(),
        binary_path=binary.lstrip('/'), binary_sha256=sha,
        model_files=[{'path': f.path, 'sha256': f.sha256} for f in pinned_model_files(runtime_receipt)],
        session_role='legacy', model_root=fragment.get('model_materialization_root'),
        server_argv=instance_argv(runtime_receipt, instance), server_env=fragment.get('server_env', {}),
        host=host, port=port, props=None,
    )


def _bytes(document):
    return (json.dumps(document, indent=2, sort_keys=True) + '\n').encode()


def register_deployment(*, data_root: Path, fragment: Mapping, model_fragment: str,
                        receipt_path: Path, instances: Mapping[str, Mapping], session_id: str,
                        proc_root: Path = Path('/proc')) -> list[Path]:
    """Serialize external registrations so two coordinators cannot claim one root."""
    with (Path(data_root)/'.vllm-resident-registration.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _register_deployment(data_root=data_root, fragment=fragment,
            model_fragment=model_fragment, receipt_path=receipt_path, instances=instances,
            session_id=session_id, proc_root=proc_root)


def _register_deployment(*, data_root: Path, fragment: Mapping, model_fragment: str,
                        receipt_path: Path, instances: Mapping[str, Mapping], session_id: str,
                        proc_root: Path) -> list[Path]:
    """Register the entire verified set; publish the coordinator last as its commit marker.

    Each instance observation carries pid, started_at and startup_stdout_sha256 from
    the external launcher's process/container record. Existing identical documents are
    idempotent; conflicting or stopped sessions require explicit reconciliation.
    """
    receipt_path = Path(receipt_path)
    receipt = json.loads(receipt_path.read_bytes())
    validate_vllm_receipt_document(receipt)
    verify_vllm_runtime_files(receipt)
    model_root = Path(fragment['model_materialization_root'])
    model_dir = Path(receipt['model']['model_dir'])
    verify_model_directory(receipt, model_root, str(model_dir.relative_to(model_root)))
    declared = receipt['deployment']['instances']
    if set(instances) != {i['instance_id'] for i in declared}:
        raise SessionError('observations must name every receipt instance exactly once')
    if len({o['pid'] for o in instances.values()}) != len(instances):
        raise SessionError('deployment instances must have distinct process IDs')
    if session_id in instances:
        raise SessionError('coordinator session ID must differ from instance IDs')
    coordinator = serving_session_dir(data_root, session_id)
    resident = find_resident_session(data_root)
    if resident is not None and resident[1]['session_id'] != session_id:
        raise SessionError('another coordinator is already registered as resident')
    pending = []
    first = None
    for instance in declared:
        name = instance['instance_id']
        observation = instances[name]
        host, port = instance['host'], instance['port']
        identity = required_identity(fragment=fragment, model_fragment=model_fragment,
                                     runtime_receipt=receipt, receipt_path=receipt_path, host=host, port=port)
        argv = verify_vllm_process_identity(observation['pid'], receipt, instance,
                                           host=host, port=port, proc_root=proc_root)
        session = SessionReceipt(
            schema=SESSION_RECEIPT_SCHEMA, study=STUDY_SLUG, session_id=name,
            binary_path=identity['binary_path'], binary_sha256=identity['binary_sha256'],
            argv=tuple(argv), pid=observation['pid'], host=host, port=port,
            model_files=pinned_model_files(receipt), started_at=observation['started_at'],
            startup_stdout_sha256=observation['startup_stdout_sha256'],
        ).validate()
        verify_vllm_session(session, receipt)
        directory = serving_session_dir(data_root, name)
        geometry = {'slots': instance['max_num_seqs'],
                    'slot_tokens': int(receipt['protocol']['server_args'][receipt['protocol']['server_args'].index('--max-model-len') + 1]),
                    'max_tokens': fragment.get('decoding_params', {}).get('max_tokens')}
        document = build_session_document(
            session_id=name, identity=identity, pid=session.pid, started_at=session.started_at,
            serving_geometry=geometry, props=None, session_receipt_path=str(directory/'session-receipt.json'),
            server_log_path='', monitor_path='',
        )
        document['deployment_parent_session_id'] = session_id
        pending.extend([(directory/'session-receipt.json', session.as_json()), (directory/'session.json', document)])
        if first is None:
            first = dict(document)
            first.pop('deployment_parent_session_id')
            first['session_id'] = session_id
            first['deployment_instance_ids'] = [i['instance_id'] for i in declared]
            first['session_receipt_path'] = str(coordinator/'session-receipt.json')
            # Existing lease and single-endpoint FIT APIs attach at the coordinator
            # directory. Its receipt represents the first verified endpoint.
            pending.append((coordinator/'session-receipt.json',
                            replace(session, session_id=session_id).as_json()))
    pending.append((coordinator/'session.json', first))
    # Validate conflicts across all files before writing any. Publication order makes
    # an interrupted partial set invisible to the generation lease until resumed.
    for path, document in pending:
        if (path.parent/'session-stop.json').exists():
            raise SessionError(f'stopped session needs reconciliation: {path.parent}')
        if path.exists() and path.read_bytes() != _bytes(document):
            raise SessionError(f'conflicting resident session evidence: {path}')
    for path, document in pending:
        try:
            publish_new(path, _bytes(document))
        except FileExistsError:
            if path.read_bytes() != _bytes(document):
                raise SessionError(f'concurrent resident session conflict: {path}')
    return [path for path, _ in pending]

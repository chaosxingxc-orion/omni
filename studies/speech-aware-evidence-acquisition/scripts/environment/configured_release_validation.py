#!/usr/bin/env python3
"""Offline, source-pinned request rendering and checkpoint compatibility precheck."""
import argparse
import hashlib
import json
import re
from pathlib import Path

SCHEMA = 'saea-configured-release-validation-v1'


def _pin(pin):
    if (not isinstance(pin, dict) or set(pin) != {'path', 'sha256'}
            or not isinstance(pin['path'], str) or not Path(pin['path']).is_absolute()
            or not isinstance(pin['sha256'], str) or not re.fullmatch('[0-9a-f]{64}', pin['sha256'])):
        raise ValueError('absolute file path and SHA256 pin required')
    path = Path(pin['path'])
    if path.is_symlink() or not path.is_file():
        raise ValueError('pinned input missing or symlink: ' + str(path))
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != pin['sha256']:
        raise ValueError('pinned input SHA256 differs: ' + str(path))
    return raw


def _root(value):
    if not isinstance(value, str) or not Path(value).is_absolute() or not Path(value).is_dir():
        raise ValueError('existing absolute root required')
    return Path(value)


def _composition(doc):
    from speech_aware_evidence_acquisition.core.config import compose
    from speech_aware_evidence_acquisition.core.experiment_assets import resolve_asset
    repo, data = _root(doc['repo']), _root(doc['data'])
    fragments = doc['fragments']
    if not isinstance(fragments, dict) or set(fragments) != {'model', 'dataset', 'baseline', 'experiment'}:
        raise ValueError('four exact fragment pins required')
    names = {}
    for kind, item in fragments.items():
        if not isinstance(item, dict) or set(item) != {'name', 'sha256'} or not re.fullmatch('[a-zA-Z0-9][a-zA-Z0-9_.-]*', item['name']):
            raise ValueError('fragment name and SHA256 required')
        path = resolve_asset(repo, f'configs/{kind}/{item["name"]}.json')
        _pin({'path': str(path), 'sha256': item['sha256']})
        names[kind] = item['name']
    result = compose(repo, **names, data_root=data)
    if result.config_hash != doc['config_sha256']:
        raise ValueError('composed config SHA256 differs')
    return repo, data, names, result


def _render(doc):
    from speech_aware_evidence_acquisition.core.model.transport import LlamaServerTransport, _llama_server_factory
    from speech_aware_evidence_acquisition.core.preparation import canonical_bytes
    from speech_aware_evidence_acquisition.core.session.receipt import read_session_receipt

    repo, data, names, cfg = _composition(doc)
    _pin(doc['session'])
    session = read_session_receipt(Path(doc['session']['path']))
    values = cfg.values
    if values.get('transport') != 'vllm':
        raise ValueError('offline shared projection supports only vllm transport')
    from speech_aware_evidence_acquisition.core.reasoning_knobs import vllm_forced_close_message_from_server_args
    from speech_aware_evidence_acquisition.core.model.vllm_transport import VLLM_REFUSED_SERVER_FLAGS
    server_args = values.get('server_args') or ()
    if not isinstance(server_args, (tuple, list)) or any(not isinstance(token, str) for token in server_args):
        raise ValueError('vllm server_args must be a string array')
    if VLLM_REFUSED_SERVER_FLAGS & {token.split('=', 1)[0] for token in server_args}:
        raise ValueError('unsupported llama server flag in vllm configuration')
    forced_close_message = vllm_forced_close_message_from_server_args(server_args)

    class OfflineTransport(LlamaServerTransport):
        def _assert_session_still_valid(self):
            # Explicit offline-only bypass: constructor/render cannot establish live identity.
            return None

    def deny_post(*args, **kwargs):
        raise RuntimeError('offline validation prohibits network POST')

    transport = _llama_server_factory(values, data, session=session, runtime_receipt={},
        runtime_root=data, transport_cls=OfflineTransport, backend_kwargs={'post': deny_post},
        forced_close_message=forced_close_message)
    rows = doc['requests']
    if not isinstance(rows, list) or not rows:
        raise ValueError('nonempty pinned requests required')
    results = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {'role', 'intent', 'expected_body_sha256', 'expected_receipt_sha256'}:
            raise ValueError('exact render request entry required')
        if not isinstance(row['role'], str) or not row['role']:
            raise ValueError('request role required')
        for key in ('expected_body_sha256', 'expected_receipt_sha256'):
            if not isinstance(row[key], str) or not re.fullmatch('[0-9a-f]{64}', row[key]):
                raise ValueError('expected render SHA256 required')
        try:
            intent = json.loads(_pin(row['intent']))
            if not isinstance(intent, dict) or not isinstance(intent.get('payload'), dict):
                raise ValueError('intent must contain payload object')
            contact = intent.get('agent_contact_id')
            if not isinstance(contact, str) or ':' + row['role'] + ':' not in contact:
                raise ValueError('intent contact role differs: ' + row['role'])
            body, receipt = transport.render_request_with_receipt(canonical_bytes(intent['payload']))
            actual_body = hashlib.sha256(canonical_bytes(body)).hexdigest()
            actual_receipt = hashlib.sha256(canonical_bytes(receipt)).hexdigest()
            results.append({'role': row['role'], 'intent_sha256': row['intent']['sha256'],
                            'body_sha256': actual_body, 'expected_body_sha256': row['expected_body_sha256'],
                            'receipt_sha256': actual_receipt, 'expected_receipt_sha256': row['expected_receipt_sha256'],
                            'ok': actual_body == row['expected_body_sha256'] and actual_receipt == row['expected_receipt_sha256']})
        except Exception as error:
            results.append({'role': row['role'], 'ok': False, 'error_type': type(error).__name__, 'error': str(error)})
    return {'ok': all(x['ok'] for x in results), 'results': results,
            'config_sha256': cfg.config_hash, 'fragments': names,
            'scope': 'shared projection render body/receipt before vllm_post wire translation; session process liveness and identity explicitly stubbed; no send, FIT or tokenizer'}


def _bridge(doc):
    from speech_aware_evidence_acquisition.core.config import protocol_hash
    from speech_aware_evidence_acquisition.core.flight.checkpoint_compatibility import CheckpointCompatibility, IMPLEMENTATION
    from speech_aware_evidence_acquisition.core.row_checkpoint import read_checkpoint

    repo, data, names, cfg = _composition(doc)
    bridge_doc = json.loads(_pin(doc['bridge']))
    for key in ('commit_object', 'tree_listing', 'source_archive', 'release_manifest', 'source_review', 'reference_pins'):
        _pin(bridge_doc['evidence'][key])
    checkpoint_path = Path(doc['checkpoint']['path'])
    _pin(doc['checkpoint'])
    checkpoint = read_checkpoint(checkpoint_path, expected_sha256=doc['checkpoint']['sha256'])
    if checkpoint.torn_tail_bytes:
        raise ValueError('checkpoint has torn tail')
    header = checkpoint.header
    attempt = doc['predecessor_attempt_id']
    declaration = {'attempt_id': attempt, 'checkpoint_sha256': checkpoint.sha256}
    bridge = CheckpointCompatibility.from_document(bridge_doc, attempt_id=doc['successor_attempt_id'])
    from speech_aware_evidence_acquisition.core.flight.checkpoint_compatibility import ImageReleaseCompatibility
    if not isinstance(bridge, ImageReleaseCompatibility):
        raise ValueError('image release bridge and explicit source review required')
    impl = {key: hashlib.sha256((repo / path).read_bytes()).hexdigest() for key, path in IMPLEMENTATION.items()}
    proof = bridge.precheck(repo_root=repo, data_root=data, current_fragments=names, header=header,
        run_id=doc['run_id'], predecessor_attempt_id=attempt,
        current_protocol_hash=protocol_hash(cfg.config_hash, cfg.values['arm'], header['split_identity_hash']),
        current_split_identity_hash=header['split_identity_hash'], current_implementation=impl, declaration=declaration)
    return {'ok': True, 'proof': proof, 'checkpoint_sha256': checkpoint.sha256,
            'checkpoint_rows': len(checkpoint.rows), 'activation_authorized': False,
            'scope': 'offline bridge precheck only; no terminal ledger verification or authorization'}


def execute(path):
    raw = Path(path).read_bytes()
    doc = json.loads(raw)
    common = {'schema', 'mode', 'repo', 'data', 'fragments', 'config_sha256'}
    if not isinstance(doc, dict) or doc.get('schema') != SCHEMA:
        raise ValueError('invalid validation schema')
    mode = doc.get('mode')
    fields = {'render': {'session', 'requests'}, 'bridge-precheck': {'bridge', 'checkpoint', 'run_id', 'predecessor_attempt_id', 'successor_attempt_id'}}
    if mode not in fields or set(doc) != common | fields[mode]:
        raise ValueError('exact validation mode and fields required')
    result = (_render if mode == 'render' else _bridge)(doc)
    return {'schema': SCHEMA, 'mode': mode, 'input_sha256': hashlib.sha256(raw).hexdigest(),
            'new_model_calls': 0, 'fit_calls': 0, 'teacher_calls': 0, 'tokenizer_calls': 0, **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = execute(args.config)
    except Exception as error:
        result = {'schema': SCHEMA, 'ok': False, 'error_type': type(error).__name__, 'error': str(error),
                  'new_model_calls': 0, 'fit_calls': 0, 'teacher_calls': 0, 'tokenizer_calls': 0}
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result['ok'] else 1)


if __name__ == '__main__':
    main()

"""Switch a pinned resident deployment's client budget without restarting engines.

One JSON configuration drives both directions. A durable pending record blocks
all later transitions after interruption; this command never auto-retries an
unknown outcome, removes a flight lease, or starts/stops serving processes.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import urllib.request

SCHEMA = 'saea-serving-budget-transition-v1'
LEAVES = ('session.json', 'session-receipt.json')
BUDGET_KEYS = {'max_tokens', 'reasoning_budget_tokens', 'answer_reserve_tokens'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def raw(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def inventory_sha256(files):
    hashes = {name: digest(blob) for name, blob in files.items()}
    return digest(json.dumps(hashes, sort_keys=True, separators=(',', ':')).encode())


def sync_directory(directory):
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def durable_mkdir(directory):
    if directory.exists():
        return
    durable_mkdir(directory.parent)
    directory.mkdir()
    sync_directory(directory.parent)


@contextmanager
def exclusive(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def pinned(pin):
    require(isinstance(pin, dict) and set(pin) == {'path', 'sha256'}, 'exact input pin required')
    path = Path(pin['path'])
    require(path.is_absolute(), 'absolute pinned input path required')
    blob = path.read_bytes()
    require(digest(blob) == pin['sha256'], 'pinned transition input changed')
    return json.loads(blob)


def quiescence(instances, *, proc_root=Path('/proc'), opener=urllib.request.urlopen):
    """Observe all declared loopback queues and sockets, refusing unreadable state."""
    ports, queues = set(), {}
    for instance in instances:
        host, port = instance['host'], instance['port']
        require(host in ('127.0.0.1', '::1', 'localhost') and type(port) is int
                and 0 < port < 65536 and port not in ports, 'distinct loopback endpoints required')
        ports.add(port)
        url_host = '['+host+']' if ':' in host else host
        with opener(f'http://{url_host}:{port}/metrics', timeout=10) as response:
            lines = response.read().decode().splitlines()
        counts = {}
        for metric in ('vllm:num_requests_running', 'vllm:num_requests_waiting'):
            values = [float(line.rsplit(' ', 1)[1]) for line in lines
                      if line.startswith(metric+'{') or line.startswith(metric+' ')]
            require(values and all(math.isfinite(v) and v == 0 for v in values),
                    'serving queue active or unknown')
            counts[metric] = sum(values)
        queues[str(port)] = counts
    require(ports, 'nonempty resident deployment required')
    for table in ('tcp', 'tcp6'):
        for line in (proc_root/'net'/table).read_text().splitlines()[1:]:
            fields = line.split()
            require(len(fields) >= 4, 'unreadable socket inventory')
            if fields[3] == '01' and any(int(x.rsplit(':', 1)[1], 16) in ports for x in fields[1:3]):
                raise ValueError('established inference connection remains')
    return {'queues': queues, 'established_connections': [], 'observed_at': time.time()}


def budget(fragment):
    return {'max_tokens': fragment['decoding_params']['max_tokens'],
            'reasoning_budget_tokens': fragment['reasoning_budget_tokens'],
            'answer_reserve_tokens': fragment['answer_reserve_tokens']}


def verify_budget_only(source, target, old, new):
    for expected in (old, new):
        require(isinstance(expected, dict) and set(expected) == BUDGET_KEYS
                and all(type(v) is int and v > 0 for v in expected.values())
                and expected['max_tokens'] == expected['reasoning_budget_tokens']+expected['answer_reserve_tokens'],
                'exact internally consistent client budgets required')
    require({old['max_tokens'], new['max_tokens']} == {4096, 16384}
            and old['reasoning_budget_tokens'] == new['reasoning_budget_tokens']
            and budget(source) == old and budget(target) == new,
            'expected 4096/16384 budget direction or reasoning differs')
    def engine(value):
        result = copy.deepcopy(value)
        result['decoding_params'].pop('max_tokens')
        result.pop('answer_reserve_tokens')
        result.pop('geometry_basis', None)
        return result
    require(engine(source) == engine(target), 'transition changes more than client budget identity')


def load_config(config):
    require(isinstance(config, dict) and set(config) == {'schema', 'repo', 'data', 'model',
        'session', 'observations', 'source_fragment', 'target_fragment_sha256',
        'expected_source_sha256', 'expected_old_budget', 'expected_new_budget', 'out'}
        and config['schema'] == SCHEMA, 'exact serving budget transition config required')
    for field in ('model', 'session'):
        require(isinstance(config[field], str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', config[field]),
                'safe model and coordinator identifiers required')
    paths = {k: Path(config[k]).resolve() for k in ('repo', 'data', 'out')}
    require(all(Path(config[k]).is_absolute() for k in paths), 'absolute transition paths required')
    require(not paths['out'].exists() and not paths['out'].is_relative_to(paths['data']/'serving')
            and not paths['out'].is_relative_to(paths['data']/'locks'), 'fresh separate output directory required')
    return paths


def transition(config):
    paths = load_config(config)
    repo, data, out = (paths[k] for k in ('repo', 'data', 'out'))
    sys.path.insert(0, str(repo/'src'))
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.atomic_file import publish_new
    from speech_aware_evidence_acquisition.core.config import read_fragment, resolve_runtime_receipt_path
    from speech_aware_evidence_acquisition.core.session.receipt import SessionReceipt
    from speech_aware_evidence_acquisition.core.session.serving import assert_identity_matches
    from speech_aware_evidence_acquisition.core.session.vllm_identity import verify_vllm_session, verify_vllm_process_identity
    from speech_aware_evidence_acquisition.core.session.vllm_resident import register_deployment, required_identity
    assert_execution_environment()
    model, session = config['model'], config['session']
    target_path = repo/'configs/model'/(model+'.json')
    require(digest(target_path.read_bytes()) == config['target_fragment_sha256'], 'target model fragment changed')
    target = read_fragment(repo, 'model', model)
    source, observations = pinned(config['source_fragment']), pinned(config['observations'])
    verify_budget_only(source, target, config['expected_old_budget'], config['expected_new_budget'])
    receipt_path = resolve_runtime_receipt_path(repo, target)
    runtime_receipt_bytes = receipt_path.read_bytes()
    runtime_receipt = json.loads(runtime_receipt_bytes)
    instances = runtime_receipt['deployment']['instances']
    members = [i['instance_id'] for i in instances]
    require(members and len(set(members)) == len(members) and session not in members
            and set(observations) == set(members), 'exact observed deployment membership required')
    require(all(isinstance(name, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', name) for name in members),
            'safe deployment members required')
    names = [session, *members]
    files = {str(Path(name)/leaf): data/'serving'/name/leaf for name in names for leaf in LEAVES}
    pending = data/'locks/serving-budget-transition.pending.json'
    lease = data/'locks/flight.lock'
    def no_lease():
        require(not os.path.lexists(lease), 'active or unknown flight lease; preserve it')
    def unchanged_inputs():
        require(digest(target_path.read_bytes()) == config['target_fragment_sha256']
                and pinned(config['source_fragment']) == source
                and pinned(config['observations']) == observations
                and receipt_path.read_bytes() == runtime_receipt_bytes,
                'transition inputs changed during verification')
    def preserve(path, blob):
        durable_mkdir(path.parent)
        publish_new(path, blob)
    def read_all():
        require(not any((data/'serving'/name/'session-stop.json').exists() for name in names),
                'stopped serving identity requires reconciliation')
        return {rel: path.read_bytes() for rel, path in files.items()}
    def verify_live(blobs, fragment, expected_budget):
        for index, name in enumerate(names):
            instance = instances[max(index-1, 0)]
            observation = observations[instance['instance_id']]
            document = json.loads(blobs[str(Path(name)/'session.json')])
            saved = SessionReceipt.from_json(json.loads(blobs[str(Path(name)/'session-receipt.json')]))
            identity = required_identity(fragment=fragment, model_fragment=model,
                runtime_receipt=runtime_receipt, receipt_path=receipt_path, host=instance['host'], port=instance['port'])
            assert_identity_matches(document, identity)
            verify_vllm_session(saved, runtime_receipt)
            require(saved.session_id == name and saved.pid == observation['pid'] == document['pid']
                    and saved.started_at == observation['started_at'] == document['started_at']
                    and saved.startup_stdout_sha256 == observation['startup_stdout_sha256']
                    and document['session_id'] == name
                    and document['serving_geometry']['max_tokens'] == expected_budget['max_tokens'],
                    'source process observation or client budget differs')
            if index == 0:
                require(document.get('deployment_instance_ids') == members, 'coordinator membership differs')
            else:
                require(document.get('deployment_parent_session_id') == session, 'member coordinator differs')
                argv = verify_vllm_process_identity(saved.pid, runtime_receipt, instance,
                                                    host=saved.host, port=saved.port)
                require(list(argv) == list(saved.argv), 'live serving argv differs')
    def replace_file(path, blob):
        temp = path.with_name('.'+path.name+'.budget-transition-new')
        publish_new(temp, blob)
        os.replace(temp, path)
        sync_directory(path.parent)
    def clear_pending():
        pending.unlink(); sync_directory(pending.parent)
    with exclusive(data/'locks/generation.guard'):
        require(not os.path.lexists(pending), 'unfinished budget transition; inspect preserved evidence, never auto-retry')
        no_lease()
        before = read_all()
        require(inventory_sha256(before) == config['expected_source_sha256'], 'source serving inventory changed')
        verify_live(before, source, config['expected_old_budget'])
        quiet = quiescence(instances)
        no_lease()
        unchanged_inputs()
        require(read_all() == before, 'source serving files changed during verification')
        durable_mkdir(out)
        for rel, blob in before.items():
            preserve(out/'original'/rel, blob)
        intent = {'schema': SCHEMA, 'config': config, 'config_sha256': digest(raw(config)),
            'source_sha256': inventory_sha256(before), 'quiescence': quiet,
            'no_engine_restart': True, 'automatic_retry': False, 'new_model_calls': 0}
        publish_new(out/'intent.json', raw(intent))
        publish_new(pending, raw({'schema': SCHEMA, 'out': str(out), 'intent_sha256': digest(raw(intent))}))
        try:
            # Coordinator disappears first; register_deployment publishes it last.
            for name in names:
                for leaf in LEAVES:
                    (data/'serving'/name/leaf).unlink()
                sync_directory(data/'serving'/name)
            register_deployment(data_root=data, fragment=target, model_fragment=model,
                receipt_path=receipt_path, instances=observations, session_id=session)
            after = read_all()
            verify_live(after, target, config['expected_new_budget'])
            no_lease()
            unchanged_inputs()
            final_quiet = quiescence(instances)
            result = {'schema': SCHEMA, 'status': 'COMPLETE',
                'old_budget': config['expected_old_budget'], 'new_budget': config['expected_new_budget'],
                'source_sha256': inventory_sha256(before), 'target_sha256': inventory_sha256(after),
                'quiescence': final_quiet, 'no_engine_restart': True, 'new_model_calls': 0}
            publish_new(out/'complete.json', raw(result))
        except BaseException as error:
            failure = {'schema': SCHEMA, 'error_type': type(error).__name__, 'error': str(error),
                       'restored_original': False, 'automatic_retry': False, 'new_model_calls': 0}
            try:
                # Preserve partial/unknown bytes before any attempt to restore.
                for rel, path in files.items():
                    if path.exists(): preserve(out/'interrupted'/rel, path.read_bytes())
                no_lease(); quiescence(instances)
                require(receipt_path.read_bytes() == runtime_receipt_bytes,
                        'runtime receipt changed; rollback process identity unknown')
                verify_live(before, source, config['expected_old_budget'])
                for rel, blob in before.items(): replace_file(files[rel], blob)
                require(read_all() == before, 'rollback bytes differ')
                verify_live(before, source, config['expected_old_budget'])
                failure.update(status='ROLLED_BACK', restored_original=True)
                publish_new(out/'failed.json', raw(failure))
                clear_pending()
            except BaseException as rollback_error:
                failure.update(status='UNKNOWN', rollback_error_type=type(rollback_error).__name__,
                               rollback_error=str(rollback_error))
                if not (out/'failed.json').exists(): publish_new(out/'failed.json', raw(failure))
                # pending remains: changing out cannot bypass the interrupted attempt.
            raise
        clear_pending()
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(transition(json.loads(args.config.read_bytes())), sort_keys=True))

"""Durable execution of registered grounded shards through ExperienceLearner.

No network or admission implementation lives here. The production caller owns
FlightRuntime admission, native FIT and metering; a native source verifier must
revalidate the whole historical lineage on every invocation. Shards are NOT a
final experience library: cross-shard consolidation remains a separate job.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import copy
import hashlib
import json
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts/analysis')]
from grounded_synthesis_plan import verify_plan
from speech_aware_evidence_acquisition.core.atomic_file import publish_new
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import (
    ExperienceLearner, LearningError, UnknownLearningContact, canonical,
    parse_operations, sha,
)

SCHEMA = 'saea-grounded-shard-execution-v1'
STAGE = 'grounded_shard_update'
USAGE = ('calls', 'prompt_tokens', 'completion_tokens')


def require(condition, message):
    if not condition:
        raise LearningError(message)


def contact_identity(plan, shard):
    return f"{plan['plan_sha256']}:shard-{shard['shard_index']}"


def registration_binding(plan, *, model_id, learning_config_sha256, custody_sha256,
                         max_operations, max_calls):
    """Derive immutable contact identity, not generation authorization."""
    require(type(max_calls) is int and max_calls == len(plan['shards']),
            'exact one-call-per-shard budget required')
    require(type(max_operations) is int and max_operations > 0, 'operation bound required')
    for pin in (learning_config_sha256, custody_sha256, plan['plan_sha256']):
        require(isinstance(pin, str) and len(pin) == 64 and all(c in '0123456789abcdef' for c in pin),
                'SHA256 registration pins required')
    return {'schema': SCHEMA, 'plan_sha256': plan['plan_sha256'], 'model_id': model_id,
            'learning_config_sha256': learning_config_sha256, 'source_custody_sha256': custody_sha256,
            'max_calls': max_calls, 'max_operations': max_operations,
            'contacts': [{'stage': STAGE, 'identity': contact_identity(plan, shard),
                          'messages_sha256': shard['messages_sha256']}
                         for shard in plan['shards']],
            'completion_cap': plan['completion_cap'], 'resident_context': plan['resident_context'],
            'final_consolidation': 'NOT_IMPLEMENTED'}


def validate_custody(custody, plan, model_id):
    """Validate the verifier's full, deduplicated history; never rewrite attempts."""
    require(isinstance(custody, dict) and custody.get('source_native_custody_verified') is True,
            'native historical source custody must be verified')
    for field in ('source_batch_sha256', 'source_operations_sha256'):
        require(custody.get(field) == plan[field], 'historical source pin differs')
    require(custody.get('model_id') == model_id, 'historical model differs')
    require(custody.get('unknown_contacts') == 0, 'historical unknown contacts cannot be imported')
    sources = custody.get('native_source_usage')
    require(isinstance(sources, list) and sources, 'complete historical source ledger required')
    seen, total = set(), dict.fromkeys(USAGE, 0)
    for row in sources:
        identity = (row.get('run_id'), row.get('attempt_id'))
        require(all(isinstance(v, str) and v for v in identity) and identity not in seen,
                'duplicate or missing historical spend identity')
        seen.add(identity)
        require(isinstance(row.get('usage'), dict) and set(row['usage']) == set(USAGE),
                'complete historical token usage required')
        for key in USAGE:
            value = row['usage'][key]
            require(type(value) is int and value >= 0, 'invalid historical usage')
            total[key] += value
    require(custody.get('original_total_usage') == total, 'historical total drops or duplicates cost')
    return total


def execute_shards(learner, plan, batch_bytes, operation_bytes, *, registration,
                   count_tokens, tokenizer_identity, verify_source_custody):
    """Execute/resume registered shards; known responses are never resampled.

    ``verify_source_custody`` performs native journal/attempt/ancestor verification
    and returns its pinned full result. A boolean or cached assertion is not an
    acceptable production verifier. Its exact output is bound by registration.
    Runtime configuration and admission must bind this same registration and the
    plan's completion cap before constructing the learner's metered callbacks.
    """
    require(isinstance(learner, ExperienceLearner), 'existing ExperienceLearner required')
    require(callable(learner.preflight), 'native FIT preflight is mandatory')
    require(callable(verify_source_custody), 'native source verifier required')
    plan, registration = copy.deepcopy(plan), copy.deepcopy(registration)
    verify_plan(plan, batch_bytes, operation_bytes, count_tokens=count_tokens,
                tokenizer_identity=tokenizer_identity)
    custody = copy.deepcopy(verify_source_custody())
    historical = validate_custody(custody, plan, learner.model_id)
    expected = registration_binding(plan, model_id=learner.model_id,
        learning_config_sha256=learner.learning_config_sha256, custody_sha256=sha(custody),
        max_operations=learner.max_operations, max_calls=learner.max_calls)
    require(registration == expected, 'registered shard binding differs')
    source_batch = json.loads(batch_bytes)
    require(source_batch.get('model_id') == learner.model_id, 'source batch model differs')
    binding = {'schema': SCHEMA, 'registration': registration, 'plan': plan,
               'source_custody': custody}
    batch_digest = sha(binding)
    aborted = threading.Event()
    original_preflight = learner.preflight

    def preflight(request):
        if aborted.is_set():
            raise LearningError('shard dispatch stopped before preflight')
        original_preflight(request)
        if aborted.is_set():
            raise LearningError('shard dispatch stopped after preflight; no intent created')

    def persist_same(path, value):
        blob = canonical(value)
        if path.exists():
            require(path.read_bytes() == blob, 'persisted shard execution evidence differs')
        else:
            publish_new(path, blob)

    with learner._exclusive_batch():
        persist_same(learner.root / 'batch.json', binding)
        learner._batch = batch_digest
        learner.last_invocation = {'new_calls': 0, 'reused_calls': 0}
        expected_requests = {}
        for shard in plan['shards']:
            identity = contact_identity(plan, shard)
            key = sha({'stage': STAGE, 'identity': identity})
            request = {'schema': 'saea-tfgrpo-learning-request-v1', 'batch_sha256': batch_digest,
                       'model_id': learner.model_id,
                       'learning_config_sha256': learner.learning_config_sha256,
                       'stage': STAGE, 'identity': identity, 'messages': shard['messages']}
            expected_requests[key] = request
        contacts = learner.root / 'contacts'
        existing = set()
        # Validate every old intent before dispatching any new work. Unknown
        # reconciliation is also completed before newly unspent shards start.
        for path in contacts.glob('*.intent.json'):
            key = path.name.removesuffix('.intent.json')
            require(key in expected_requests and path.read_bytes() == canonical(expected_requests[key]),
                    'existing shard intent is foreign or changed')
            existing.add(key)
        require(all(p.name.removesuffix('.response.json') in existing
                    for p in contacts.glob('*.response.json')), 'orphan shard response')
        results = {}
        progress = learner.root / 'progress'
        progress.mkdir(exist_ok=True)

        def snapshot(status, error=None):
            accounting = learner.accounting()
            combined = {k: historical[k] + accounting['known_usage'][k] for k in USAGE}
            record = {'schema': SCHEMA, 'status': status, 'batch_sha256': batch_digest,
                      'completed_shards': sorted(results), 'total_shards': len(plan['shards']),
                      'historical_usage': historical, 'current_accounting': accounting,
                      'cumulative_known_usage': combined, 'candidate_publishable': False,
                      'final_consolidation': 'NOT_IMPLEMENTED', 'error_type': error}
            index = len(list(progress.glob('*.json')))
            publish_new(progress / f'{index:06d}.json', canonical(record))
            return record

        def execute_one(shard):
            text = learner._call(STAGE, contact_identity(plan, shard), shard['messages'])
            operations = parse_operations(text, source_batch['parent']['items'],
                                          max_operations=learner.max_operations)
            result = {'shard_index': shard['shard_index'], 'messages_sha256': shard['messages_sha256'],
                      'source_group_indices': shard['group_indices'],
                      'source_operation_indices': shard['operation_indices'],
                      'output_operations': operations, 'response_text_sha256': sha(text),
                      'candidate_publishable': False}
            persist_same(learner.root / 'shards' / f"{shard['shard_index']:04d}.json", result)
            return result

        def one(shard):
            try:
                return execute_one(shard)
            except BaseException:
                aborted.set()
                raise

        learner.preflight = preflight
        try:
            # Read known responses first. No unspent model call can slip past
            # an earlier unknown outcome merely because the pool runs in parallel.
            pending = []
            for shard in plan['shards']:
                key = sha({'stage': STAGE, 'identity': contact_identity(plan, shard)})
                if key in existing:
                    results[shard['shard_index']] = one(shard)
                else:
                    pending.append(shard)
            snapshot('RUNNING')
            with ThreadPoolExecutor(max_workers=learner.concurrency) as pool:
                active = {}
                failure = None
                while pending or active:
                    while failure is None and not aborted.is_set() and pending and len(active) < learner.concurrency:
                        shard = pending.pop(0)
                        active[pool.submit(one, shard)] = shard
                    completed, _ = wait(active, return_when=FIRST_COMPLETED) if active else (set(), set())
                    for future in completed:
                        shard = active.pop(future)
                        try:
                            results[shard['shard_index']] = future.result()
                        except BaseException as error:
                            if failure is None:
                                failure = error
                            aborted.set()
                    if failure is not None:
                        pending.clear()
                    # accounting() requires a quiescent contact inventory: a
                    # peer may atomically publish its response during a scan.
                    if not active:
                        snapshot('DRAINING' if failure is not None else 'RUNNING',
                                 type(failure).__name__ if failure else None)
                if failure is not None:
                    raise failure
            record = snapshot('SHARDS_COMPLETE_NOT_CONSOLIDATED')
            final = {**record, 'shard_results': [results[i] for i in sorted(results)]}
            persist_same(learner.root / 'shards-complete.json', final)
            return final
        except BaseException as error:
            aborted.set()
            # Contacts remain authoritative even if damaged accounting prevents
            # a convenience progress record. Do not hide the original failure.
            try:
                snapshot('STOPPED', type(error).__name__)
            except BaseException:
                pass
            raise
        finally:
            learner.preflight = original_preflight

"""Distinguish never-started Direct FIT from complete or unknown FIT custody."""
from __future__ import annotations

from dataclasses import asdict
import json

from .contact_journal import canonical, comparable_request, digest
from .preparation import PreparationError, load_artifact, digest as artifact_digest


def allow_first_direct_fit(runtime, replay, path, journal, custody_path, fit_plan):
    """Permit a first FIT only with zero generation spend and no FIT evidence.

    The fixed FIT identity, exposure ledger and live generation lease still
    apply. Any partial evidence follows the strict reuse path and never retries.
    """
    runtime.assert_generation_lease()
    replay.assert_all_contacts_observed()
    if any(p.exists() or p.is_symlink() for p in (path, journal, custody_path)):
        return False
    if replay.request_ids or replay.source_actual_usage != {'calls': 0, 'audio_seconds': 0.0}:
        raise PreparationError('spent Direct predecessor has no FIT custody; no tokenizer resend')
    # Source ancestry was independently pinned as all-zero by NativeContactReplay.
    runtime._gate_touch(runtime._gate, fit_plan)
    runtime._require_exposure_row_unconsumed(fit_plan.run_id)
    runtime.assert_generation_lease()
    replay.assert_all_contacts_observed()
    return True


def reuse_direct_fit(runtime, replay, path, journal, custody_path, payloads, inputs, fit_plan):
    """Require original durable success; partial/missing custody never retries."""
    from .preparation_fit import boundary_identity
    try:
        runtime.assert_generation_lease()
        replay.assert_all_contacts_observed()
        if not all(p.is_file() for p in (path, journal, custody_path)):
            raise PreparationError('Direct predecessor FIT incomplete or unknown; no tokenizer resend')
        blob = custody_path.read_bytes()
        document = json.loads(blob)
        original = tuple(document['payloads'])
        source_attempts = {replay.source_plan.attempt_id}
        ancestry = replay.source_binding().get('empty_ancestry')
        while ancestry:
            source_attempts.add(ancestry['source']['plan']['attempt_id'])
            ancestry = ancestry['source'].get('empty_ancestry')
        if (document.get('schema') != 'saea-direct-fit-custody-v1'
                or document['generation_run_id'] != replay.source_plan.run_id
                or document['generation_attempt_id'] not in source_attempts
                or canonical([comparable_request(p) for p in original]) !=
                   canonical([comparable_request(p) for p in payloads])):
            raise PreparationError('Direct predecessor FIT request or generation identity changed')
        if canonical(document['session_identity']) != canonical(runtime.session_required_identity()):
            raise PreparationError('Direct predecessor FIT service identity changed')
        expected = {**runtime.request_fit_inputs(original),
            'live_phase': 'direct', 'fit_plan': asdict(fit_plan),
            'direct_only': inputs['direct_only']}
        if canonical(document['inputs']) != canonical(expected):
            raise PreparationError('Direct predecessor FIT configuration or source changed')
        receipt = document['receipt']
        if receipt['path'] != str(path) or receipt.get('contact_class') != 'tokenizer':
            raise PreparationError('Direct predecessor FIT receipt path or contact class changed')
        output = load_artifact(path, expected_sha256=receipt['sha256'],
            kind='request-fit', inputs=expected)
        rows = output.get('rows', [])
        n = len(payloads)
        if (output.get('every_request_fits') is not True
                or output.get('boundary_sha256') != boundary_identity(original)
                or output.get('sample_count') != n or len(rows) != n
                or output.get('contacts') != n or output.get('cumulative_contacts') != n
                or output.get('generation_contacts') != 0 or output.get('phase') != 'direct'
                or output.get('session_id') != runtime._session.session_id
                or output.get('session_identity_sha256') !=
                   artifact_digest(document['session_identity'])):
            raise PreparationError('Direct predecessor FIT output does not certify complete wave')
        endpoint_blob = journal.read_bytes()
        endpoints = [json.loads(line) for line in endpoint_blob.splitlines()]
        if digest(endpoint_blob) != document['endpoint_journal_sha256'] or len(endpoints) != n:
            raise PreparationError('Direct predecessor FIT endpoint journal changed or incomplete')
        for index, endpoint in enumerate(endpoints, 1):
            expected_endpoint = {'contact': index, 'method': 'POST', 'route': '/tokenize',
                'run_id': fit_plan.run_id, 'attempt_id': fit_plan.attempt_id,
                'generation_run_id': replay.source_plan.run_id,
                'session_id': runtime._session.session_id, 'phase': 'direct'}
            if endpoint != expected_endpoint:
                raise PreparationError('Direct predecessor FIT endpoint identity changed')
        runtime.assert_generation_lease()
        return {**receipt, 'direct_fit_reuse': {
            'schema': 'saea-direct-fit-reuse-v1', 'source_path': str(custody_path),
            'source_sha256': digest(blob), 'new_tokenizer_contacts': 0}}
    except (KeyError, TypeError, ValueError, OSError) as error:
        if isinstance(error, PreparationError):
            raise
        raise PreparationError(f'cannot reuse Direct predecessor FIT: {error}') from error

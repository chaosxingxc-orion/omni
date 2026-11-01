"""Verify historical contact custody before excluding calls from fresh spend."""
from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Mapping

from ...contracts import ExecutionPlan
from ..contact_journal import NativeContactReplay, canonical, digest
from . import BundleError


def verify_native_replay(manifest, records, plan):
    markers = {rid.removesuffix('/contact-replay'): value
               for rid, value in records.items() if rid.endswith('/contact-replay')}
    cost = manifest.get('cost', {})
    source = cost.get('native_replay_source') if isinstance(cost, Mapping) else None
    totals = cost.get('replayed_predecessor_cost') if isinstance(cost, Mapping) else None
    if source is None and totals is None and not markers:
        return frozenset()
    try:
        if not isinstance(source, Mapping) or not isinstance(totals, Mapping):
            raise ValueError('replay markers and costs require bound predecessor custody')
        if source.get('schema') != 'saea-native-replay-source-v1':
            raise ValueError('unsupported native replay source')
        source_plan = ExecutionPlan(**source['plan'])
        target = manifest['attempt']
        target_path = Path(target['state_path'])
        if (not target_path.is_absolute() or target.get('attempt_id') != plan.attempt_id
                or target_path.name != plan.attempt_id + '.json'
                or target_path.parent.name != plan.run_id
                or target_path.parent.parent.name != 'attempts'):
            raise ValueError('recovery attempt path/identity mismatch')
        data_root = target_path.parent.parent.parent
        source_path = data_root / 'attempts' / plan.run_id / (source_plan.attempt_id + '.json')
        if (str(source_path.resolve()) != source['attempt_path']
                or digest(source_path.read_bytes()) != source['attempt_sha256']):
            raise ValueError('predecessor attempt bytes changed')
        replay = NativeContactReplay(data_root, source_plan, plan, manifest['arm'])
        if replay.source_binding() != source:
            raise ValueError('predecessor usage/plan differs from its source binding')
        # Complete-row reuse has its own accounting contract. Do not mix it
        # with contact replay until their disjointness is explicitly proved.
        if manifest.get('predecessor_attempt') or manifest.get('predecessor_attempts'):
            raise ValueError('combined row/contact recovery is not supported')
        expected = dict(calls=0, audio_seconds=0., latency_seconds=0., prompt_tokens=0, completion_tokens=0)
        for rid, marker in markers.items():
            if marker != replay.receipt(rid):
                raise ValueError('replay receipt differs from durable source')
            observed = replay.source.inspect(rid)
            intent, response = observed['intent'], observed['response']
            payload = intent['payload']
            if records.get(rid + '/request') != {
                    'arm': manifest['arm'], 'payload': payload,
                    'payload_sha256': intent['request_sha256']}:
                raise ValueError('replayed request differs from durable intent')
            raw = json.loads(observed['response_bytes'])
            text = raw['text']
            usage = {str(k): int(v) for k, v in dict(raw.get('usage', {})).items()}
            if not isinstance(text, str) or records.get(rid + '/response') != {
                    'text': text, 'usage': usage,
                    'response_sha256': response['response_sha256'],
                    'response_text_sha256': digest(text.encode())}:
                raise ValueError('replayed response differs from durable transport bytes')
            contact_cost = dict(calls=1, audio_seconds=float(payload['audio_seconds']),
                latency_seconds=response['latency_seconds'],
                prompt_tokens=usage.get('prompt_tokens', 0),
                completion_tokens=usage.get('completion_tokens', 0))
            traced_cost = records.get(rid + '/cost', {})
            if any(traced_cost.get(key) != value for key, value in contact_cost.items()):
                raise ValueError('replayed contact cost differs from durable response')
            for key, value in contact_cost.items():
                expected[key] += value
        # Same values, but tolerate floating summation order of concurrent contacts.
        import math
        if set(totals) != set(expected) or any(
                type(totals[key]) not in (int, float) or not math.isfinite(totals[key])
                or not math.isclose(totals[key], value, rel_tol=1e-12, abs_tol=1e-9)
                for key, value in expected.items()):
            raise ValueError('replayed predecessor totals do not recompute')
        return frozenset(markers)
    except (KeyError, TypeError, ValueError, OSError, AttributeError) as error:
        raise BundleError(f'native replay custody: {error}') from error

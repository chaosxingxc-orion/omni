"""Retrospective proof for a reviewed pre-dispatch session-check failure.

This is distinct from a live transport event. Missing HTTP logs alone prove
nothing. Require positive nested failure events, exact reviewed source bytes,
producer/launch/lease identity and a terminal attempt. No request is sent here.
"""
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re

SCHEMA = 'saea-contact-reconciliation-v1'
POLICY = 'reviewed-session-render-failure-v1'
PREFIX = 'studies/speech-aware-evidence-acquisition/src/speech_aware_evidence_acquisition/'
# These implementations were reviewed together: render checks session identity
# before constructing a body; __call__ does not catch render failures or POST
# until render returns; wrappers do not retry; timing errors propagate.
REVIEWED = {
    'core/model/transport.py': '2472c0fdd7efb1daa5b2c93c54a9453f2e87a8d1b126f6c953050b4735abd122',
    'core/model/adapter.py': 'e3797474fb2c2da8339e443cb5867850907f4817dbb363b90793cedaf5ddefd5',
    'core/model/vllm_transport.py': '4dc935cc473c05355eb2779c082562211ba22d26fea19182f8a23f3c0dd78a27',
    'core/stage_timing.py': '966d05435a1516c2567207c08a1dfcab56c97266484628cb439ffb25c28ec017',
    'core/session/vllm_identity.py': 'bc9b71061380c464a5ef2eb233091a88cab6ffd6d9e81dca021d9a5cca108b93',
    'core/session/worker.py': '935efb92cfaa641d7cb8db1e2396a69a4c2fd6ce64ed6b139842307cf5641cec',
    'core/session/resource_guard.py': '2753998c25746e5cfd146a2f86c4977277158807ea7f1a743f2d34449f07f9ff',
    'core/model/vllm_router_transport.py': '67e6b1085f036b9676169e08a119d29865a0e345eabd7a8c89dc7997b33facc0',
}
EVIDENCE = frozenset({'timing', 'timing_config', 'attempt', 'lease', 'container', 'launch', 'source_manifest'})


def _require(value, message):
    if not value:
        raise ValueError('contact reconciliation: ' + message)


def _hash(blob):
    return hashlib.sha256(blob).hexdigest()


def _read(pin):
    _require(isinstance(pin, dict) and set(pin) == {'path', 'sha256'}, 'exact evidence pin required')
    p = Path(pin['path'])
    _require(p.is_absolute() and p.is_file() and not p.is_symlink() and p.resolve() == p,
             'regular absolute evidence path required')
    raw = p.read_bytes()
    _require(_hash(raw) == pin['sha256'], 'evidence bytes changed')
    return raw


def _stamp(value):
    return datetime.fromisoformat(re.sub(r'(\.\d{6})\d+', r'\1', value).replace('Z', '+00:00')).timestamp()


def _host_path(launch, logical):
    candidates = [(len(m['target']), m) for m in launch['mounts']
                  if Path(logical).is_relative_to(Path(m['target']))]
    _require(candidates, 'evidence path is not mounted by producer')
    m = max(candidates, key=lambda row: row[0])[1]
    return str(Path(m['source']) / Path(logical).relative_to(m['target']))


def validate(document, *, intent_blob, identity):
    """Recompute proof from immutable evidence; caller checks journal conflicts."""
    _require(isinstance(document, dict) and set(document) == {
        'schema', 'policy', 'identity', 'request_id', 'request_sha256',
        'intent_sha256', 'verdict', 'evidence'}, 'exact reconciliation document required')
    intent = json.loads(intent_blob)
    _require(document['schema'] == SCHEMA and document['policy'] == POLICY
             and document['verdict'] == 'not_sent' and document['identity'] == identity
             and document['request_id'] == intent['request_id']
             and document['request_sha256'] == intent['request_sha256']
             and document['intent_sha256'] == _hash(intent_blob), 'request identity differs')
    pins = document['evidence']
    _require(isinstance(pins, dict) and set(pins) == EVIDENCE, 'complete evidence set required')
    blobs = {name: _read(pin) for name, pin in pins.items()}
    docs = {name: json.loads(raw) for name, raw in blobs.items() if name != 'timing'}
    launch, manifest, lease, attempt = (docs[k] for k in ('launch', 'source_manifest', 'lease', 'attempt'))
    inspect = docs['container']
    _require(isinstance(inspect, list) and len(inspect) == 1, 'one exact container required')
    container = inspect[0]; state = container['State']; env = launch['environment']
    _require(state['Status'] == 'exited' and state['Running'] is False and state['Pid'] == 0,
             'producer is not terminal')
    _require(container['Config']['Labels']['saea.launch.config'] == _hash(json.dumps(launch, sort_keys=True, separators=(',', ':')).encode())
             and container['Config']['Image'] == launch['image']
             and container['HostConfig']['PidMode'] == 'host', 'producer launch differs')
    actual_env = dict(item.split('=', 1) for item in container['Config']['Env'])
    _require(all(actual_env.get(k) == v for k, v in env.items()), 'producer environment differs')
    _require(attempt['state'] == 'ABORTED' and all(
        attempt[k] == lease[k] == identity[k] for k in ('run_id', 'attempt_id')),
        'terminal attempt or lease identity differs')
    _require(type(lease['pid']) is int and lease['pid'] > 0
             and lease['study_head'] == launch['expected_commit'] == manifest['commit'],
             'source or producer identity differs')
    _require(manifest['schema'] == 'saea-release-source-v1'
             and launch['source_identity']['manifest_sha256'] == _hash(blobs['source_manifest'])
             and env['SAEA_RELEASE_MANIFEST_SHA256'] == _hash(blobs['source_manifest'])
             and all(manifest['files'].get(PREFIX + k) == sha for k, sha in REVIEWED.items()),
             'unreviewed source implementation')
    _require(env['SAEA_OBSERVABILITY_RUN_ID'] == identity['run_id']
             and env['SAEA_OBSERVABILITY_ATTEMPT_ID'] == identity['attempt_id']
             and env['SAEA_OBSERVABILITY_MODE'] == 'debug', 'producer timing identity differs')
    # Evidence copies may be archived anywhere. Their origin is a separately
    # pinned producer path; validate that source/timing paths were real mounts.
    _host_path(launch, env['SAEA_STAGE_TIMING_PATH'])
    source_host = _host_path(launch, env['SAEA_RELEASE_MANIFEST'])
    source_mount = next(m for m in launch['mounts'] if m['target'] == '/opt/saea-release')
    _require(source_mount['readonly'] is True and any(m['Destination'] == '/opt/saea-release'
        and m['Source'] == source_mount['source'] and m['RW'] is False for m in container['Mounts']),
        'producer source was not the pinned readonly mount')
    _require(source_host.endswith('/source-manifest.json'), 'source manifest path differs')
    effective = docs['timing_config']['effective']
    _require(effective.get('mode') == 'debug' and effective.get('client_detail') is True
             and all(effective.get(k) == identity[k] for k in ('run_id', 'attempt_id'))
             and docs['timing_config']['configuration_sha256'] == _hash(json.dumps(effective, sort_keys=True).encode()),
             'timing configuration differs')
    raw = blobs['timing']
    _require(raw.endswith(b'\n'), 'torn timing evidence')
    rows = [json.loads(line) for line in raw.splitlines()]
    target = [r for r in rows if r.get('request_id') == intent['request_id']]
    _require(target, 'no positive request evidence')
    for r in target:
        _require(r.get('schema') == 'saea-stage-timing-v1' and r.get('observability_mode') == 'debug'
                 and r.get('pid') == lease['pid'] and r.get('sample_id') == intent['sample_id']
                 and all(r.get(k) == identity[k] for k in ('run_id', 'attempt_id')),
                 'timing request/producer identity differs')
        _require(all(type(r.get(k)) in (int, float) and math.isfinite(r[k]) and r[k] >= 0
                     for k in ('started_monotonic', 'elapsed_seconds')), 'invalid timing interval')
    _require(len({r['thread'] for r in target}) == 1, 'request crossed unreviewed thread boundary')
    def one(stage, status):
        found = [r for r in target if r['stage'] == stage and r['status'] == status]
        _require(len(found) == 1, 'positive unique ' + stage + ' failure required')
        return found[0]
    request, transport, render = [one(stage, 'error') for stage in ('request_total','transport_total','request_render')]
    session, namespace, probe = [one(stage, 'error') for stage in ('session_total','session_namespace','session_namespace_probe')]
    _require(all(sum(r['stage'] == stage for r in target) == 1
                 for stage in ('request_total', 'transport_total', 'request_render')),
             'multiple attempts or renders for same request')
    _require(namespace['parent_span_id'] == session['span_id'] and probe['parent_span_id'] == namespace['span_id'],
             'session failure ancestry differs')
    def contains(outer, inner):
        return (outer['started_monotonic'] <= inner['started_monotonic']
                and inner['started_monotonic'] + inner['elapsed_seconds'] <=
                outer['started_monotonic'] + outer['elapsed_seconds'] + 1e-8)
    _require(all(contains(a, b) for a, b in zip(
        (request, transport, render, session, namespace), (transport, render, session, namespace, probe))),
        'failure intervals do not nest before dispatch')
    _require(not any(r['stage'].startswith(('http_', 'client_http_', 'response_'))
                     or r['stage'] == 'request_json_encode' for r in target),
             'post-render activity contradicts not-sent proof')
    wall = probe['started_unix_ns'] / 1e9
    _require(_stamp(state['StartedAt']) <= _stamp(lease['started_at']) <= wall
             <= wall + probe['elapsed_seconds'] <= _stamp(state['FinishedAt']), 'producer lifetime differs')
    return {'state': 'not_sent', 'basis': POLICY, 'request_id': intent['request_id'],
            'evidence_sha256': {k: p['sha256'] for k, p in pins.items()}, 'new_model_calls': 0}

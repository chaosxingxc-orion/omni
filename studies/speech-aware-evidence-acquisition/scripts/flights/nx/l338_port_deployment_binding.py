"""Validate measured operational inputs before replacing a port package's placeholders."""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

from speech_aware_evidence_acquisition.core.deployment import load_deployment_profile
from speech_aware_evidence_acquisition.core.session.serving import read_session_document


def pinned_json(pin, root):
    path = Path(root) / pin['relpath']
    if Path(pin['relpath']).is_absolute():
        raise ValueError('binding evidence path must be relative')
    if path.resolve().is_relative_to(Path(root).resolve()) is False:
        raise ValueError('binding evidence must stay under its declared root')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != pin['sha256']:
        raise ValueError(f'binding evidence hash mismatch: {path}')
    return json.loads(raw)


def finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{label} must be finite and readable')
    return value


def validate_provenance(binding, profile, receipt, session, repo, data_root):
    """Connect both measurement starts to the recorded deployment and member PIDs."""
    profile_path = Path(repo)/'configs/deployment'/f'{profile.profile_id}.json'
    raw_profile_sha = hashlib.sha256(profile_path.read_bytes()).hexdigest()
    starts = {kind: pinned_json(binding[kind+'_start'], data_root) for kind in ('normal', 'worst')}
    parent = pinned_json(binding['recorded_chain'], data_root)
    if starts['normal']['parent_status_sha256'] != binding['recorded_chain']['sha256']:
        raise ValueError('normal start does not bind the recorded deployment')
    if starts['worst']['parent_started_sha256'] != binding['normal_start']['sha256']:
        raise ValueError('worst start does not bind the normal attempt')
    for kind, start in starts.items():
        if (start['receipt_sha256'] != binding['runtime_receipt']['sha256'] or
                start['profile_sha256'] != raw_profile_sha):
            raise ValueError(f'{kind} start has a different deployment receipt/profile')
        if not start.get('git_head') or len(start.get('driver_sha256', '')) != 64:
            raise ValueError(f'{kind} start lacks driver provenance')
    observations = starts['worst']['deployment_observed']
    if [item['Name'].lstrip('/') for item in observations] != [i.name for i in profile.instances]:
        raise ValueError('worst start does not cover the deployment')
    members = binding['members']
    if set(members) != {i.name for i in profile.instances}:
        raise ValueError('binding must pin every session member')
    lines = parent['steps']['deploy-status']['decision']['status_lines']
    for instance, observed in zip(profile.instances, observations, strict=True):
        member = pinned_json(members[instance.name], data_root)
        read_session_document(Path(data_root)/members[instance.name]['relpath'])
        identity = dict(member['identity'])
        identity['host'], identity['port'] = session['identity']['host'], session['identity']['port']
        argv = list(identity['server_argv'])
        if argv.count('--port') != 1 or argv[argv.index('--port')+1] != str(instance.port):
            raise ValueError('session member argv port differs')
        argv[argv.index('--port')+1] = str(session['identity']['port'])
        identity['server_argv'] = argv
        if (member['session_id'] != instance.name or
                member['deployment_parent_session_id'] != session['session_id'] or
                identity != session['identity'] or member['identity']['port'] != instance.port):
            raise ValueError('session member identity or parent differs')
        state = observed['State']
        if (state['Pid'] != member['pid'] or state['StartedAt'] != member['started_at'] or
                not state['Running'] or observed['RestartCount'] != 0):
            raise ValueError('measured deployment differs from registered member process')
        if not any(instance.name+' ' in line and 'started='+state['StartedAt']+']' in line for line in lines):
            raise ValueError('measurement deployment differs from original startup')
    return starts


def validate_binding(binding, repo, data_root):
    """Require successful normal/worst evidence for the exact runtime and width.

    The lock-row rate is a two-contact equivalent from the synthetic measurement,
    not an observed SLURP row rate. Smoke sign-off checks actual task cost next.
    """
    if binding.get('schema') != 'saea-port-deployment-binding-v1':
        raise ValueError('unknown deployment binding schema')
    profile = load_deployment_profile(binding['profile'], Path(repo), data_root=Path(data_root))
    model_pin = binding.get('model_fragment')
    if model_pin is not None:
        expected_path = f'configs/model/{profile.model_fragment}.json'
        if model_pin.get('relpath') != expected_path:
            raise ValueError('binding model fragment differs from deployment profile')
        pinned_json(model_pin, repo)
    if profile.sha256 != binding['profile_sha256']:
        raise ValueError('deployment profile hash differs from the binding')
    level = binding['admitted_level']
    if isinstance(level, bool) or not isinstance(level, int) or not 1 <= level <= (profile.admitted_level or profile.max_num_seqs):
        raise ValueError('binding admitted level exceeds the engine ceiling')
    receipt = pinned_json(binding['runtime_receipt'], repo)
    for member, instance in zip(profile.instances, receipt['deployment']['instances'], strict=True):
        if (member.name, member.port, list(member.cards), profile.max_num_seqs, profile.tensor_parallel_size) != (
                instance['instance_id'], instance['port'], instance['device_ids'], instance['max_num_seqs'], instance['tensor_parallel_size']):
            raise ValueError('binding profile and receipt deployment differ')
    session = pinned_json(binding['session'], data_root)
    read_session_document(Path(data_root)/binding['session']['relpath'])
    if (session['session_id'] != binding['session_id'] or
            session['identity']['runtime_receipt_sha256'] != binding['runtime_receipt']['sha256'] or
            session['deployment_instance_ids'] != [i.name for i in profile.instances]):
        raise ValueError('binding coordinator does not match runtime receipt/profile')
    starts = validate_provenance(binding, profile, receipt, session, repo, data_root)
    measured = {}
    for kind in ('normal', 'worst'):
        evaluation = pinned_json(binding[kind], data_root)
        measure = evaluation['measure']
        summary = measure['summary']
        if evaluation.get('passes') is not True or evaluation.get('reasons'):
            raise ValueError(f'{kind} measurement did not pass')
        report_pin = binding[kind + '_report']
        if report_pin['sha256'] != measure['report_sha256']:
            raise ValueError('evaluation report hash differs from binding')
        raw_report = pinned_json(report_pin, data_root)
        settings = raw_report['settings']
        expected_urls = [profile.base_url(i) for i in profile.instances]
        if (raw_report.get('synthetic') is not True or raw_report.get('dataset_rows') != 0 or
                raw_report.get('endpoint_died') or len(raw_report['levels']) != 1 or
                settings['model'] != receipt['protocol']['served_model_name'] or
                settings['base_urls'] != expected_urls or settings['thinking_budget'] != 3072 or
                settings['max_tokens'] != 4096 or settings['levels'] != [level] or
                settings['min_tokens'] != (4096 if kind == 'worst' else None)):
            raise ValueError(f'{kind} report settings do not describe the registered synthetic base')
        if datetime.fromisoformat(raw_report['generated_at']) <= datetime.fromisoformat(starts[kind]['started'].replace('Z', '+00:00')):
            raise ValueError(f'{kind} report predates its start')
        original = dict(raw_report['levels'][0], instances=len(profile.instances))
        if original != summary:
            raise ValueError(f'{kind} summary differs from its measured report')
        if summary['concurrency_per_instance'] != level or measure['instances'] != len(profile.instances):
            raise ValueError('binding width differs from measured width')
        if (summary.get('errors') or summary.get('timeouts') or summary.get('aborted') or
                summary.get('preemptions_delta') != 0 or summary['ok'] != summary['requests'] or summary.get('ok', 0) < level * len(profile.instances) * 3):
            raise ValueError(f'{kind} evidence fails admission invariants')
        if len(raw_report['requests']) != summary['requests'] or any(r['status'] != 'ok' for r in raw_report['requests']):
            raise ValueError(f'{kind} request records are incomplete')
        waiting = max(finite(summary.get('waiting_steady_state', {}).get('steady_peak'), 'steady waiting'),
                      finite(measure['poll_waiting_steady'], 'poll steady waiting'))
        if waiting is None or (level == profile.max_num_seqs and waiting > 0):
            raise ValueError(f'{kind} waiting evidence is unreadable or ceiling-bound')
        if (finite(summary['latency_s']['max'], 'latency') >= 420 or finite(summary['e2e_tok_s']['p10'], 'p10') < 10
                or finite(measure['min_hbm_free_mib'], 'free HBM') < 2048):
            raise ValueError(f'{kind} speed or memory threshold failed')
        if measure['names'] != [i.name for i in profile.instances]:
            raise ValueError('measurement instance names differ')
        if kind == 'worst':
            before, after, gauges, polled = (summary['metrics_before'], summary['metrics_after'],
                                            summary['gauges'], measure['poll_kv_peak'])
            if any(len(values) != len(profile.instances) for values in (before, after, gauges, polled)):
                raise ValueError('worst-case evidence omits an instance')
            for index in range(len(profile.instances)):
                pre = finite(after[index]['preemptions'], 'preemptions') - finite(before[index]['preemptions'], 'preemptions')
                kv = max(finite(gauges[index]['kv_cache_usage_perc']['max'], 'KV'), finite(polled[index], 'poll KV'))
                if pre != 0 or kv > 0.9:
                    raise ValueError('worst-case per-instance admission failed')
        if measure.get('max_tokens_sent') != 4096 or (kind == 'worst' and measure.get('min_tokens_sent') != 4096):
            raise ValueError('binding evidence is not the 4096-token base')
        measured[kind] = summary
    normal = measured['normal']
    rate = normal['ok'] / finite(normal['wall_s'], 'wall seconds') * 60 / len(profile.instances) / 2
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError('invalid measured rate')
    return {**binding, 'lock_rows_per_minute_per_instance': rate,
            'rate_basis': 'synthetic completed contacts / wall seconds * 60 / instances / 2; task cost checked by smoke',
            'instances': len(profile.instances), 'tensor_parallel_size': profile.tensor_parallel_size,
            'server_max_num_seqs': profile.max_num_seqs, 'devices': list(profile.cards)}


def bind_family(family, binding):
    """Bind this writer process only; default historical derivations stay unchanged."""
    family.DEPLOYMENT_BINDING = binding
    if binding.get('model_fragment') is not None:
        family.MODEL_FRAGMENT = Path(binding['model_fragment']['relpath']).stem
        for spec in family.CELLS.values():
            if 'model' in spec:
                spec['model'] = family.MODEL_FRAGMENT
    family.SERVICE_SESSION = binding['session_id']
    family.DEPLOYMENT_REF = f"{binding['profile']} at client admission {binding['admitted_level']} per instance"
    family.PORT_DEPLOYMENT_PROFILE = {
        'name': binding['profile'], 'status': 'MEASURED',
        'fields': {k: binding[k] for k in ('instances', 'tensor_parallel_size', 'devices')},
        'admitted_level': binding['admitted_level'], 'max_num_seqs': binding['server_max_num_seqs'],
        'evidence': binding,
        'filled_by': 'hash-pinned normal and worst-case measurements; original server profile unchanged',
        'not_a_fact_here': 'Deployment is now bound to measured evidence outside the model protocol.',
    }
    family.CARD_HOUR_FORMULA = {
        'arm': f"ceil_0.1(1.3 * rows * {family.ARM_MIX_ROWS} * {binding['tensor_parallel_size']} / {binding['lock_rows_per_minute_per_instance']:.12g} / 60)",
        'smoke': f"{family.SMOKE_WALL_CLOCK_HOURS} wall hours * {binding['instances']} instances * {binding['tensor_parallel_size']} cards",
    }
    for key, spec in family.CELLS.items():
        spec['session'] = binding['session_id']
        spec['plan_gpu_hours_placeholder'] = (
            family.provisional_smoke_card_hours(instances=binding['instances'], tensor_parallel_size=binding['tensor_parallel_size'])
            if key == 'smoke' else family.provisional_card_hours(spec['rows'],
                lock_rows_per_minute_per_instance=binding['lock_rows_per_minute_per_instance'],
                tensor_parallel_size=binding['tensor_parallel_size']))

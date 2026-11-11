"""Offline S1 journal census, scoped costs and atomic result handoff.

No model transport or generation admission is performed here. Parent receipts
remain immutable and are referenced separately from the current scientific run.
"""
from __future__ import annotations

import base64
from collections import Counter
import json
import os
from pathlib import Path
import signal

from .configured_s1 import file_sha, read, require, save, sha


def census(lanes, total_samples, total_requests):
    usage, finish, states = Counter(), Counter(), Counter(observed=0, unknown=0, absent=0)
    latency, no_usage, errors = 0.0, [], []
    for lane in lanes:
        run = Path(lane['rows']).parent
        config = read(run.parent / 'runtime-config.json')
        folder = run / 'contact-journals' / config['run_id'] / config['attempt_id']
        intents = {p.name[:-12]: p for p in folder.glob('*.intent.json')}
        responses = {p.name[:-14]: p for p in folder.glob('*.response.json')}
        require(set(responses) <= set(intents), 'response without intent')
        require(len(intents) <= lane['total_requests'], 'too many intents')
        states['observed'] += len(responses)
        states['unknown'] += len(set(intents) - set(responses))
        states['absent'] += lane['total_requests'] - len(intents)
        identity = None
        for key, path in responses.items():
            response, intent = read(path), read(intents[key])
            raw = base64.b64decode(response['response_b64'], validate=True)
            require(sha(raw) == response['response_sha256'] and
                    response['identity'] == intent['identity'] and
                    response['request_id'] == intent['request_id'], 'raw journal binding')
            if identity is None:
                identity = response['identity']
            require(identity == response['identity'], 'mixed journal identity')
            body = json.loads(raw)
            choice = body.get('choices', [{}])[0]
            finish[choice.get('finish_reason', 'HTTP-error')] += 1
            latency += response['latency_seconds']
            provided = body.get('usage') or {}
            if not provided:
                no_usage.append(response['request_id'])
            for name, value in provided.items():
                if type(value) in (int, float):
                    usage[name] += value
            if 'error' in body:
                errors.append(dict(request_id=response['request_id'], error=body['error']))
    require(sum(l['total_requests'] for l in lanes) == total_requests, 'request denominator')
    return dict(schema='s1-scoped-journal-cost-v2', states=dict(states), total_requests=total_requests,
                total_samples=total_samples, usage=dict(usage), finish_reasons=dict(finish),
                logical_http_seconds=latency, requests_without_usage=len(no_usage), HTTP_errors=errors,
                cards_per_instance=lanes[0]['cards'], instances=len(lanes),
                total_cards=sum(l['cards'] for l in lanes),
                card_hours=sum(l['card_hours'] for l in lanes),
                wall_seconds=max(l['wall_seconds'] for l in lanes), money=None,
                lanes=[{k: v for k, v in l.items() if k != 'launch'} for l in lanes])


def scope_costs(current, parent_path, premodel_path):
    """Keep legacy top-level current fields for the original scorer contract."""
    parent, premodel = read(parent_path), read(premodel_path)
    require(premodel['model_calls'] == 0, 'pre-model receipt has model calls')
    prior = parent.get('prior_attempts', [])
    parent_requests = parent['states']['observed'] + sum(p['requests'] for p in prior)
    parent_hours = parent['card_hours'] + sum(p['card_hours'] for p in prior)
    require(parent_requests == parent['all_attempt_requests'], 'parent request accounting')
    require(abs(parent_hours - parent['all_attempt_card_hours']) < 1e-9, 'parent card accounting')
    result = dict(current)
    result.update(cost_scope='current_scientific_run_only', prior_attempts=[],
                  all_attempt_requests=current['states']['observed'],
                  all_attempt_card_hours=current['card_hours'],
                  pre_model_failures=dict(path=str(premodel_path), sha256=file_sha(premodel_path),
                      model_calls=0, bind_wall_seconds=premodel['total_bind_wall_seconds'],
                      cpu_time_seconds=None, money=None,
                      note='Wall duration and CPU limits are recorded; CPU consumption and full reserved-device cost are unknown.'),
                  parent_references=[dict(path=str(parent_path), sha256=file_sha(parent_path),
                      successful_requests=parent['states']['observed'], usage=parent['usage'],
                      successful_card_hours=parent['card_hours'], rejected_attempts=prior,
                      all_attempt_requests=parent_requests, all_attempt_card_hours=parent_hours)],
                  lineage_known_totals=dict(requests=current['states']['observed'] + parent_requests,
                      card_hours=current['card_hours'] + parent_hours,
                      note='Known generation plus parent rejected-attempt intervals; excludes unknown pre-model reservation and CPU cost.'),
                  historical_predecessor_costs='Complete parent success and rejected attempts referenced separately; never charged as current inference.')
    return result


def result_interface(successor, terminal_path, archival_complete):
    require(archival_complete is True, 'archive not complete')
    terminal = read(terminal_path)
    states = terminal['states']
    require(states['unknown'] == 0 and states['absent'] == 0, 'unresolved or absent requests')
    require(set(successor['manifests']) == set('ABC'), 'three explicit manifests required')
    summary = read(successor['summary'])
    require(summary['cost']['runtime_receipt']['states'] == states, 'score/cost inventory differs')
    paths = [successor[k] for k in ('scores', 'summary', 'predictions')]
    paths += list(successor['manifests'].values()) + [str(terminal_path)]
    pins = {str(p): file_sha(p) for p in paths}
    complete = not terminal['HTTP_errors'] and all(l['status'] == 'COMPLETE' for l in terminal['lanes'])
    return dict(schema='s1-paired-result-interface-v1', status='READY', successor=successor,
                terminal=dict(path=str(terminal_path), status='COMPLETE' if complete else 'FAILED_TERMINAL',
                              archival_complete=True), pins=pins)


def notify_watcher(receipt_path):
    """Use pidfd after boot/start/argv validation; never signal a reused PID."""
    receipt = read(receipt_path)
    require(Path('/proc/sys/kernel/random/boot_id').read_text().strip() == receipt['boot_id'], 'watcher boot changed')
    pid = receipt['pid']
    fd = os.pidfd_open(pid)
    try:
        stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        argv = Path(f'/proc/{pid}/cmdline').read_bytes().rstrip(bytes([0])).split(bytes([0]))
        require(stat[0] != 'Z' and stat[19] == str(receipt['proc_start_ticks']), 'watcher PID identity changed')
        require([a.decode() for a in argv] == receipt['argv'], 'watcher argv changed')
        signal.pidfd_send_signal(fd, signal.SIGUSR1)
    finally:
        os.close(fd)
    return dict(status='SIGUSR1_SENT_IDENTITY_VERIFIED', pid=pid, receipt_sha256=file_sha(receipt_path))

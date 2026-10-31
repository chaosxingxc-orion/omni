"""Prepare the complete R01C retrieval-encoding package without encoder contact.

Derive from the accepted Fold B encoding protocol. Preserve the frozen encoder,
geometry and window policy; bind the actual 625-row inventory and canonical lock.
Draft publication is create-only and does not modify the exposure ledger.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
for directory in (REPO / 'src', REPO / 'scripts/flights/nx'):
    sys.path.insert(0, str(directory))
import fold_a_encoding_write_registration as predecessor
from speech_aware_evidence_acquisition.core.atomic_file import publish_new
from speech_aware_evidence_acquisition.core.preparation_speech_queries import segment_plan

RUN = 'SAEA-SLURP-STAGE3-CONSTRUCTION-R01C-SPEECH-QUERY-ENCODING-NPU-PORT-V1'
SESSION = 'local-nemotron-cpu-slurp-tfrl-r01c-query-npu-port-v1'
ROSTER = 'docs/receipts/rosters/slurp-stage3-construction-r01c.json'
DATASET = 'configs/dataset/slurp-speech-native-stage3-construction-r01c.json'
INVENTORY = 'coordination/slurp-tfrl-r01c-speech-query-source-inventory-npu-port-v1.json'
PROTOCOL = 'configs/preparation/slurp-tfrl-r01c-speech-query-encoding-protocol-npu-port-v1.json'
PACKAGE = 'docs/readiness/2026-10-31-slurp-tfrl-r01c-encoding-npu-port-v1'
REGISTRATION = PACKAGE + '/registration-v1.json'
AUTHORITY = 'docs/decisions/2026-10-30-owner-full-test-and-first-system-gain.md'
CHANGED = {*predecessor.EXPECTED_MOVED_LEAVES, 'expected_unique_spans', 'cache_relpath'}


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return predecessor.sha256_bytes(path.read_bytes())


def within(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('input path escapes the data root')
    return path


def validate_inventory(repo, data, protocol):
    roster, dataset, inventory = read(repo / ROSTER), read(repo / DATASET), read(data / INVENTORY)
    ids = [item['sample_id'] for item in roster['items']]
    if (roster['roster_id'] != 'SLURP-STAGE3-CONSTRUCTION-R01C' or len(ids) != 625
            or len(set(ids)) != 625 or any(not sid.startswith('slurp:train:') for sid in ids)):
        raise ValueError('exact complete TRAIN R01C roster required')
    for rel in ('slurp-evo-fold-a-v2.json', 'slurp-evo-fold-b-v2.json'):
        forbidden = read(repo / 'docs/receipts/rosters' / rel)['items']
        utterances = {':'.join(row['sample_id'].split(':')[:3]) for row in forbidden}
        if any(':'.join(sid.split(':')[:3]) in utterances for sid in ids):
            raise ValueError('R01C must be utterance-disjoint from Fold A and Fold B')
    if (inventory.get('schema') != 'saea-speech-query-source-inventory-v1'
            or inventory.get('count') != 625 or inventory.get('carrier_lock_key') != 'slurp'
            or inventory.get('split_name') != 'STAGE3-CONSTRUCTION-R01C' or inventory.get('split_role') != 'discovery'
            or inventory.get('split_identity_hash') != roster['split_identity_hash']
            or inventory.get('roster_sha256') != sha(repo / ROSTER)
            or inventory.get('roster_receipt') != {'relpath': ROSTER, 'sha256': sha(repo / ROSTER)}
            or any(inventory.get(key) != 0 for key in ('model_contacts', 'tokenizer_contacts', 'network_contacts', 'audio_decodes'))):
        raise ValueError('inventory is not the complete model-free R01C source')
    pin = inventory['dataset_config']
    if pin.get('relpath') != DATASET or pin.get('sha256') != sha(repo / DATASET):
        raise ValueError('inventory dataset hash differs')
    rows = inventory['rows']
    by_id = {row['sample_id']: row for row in dataset['items']}
    if (len(rows) != 625 or {r['id'] for r in rows} != set(ids)
            or len(by_id) != 625 or set(by_id) != set(ids)
            or dataset['split_identity_hash'] != roster['split_identity_hash']):
        raise ValueError('inventory, runtime view and roster must contain the same unique recordings')
    spans = set()
    for row in rows:
        original = by_id[row['id']]
        if (row['audio_sha256'] != original['audio_sha256'] or row['media_relpath'] != original['media_relpath']
                or type(row['source_frames']) is not int or row['source_frames'] <= 0
                or type(row['source_sample_rate_hz']) is not int or row['source_sample_rate_hz'] <= 0
                or row['source_sample_rate_hz'] != original['sample_rate_hz']
                or abs(row['source_frames'] / row['source_sample_rate_hz'] - original['audio_seconds']) > 0.01):
            raise ValueError('source geometry or media identity differs from the runtime view')
        within(data, row['media_relpath'])
        for span in segment_plan(row['source_frames'], row['source_sample_rate_hz'],
                                 max_seconds=protocol['max_seconds'], overlap_seconds=protocol['overlap_seconds']):
            spans.add((row['audio_sha256'], row['source_frames'], row['source_sample_rate_hz'], span['start_frame'], span['end_frame']))
    total = sum(row['source_frames'] / row['source_sample_rate_hz'] for row in rows)
    if not math.isfinite(inventory['audio_seconds_total']) or not math.isclose(total, inventory['audio_seconds_total'], abs_tol=1e-6):
        raise ValueError('inventory audio total differs from source frames')
    return roster, inventory, len(spans), sum((end - start) / rate for _, _, rate, start, end in spans)


def derive(repo, data, lock_path):
    source_path = repo / predecessor.FOLD_B_FRAGMENT
    if sha(source_path) != predecessor.FOLD_B_FRAGMENT_SHA256:
        raise ValueError('accepted Fold B encoding protocol changed')
    if sha(lock_path) != predecessor.LIVE_LOCK_SHA256:
        raise ValueError('canonical dataset lock changed; review its successor first')
    base = read(source_path)
    roster, inventory, contacts, seconds = validate_inventory(repo, data, base)
    encoder = within(data, base['encoder']['path'].removeprefix('${SAEA_DATA_ROOT}/'))
    if sha(encoder) != base['encoder']['sha256']:
        raise ValueError('frozen encoder inventory differs')
    spec = copy.deepcopy(base)
    spec.update(roster_sha256=sha(repo / ROSTER), expected_unique_spans=contacts,
        sources={'path': '${SAEA_DATA_ROOT}/' + INVENTORY, 'sha256': sha(data / INVENTORY)},
        output_relpath=f'prepared/{RUN}/index.json', query_rows_relpath=f'prepared/{RUN}/queries.jsonl',
        cache_relpath='prepared/slurp-tfrl-r01c-speech-query-span-cache-npu-port-v1')
    if set(predecessor.moved_leaves(base, spec)) != CHANGED:
        raise ValueError('protocol differs beyond the declared roster/output/cache leaves')
    raw = predecessor.canonical(spec)
    digest = predecessor.sha256_bytes(raw)
    plan = {'schema': 'saea-execution-plan-v1', 'study': 'speech-aware-evidence-acquisition',
        'run_id': RUN, 'attempt_id': 'encode-v1', 'carrier_lock_key': 'slurp',
        'execution_profile': 'bounded-discovery-probe', 'split_role': 'discovery',
        'split_identity_hash': roster['split_identity_hash'], 'protocol_hash': digest,
        'planned_model_calls': contacts, 'planned_speech_audio_seconds': math.ceil(seconds), 'planned_gpu_hours': 2.0}
    cells = ['2026-10-31', RUN, 'bounded-discovery-probe', 'R01C frozen retrieval-query encoding; no answering core or scoring',
        'Frozen local Omni-Embed-Nemotron-3B CPU fp32 causal encoder', 'slurp TRAIN R01C construction',
        'discovery', roster['split_identity_hash'], 'no', str(contacts), str(math.ceil(seconds)), '2.0',
        'Historical construction exposure retained; regenerate current-port query vectors.', REGISTRATION,
        f'DRAFT; no contact before concrete registration under {AUTHORITY}. Protocol SHA256 {digest}. One attempt; no replay.']
    lock = {'path': str(lock_path), 'sha256': sha(lock_path)}
    entry = {'execution_plan': plan, 'session_id': SESSION,
        'protocol': {'relpath': PROTOCOL, 'sha256': digest}, 'roster_receipt': {'relpath': ROSTER, 'sha256': sha(repo / ROSTER)},
        'dataset_config': {'relpath': DATASET, 'sha256': sha(repo / DATASET)}, 'source_inventory': spec['sources'],
        'actual_source_audio_seconds': inventory['audio_seconds_total'], 'window_audio_seconds': seconds,
        'output_relpath': spec['output_relpath'], 'query_rows_relpath': spec['query_rows_relpath'], 'ledger_row_cells': cells}
    registration = {'schema': 'saea-prospective-tool-registration-v1', 'status': 'DRAFT-NOT-REGISTERED',
        'authority': AUTHORITY, 'plans': [entry], 'umbrella_lock': lock, 'encoder': spec['encoder'],
        'runtime_entry': 'python -m speech_aware_evidence_acquisition.core.speech_preparation_runtime',
        'source_protocol': {'relpath': predecessor.FOLD_B_FRAGMENT, 'sha256': sha(source_path)},
        'changed_protocol_leaves': sorted(CHANGED), 'actual_npu_hours': 0,
        'limits': ['Two hours is a wall-time ceiling, not estimated or measured runtime.',
            'Existing encoder pool is cores 128-191; it overlaps current serving/client cores. Schedule at a released boundary or register a separate CPU inventory.',
            'The encoder runtime acquires the global generation lease; do not race the Direct/Agent chain.',
            'This package does not admit model generation, bank updates, TFRL iteration or a readiness flip.']}
    header = '| ' + ' | '.join(predecessor.LEDGER_COLUMNS) + ' |\n| ' + ' | '.join(['---'] * 15) + ' |\n'
    ledger = '# R01C encoding draft exposure row\n\nNot appended to the live ledger.\n\n' + header + '| ' + ' | '.join(cells) + ' |\n'
    return {PROTOCOL: raw, REGISTRATION: predecessor.canonical(registration), PACKAGE + '/encoding-ledger-row-draft-v1.md': ledger.encode()}


def publish(repo, outputs):
    # Check every conflict before publishing any file. A partial identical draft can finish.
    for relative, blob in outputs.items():
        path = repo / relative
        if path.exists() and path.read_bytes() != blob:
            raise ValueError('existing draft differs; preserve it and register a successor')
    for relative, blob in outputs.items():
        path = repo / relative
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            publish_new(path, blob)
            path.chmod(0o644)


def main():
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    assert_execution_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=REPO)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--lock-path', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    outputs = derive(args.repo_root, args.data_root, args.lock_path)
    if not args.dry_run:
        publish(args.repo_root, outputs)
    print(json.dumps({'files': {p: predecessor.sha256_bytes(blob) for p, blob in outputs.items()},
        'rows': 625, 'dry_run': args.dry_run, 'new_contacts': 0, 'registered': False}))


if __name__ == '__main__':
    main()

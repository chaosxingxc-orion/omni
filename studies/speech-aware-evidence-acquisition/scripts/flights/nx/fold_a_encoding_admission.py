"""Register the unspent Fold A encoder under the owner's D3 decision.

Keep the earlier draft bytes intact. This tool prepares a separate registration
and exposure row, then builds a clean-commit runtime config. It never starts an
encoder: the caller must wait for the Direct/Agent process and lease boundary.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
for extra in (REPO / 'src', Path(__file__).parent):
    sys.path.insert(0, str(extra))
import fold_a_encoding_write_registration as draft
from speech_aware_evidence_acquisition.core.atomic_file import publish_new
from speech_aware_evidence_acquisition.core.preparation_speech_queries import segment_plan

PACKAGE = 'docs/readiness/2026-10-31-fold-a-encoding-admission-npu-v1'
REGISTRATION = PACKAGE + '/registration-v1.json'
AUTHORITY = 'docs/decisions/2026-10-30-owner-full-test-and-first-system-gain.md'
LEDGER = 'docs/exposure-ledger.md'


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return draft.sha256_bytes(Path(path).read_bytes())


def clean_head(repo):
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo).strip():
        raise ValueError('encoder registration/config requires a clean study checkout')
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()


def unspent(repo, data):
    for folder in ('attempts', 'preparation-runs', 'prepared', 'runs'):
        if (data / folder / draft.FOLD_A_RUN).exists():
            raise ValueError('Fold A encoding identity already has artifacts; inspect without replay')


def validated_inputs(repo, data, lock):
    outputs = draft.derive_all(repo, data, lock)
    if any((repo / rel).read_bytes() != blob for rel, blob in outputs.items()):
        raise ValueError('original Fold A admission trio differs from its derivation')
    spec = read(repo / draft.FOLD_A_FRAGMENT)
    inventory = read(data / draft.INVENTORY_RELPATH)
    roster = read(repo / draft.ROSTER_RECEIPT)
    dataset = read(repo / draft.DATASET_CONFIG)
    ids = [row['sample_id'] for row in roster['items']]
    rows = inventory['rows']
    by_id = {row['sample_id']: row for row in dataset['items']}
    if (len(ids) != 1080 or len(set(ids)) != 1080 or any(not sid.startswith('slurp:train:') for sid in ids)
            or len(rows) != 1080 or {r['id'] for r in rows} != set(ids)
            or len(dataset['items']) != 1080 or set(by_id) != set(ids)):
        raise ValueError('complete unique TRAIN Fold A roster, source inventory and view required')
    spans = set()
    for row in rows:
        source = by_id[row['id']]
        if (row['audio_sha256'] != source['audio_sha256'] or row['media_relpath'] != source['media_relpath']
                or row['source_sample_rate_hz'] != source['sample_rate_hz']
                or abs(row['source_frames'] / row['source_sample_rate_hz'] - source['audio_seconds']) > 0.01):
            raise ValueError('Fold A source audio identity or geometry changed')
        if not (data / row['media_relpath']).resolve().is_relative_to(data.resolve()):
            raise ValueError('Fold A audio path escapes the data root')
        for window in segment_plan(row['source_frames'], row['source_sample_rate_hz'],
                                   max_seconds=spec['max_seconds'], overlap_seconds=spec['overlap_seconds']):
            spans.add((row['audio_sha256'], row['source_sample_rate_hz'], window['start_frame'], window['end_frame']))
    entry = read(repo / draft.FOLD_A_REGISTRATION)['plans'][0]
    plan = entry['execution_plan']
    seconds = sum((end - start) / rate for _, rate, start, end in spans)
    if (len(spans) != spec['expected_unique_spans'] or len(spans) != plan['planned_model_calls']
            or seconds > plan['planned_speech_audio_seconds']):
        raise ValueError('Fold A encoder span count or audio budget differs')
    encoder_path = Path(spec['encoder']['path'].replace('${SAEA_DATA_ROOT}', str(data)))
    if not encoder_path.resolve().is_relative_to(data.resolve()) or sha(encoder_path) != spec['encoder']['sha256']:
        raise ValueError('frozen CPU encoder inventory differs')
    encoder = read(encoder_path)
    if (encoder['schema'], encoder['workers'], encoder['threads_per_worker'], encoder['first_core']) != (
            'saea-cpu-encoder-inventory-v1', 4, 16, 128):
        raise ValueError('registered CPU pool geometry changed')
    return entry, spec


def build_registration(repo, data, lock, *, head, at):
    unspent(repo, data)
    if (repo / REGISTRATION).exists():
        raise ValueError('Fold A successor already registered; do not replay')
    ledger = (repo / LEDGER).read_bytes()
    if f'| {draft.FOLD_A_RUN} |' in ledger.decode():
        raise ValueError('Fold A encoding already has a ledger row')
    entry, spec = validated_inputs(repo, data, lock)
    cells = list(entry['ledger_row_cells'])
    cells[0], cells[13] = at[:10], REGISTRATION
    cells[14] = (f'Registered before contact by root under {AUTHORITY}; recorded at {at}. '
        f'Protocol SHA256 {entry["protocol"]["sha256"]}. One encode-v1 attempt; no retry, fallback or scoring. '
        'CPU fp32, frozen causal geometry, 4 workers x16 threads on128-191; '
        'wait for complete Direct/Agent success, parent exit and global lease release. '
        'planned_gpu_hours=2 is the wall-time ceiling; actual NPU hours=0. '
        'This records current authority, not a historical owner signature.')
    registered = copy.deepcopy(entry)
    registered['ledger_row_cells'] = cells
    document = {'schema': 'saea-prospective-tool-registration-v1', 'status': 'REGISTERED-BEFORE-CONTACT',
        'authority': {'relpath': AUTHORITY, 'sha256': sha(repo / AUTHORITY)},
        'registered_at_utc': at, 'registration_base_commit': head,
        'source_draft': {'relpath': draft.FOLD_A_REGISTRATION, 'sha256': sha(repo / draft.FOLD_A_REGISTRATION)},
        'plans': [registered], 'umbrella_lock': {'path': str(lock), 'sha256': sha(lock)},
        'encoder': spec['encoder'], 'actual_npu_hours': 0,
        'launch_boundary': 'Agent successor done, exact parent exited, no generation lease; no concurrent serving requests.',
        'claims_system_lock': False}
    header = '| ' + ' | '.join(draft.LEDGER_COLUMNS) + ' |\n| ' + ' | '.join(['---'] * 15) + ' |\n'
    row = '| ' + ' | '.join(cells) + ' |\n'
    addition = '\n\n## Fold A encoding under the initial-system gain decision\n\n' + header + row
    return {REGISTRATION: draft.canonical(document), LEDGER: ledger + addition.encode()}


def runtime_config(repo, data, *, head):
    from speech_aware_evidence_acquisition.contracts import ExecutionPlan
    from speech_aware_evidence_acquisition.contracts.tool_plan import assert_registered_tool_plan

    unspent(repo, data)
    registered = read(repo / REGISTRATION)
    lock = Path(registered['umbrella_lock']['path'])
    entry, _ = validated_inputs(repo, data, lock)
    actual = registered['plans'][0]
    if ({k: v for k, v in actual.items() if k != 'ledger_row_cells'}
            != {k: v for k, v in entry.items() if k != 'ledger_row_cells'}):
        raise ValueError('registered encoder entry differs from verified frozen draft')
    if registered['status'] != 'REGISTERED-BEFORE-CONTACT' or registered['authority'] != {
            'relpath': AUTHORITY, 'sha256': sha(repo / AUTHORITY)}:
        raise ValueError('Fold A registration authority differs')
    config = {'plan': actual['execution_plan'], 'session_id': actual['session_id'], 'study_head': head,
        'protocol': {'path': str(repo / actual['protocol']['relpath']), 'sha256': actual['protocol']['sha256']},
        'umbrella_lock': registered['umbrella_lock']}
    assert_registered_tool_plan(source_root=repo, exposure_ledger=repo / LEDGER,
        umbrella_lock=lock, umbrella_lock_sha256=config['umbrella_lock']['sha256'],
        protocol_path=Path(config['protocol']['path']), protocol_sha256=config['protocol']['sha256'],
        plan=ExecutionPlan(**config['plan']))
    return config


def main():
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    assert_execution_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['register', 'config'])
    parser.add_argument('--repo-root', type=Path, default=REPO)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--lock-path', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    repo, data, lock = args.repo_root.resolve(), args.data_root.resolve(), args.lock_path.resolve()
    head = clean_head(repo)
    if args.stage == 'register':
        outputs = build_registration(repo, data, lock, head=head, at=datetime.now(timezone.utc).isoformat())
        if not args.dry_run:
            target = repo / REGISTRATION
            target.parent.mkdir(parents=True, exist_ok=True)
            publish_new(target, outputs[REGISTRATION])
            target.chmod(0o644)
            # A partial write is visible and refuses replay; inspect before recovery.
            (repo / LEDGER).write_bytes(outputs[LEDGER])
        print(json.dumps({'files': {key: draft.sha256_bytes(blob) for key, blob in outputs.items()},
            'dry_run': args.dry_run, 'encoder_contacts': 0}))
    else:
        config = runtime_config(repo, data, head=head)
        blob = draft.canonical(config)
        if not args.dry_run:
            if args.out is None:
                parser.error('config publication requires --out')
            out = args.out.resolve()
            if not out.is_relative_to(data / 'coordination') or out.is_relative_to(repo):
                raise ValueError('runtime config must be external under owned coordination')
            out.parent.mkdir(parents=True, exist_ok=True)
            publish_new(out, blob)
        print(json.dumps({'config_sha256': draft.sha256_bytes(blob), 'study_head': head,
            'dry_run': args.dry_run, 'encoder_contacts': 0}))


if __name__ == '__main__':
    main()

"""Build Fold A aliases and content vectors from the current sealed CPU encoding.

No model, tokenizer, scorer, audio decoding or network contact. Existing partial
outputs require reconciliation; this entry point never replays a preparation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
RUN = 'SAEA-SLURP-TFRL-FOLD-A-SPEECH-QUERY-ENCODING-NPU-PORT-V1'
ALIAS_RUN = 'SAEA-SLURP-INTENT-BOUNDARY-TRAIN-FOLD-A-ALIAS-NPU-PORT-V1'
CONTENT_RUN = 'SAEA-SLURP-FOLD-A-INPUTS-PREPARE-NPU-PORT-V1'
PROTOCOL = 'configs/preparation/slurp-tfrl-fold-a-speech-query-encoding-protocol-npu-port-v1.json'
ROSTER = 'docs/receipts/rosters/slurp-evo-fold-a-v2.json'
DATASET = 'configs/dataset/slurp-speech-native-evo-fold-a-1080.json'
RECEIPT = f'preparation-runs/{RUN}/encode-v1/receipt.json'
SEGMENTS = 'vectors/slurp-intent-boundaries-train-fold-a-1080-npu-port-v1/query-vectors-speech-segments.jsonl'
CONTENT = 'vectors/slurp-stage3-filler-fold-a-npu-port-v1/query-vectors-speech-content-v1.jsonl'
ALIAS_RECEIPT = f'prepared/{ALIAS_RUN}/query-dimension-alias.json'
PROVENANCE = str(Path(CONTENT).with_suffix('.provenance.json'))
PARENT = 'work/codex/fold-a-encoding-after-agent-v1'


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def raw(document):
    return (json.dumps(document, indent=2, sort_keys=True) + '\n').encode()


def confined(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('preparation path escapes owned root')
    return path


def verify_rows(rows, sources, dimension):
    identity = lambda row: {key: row[key] for key in
        ('id', 'audio_sha256', 'source_frames', 'source_sample_rate_hz')}
    expected = {row['id']: identity(row) for row in sources}
    actual = {row['id']: identity(row) for row in rows}
    if len(rows) != 1080 or len(actual) != len(rows) or actual != expected:
        raise ValueError('exact complete Fold A source identities required')
    for row in rows:
        segments = row['segments']
        if len(segments) != 1:
            raise ValueError('content derivative requires one complete speech segment')
        segment = segments[0]
        if segment['start_frame'] != 0 or segment['end_frame'] != row['source_frames']:
            raise ValueError('query segment does not cover the complete recording')
        if set(segment['vectors']) != {'content'}:
            raise ValueError('source query dimensions changed')
        vector = segment['vectors']['content']
        if len(vector) != dimension or any(isinstance(v, bool) or not isinstance(v, (float, int))
                or not math.isfinite(v) for v in vector):
            raise ValueError('source query vector geometry or numeric values differ')


def source_config(repo, data, encoder_expected):
    """Validate the completed encoder before producing either derivative."""
    status = read(data / PARENT / 'status.json')
    if status['state'] != 'done':
        raise ValueError('encoder follower has not completed')
    if read(data / PARENT / 'formal.exit.json')['exit'] != 0:
        raise ValueError('encoder formal invocation did not succeed')
    complete = read(data / PARENT / 'complete.json')
    receipt_path = data / RECEIPT
    if (complete['rows'] != 1080 or complete['run_id'] != RUN
            or Path(complete['receipt']).resolve() != receipt_path.resolve()
            or complete['receipt_sha256'] != sha(receipt_path)):
        raise ValueError('encoder completion receipt differs')
    expected = read(encoder_expected)
    started = read(data / PARENT / 'started.json')
    if started['expected'] != expected or status['pid'] != started['pid']:
        raise ValueError('encoder predecessor identity differs')
    receipt = read(receipt_path)
    protocol = read(repo / PROTOCOL)
    for key, value in {
        'status': 'SEALED', 'run_id': RUN, 'attempt_id': 'encode-v1',
        'study_head': expected['source_commit'], 'protocol_sha256': sha(repo / PROTOCOL),
        'encoder_inventory_sha256': protocol['encoder']['sha256'],
        'source_inventory_sha256': protocol['sources']['sha256'],
    }.items():
        if receipt.get(key) != value:
            raise ValueError('encoder receipt binding differs: ' + key)
    if read(data / 'attempts' / RUN / 'encode-v1.json')['state'] != 'FINALIZED':
        raise ValueError('encoder attempt not finalized')
    query_path = confined(data, protocol['query_rows_relpath'])
    index_path = confined(data, protocol['output_relpath'])
    if (Path(receipt['query_rows_path']).resolve() != query_path
            or sha(query_path) != receipt['query_rows_sha256']
            or sha(index_path) != receipt['index_sha256']):
        raise ValueError('encoder output seal changed')
    source_path = confined(data, protocol['sources']['path'].removeprefix('${SAEA_DATA_ROOT}/'))
    if sha(source_path) != protocol['sources']['sha256']:
        raise ValueError('encoder source inventory changed')
    rows = [json.loads(line) for line in query_path.read_text().splitlines()]
    verify_rows(rows, read(source_path)['rows'], protocol['dimension'])
    if read(index_path)['output']['rows'] != rows:
        raise ValueError('encoder index does not match query export')
    roster = read(repo / ROSTER)
    if (sha(repo / ROSTER) != protocol['roster_sha256']
            or {row['id'] for row in rows} != {row['sample_id'] for row in roster['items']}):
        raise ValueError('query rows do not match frozen Fold A roster')
    config = {
        'schema': 'saea-sealed-segment-vector-dimension-alias-v1',
        'source': {'kind': 'sealed-segment-vectors', 'dimension': 'content',
                   'relpath': protocol['query_rows_relpath'], 'sha256': sha(query_path)},
        'target_dimension': 'intent-boundary', 'expected_rows': 1080,
        'vector_width': protocol['dimension'], 'output_relpath': SEGMENTS,
        'receipt_relpath': ALIAS_RECEIPT,
        'source_receipts': [{'root': scope, 'relpath': name, 'sha256': sha(base / name)}
                            for scope, base, name in (('data', data, RECEIPT),
                            ('repository', repo, ROSTER), ('repository', repo, DATASET))],
    }
    return config, rows


def prepare(repo, data, encoder_expected, out, source_commit):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.atomic_file import publish_new
    from speech_aware_evidence_acquisition.core.preparation_vectors import prepare_vector_dimension_alias
    sys.path.insert(0, str(repo / 'scripts'))
    from build_slurp_query_content_derivative import build_query_content_derivative

    assert_execution_environment()
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip() != source_commit
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo).strip()):
        raise ValueError('preparation source must be the pinned clean commit')
    out = confined(data, str(out))
    if out.is_relative_to(repo.resolve()) or out.exists():
        raise ValueError('fresh external preparation output required')
    for name in (SEGMENTS, CONTENT, ALIAS_RECEIPT, PROVENANCE):
        if confined(data, name).exists():
            raise ValueError('existing derivative output; inspect without replay')
    config, rows = source_config(repo, data, encoder_expected)
    out.mkdir(parents=True)
    publish_new(out / 'intent.json', raw({'source_commit': source_commit,
        'encoder_expected_sha256': sha(encoder_expected), 'alias_config': config,
        'model_contacts': 0, 'encoder_contacts': 0}))
    try:
        alias = prepare_vector_dimension_alias(config, repo_root=repo, data_root=data)
        result = build_query_content_derivative(data, source_relpath=SEGMENTS,
            source_sha256=sha(data / SEGMENTS), out_dir_relpath=str(Path(CONTENT).parent),
            out_name=Path(CONTENT).name, ledger_row=CONTENT_RUN, check=False)
        flat = [json.loads(line) for line in (data / CONTENT).read_text().splitlines()]
        if (len(flat) != 1080 or {row['id']: row['vectors']['content'] for row in flat}
                != {row['id']: row['segments'][0]['vectors']['content'] for row in rows}):
            raise ValueError('content derivative changed source vector values')
        if (alias['output']['values_unchanged'] is not True
                or alias['output']['max_absolute_delta'] != 0
                or result['row_count'] != 1080):
            raise ValueError('derivative validation differs')
        pins = {name: sha(data / name) for name in
                (RECEIPT, config['source']['relpath'], SEGMENTS, ALIAS_RECEIPT, CONTENT, PROVENANCE)}
        record = {'schema': 'saea-fold-a-retrieval-inputs-v1', 'source_commit': source_commit,
            'rows': 1080, 'pins': pins, 'model_contacts': 0, 'encoder_contacts': 0,
            'tokenizer_contacts': 0, 'scorer_contacts': 0, 'values_unchanged': True,
            'claims_system_lock': False,
            'next_required': 'Close model-free preparation ledger rows, derive the current A0 comparator and build agreement/supply before Fold A evaluation.'}
        publish_new(out / 'complete.json', raw(record))
        return record
    except BaseException as error:
        publish_new(out / 'failed.json', raw({'reason': str(error), 'automatic_retry': False}))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root', type=Path, default=REPO)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--encoder-expected', type=Path, required=True)
    p.add_argument('--source-commit', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    result = prepare(a.repo_root.resolve(), a.data_root.resolve(), a.encoder_expected, a.out, a.source_commit)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()

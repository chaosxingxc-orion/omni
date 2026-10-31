"""Adopt sealed isolated R01C encoding with model-free shared vector builders.

Verify the pinned encoder source, registration, terminal attempt and every vector
before writing. No encoder retry or implicit TFRL flight admission is performed.
"""
import argparse
from pathlib import Path
import sys
import subprocess

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / 'src'), str(Path(__file__).parent)]
from fold_a_retrieval_inputs import read, sha, raw, confined, verify_rows
from r01c_cpu_overlap import RUN, PROTOCOL, REGISTRATION
from r01c_encoding_write_registration import ROSTER, DATASET, validate_inventory

RECEIPT = f'preparation-runs/{RUN}/encode-v1/receipt.json'
VECTOR_DIR = 'vectors/slurp-stage3-construction-r01c-npu-port-v1'
SEGMENTS = VECTOR_DIR + '/query-vectors-speech-segments.jsonl'
CONTENT = VECTOR_DIR + '/query-vectors-speech-content-v1.jsonl'
ALIAS_RECEIPT = 'prepared/SAEA-SLURP-STAGE3-CONSTRUCTION-R01C-ALIAS-NPU-PORT-V1/query-dimension-alias.json'
CONTENT_RUN = 'SAEA-SLURP-STAGE3-CONSTRUCTION-R01C-INPUTS-PREPARE-NPU-PORT-V1'
PROVENANCE = str(Path(CONTENT).with_suffix('.provenance.json'))


def source_config(encoder_repo, data, runtime_config, runtime_sha, exit_path):
    if sha(runtime_config) != runtime_sha:
        raise ValueError('encoder runtime config pin differs')
    config = read(runtime_config)
    if (subprocess.check_output(['git', '-c', 'safe.directory=' + str(encoder_repo), 'rev-parse', 'HEAD'], cwd=encoder_repo, text=True).strip()
            != config['study_head'] or subprocess.check_output(
                ['git', '-c', 'safe.directory=' + str(encoder_repo), 'status', '--porcelain'], cwd=encoder_repo).strip()):
        raise ValueError('encoder source is not the pinned clean commit')
    if exit_path.read_text().strip() != '0':
        raise ValueError('encoder invocation not successfully complete')
    protocol_path = encoder_repo / PROTOCOL
    protocol = read(protocol_path)
    if (Path(config['protocol']['path']).resolve() != protocol_path.resolve()
            or config['protocol']['sha256'] != sha(protocol_path)):
        raise ValueError('encoder protocol binding differs')
    registration = read(encoder_repo / REGISTRATION)
    if config['plan'] != registration['plans'][0]['execution_plan']:
        raise ValueError('registered encoder plan differs')
    plan = config['plan']
    receipt = read(data / RECEIPT)
    for key, expected in {'status': 'SEALED', 'run_id': RUN, 'attempt_id': 'encode-v1',
            'protocol_sha256': sha(protocol_path), 'study_head': config['study_head'],
            'encoder_inventory_sha256': protocol['encoder']['sha256'],
            'source_inventory_sha256': protocol['sources']['sha256'],
            'cpu_isolation': protocol['cpu_isolation'], 'logical_encoder_contacts': 625}.items():
        if receipt.get(key) != expected:
            raise ValueError('encoder receipt binding differs: ' + key)
    if (plan['run_id'] != RUN or plan['attempt_id'] != 'encode-v1'
            or plan['protocol_hash'] != sha(protocol_path)
            or read(data / 'attempts' / RUN / 'encode-v1.json')['state'] != 'FINALIZED'):
        raise ValueError('encoder attempt identity or finalization differs')
    for field in ('sources', 'encoder'):
        path = confined(data, protocol[field]['path'].removeprefix('${SAEA_DATA_ROOT}/'))
        if sha(path) != protocol[field]['sha256']:
            raise ValueError('encoder input pin differs: ' + field)
    roster, inventory, spans, _ = validate_inventory(encoder_repo, data, protocol)
    if (spans != 625 or protocol['expected_unique_spans'] != 625
            or protocol['roster_sha256'] != sha(encoder_repo / ROSTER)
            or plan['split_identity_hash'] != roster['split_identity_hash']):
        raise ValueError('encoder roster identity differs')
    query = confined(data, protocol['query_rows_relpath'])
    index = confined(data, protocol['output_relpath'])
    if (Path(receipt['query_rows_path']).resolve() != query
            or sha(query) != receipt['query_rows_sha256'] or sha(index) != receipt['index_sha256']):
        raise ValueError('encoder output seal changed')
    import json
    rows = [json.loads(line) for line in query.read_text().splitlines()]
    verify_rows(rows, inventory['rows'], protocol['dimension'], expected_count=625)
    if read(index)['output']['rows'] != rows:
        raise ValueError('encoder index differs from query export')
    alias = {'schema': 'saea-sealed-segment-vector-dimension-alias-v1',
        'source': {'kind': 'sealed-segment-vectors', 'dimension': 'content',
            'relpath': protocol['query_rows_relpath'], 'sha256': sha(query)},
        'target_dimension': 'intent-boundary', 'expected_rows': 625,
        'vector_width': protocol['dimension'], 'output_relpath': SEGMENTS,
        'receipt_relpath': ALIAS_RECEIPT,
        'source_receipts': [{'root': scope, 'relpath': name, 'sha256': sha(base / name)}
            for scope, base, name in [('data', data, RECEIPT),
                ('repository', encoder_repo, ROSTER), ('repository', encoder_repo, DATASET),
                ('repository', encoder_repo, PROTOCOL), ('repository', encoder_repo, REGISTRATION)]]}
    return alias, rows


def prepare(encoder_repo, data, runtime_config, runtime_sha, exit_path, out):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.atomic_file import publish_new
    from speech_aware_evidence_acquisition.core.preparation_vectors import prepare_vector_dimension_alias
    sys.path.insert(0, str(REPO / 'scripts'))
    from build_slurp_query_content_derivative import build_query_content_derivative
    assert_execution_environment()
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO).strip():
        raise ValueError('derivative implementation must be committed and clean')
    tool_head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    out = confined(data, str(out))
    if not out.is_relative_to((data / 'work/codex').resolve()) or out.exists():
        raise ValueError('fresh owned external output required')
    for name in (SEGMENTS, CONTENT, ALIAS_RECEIPT, PROVENANCE):
        if confined(data, name).exists():
            raise ValueError('existing derivative; inspect partial or complete result without replay')
    alias_config, rows = source_config(encoder_repo, data, runtime_config, runtime_sha, exit_path)
    out.mkdir(parents=True)
    publish_new(out / 'intent.json', raw({'tool_head': tool_head,
        'runtime_config_sha256': runtime_sha, 'alias_config': alias_config, 'new_model_contacts': 0}))
    try:
        alias = prepare_vector_dimension_alias(alias_config, repo_root=encoder_repo, data_root=data)
        result = build_query_content_derivative(data, source_relpath=SEGMENTS,
            source_sha256=sha(data / SEGMENTS), out_dir_relpath=VECTOR_DIR,
            out_name=Path(CONTENT).name, ledger_row=CONTENT_RUN, check=False)
        import json
        flat = [json.loads(line) for line in (data / CONTENT).read_text().splitlines()]
        if (len(flat) != 625 or {r['id']: r['vectors']['content'] for r in flat}
                != {r['id']: r['segments'][0]['vectors']['content'] for r in rows}
                or alias['output']['values_unchanged'] is not True
                or alias['output']['max_absolute_delta'] != 0 or result['row_count'] != 625):
            raise ValueError('derivative changed source vectors')
        record = {'schema': 'saea-r01c-retrieval-inputs-v1', 'rows': 625,
            'tool_head': tool_head, 'encoder_run_id': RUN, 'runtime_config_sha256': runtime_sha,
            'pins': {name: sha(data / name) for name in
                (RECEIPT, alias_config['source']['relpath'], SEGMENTS, CONTENT, ALIAS_RECEIPT, PROVENANCE)},
            'values_unchanged': True, 'new_model_contacts': 0, 'new_encoder_contacts': 0,
            'claims_tfrl_admitted': False,
            'next_required': 'Bind the actual isolated encoder receipt and minus-R01 pool, build agreement and own supply; retain A0-first admission.'}
        publish_new(out / 'complete.json', raw(record))
        return record
    except BaseException as error:
        publish_new(out / 'failed.json', raw({'reason': str(error), 'automatic_retry': False}))
        raise


if __name__ == '__main__':
    import json
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('encoder-repo', 'data', 'runtime-config', 'exit-path', 'out'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--runtime-sha', required=True)
    print(json.dumps(prepare(**vars(p.parse_args()))))

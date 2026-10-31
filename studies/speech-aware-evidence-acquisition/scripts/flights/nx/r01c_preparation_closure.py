"""Verify completed R01C preparation and export a source-preserving ledger closure.

Original encoder registration remains immutable. Derivative ledger omissions are
reported retrospectively, never relabelled as prospective registrations.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / 'src'), str(Path(__file__).parent)]
import r01c_retrieval_inputs as inputs
import r01c_current_agreement as agreement
from speech_aware_evidence_acquisition.core.atomic_file import publish_new
read, sha, raw, confined = inputs.read, inputs.sha, inputs.raw, inputs.confined
PACKAGE = 'docs/readiness/2026-10-31-r01c-preparation-closure-v1'
REPORT = PACKAGE + '/closure-v1.json'
ALIAS_RUN = 'SAEA-SLURP-STAGE3-CONSTRUCTION-R01C-ALIAS-NPU-PORT-V1'


def verify_derivatives(data, source_rows, retrieval, agreement_complete, ids):
    if (retrieval['schema'] != 'saea-r01c-retrieval-inputs-v1' or retrieval['rows'] != 625
            or retrieval['encoder_run_id'] != inputs.RUN or not retrieval['values_unchanged']
            or retrieval['new_model_contacts'] != 0 or retrieval['new_encoder_contacts'] != 0):
        raise ValueError('completed model-free R01C derivative required')
    required = {inputs.RECEIPT, inputs.SEGMENTS, inputs.CONTENT, inputs.ALIAS_RECEIPT, inputs.PROVENANCE}
    if not required <= set(retrieval['pins']):
        raise ValueError('incomplete derivative pins')
    for rel, digest in retrieval['pins'].items():
        if sha(confined(data, rel)) != digest:
            raise ValueError('derivative pin changed')
    source = {r['id']: r['segments'][0]['vectors']['content'] for r in source_rows}
    segments = [json.loads(x) for x in (data / inputs.SEGMENTS).read_text().splitlines()]
    flat = [json.loads(x) for x in (data / inputs.CONTENT).read_text().splitlines()]
    if (len(source_rows) != 625 or len(source) != 625 or set(source) != set(ids)
            or len(segments) != 625 or len(flat) != 625
            or {r['id']: r['segments'][0]['vectors']['intent-boundary'] for r in segments} != source
            or {r['id']: r['vectors']['content'] for r in flat} != source):
        raise ValueError('derivative values or complete roster differ')
    alias = read(data / inputs.ALIAS_RECEIPT)
    if (alias['contact_class'] != 'model-free' or alias['output']['row_count'] != 625
            or alias['output']['output_sha256'] != sha(data / inputs.SEGMENTS)
            or alias['output']['values_unchanged'] is not True
            or alias['output']['max_absolute_delta'] != 0
            or any(alias['output'][k] != 0 for k in ('encoder_contacts', 'model_contacts', 'tokenizer_contacts'))):
        raise ValueError('alias receipt differs')
    provenance = read(data / inputs.PROVENANCE)
    if (provenance['exposure_ledger_row'] != inputs.CONTENT_RUN
            or provenance['row_count'] != 625 or provenance['parent_sha256'] != sha(data / inputs.SEGMENTS)
            or provenance['sha256'] != sha(data / inputs.CONTENT)):
        raise ValueError('content provenance differs')
    if (agreement_complete['status'] != 'complete' or agreement_complete['rows'] != 625
            or agreement_complete['artifact_id'] != agreement.ARTIFACT
            or agreement_complete['model_contacts'] != 0
            or agreement_complete['artifact_sha256'] != sha(data / agreement.OUTPUT / 'agreement.jsonl')
            or agreement_complete['receipt_sha256'] != sha(data / agreement.OUTPUT / 'receipt.json')):
        raise ValueError('agreement completion differs')
    return agreement.verify_artifact(data, read(data / agreement.OUTPUT / 'receipt.json'), ids)


def closed_encoder_cells(original, receipt_sha):
    cells = copy.deepcopy(original)
    if len(cells) != 15 or cells[1] != inputs.RUN or cells[8] != 'no' or cells[9:12] != ['625', '2727', '2.0']:
        raise ValueError('exact original unconsumed registration required')
    cells[8] = 'yes'
    cells[14] += (' COMPLETED: 625 encoder contacts, 2726.8471875 audio seconds, 0 NPU hours; '
                  'original prospective registration preserved. SEALED receipt SHA256 ' + receipt_sha + '; closure ' + REPORT + '.')
    return cells


def export(encoder_repo, data, runtime_config, runtime_sha, exit_path, retrieval_complete, agreement_completion, out):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from r01c_encoding_write_registration import predecessor
    assert_execution_environment()
    out = confined(data, str(out))
    if not out.is_relative_to(data.resolve() / 'work/codex') or out.exists():
        raise ValueError('fresh owned closure output required')
    alias_config, source_rows = inputs.source_config(encoder_repo, data, runtime_config, runtime_sha, exit_path)
    retrieval, agreed = read(retrieval_complete), read(agreement_completion)
    if retrieval['runtime_config_sha256'] != runtime_sha:
        raise ValueError('retrieval encoder runtime differs')
    ids = [x['sample_id'] for x in read(encoder_repo / inputs.ROSTER)['items']]
    rows = verify_derivatives(data, source_rows, retrieval, agreed, ids)
    original = read(encoder_repo / inputs.REGISTRATION)['plans'][0]['ledger_row_cells']
    original_line = '| ' + ' | '.join(original) + ' |'
    if (encoder_repo / 'docs/exposure-ledger.md').read_text().splitlines().count(original_line) != 1:
        raise ValueError('original prospective encoder row missing or duplicated')
    cells = closed_encoder_cells(original, sha(data / inputs.RECEIPT))
    derivative_rows = []
    for run, detail in [(ALIAS_RUN, '625 sealed segment vectors; values unchanged'),
                        (inputs.CONTENT_RUN, '625 content vectors; values unchanged'),
                        (agreement.ARTIFACT, '625 self-excluded agreement rows; no labels read')]:
        derivative_rows.append(['2026-10-31', run, 'model-free-check', detail,
            'Local arithmetic on sealed R01C inputs', 'slurp TRAIN R01C construction', 'discovery', original[7],
            'yes', '0', '0', '0.0', 'Original encoder exposure retained; no new model or encoder contacts.', REPORT,
            'COMPLETED; retrospective ledger synchronization from pinned preparation evidence. '
            'This closure does not assert prior ledger registration or admit generation.'])
    pins = dict(retrieval['pins'])
    for rel in (agreement.OUTPUT + '/receipt.json', agreement.OUTPUT + '/agreement.jsonl'):
        pins[rel] = sha(data / rel)
    report = {'schema': 'saea-r01c-preparation-closure-v1', 'rows': 625, 'new_model_contacts': 0,
        'encoder_source_commit': read(runtime_config)['study_head'],
        'registration_sha256': sha(encoder_repo / inputs.REGISTRATION),
        'protocol_sha256': sha(encoder_repo / inputs.PROTOCOL), 'runtime_config_sha256': runtime_sha,
        'retrieval_completion_sha256': sha(retrieval_complete), 'agreement_completion_sha256': sha(agreement_completion),
        'pins': pins, 'derivatives_values_unchanged': True, 'agreement_rows': len(rows),
        'ledger_rows': [inputs.RUN, ALIAS_RUN, inputs.CONTENT_RUN, agreement.ARTIFACT],
        'derivative_registration_scope': 'Retrospective ledger synchronization; original preparation evidence remains immutable.',
        'generation_admitted': False}
    header = '| ' + ' | '.join(predecessor.LEDGER_COLUMNS) + ' |\n| ' + ' | '.join(['---'] * 15) + ' |\n'
    table = '\n## Verified R01C preparation closure\n\n' + header
    table += '\n'.join('| ' + ' | '.join(row) + ' |' for row in [cells, *derivative_rows]) + '\n'
    outputs = {inputs.REGISTRATION: (encoder_repo / inputs.REGISTRATION).read_bytes(),
               inputs.PROTOCOL: (encoder_repo / inputs.PROTOCOL).read_bytes(), REPORT: raw(report),
               PACKAGE + '/ledger-addition.md': table.encode()}
    out.mkdir(parents=True)
    for rel, blob in outputs.items():
        path = out / rel; path.parent.mkdir(parents=True, exist_ok=True); publish_new(path, blob)
    publish_new(out / 'complete.json', raw({'files': {rel: sha(out / rel) for rel in outputs},
        'rows': 625, 'generation_admitted': False, 'new_model_contacts': 0}))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('encoder-repo', 'data', 'runtime-config', 'exit-path', 'retrieval-complete', 'agreement-completion', 'out'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--runtime-sha', required=True)
    print(json.dumps(export(**vars(p.parse_args()))))

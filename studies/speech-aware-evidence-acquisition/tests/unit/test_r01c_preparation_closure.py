import copy
import json
from pathlib import Path
import sys
import pytest
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/flights/nx'))
import r01c_preparation_closure as m


def test_encoder_closure_keeps_original_authority_and_budget():
    original = ['2026-10-31', m.inputs.RUN, 'profile', 'scope', 'encoder', 'train', 'discovery',
                'a'*64, 'no', '625', '2727', '2.0', 'history', 'registration', 'countersigned']
    closed = m.closed_encoder_cells(original, 'b'*64)
    assert original[8] == 'no' and closed[8] == 'yes'
    assert closed[:8] == original[:8] and closed[9:14] == original[9:14]
    assert closed[14].startswith(original[14])
    original[9] = '624'
    with pytest.raises(ValueError): m.closed_encoder_cells(original, 'b'*64)


def fixture(data):
    ids = [f'slurp:train:fixture:{i}' for i in range(625)]
    source = [{'id': i, 'segments': [{'vectors': {'content': [float(n), 1.]}}]} for n,i in enumerate(ids)]
    segments = [{'id': r['id'], 'segments': [{'vectors': {'intent-boundary': r['segments'][0]['vectors']['content']}}]} for r in source]
    flat = [{'id': r['id'], 'vectors': {'content': r['segments'][0]['vectors']['content']}} for r in source]
    def write(rel, value, lines=False):
        p = data / rel; p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(''.join(json.dumps(x)+'\n' for x in value) if lines else json.dumps(value))
    write(m.inputs.RECEIPT, {})
    write(m.inputs.SEGMENTS, segments, True); write(m.inputs.CONTENT, flat, True)
    write(m.inputs.ALIAS_RECEIPT, {'contact_class':'model-free', 'output':{'row_count':625,
        'output_sha256':m.sha(data/m.inputs.SEGMENTS), 'values_unchanged':True, 'max_absolute_delta':0,
        'encoder_contacts':0, 'model_contacts':0, 'tokenizer_contacts':0}})
    write(m.inputs.PROVENANCE, {'exposure_ledger_row':m.inputs.CONTENT_RUN, 'row_count':625,
        'parent_sha256':m.sha(data/m.inputs.SEGMENTS), 'sha256':m.sha(data/m.inputs.CONTENT)})
    write(m.agreement.OUTPUT+'/receipt.json', {})
    write(m.agreement.OUTPUT+'/agreement.jsonl', [])
    retrieval = {'schema':'saea-r01c-retrieval-inputs-v1','rows':625, 'encoder_run_id':m.inputs.RUN,
        'values_unchanged':True,'new_model_contacts':0,'new_encoder_contacts':0,
        'pins':{r:m.sha(data/r) for r in (m.inputs.RECEIPT,m.inputs.SEGMENTS,m.inputs.CONTENT,m.inputs.ALIAS_RECEIPT,m.inputs.PROVENANCE)}}
    agreed = {'status':'complete','rows':625,'artifact_id':m.agreement.ARTIFACT,'model_contacts':0,
        'artifact_sha256':m.sha(data/m.agreement.OUTPUT/'agreement.jsonl'),
        'receipt_sha256':m.sha(data/m.agreement.OUTPUT/'receipt.json')}
    return source, retrieval, agreed, ids


@pytest.mark.parametrize('fault', [None, 'value', 'duplicate', 'missing_pin', 'contact', 'changed_pin', 'agreement'])
def test_closure_revalidates_content_before_accepting_receipts(tmp_path, monkeypatch, fault):
    source, retrieval, agreed, ids = fixture(tmp_path)
    monkeypatch.setattr(m.agreement, 'verify_artifact', lambda *_: ids)
    if fault == 'value': source[0]['segments'][0]['vectors']['content'][0] = -1.
    elif fault == 'duplicate': source[-1] = copy.deepcopy(source[0])
    elif fault == 'missing_pin': retrieval['pins'].pop(m.inputs.SEGMENTS)
    elif fault == 'contact': retrieval['new_encoder_contacts'] = 1
    elif fault == 'changed_pin': (tmp_path/m.inputs.CONTENT).write_text('[]')
    elif fault == 'agreement': agreed['rows'] = 624
    if fault:
        with pytest.raises(ValueError): m.verify_derivatives(tmp_path, source, retrieval, agreed, ids)
    else:
        assert len(m.verify_derivatives(tmp_path, source, retrieval, agreed, ids)) == 625

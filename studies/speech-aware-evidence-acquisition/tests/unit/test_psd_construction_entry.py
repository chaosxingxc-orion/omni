import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/analysis'))
from intent_slot_workload import NodeWorkload


def test_existing_construction_roster_and_pinned_query_identity(tmp_path):
    roster_path = 'docs/receipts/rosters/slurp-stage3-construction-r01c.json'
    dataset = 'slurp-speech-native-stage3-construction-r01c'
    view_path = f'configs/dataset/{dataset}.json'
    roster = json.loads((ROOT / roster_path).read_text())
    view = json.loads((ROOT / view_path).read_text())
    ids = [r['sample_id'] for r in roster['items']]
    def write(root, path, value):
        p = root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value))
        return {'relpath': path, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
    rp = write(tmp_path, roster_path, roster)
    vp = write(tmp_path, view_path, view)
    qpath = tmp_path / 'queries.jsonl'
    qpath.write_text(''.join(json.dumps({'id': r['sample_id'], 'audio_sha256': r['audio_sha256'], 'source_sample_rate_hz': r['sample_rate_hz']}) + '\n' for r in view['items']))
    query = {'relpath': 'queries.jsonl', 'sha256': hashlib.sha256(qpath.read_bytes()).hexdigest(), 'kind': 'sealed-segment-vectors'}
    source = {'policy': 'phase-native-module-v1-c2', 'knowledge_stores': {
        'intent-boundaries': {'phase_supply': {'perception': {'query_source': query}}},
        'train-exemplars': {'phase_supply': {'solver': {'query_source': query}}}}}
    sp = write(tmp_path, 'source.json', source)
    material = {'schema': 'saea-intent-slot-fold-material-v1', 'source_baseline': sp, 'source_roster': rp, 'dataset': vp}
    mp = write(tmp_path, 'material.json', material)
    config = {'schema':'saea-intent-slot-workload-v1', 'name':'construction-test', 'run_id':'CONSTRUCTION-TEST', 'sample_ids':ids, 'runtime_binding':{}, 'execution':{'admitted_level':1}, 'material':mp, 'node_arm':'enhanced-p-slot-decision', 'population':{'role':'construction','dataset':dataset,'dataset_sha256':vp['sha256'],'roster':rp,'source_baseline':sp}}
    job = NodeWorkload(config)
    _, checked = job.material_checked(tmp_path, tmp_path, mp)
    assert [r['sample_id'] for r in checked['view']['items']] == ids
    assert len(ids) == 625
    qpath.write_text(qpath.read_text() + '{}\n')
    with pytest.raises(ValueError, match='query source digest'):
        job.material_checked(tmp_path, tmp_path, mp)

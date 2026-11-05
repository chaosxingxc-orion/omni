import copy,json,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/analysis'))
from intent_slot_workload import NodeWorkload
from intent_slot_read import protocol_status

@pytest.mark.parametrize('fold',['a','b'])
def test_fixed_fold_is_complete_and_source_pinned(fold):
    config=json.loads((ROOT/f'configs/evaluation/fold-baseline-v1/fold-{fold}-workload.json').read_text())
    job=NodeWorkload(config)
    assert job.ROWS==1080 and job.CONTACTS==3
    assert config['sample_ids']==[r['sample_id'] for r in json.loads((ROOT/f'configs/dataset/slurp-speech-native-evo-fold-{fold}-1080.json').read_text())['items']]
    bad=copy.deepcopy(config);bad['sample_ids'].append(bad['sample_ids'][0])
    with pytest.raises(ValueError):NodeWorkload(bad)

def test_refusal_and_storage_complete_are_not_protocol_complete():
    assert protocol_status({'role_visits':[],'terminal_disposition':{'reason':'seal_refusal'}},{})=='protocol_refusal'
    assert protocol_status({'role_visits':[]},{})=='incomplete_protocol'

import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts' / 'analysis'))
from task_geometry_mask import masks, tokens


def test_tokens_and_rare_and_no_empty():
    assert tokens('PLAY-ing café_23') == {'play', 'ing', 'café', '23'}
    members = [{'intent': 'large', 'utterance_id': str(i)} for i in range(33)] + [{'intent': 'rare', 'utterance_id': '33'}]
    rows = {str(i): {'sentence': 'unrelated'} for i in range(34)}
    cards = {x: {'facet': {'representation': x + ' unique'}} for x in ('large', 'rare', 'empty')}
    top = np.tile(np.array([0, 1, 2, 3, 4]), (34, 1))
    result, fallback, _, _ = masks(members, rows, cards, top, [8], [3], 32)
    assert all(v.all() for v in result.values())
    assert fallback['joint_df8_peer3'] == ['large']
    assert fallback['geometric_peer3'] == []

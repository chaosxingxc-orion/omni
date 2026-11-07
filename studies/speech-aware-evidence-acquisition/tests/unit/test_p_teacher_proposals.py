import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / 'scripts/analysis'))
from p_teacher_proposals import source_roles, card_patch, sha
import copy


def test_counterexamples_keep_all_roles_and_require_success():
    groups = [{'query_id': q, 'evidence': {'reference_feedback_private': {'gold_candidate_recall': ok}}}
              for q, ok in [('success', True), ('failure', False)]]
    roles = source_roles(['failure', 'success'], groups)
    assert roles == {'all_counterexample_query_ids': ['failure', 'success'],
                     'protected_success_query_ids': ['success'], 'failure_query_ids': ['failure']}
    with pytest.raises(ValueError):
        source_roles(['failure'], groups)
    with pytest.raises(ValueError):
        source_roles(['success', 'unknown'], groups)


def _permutation():
    sections = ['Intent: demo', 'Activation boundary: α', 'Confusable exclusions: beta',
                'Uncertainty behavior: gamma', 'Limitations: delta']
    old = '\n'.join(sections)
    order = [0, 2, 1, 3, 4]
    new = '\n'.join(sections[i] for i in order)
    card = {'id': 'card', 'source_uri': 'source', 'content': old, 'content_sha256': sha(old.encode())}
    refs = []
    for i in order:
        a = old.index(sections[i]); c = new.index(sections[i])
        refs.append({'card_id': 'card', 'source_uri': 'source', 'content_sha256': card['content_sha256'],
                     'old_start': a, 'old_end': a + len(sections[i]), 'new_start': c, 'new_end': c + len(sections[i])})
    return card, {'operation': 'REORDER_EXISTING_SECTIONS', 'target_id': 'card', 'old': old,
                  'new': new, 'source_refs': refs, 'semantic_equivalence': 'permutation'}


def test_section_separator_normalization_keeps_wire_and_proof():
    card, op = _permutation()
    original = copy.deepcopy(op)
    result = card_patch(op, {'card': card})
    assert op == original and result['source_refs'] == original['source_refs']
    assert result['old_content_sha256'] == sha(op['old'].encode())
    assert result['new_content_sha256'] == sha(op['new'].encode())
    assert sum(x['excluded_separator'] for x in result['derived_section_proof']['sections']) == 4


def test_equal_gap_count_cannot_move_header_separator_to_another_boundary():
    card, op = _permutation()
    refs = op['source_refs']
    # Include the second section's original LF, then shift the header's gap
    # after it.  The old count-only check saw three gaps and three exclusions.
    refs[1]['old_end'] += 1
    pieces = [op['old'][r['old_start']:r['old_end']] for r in refs]
    gaps = ['', '\n', '\n', '\n']
    new = ''
    for i, (ref, piece) in enumerate(zip(refs, pieces)):
        ref['new_start'] = len(new)
        new += piece
        ref['new_end'] = len(new)
        if i < len(gaps):
            new += gaps[i]
    op['new'] = new
    assert new.startswith(pieces[0] + pieces[1])
    assert sum(bool(g) for g in gaps) == 3
    with pytest.raises(ValueError, match='section boundary'):
        card_patch(op, {'card': card})


@pytest.mark.parametrize('damage', ['omit', 'duplicate', 'overlap', 'insert', 'modify', 'cross_card', 'sha', 'offset', 'header'])
def test_section_permutation_rejects_damage(damage):
    card, op = _permutation(); refs = op['source_refs']
    if damage == 'omit': refs.pop()
    elif damage == 'duplicate': refs[-1] = copy.deepcopy(refs[-2])
    elif damage == 'overlap': refs[1]['new_start'] -= 1
    elif damage == 'insert': op['new'] += '!'
    elif damage == 'modify': op['new'] = op['new'].replace('beta', 'zeta')
    elif damage == 'cross_card': refs[1]['card_id'] = 'other'
    elif damage == 'sha': refs[1]['content_sha256'] = '0' * 64
    elif damage == 'offset': refs[1]['old_start'] += 1
    elif damage == 'header': refs[0]['new_start'] = 1
    with pytest.raises(ValueError): card_patch(op, {'card': card})

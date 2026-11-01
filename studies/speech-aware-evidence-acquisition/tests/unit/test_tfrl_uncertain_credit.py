"""Behavioral checks for learning eligibility distinct from rule publication."""
from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts/analysis'))
from tfrl_uncertain_credit import propose, ARMS


def score(intent=1, tp=1, fp=0, fn=0):
    return {'scenario_action_exact': intent,
            **{f'slu_f1_{kind}_{stat}': value for kind in ('word', 'char')
               for stat, value in [('tp', tp), ('fp', fp), ('fn', fn)]},
            **{f'slot_f1_exact_{stat}': value for stat, value in [('tp', tp), ('fp', fp), ('fn', fn)]}}


def arms(**overrides):
    return {a: overrides.get(a, score()) for a in ARMS}


def test_unstable_error_is_retained_without_publishing_lucky_candidate():
    result = propose('x', arms(lock=score(fn=1)))
    assert result['learning_candidate']
    assert result['control_score_varies']
    assert result['contrast_candidates'] == []  # equals aa, only beats lock
    assert result['next_action'] == 'diagnose_or_repeat'
    assert not result['publishable_rule']
    assert result['independent_full_system_repeats'] == 0


def test_unstable_controls_can_support_a_contrast_but_not_a_rule():
    result = propose('x', arms(lock=score(fn=2), aa=score(fn=1)))
    assert result['learning_candidate'] and result['contrast_candidates']
    assert result['next_action'] == 'cross_fold_validate_contrast'
    assert not result['confirmed_hard_example'] and not result['publishable_rule']


def test_no_correct_candidate_still_preserves_learning_problem():
    result = propose('x', {a:score(intent=0) for a in ARMS})
    assert result['learning_candidate']
    assert result['next_action'] == 'diagnose_or_repeat'


def test_intent_only_empty_slot_repair_is_not_lost_to_f1_tie():
    result = propose('x', {a:score(intent=int(a.startswith('s')),tp=0) for a in ARMS})
    assert result['contrast_candidates'][0]['intent_repair_above_both_controls']
    assert result['contrast_candidates'][0]['f1_gain_above_control_max'] == 0


def test_easy_controls_are_not_added_for_candidate_noise():
    result = propose('x', arms(s1=score(intent=0)))
    assert not result['learning_candidate']
    assert result['next_action'] == 'keep_for_regression'


def test_better_slot_score_with_wrong_intent_is_not_positive():
    result = propose('x', {a:score(intent=0) if a.startswith('s') else score(fn=1) for a in ARMS})
    assert result['contrast_candidates'] == []


@pytest.mark.parametrize('bad', [float('nan'), -1, True])
def test_invalid_counts_refused(bad):
    values = arms()
    values['aa']['slot_f1_exact_fn'] = bad
    with pytest.raises(ValueError): propose('x', values)


def test_missing_control_and_extra_arm_refused():
    for values in ({a:score() for a in ARMS if a!='aa'}, {**arms(), 'vote':score()}):
        with pytest.raises(ValueError): propose('x', values)

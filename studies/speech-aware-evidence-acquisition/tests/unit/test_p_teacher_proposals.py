import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / 'scripts/analysis'))
from p_teacher_proposals import source_roles


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

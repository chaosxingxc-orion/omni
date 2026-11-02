"""The recorded counterexample audit must refuse silently changed inputs."""
import pytest
from scripts.analysis.review_conditioned_h1 import review


def test_changed_candidate_is_refused_before_reading_semantics(tmp_path):
    paths = [tmp_path / name for name in ('candidate', 'material', 'roster')]
    for path in paths:
        path.write_text('{}')
    with pytest.raises(ValueError, match='candidate or H1 roster changed'):
        review(*paths)

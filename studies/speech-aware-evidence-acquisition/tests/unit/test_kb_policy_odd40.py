import pytest
from speech_aware_evidence_acquisition.core.base import KNOWLEDGE_FUSERS
from speech_aware_evidence_acquisition.rails.errors import RailsError
from speech_aware_evidence_acquisition.rails.kb_policy import ODD40_FUSE_POLICY_ID,FuseKnobs,fuse_odd40_factory,fuse_supplyfilter_factory
from test_kb_policy_target_cap import _store,_decide


def test_rank_selection_not_identifier_position():
    snapshot=_store(['x']*40);order=list(reversed(range(40)))
    result=_decide(fuse_odd40_factory(FuseKnobs(0,40,40,40)),snapshot,order=order)
    assert result.selected_value_ids==tuple(sorted(f'value:{i:03d}' for i in order[::2]))
    assert result.fusion_policy_id==ODD40_FUSE_POLICY_ID
    assert len(result.supporting_facet_ids)==20
    assert KNOWLEDGE_FUSERS.resolve(ODD40_FUSE_POLICY_ID)==fuse_odd40_factory


def test_short_recall_falls_back_exact_original20():
    snapshot=_store(['x']*39);order=list(reversed(range(39)))
    actual=_decide(fuse_odd40_factory(FuseKnobs(0,40,40,40)),snapshot,order=order)
    original=_decide(fuse_supplyfilter_factory(FuseKnobs(0,20,20,20)),snapshot,order=order)
    assert actual.selected_value_ids==original.selected_value_ids


def test_visible_values_are_never_reintroduced():
    snapshot=_store(['x']*40)
    actual=_decide(fuse_odd40_factory(FuseKnobs(0,40,40,40)),snapshot,order=list(range(40)),retrieved=('value:000',))
    assert 'value:000' not in actual.selected_value_ids
    assert len(actual.selected_value_ids)==19


@pytest.mark.parametrize('knobs',[None,FuseKnobs(0,20,20,20),FuseKnobs(.1,40,40,40)])
def test_other_interventions_rejected(knobs):
    with pytest.raises(RailsError):fuse_odd40_factory(knobs)

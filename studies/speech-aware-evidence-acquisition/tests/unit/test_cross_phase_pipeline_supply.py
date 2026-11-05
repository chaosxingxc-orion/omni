from dataclasses import replace
from types import SimpleNamespace
import pytest
from speech_aware_evidence_acquisition.rails.knowledge import QueryState
from speech_aware_evidence_acquisition.rails.runner import _restrict_pipeline_cross_phase_supply, RailsRunnerError
from speech_aware_evidence_acquisition.rails.supply import SupplyPacket, SupplyItem, SupplyDose, SupplyForm
from test_phase_supply_repeat_visit import _exemplar_snapshot, _prepared_supply, EXEMPLAR_VALUE_IDS

def context(snapshot, ids):
    q=QueryState(query_id='query-id',query_sample_id='query',speech_packet_id='packet',speech_packet_sha256='a'*64,use_split_identity_hash='c'*64, retrieved_value_ids=tuple(ids))
    return SimpleNamespace(query_state=q,knowledge_snapshot_sha256=snapshot.content_hash())

def test_cross_phase_existing_and_new_are_bound_and_ordered():
    s=_exemplar_snapshot();ctx=context(s,EXEMPLAR_VALUE_IDS[:3]);sup=_prepared_supply(s)('solver',ctx)
    out=_restrict_pipeline_cross_phase_supply(sup,ctx,s,s.content_hash())
    assert out.fusion_decision.selected_value_ids==EXEMPLAR_VALUE_IDS[3:]
    assert out.rank_result==sup.rank_result
    assert out.fusion_decision.supporting_facet_ids==sup.fusion_decision.supporting_facet_ids
    assert out.fusion_decision.rank_result_sha256==out.rank_result.content_hash()

def test_all_visible_does_not_acquire_twice():
    s=_exemplar_snapshot();ctx=context(s,EXEMPLAR_VALUE_IDS);sup=_prepared_supply(s)('solver',ctx)
    assert _restrict_pipeline_cross_phase_supply(sup,ctx,s,s.content_hash()) is None

def test_wrong_snapshot_or_conflicting_packet_rejected():
    s=_exemplar_snapshot();ctx=context(s,EXEMPLAR_VALUE_IDS[:1]);sup=_prepared_supply(s)('solver',ctx)
    with pytest.raises(RailsRunnerError,match='snapshot identity'):
        _restrict_pipeline_cross_phase_supply(sup,ctx,s,'f'*64)
    packet=SupplyPacket(form=SupplyForm.TEXT,dose=SupplyDose(n_items=1,token_count=1),prompt_block='wrong',assembly_template_id='test',items=(SupplyItem(item_id='x',text='wrong',evidence_id=EXEMPLAR_VALUE_IDS[0]),))
    with pytest.raises(RailsRunnerError,match='content differs'):
        _restrict_pipeline_cross_phase_supply(replace(sup,packet=packet),ctx,s,s.content_hash())


def test_different_content_same_id_snapshot_rejected():
    s=_exemplar_snapshot();ctx=context(s,EXEMPLAR_VALUE_IDS[:1]);sup=_prepared_supply(s)('solver',ctx)
    altered=replace(s,values=(replace(s.values[0],content='different'),)+s.values[1:])
    # Runtime calculates the identity from actual snapshot once at startup.
    with pytest.raises(RailsRunnerError,match='snapshot identity'):
        _restrict_pipeline_cross_phase_supply(sup,ctx,altered,altered.content_hash())

def test_rank_and_facet_conflict_rejected_before_filtering():
    s=_exemplar_snapshot();ctx=context(s,EXEMPLAR_VALUE_IDS[:1]);sup=_prepared_supply(s)('solver',ctx)
    with pytest.raises(Exception,match='RankResult'):
        _restrict_pipeline_cross_phase_supply(replace(sup,fusion_decision=replace(sup.fusion_decision,rank_result_sha256='f'*64)),ctx,s,s.content_hash())

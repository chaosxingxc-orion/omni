from __future__ import annotations

import copy

import pytest

from tfrl_evidence_card_consumer import EvidenceCardError, consume_evidence_cards


REQUEST = {"intent": "weather_query", "slots": ["place_name"]}
LOCK = {"intent": "weather_query", "entities": []}


def card(source="b", answer=None, evidence=None):
    return {
        "source": source,
        "intent": "weather_query",
        "slot": "place_name",
        "relation": REQUEST,
        "evidence": ["physical-place"] if evidence is None else evidence,
        "answer": {"intent": "weather_query", "entities": [{"filler": "Paris"}]}
        if answer is None else answer,
    }


def test_orders_cards_by_source_and_does_not_mutate_input():
    cards = [card("z"), card("a")]
    before = copy.deepcopy(cards)
    result = consume_evidence_cards(cards, request_relation=REQUEST, lock_answer=LOCK)
    assert result["mode"] == "conditioned"
    assert [row["source"] for row in result["evidence_cards"]] == ["a", "z"]
    assert cards == before


@pytest.mark.parametrize("cards,reason", [([], "no-matching-evidence"), ([card(evidence=[])], "missing-evidence")])
def test_missing_evidence_falls_back_to_lock(cards, reason):
    result = consume_evidence_cards(cards, request_relation=REQUEST, lock_answer=LOCK)
    assert result["mode"] == "lock" and result["fallback_reason"] == reason
    assert result["answer"] == LOCK


def test_wrong_relation_is_ignored_without_cartesian_expansion():
    other = card()
    other["relation"] = {"intent": "weather_query", "slots": ["date"]}
    result = consume_evidence_cards([other], request_relation=REQUEST, lock_answer=LOCK)
    assert result["mode"] == "lock"


def test_conflicting_duplicate_cards_fall_back():
    result = consume_evidence_cards(
        [card(answer={"value": "a"}), card(answer={"value": "b"})],
        request_relation=REQUEST,
        lock_answer=LOCK,
    )
    assert result["fallback_reason"] == "conflicting-evidence"


def test_malformed_card_relation_fails_closed_at_cpu_boundary():
    bad = card()
    bad["relation"] = {"intent": "weather_query", "slots": []}
    with pytest.raises(EvidenceCardError):
        consume_evidence_cards([bad], request_relation=REQUEST, lock_answer=LOCK)

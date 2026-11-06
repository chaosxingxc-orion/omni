"""Resumed deliveries retain the raw contact and the derived identity."""
import copy
import hashlib

import pytest

from speech_aware_evidence_acquisition.core.bundle import BundleError
from speech_aware_evidence_acquisition.core.bundle.outputs import _verify_outputs_trace_binding
from speech_aware_evidence_acquisition.core.delivery import (
    native_abstention_records, native_context_abstention_records,
    native_seal_refusal_abstention_records,
)


def _case(sample, kind, parent="flight-0088"):
    h = lambda text: hashlib.sha256(text.encode()).hexdigest()
    contact = dict(agent_contact_id="agent", request_id=parent, request_sha256=h("request"),
                   response_sha256=h("response"), response_text_sha256=h("raw"))
    trace = {parent + "/request": {"arm": "speech", "payload": {"sample_id": sample},
                                    "payload_sha256": contact["request_sha256"]},
             parent + "/response": {"text": "raw", "response_sha256": contact["response_sha256"],
                                     "response_text_sha256": contact["response_text_sha256"]}}
    disposition = dict(reason="finite_completion", finish_reason="length", role_id="perception",
                       stage="perception", revision_count=0, planner_accepted=False,
                       rejected_contact=contact)
    if kind == "seal":
        disposition.pop("finish_reason")
        disposition.update(reason="seal_refusal", seal_refusal={"error_class": "Error", "message": "refused"})
        derived = native_seal_refusal_abstention_records(sample_id=sample, source_contact=contact,
                                                         seal_refusal=disposition["seal_refusal"])
    elif kind == "context":
        disposition = dict(reason="context_compaction", source="dispatch-budget")
        derived = native_context_abstention_records(sample_id=sample, disposition=disposition)
        trace = {}
    else:
        derived = native_abstention_records(sample_id=sample, source_contact=contact, finish_reason="length")
    request_id, request, response = derived
    trace.update({request_id + "/request": request, request_id + "/response": response})
    row = dict(sample_id=sample, request_id=request_id, request_sha256=request["payload_sha256"],
               text=response["text"], response_sha256=response["response_sha256"],
               response_text_sha256=response["response_text_sha256"],
               native_episode_receipt={"terminal_disposition": disposition})
    return row, trace


@pytest.mark.parametrize("kind", ["finite", "seal", "context"])
def test_multi_ancestor_derived_and_raw_answer(kind):
    row, first = _case("s1", kind)
    other, second = _case("s2", "seal")  # same request IDs, different attempt
    answer = dict(sample_id="s3", request_id="flight-1", text="answer", request_sha256="a",
                  response_sha256="b", response_text_sha256="c")
    recovery = {"flight-1/request": {"payload": {"sample_id": "s3"}, "payload_sha256": "a"},
                "flight-1/response": {"text": "answer", "response_sha256": "b", "response_text_sha256": "c"}}
    _verify_outputs_trace_binding((row, other, answer), recovery, {"s1": first, "s2": second}, frozenset({"s1", "s2"}))


@pytest.mark.parametrize("fault", ["wrong-sample", "parent-sample", "parent-hash", "parent-response",
                                  "wrong-parent", "derived-hash", "derived-response", "conflict", "wrong-attempt", "reflown", "derived-reflown"])
def test_reject_forged_or_ambiguous_custody(fault):
    row, trace = _case("s1", "seal")
    request = trace[row["request_id"] + "/request"]
    recovery = {}
    if fault == "wrong-sample": row["sample_id"] = "s2"
    elif fault == "parent-sample": trace["flight-0088/request"]["payload"]["sample_id"] = "s2"
    elif fault == "parent-hash": trace["flight-0088/request"]["payload_sha256"] = "0" * 64
    elif fault == "parent-response": trace["flight-0088/response"]["response_sha256"] = "0" * 64
    elif fault == "wrong-parent": request["source"]["source_request_id"] = "other"
    elif fault == "derived-hash": request["payload_sha256"] = "0" * 64
    elif fault == "derived-response": trace[row["request_id"] + "/response"]["text"] = "forged"
    elif fault == "conflict": request["payload"] = {"sample_id": "s1"}
    elif fault == "wrong-attempt": trace = _case("s2", "seal")[1]
    elif fault == "reflown": recovery = copy.deepcopy(trace)
    elif fault == "derived-reflown": recovery = {row["request_id"] + "/request": copy.deepcopy(request)}
    with pytest.raises(BundleError):
        _verify_outputs_trace_binding((row,), recovery, {row["sample_id"]: trace}, frozenset({row["sample_id"]}))

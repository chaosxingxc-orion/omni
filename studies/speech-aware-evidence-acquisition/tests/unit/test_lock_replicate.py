"""Noise model M1: the registered REPLICATE of lock (``arm_kind: replicate``) beside the single twin.

Layers (model-free): 1. the rails' parser accepts the twin plus up to R = 2 request-identical copies and refuses every other shape by name; 2. the loader adds the replicate's
Perception and Solver contact to the ceiling; 3. the receipt validator takes the replicate ids and refuses a mismatch; 4. the module compiler's sampling frame names the
replicate, refuses a malformed one and resolves c17 to its own module-family program (never to the flown re-listen program of the same ceiling); 5. the PORT family
(and the port derive) carry the replicate while the historical family and arm table are untouched.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
for _extra in (REPO / "scripts" / "flights" / "nx", REPO / "scripts" / "flights" / "direct", REPO / "scripts" / "flights" / "tfrl", REPO / "scripts" / "flights" / "batch",
               REPO / "scripts" / "analysis", REPO / "src"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import l338_family as fam  # noqa: E402
import l338_port_derive_cells as pdc  # noqa: E402
import l338_port_family as pf  # noqa: E402
import test_l338_sampling_variants as sv  # noqa: E402  (the sampling-column fixtures)

from speech_aware_evidence_acquisition.rails import consumption_module as cm  # noqa: E402
from speech_aware_evidence_acquisition.rails.native_episode_config import load_native_episode_config  # noqa: E402
from speech_aware_evidence_acquisition.rails.native_episode_receipt import (  # noqa: E402
    MAX_LOCK_REPLICATES,
    perception_branch_twin_and_replicates,
    validate_perception_branch_receipt,
)
from speech_aware_evidence_acquisition.rails.supply_variants import (  # noqa: E402
    SAMPLING_ARM_KIND_REPLICATE,
    PerceptionBranchError,
    parse_perception_sampling_variants,
    perception_branch_ceiling,
    perception_branch_receipt,
    sampling_extra_contacts,
)

REP = {"arm_id": "lock-rep2", "replicate_of": "lock", "arm_kind": "replicate"}


def block(*, replicates=(REP,), twin=True, candidates=None):
    rows = list(candidates if candidates is not None else sv._block(twin=False)["arms"][:-1])
    if twin:
        rows.append({"arm_id": "aa", "replicate_of": "lock"})
    rows += [copy.deepcopy(r) for r in replicates]
    rows.append({"arm_id": "lock", "deliver": True})
    return {"node": "perception", "arms": rows, "max_supplied_items": 4, "fit_estimate": dict(sv.FIT)}


def parse(**kwargs):
    return parse_perception_sampling_variants(block(**kwargs), phase_id="perception")


# --------------------------------------------------------------------------- 1. the parser


def test_the_parser_accepts_the_twin_and_one_replicate_in_declared_order():
    group = parse()
    assert group.branch_ids == ("l1-support", "l2-union", "aa", "lock-rep2", "lock") and group.branch_count == 5 and group.level == 2
    assert group.twin_branch.branch_id == "aa" and not group.twin_branch.is_replicate and group.twin_branch.is_primary_twin
    assert [b.branch_id for b in group.replicate_branches] == ["lock-rep2"] and group.has_twin
    rep = group.branch("lock-rep2")
    # request identity for the runtime: a replicate is a twin to every path that renders A's own request, and a B branch to none
    assert rep.is_twin and rep.is_replicate and not rep.is_primary_twin and not rep.is_b and not rep.deliver and rep.replicate_of == "lock"
    assert rep.perception_samples is None and rep.solver_samples is None and rep.knobs == () and rep.max_supplied_items == group.delivering_branch.max_supplied_items
    assert [b.branch_id for b in group.branches if b.is_b] == ["l1-support", "l2-union"]
    assert group.delivering_branch.branch_id == "lock" and group.renders_unvaried_evidence and group.is_sampling


def test_the_replicate_adds_one_perception_and_one_solver_contact_to_the_ceiling():
    plain, with_rep = parse(replicates=()), parse()
    assert sampling_extra_contacts(plain) == sampling_extra_contacts(with_rep) == 3 + 4                   # the sampling axes are the candidates' alone
    base = perception_branch_ceiling(base_ceiling=2, branch_count=plain.branch_count, level=2) + sampling_extra_contacts(plain)
    more = perception_branch_ceiling(base_ceiling=2, branch_count=with_rep.branch_count, level=2) + sampling_extra_contacts(with_rep)
    assert (base, more) == (15, 17) and more - base == 2                                                  # +1 Perception, +1 Solver
    from speech_aware_evidence_acquisition.rails.supply_variants import perception_contact_slots, solver_contact_slots

    assert len(perception_contact_slots(with_rep)) - len(perception_contact_slots(plain)) == 1
    assert len(solver_contact_slots(with_rep)) - len(solver_contact_slots(plain)) == 1


def test_a_block_without_arm_kind_parses_exactly_as_before_and_serialises_without_it():
    group = parse(replicates=())
    assert not group.replicate_branches and all("arm_kind" not in b.as_json() for b in group.branches)
    assert group.twin_branch.as_json()["replicate_of"] == "lock"
    assert parse().branch("lock-rep2").as_json()["arm_kind"] == SAMPLING_ARM_KIND_REPLICATE


def test_at_most_two_request_identical_replicates_are_admitted():
    assert MAX_LOCK_REPLICATES == 2
    third = {"arm_id": "lock-rep3", "replicate_of": "lock", "arm_kind": "replicate"}
    with pytest.raises(PerceptionBranchError, match="at most 2 request-identical replicates"):
        parse(replicates=(REP, third))
    two_twins = block(replicates=())
    two_twins["arms"].insert(-1, {"arm_id": "aa2", "replicate_of": "lock"})
    with pytest.raises(PerceptionBranchError, match="exactly one twin"):
        parse_perception_sampling_variants(two_twins, phase_id="perception")
    with pytest.raises(PerceptionBranchError, match="exactly one twin"):
        parse(twin=False)                                                                                   # a replicate does not replace the twin


@pytest.mark.parametrize("extra, message", [
    ({"seed": 7}, "unknown keys: \\['seed'\\]"),
    ({"decoding": {"temperature": 0.6}}, "unknown keys: \\['decoding'\\]"),
    ({"overrides": {"x": 1}}, "unknown keys: \\['overrides'\\]"),
    ({"deliver": True}, "both deliver: true and replicate_of"),
    ({"solver_samples": copy.deepcopy(sv.L1)}, "no sampling axis"),
    ({"perception_samples": copy.deepcopy(sv.L2)}, "no sampling axis"),
])
def test_a_replicate_that_differs_from_lock_in_anything_but_its_id_is_refused(extra, message):
    with pytest.raises(PerceptionBranchError, match=message):
        parse(replicates=(dict(REP, **extra),))


@pytest.mark.parametrize("edit, message", [
    (lambda r: r.update(replicate_of="aa"), "not the delivering arm"),
    (lambda r: r.update(arm_kind="twin"), "arm_kind 'twin' is not registered"),
    (lambda r: r.update(arm_kind="copy"), "is not registered"),
    (lambda r: r.pop("replicate_of"), "without replicate_of"),
    (lambda r: r.update(arm_id="aa"), "distinct"),
])
def test_a_malformed_replicate_is_refused_by_name(edit, message):
    row = copy.deepcopy(REP)
    edit(row)
    with pytest.raises(PerceptionBranchError, match=message):
        parse(replicates=(row,))


def test_arm_kind_on_a_candidate_or_before_the_twin_is_refused():
    candidates = sv._block(twin=False)["arms"][:-1]
    candidates[0] = dict(candidates[0], arm_kind="replicate")
    with pytest.raises(PerceptionBranchError, match="without replicate_of"):
        parse(candidates=candidates)
    early = block()
    rows = early["arms"]
    rows[2], rows[3] = rows[3], rows[2]                                                                     # the replicate before the twin
    with pytest.raises(PerceptionBranchError, match="after the twin"):
        parse_perception_sampling_variants(early, phase_id="perception")


# --------------------------------------------------------------------------- 2. the loader


def test_the_loader_takes_the_replicate_into_the_ceiling():
    group = parse()
    ceiling = perception_branch_ceiling(base_ceiling=2, branch_count=group.branch_count, level=2) + sampling_extra_contacts(group)
    config = load_native_episode_config(sv._values(block()), sv._program(ceiling))
    assert config.max_contacts == ceiling == 17 and [b.branch_id for b in config.perception_branch_group.replicate_branches] == ["lock-rep2"]
    with pytest.raises(ValueError, match="program contact ceiling must equal"):
        load_native_episode_config(sv._values(block()), sv._program(ceiling - 2))
    third = {"arm_id": "lock-rep3", "replicate_of": "lock", "arm_kind": "replicate"}
    with pytest.raises(ValueError, match="at most 2 request-identical replicates"):
        load_native_episode_config(sv._values(block(replicates=(REP, third))), sv._program(19))


# --------------------------------------------------------------------------- 3. the receipt


def _receipt(group, edit=None):
    rows = sv._rows(group)
    for row in rows:
        if row["branch_id"] in ("aa", "lock-rep2", "lock"):
            row["supplied_value_ids"], row["supplied_count"] = ["v1", "v2"], 2
    for index, mutation in (edit or {}).items():
        rows[index].update(mutation)
    return rows, perception_branch_receipt(group=group, phase_id="perception", records=rows)


def _validate(group, receipt, rows, **kwargs):
    twin, replicates = perception_branch_twin_and_replicates(rows)
    validate_perception_branch_receipt(receipt, branch_ids=list(group.branch_ids), delivered_branch_id="lock", twin_branch_id=twin, phase_id="perception", level=2,
                                       replicate_branch_ids=kwargs.get("replicates", replicates))


def test_a_v6_receipt_with_a_replicate_validates_and_the_helper_splits_twin_and_replicates():
    group = parse()
    rows, receipt = _receipt(group)
    assert perception_branch_twin_and_replicates(rows) == ("aa", ("lock-rep2",))
    assert perception_branch_twin_and_replicates([{"branch_id": "x"}]) == (None, ())
    assert [r["replicate_of"] for r in rows] == [None, None, "lock", "lock", None] and rows[3]["sampling"] is None
    _validate(group, receipt, rows)
    assert receipt["branch_count"] == 5


def test_the_validator_refuses_every_replicate_mismatch():
    group = parse()
    rows, receipt = _receipt(group)
    with pytest.raises(ValueError, match="only the declared twin branch may carry replicate_of"):
        _validate(group, receipt, rows, replicates=())                                                      # a caller that predates M1
    with pytest.raises(ValueError, match="undeclared, repeated or delivering replicate"):
        _validate(group, receipt, rows, replicates=("nope",))
    with pytest.raises(ValueError, match="undeclared, repeated or delivering replicate"):
        _validate(group, receipt, rows, replicates=("lock",))
    with pytest.raises(ValueError, match="at most 2 are registered"):
        _validate(group, receipt, rows, replicates=("lock-rep2", "l1-support"))
    with pytest.raises(ValueError, match="undeclared twin branch"):
        validate_perception_branch_receipt(receipt, branch_ids=list(group.branch_ids), delivered_branch_id="lock", twin_branch_id=None, phase_id="perception", level=2,
                                           replicate_branch_ids=("lock-rep2",))
    bad_rows, bad_receipt = _receipt(group, {3: {"supplied_value_ids": ["x", "y"]}})
    with pytest.raises(ValueError, match="does not resolve exactly the delivering branch's ids"):
        _validate(group, bad_receipt, bad_rows)
    no_link_rows, no_link = _receipt(group, {3: {"replicate_of": "aa"}})
    with pytest.raises(ValueError, match="does not replicate the delivering branch"):
        _validate(group, no_link, no_link_rows)


def test_a_receipt_without_a_replicate_validates_exactly_as_before():
    group = parse(replicates=())
    rows, receipt = _receipt(group)
    assert perception_branch_twin_and_replicates(rows) == ("aa", ())
    _validate(group, receipt, rows)
    validate_perception_branch_receipt(receipt, branch_ids=list(group.branch_ids), delivered_branch_id="lock", twin_branch_id="aa", phase_id="perception", level=2)


# --------------------------------------------------------------------------- 4. the module compiler's frame


FRAME = {"max_supplied_items": 55, "twin": True, "fit_estimate": dict(fam.SAMPLING_FIT_ESTIMATE)}


@pytest.mark.parametrize("edit, message", [
    (lambda f: f.update(replicates=[]), "1..1 distinct"),
    (lambda f: f.update(replicates=["a", "b"]), "1..1 distinct"),
    (lambda f: f.update(replicates=["a", "a"]), "1..1 distinct"),
    (lambda f: f.update(replicates="lock-rep2"), "1..1 distinct"),
    (lambda f: f.update(replicates=[""]), "1..1 distinct"),
    (lambda f: f.update(replicates=["lock-rep2"], twin=False), "requires twin: true"),
    (lambda f: f.update(extra=1), "keys must be exactly"),
])
def test_the_sampling_frame_refuses_a_malformed_replicates_entry(edit, message):
    frame = copy.deepcopy(FRAME)
    edit(frame)
    with pytest.raises(cm.ConsumptionModuleConfigError, match=message):
        cm._parse_sampling_frame(frame, has_sampling_columns=True)


def test_the_sampling_frame_without_replicates_is_unchanged_and_with_one_is_canonical():
    historical = cm._parse_sampling_frame(copy.deepcopy(FRAME), has_sampling_columns=True)
    assert "replicates" not in json.loads(historical)
    with_rep = cm._parse_sampling_frame(dict(copy.deepcopy(FRAME), replicates=["lock-rep2"]), has_sampling_columns=True)
    assert json.loads(with_rep)["replicates"] == ["lock-rep2"] and historical != with_rep


def test_the_port_cells_compile_to_c17_with_the_replicate_and_the_flown_relisten_program_is_not_chosen():
    for key in pf.ALL_KEYS:
        doc = pdc.port_baseline(key)
        block_ = doc["native_episode"]["consumption_module"]
        assert block_["sampling_frame"]["replicates"] == list(pf.REPLICATE_ARMS)
        assert doc["policy"] == pf.PROGRAM == "phase-native-module-v1-c17" != "phase-native-candidates-review-revise-relisten-v1"
        arms = doc["native_episode"]["roles"]["perception"]["sampling_variants"]["arms"]
        assert [a["arm_id"] for a in arms] == list(pf.ARM_ORDER)
        assert arms[3] == {"arm_id": "lock-rep2", "arm_kind": "replicate", "replicate_of": "lock"} and arms[2] == {"arm_id": "aa", "replicate_of": "lock"}
        assert doc["native_contact_fit"]["contact_cap"] == pf.fit_calls(key)
        group = parse_perception_sampling_variants(doc["native_episode"]["roles"]["perception"]["sampling_variants"], phase_id="perception")
        assert 2 + 2 * (group.branch_count - 1) + sampling_extra_contacts(group) == pf.CONTACTS_PER_ROW == 17


def test_the_historical_cells_still_compile_to_c15_without_a_replicate():
    import l338_derive_cells as dc

    result = dc.derive_baseline("arm", write=False)
    assert result["program"] == fam.PROGRAM == "phase-native-module-v1-c15" and result["ceiling"] == 15
    assert "replicates" not in result["block"]["sampling_frame"]
    assert [a["arm_id"] for a in result["document"]["native_episode"]["roles"]["perception"]["sampling_variants"]["arms"]] == list(fam.ARM_ORDER)


# --------------------------------------------------------------------------- 5. the port family


def test_the_port_arm_table_is_the_historical_one_plus_the_replicate_and_the_historical_family_is_untouched():
    assert pf.REPLICATE_ARMS == ("lock-rep2",) and pf.ARM_ORDER == ("l1-support", "l2-union", "aa", "lock-rep2", "lock") and pf.ARM_ORDER[-1] == fam.DELIVERING
    assert fam.ARM_ORDER == ("l1-support", "l2-union", "aa", "lock") and fam.K == 4 and fam.CONTACTS_PER_ROW == 15 and fam.PROGRAM == "phase-native-module-v1-c15"
    assert pf.REPLICATE_SET == ("lock", "aa", "lock-rep2") and pf.K == 5
    assert (pf.PERCEPTION_CONTACTS_PER_ROW, pf.SOLVER_CONTACTS_PER_ROW, pf.CONTACTS_PER_ROW) == (8, 9, 17)
    assert (fam.PERCEPTION_CONTACTS_PER_ROW, fam.SOLVER_CONTACTS_PER_ROW) == (7, 8)
    assert pf.PROGRAM == "phase-native-module-v1-c17" and pf.ARM_MIX_ROWS == 8.52 and fam.ARM_MIX_ROWS == 7.52
    assert (pf.generation_calls("arm"), pf.fit_calls("arm"), pf.audio_seconds("arm")) == (36720, 18360, 107264)
    assert (pf.generation_calls("smoke"), pf.fit_calls("smoke"), pf.audio_seconds("smoke")) == (136, 68, 347)
    assert (fam.original_generation_calls("arm"), fam.fit_contact_cap(1080), fam.audio_cap("arm")) == (32400, 64800, 94644)
    arms_json = json.loads((REPO / "scripts/flights/nx/l338_arms.json").read_text(encoding="utf-8"))
    assert "lock-rep2" not in json.dumps(arms_json) and arms_json["twin"] == "aa"


def test_the_port_sampling_block_is_parsed_by_the_rails_and_the_replicate_is_not_a_candidate():
    block_ = pf.sampling_block()
    assert [a["arm_id"] for a in block_["arms"]] == list(pf.ARM_ORDER) and block_["arms"][3] == REP
    group = pf.SAMPLING_GROUP
    assert tuple(group.branch_ids) == pf.ARM_ORDER and [b.branch_id for b in group.replicate_branches] == list(pf.REPLICATE_ARMS)
    assert pf.CANDIDATE_ARMS == fam.CANDIDATE_ARMS and "lock-rep2" not in pf.CANDIDATE_ARMS
    assert fam.sampling_block()["arms"][-2:] == [{"arm_id": "aa", "replicate_of": "lock"}, {"arm_id": "lock", "deliver": True}]


def test_the_port_signoff_conditions_expect_the_replicate_arm():
    conditions = pf.signoff_conditions()
    assert conditions["expected_arms"] == sorted(pf.ARM_ORDER) and "lock-rep2" in conditions["expected_arms"]
    assert "lock-rep2" not in fam.signoff_conditions()["expected_arms"]

"""TFRL-v2 loop adapters (``scripts/flights/nx/tfrl_v2_loop_adapters.py``, task T3b): model-free, no card.

Proved here, with fake trace lines, a fake tokenizer client and a fake resolver: the loop driver runs end to end over the new sampler and distiller
(plan per half per iteration, one bank per source half, the union as the loop state); the iteration record carries the bank receipts and the budget
line; the second iteration's plan of a half consumes the bank of the OTHER half; the loop halts while a half's flight has left no trace lines and a
re-run is a no-op; an estimate-measured session and an unknown item id are refused; the existing stub / fixture distillers leave the iteration
record untouched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
NX = REPO / "scripts" / "flights" / "nx"
for _p in (str(REPO / "src"), str(REPO / "scripts" / "analysis"), str(NX), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from speech_aware_evidence_acquisition.contracts import rollout_trace as rt  # noqa: E402
import tfrl_port_loop as pl  # noqa: E402
import tfrl_v2_common as cm  # noqa: E402
import tfrl_v2_loop_adapters as ad  # noqa: E402
import tfrl_v2_sampler as sp  # noqa: E402
import tfrl_v2_text as tx  # noqa: E402
import test_tfrl_port_loop as lt  # noqa: E402  (fixtures only, as a namespace)
import test_tfrl_v2_distiller as base  # noqa: E402  (fixtures only, as a namespace)

H1, H2 = cm.HALF_H1, cm.HALF_H2
ROWS_PER_HALF = 4


class FakeTokenizer:
    def request_json(self, method, path, payload):
        text = payload["messages"][0]["content"][0]["text"]
        n = -(-len(text) // 4) + 7
        return {"count": n, "tokens": list(range(n))}


def half_fixture(prefix):
    fx = base.Fixture()
    for i in range(ROWS_PER_HALF):
        good = base.ans("alarm", "set", [("date", "monday")])
        fx.add(f"{prefix}{i:03d}", 0.5, [(1.0, True, good, (f"{prefix}-item-{i}",)), (0.5, True, base.ans("alarm", "set"), ()),
                                        (0.5, True, base.ans("alarm", "set"), ()), (0.0, False, base.ans("alarm", "query"), ())])
    return fx


FIXTURES = {H1: half_fixture("a"), H2: half_fixture("b")}


def rosters():
    return {half: {"roster_id": cm.HALF_ROSTER_ID[half], "split_identity_hash": ("a" if half == H1 else "b") * 64,
                   "items": [{"sample_id": r, "audio_seconds": 2.0} for r in fx.rows]} for half, fx in FIXTURES.items()}


def resolver():
    ids = {item["item_id"] for fx in FIXTURES.values() for c in fx.contacts for item in c["supply_items"]}
    return tx.ItemTextResolver({i: f"{i} | text of {i} | date=monday" for i in sorted(ids)})


class FakeLines:
    """Flight lines held in memory: ``flown`` is the set of (half, iteration) whose flight has left its trace."""

    def __init__(self, flown):
        self.flown = set(flown)

    def lines(self, half, iteration):
        if (half, iteration) not in self.flown:
            return None
        return list(FIXTURES[half].contacts), list(FIXTURES[half].scores)

    def sha256s(self, half, iteration):
        return [cm.content_sha256([half, iteration])]


def session(tmp_path, flown, **kw):
    kw.setdefault("measure", tx.VllmTokenizeMeasure(FakeTokenizer()))
    return ad.V2Session(root=tmp_path / "v2", rosters=rosters(), flight_lines=FakeLines(flown), resolver=resolver(), **kw)


def cfg(tmp_path, iterations=2):
    return pl.LoopConfig(data_root=tmp_path, iterations=iterations, rows=2 * ROWS_PER_HALF, repo_root=REPO)


def lock_scores():
    return {r: 0.5 for fx in FIXTURES.values() for r in fx.rows}


ALL_FLOWN = [(h, i) for h in cm.HALVES for i in (1, 2, 3)]


def run(tmp_path, reader, *, flown=ALL_FLOWN, iterations=2):
    s = session(tmp_path, flown)
    return s, pl.run_loop(cfg(tmp_path, iterations), ad.V2Sampler(s), ad.V2Distiller(s), reader, lock_row_f1=lock_scores(), out=lambda t: None)


# ---- the sampler ---------------------------------------------------------------------------------------------------------------------------


def test_sampler_returns_one_valid_trajectory_per_row_and_k_with_the_row_credit(tmp_path):
    s = session(tmp_path, ALL_FLOWN)
    request = pl.SamplerRequest(1, sorted(lock_scores()), 4, 0.7, {"rules": []}, {"by_scenario": {}}, lock_scores())
    lines = ad.V2Sampler(s).sample(request)
    pl.check_sample(lines, request)
    assert len(lines) == 2 * ROWS_PER_HALF * 4 and all(pl.validate_trajectory(t) == [] for t in lines)
    best = next(t for t in lines if t.row_id == "a000" and t.k_index == 0)
    worst = next(t for t in lines if t.row_id == "a000" and t.k_index == 3)
    assert best.reward == 0.5 and worst.reward == -0.5 and best.lock_row_f1 == 0.5 and best.trace["reason"] == "sampled"
    assert best.trace["run_id"] == sp.run_id_for(H1, 1) and next(t for t in lines if t.row_id == "b000").trace["run_id"] == sp.run_id_for(H2, 1)
    assert best.trace["trajectory_id"] == "tfrl-v2/i1/a000/k0" and best.rule_ids == ()
    assert pl.credit_summary(lines)["positives"] == 2 * ROWS_PER_HALF                      # one best trajectory per row beats the lock
    for half in cm.HALVES:                                                                  # the plans are written write-once for both halves
        plan = json.loads(s.plan_path(1, half).read_text(encoding="utf-8"))
        assert plan["half"] == half and plan["k"] == 4 and plan["banks"]["bank_measure"] is None and plan["banks"]["supply_bound_items"] == 51


def test_sampler_waits_for_the_flight_and_writes_both_plans_first(tmp_path):
    s = session(tmp_path, [(H1, 1)])
    request = pl.SamplerRequest(1, sorted(lock_scores()), 4, 0.7, {}, {}, lock_scores())
    with pytest.raises(ad.FlightNotFlown, match="no trace lines yet for \\['R01C-H2'\\]"):
        ad.V2Sampler(s).sample(request)
    assert isinstance(ad.FlightNotFlown("x"), pl.SamplerNotImplemented)
    assert s.plan_path(1, H1).is_file() and s.plan_path(1, H2).is_file()


def test_sampler_refuses_rows_that_are_not_the_two_halves(tmp_path):
    s = session(tmp_path, ALL_FLOWN)
    with pytest.raises(pl.PortLoopError, match="not exactly the rows"):
        ad.V2Sampler(s).sample(pl.SamplerRequest(1, ["a000"], 4, 0.7, {}, {}, {"a000": 0.5}))


def test_session_refuses_an_estimate_measure(tmp_path):
    with pytest.raises(pl.PortLoopError, match="needs 'vllm-tokenize'"):
        session(tmp_path, ALL_FLOWN, measure=tx.EstimateMeasure())


def test_trace_file_lines_read_a_flights_trace_and_report_none_until_it_exists(tmp_path):
    source = ad.TraceFileLines(tmp_path)
    assert source.lines(H1, 1) is None and source.sha256s(H1, 1) == []
    path = source.path(H1, 1)
    path.parent.mkdir(parents=True)
    fx = FIXTURES[H1]
    path.write_text("".join(rt.dumps_line(line) for line in fx.contacts + fx.scores), encoding="utf-8")
    contacts, scores = source.lines(H1, 1)
    assert len(contacts) == len(fx.contacts) and len(scores) == len(fx.scores) and len(source.sha256s(H1, 1)) == 1


# ---- the loop over the adapters ----------------------------------------------------------------------------------------------------------


def test_round_trip_two_iterations_carry_the_bank_receipts_and_the_budget_line(tmp_path):
    reader = lt.FixtureReader({1: lt.read(), 2: lt.read(lt.credit(slu=3.0, slu_lo=1.0))})
    s, report = run(tmp_path, reader)
    assert report.status == "complete" and [i["verdict"] for i in report.iterations] == ["accepted", "accepted"]
    for record in report.iterations:
        receipts = record["bank_receipts"]
        assert set(receipts) == {H1, H2, "union"}
        assert receipts[H1]["source_halves"] == [H1] and receipts[H2]["source_halves"] == [H2] and receipts["union"]["source_halves"] == [H1, H2]
        assert {r["measure"] for r in receipts.values()} == {"vllm-tokenize"} and {r["exemplar_text"] for r in receipts.values()} == {"resolved"}
        assert all(len(r["receipt_file_sha256"]) == 64 and r["rules"] >= 1 and r["exemplar_entries"] >= 1 for r in receipts.values())
        assert record["budget_line"].startswith(f"iteration {record['iteration']} both halves program c7: {2 * ROWS_PER_HALF} rows x 7 contacts (K=4, with aa+lock)")
        assert f"{2 * ROWS_PER_HALF * 7} calls" in record["budget_line"] and "card-hours" in record["budget_line"]
    # the files behind the receipts
    for t in (1, 2):
        for name in (H1, H2, "union"):
            assert (s.bank_dir(t, name) / "bank-receipt.json").is_file()
        assert cm.sha256_file(s.bank_dir(t, "union") / "bank-receipt.json") == report.iterations[t - 1]["bank_receipts"]["union"]["receipt_file_sha256"]
    # one bank per source half; the half consumes the bank of the OTHER half at the next iteration
    plan_h1 = json.loads(s.plan_path(2, H1).read_text(encoding="utf-8"))
    plan_h2 = json.loads(s.plan_path(2, H2).read_text(encoding="utf-8"))
    assert plan_h1["banks"]["bank_source_halves"] == [H2] and plan_h2["banks"]["bank_source_halves"] == [H1]
    assert plan_h1["banks"]["bank_measure"] == "vllm-tokenize" and plan_h1["banks"]["supply_bound_items"] == 43
    h2_receipt = json.loads((s.bank_dir(1, H2) / "bank-receipt.json").read_text(encoding="utf-8"))
    assert plan_h1["banks"]["exemplar_bank_sha256"] == h2_receipt["exemplar_bank"]["content_sha256"]
    # the union is the state the loop hashes and the Fold B flight consumes
    union = json.loads((s.bank_dir(1, "union") / "bank-receipt.json").read_text(encoding="utf-8"))
    cm.check_union_for_fold_b(union, ["fold-b-row"], half_row_ids={h: FIXTURES[h].rows for h in cm.HALVES})
    for half in cm.HALVES:
        with pytest.raises(cm.CrossFitError):
            cm.check_cross_fit(union, half)
    # the consumed rule ids reach the iteration-2 trajectories
    traj = [json.loads(x) for x in (tmp_path / pl.ANALYSIS_ROOT / "i2" / "trajectories.jsonl").read_text(encoding="utf-8").splitlines()]
    assert traj and all(len(t["rule_ids"]) >= 1 for t in traj)
    # a re-run is a no-op: same receipts, same banks, same plans
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    s2, report2 = run(tmp_path, reader)
    assert report2.status == "complete" and report2.iterations == report.iterations
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


def test_distil_receipt_of_the_loop_carries_the_receipts_and_the_budget_line(tmp_path):
    reader = lt.FixtureReader({1: lt.read()})
    run(tmp_path, reader, iterations=1)
    receipt = json.loads((tmp_path / pl.ANALYSIS_ROOT / "i1" / "distil-receipt.json").read_text(encoding="utf-8"))
    assert set(receipt["bank_receipts"]) == {H1, H2, "union"} and "both halves program c7" in receipt["budget_line"]
    assert receipt["contrast_pairs"] == 2 * ROWS_PER_HALF


def test_loop_halts_while_a_half_has_not_been_flown_and_resumes_when_it_has(tmp_path):
    reader = lt.FixtureReader({1: lt.read()})
    s, report = run(tmp_path, reader, flown=[(H1, 1)], iterations=1)
    assert report.status == "halted" and "no trace lines yet" in report.halt_reason
    assert s.plan_path(1, H1).is_file() and s.plan_path(1, H2).is_file() and not (tmp_path / pl.ANALYSIS_ROOT / "i1" / "trajectories.jsonl").exists()
    _, report = run(tmp_path, reader, flown=[(H1, 1), (H2, 1)], iterations=1)
    assert report.status == "complete"


def test_loop_awaiting_read_still_carries_the_receipts_and_the_budget_line(tmp_path):
    _, report = run(tmp_path, lt.FixtureReader({}), iterations=1)
    assert report.status == "awaiting-read"
    record = report.iterations[0]
    assert record["verdict"] == "awaiting-read" and set(record["bank_receipts"]) == {H1, H2, "union"} and "card-hours" in record["budget_line"]


def test_distiller_refuses_an_unknown_item_id(tmp_path):
    s = session(tmp_path, ALL_FLOWN)
    s.resolver = tx.ItemTextResolver({"only-this": "text"})
    request = pl.SamplerRequest(1, sorted(lock_scores()), 4, 0.7, {}, {}, lock_scores())
    lines = ad.V2Sampler(s).sample(request)
    with pytest.raises(pl.PortLoopError, match="unknown supplied item id"):
        ad.V2Distiller(s).distil(1, lines, {}, {})


def test_iteration_two_needs_the_bank_of_the_previous_iteration(tmp_path):
    s = session(tmp_path, ALL_FLOWN)
    request = pl.SamplerRequest(2, sorted(lock_scores()), 4, 0.7, {}, {}, lock_scores())
    with pytest.raises(pl.PortLoopError, match="consumes the bank of R01C-H2 at iteration 1"):
        ad.V2Sampler(s).sample(request)


def test_a_distiller_without_receipts_leaves_the_iteration_record_unchanged(tmp_path):
    reader = lt.FixtureReader({1: lt.read()})
    report = pl.run_loop(lt.cfg_for(tmp_path, iterations=1), lt.FixtureSampler(), lt.FixtureDistiller(), reader, lock_row_f1=lt.lock_scores(), out=lambda s: None)
    assert report.iterations[0].keys() == {"iteration", "verdict", "failed", "state_id", "construction_gain_pp"}
    assert pl.result_extras(pl.DistilResult(rule_bank={}, exemplar_bank={})) == {}

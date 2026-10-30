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
import tfrl_v2_distiller as dd  # noqa: E402
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
                   "items": [{"sample_id": r, "audio_seconds": 2.0, "stratum": "alarm"} for r in fx.rows]} for half, fx in FIXTURES.items()}


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


# ---- rule-variants mode through the session ----------------------------------------------------------------------------------------------


def test_rule_variants_session_gives_each_trajectory_its_variant_from_iteration_two(tmp_path):
    reader = lt.FixtureReader({1: lt.read(), 2: lt.read(lt.credit(slu=3.0, slu_lo=1.0))})
    s = session(tmp_path, ALL_FLOWN, variation="rule-variants", variants_fn=dd.rule_variants_seed_bank)
    report = pl.run_loop(cfg(tmp_path, 2), ad.V2Sampler(s), ad.V2Distiller(s), reader, lock_row_f1=lock_scores(), out=lambda t: None)
    assert report.status == "complete"
    p1 = json.loads(s.plan_path(1, H1).read_text(encoding="utf-8"))
    assert p1["variation"] == "seed-only" and all("rule_variant" not in c for r in p1["row_plan"] for c in r["contacts"])      # no bank yet
    p2 = json.loads(s.plan_path(2, H1).read_text(encoding="utf-8"))
    assert p2["variation"] == "rule-variants" and p2["banks"]["seed_rule_bank_sha256"] is not None
    row = p2["row_plan"][0]
    solvers = [c for c in row["contacts"] if c["node"] == "solver"]
    assert [c["rule_variant"]["kind"] for c in solvers] == ["none", "full", "scenario", "top-half"]
    traj = [json.loads(x) for x in (tmp_path / pl.ANALYSIS_ROOT / "i2" / "trajectories.jsonl").read_text(encoding="utf-8").splitlines()]
    mine = {t["k_index"]: t["rule_ids"] for t in traj if t["trace"]["row_id"] == row["row_id"]}
    assert mine[0] == [] and len(mine[1]) >= 1 and mine[2] == mine[1] and set(mine[3]) <= set(mine[1])        # all fixture rules are alarm rules


def test_rule_variants_session_refuses_k_other_than_four_once_a_bank_exists(tmp_path):
    reader = lt.FixtureReader({1: lt.read()})
    s = session(tmp_path, ALL_FLOWN, variation="rule-variants", variants_fn=dd.rule_variants_seed_bank)
    pl.run_loop(cfg(tmp_path, 1), ad.V2Sampler(s), ad.V2Distiller(s), reader, lock_row_f1=lock_scores(), out=lambda t: None)
    request = pl.SamplerRequest(2, sorted(lock_scores()), 8, 0.7, {}, {}, lock_scores())
    with pytest.raises(pl.PortLoopError, match="needs K = 4"):
        ad.V2Sampler(s).sample(request)


# ---- the tokenizer driver and the --v2 command line ---------------------------------------------------------------------------------------


class RecordingFactory:
    """A ``client_factory`` for ``GuardedTokenizer``: records its construction arguments and answers /tokenize like the fake tokenizer; no socket."""

    def __init__(self):
        self.built = []
        self.client = FakeTokenizer()

    def __call__(self, **kwargs):
        self.built.append(kwargs)
        return self.client


def test_tokenize_client_is_the_registered_guarded_tokenizer_over_the_loopback_client():
    from speech_aware_evidence_acquisition.core.preparation_fit import GuardedTokenizer

    factory = RecordingFactory()
    client = ad.tokenize_client("http://127.0.0.1:8123", client_factory=factory, contact_cap=2)
    assert isinstance(client, GuardedTokenizer) and client.contact_cap == 2
    assert factory.built == [{"host": "127.0.0.1", "port": 8123, "timeout_seconds": 600.0}]
    assert client.request_json("POST", "/tokenize", {"messages": [{"role": "user", "content": [{"type": "text", "text": "abcd"}]}]})["count"] == 1 + 7
    with pytest.raises(Exception, match="non-tokenizer route"):
        client.request_json("POST", "/v1/chat/completions", {})                                         # the guard refuses generation routes
    for bad in ("https://h:1", "http://h", "http://h:1/path", "ftp://h:1"):
        with pytest.raises(pl.PortLoopError, match="must be http://host:port"):
            ad.tokenize_client(bad, client_factory=factory)
    assert ad.tokenize_client("localhost:9000", client_factory=factory).client is factory.client           # a bare host:port is read as http


def test_build_measure_selects_the_measure_and_never_contacts_the_endpoint_at_construction():
    factory = RecordingFactory()
    assert ad.build_measure("estimate").name == "estimate"
    measure = ad.build_measure("vllm-tokenize", endpoint="http://127.0.0.1:1", model="m", client_factory=factory)
    assert measure.name == "vllm-tokenize" and measure.calls == 0
    assert measure.count("abcdefgh") == 2 + 7 and measure.calls == 1
    with pytest.raises(pl.PortLoopError, match="needs a tokenize endpoint"):
        ad.build_measure("vllm-tokenize")
    with pytest.raises(pl.PortLoopError, match="unknown measure"):
        ad.build_measure("chars")


def test_build_session_uses_the_committed_rosters_and_the_injected_resolver_and_client(tmp_path):
    factory = RecordingFactory()
    s = ad.build_session(data_root=tmp_path, analysis_root=tmp_path / "an", measure="vllm-tokenize", tokenize_endpoint="http://127.0.0.1:1",
                         variation="rule-variants", repo_root=REPO, resolver=tx.ItemTextResolver({"x": "t"}), client_factory=factory)
    assert s.root == tmp_path / "an" / "v2" and sorted(s.rosters) == sorted(cm.HALVES) and s.variation == "rule-variants"
    assert s.variants_fn is dd.rule_variants_seed_bank and all(len(v) == 64 for v in s.roster_sha256.values())
    assert [len(s.rosters[h]["items"]) for h in cm.HALVES] == [313, 312] and len(s.column_rows()) == 625
    assert ad.build_session(data_root=tmp_path, analysis_root=tmp_path, measure="vllm-tokenize", tokenize_endpoint="http://127.0.0.1:1", repo_root=REPO,
                            resolver=tx.ItemTextResolver({"x": "t"}), client_factory=factory).variants_fn is None
    with pytest.raises(pl.PortLoopError, match="needs 'vllm-tokenize'"):
        ad.build_session(data_root=tmp_path, analysis_root=tmp_path, measure="estimate", repo_root=REPO, resolver=tx.ItemTextResolver({"x": "t"}))


def test_cli_v2_dry_run_prints_the_budget_lines_and_writes_nothing(tmp_path, capsys):
    before = sorted(tmp_path.rglob("*"))
    assert pl.main(["--v2", "--dry-run", "--iterations", "2", "--data-root", str(tmp_path), "--analysis-root", str(tmp_path / "an")]) == 0
    text = capsys.readouterr().out
    assert "iteration 1 R01C-H1 program c7" in text and "313 rows x 7 contacts (K=4, with aa+lock)" in text
    assert "iteration 1 both halves program c7: 625 rows x 7 contacts" in text and "iteration 2 both halves program c7: 625 rows" in text
    assert "rule-variants mode" in text and "card-hours" in text
    assert sorted(tmp_path.rglob("*")) == before                                                      # nothing written, no data root needed either
    assert pl.main(["--v2", "--dry-run", "--iterations", "1", "--k", "8", "--variation", "seed-only"]) == 0
    assert "625 rows x 11 contacts (K=8" in capsys.readouterr().out


def test_cli_without_v2_keeps_the_stub_dry_run(capsys):
    assert pl.main(["--dry-run", "--iterations", "1"]) == 0
    assert "TODO-T3 stub" in capsys.readouterr().out


def test_cli_v2_argument_rules(tmp_path, capsys):
    with pytest.raises(SystemExit):
        pl.main(["--v2", "--data-root", str(tmp_path)])                                                 # vllm-tokenize needs an endpoint
    assert "requires --tokenize-endpoint" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        pl.main(["--v2", "--data-root", str(tmp_path), "--tokenize-endpoint", "http://127.0.0.1:1", "--k", "8"])
    assert "needs K=4" in capsys.readouterr().err
    assert pl.main(["--v2", "--data-root", str(tmp_path), "--measure", "estimate"]) == 2                 # the session refuses an estimate measure
    assert "REFUSED" in capsys.readouterr().err


def test_cli_v2_halts_awaiting_the_lock_scores_through_the_injected_client(tmp_path, capsys):
    factory = RecordingFactory()
    code = pl.main(["--v2", "--iterations", "1", "--data-root", str(tmp_path), "--analysis-root", str(tmp_path / "an"),
                    "--tokenize-endpoint", "http://127.0.0.1:1", "--tokenize-model", "m", "--base-seed", "7"],
                   v2_overrides={"resolver": tx.ItemTextResolver({"x": "t"}), "client_factory": factory})
    out = capsys.readouterr().out
    assert code == 3 and '"status": "halted"' in out and "construction-lock-row-scores.json" in out
    assert factory.built == [{"host": "127.0.0.1", "port": 1, "timeout_seconds": 600.0}] and factory.client is not None


def test_cli_v2_runs_the_loop_to_the_first_missing_flight(tmp_path, capsys):
    """With the lock scores of the real 625 rows present, the V2 sampler writes both iteration-1 plans and halts at the missing flights."""

    import tfrl_v2_loop_adapters as adapters

    docs, _ = adapters.load_half_rosters(REPO)
    rows = sorted(str(i["sample_id"]) for h in cm.HALVES for i in docs[h]["items"])
    adir = tmp_path / "an"
    adir.mkdir()
    (adir / "construction-lock-row-scores.json").write_text(json.dumps({"arms": {"lock": {r: 0.5 for r in rows}}}), encoding="utf-8")
    code = pl.main(["--v2", "--iterations", "1", "--data-root", str(tmp_path), "--analysis-root", str(adir), "--tokenize-endpoint", "http://127.0.0.1:1",
                    "--base-seed", "7"], v2_overrides={"resolver": tx.ItemTextResolver({"x": "t"}), "client_factory": RecordingFactory()})
    assert code == 3 and "no trace lines yet" in capsys.readouterr().out
    plan = json.loads((adir / "v2" / "plans" / "i1-R01C-H1.json").read_text(encoding="utf-8"))
    assert plan["base_seed"] == 7 and plan["k"] == 4 and (adir / "v2" / "plans" / "i1-R01C-H2.json").is_file()

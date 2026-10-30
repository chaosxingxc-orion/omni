"""The TFRL-v2 port loop driver (``scripts/flights/nx/tfrl_port_loop.py``, task T5): model-free, no data root, no card.

Proved here:

* ``--dry-run`` prints the iteration plan (at most 3 iterations, K trajectories per row, the construction column identity, the stop rule)
  and touches no file; more than 3 iterations is refused;
* the acceptance wrapper on a synthetic fixture: accept; refuse on the within-run floor; refuse on the intent guard; refuse on the twin
  (a twin the child does not beat, and a twin outside the 1.2 pp admissibility bound); refuse on a negative per-row delta; the M1
  replicate variance widens the interval and can flip an accept into a refusal;
* the sampler call site is a stub that fails closed (TODO-T3), and its result is checked (K lines per row, consistent rewards);
* a full loop on fixtures: accept at iteration 1, state admitted into the write-once archive, a refusal at iteration 2 stops the loop,
  a missing Fold B read halts it as ``awaiting-read``, and a re-run is a no-op;
* a static check: no Windows-side path and no non-ASCII text in the driver or in this file.

Fixtures are synthetic and aggregate-only.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
NX = REPO / "scripts" / "flights" / "nx"
for _p in (str(REPO / "src"), str(NX)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from speech_aware_evidence_acquisition.contracts import rollout_trace as rt  # noqa: E402
import tfrl_port_loop as pl  # noqa: E402

DRIVER = NX / "tfrl_port_loop.py"
ROWS = 1080


def ci(point, lo, hi=None):
    return {"point": point, "lo": lo, "hi": point + 1.0 if hi is None else hi}


def credit(slu=2.0, slu_lo=0.5, intent=0.3, intent_lo=-0.2):
    return {
        "slu_f1": 72.0, "intent": 85.0,
        "vs_lock": {"slu": ci(slu, slu_lo), "intent": ci(intent, intent_lo)},
        "vs_twin": None, "vs_parent": None,
        "per_scenario": {"alarm": {"slu": 1.0, "intent": 0.0}},
        "false_fill": {"n": 10, "delta_pp": -0.5}, "non_empty_exact": {"n": 90, "delta_pp": 1.0},
        "contacts_per_row": 2.0,
    }


def read(child=None, twin=None, **kw):
    return pl.CandidateRead(run_id="run-fx", attempt=1, rows=ROWS, credit=child or credit(), twin_credit=twin or credit(0.2, -1.0, 1.4), **kw)


# ---- dry run ---------------------------------------------------------------------------------------------------------------

def test_dry_run_prints_the_iteration_plan(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("SPEECHRL_DATA_DIR", str(tmp_path / "does-not-exist"))
    assert pl.main(["--dry-run"]) == 0
    text = capsys.readouterr().out
    for t in (1, 2, 3):
        assert f"iteration {t}/3" in text
    assert "625 rows x K=4 at temperature 0.7 = 2500" in text
    assert "SLURP-STAGE3-CONSTRUCTION-R01C" in text
    assert "slurp-native-stage3-tfrl-construction-lock-r01c-npu-port-v1" in text
    assert "total Solver contacts if all 3 iterations run: 7500" in text
    assert "kill the family" in text and "Fold A confirmation" in text
    assert not (tmp_path / "does-not-exist").exists()


def test_dry_run_two_iterations_and_refuse_more_than_three(capsys):
    assert pl.main(["--dry-run", "--iterations", "2"]) == 0
    text = capsys.readouterr().out
    assert "iteration 2/2" in text and "iteration 3" not in text
    with pytest.raises(SystemExit):
        pl.main(["--dry-run", "--iterations", "4"])
    with pytest.raises(pl.PortLoopError):
        pl.LoopConfig(data_root=Path("."), iterations=4)


def test_construction_identity_is_the_port_rule_applied_to_the_historical_ids():
    c = pl.CONSTRUCTION
    assert c["rows"] == 625 and c["must_reflight_on_port"] is True
    assert c["port_cell"] == c["historical_cell"][: -len("-v1")] + "-npu-port-v1"
    assert c["port_run_id"] == c["historical_run_id"][: -len("-V1")] + "-NPU-PORT-V1"


# ---- acceptance wrapper -----------------------------------------------------------------------------------------------------

def test_accept():
    verdict = pl.accept_candidate(read())
    assert verdict["accepted"] is True and verdict["failed"] == [] and verdict["status"] == "accepted"
    assert abs(verdict["rules"]["slu_floor_pp"] - 1.49) < 0.001           # the rules are read from the floor table, not copied


def test_refuse_on_floor():
    verdict = pl.accept_candidate(read(credit(slu=1.0, slu_lo=0.2)))
    assert verdict["accepted"] is False and verdict["failed"] == ["b_point_floor"]


def test_refuse_on_intent_guard():
    verdict = pl.accept_candidate(read(credit(intent=0.3, intent_lo=-0.9)))
    assert verdict["failed"] == ["e_intent_guard"]
    verdict = pl.accept_candidate(read(credit(intent=-0.1, intent_lo=-0.3)))      # point under zero: also refused
    assert verdict["failed"] == ["e_intent_guard"]


def test_refuse_on_twin():
    # a twin above the child: the child does not beat it, and the twin is outside the 1.2 pp bound
    verdict = pl.accept_candidate(read(twin=credit(2.5, 1.0, 4.0)))
    assert verdict["accepted"] is False
    assert "c_beats_twin" in verdict["failed"] and "f_twin_admissible" in verdict["failed"]
    # a twin inside the bound that the child beats is fine; a twin outside the bound refuses even when beaten
    assert pl.accept_candidate(read(twin=credit(1.2, 0.0, 2.4)))["accepted"] is True
    verdict = pl.accept_candidate(read(credit(slu=4.0, slu_lo=2.0), twin=credit(1.5, 0.0, 3.0)))
    assert verdict["failed"] == ["f_twin_admissible"]


def test_refuse_on_negative_per_row_delta_and_replicate_widening():
    assert pl.accept_candidate(read(per_row_delta_pp=0.3))["accepted"] is True
    assert pl.accept_candidate(read(per_row_delta_pp=-0.1))["failed"] == ["g_per_row_not_negative"]
    narrow = read(credit(slu=2.0, slu_lo=0.2))
    assert pl.accept_candidate(narrow)["accepted"] is True
    wide = read(credit(slu=2.0, slu_lo=0.2), replicate_sd_pp=0.6)               # 1.96 * 0.6 widens the lower bound below 0
    verdict = pl.accept_candidate(wide)
    assert verdict["widened"] is True and verdict["failed"] == ["a_ci_lb"]


def test_parent_must_be_beaten():
    parent = credit(3.0, 1.0)
    child = credit(2.0, 0.5)
    child["vs_parent"] = {"slu": ci(-0.5, -1.5), "intent": ci(0.0, -0.5)}
    verdict = pl.accept_candidate(read(child, parent_credit=parent))
    assert verdict["failed"] == ["d_beats_parent"]


def test_fold_a_confirmation():
    assert pl.fold_a_confirms(credit(1.0, 0.1)) is True
    assert pl.fold_a_confirms(credit(1.0, -0.1)) is False


# ---- sampler contract ------------------------------------------------------------------------------------------------------

def line(row, k, f1, lock_f1, iteration=1):
    trace = {"schema": rt.SCHEMA, "kind": rt.KIND_TFRL, "ts": "2026-10-30T00:00:00Z", "run_id": "run-fx", "attempt_id": "attempt-fx", "row_id": row,
             "arm": pl.TRACE_ARM, "iteration": iteration, "trajectory_id": f"i{iteration}-{row}-{k}", "reward": f1 - lock_f1,
             "selected": False, "reason": None}
    return pl.Trajectory(trace=trace, k_index=k, temperature=0.7, row_f1=f1, lock_row_f1=lock_f1, rule_ids=("r1",), exemplar_ids=())


class FixtureSampler:
    def __init__(self, mode="good"):
        self.mode = mode

    def sample(self, request):
        out = []
        for row in request.column_rows:
            lock = request.lock_row_f1[row]
            for k in range(request.k):
                f1 = min(1.0, lock + 0.25) if k == 0 else max(0.0, lock - 0.25) if k == 1 else lock
                out.append(line(row, k, f1, lock, request.iteration))
        if self.mode == "short":
            out = out[:-1]
        return out


def test_stub_sampler_fails_closed():
    req = pl.SamplerRequest(1, ["a"], 4, 0.7, {}, {}, {"a": 0.5})
    with pytest.raises(pl.SamplerNotImplemented):
        pl.StubSampler().sample(req)


def test_validate_trajectory_uses_the_t1_contract_and_check_sample():
    good = line("a", 0, 0.75, 0.5)
    assert pl.validate_trajectory(good) == []
    bad_reward = pl.Trajectory(trace={**good.trace, "reward": 9.0}, k_index=0, temperature=0.7, row_f1=0.75, lock_row_f1=0.5)
    assert any("reward" in p for p in pl.validate_trajectory(bad_reward))
    missing = pl.Trajectory(trace={k: v for k, v in good.trace.items() if k != "row_id"}, k_index=0, temperature=0.7, row_f1=0.75, lock_row_f1=0.5)
    assert any("trace line" in p for p in pl.validate_trajectory(missing))
    extra = pl.Trajectory(trace={**good.trace, "row_f1": 0.75}, k_index=0, temperature=0.7, row_f1=0.75, lock_row_f1=0.5)      # the T1 line is closed
    assert any("unknown fields" in p for p in pl.validate_trajectory(extra))
    wrong_kind = pl.Trajectory(trace={**good.trace, "kind": "score"}, k_index=0, temperature=0.7, row_f1=0.75, lock_row_f1=0.5)
    assert pl.validate_trajectory(wrong_kind)
    req = pl.SamplerRequest(1, ["a"], 4, 0.7, {}, {}, {"a": 0.5})
    pl.check_sample(FixtureSampler().sample(req), req)
    with pytest.raises(pl.PortLoopError):
        pl.check_sample(FixtureSampler("short").sample(req), req)
    summary = pl.credit_summary(FixtureSampler().sample(req))
    assert (summary["positives"], summary["negatives"], summary["contrast_rows"]) == (1, 0, 1)   # lock wrong (0.5): no negatives by definition


def test_negative_means_breaking_a_row_lock_gets_right():
    req = pl.SamplerRequest(1, ["a"], 4, 0.7, {}, {}, {"a": 1.0})
    summary = pl.credit_summary(FixtureSampler().sample(req))
    assert (summary["positives"], summary["negatives"]) == (0, 1)


# ---- the loop on fixtures --------------------------------------------------------------------------------------------------

class FixtureDistiller:
    def __init__(self, gain=2.0):
        self.gain = gain

    def distil(self, iteration, trajectories, rule_bank, exemplar_bank):
        rules = {"rules": [{"rule_id": f"r{iteration}", "kind": "consumption-instruction", "text": "synthetic"}]}
        return pl.DistilResult(rule_bank=rules, exemplar_bank={"by_scenario": {"alarm": [f"e{iteration}"]}}, vote_n=3,
                               contrast_pairs=1, construction_gain_pp=self.gain)


class FixtureReader:
    def __init__(self, reads):
        self.reads = reads

    def read(self, iteration, state_digests):
        return self.reads.get(iteration)


def cfg_for(tmp_path, rows=6, iterations=3):
    return pl.LoopConfig(data_root=tmp_path, iterations=iterations, rows=rows, repo_root=REPO)


def lock_scores(rows=6):
    return {f"row{i}": 0.5 for i in range(rows)}


def test_loop_accept_then_refuse_stops_and_admits_both_states(tmp_path):
    reader = FixtureReader({1: read(), 2: read(credit(slu=0.5, slu_lo=-0.5))})
    report = pl.run_loop(cfg_for(tmp_path), FixtureSampler(), FixtureDistiller(), reader, lock_row_f1=lock_scores(), out=lambda s: None)
    assert report.status == "stopped" and "nothing accepted" in report.halt_reason
    assert [i["verdict"] for i in report.iterations] == ["accepted", "refused"]
    assert len(report.accepted_states) == 1
    archive = tmp_path / pl.ANALYSIS_ROOT / "archive"
    states = sorted(archive.glob("state-*.json"))
    assert len(states) == 3                                               # root + accepted + refused
    verdicts = sorted(json.loads(p.read_text())["verdict"] for p in states)
    assert verdicts == ["accepted", "accepted", "refused"]
    # no text-bearing key in anything written
    for p in (tmp_path / pl.ANALYSIS_ROOT).rglob("*.json"):
        assert "utterance" not in p.read_text()
    # a re-run is a no-op (same inputs, same receipts, same archive)
    before = {p: p.read_bytes() for p in (tmp_path / pl.ANALYSIS_ROOT).rglob("*") if p.is_file()}
    report2 = pl.run_loop(cfg_for(tmp_path), FixtureSampler(), FixtureDistiller(), reader, lock_row_f1=lock_scores(), out=lambda s: None)
    assert report2.status == "stopped" and report2.iterations == report.iterations
    after = {p: p.read_bytes() for p in (tmp_path / pl.ANALYSIS_ROOT).rglob("*") if p.is_file()}
    assert before == after


def test_loop_halts_while_the_fold_b_read_is_absent(tmp_path):
    report = pl.run_loop(cfg_for(tmp_path), FixtureSampler(), FixtureDistiller(), FixtureReader({}), lock_row_f1=lock_scores(), out=lambda s: None)
    assert report.status == "awaiting-read"
    assert (tmp_path / pl.ANALYSIS_ROOT / "i1" / "trajectories.jsonl").is_file()          # the sample step is checkpointed


def test_loop_with_the_stub_sampler_halts_closed(tmp_path):
    report = pl.run_loop(cfg_for(tmp_path), pl.StubSampler(), pl.StubDistiller(), FixtureReader({}), lock_row_f1=lock_scores(), out=lambda s: None)
    assert report.status == "halted" and "TODO-T3" in report.halt_reason


def test_loop_kill_criterion_on_iteration_one_gain(tmp_path):
    reader = FixtureReader({1: read()})
    report = pl.run_loop(cfg_for(tmp_path), FixtureSampler(), FixtureDistiller(gain=0.4), reader, lock_row_f1=lock_scores(), out=lambda s: None)
    assert report.status == "stopped" and "kill criterion" in report.halt_reason


def test_loop_requires_the_construction_lock_row_scores(tmp_path):
    report = pl.run_loop(cfg_for(tmp_path, rows=625), FixtureSampler(), FixtureDistiller(), FixtureReader({}), out=lambda s: None)
    assert report.status == "halted" and "re-flown on the port" in report.halt_reason


def test_loop_three_iterations_complete(tmp_path):
    reader = FixtureReader({1: read(), 2: read(credit(slu=3.0, slu_lo=1.0)), 3: read(credit(slu=4.0, slu_lo=1.5), twin=credit(0.5, -0.5, 1.5))})
    report = pl.run_loop(cfg_for(tmp_path), FixtureSampler(), FixtureDistiller(), reader, lock_row_f1=lock_scores(), out=lambda s: None)
    assert report.status == "complete" and len(report.accepted_states) == 3


# ---- static: no Windows-side path, English only --------------------------------------------------------------------------

def _forbidden_patterns():
    # assembled from pieces so this file does not contain the strings it forbids
    return [
        re.compile("/" + "mnt" + "/[a-z]/"),             # a Windows drive mounted into a Linux guest
        re.compile("/" + "home" + "/chao"),
        re.compile("chao" + "_workspace"),
        re.compile("[A-Za-z]:" + r"\\"),
        re.compile("(?i)" + "w" + "sl"),
        re.compile("(?i)" + "power" + "shell"),
        re.compile("(?i)" + "llama" + r"[._ -]?cpp"),
        re.compile("(?i)" + "q4_k" + "_m"),
    ]


@pytest.mark.parametrize("path", [DRIVER, Path(__file__)])
def test_no_windows_path_and_english_only(path):
    text = path.read_text(encoding="utf-8")
    for pattern in _forbidden_patterns():
        assert not pattern.search(text), f"{path.name}: matches {pattern.pattern}"
    assert not re.search(r"[\u4e00-\u9fff]", text)
    assert text.isascii()


def test_driver_default_data_root_comes_from_the_environment_only():
    text = DRIVER.read_text(encoding="utf-8")
    assert "SPEECHRL_DATA_DIR" in text
    assert "default=\"/" not in text and "default='/" not in text

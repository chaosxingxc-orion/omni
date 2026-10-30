"""The base-lock filler (``scripts/readiness/fill_base_lock_from_recorded_runs.py``) on a fake record_runs.py chain.

Model-free and CPU-only: the fixture writes a chain directory in the layout ``work/scripts/record_runs.py`` produces
(``config.json``, ``run-list.json``, ``chain-status.json``, ``manifest.json``, ``steps/<step>.{done,failed}.json``, the
verification report, the runtime receipt and the level reports beside the chain directory) plus a small skeleton
base-lock record and S8 document, then runs the filler's ``main``.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "readiness" / "fill_base_lock_from_recorded_runs.py"
_spec = importlib.util.spec_from_file_location("fill_base_lock_from_recorded_runs", SCRIPT)
fill = importlib.util.module_from_spec(_spec)
sys.modules["fill_base_lock_from_recorded_runs"] = fill
_spec.loader.exec_module(fill)

PROFILE = "p-4xtp2-v3"
OTHER = "p-4xtp2-v4"
TS = "20261030T120000Z"
CHECKS = ["image", "versions", "cann", "weights", "endpoints", "protocol", "deployment", "env", "kv", "template", "receipt"]
PROTO = "f" * 64


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def write_json(path: Path, doc) -> bytes:
    raw = (json.dumps(doc, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return raw


def skeleton_lock() -> dict:
    todo = lambda what: f"TODO-R0: {what}"
    engine = {"kv_tokens": todo("kv"), "block_size": todo("block"), "cudagraph_capture_sizes": todo("sizes"),
              "execute_model_timeout_seconds": None}
    return {
        "_skeleton": {"status": "SKELETON. Leaves whose value starts with 'TODO-R0:' are placeholders."},
        "schema": fill.LOCK_SCHEMA,
        "image": {"ref": "saea-base:20261027", "created": todo("Created")},
        "hosts": {"A800I-03": {"card_count": 8, "cann_files": {"/etc/ascend_install.info": {
            "bytes": todo("bytes"), "sha256": todo("sha")}}}},
        "protocol": {"served_model_name": "qwen3-omni-30b-a3b-thinking", "protocol_sha256": todo("protocol"),
                     "decoding": {"max_tokens": 7168}},
        "receipt": {"path": todo("receipt path")},
        "template": {"tails": {"thinking_true": todo("tail on"), "thinking_false": todo("tail off")}},
        "profiles": {
            PROFILE: {"profile_sha256": "a" * 64, "instances": {
                "i0": {"max_num_seqs": 32, "engine": copy.deepcopy(engine)},
                "i1": {"max_num_seqs": 32, "engine": copy.deepcopy(engine)}}},
            OTHER: {"profile_sha256": "b" * 64, "instances": {"i0": {"max_num_seqs": 24, "engine": copy.deepcopy(engine)}}},
        },
    }


def observed_facts(logs: Path) -> dict:
    eng = lambda kv: {"kv_tokens": kv, "block_size": 128, "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 24, 32],
                      "execute_model_timeout_seconds": None}
    return {
        "image": {"image": {"ref": "saea-base:20261027", "created": "2026-10-27T12:02:29Z"}},
        "cann": {"hosts": {"A800I-03": {"cann_files": {"/etc/ascend_install.info": {"bytes": 123, "sha256": "c" * 64}}}}},
        "protocol": {"protocol": {"served_model_name": "qwen3-omni-30b-a3b-thinking", "protocol_sha256": PROTO,
                                  "decoding": {"max_tokens": 7168}}},
        "deployment": {"hosts": {"A800I-03": {"card_count": 8}}, "profiles": {PROFILE: {
            "profile_sha256": "a" * 64, "instances": {"i0": {"max_num_seqs": 32}, "i1": {"max_num_seqs": 32}}}}},
        "kv": {"profiles": {PROFILE: {"instances": {"i0": {"engine": eng(551000)}, "i1": {"engine": eng(551232)}}}}},
        "template": {"template": {"tails": {"thinking_true": "<|im_start|>assistant\n", "thinking_false": "</think>\n\n"}}},
        "receipt": {"receipt": {"path": str(logs / f"receipt-{PROFILE}-{TS}.json")}},
    }


def level_row(logs: Path, run: dict, level: int, *, passes=True, worst=False, admissible=None, reasons=()) -> dict:
    mint = f"-mintok{run['min_tokens']}" if run.get("min_tokens") else ""
    name = f"sweep-rec-{run['id']}{mint}-{TS}-L{level}.json"
    raw = write_json(logs / name, {"levels": [{"level": level}], "fixture": True})
    inst = 2
    ok = run["passes"] * level * inst * 3
    return {
        "concurrency_per_instance": level, "instances": inst, "max_num_seqs": 32, "requests": ok, "ok": ok,
        "errors": 0, "timeouts": 0, "wall_s": 600.0, "host_output_tok_s": 1626.37 + level,
        "e2e_tok_s": {"p50": 17.79, "p10": 16.59}, "latency_s": {"p50": 403.0, "p95": 434.5, "max": 439.4},
        "ttft_s": {"p95": 9.876}, "preemptions_delta": 0.0,
        "per_instance": [{"name": "i0", "kv_peak": 0.7712, "preemptions_delta": 0.0},
                         {"name": "i1", "kv_peak": 0.7572, "preemptions_delta": 0.0}],
        "min_hbm_free_mib_per_card": 4126, "num_requests_waiting_peak": 10.0, "ceiling_bound": False,
        "worst_case": worst, "worst_case_admissible": admissible, "passes": passes, "failing_reasons": list(reasons),
        "report": name, "report_sha256": sha(raw), "passes_requested": run["passes"],
        "greedy_probe": {"sequential_before_vs_after": [
            {"requests": 16, "byte_identical": 16, "mismatched_indices": [], "max_abs_logprob_diff": 0.0},
            {"requests": 16, "byte_identical": 15, "mismatched_indices": [3], "max_abs_logprob_diff": 0.25}]},
    }


RUNS = [
    {"id": "R-A2-L32", "levels": [32], "passes": 3, "min_tokens": None, "sweep": {}, "admissibility": False,
     "on_fail": "stop_chain", "max_latency_s": 420.0, "note": ""},
    {"id": "R-A3-b6144", "levels": [32, 24], "passes": 1, "min_tokens": 7168,
     "sweep": {"max_tokens": 7168, "thinking_budget": 6144}, "admissibility": True, "on_fail": "continue",
     "max_latency_s": 735.0, "note": ""},
]


DRIVER = Path("/mnt/cephfs/xingchao/omni/work/scripts/record_runs.py")


@pytest.mark.skipif(not DRIVER.is_file(), reason="record_runs.py is only on the shared CephFS")
def test_r0_steps_and_report_names_follow_the_driver():
    import re

    text = DRIVER.read_text(encoding="utf-8")
    body = text[text.index("    def r0(self) -> None:"):text.index("    # --- levels")]
    assert tuple(re.findall(r'\("([a-z-]+)", self\.r0_', body)) == fill.R0_STEPS
    assert 'EXPECTED_CHECKS = 11' in text and '"verdict": why, "report": self.paths["verification"]' in text
    assert '"verification": f"{logs}/base-verification-{profile[\'profile_id\']}-rec-{ts}.json"' in text
    assert 'f"{step}.done.json"' in text and '"row": row' in text


class Fixture:
    def __init__(self, root: Path, *, verify_failed=(), bootstrap=False, facts=None, lock=None) -> None:
        self.root, self.logs = root, root / "logs"
        self.chain = self.logs / f"record-runs-{TS}"
        self.lock_path = root / "docs" / "base-lock-v1.json"
        self.s8_path = root / "docs" / "speed-baseline.md"
        self.lock = lock or skeleton_lock()
        self.lock_raw = write_json(self.lock_path, self.lock)
        self.s8_text = ("# S8\n\n## 1. Recorded table (EMPTY)\n\nIntro paragraph.\n\n"
                        "Branch (ii)/(iii)/(iv) of plan 3.10 (arm A resident):\n\n| run | level |\n|---|---|\n| R-A1 | 8 |\n\n"
                        "## 2. Selection rule\n\nText.\n")
        self.s8_path.write_text(self.s8_text, encoding="utf-8")
        steps = self.chain / "steps"
        steps.mkdir(parents=True)
        run_list = {"schema": "saea-record-run-list-v1", "profile": PROFILE, "admitted_level": 32, "runs": RUNS,
                    "seed": "20261028", "example": False, "label": "fixture", "r0": {"base_lock": str(self.lock_path)}}
        write_json(self.chain / "config.json", {"ts": TS, "mode": "record"})
        write_json(self.chain / "run-list.json", run_list)
        write_json(self.chain / "chain-status.json", {"state": "failed" if bootstrap else "done"})
        self.receipt = self.logs / f"receipt-{PROFILE}-{TS}.json"
        write_json(self.receipt, {"protocol_sha256": PROTO})
        self.report_path = self.logs / f"base-verification-{PROFILE}-rec-{TS}.json"
        facts = observed_facts(self.logs) if facts is None else facts
        checks = {}
        for name in CHECKS:
            failures = [f"{name}: injected failure"] if name in verify_failed else []
            checks[name] = {"verdict": "fail" if failures else "pass", "failures": failures,
                            "facts": facts.get(name, {}), "details": {}, "claims_checked": 1}
        if bootstrap:  # what the verifier reports against the skeleton: every placeholder leaf fails its check
            for name, doc in facts.items():
                for path, value in fill.leaves(doc):
                    try:
                        expected = fill.lookup(self.lock, path)
                    except KeyError:
                        continue
                    if fill.is_placeholder(expected):
                        checks[name]["verdict"] = "fail"
                        checks[name]["failures"].append(
                            f"{' > '.join(path)}: base-lock says {expected!r}, observed {value!r}")
        failed = sorted(n for n, c in checks.items() if c["verdict"] != "pass")
        self.report = {"schema": "saea-engineering-base-verification-v1", "selected_checks": CHECKS,
                       "skipped_checks": {}, "checks": checks, "failed_checks": failed,
                       "verdict": "fail" if failed else "pass",
                       "inputs": {"profile_id": PROFILE, "base_lock_sha256": sha(self.lock_raw)}}
        report_raw = write_json(self.report_path, self.report)
        r0 = {"gate-cards": {}, "disk": {}, "deploy-up": {}, "health": {}, "deploy-status": {}, "startup-lines": {},
              "receipt": {"receipt": str(self.receipt), "sha256": sha(self.receipt.read_bytes())}}
        if not bootstrap:
            r0["verify-base"] = {"report": str(self.report_path), "sha256": sha(report_raw), "verdict": "11/11 PASS"}
            r0["smoke"] = {"results": {}}
            r0["manifest"] = {"manifest": str(self.chain / "manifest.json")}
        else:
            write_json(steps / "verify-base.failed.json", {"step": "verify-base", "reason": "exit 1; failed checks"})
        for name, decision in r0.items():
            write_json(steps / f"{name}.done.json", {"step": name, "decision": decision})
        if bootstrap:
            return
        write_json(self.chain / "manifest.json", {
            "schema": "saea-record-run-manifest-v1", "profile": PROFILE, "profile_sha256": "d" * 64,
            "max_num_seqs_ceiling": 32, "admitted_level": 32, "git_head": "1" * 40, "host": "A800I-03",
            "client_seed": "20261028", "client_decoding_seed_default": 20260913,
            "server_seeds": {"i0": "0", "i1": "0"}, "probe": {"decoding_seed": 777, "send": "sequential",
                                                             "bodies_sha256": "e" * 64},
            "receipt": str(self.receipt), "verification": str(self.report_path),
            "npu_smi_before": {"0": 3440, "1": 3441}, "npu_smi_after": {"0": 61409, "1": 61352}, "run_list": run_list})
        self.rows = {}
        for run in RUNS:
            for level in run["levels"]:
                worst = bool(run["admissibility"])
                bad = run["id"] == "R-A3-b6144" and level == 32
                row = level_row(self.logs, run, level, passes=not bad, worst=worst,
                                admissible=(not bad) if worst else None,
                                reasons=["6 preemption(s)", "admissibility: KV peak 100.0 % > 90 % on i0"] if bad else ())
                self.rows[(run["id"], level)] = row
                write_json(steps / f"{run['id']}-L{level}.done.json",
                           {"step": f"{run['id']}-L{level}", "decision": {"level": level, "passes": row["passes"], "row": row}})

    def args(self, *extra: str) -> list[str]:
        return ["--chain-dir", str(self.chain), "--base-lock", str(self.lock_path), "--speed-baseline", str(self.s8_path),
                *extra]

    def lock_now(self) -> dict:
        return json.loads(self.lock_path.read_text())


def run(fx: Fixture, *extra: str) -> int:
    return fill.main(fx.args(*extra))


# ---------------------------------------------------------------- the normal path
def test_fills_every_placeholder_of_the_chain_profile_and_nothing_else(tmp_path, capsys):
    fx = Fixture(tmp_path)
    assert run(fx) == 0
    lock = fx.lock_now()
    assert lock["image"]["created"] == "2026-10-27T12:02:29Z"
    assert lock["protocol"]["protocol_sha256"] == PROTO
    assert lock["receipt"]["path"] == str(fx.receipt)
    assert lock["template"]["tails"] == {"thinking_true": "<|im_start|>assistant\n", "thinking_false": "</think>\n\n"}
    assert lock["hosts"]["A800I-03"]["cann_files"]["/etc/ascend_install.info"] == {"bytes": 123, "sha256": "c" * 64}
    assert lock["profiles"][PROFILE]["instances"]["i1"]["engine"]["kv_tokens"] == 551232
    assert lock["profiles"][PROFILE]["instances"]["i0"]["engine"]["cudagraph_capture_sizes"] == [1, 2, 4, 8, 16, 24, 32]
    # the other profile's placeholders and every non-placeholder leaf are untouched
    assert lock["profiles"][OTHER] == fx.lock["profiles"][OTHER]
    assert lock["_skeleton"] == fx.lock["_skeleton"]
    before = dict(fill.leaves(fx.lock))
    for path, value in fill.leaves(lock):
        if not fill.is_placeholder(before[path]):
            assert value == before[path], path
    # canonical form, so the diff is leaf-only
    assert fx.lock_path.read_text() == fill.canonical(lock)


def test_receipt_records_chain_inputs_and_filled_fields(tmp_path):
    fx = Fixture(tmp_path)
    assert run(fx) == 0
    receipt = json.loads((fx.chain / "fill-receipt.json").read_text())
    assert receipt["schema"] == fill.RECEIPT_SCHEMA and receipt["mode"] == "record"
    assert receipt["chain_dir"] == str(fx.chain.resolve()) and receipt["profile"] == PROFILE
    filled = {tuple(f["field"]) for f in receipt["filled_fields"]}
    assert ("protocol", "protocol_sha256") in filled and len(filled) == 13
    assert all("checks." in f["source"] for f in receipt["filled_fields"])
    assert receipt["inputs"][str(fx.report_path.resolve())] == sha(fx.report_path.read_bytes())
    assert receipt["inputs"][str(fx.lock_path.resolve())] == sha(fx.lock_raw)
    out = receipt["outputs"][str(fx.lock_path.resolve())]
    assert out == {"before": sha(fx.lock_raw), "after": sha(fx.lock_path.read_bytes())}
    assert len(receipt["placeholders_left_other_profiles"]) == 3 and receipt["placeholders_left_unmapped"] == []
    assert receipt["verification"]["verdict"] == "11/11 PASS" and receipt["s8_rows"] == 3


def test_s8_block_is_inserted_before_the_branch_tables_with_one_row_per_level(tmp_path):
    fx = Fixture(tmp_path)
    assert run(fx) == 0
    text = fx.s8_path.read_text()
    assert text.count(fill.BEGIN) == 1 and text.index(fill.BEGIN) < text.index("Branch (ii)/(iii)/(iv)")
    assert text.replace(text[text.index(fill.BEGIN):text.index(fill.END) + len(fill.END) + 2], "") == fx.s8_text
    block = text[text.index(fill.BEGIN):text.index(fill.END)]
    rows = [line for line in block.splitlines() if line.startswith("| R-")]
    assert len(rows) == 3
    r_a2 = rows[0].split(" | ")
    assert r_a2[1:4] == ["32", "64", "3 pass(es), 576/576 ok"]
    assert r_a2[6] == "1,658.4" and r_a2[7] == "829.2" and r_a2[8] == "28.80"  # 576 / 600 s x 60 / 2
    assert "16/16, 15/16" in rows[0] and rows[0].endswith("| PASS |")
    assert "0/0" in rows[0] and "77.1/75.7 %" in rows[0]
    assert "FAIL: 6 preemption(s); admissibility: KV peak 100.0 % > 90 % on i0 (admissible: no)" in rows[1]
    assert rows[2].endswith("| PASS (admissible: yes) |") and "worst case `--min-tokens 7168`, max_tokens 7168, budget 6144" in rows[2]
    assert f"`protocol_sha256` `{PROTO}`" in block and "11/11 PASS" in block and "server `--seed` 0" in block


def test_dry_run_prints_the_field_map_and_writes_nothing(tmp_path, capsys):
    fx = Fixture(tmp_path)
    assert run(fx, "--dry-run") == 0
    out = capsys.readouterr().out
    doc = json.loads(out[:out.index(fill.BEGIN)])
    fields = {f["field"]: f for f in doc["field_map"]}
    assert fields["protocol > protocol_sha256"]["value"] == PROTO
    assert fields["protocol > protocol_sha256"]["source"].endswith("checks.protocol.facts")
    assert fields[f"profiles > {PROFILE} > instances > i0 > engine > kv_tokens"]["source"].endswith("checks.kv.facts")
    assert doc["unmapped"] == [] and doc["other_profiles"] == 3
    assert fx.lock_path.read_bytes() == fx.lock_raw and fx.s8_path.read_text() == fx.s8_text
    assert not (fx.chain / "fill-receipt.json").exists()


def test_rerun_is_idempotent_and_the_receipt_is_never_overwritten(tmp_path, capsys):
    fx = Fixture(tmp_path)
    assert run(fx) == 0
    lock_after, s8_after = fx.lock_path.read_bytes(), fx.s8_path.read_bytes()
    assert run(fx) == 2 and "fill-receipt.json exists" in capsys.readouterr().err
    second = tmp_path / "second-receipt.json"
    assert run(fx, "--receipt-out", str(second)) == 0
    assert fx.lock_path.read_bytes() == lock_after and fx.s8_path.read_bytes() == s8_after
    assert json.loads(second.read_text())["filled_fields"] == []


def test_identical_inputs_give_identical_outputs(tmp_path):
    fx = Fixture(tmp_path)
    outputs = []
    for _ in range(2):
        fx.lock_path.write_bytes(fx.lock_raw)
        fx.s8_path.write_text(fx.s8_text, encoding="utf-8")
        (fx.chain / "fill-receipt.json").unlink(missing_ok=True)
        assert run(fx) == 0
        outputs.append((fx.lock_path.read_bytes(), fx.s8_path.read_bytes(), (fx.chain / "fill-receipt.json").read_bytes()))
    assert outputs[0] == outputs[1]


def test_kv_tokens_within_one_percent_is_not_a_difference(tmp_path):
    lock = skeleton_lock()
    lock["profiles"][PROFILE]["instances"]["i0"]["engine"]["kv_tokens"] = 553000  # observed 551000: 0.36 %
    fx = Fixture(tmp_path, lock=lock)
    assert run(fx) == 0
    assert fx.lock_now()["profiles"][PROFILE]["instances"]["i0"]["engine"]["kv_tokens"] == 553000


# ---------------------------------------------------------------- refusals (nothing written)
def assert_untouched(fx: Fixture) -> None:
    assert fx.lock_path.read_bytes() == fx.lock_raw and fx.s8_path.read_text() == fx.s8_text
    assert not (fx.chain / "fill-receipt.json").exists()


@pytest.mark.parametrize("step", fill.R0_STEPS)
def test_refuses_when_an_r0_step_is_missing(tmp_path, capsys, step):
    fx = Fixture(tmp_path)
    (fx.chain / "steps" / f"{step}.done.json").unlink()
    assert run(fx) == 2
    err = capsys.readouterr().err
    assert "required R0 step(s) not done" in err and repr(step) in err
    assert_untouched(fx)


def test_refuses_when_verify_base_is_not_eleven_of_eleven(tmp_path, capsys):
    fx = Fixture(tmp_path, verify_failed=("protocol",))
    assert run(fx) == 2
    assert "not 11/11 PASS" in capsys.readouterr().err
    assert_untouched(fx)


def test_refuses_when_fewer_than_eleven_checks_were_selected(tmp_path, capsys):
    fx = Fixture(tmp_path)
    report = json.loads(fx.report_path.read_text())
    report["selected_checks"] = [c for c in CHECKS if c != "weights"]
    report["skipped_checks"] = {"weights": "--rehash-weights not given"}
    raw = write_json(fx.report_path, report)
    write_json(fx.chain / "steps" / "verify-base.done.json", {"decision": {
        "report": str(fx.report_path), "sha256": sha(raw), "verdict": "11/11 PASS"}})
    assert run(fx) == 2
    assert "10 of 11 checks selected" in capsys.readouterr().err
    assert_untouched(fx)


def test_refuses_when_the_report_hash_differs_from_the_step(tmp_path, capsys):
    fx = Fixture(tmp_path)
    fx.report_path.write_text(fx.report_path.read_text() + " ")
    assert run(fx) == 2
    assert "differs from the verify-base step" in capsys.readouterr().err
    assert_untouched(fx)


def test_never_overwrites_a_non_placeholder_leaf(tmp_path, capsys):
    lock = skeleton_lock()
    lock["image"]["created"] = "2026-01-01T00:00:00Z"  # a filled value that the observation contradicts
    fx = Fixture(tmp_path, lock=lock)
    assert run(fx) == 2
    assert "image > created: record '2026-01-01T00:00:00Z'" in capsys.readouterr().err
    assert_untouched(fx)


def test_refuses_a_missing_run_level(tmp_path, capsys):
    fx = Fixture(tmp_path)
    (fx.chain / "steps" / "R-A3-b6144-L24.done.json").unlink()
    assert run(fx) == 2
    assert "['R-A3-b6144-L24']" in capsys.readouterr().err
    assert_untouched(fx)


def test_refuses_a_level_report_whose_hash_differs_from_its_row(tmp_path, capsys):
    fx = Fixture(tmp_path)
    (fx.logs / fx.rows[("R-A2-L32", 32)]["report"]).write_text("{}")
    assert run(fx) == 2
    assert "level report" in capsys.readouterr().err
    assert_untouched(fx)


def test_refuses_a_running_chain(tmp_path, capsys):
    fx = Fixture(tmp_path)
    write_json(fx.chain / "chain-status.json", {"state": "running"})
    assert run(fx) == 2
    assert "still running" in capsys.readouterr().err


def test_refuses_conflicting_observations(tmp_path, capsys):
    facts = observed_facts(tmp_path / "logs")
    facts["deployment"]["profiles"][PROFILE]["instances"]["i0"]["engine"] = {"block_size": 64}
    fx = Fixture(tmp_path, facts=facts)
    assert run(fx) == 2
    assert "conflicting observations" in capsys.readouterr().err


def test_refuses_a_non_canonical_record(tmp_path, capsys):
    fx = Fixture(tmp_path)
    fx.lock_path.write_text(json.dumps(fx.lock))
    fx.lock_raw = fx.lock_path.read_bytes()
    assert run(fx) == 2
    assert "canonical form" in capsys.readouterr().err


def test_refuses_a_different_existing_block(tmp_path, capsys):
    fx = Fixture(tmp_path)
    fx.s8_text = fx.s8_text.replace("Branch (ii)", f"{fill.BEGIN}\nold\n{fill.END}\nBranch (ii)")
    fx.s8_path.write_text(fx.s8_text)
    assert run(fx) == 2
    assert "DIFFERENT generated block" in capsys.readouterr().err
    assert_untouched(fx)


def test_refuses_a_protocol_hash_that_is_not_the_runtime_receipts(tmp_path, capsys):
    fx = Fixture(tmp_path)
    write_json(fx.receipt, {"protocol_sha256": "0" * 64})
    assert run(fx) == 2
    assert "is not the runtime receipt's" in capsys.readouterr().err


def test_skipped_and_died_levels_are_rendered_as_data(tmp_path):
    fx = Fixture(tmp_path)
    steps = fx.chain / "steps"
    write_json(steps / "R-A3-b6144-L24.done.json", {"decision": {"skipped": "level 32 failed (on_fail stop_run)"}})
    (steps / "R-A2-L32-L32.done.json").unlink()
    write_json(steps / "R-A2-L32-L32.failed.json", {"failing_reasons": ["endpoint_died:i1"], "passes": False})
    assert run(fx) == 0
    text = fx.s8_path.read_text()
    block = text[text.index(fill.BEGIN):text.index(fill.END)]
    rows = [line for line in block.splitlines() if line.startswith("| R-")]
    assert len(rows) == 3 and rows[0].endswith("| FAIL: endpoint_died:i1 |") and rows[2].endswith("| not run: level 32 failed (on_fail stop_run) |")
    assert all(len(line.split(" | ")) == len(fill.HEADER) for line in rows)


# ---------------------------------------------------------------- bootstrap (first R0 against the skeleton)
def test_bootstrap_fills_lock_placeholders_from_a_placeholder_only_failure(tmp_path):
    fx = Fixture(tmp_path, bootstrap=True)
    assert run(fx) == 2  # the default mode refuses: verify-base is not done
    assert run(fx, "--bootstrap") == 0
    lock = fx.lock_now()
    assert lock["protocol"]["protocol_sha256"] == PROTO
    assert lock["profiles"][PROFILE]["instances"]["i0"]["engine"]["block_size"] == 128
    assert fx.s8_path.read_text() == fx.s8_text  # no S8 in bootstrap
    receipt = json.loads((fx.chain / "fill-receipt.json").read_text())
    assert receipt["mode"] == "bootstrap" and not receipt["s8_block_written"]
    assert receipt["verification"]["report"] == str(fx.report_path)
    assert any("re-run verify-base" in note for note in receipt["notes"])


def test_bootstrap_refuses_a_real_mismatch(tmp_path, capsys):
    fx = Fixture(tmp_path, bootstrap=True)
    report = json.loads(fx.report_path.read_text())
    report["checks"]["protocol"]["verdict"] = "fail"
    report["checks"]["protocol"]["failures"].append(
        "protocol > decoding > max_tokens: base-lock says 7168, observed 4096")
    write_json(fx.report_path, report)
    assert run(fx, "--bootstrap") == 2
    assert "not placeholder mismatches" in capsys.readouterr().err
    assert_untouched(fx)


def test_bootstrap_refuses_a_record_changed_since_the_verification(tmp_path, capsys):
    fx = Fixture(tmp_path, bootstrap=True)
    lock = fx.lock_now()
    lock["_skeleton"]["status"] = "edited after R0"
    fx.lock_raw = write_json(fx.lock_path, lock)
    assert run(fx, "--bootstrap") == 2
    assert "did not run against the record as it is now" in capsys.readouterr().err

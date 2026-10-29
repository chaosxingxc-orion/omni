#!/usr/bin/env python3
"""Greedy-decoding self-agreement probe (plan 3.9, D1 and D3).  PRELIMINARY tooling: no dataset row, no exposure entry.

Sends the same 16 fixed requests that e_crash_chain.py's determinism floor uses (8 trajectories x 2 contacts,
temperature 0, decoding seed 777, max_tokens 64, logprobs on) to ONE endpoint and records full text, token ids (when the
server returns them) and logprobs; compares two recorded runs.

  send     --url URL --out FILE (--sequential | --burst) [--bodies-file F] [--top-logprobs 2] [--no-token-ids]
  compare  A.json B.json [--out FILE]           byte-identical count, mismatched indices, first divergent token,
                                                max |delta logprob|, top-2 gap at the divergence
  matrix   --run label=path ... [--out FILE]    all pairs, grouped by mode (D3)
  d3       --url URL --out FILE --run-dir DIR   preflight idle check, 3 sequential + 3 burst runs, matrix, one JSON
                                                (run on the NPU host, pinned, detached; never starts or stops a server)

Legacy files (E1 eq outputs) carry no token ids and no top-logprobs: the first divergent position then falls back to a
sampled-logprob proxy (first position with |delta| > --lp-tol) and says so in `divergence_basis`.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import itertools
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.dont_write_bytecode = True
SCRIPTS = Path(__file__).resolve().parent
LP_TOL_DEFAULT = 1e-2
TOKEN_ID_RE = re.compile(r"^token_id:(\d+)$")


# ----------------------------------------------------------------------------------------------- pure compare logic
def _key(r: dict) -> tuple:
    """Same identity key as e_crash_chain.compare_eq: text, finish reason, completion token count."""
    return (r.get("reasoning"), r.get("content"), r.get("finish_reason"), r.get("completion_tokens"))


def _tokens(r: dict):
    """Comparable token sequence of one result: ints if ids were returned, else raw token strings, else None."""
    if r.get("token_ids"):
        return list(r["token_ids"])
    if r.get("tokens"):
        return list(r["tokens"])
    return None


def _text(r: dict) -> str:
    return (r.get("reasoning") or "") + "\u0000" + (r.get("content") or "")


def _first_diff(a, b):
    """Index of the first differing element, or min(len) if one is a strict prefix of the other, or None if equal."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def _gap(top) -> float | None:
    """logprob(top-1) - logprob(top-2) of one position's candidate list, None if fewer than two candidates."""
    if not top or len(top) < 2:
        return None
    vals = sorted((c["logprob"] for c in top), reverse=True)
    return vals[0] - vals[1]


def compare_one(a: dict, b: dict, lp_tol: float = LP_TOL_DEFAULT) -> dict:
    """Detail for one request pair (index-aligned)."""
    out = {"identical": a.get("status") == "ok" and b.get("status") == "ok" and _key(a) == _key(b)}
    la, lb = a.get("logprobs") or [], b.get("logprobs") or []
    if la and len(la) == len(lb):  # same rule as e_crash_chain.compare_eq, so the legacy 1.97 is reproducible
        out["max_abs_logprob_diff"] = max(abs(x - y) for x, y in zip(la, lb))
    else:
        out["max_abs_logprob_diff"] = None
    if out["identical"]:
        return out
    ta, tb = _tokens(a), _tokens(b)
    pos, basis = None, None
    if ta is not None and tb is not None:
        pos, basis = _first_diff(ta, tb), "token_ids" if isinstance(ta[0], int) else "token_strings"
    elif la and lb:
        pos = next((i for i, (x, y) in enumerate(zip(la, lb)) if abs(x - y) > lp_tol), None)
        if pos is None and len(la) != len(lb):
            pos = min(len(la), len(lb))
        basis = f"sampled_logprob_proxy(|delta|>{lp_tol:g})"
    out["first_divergent_token"], out["divergence_basis"] = pos, basis
    ca, cb = _text(a), _text(b)
    out["first_divergent_char"] = _first_diff(ca, cb)
    n = min(len(la), len(lb))
    if pos is not None and n:
        pre = [abs(x - y) for x, y in zip(la[:pos], lb[:pos])]
        out["max_abs_logprob_diff_before_divergence"] = max(pre) if pre else 0.0
        out["logprob_at_divergence"] = [la[pos] if pos < len(la) else None, lb[pos] if pos < len(lb) else None]
    ga = _gap((a.get("top_logprobs") or [None] * (pos + 1))[pos]) if pos is not None and a.get("top_logprobs") and pos < len(a["top_logprobs"]) else None
    gb = _gap((b.get("top_logprobs") or [None] * (pos + 1))[pos]) if pos is not None and b.get("top_logprobs") and pos < len(b["top_logprobs"]) else None
    out["top2_gap_at_divergence"] = [ga, gb] if (ga is not None or gb is not None) else None
    if pos is not None and ta is not None and tb is not None:
        out["tokens_at_divergence"] = [ta[pos] if pos < len(ta) else None, tb[pos] if pos < len(tb) else None]
    return out


def compare_runs(a: list[dict], b: list[dict], lp_tol: float = LP_TOL_DEFAULT) -> dict:
    """Compare two lists of recorded completions (same order)."""
    n = min(len(a), len(b))
    details = {}
    exact, mismatched, diffs = 0, [], []
    for i in range(n):
        d = compare_one(a[i], b[i], lp_tol)
        if d["identical"]:
            exact += 1
        else:
            mismatched.append(i)
            details[str(i)] = d
        if d["max_abs_logprob_diff"] is not None:
            diffs.append(d["max_abs_logprob_diff"])
    return {"requests": n, "count_a": len(a), "count_b": len(b), "byte_identical": exact, "mismatched_indices": mismatched,
            "max_abs_logprob_diff": max(diffs) if diffs else None, "mismatches": details}


def load_results(path: str) -> list[dict]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    return doc["results"] if isinstance(doc, dict) else doc


def matrix(runs: dict[str, list[dict]], modes: dict[str, str], lp_tol: float = LP_TOL_DEFAULT) -> dict:
    """All unordered pairs, grouped: within each mode and across modes."""
    pairs = []
    for x, y in itertools.combinations(runs, 2):
        c = compare_runs(runs[x], runs[y], lp_tol)
        group = f"within-{modes[x]}" if modes[x] == modes[y] else "across-modes"
        pairs.append({"a": x, "b": y, "group": group, "byte_identical": c["byte_identical"], "requests": c["requests"],
                      "mismatched_indices": c["mismatched_indices"], "max_abs_logprob_diff": c["max_abs_logprob_diff"],
                      "first_divergent_token": {i: m.get("first_divergent_token") for i, m in c["mismatches"].items()},
                      "top2_gap_at_divergence": {i: m.get("top2_gap_at_divergence") for i, m in c["mismatches"].items()}})
    return {"pairs": pairs}


def classify(pairs: list[dict]) -> str:
    """D3 rule (plan 3.9): sequential 16/16 across repeats with burst disagreeing = batch composition;
    sequential disagreeing = kernel or cache nondeterminism."""
    seq = [p for p in pairs if p["group"] == "within-sequential"]
    bur = [p for p in pairs if p["group"] == "within-burst"]
    seq_ok = bool(seq) and all(p["byte_identical"] == p["requests"] for p in seq)
    bur_ok = bool(bur) and all(p["byte_identical"] == p["requests"] for p in bur)
    if seq_ok and bur_ok:
        cross = [p for p in pairs if p["group"] == "across-modes"]
        if cross and all(p["byte_identical"] == p["requests"] for p in cross):
            return "deterministic: all runs identical in both modes"
        return "sequential and burst each self-consistent but differ from each other (batch composition, stable per mode)"
    if seq_ok:
        return "batch-composition nondeterminism: sequential identical across repeats, burst disagrees"
    return "kernel-or-cache nondeterminism: sequential repeats disagree"


# ----------------------------------------------------------------------------------------------------- HTTP / bodies
def http_get(url: str, timeout: float = 10.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def metrics_snapshot(url: str) -> dict:
    """running/waiting gauges plus cache and preemption counters (summed over label sets)."""
    text = http_get(url + "/metrics")
    want = re.compile(r"^(vllm:(?:num_requests_running|num_requests_waiting|num_preemptions_total|prefix_cache_(?:queries|hits)_total|"
                      r"mm_cache_(?:queries|hits)_total|external_prefix_cache_(?:queries|hits)_total))\{.*?\}\s+([0-9.eE+-]+)$")
    snap: dict[str, float] = {}
    for line in text.splitlines():
        m = want.match(line)
        if m:
            snap[m.group(1)] = snap.get(m.group(1), 0.0) + float(m.group(2))
    return snap


def build_bodies() -> list[dict]:
    sys.path.insert(0, str(SCRIPTS))
    import e_crash_chain as ec  # the runner that produced the E1 floor probes; its request builder is reused as is
    return ec.eq_bodies(ec.load_sweep_module(), ec.MODEL)


def load_or_build_bodies(path: str | None) -> list[dict]:
    if path and Path(path).exists():
        return json.loads(Path(path).read_text(encoding="utf-8"))
    bodies = build_bodies()
    if path:
        Path(path).write_text(json.dumps(bodies), encoding="utf-8")
    return bodies


def bodies_digest(bodies: list[dict]) -> str:
    return hashlib.sha256(json.dumps(bodies, sort_keys=True).encode("utf-8")).hexdigest()


def post(url: str, body: dict, timeout: float = 900.0) -> dict:
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            doc = json.loads(resp.read().decode("utf-8"))
    except Exception as error:  # noqa: BLE001
        detail = ""
        if hasattr(error, "read"):
            try:
                detail = error.read().decode("utf-8", "replace")[:300]
            except Exception:  # noqa: BLE001
                pass
        return {"status": "error", "error": repr(error)[:300] + detail, "elapsed_s": round(time.time() - t0, 3)}
    choice = (doc.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content = ((choice.get("logprobs") or {}).get("content")) or []
    content = [t for t in content if isinstance(t, dict)]
    raw = [t.get("token") for t in content]
    ids = [int(m.group(1)) for m in (TOKEN_ID_RE.match(x or "") for x in raw) if m]
    have_ids = bool(raw) and len(ids) == len(raw)
    top = [[{"token": c.get("token"), "logprob": c.get("logprob")} for c in (t.get("top_logprobs") or [])[:2]] for t in content]
    return {"status": "ok", "reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "",
            "content": msg.get("content") or "", "finish_reason": choice.get("finish_reason"),
            "completion_tokens": (doc.get("usage") or {}).get("completion_tokens"),
            "logprobs": [t.get("logprob") for t in content],
            "token_ids": ids if have_ids else None, "tokens": None if have_ids else raw,
            "top_logprobs": top if any(top) else None, "elapsed_s": round(time.time() - t0, 3)}


def send(url: str, bodies: list[dict], mode: str, top_logprobs: int, token_ids: bool) -> list[dict]:
    prepared = []
    for b in bodies:
        b = dict(b)
        if top_logprobs:
            b["top_logprobs"] = top_logprobs
        if token_ids:
            b["return_tokens_as_token_ids"] = True
        prepared.append(b)
    results: list[dict | None] = [None] * len(prepared)
    if mode == "sequential":
        for i, b in enumerate(prepared):
            results[i] = post(url, b)
    else:
        with cf.ThreadPoolExecutor(max_workers=len(prepared)) as pool:
            futures = {pool.submit(post, url, b): i for i, b in enumerate(prepared)}
            for f in cf.as_completed(futures):
                results[futures[f]] = f.result()
    for i, r in enumerate(results):
        r["index"] = i
    return results  # type: ignore[return-value]


def write_atomic(path: str, text: str) -> None:
    tmp = f"{path}.tmp{os.getpid()}"
    Path(tmp).write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def run_doc(url: str, mode: str, bodies: list[dict], results: list[dict], extra: dict | None = None) -> dict:
    return {"url": url, "mode": mode, "bodies_sha256": bodies_digest(bodies), "n": len(bodies),
            "started_utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()), "results": results, **(extra or {})}


# ----------------------------------------------------------------------------------------------------------- d3
def idle(url: str) -> tuple[bool, dict]:
    s = metrics_snapshot(url)
    return s.get("vllm:num_requests_running", -1) == 0 and s.get("vllm:num_requests_waiting", -1) == 0, s


def cmd_d3(a) -> int:
    url = a.url.rstrip("/")
    run_dir = Path(a.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    ok, snap = idle(url)
    if not ok:
        write_atomic(a.out, json.dumps({"label": "PRELIMINARY", "state": "refused_busy", "metrics": snap}, indent=1) + "\n")
        print("instance busy, refusing", snap)
        return 3
    bodies = load_or_build_bodies(str(run_dir / "bodies.json"))
    order = [("S1", "sequential"), ("S2", "sequential"), ("S3", "sequential"),
             ("B1", "burst"), ("B2", "burst"), ("B3", "burst")]
    runs, modes, log = {}, {}, []
    for label, mode in order:
        ok, before = idle(url)
        if not ok:
            log.append({"label": label, "aborted": "instance became busy", "metrics": before})
            break
        t0 = time.time()
        res = send(url, bodies, mode, a.top_logprobs, not a.no_token_ids)
        _, after = idle(url)
        write_atomic(str(run_dir / f"{label}.json"), json.dumps(run_doc(url, mode, bodies, res), indent=1) + "\n")
        runs[label], modes[label] = res, mode
        log.append({"label": label, "mode": mode, "wall_s": round(time.time() - t0, 2),
                    "ok": sum(r["status"] == "ok" for r in res), "metrics_before": before, "metrics_after": after})
        print(label, mode, log[-1]["ok"], "ok", log[-1]["wall_s"], "s", flush=True)
    m = matrix(runs, modes, a.lp_tol) if len(runs) > 1 else {"pairs": []}
    summary = {}
    for g in ("within-sequential", "within-burst", "across-modes"):
        ps = [p for p in m["pairs"] if p["group"] == g]
        summary[g] = {"pairs": len(ps), "identical_counts": {f"{p['a']}-{p['b']}": p["byte_identical"] for p in ps}}
    doc = {"label": "PRELIMINARY", "state": "done" if len(runs) == len(order) else "partial", "url": url,
           "bodies_sha256": bodies_digest(bodies), "order": [o[0] for o in order], "run_log": log,
           "matrix": m["pairs"], "summary": summary, "classification": classify(m["pairs"]) if m["pairs"] else None,
           "note": "run order S1 S2 S3 B1 B2 B3 on a warm instance; S1 may meet cold prefix and multimodal caches"}
    write_atomic(a.out, json.dumps(doc, indent=1) + "\n")
    print(doc["classification"])
    return 0


# ----------------------------------------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("send")
    s.add_argument("--url", required=True)
    s.add_argument("--out", required=True)
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--sequential", action="store_true")
    g.add_argument("--burst", action="store_true")
    s.add_argument("--bodies-file", help="cache of the request bodies: written when missing, loaded when present")
    s.add_argument("--top-logprobs", type=int, default=2)
    s.add_argument("--no-token-ids", action="store_true")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--out")
    c.add_argument("--lp-tol", type=float, default=LP_TOL_DEFAULT)
    m = sub.add_parser("matrix")
    m.add_argument("--run", action="append", required=True, help="label=path (mode read from the file, else from the label letter)")
    m.add_argument("--out")
    m.add_argument("--lp-tol", type=float, default=LP_TOL_DEFAULT)
    d = sub.add_parser("d3")
    d.add_argument("--url", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--run-dir", required=True)
    d.add_argument("--top-logprobs", type=int, default=2)
    d.add_argument("--no-token-ids", action="store_true")
    d.add_argument("--lp-tol", type=float, default=LP_TOL_DEFAULT)
    a = p.parse_args(argv)
    if a.cmd == "send":
        bodies = load_or_build_bodies(a.bodies_file)
        mode = "sequential" if a.sequential else "burst"
        res = send(a.url.rstrip("/"), bodies, mode, a.top_logprobs, not a.no_token_ids)
        write_atomic(a.out, json.dumps(run_doc(a.url, mode, bodies, res), indent=1) + "\n")
        print(f"{mode}: {sum(r['status'] == 'ok' for r in res)}/{len(res)} ok -> {a.out}")
        return 0
    if a.cmd == "compare":
        out = compare_runs(load_results(a.a), load_results(a.b), a.lp_tol)
        text = json.dumps(out, indent=1) + "\n"
        if a.out:
            write_atomic(a.out, text)
        print(text, end="")
        return 0
    if a.cmd == "matrix":
        runs, modes = {}, {}
        for spec in a.run:
            label, path = spec.split("=", 1)
            doc = json.loads(Path(path).read_text(encoding="utf-8"))
            runs[label] = doc["results"] if isinstance(doc, dict) else doc
            modes[label] = (doc.get("mode") if isinstance(doc, dict) else None) or ("burst" if label.upper().startswith("B") else "sequential")
        out = matrix(runs, modes, a.lp_tol)
        out["classification"] = classify(out["pairs"])
        text = json.dumps(out, indent=1) + "\n"
        if a.out:
            write_atomic(a.out, text)
        print(text, end="")
        return 0
    return cmd_d3(a)


if __name__ == "__main__":
    sys.exit(main())

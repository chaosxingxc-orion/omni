#!/usr/bin/env python3
"""slurp_tfrl_v2_port_vote_score.py -- score the derived `vote` column of the T construction read so the construction gain is a number (plan section 13 ruling 3 (b)).

The construction stage of ``slurp_tfrl_v2_port_read.py`` derives, model-free, the vote over the four sampled trajectories of each usable row and writes it, write-once, as
``vote-answers-i<t>-<H1|H2>.jsonl`` (schema ``saea-tfrl-v2-vote-answers-v1``). Those files are ANSWERS, not scores. This driver turns each into a complete-roster derived
outputs copy (``{"sample_id", "text"}`` rows, the canonical rendering of ``stage3_rollout_score.render_answer``; a roster row with no vote -- a row the construction read could not
use -- gets the canonical empty answer and is never read, because the gain is taken over the usable rows only), scores it ONCE through the registered official scorer
(``core.scoring_batch.score_complete_derived_copy``: bound to the verified, finalized parent bundle of that half's construction flight, an exclusive source-bound receipt, a durable
start marker before the scorer is contacted), and writes the column ``arm-vote-<scorer>-scores.json`` into the half's columns directory -- the exact file name and shape the reader
loads for any arm (``slurp_l338_column_read.COLUMN_SCORE_NAME``).

Then it computes the construction gain, SLU-F1(vote) - SLU-F1(lock) in pp pooled over the usable rows of both halves, with the reader's own function, and writes the write-once
``construction-gain-i<t>.json`` (gain, the gate word CONTINUE / STOP, the inputs and their hashes). The reader's Fold B stage prefers that file over the null gain of the
construction read (which is write-once and was written before the vote column existed): with the vote column scored the stop word is CONTINUE (the gate PASSES: gain >= +1.0 pp
after rounding to 6 decimals) or STOP, never UNDETERMINED. The `lock` column of each half must already be scored (stage3_rollout_score.py --mode supply-variants on the half's flight,
into the same columns directory); until it is, the driver scores the vote column and reports the gain as null.

Nothing here contacts a model, a server, a tokenizer or an encoder; the only contact is the official scorer on a derived copy (one per half). No answer text is printed.

    python -B scripts/analysis/slurp_tfrl_v2_port_vote_score.py --iteration 1 --data-root $SPEECHRL_DATA_DIR [--half R01C-H1 | --half both] [--scorer slurp-slu-official-train]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
for _path in (str(_REPO / "src"), str(_HERE), str(_REPO / "scripts" / "flights" / "nx")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import slurp_tfrl_v2_port_read as reader  # noqa: E402
import stage3_rollout_score as s3  # noqa: E402
import tfrl_v2_common as cm  # noqa: E402
import tfrl_v2_port_family as tf  # noqa: E402
import tfrl_v2_port_rails as rails  # noqa: E402

ReadError = reader.ReadError
GAIN_SCHEMA = "saea-tfrl-v2-construction-gain-v1"
VOTE_ARM = reader.VOTE_ARM
COPY_NAME = "arm-{arm}-outputs.jsonl"
MANIFEST_NAME = "vote-render-manifest.json"
GAIN_NAME = "construction-gain-i{iteration}.json"
HALF_KEY = {cm.HALF_H1: "ch1", cm.HALF_H2: "ch2"}

#: ``(outputs_copy, result_path) -> official scorer result`` -- injected by tests; the default is the bound derived-copy scorer.
ScoreFn = Callable[[Path, Path], Mapping[str, Any]]


def vote_answers_path(root: Path, iteration: int, half: str) -> Path:
    return Path(root) / f"vote-answers-i{iteration}-{half.split('-')[1]}.jsonl"


def load_votes(path: Path) -> dict[str, dict]:
    """``{row_id: vote record}`` of one half's ``vote-answers`` file; refuses a wrong schema, a duplicate row and a malformed answer."""

    votes: dict[str, dict] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("schema") != reader.VOTE_SCHEMA:
            raise ReadError(f"{path}: a vote line is not {reader.VOTE_SCHEMA}")
        row_id = record.get("row_id")
        entities = record.get("entities")
        if (not isinstance(row_id, str) or not row_id or row_id in votes or not isinstance(record.get("scenario"), str) or not isinstance(record.get("action"), str)
                or not isinstance(entities, list) or any(not isinstance(e, Mapping) or set(e) != {"type", "filler"} for e in entities)):
            raise ReadError(f"{path}: a vote line is malformed or repeats a row")
        votes[row_id] = record
    return votes


def vote_copy_rows(votes: Mapping[str, Mapping], roster_ids: Sequence[str]) -> list[dict]:
    """The derived outputs copy: one ``{"sample_id", "text"}`` row per roster id, sorted; the canonical answer for a voted row and the canonical empty answer otherwise.

    A vote row outside the roster is refused (the copy must be exactly the parent roster's column)."""

    stray = sorted(set(votes) - set(roster_ids))
    if stray:
        raise ReadError(f"{len(stray)} vote rows are not roster rows (first: {stray[0]})")
    return [{"sample_id": row_id, "text": s3.render_answer(votes[row_id]) if row_id in votes else s3.EMPTY_ANSWER} for row_id in sorted(roster_ids)]


def render_vote_copy(votes_path: Path, roster: Mapping[str, Any], out_dir: Path, *, half: str, iteration: int) -> dict:
    """Write ``arm-vote-outputs.jsonl`` for one half (idempotent: an existing copy with the same bytes is kept, a different one is refused) and its manifest entry."""

    roster_ids = [str(item["sample_id"]) for item in roster["items"]]
    votes = load_votes(votes_path)
    rows = vote_copy_rows(votes, roster_ids)
    text = "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / COPY_NAME.format(arm=VOTE_ARM)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ReadError(f"{path} exists with different bytes (the vote copy is write-once)")
    else:
        path.write_text(text, encoding="utf-8", newline="\n")
    return {"half": half, "iteration": iteration, "path": str(path), "sha256": digest, "rows": len(rows), "voted_rows": len(votes),
            "placeholder_rows": len(rows) - len(votes), "vote_answers_sha256": s3.sha256_file(Path(votes_path))}


def bound_scorer(*, parent_manifest: Path, entry: Mapping[str, Any], scorer_id: str, lock: Mapping[str, object], data_root: Path, roster_sha256: str) -> ScoreFn:
    """The production ``ScoreFn``: the official scorer on the derived copy, bound to the verified parent bundle of the half's construction flight."""

    def score(outputs_copy: Path, result_path: Path) -> Mapping[str, Any]:
        from speech_aware_evidence_acquisition.core.scoring_batch import score_complete_derived_copy  # noqa: PLC0415

        return score_complete_derived_copy(
            parent_manifest=Path(parent_manifest), outputs_copy=Path(outputs_copy), expected_copy_sha256=entry["sha256"], scorer_id=scorer_id, result_path=Path(result_path),
            lock=lock, data_root=Path(data_root),
            column_binding={"column": VOTE_ARM, "half": entry["half"], "iteration": entry["iteration"], "vote_answers_sha256": entry["vote_answers_sha256"],
                            "roster_sha256": roster_sha256, "visit": "derived-vote-over-the-sampled-trajectories"},
        )

    return score


def score_vote_column(entry: Mapping[str, Any], columns: Path, scorer_id: str, score_fn: ScoreFn) -> dict:
    """Score the half's vote copy once; return the per-row table the reader loads (``reader._load_scores`` shape). Reads back an existing column file instead of re-scoring."""

    result_path = Path(columns) / reader.COLUMN_SCORE_NAME.format(arm=VOTE_ARM, scorer=scorer_id)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if result_path.is_file():
        document = json.loads(result_path.read_text(encoding="utf-8"))
    else:
        result = score_fn(Path(entry["path"]), result_path)
        if not result_path.is_file():                       # an injected scorer writes nothing; the bound scorer has written the file with its receipt
            result_path.write_text(json.dumps({scorer_id: result}, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        document = json.loads(result_path.read_text(encoding="utf-8"))
    rows = reader.base.rows_table(document, scorer_id)
    expected = {str(i["sample_id"]) for i in json.loads(Path(entry["roster_path"]).read_text(encoding="utf-8"))["items"]}
    if set(rows) != expected:
        raise ReadError(f"{result_path}: the scored vote column is not the half's complete roster")
    return rows


def gain_document(iteration: int, vote_rows: Mapping | None, lock_rows: Mapping | None, usable_ids: Sequence[str], inputs: Mapping) -> dict:
    """``construction-gain-i<t>.json``: the reader's own gain and gate over the usable rows of both halves."""

    gain = reader.construction_gain(vote_rows, lock_rows, usable_ids)
    return {"schema": GAIN_SCHEMA, "status": "READING (development numbers; claims_benefit false)", "read_run_id": tf.READ_RUN_ID, "iteration": iteration,
            "construction_gain": gain, "stop": reader.stop_decision(iteration, gain["gain_pp"], None), "inputs": dict(inputs),
            "note": ("the vote column was scored on a derived copy by the official scorer; the gate word is CONTINUE (pass) or STOP; the Fold B read takes the gain from here "
                     "because construction-i<t>.json is write-once and was written before the column existed")}


def run(args: argparse.Namespace, score_fn_for: Callable[[str, Mapping[str, Any], str], ScoreFn] | None = None) -> dict:
    root = args.analysis_root or (args.data_root / rails.ANALYSIS_ROOT)
    halves = cm.HALVES if args.half == "both" else (args.half,)
    gain_path = root / GAIN_NAME.format(iteration=args.iteration)
    if gain_path.exists() and args.half == "both":
        raise ReadError(f"{gain_path} exists (write-once)")
    entries, vote_rows, lock_rows, usable, inputs = {}, {}, {}, [], {}
    lock_complete = True
    for half in halves:
        votes_path = vote_answers_path(root, args.iteration, half)
        if not votes_path.is_file():
            raise ReadError(f"missing {votes_path}; run the construction stage of slurp_tfrl_v2_port_read.py first")
        roster_path = args.repo_root / cm.HALF_ROSTER_RELPATH[half]
        roster = json.loads(roster_path.read_text(encoding="utf-8"))
        columns = reader.columns_dir(args.data_root, args.iteration, half)
        entry = render_vote_copy(votes_path, roster, columns, half=half, iteration=args.iteration)
        entry["roster_path"] = str(roster_path)               # read back by score_vote_column; dropped from the printed summary
        spec = tf.CELLS[HALF_KEY[half]]
        parent = reader.base.run_paths(args.data_root, spec["run"], spec["attempt"])["manifest"]
        score_fn = (score_fn_for(half, entry, args.scorer) if score_fn_for is not None
                    else bound_scorer(parent_manifest=parent, entry=entry, scorer_id=args.scorer, lock=args.lock, data_root=args.data_root, roster_sha256=s3.sha256_file(roster_path)))
        rows = score_vote_column(entry, columns, args.scorer, score_fn)
        entries[half] = entry
        usable_half = list(load_votes(votes_path))
        usable += usable_half
        vote_rows.update({i: rows[i] for i in usable_half})
        inputs[f"vote_answers:{half}"] = reader._input(votes_path, root)
        inputs[f"vote_column:{half}"] = reader._input(columns / reader.COLUMN_SCORE_NAME.format(arm=VOTE_ARM, scorer=args.scorer), args.data_root)
        lock = reader._load_scores(columns, tf.DELIVERING, args.scorer)
        if lock is None:
            lock_complete = False
        else:
            lock_rows.update(lock)
            inputs[f"lock_column:{half}"] = reader._input(columns / reader.COLUMN_SCORE_NAME.format(arm=tf.DELIVERING, scorer=args.scorer), args.data_root)
    summary = {"halves": {h: {k: v for k, v in e.items() if k != "roster_path"} for h, e in entries.items()}}
    if args.half != "both" or not lock_complete:
        summary.update(gain_pp=None, gate=reader.WORD_UNDETERMINED, written=None,
                       note="the gain needs both halves' vote and lock columns" if args.half != "both" else "the lock column of a half is not scored yet; score it with stage3_rollout_score.py --mode supply-variants")
        return summary
    document = gain_document(args.iteration, vote_rows, lock_rows, usable, inputs)
    reader._write_once(gain_path, reader._json_text(document))
    summary.update(gain_pp=document["construction_gain"]["gain_pp"], gate=document["stop"]["gate"]["word"], written=str(gain_path))
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--iteration", type=int, default=tf.ITERATION)
    parser.add_argument("--half", choices=(*cm.HALVES, "both"), default="both")
    parser.add_argument("--data-root", type=Path, default=Path(reader.hist._data_root_env()))
    parser.add_argument("--repo-root", type=Path, default=_REPO)
    parser.add_argument("--analysis-root", type=Path, default=None, help=f"default <data root>/{rails.ANALYSIS_ROOT}")
    parser.add_argument("--scorer", default=reader.SCORER)
    args = parser.parse_args(argv)
    try:
        from speech_aware_evidence_acquisition.data.lock import load_lock, umbrella_lock_path  # noqa: PLC0415

        args.lock = load_lock(umbrella_lock_path(args.repo_root))
        print(json.dumps(run(args), sort_keys=True))
        return 0
    except (ReadError, FileExistsError) as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

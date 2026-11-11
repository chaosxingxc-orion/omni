# S1 type evaluation

These CPU-only tools evaluate native `s1-handoff-v1` aggregates without model calls. Gold is a separate JSONL file of `{sample_id, gold_types}` objects. Folds are a JSON array of `{fold_id, types}` objects, or a knowledge object with a `folds` field. Each type belongs to exactly one fold.

A prediction contains `sample_id`, `complete`, `selected_types`, and `s1`: one result per fold with `branch`, `status`, `selected_types`, and optional `stage`, `sample_id`, `finish_reason`, `request_id`, `usage`, `stage_timing`, and `transport_timing` fields. Only `valid_empty` and `valid_nonempty` count as complete. Missing, failed, truncated, unknown, or partial folds retain the full reference denominator and cannot earn exact-set or complete-coverage credit.

```bash
python scripts/analysis/s1_stage_evaluation/score.py \
  --gold /data/gold.jsonl --folds /data/folds.json \
  --predictions /data/aggregates.jsonl --out /data/evaluation

python scripts/analysis/s1_stage_evaluation/collect.py \
  --gold /data/gold.jsonl --folds /data/folds.json \
  --rows /data/lane-a/rows --rows /data/lane-b/rows \
  --rows /data/lane-c/rows --out /data/evaluation
```

Both entrypoints also support `python -m scripts.analysis.s1_stage_evaluation.<score|collect>` from the study root. The collector accepts directories of per-sample JSON files or explicit JSON/JSONL files. It reuses identical replicas and rejects conflicting sample results. `--runtime-cost` optionally preserves a full runtime cost receipt alongside observed aggregate usage; aggregate rows alone cannot account for paid requests that produced no aggregate.

The summary reports exact type sets, micro precision/recall/F1, macro metrics over reference-supported types, complete coverage of nonempty references, separately labelled all-sample coverage, type-count agreement, unmatched-candidate burden, missing types, empty-reference correctness, per-type/per-fold metrics, protocol states, truncation, and observed costs. `reference_unmatched` describes reference annotation burden and does not establish semantic incorrectness. Missing timing, monetary cost, and runtime estimates remain unknown. Data scope and any historical selection must be stated in the experiment report; learning-set results are not generalization estimates.

Run focused tests with:

```bash
python -m unittest discover -s tests/unit -p 'test_s1_stage_*.py' -v
```

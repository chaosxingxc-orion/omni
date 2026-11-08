# Task expression and geometry bank masks

This offline analysis reproduces the fixed 1080-query, 8,554-member,
60-label experiment. It builds lexical and nearest-neighbor masks, freezes
all rankings before opening gold, then scores the frozen rankings. It needs
Python, NumPy and the adjacent `question_bank_selection*.py` modules.
The original CPU results rejected all 11 variants; this tool does not
establish a model improvement.

Prepare a JSON config with the following structure. Replace every placeholder
with your own file path and SHA-256; all inputs, including gold, are pinned in
the same config before ranking. `geometry_identity` is an optional additional
pin, checked for integrity but not interpreted by the algorithm.

```json
{
  "schema": "task-geometry-mask-v1",
  "output": "OUTPUT_DIRECTORY",
  "df": [8, 10, 12],
  "peer": [2, 3],
  "rare_max": 32,
  "policy": {"description": "fixed policy recorded in mask-summary"},
  "inputs": {
    "matrix": {"path": "MATRIX.npy", "sha256": "SHA256"},
    "members": {"path": "MEMBERS.json", "sha256": "SHA256"},
    "parent": {"path": "PARENT.json", "sha256": "SHA256"},
    "metadata": {"path": "METADATA.jsonl", "sha256": "SHA256"},
    "geometry": {"path": "GEOMETRY.npz", "sha256": "SHA256"},
    "cards": {"path": "CARDS.json", "sha256": "SHA256"},
    "gold": {"path": "GOLD.json", "sha256": "SHA256"}
  }
}
```

`members` is an ordered JSON array of objects with `intent`, `utterance_id`
and `sentence_sha256`. Metadata JSONL contains `slurp_id` and `sentence`;
IDs must match members after conversion to strings. Cards are a label-keyed
object whose entries contain `facet.representation`. Parent contains ordered
`intents`, unique `sample_ids`, and `orders.question-gepa-coverage-gated-depth`
(rows of label indexes). Matrix shape is `(1080, 8554)`, aligned to parent
sample IDs and member order. Geometry NPZ contains `topidx`, indexed by member
and with at least five columns of valid member indexes; it must already encode
the intended source exclusion and neighbor ordering. Gold is a JSON array with
`sample_id` and `gold_intent`, covering the same samples and each of 60 labels.
Input alignment and correct source exclusion are the caller's responsibility;
SHA pins ensure files stay unchanged, not that their contents are correct.

```sh
python scripts/analysis/task_geometry_mask.py rank --config CONFIG.json
python scripts/analysis/task_geometry_mask.py score --config CONFIG.json
```

Run from the study root with a new output directory. Rank checks the exact
parent ranking, saves masks and rankings, and records their hashes in
`rank-freeze.json` without reading gold. Score verifies that freeze, config,
and inputs still match before writing `score.json`. Keep both phases on the
same unchanged config. Ranking uses strictly positive similarities and the
original class coverage denominator; rare classes and otherwise-empty supported
classes are retained. Assertions enforce the fixed experimental dimensions;
run Python without `-O`.

# L338 NPU port, Fold A confirmation: the lever column, preregistration (DRAFT / HELD)

Registered 2026-10-30 at study commit `28bf317600000000000000000000000000000000` by script. Development numbers only.

Runs: `SAEA-SLURP-STAGE3-L338-LEVER-COLUMN-FOLD-A-NPU-PORT-V1` / `SAEA-SLURP-STAGE3-L338-LEVER-COLUMN-FOLD-A-FIT-NPU-PORT-V1`.

The package is `docs/readiness/2026-10-30-slurp-l338-fold-a-confirmation-npu-port-v1/`; the read rule is the Fold B port package's, with the confirmation rule of plan M2.

## The measurement declaration

```json
{
  "stage3_measurement": {
    "SAEA-SLURP-STAGE3-L338-LEVER-COLUMN-FOLD-A-NPU-PORT-V1": {
      "aa_arm_id": "aa",
      "claim": "nominated",
      "claims_benefit": false,
      "confirmation_of": "SAEA-SLURP-STAGE3-L338-LEVER-COLUMN-FOLD-B-NPU-PORT-V1",
      "expected_effect_pp": 0.0,
      "measurement_design": "same-flight-node-column",
      "note": "A per-arm nomination, not a benefit claim: the round maps two sampling levers on the Thinking core, so no effect is assumed (0.0 pp SLU-F1) and the deliverable is the paired contrast with its cost. The null is the in-flight twin aa; the design resolves a per-arm contrast of 1.49 pp SLU / 0.65 pp intent at n = 1,080 (decision 190 node-column floors). L1 is judged on SLU with an intent guard, L2 on intent with an SLU guard. The summary key template_variants names the node-column summary of the effect-floor gate, not the template axis. Noise model M1: lock-rep2 is a second request-identical replicate of lock; the CI95 carries the replicate variance of the three copies. Fold A CONFIRMATION (plan section 8 R2 / M2): one pass of the same arm table on TRAIN Fold A (SLURP-EVO-FOLD-A-V2); a lever is accepted only if its Fold A lower CI95 is above zero; the optimism term is the Fold B delta minus the Fold A delta. Fold A is a held-out discovery fold, rollout-exposed by prior cells, never a test claim.",
      "primary_metric": "official SLURP pooled SLU-F1 (micro; scorer slurp-slu-official-train, the pinned official evaluate.py), natural-mixture weighting, with intent accuracy (scenario_action) beside it",
      "primary_metric_key": "slu",
      "primary_metric_key_by_arm": {
        "l1-support": "slu",
        "l2-union": "intent"
      },
      "read_roster": "SLURP-EVO-FOLD-A-V2",
      "replicate_arm_ids": [
        "lock-rep2"
      ],
      "rows": 1080,
      "template_variants": {
        "candidate_arms": 2,
        "node": "perception",
        "twin": true
      },
      "tuning_roster": "none",
      "variant_cost": {
        "extra_audio_seconds": 47321.96,
        "extra_contacts_per_row": 15
      },
      "variant_kind": "solver-sampling",
      "variant_kind_by_arm": {
        "l1-support": "solver-sampling",
        "l2-union": "perception-sampling"
      }
    }
  }
}
```

# D NPU port: direct-only on the BF16 / vLLM-Ascend Thinking core, preregistration (DRAFT / HELD)

Registered 2026-10-30 at study commit `534b543ff154ca23be8287c1730b040940a3d038` by script. Development numbers only.

Runs: `SAEA-SLURP-DIRECT-ONLY-FOLD-B-SMOKE-4-NPU-PORT-V1` / `SAEA-SLURP-DIRECT-ONLY-FOLD-B-SMOKE-4-FIT-NPU-PORT-V1`; `SAEA-SLURP-DIRECT-ONLY-FOLD-B-NPU-PORT-V1` / `SAEA-SLURP-DIRECT-ONLY-FOLD-B-FIT-NPU-PORT-V1`; `SAEA-SLURP-DIRECT-ONLY-FOLD-B-AA-NPU-PORT-V1` / `SAEA-SLURP-DIRECT-ONLY-FOLD-B-AA-FIT-NPU-PORT-V1`; `SAEA-SLURP-DIRECT-ONLY-FOLD-B-AB-NPU-PORT-V1` / `SAEA-SLURP-DIRECT-ONLY-FOLD-B-AB-FIT-NPU-PORT-V1`.

The package is `docs/readiness/2026-10-30-slurp-direct-only-fold-b-npu-port-v1/`; the read rule is the `read_rule` block of its `registered-facts-v1.json`.

## The measurement declaration

```json
{
  "stage3_measurement": {
    "SAEA-SLURP-DIRECT-ONLY-FOLD-B-AA-NPU-PORT-V1": {
      "claims_benefit": false,
      "expected_effect_pp": 0.0,
      "measurement_design": "cross-flight-with-in-chain-replicate",
      "note": "A capability read of the bare core on the port, no mechanism and no claim: expected_effect_pp is 0.0 and claims_benefit false. lock, aa and ab are three request-identical passes in one chain (noise model M1); the replicate variance enters every CI95 of the A-to-D and T-to-D contrasts.",
      "primary_metric": "official SLURP pooled SLU-F1 (micro; scorer slurp-slu-official-train, the pinned official evaluate.py), natural-mixture weighting, with intent accuracy (scenario_action_exact) beside it; the per-row F1 is a repo diagnostic reported beside the pooled F1, never an official SLURP number",
      "read_roster": "SLURP-EVO-FOLD-B-V2",
      "replicate_run_id": "SAEA-SLURP-DIRECT-ONLY-FOLD-B-NPU-PORT-V1",
      "rows": 1080,
      "tuning_roster": "none"
    },
    "SAEA-SLURP-DIRECT-ONLY-FOLD-B-AB-NPU-PORT-V1": {
      "claims_benefit": false,
      "expected_effect_pp": 0.0,
      "measurement_design": "cross-flight-with-in-chain-replicate",
      "note": "A capability read of the bare core on the port, no mechanism and no claim: expected_effect_pp is 0.0 and claims_benefit false. lock, aa and ab are three request-identical passes in one chain (noise model M1); the replicate variance enters every CI95 of the A-to-D and T-to-D contrasts.",
      "primary_metric": "official SLURP pooled SLU-F1 (micro; scorer slurp-slu-official-train, the pinned official evaluate.py), natural-mixture weighting, with intent accuracy (scenario_action_exact) beside it; the per-row F1 is a repo diagnostic reported beside the pooled F1, never an official SLURP number",
      "read_roster": "SLURP-EVO-FOLD-B-V2",
      "replicate_run_id": "SAEA-SLURP-DIRECT-ONLY-FOLD-B-NPU-PORT-V1",
      "rows": 1080,
      "tuning_roster": "none"
    },
    "SAEA-SLURP-DIRECT-ONLY-FOLD-B-NPU-PORT-V1": {
      "claims_benefit": false,
      "expected_effect_pp": 0.0,
      "measurement_design": "cross-flight-with-in-chain-replicate",
      "note": "A capability read of the bare core on the port, no mechanism and no claim: expected_effect_pp is 0.0 and claims_benefit false. lock, aa and ab are three request-identical passes in one chain (noise model M1); the replicate variance enters every CI95 of the A-to-D and T-to-D contrasts.",
      "primary_metric": "official SLURP pooled SLU-F1 (micro; scorer slurp-slu-official-train, the pinned official evaluate.py), natural-mixture weighting, with intent accuracy (scenario_action_exact) beside it; the per-row F1 is a repo diagnostic reported beside the pooled F1, never an official SLURP number",
      "read_roster": "SLURP-EVO-FOLD-B-V2",
      "replicate_run_id": "SAEA-SLURP-DIRECT-ONLY-FOLD-B-AA-NPU-PORT-V1",
      "rows": 1080,
      "tuning_roster": "none"
    },
    "SAEA-SLURP-DIRECT-ONLY-FOLD-B-SMOKE-4-NPU-PORT-V1": {
      "claims_benefit": false,
      "expected_effect_pp": 0.0,
      "measurement_design": "form-proof",
      "note": "Four Fold B rows prove FORM only: the raw-audio direct-only request form, the live phase fit of the direct-only raw gate and the seal work, and what one Solver contact costs under the port budget. Scored by no one; four rows claim nothing under instrument rule 5.",
      "primary_metric": "none: a four-row decode-cost smoke reads no metric",
      "read_roster": "SLURP-EVO-FOLD-B-V2",
      "rows": 4,
      "tuning_roster": "none"
    }
  }
}
```

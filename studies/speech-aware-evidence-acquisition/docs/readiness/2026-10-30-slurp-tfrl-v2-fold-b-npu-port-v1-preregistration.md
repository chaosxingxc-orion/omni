# T cell, iteration 1: the agent system plus TFRL-v2, preregistration (DRAFT / HELD)

Registered 2026-10-30 at study commit `534b543ff154ca23be8287c1730b040940a3d038` by script. Development numbers only; claims_benefit false.

Runs: `SAEA-SLURP-TFRL-V2-CONSTRUCTION-R01C-SMOKE-4-NPU-PORT-V1` / `SAEA-SLURP-TFRL-V2-CONSTRUCTION-R01C-SMOKE-4-FIT-NPU-PORT-V1`; `SAEA-SLURP-TFRL-V2-CONSTRUCTION-R01C-I1-H1-NPU-PORT-V1` / `SAEA-SLURP-TFRL-V2-CONSTRUCTION-R01C-I1-H1-FIT-NPU-PORT-V1`; `SAEA-SLURP-TFRL-V2-CONSTRUCTION-R01C-I1-H2-NPU-PORT-V1` / `SAEA-SLURP-TFRL-V2-CONSTRUCTION-R01C-I1-H2-FIT-NPU-PORT-V1`; `SAEA-SLURP-TFRL-V2-FOLD-B-SMOKE-4-NPU-PORT-V1` / `SAEA-SLURP-TFRL-V2-FOLD-B-SMOKE-4-FIT-NPU-PORT-V1`; `SAEA-SLURP-TFRL-V2-FOLD-B-I1-NPU-PORT-V1` / `SAEA-SLURP-TFRL-V2-FOLD-B-I1-FIT-NPU-PORT-V1`.

The package is `docs/readiness/2026-10-30-slurp-tfrl-v2-fold-b-npu-port-v1/`; the acceptance and kill rules are in its README and facts.

## The measurement declaration

```json
{
  "stage3_measurement": {
    "SAEA-SLURP-TFRL-V2-CONSTRUCTION-R01C-I1-H1-NPU-PORT-V1": {
      "claims_benefit": false,
      "expected_effect_pp": 0.0,
      "measurement_design": "material-source",
      "note": "A measurement-free trajectory-sampling column on the construction roster: it carries no contrast, no floor and no benefit claim; its sealed per-row scores feed the model-free credit, the contrast pairs and the exemplar distillation. The construction gain is read from the sealed scores by the acceptance rule.",
      "primary_metric": "none: per-row scores are read OFFLINE as the credit and distillation material of TFRL-v2",
      "read_roster": "SLURP-STAGE3-CONSTRUCTION-R01C-H1",
      "rows": 313,
      "tuning_roster": "none"
    },
    "SAEA-SLURP-TFRL-V2-CONSTRUCTION-R01C-I1-H2-NPU-PORT-V1": {
      "claims_benefit": false,
      "expected_effect_pp": 0.0,
      "measurement_design": "material-source",
      "note": "A measurement-free trajectory-sampling column on the construction roster: it carries no contrast, no floor and no benefit claim; its sealed per-row scores feed the model-free credit, the contrast pairs and the exemplar distillation. The construction gain is read from the sealed scores by the acceptance rule.",
      "primary_metric": "none: per-row scores are read OFFLINE as the credit and distillation material of TFRL-v2",
      "read_roster": "SLURP-STAGE3-CONSTRUCTION-R01C-H2",
      "rows": 312,
      "tuning_roster": "none"
    },
    "SAEA-SLURP-TFRL-V2-CONSTRUCTION-R01C-SMOKE-4-NPU-PORT-V1": {
      "claims_benefit": false,
      "expected_effect_pp": 0.0,
      "measurement_design": "form-proof",
      "note": "Four rows prove FORM only: the phase-native-module-v1-c7 module program with the Solver decoding group s1, s2, s3, s4 (temperature 0.7, top_p 0.80 / 0.90 / 0.95 / 1.00, top_k 20, the lock seed), the twin aa and the delivering lock arm LAST loads on the port deployment, no node hits its cap, the live phase fit passes and the receipt seals. Scored by no one; four rows claim nothing under instrument rule 5.",
      "primary_metric": "none (structural form, per-arm decode cost and the request form only)",
      "read_roster": "SLURP-STAGE3-ROLL-AA-R01C-SMOKE-4-V1",
      "rows": 4,
      "tuning_roster": "none"
    },
    "SAEA-SLURP-TFRL-V2-FOLD-B-I1-NPU-PORT-V1": {
      "aa_arm_id": "aa",
      "claim": "nominated",
      "claims_benefit": false,
      "expected_effect_pp": 2.0,
      "measurement_design": "same-flight-node-column",
      "note": "A nomination read, not a benefit claim: expected_effect_pp 2.0 is the low end of the plan's +2 to +4 pp if consumption is the bottleneck (else 0). The null is the in-flight twin aa and the replicate lock-rep2; the design resolves 1.49 pp SLU / 0.65 pp intent at n = 1,080 (decision 190 node-column floors), with the replicate variance in every CI95.",
      "primary_metric": "official SLURP pooled SLU-F1 (micro; scorer slurp-slu-official-train, the pinned official evaluate.py), natural-mixture weighting, with intent accuracy (scenario_action) beside it",
      "primary_metric_key": "slu",
      "read_roster": "SLURP-EVO-FOLD-B-V2",
      "replicate_arm_ids": [
        "lock-rep2"
      ],
      "rows": 1080,
      "template_variants": {
        "candidate_arms": 1,
        "node": "perception",
        "twin": true
      },
      "tuning_roster": "none",
      "variant_cost": {
        "extra_audio_seconds": 25238.38,
        "extra_contacts_per_row": 8
      },
      "variant_kind": "solver-sampling"
    },
    "SAEA-SLURP-TFRL-V2-FOLD-B-SMOKE-4-NPU-PORT-V1": {
      "claims_benefit": false,
      "expected_effect_pp": 0.0,
      "measurement_design": "form-proof",
      "note": "Four rows prove FORM only: the phase-native-module-v1-c10 module program with the candidate t-v2 (one Perception contact and a Solver vote of n = 3 at temperature 0.6, top_p 0.95, top_k 20, commit saea-commit-plurality-support-v1 with min_support 2), the twin aa, the registered replicate lock-rep2 and the delivering lock arm LAST loads on the port deployment, no node hits its cap, the live phase fit passes and the receipt seals. Scored by no one; four rows claim nothing under instrument rule 5.",
      "primary_metric": "none (structural form, per-arm decode cost and the request form only)",
      "read_roster": "SLURP-STAGE3-FILLER-FOLD-B-C01-SMOKE-4-V1",
      "rows": 4,
      "tuning_roster": "none"
    }
  }
}
```

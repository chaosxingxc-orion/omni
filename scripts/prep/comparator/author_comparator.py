#!/usr/bin/env python3
"""Author the agreement-free port COMPARATOR cell from the raw-port baseline.

Model-free, repository-side only: reads two committed fragments and writes two
new ones into the worktree. Nothing under the data root is touched.
"""
import json, sys
from pathlib import Path

WT = Path("/mnt/cephfs/xingchao/omni/work/dev/wt-port/studies/speech-aware-evidence-acquisition")
BASE = WT / "configs/baseline/slurp-native-stage3-l338-lever-column-fold-b-baseline-raw-port-v1.json"
EXP = WT / "configs/experiment/slurp-native-stage3-l338-lever-column-fold-b-npu-port-v1.json"
OUT_BASE = WT / "configs/baseline/slurp-native-stage3-l338-lever-column-fold-b-comparator-npu-port-v1.json"
OUT_EXP = WT / "configs/experiment/slurp-native-stage3-l338-lever-column-fold-b-comparator-npu-port-v1.json"

KUS3_SHA = sys.argv[1] if len(sys.argv) > 1 else "PENDING-MF7B-NPU-PORT-SEAL"

IB_KEYS = ("prepared/SAEA-SLURP-INTENT-BOUNDARY-KEY-ENCODING-NPU-PORT-V1/facets.jsonl",
           "30e29867b1c94bbf5a9df32f92a4a830adbba9dda9ca0064fed44b1b8c3a3781",
           "SAEA-SLURP-INTENT-BOUNDARY-KEY-ENCODING-NPU-PORT-V1")
IB_QUERY = ("vectors/slurp-intent-boundaries-train-fold-b-1080-npu-port-v1/query-vectors-speech-segments.jsonl",
            "fe23703a429d4a655b51d0cee921f28eba0e5be8a5511d9fe0f13dc73eb28181",
            "SAEA-SLURP-TFRL-FOLD-B-SPEECH-QUERY-ENCODING-NPU-PORT-V1")
EX_QUERY = ("vectors/slurp-stage3-filler-fold-b-npu-port-v1/query-vectors-speech-content-v1.jsonl",
            "e1d3230ed517685f6d0e963bb48a0a5bf205e3c31a7c7aaae22e88b1ac27196f",
            "SAEA-SLURP-TFRL-FOLD-B-SPEECH-QUERY-ENCODING-NPU-PORT-V1")

RECEIPTS = [
    ("knowledge/slurp/semantic/intent-boundaries-npu-port-v1/completion.json", "data",
     "7b7e4744caec975d9fa000ec14c4f47f2bb072d1f129016e114382e98042b294"),
    ("knowledge/slurp/semantic/intent-boundaries-npu-port-v1/release.json", "data",
     "22242e839627a895d3bcc450002a90fd64ceddef606446febe19a5b8a895dc06"),
    ("knowledge/slurp/semantic/slot-contracts-npu-port-v1/completion.json", "data",
     "894937819636aaeed206d96e4e806be03712341a9ed94312d919201807272d19"),
    ("knowledge/slurp/semantic/slot-contracts-npu-port-v1/release.json", "data",
     "19723a2a8996bb189e9c6312ae56903a87c9effc323bdde4bf28aaa2a8952341"),
    ("preparation-runs/SAEA-SLURP-INTENT-BOUNDARY-KEY-ENCODING-NPU-PORT-V1/encode-v1/receipt.json", "data",
     "8aff65483afd4e1aba98ecb8456976650783dea1a2e87aea003b2a73fb887964"),
    ("preparation-runs/SAEA-SLURP-TFRL-FOLD-B-SPEECH-QUERY-ENCODING-NPU-PORT-V1/encode-v1/receipt.json", "data",
     "3e52357c4871b2cb8e7c67b9586b6001cefc484090b8d2bb64831302dcd84da5"),
    ("prepared/SAEA-SLURP-INTENT-BOUNDARY-TRAIN-FOLD-B-ALIAS-NPU-PORT-V1/query-dimension-alias.json", "data",
     "f59dc040b05e2e87637a91d8d677392c50a010701986a8db33bbdfdfdd008adf"),
    ("vectors/slurp-stage3-filler-fold-b-npu-port-v1/query-vectors-speech-content-v1.provenance.json", "data",
     "59c285768247d3f6abb0fea68dfbd5e2f0246074038d7318218f834e232c890c"),
    ("docs/receipts/knowledge/slurp-kb-size-pool-annotated-exemplars.json", "repository",
     "48d0f34cf536ebf9806d55b5dcced1b607080bec3c68ca7dea27cdb01478cc72"),
    ("docs/receipts/rosters/slurp-evo-fold-b-v2.json", "repository",
     "47b89b9e33b1aed553ad94a1819acba6ac8837130ca546c33219d91a1ab944a8"),
]

WHAT_THIS_IS = (
    "The agreement-free COMPARATOR of the NPU port: the raw-port Fold B lever-column baseline with the "
    "boundary channel's agreement_source REMOVED and w_agree 0.0, every retrieval input re-pinned to the "
    "port's own sealed artifacts (written boundary keys, Fold B speech-query segment vectors, the "
    "model-free content derivative and the port MF-7b key vectors). NEVER FLOWN: allowed stage check "
    "only, readiness held forever, no exposure row, no admission receipt, no budget authority. It exists "
    "for one reason: scripts/build_slurp_intent_exemplar_agreement.py refuses to form a vote out of a "
    "vote, so the port's own agreement table needs a committed source identity that declares no "
    "agreement_source. This is the port's analogue of the historical FILLER BASE "
    "slurp-native-stage3-filler-fold-b-base-v1, which played the same role for "
    "SAEA-SLURP-INTENT-EXEMPLAR-AGREEMENT-FOLD-B-V1."
)
IB_KEY_SIGNAL = (
    "text: sealed released TRAIN-derived intent-boundary facet vectors re-encoded on the port stack "
    "(Omni-Embed-Nemotron-3B, CPU float32, forced causal text geometry; preparation run "
    "SAEA-SLURP-INTENT-BOUNDARY-KEY-ENCODING-NPU-PORT-V1); the key side is knowledge text."
)
IB_QUERY_SIGNAL = (
    "speech: complete original TRAIN Fold B recordings represented by sealed full-utterance segments in "
    "the intent-boundary retrieval dimension, re-encoded on the port stack (preparation run "
    "SAEA-SLURP-TFRL-FOLD-B-SPEECH-QUERY-ENCODING-NPU-PORT-V1) and carried into the segment file by the "
    "model-free query-dimension alias SAEA-SLURP-INTENT-BOUNDARY-TRAIN-FOLD-B-ALIAS-NPU-PORT-V1; segment "
    "identities are preserved and no flattening, re-encoding, transcript, answer or future turn enters "
    "this query."
)


def main() -> int:
    base = json.loads(BASE.read_text(encoding="utf-8"))
    exp = json.loads(EXP.read_text(encoding="utf-8"))

    # The module compiler re-applies agree-vote-seed (agreement_source, w_agree, k) and
    # centroid-boundary-key (facet_source) onto the boundary channel whenever those components
    # are armed, so disarming them is what actually makes this cell agreement-free; editing the
    # composed spec alone would be overwritten at compile time.
    components = base["native_episode"]["consumption_module"]["components"]
    disarmed = []
    for index, component in enumerate(components):
        if component.get("component_id") in ("agree-vote-seed", "centroid-boundary-key"):
            if component.get("state") != "on":
                raise SystemExit(f"{component['component_id']} is not armed in the source baseline")
            components[index] = {"component_id": component["component_id"], "state": "off"}
            disarmed.append(component["component_id"])
    if sorted(disarmed) != ["agree-vote-seed", "centroid-boundary-key"]:
        raise SystemExit(f"expected both seed components, disarmed {disarmed}")

    ib = base["knowledge_stores"]["intent-boundaries"]["phase_supply"]["perception"]
    if "agreement_source" not in ib:
        raise SystemExit("source baseline already declares no agreement_source")
    del ib["agreement_source"]
    ib["rank_weights"] = dict(ib["rank_weights"], w_agree=0.0)
    ib["facet_source"] = {"embedded_signal": IB_KEY_SIGNAL, "kind": "sealed-vectors",
                          "relpath": IB_KEYS[0], "sha256": IB_KEYS[1],
                          "tool_contact_ledger_row": IB_KEYS[2]}
    ib["query_source"] = {"embedded_signal": IB_QUERY_SIGNAL, "kind": "sealed-segment-vectors",
                          "relpath": IB_QUERY[0], "sha256": IB_QUERY[1],
                          "tool_contact_ledger_row": IB_QUERY[2]}

    ex = base["knowledge_stores"]["train-exemplars"]["phase_supply"]["solver"]
    if ex["rank_weights"].get("w_agree") != 0.0 or "agreement_source" in ex:
        raise SystemExit("the vote channel must stay agreement-free")
    ex["facet_source"] = dict(ex["facet_source"], sha256=KUS3_SHA,
                              tool_contact_ledger_row="SAEA-SLURP-MF7B-NEMOTRON-EMBEDDING-NPU-PORT-V1")
    ex["query_source"] = dict(ex["query_source"], relpath=EX_QUERY[0], sha256=EX_QUERY[1],
                              tool_contact_ledger_row=EX_QUERY[2])

    base["preparation_source_receipts"] = [
        {"relpath": r, "root": root, "sha256": s} for r, root, s in RECEIPTS
    ]
    base["native_contact_fit"] = {
        "attempt_id": "attempt-slurp-native-stage3-l338-lever-column-fold-b-comparator-npu-port-fit-v1",
        "contact_cap": base["native_contact_fit"]["contact_cap"],
        "max_speech_packets": base["native_contact_fit"]["max_speech_packets"],
        "run_id": "SAEA-SLURP-STAGE3-L338-LEVER-COLUMN-FOLD-B-COMPARATOR-NPU-PORT-FIT-V1",
    }
    base["what_this_is"] = WHAT_THIS_IS
    OUT_BASE.write_text(json.dumps(base, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    cell = exp["cells"][0]
    cell["allowed_execution_stages"] = ["check"]
    cell["attempt_id"] = "attempt-slurp-native-stage3-l338-lever-column-fold-b-comparator-npu-port-v1"
    cell["baseline_fragment"] = OUT_BASE.stem
    cell["cell_id"] = OUT_EXP.stem
    cell["run_id"] = "SAEA-SLURP-STAGE3-L338-LEVER-COLUMN-FOLD-B-COMPARATOR-NPU-PORT-V1"
    cell["readiness"] = {
        "reason": (
            "NEVER FLOWN. Allowed stage check only; readiness held forever; no exposure row, no "
            "admission receipt, no decision receipt, no budget authority and no prepared first-phase "
            "supply. It is the committed agreement-free source identity of the port's own exemplar-intent "
            "agreement table prepared/SAEA-SLURP-INTENT-EXEMPLAR-AGREEMENT-FOLD-B-NPU-PORT-V1/"
            "agreement.jsonl, built model-free by scripts/build_slurp_intent_exemplar_agreement.py, "
            "exactly as the historical FILLER BASE slurp-native-stage3-filler-fold-b-base-v1 was the "
            "source identity of SAEA-SLURP-INTENT-EXEMPLAR-AGREEMENT-FOLD-B-V1. Nothing here authorises a "
            "flight, a readiness flip or an admission entry."
        ),
        "requirements": [],
        "state": "held",
    }
    exp["cells"] = [cell]
    exp["experiment_id"] = "SAEA-SLURP-STAGE3-L338-LEVER-COLUMN-FOLD-B-COMPARATOR-NPU-PORT-V1"
    exp["date"] = "2026-10-28"
    exp["purpose"] = WHAT_THIS_IS
    exp["core_read_design"] = {
        "reading": "none: this cell is never flown and produces no read",
        "this_cell": "agreement-free comparator; a model-free builder input only",
    }
    OUT_EXP.write_text(json.dumps(exp, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    raw = OUT_BASE.read_text(encoding="utf-8") if OUT_BASE.exists() else ""
    for path in (OUT_BASE, OUT_EXP):
        print(f"wrote {path.relative_to(WT)}  bytes={path.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Re-pin the raw-port Fold B baseline and the port experiment cell to the port artifacts.

Model-free, repository-side only. Reads a sealed-sha JSON file and rewrites two committed
fragments in the worktree. Nothing under the data root is touched.

Every pin site is named explicitly and verified to have moved, so a silently missed site is a
failure rather than a stale historical pin: the boundary channel (facet_source, query_source,
agreement_source), the exemplar channel (facet_source, query_source) AND the two module-graph
components the compiler re-applies over the boundary channel (agree-vote-seed.agreement_source,
centroid-boundary-key.facet_source).

usage: repin_port_cell.py <sealed-shas.json>
"""
import json, re, sys
from pathlib import Path

WT = Path("/mnt/cephfs/xingchao/work/dev/wt-port/studies/speech-aware-evidence-acquisition")
BASE = WT / "configs/baseline/slurp-native-stage3-l338-lever-column-fold-b-baseline-raw-port-v1.json"
EXP = WT / "configs/experiment/slurp-native-stage3-l338-lever-column-fold-b-npu-port-v1.json"

CENTROIDS = "prepared/SAEA-SLURP-INTENT-BOUNDARY-CENTROID-KEYS-NPU-PORT-V1/facets.jsonl"
CENTROID_ROW = "SAEA-SLURP-INTENT-BOUNDARY-CENTROID-KEYS-NPU-PORT-V1"
AGREEMENT = "prepared/SAEA-SLURP-INTENT-EXEMPLAR-AGREEMENT-FOLD-B-NPU-PORT-V1/agreement.jsonl"
AGREEMENT_ROW = "SAEA-SLURP-INTENT-EXEMPLAR-AGREEMENT-FOLD-B-NPU-PORT-V1"
IB_QUERY = "vectors/slurp-intent-boundaries-train-fold-b-1080-npu-port-v1/query-vectors-speech-segments.jsonl"
IB_QUERY_SHA = "fe23703a429d4a655b51d0cee921f28eba0e5be8a5511d9fe0f13dc73eb28181"
EX_QUERY = "vectors/slurp-stage3-filler-fold-b-npu-port-v1/query-vectors-speech-content-v1.jsonl"
EX_QUERY_SHA = "e1d3230ed517685f6d0e963bb48a0a5bf205e3c31a7c7aaae22e88b1ac27196f"
KUS3 = "vectors/slurp-kus3/facet-vectors.jsonl"
MF7B_ROW = "SAEA-SLURP-MF7B-NEMOTRON-EMBEDDING-NPU-PORT-V1"
SPEECH_ROW = "SAEA-SLURP-TFRL-FOLD-B-SPEECH-QUERY-ENCODING-NPU-PORT-V1"
IB_KEY_ROW = "SAEA-SLURP-INTENT-BOUNDARY-KEY-ENCODING-NPU-PORT-V1"

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

REQUIRED = ("kus3_facet_vectors", "centroid_facets", "centroid_receipt",
            "agreement", "agreement_receipt")


def sha(value: str, field: str) -> str:
    if not re.fullmatch("[0-9a-f]{64}", str(value)):
        raise SystemExit(f"{field} is not a sealed sha256: {value!r}")
    return str(value)


def main() -> int:
    sealed = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED if k not in sealed]
    if missing:
        raise SystemExit(f"sealed shas missing: {missing}")
    for key in REQUIRED:
        sha(sealed[key], key)

    base = json.loads(BASE.read_text(encoding="utf-8"))
    before = json.dumps(base, sort_keys=True)
    moved: list[str] = []

    ib = base["knowledge_stores"]["intent-boundaries"]["phase_supply"]["perception"]
    ib["facet_source"] = dict(ib["facet_source"], relpath=CENTROIDS,
                              sha256=sealed["centroid_facets"],
                              tool_contact_ledger_row=CENTROID_ROW)
    moved.append("intent-boundaries.perception.facet_source")
    ib["agreement_source"] = dict(ib["agreement_source"], relpath=AGREEMENT,
                                  sha256=sealed["agreement"],
                                  tool_contact_ledger_row=AGREEMENT_ROW)
    moved.append("intent-boundaries.perception.agreement_source")
    ib["query_source"] = dict(ib["query_source"], relpath=IB_QUERY, sha256=IB_QUERY_SHA,
                              tool_contact_ledger_row=SPEECH_ROW)
    moved.append("intent-boundaries.perception.query_source")

    ex = base["knowledge_stores"]["train-exemplars"]["phase_supply"]["solver"]
    ex["facet_source"] = dict(ex["facet_source"], relpath=KUS3,
                              sha256=sealed["kus3_facet_vectors"],
                              tool_contact_ledger_row=MF7B_ROW)
    moved.append("train-exemplars.solver.facet_source")
    ex["query_source"] = dict(ex["query_source"], relpath=EX_QUERY, sha256=EX_QUERY_SHA,
                              tool_contact_ledger_row=SPEECH_ROW)
    moved.append("train-exemplars.solver.query_source")

    seen = set()
    for component in base["native_episode"]["consumption_module"]["components"]:
        cid = component.get("component_id")
        if cid == "agree-vote-seed":
            if component.get("state") != "on":
                raise SystemExit("agree-vote-seed is not armed; the flown cell needs it on")
            component["params"]["agreement_source"] = dict(ib["agreement_source"])
            seen.add(cid)
            moved.append("consumption_module.agree-vote-seed.params.agreement_source")
        elif cid == "centroid-boundary-key":
            if component.get("state") != "on":
                raise SystemExit("centroid-boundary-key is not armed; the flown cell needs it on")
            component["params"]["facet_source"] = dict(ib["facet_source"])
            seen.add(cid)
            moved.append("consumption_module.centroid-boundary-key.params.facet_source")
    if seen != {"agree-vote-seed", "centroid-boundary-key"}:
        raise SystemExit(f"module-graph pin sites not found: {sorted(seen)}")

    base["preparation_source_receipts"] = sorted(
        [{"relpath": r, "root": root, "sha256": s} for r, root, s in RECEIPTS]
        + [
            {"relpath": "prepared/SAEA-SLURP-INTENT-EXEMPLAR-AGREEMENT-FOLD-B-NPU-PORT-V1/receipt.json",
             "root": "data", "sha256": sealed["agreement_receipt"]},
            {"relpath": "prepared/SAEA-SLURP-INTENT-BOUNDARY-CENTROID-KEYS-NPU-PORT-V1/receipt.json",
             "root": "data", "sha256": sealed["centroid_receipt"]},
        ],
        key=lambda item: (item["root"], item["relpath"]),
    )
    moved.append("preparation_source_receipts")

    if json.dumps(base, sort_keys=True) == before:
        raise SystemExit("nothing moved; refusing to rewrite")
    BASE.write_text(json.dumps(base, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")

    exp = json.loads(EXP.read_text(encoding="utf-8"))
    cell = exp["cells"][0]
    hit = 0
    for item in cell["readiness"]["requirements"]:
        if item.get("kind") == "preparation" and "AGREEMENT" in str(item.get("relpath", "")):
            item["relpath"], item["sha256"] = AGREEMENT, sealed["agreement"]
            hit += 1
    if hit != 1:
        raise SystemExit(f"expected one agreement requirement, matched {hit}")
    if "phase_supply" in sealed:
        for item in cell["readiness"]["requirements"]:
            if item.get("kind") == "preparation" and "phase-supply" in str(item.get("relpath", "")):
                item["relpath"] = (
                    "prepared/SAEA-SLURP-STAGE3-L338-LEVER-COLUMN-FOLD-B-NPU-PORT-V1/phase-supply-v1.json"
                )
                item["sha256"] = sha(sealed["phase_supply"], "phase_supply")
    EXP.write_text(json.dumps(exp, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")

    for site in moved:
        print("re-pinned", site)
    print("re-pinned experiment readiness: agreement" + (" + phase supply" if "phase_supply" in sealed else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

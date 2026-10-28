"""In-memory validation of the port's authored semantic sources before any registered preparation.

Compiles each source with the study's own builders against the historical permitted rosters and the three
forbidden rosters, measures rendered section sizes against the historical store measurements, and runs the
containment rule against the anchor control turns and the task-contract template.

usage (port_py.sh): validate_port_sources.py <data-root> <repo-root> <source-dir>
"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
from pathlib import Path

from speech_aware_evidence_acquisition.analysis.slurp_slot_vocabulary import containment_collisions
from speech_aware_evidence_acquisition.core.instructions import render_template_with_fill
from speech_aware_evidence_acquisition.core.semantic_knowledge_preparation import _Pin, _roster
from speech_aware_evidence_acquisition.data.lock import load_lock, umbrella_lock_path
from speech_aware_evidence_acquisition.rails.runner import _protected_knowledge_material
from speech_aware_evidence_acquisition.rails.semantic_knowledge_build import (
    AuthoredSemanticKnowledgeSpec,
    build_intent_boundary_snapshot,
    build_slot_contract_snapshot,
)

PREPARE = "prepared/SAEA-SLURP-SEMANTIC-AUTHORING-PREPARE-V1"
FORBIDDEN = ("docs/receipts/rosters/slurp-evo-fold-a-v2.json", "docs/receipts/rosters/slurp-evo-fold-b-v2.json",
             "docs/receipts/probe-samples/DEV-1000.json")
CONTROL = ("configs/probes/slurp_native_role_perception_v3.txt", "configs/probes/slurp_native_role_planner_v1.txt",
           "configs/probes/slurp_native_role_reviewer_v1.txt", "configs/probes/slurp_native_role_solver_v2.txt",
           "configs/probes/slurp_native_role_context_v3.txt")
HISTORICAL = {  # docs/receipts/knowledge-index-design-v1.json store_measurements and the v2 review section 7
    "intent v1": {"total": 42133, "words": 5740, "Activation boundary": 4244, "Confusable exclusions": 7301,
                  "Uncertainty behavior": 11940, "Limitations": 12362},
    "slot v1": {"total": 80049, "words": 11647},
    "slot v2": {"total": 83351, "words": 12137, "Intent bindings and semantic roles": 24877, "Extraction cues": 12928,
                "Admissible formats": 5352, "Missing behavior": 8513, "Ambiguity behavior": 6997, "Limitations": 16725},
}


def roster(root: Path, relpath: str, rootname: str, repo: Path, *, construction: bool):
    raw = (root / relpath).read_bytes()
    pin = _Pin(relpath=relpath, sha256=hashlib.sha256(raw).hexdigest(), path=root / relpath, root=rootname)
    return _roster(pin, raw, name=relpath, carrier="slurp", require_construction=construction, repo_root=repo)


def sections(content: str, names: tuple[str, ...]) -> dict[str, int]:
    """Characters of each section body (the lines after its heading, or the inline text)."""

    lines, out, current = content.split("\n"), {name: 0 for name in names}, None
    for line in lines[1:]:
        head = next((name for name in names if line.startswith(name + ":")), None)
        if head:
            current = head
            inline = line[len(head) + 1:].strip()
            out[head] += len(inline)
        elif current:
            out[current] += len(line[2:]) if line.startswith("- ") else len(line)
    return out


def main() -> int:
    data_root, repo, source_dir = map(Path, sys.argv[1:4])
    forbidden = set()
    for relpath in FORBIDDEN:
        forbidden.update(roster(repo, relpath, "repo", repo, construction=False).sample_ids)
    control = {relpath: (repo / relpath).read_text(encoding="utf-8") for relpath in CONTROL}
    control["context_v3 + slurp-label-inventory"] = render_template_with_fill(
        "probes/slurp_native_role_context_v3.txt", "slurp-label-inventory", repo,
        load_lock(umbrella_lock_path(repo)), data_root)
    jobs = (("intent v1", "authored-intent-boundaries-v1.json", "intent-support-construction-roster.json",
             build_intent_boundary_snapshot,
             ("Activation boundary", "Confusable exclusions", "Uncertainty behavior", "Limitations")),
            ("slot v1", "authored-slot-contracts-v1.json", "slot-support-construction-roster.json",
             build_slot_contract_snapshot, ()),
            ("slot v2", "authored-slot-contracts-v2.json", "slot-support-construction-roster.json",
             build_slot_contract_snapshot,
             ("Intent bindings and semantic roles", "Extraction cues", "Admissible formats", "Missing behavior",
              "Ambiguity behavior", "Limitations")))
    report = {}
    for label, source, permitted_name, builder, names in jobs:
        document = json.loads((source_dir / source).read_bytes())
        permitted = roster(data_root, f"{PREPARE}/{permitted_name}", "data", repo, construction=True)
        assert document["source_split_identity_hash"] == permitted.split_identity_hash
        snapshot = builder(
            AuthoredSemanticKnowledgeSpec(carrier_lock_key="slurp",
                                          source_split_identity_hash=permitted.split_identity_hash,
                                          permitted_source_sample_ids=permitted.sample_ids),
            rows=document["rows"], forbidden_sample_ids=frozenset(forbidden))
        contents = [value.content for value in snapshot.values]
        measured = {"values": len(contents), "source_recordings": len(snapshot.source_sample_ids),
                    "total": sum(map(len, contents)), "words": sum(len(c.split()) for c in contents),
                    "content_sha256": snapshot.content_hash()}
        for name in names:
            measured[name] = sum(sections(c, names)[name] for c in contents)
        measured["card_chars_min_median_max"] = [min(map(len, contents)), statistics.median(map(len, contents)),
                                                 max(map(len, contents))]
        material = _protected_knowledge_material(snapshot)
        # the runner's own rule: casefolded material inside the casefolded instruction
        measured["containment_collisions"] = {
            name: list(containment_collisions(material, text.casefold())) for name, text in control.items()}
        measured["historical"] = HISTORICAL[label]
        report[label] = measured
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

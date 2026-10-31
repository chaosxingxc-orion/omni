"""Explicit namespace selection for shared Agent campaign tools.

Only two registered engineering families are recognized. Selection never grants
contact authority; ordinary registration, mint, readiness and flight gates apply.
The launcher passes SAEA_AGENT_CAMPAIGN_FAMILY to every subprocess explicitly.
"""
from dataclasses import dataclass
import os

ENV = "SAEA_AGENT_CAMPAIGN_FAMILY"
DEFAULT = "naturalend-v1"
WIDTH = "phase-width-v1"
WIDTH2 = "phase-width-v2"
WIDTH3 = "phase-width-v3"
WIDTH4 = "phase-width-v4"
WIDTH5 = "phase-width-v5"


@dataclass(frozen=True)
class CampaignFamily:
    name: str
    package: str
    branch_read: str
    paired_package: str
    paired_read: str


FAMILIES = {
    DEFAULT: CampaignFamily(DEFAULT,
        "docs/readiness/2026-10-30-slurp-agent-fold-b-naturalend-npu-v1",
        "SAEA-SLURP-L338-FOLD-B-NATURALEND-COLUMNS-READ-NPU-V1",
        "docs/readiness/2026-10-30-slurp-first-system-paired-read-npu-v1",
        "SAEA-SLURP-FIRST-SYSTEM-PAIRED-READ-NPU-V1"),
    WIDTH2: CampaignFamily(WIDTH2,
        "docs/readiness/2026-10-31-slurp-agent-fold-b-phase-width-v2-npu-v1",
        "SAEA-SLURP-L338-FOLD-B-PHASE-WIDTH-V2-COLUMNS-READ-NPU-V1",
        "docs/readiness/2026-10-31-slurp-first-system-phase-width-v2-paired-read-npu-v1",
        "SAEA-SLURP-FIRST-SYSTEM-PHASE-WIDTH-V2-PAIRED-READ-NPU-V1"),
    WIDTH3: CampaignFamily(WIDTH3,
        "docs/readiness/2026-11-01-slurp-agent-fold-b-phase-width-v3-npu-v1",
        "SAEA-SLURP-L338-FOLD-B-PHASE-WIDTH-V3-COLUMNS-READ-NPU-V1",
        "docs/readiness/2026-11-01-slurp-first-system-phase-width-v3-paired-read-npu-v1",
        "SAEA-SLURP-FIRST-SYSTEM-PHASE-WIDTH-V3-PAIRED-READ-NPU-V1"),
    WIDTH4: CampaignFamily(WIDTH4,
        "docs/readiness/2026-11-01-slurp-agent-fold-b-phase-width-v4-npu-v1",
        "SAEA-SLURP-L338-FOLD-B-PHASE-WIDTH-V4-COLUMNS-READ-NPU-V1",
        "docs/readiness/2026-11-01-slurp-first-system-phase-width-v4-paired-read-npu-v1",
        "SAEA-SLURP-FIRST-SYSTEM-PHASE-WIDTH-V4-PAIRED-READ-NPU-V1"),
    WIDTH5: CampaignFamily(WIDTH5,
        "docs/readiness/2026-10-31-slurp-agent-fold-b-phase-width-v5-npu-v1",
        "SAEA-SLURP-L338-FOLD-B-PHASE-WIDTH-V5-COLUMNS-READ-NPU-V1",
        "docs/readiness/2026-10-31-slurp-first-system-phase-width-v5-paired-read-npu-v1",
        "SAEA-SLURP-FIRST-SYSTEM-PHASE-WIDTH-V5-PAIRED-READ-NPU-V1"),
    WIDTH: CampaignFamily(WIDTH,
        "docs/readiness/2026-10-31-slurp-agent-fold-b-phase-width-npu-v1",
        "SAEA-SLURP-L338-FOLD-B-PHASE-WIDTH-COLUMNS-READ-NPU-V1",
        "docs/readiness/2026-10-31-slurp-first-system-phase-width-paired-read-npu-v1",
        "SAEA-SLURP-FIRST-SYSTEM-PHASE-WIDTH-PAIRED-READ-NPU-V1"),
}


def resolve(name):
    if name not in FAMILIES:
        raise ValueError("unknown Agent campaign family: " + repr(name))
    return FAMILIES[name]


def selected():
    return resolve(os.environ.get(ENV, DEFAULT))

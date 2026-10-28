#!/usr/bin/env python3
"""Mint the root-admission decision receipt of the NPU-port twin of L338 on the data root from the port package's draft (MAIN THREAD only).

Owner decision 2026-10-28 item 5 authorises this receipt; the MAIN THREAD mints it, never an agent. The 268-B minting logic
(``thinking_system_mint_receipt.mint``) with the port family (``l338_port_family``): copies
``docs/readiness/2026-10-28-slurp-l338-lever-column-fold-b-npu-port-v1/root-admission-draft-v1.json`` to
``$SAEA_DATA_ROOT/coordination/slurp-l338-lever-column-fold-b-npu-port-v1/root-admission-v1.json``, sets ``status`` to ``decided`` and ``recorded_at_utc``
to now, re-hashes every antecedent that names a repository file, and refuses to overwrite an existing receipt. ``eligible_run_ids`` (the port smoke and the
port flight, each with its FIT identity, in chain order) and the caps are the draft's.

Mint ONLY after the owner countersigns the DRAFT rows and the measured speed sweep has filled PORT_DEPLOYMENT_PROFILE (the draft's accelerator bound is
PROVISIONAL until then). The study root defaults to this repository (``SAEA_STUDY_ROOT`` overrides it).

    SAEA_DATA_ROOT=<data root> python -B scripts/flights/nx/l338_port_mint_receipt.py [--coordinator NAME] [--dry]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "direct"))

import l338_port_family as pf  # noqa: E402
import thinking_system_mint_receipt as base  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--coordinator", default="Fable5.1")
    parser.add_argument("--dry", action="store_true", help="print the receipt, write nothing")
    args = parser.parse_args(argv)
    root = (os.environ.get("SAEA_STUDY_ROOT") or str(pf.REPO_ROOT)).rstrip("/")
    draft = json.loads((Path(root) / pf.PACKAGE / "root-admission-draft-v1.json").read_text(encoding="utf-8"))
    if draft.get("eligible_run_ids") != pf.eligible_run_ids() or draft.get("decision_id") != pf.DECISION_ID:
        raise SystemExit("the port draft's decision id or eligible run ids are not the port family's; re-run l338_port_write_registration.py")
    out = os.path.join(base.DATA, pf.DECISION_RELPATH)
    if os.path.exists(out) and not args.dry:
        raise SystemExit(f"refusing to overwrite an existing receipt: {out}")
    summary = base.mint(root, base.DATA, args.coordinator, args.dry, family=pf, tool=pf.TOOL_MINT)
    if args.dry:
        print(summary.pop("text"))
        print(f"(dry) would write {summary['out']}; re-hashed {summary['rehashed_paths']} antecedent path(s)")
        return 0
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

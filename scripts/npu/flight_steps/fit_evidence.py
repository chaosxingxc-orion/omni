#!/usr/bin/env python3
"""Read-only check of the FIT evidence a worker or campaign leaves (stdlib only, no side effects).

The native contact FIT of these cells is an inline gate of the worker/campaign under the same lease
(core.preparation_fit.build_native_contact_fit_gate); it is not a separate model run. After the worker it leaves
``prepared/<FIT run>/<FIT attempt>/`` and the worker's ``runs/<run>/<attempt>/flight-receipts.json``.
usage: fit_evidence.py --data-root D --run R --attempt A --fit-run FR --fit-attempt FA --contact-cap N
Exit 0 when both exist and the FIT directory is non-empty; 1 otherwise.
"""
import argparse
import json
import sys
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "run", "attempt", "fit-run", "fit-attempt"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--contact-cap", type=int, required=True)
    args = parser.parse_args(argv)
    root = Path(args.data_root)
    fit_dir = root / "prepared" / args.fit_run / args.fit_attempt
    receipts = root / "runs" / args.run / args.attempt / "flight-receipts.json"
    files = sorted(p for p in fit_dir.rglob("*") if p.is_file()) if fit_dir.is_dir() else []
    report = {
        "fit_dir": str(fit_dir), "fit_files": len(files), "fit_bytes": sum(p.stat().st_size for p in files),
        "flight_receipts": str(receipts), "flight_receipts_present": receipts.is_file(),
        "registered_contact_cap": args.contact_cap,
    }
    print(json.dumps(report, sort_keys=True))
    problems = []
    if not files:
        problems.append(f"no FIT evidence under {fit_dir}")
    if not receipts.is_file():
        problems.append(f"no flight receipts at {receipts}")
    for problem in problems:
        print(f"FIT evidence missing: {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Print one value from the local deployment profile, for shell entry points.

The deployment profile is the only place a machine's paths and hosts live, and
``local_profile.sh`` is the only shell bridge to it.  This exists so a shell
script can ask for one field without re-implementing the JSON rules: the same
loader the experiment CLI uses validates the file, so a typo, a relative path
or a credential-looking key fails here exactly as it does at parse time.

usage: local_get.py --key storage.shared_root [--config PATH]
       local_get.py --keys storage.shared_root roots.data [--config PATH]
       local_get.py --list [--config PATH]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# The bridge ships next to the host scripts, which may be deployed apart from the
# repository (``work/scripts/lib`` next to ``work/configs``).  OMNI_BASE_ROOT names
# the tree that holds ``tools/experiments`` and ``configs/``; local_profile.sh sets
# it from its own location, and a deployment may point it elsewhere.
REPO = Path(os.environ.get("OMNI_BASE_ROOT") or Path(__file__).resolve().parents[2]).resolve()
sys.path.insert(0, str(REPO))

try:
    from tools.experiments.config import ConfigError  # noqa: E402
    from tools.experiments.local import FIELDS, load_local_config  # noqa: E402
except ImportError as error:  # a bare copy of the bridge cannot validate the profile
    print(f"local profile: cannot import tools.experiments from {REPO} ({error}); "
          f"set OMNI_BASE_ROOT to the repository that holds tools/ and configs/", file=sys.stderr)
    raise SystemExit(2)

DEFAULT = REPO / "configs" / "local" / "deployment.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--key", action="append", help="dotted field name; repeatable")
    group.add_argument("--all", action="store_true", help="print every declared field as KEY=VALUE")
    group.add_argument("--list", action="store_true", help="print the declared field names")
    args = parser.parse_args(argv)

    if args.list:
        for name in sorted(FIELDS):
            print(name)
        return 0

    try:
        profile = load_local_config(args.config)
    except ConfigError as error:
        print(f"local profile: {error}", file=sys.stderr)
        return 2
    values = profile["values"]
    if args.all:
        # Only the keys this profile actually declares: an omitted optional field is
        # simply not injected, and the caller that needs it gets the exact key to add.
        # Every key the loader accepted is published, not just the fixed FIELDS --
        # the open groups (``paths.*``, ``ports.*``) are how a deployment carries its
        # own directories and port groups, and the shell bridge reads them through
        # ``lp_get``, which goes through this listing.
        args.key = sorted(values)
    missing = [name for name in args.key if name not in values]
    if missing:
        print(f"local profile {args.config}: no value for {', '.join(missing)}; "
              f"add it in the deployment profile", file=sys.stderr)
        return 2

    if args.all:
        for name in args.key:
            print(f"{name}={values[name]}")
        return 0
    for name in args.key:
        print(values[name])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
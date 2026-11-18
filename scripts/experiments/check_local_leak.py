#!/usr/bin/env python3
"""Fail when local machine facts appear inside the engineering base.

The base must be usable from any deployment: hosts, home directories, shared
mounts, loopback service addresses and machine labels belong in the local
deployment profile (``configs/local/deployment.json``), referenced through
``${local.<dotted.key>}``.

Files that still carry such literals are listed, with a reason, in
``configs/quality/local-leak-allowlist.json``.  That list may only shrink: an
entry with no remaining finding is itself an error, so cleared files must be
removed from it in the same change.

An allowlist entry is either one file (``files``) or a whole imported tree
(``trees``).  The tree form exists for exactly one kind of thing: a study tree
imported verbatim at a pinned upstream commit, whose own history is sealed
evidence and is not rewritten here.  A tree entry is stale by the same rule --
if nothing beneath it is flagged any more, the entry must go.

A single line may opt out in place with ``local-leak: allow <reason>``.  That is
for the narrow case the allowlist cannot express -- a hermetic fixture whose
loopback address is its own ephemeral test server, not a deployment endpoint --
and the reason is mandatory: every inline exemption is reported under
``inline`` so a reader can audit the whole set in one scan.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ALLOWLIST = "configs/quality/local-leak-allowlist.json"
SUFFIXES = {".py", ".sh", ".bash", ".json", ".yaml", ".yml", ".toml", ".md", ".cfg", ".ini", ".txt"}
# Extension-less files that are still part of the base and carry machine facts
# (an image ENV pinning a shared mount, for one).
NAMED_FILES = {"Dockerfile", "Containerfile", "Makefile"}
SKIP_DIRECTORIES = {"__pycache__", ".git", ".pytest_cache", "node_modules"}
# The scanner necessarily contains the literals it looks for: it is the one file
# where a shared mount inside a string is a rule, not a leak.
SKIP_FILES = {"scripts/experiments/check_local_leak.py"}

# Each rule names the local fact it detects.  A match is reported as
# file:line:rule with the value that should have come from the profile.
RULES = (
    ("shared-mount", re.compile(r"/mnt/[A-Za-z0-9_.\-]+"), "storage.shared_root / storage.archive_root"),
    ("home-path", re.compile(r"/home/[A-Za-z0-9_.\-]+"), "storage.hot_root or roots.*"),
    ("hot-state-root", re.compile(r"/var/lib/omni-local-state"), "storage.hot_root"),
    ("machine-label", re.compile(r"\bA800I(?:-[A-Za-z0-9]+)?\b"), "hosts.*"),
    ("loopback", re.compile(r"\b127\.0\.0\.1\b"), "services.<name>.host"),
    # A host alias is only a local fact where it names a machine: as an ssh/scp
    # target, an ``@host`` suffix, or a host field.  A bare ``"npu"`` is this
    # codebase's resource kind (``cpu`` vs ``npu``) and is not host information.
    ("host-alias", re.compile(r"\bssh\s+(?:-\S+\s+)*(?:[a-z0-9_.\-]+@)?(?:npu2?|npu-[0-9]+)\b"
                              r"|\bscp\b[^\n]*\b(?:npu2?|npu-[0-9]+):"
                              r"|['\"]host['\"]\s*:\s*['\"](?:npu2?|npu-[0-9]+)['\"]"
                              r"|@(?:npu2|npu-[0-9]+)\b"),
     "hosts.training / hosts.rollout"),
)

# A line that already defers to configuration is not a leak: the placeholder and
# macro forms are exactly the correct spelling.
EXEMPT = ("${local.", "${shared_root}", "${run_root}", "${config_dir}", "/REPLACE_WITH_",
          "$HOME", "${HOME}", "%(home)s")

# ``local-leak: allow <reason>`` exempts one line: a hermetic fixture that owns
# its own ephemeral loopback server rather than naming a deployment endpoint.
# The reason must be non-empty, and every such line is reported for audit.
INLINE = re.compile(r"local-leak:\s*allow\s+(\S.*?)\s*$")


def findings(root, paths):
    """Scan explicit files; return one record per offending line, plus inline exemptions."""
    root = Path(root).resolve()
    results, inline = [], []
    for name in sorted(paths):
        path = root / name
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as error:
            results.append({"path": name, "line": 0, "rule": "unreadable", "match": str(error)})
            continue
        for number, line in enumerate(lines, 1):
            allowed = INLINE.search(line)
            if allowed:
                inline.append({"path": name, "line": number, "reason": allowed.group(1)})
            if any(marker in line for marker in EXEMPT):
                continue
            for rule, pattern, hint in RULES:
                # The inline marker may only retire a loopback finding: a fixture's
                # own ephemeral server is not an endpoint, but a shared mount or a
                # home directory never becomes correct by annotating it.
                if allowed and rule == "loopback":
                    continue
                for match in pattern.finditer(line):
                    results.append({"path": name, "line": number, "rule": rule,
                                    "match": match.group(0), "hint": f"move to {hint} in the local profile"})
    return results, inline


def tracked_files(root, scopes):
    """Yield repo-relative paths in the scanned scopes; never follows symlinks out."""
    root = Path(root).resolve()
    seen = []
    for scope in scopes:
        base = root / scope
        candidates = [base] if base.is_file() else sorted(base.rglob("*")) if base.is_dir() else []
        for path in candidates:
            if not path.is_file():
                continue
            if path.suffix not in SUFFIXES and path.name not in NAMED_FILES:
                continue
            name = str(path.relative_to(root))
            if name in SKIP_FILES or SKIP_DIRECTORIES & set(path.relative_to(root).parts):
                continue
            seen.append(name)
    return seen


def load_allowlist(root):
    """Return (files, trees); both validated, so a bare path is never silently accepted."""
    path = Path(root) / ALLOWLIST
    if not path.is_file():
        return {}, {}
    data = json.loads(path.read_text(encoding="utf-8"))
    files = data.get("files", {})
    trees = data.get("trees", {})
    for kind, entries in (("files", files), ("trees", trees)):
        for name, entry in entries.items():
            for field in ("reason", "owner", "date"):
                if not isinstance(entry, dict) or not entry.get(field):
                    raise ValueError(f"{ALLOWLIST}: {kind}: {name} needs reason, owner and date")
    return files, trees


def under(name, prefix):
    """A tree entry covers the directory itself and everything beneath it, nothing else."""
    return name == prefix or name.startswith(prefix.rstrip("/") + "/")


def check(root, scopes=("tools", "scripts", "configs", "tests", "docker", "adapters",
                        "studies/speech-aware-evidence-acquisition/configs",
                        "studies/speech-aware-evidence-acquisition/scripts")):
    root = Path(root).resolve()
    files, trees = load_allowlist(root)
    tracked = tracked_files(root, scopes)
    found, inline = findings(root, tracked)
    reported = [item for item in found
                if item["path"] not in files and not any(under(item["path"], t) for t in trees)]
    by_path = {}
    for item in found:
        by_path.setdefault(item["path"], []).append(item)
    # Both forms shrink the same way: an entry earns its place only while it still
    # covers a finding, so clearing a file or a whole tree says so in the same change.
    stale = sorted(set(files) - set(by_path))
    stale += sorted(t for t in trees if not any(under(name, t) for name in by_path))
    # Report a live tree once, with its file count, rather than one line per file:
    # what a reader has to audit is the exception, and for a sealed import that is
    # the tree, not its 2,000-odd paths.
    live_trees = {}
    for name in by_path:
        for t in trees:
            if under(name, t):
                live_trees[t] = live_trees.get(t, 0) + 1
    return {"clean": not reported and not stale, "violations": reported, "stale_allowlist": stale,
            "allowed": sorted(n for n in by_path if n in files),
            "allowed_trees": live_trees, "inline": inline, "scanned": len(tracked)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(REPO))
    parser.add_argument("--scope", action="append", help="repo-relative path to scan; repeatable")
    args = parser.parse_args(argv)
    scopes = tuple(args.scope) if args.scope else ("tools", "scripts", "configs", "tests", "docker", "adapters",
                                                   "studies/speech-aware-evidence-acquisition/configs",
                                                   "studies/speech-aware-evidence-acquisition/scripts")
    result = check(args.root, scopes)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if result["clean"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
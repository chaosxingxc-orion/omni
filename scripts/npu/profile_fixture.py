#!/usr/bin/env python3
"""A throwaway deployment profile for the offline launcher tests.

The launchers (``npu_serve.sh``, ``npu_serve_e.sh``) take every path, the image
and the bind address from the local deployment profile, so an offline test that
runs one must supply a profile through ``OMNI_LOCAL_CONFIG``.  This builds one
in a temporary directory: placeholders for everything the test does not
exercise, and -- when a sealed arm-A capture is available -- the model root and
the bind host read out of that capture, so the printed argv can still be
compared with the recorded launch byte for byte.

Nothing here is a machine value.  The repository therefore carries no path, host
or image reference, and the tests run on a clean clone.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

SCHEMA = "omni-local-deployment-v1"
DIGEST = "sha256:" + "0" * 64
_ENSURED: Path | None = None


def _capture_roots(capture: Path) -> tuple[str, str] | None:
    """The model root and bind host a captured arm-A launch used, or None."""
    try:
        record = json.loads(Path(capture).read_text(encoding="utf-8"))
        cmd = record["instances"][0]["cmd"]
        model_dir = Path(cmd[cmd.index("serve") + 1])
        return str(model_dir.parent), cmd[cmd.index("--host") + 1]
    except (OSError, ValueError, KeyError, IndexError):
        return None


def write(directory, capture=None, source=None) -> Path:
    """Write a schema-valid profile under ``directory`` and return its path.

    ``roots.source`` defaults to this repository: the study tree lives inside it,
    so the offline tests read the real deployment profiles and run lists.  Every
    other path points into the temporary directory.
    """
    base = Path(directory).resolve()
    base.mkdir(parents=True, exist_ok=True)
    derived = _capture_roots(capture) if capture else None
    model_root, service_host = derived or (str(base / "models"), "fixture-host")
    source_root = str(Path(source).resolve() if source else Path(__file__).resolve().parents[2])
    profile = {
        "schema": SCHEMA,
        "hosts": {"training": "fixture-training", "rollout": "fixture-rollout"},
        "storage": {"shared_root": str(base / "shared"), "hot_root": str(base / "hot"),
                    "archive_root": str(base / "archive")},
        "roots": {"model": model_root, "data": str(base / "data"), "source": source_root},
        "image": {"reference": "fixture-image:1", "digest": DIGEST},
        "services": {"omni": {"host": service_host, "port": 8200}},
        "identity": {"uid": os.getuid(), "gid": os.getgid()},
    }
    path = base / "deployment.json"
    path.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def ensure(capture=None) -> Path:
    """A profile for the whole test process, without disturbing a real deployment.

    A test module that exercises a host tool imports this first.  When the
    environment already names a profile, or the repository carries one, that
    file is used unchanged; otherwise a temporary fixture is written and
    exported through ``OMNI_LOCAL_CONFIG`` so the tool can read it.
    """
    global _ENSURED
    if _ENSURED is not None:
        return _ENSURED
    explicit = os.environ.get("OMNI_LOCAL_CONFIG")
    repo = Path(__file__).resolve().parents[2]
    candidate = Path(explicit) if explicit else repo / "configs" / "local" / "deployment.json"
    if candidate.is_file():
        _ENSURED = candidate
        return _ENSURED
    _ENSURED = write(tempfile.mkdtemp(prefix="omni-profile-"), capture=capture)
    os.environ["OMNI_LOCAL_CONFIG"] = str(_ENSURED)
    return _ENSURED

"""Local deployment profile: the only place host, path and serving facts may live.

A deployment profile carries what differs between machines and people.  It is
never committed; ``configs/local/deployment.example.json`` is the committed
template.  Experiment configurations reference these values only through
``${local.<dotted.key>}`` macros, so the public base holds no local address,
path, port or host alias.

The file is read exactly once, at parse time.  Every later stage reads the
frozen ``CONFIG.json`` snapshot, so a run stays reproducible after the profile
is edited or removed.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .config import ConfigError, _unique_pairs, canonical

SCHEMA = "omni-local-deployment-v1"
_DIGEST = re.compile(r"sha256:[a-fA-F0-9]{64}\Z")
_CREDENTIAL = re.compile(
    r"(token|password|passwd|secret|api[_-]?key|apikey|credential|private[_-]?key)", re.IGNORECASE)

# Declared fields.  A field omitted from the file is simply not injected; the
# experiment that references it then fails with the exact key to add.  Unknown
# fields are rejected so a typo cannot silently do nothing.
FIELDS = {
    "hosts.training": "host",
    "hosts.rollout": "host",
    "storage.shared_root": "path",
    "storage.hot_root": "path",
    "storage.archive_root": "path",
    "roots.model": "path",
    "roots.data": "path",
    "roots.source": "path",
    "image.reference": "text",
    "image.digest": "digest",
    "services.omni.host": "host",
    "services.omni.port": "port",
    "identity.uid": "uid",
    "identity.gid": "gid",
    # The machine's own name, when a tool must refuse to run elsewhere.  A
    # deployment that omits it simply cannot perform that check.
    "identity.hostname": "host",
}
REQUIRED = (
    "storage.shared_root",
    "storage.hot_root",
    "storage.archive_root",
    "roots.model",
    "roots.data",
    "image.reference",
    "image.digest",
)
# Open groups: a deployment's own extra directories and its own groups of
# serving ports.  Their names are deployment-specific, so the groups are open:
# ``paths.<name>`` is any absolute path, ``ports.<name>`` is any list of distinct
# ports.  This keeps a host utility from inventing a fifth hard-coded literal
# while staying inside the one profile file.
EXTRA_PREFIXES = {"paths.": "path", "ports.": "port_list"}


def _flatten(value, prefix, output):
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ConfigError("local profile: keys must be nonempty strings")
        name = key if not prefix else prefix + "." + key
        if isinstance(item, dict) and item:
            _flatten(item, name, output)
        elif isinstance(item, dict):
            # An empty object carries no value; surface it as an unknown field so a
            # placeholder group cannot silently do nothing.
            output[name] = {}
        else:
            output[name] = item


def _path(value, name):
    if not isinstance(value, str) or not value or "\0" in value or "\n" in value:
        raise ConfigError(f"local profile {name}: absolute clean path required")
    candidate = Path(value)
    if not candidate.is_absolute() or ".." in candidate.parts:
        raise ConfigError(f"local profile {name}: absolute clean path required")
    return value


def _check(name, kind, value):
    if _CREDENTIAL.search(name.rsplit(".", 1)[-1]):
        raise ConfigError(f"local profile {name}: credentials are not allowed in a deployment profile")
    if isinstance(value, str) and "${" in value:
        raise ConfigError(f"local profile {name}: macro expansion is not allowed in a deployment profile")
    if kind == "path":
        return _path(value, name)
    if kind == "digest":
        if not isinstance(value, str) or not _DIGEST.fullmatch(value):
            raise ConfigError(f"local profile {name}: sha256:<64 hex> digest required")
        return value.lower()
    if kind == "host":
        if not isinstance(value, str) or not value.strip() or any(c in value for c in " \t\r\n\0"):
            raise ConfigError(f"local profile {name}: host name required")
        return value
    if kind == "port":
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535:
            raise ConfigError(f"local profile {name}: integer port 1-65535 required")
        return value
    if kind in ("uid", "gid"):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ConfigError(f"local profile {name}: non-negative integer required")
        return value
    if kind == "port_list":
        if not isinstance(value, list) or not value:
            raise ConfigError(f"local profile {name}: a nonempty list of ports required")
        ports = [_check(name, "port", item) for item in value]
        if len(set(ports)) != len(ports):
            raise ConfigError(f"local profile {name}: ports must be distinct")
        return ports
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"local profile {name}: nonempty string required")
    return value


def load_local_config(filename):
    """Load and strictly validate a deployment profile.  No filesystem writes."""
    source = Path(filename).resolve()
    try:
        raw = json.loads(source.read_text(encoding="utf-8"), object_pairs_hook=_unique_pairs,
                         parse_constant=lambda v: (_ for _ in ()).throw(ConfigError(f"invalid number: {v}")))
    except (OSError, ValueError) as error:
        raise ConfigError(f"cannot load local profile: {error}") from error
    if not isinstance(raw, dict):
        raise ConfigError("local profile: expected object")
    if raw.pop("schema", None) != SCHEMA:
        raise ConfigError(f"local profile schema must be {SCHEMA}")
    flat = {}
    _flatten(raw, "", flat)
    unknown = {name for name in flat
               if name not in FIELDS and not any(name.startswith(p) and len(name) > len(p)
                                                 for p in EXTRA_PREFIXES)}
    if unknown:
        raise ConfigError(f"local profile: unknown fields {sorted(unknown)}")
    missing = [name for name in REQUIRED if name not in flat]
    if missing:
        # Name the key the way an experiment references it, not only the way the
        # profile spells it: the newcomer meets these first, from an experiment that
        # never ran, and the two spellings differ by the ``local.`` prefix.
        macros = ", ".join("${local." + name + "}" for name in missing)
        raise ConfigError(f"local profile {source}: missing required fields {missing}; "
                          f"add them in the deployment profile ({macros}) and pass it with --local-config")

    def kind_of(name):
        if name in FIELDS:
            return FIELDS[name]
        prefix = next(p for p in EXTRA_PREFIXES if name.startswith(p))
        return EXTRA_PREFIXES[prefix]

    values = {name: _check(name, kind_of(name), flat[name]) for name in sorted(flat)}
    body = {"schema": SCHEMA, **values}
    return {"path": str(source), "values": values,
            "file_sha256": hashlib.sha256(canonical(body)).hexdigest()}


def used_digest(values):
    """Digest over the subset of profile values an experiment actually referenced."""
    return hashlib.sha256(canonical(values)).hexdigest()

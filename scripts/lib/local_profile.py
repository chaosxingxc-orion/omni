#!/usr/bin/env python3
"""Read the local deployment profile from Python host tools.

The shell entry points reach the profile through ``local_profile.sh``; Python
host tools (``flight_steps``, ``engine_matrix``, ``prelim_chain``, ...) use this
module so both halves resolve the same file with the same rules.

Resolution order, identical to the shell bridge:

1. ``OMNI_LOCAL_CONFIG`` — an explicit profile path;
2. ``$OMNI_BASE_ROOT/configs/local/deployment.json``;
3. ``<repository>/configs/local/deployment.json``.

There is no fallback value.  A caller that needs a field asks for it by name and
gets the same strict validation the experiment CLI applies; a missing profile or
field raises ``ProfileError`` with the path and the key to add.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_BASE = Path(os.environ.get("OMNI_BASE_ROOT") or REPO).resolve()
if str(_BASE) not in sys.path:
    sys.path.insert(0, str(_BASE))

from tools.experiments.config import ConfigError  # noqa: E402
from tools.experiments.local import load_local_config  # noqa: E402


class ProfileError(RuntimeError):
    """The deployment profile is missing, malformed or lacks a requested field."""


_OVERRIDE: str | None = None
_CACHE: dict | None = None


def profile_path() -> Path:
    """The profile this process will read; never guessed beyond the documented order."""
    explicit = _OVERRIDE or os.environ.get("OMNI_LOCAL_CONFIG")
    if explicit:
        return Path(explicit)
    return _BASE / "configs" / "local" / "deployment.json"


def load() -> dict:
    """Every declared field of the profile, validated.  Read once per process."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    path = profile_path()
    try:
        _CACHE = load_local_config(path)["values"]
    except ConfigError as error:
        raise ProfileError(f"{path}: {error}") from error
    return _CACHE


def get(key: str, values: dict | None = None) -> str:
    """One required field; the error names the file and the exact key."""
    values = load() if values is None else values
    if key not in values:
        raise ProfileError(f"{profile_path()}: no value for {key}; add it in the deployment profile")
    return values[key]


def optional(key: str, default=None, values: dict | None = None):
    """One field that a deployment may omit, with an explicit caller-side default."""
    values = load() if values is None else values
    return values.get(key, default)


def add_argument(parser, help_text: str = "local deployment profile (default: $OMNI_LOCAL_CONFIG "
                                           "or configs/local/deployment.json)") -> None:
    """Give a host tool the standard ``--local-config`` flag."""
    parser.add_argument("--local-config", default=None, help=help_text)


def use(argument: str | None) -> dict:
    """Load the profile, honouring an explicit ``--local-config`` argument."""
    global _OVERRIDE, _CACHE
    if argument:
        _OVERRIDE = argument
        _CACHE = None
    return load()


_MACRO = re.compile(r"\$\{([^}]+)\}")


def expand(value, values: dict | None = None):
    """Expand ``${local.<dotted.key>}`` macros in a host data file.

    Queue and run-list documents are data, not experiment configurations, but
    they name the same deployment paths; they use the same one namespace and the
    same strictness, so a deployment writes each path in the profile exactly
    once.  An undeclared or non-``local.`` macro raises instead of reaching a
    command line as a literal.  Only ``${`` in a string triggers a profile read,
    so a document that uses none needs no profile.
    """
    if isinstance(value, str):
        def replace(match):
            name = match.group(1)
            if not name.startswith("local."):
                raise ProfileError(
                    f"undeclared variable: ${{{name}}}; host data files may only use ${{local.<key>}}")
            return str(get(name[len("local."):], values))
        result = _MACRO.sub(replace, value)
        if "${" in result:
            raise ProfileError("malformed variable expansion")
        return result
    if isinstance(value, list):
        return [expand(v, values) for v in value]
    if isinstance(value, dict):
        if any("${" in k for k in value):
            raise ProfileError("variables are not allowed in object keys")
        return {k: expand(v, values) for k, v in value.items()}
    return value
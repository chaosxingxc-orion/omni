"""Flight width: how many rows one flight keeps in flight (Gap C, Decision 2 of the 2026-10-30 plan).

Width is a DEPLOYMENT fact: ``instances x admitted level``. The hashed baseline keeps ``serving_geometry.slots``
(``serving_slots``, 4 for the Fold B baseline) as the GEOMETRY fact and it is not reinterpreted here; the deployment
profile only widens how many rows the client keeps in flight against the resident servers. No protocol hash moves.

``admitted_level`` is an OPTIONAL deployment-profile field (requests per instance the owner admitted, at or below the
per-instance scheduling ceiling ``max_num_seqs``). This module reads it straight from the profile JSON so it needs no
change in :mod:`..deployment`. Absent field (or no profile) -> the caller's ``fallback_slots`` is the width.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

#: The optional deployment-profile key this module reads.
ADMITTED_LEVEL_KEY = "admitted_level"


class FlightWidthError(ValueError):
    """The deployment profile cannot give a coherent flight width (bad field, or an admitted level above the ceiling)."""


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise FlightWidthError(f"{field} must be an integer >= 1, got {value!r}")
    return value


def _profile_document(source: object) -> Mapping[str, object]:
    """The profile's JSON document: a mapping (raw profile or deployment record), a profile id or path, or a loaded
    ``DeploymentProfile`` (whose ``profile_id`` names the JSON file)."""

    if isinstance(source, Mapping):
        return source
    # Imported lazily and read-only: only the id -> path resolution is used.
    from ..deployment import resolve_profile_path

    if isinstance(source, (str, Path)):
        reference = source
    else:
        reference = getattr(source, "profile_id", None)
        if not isinstance(reference, str):
            raise FlightWidthError(
                f"deployment profile must be a mapping, a profile id/path or a DeploymentProfile, got {type(source).__name__}"
            )
    path = resolve_profile_path(reference)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise FlightWidthError(f"cannot read deployment profile {path}: {error}") from error
    if not isinstance(document, dict):
        raise FlightWidthError(f"deployment profile {path} must be a JSON object")
    return document


def flight_width(deployment_profile_or_record: object, fallback_slots: int) -> int:
    """Rows in flight for one flight: ``len(instances) x admitted_level`` when the profile declares ``admitted_level``,
    otherwise ``fallback_slots`` (the hashed ``serving_slots``). ``None`` means no profile is attached.

    Refuses an ``admitted_level`` above the per-instance ceiling (``max_num_seqs``, or ``max_num_seqs_per_instance`` in a
    deployment record), a non-integer level, and a profile with no instances."""

    fallback = _positive_int(fallback_slots, "fallback_slots")
    if deployment_profile_or_record is None:
        return fallback
    document = _profile_document(deployment_profile_or_record)
    admitted = document.get(ADMITTED_LEVEL_KEY)
    if admitted is None:
        return fallback
    admitted = _positive_int(admitted, ADMITTED_LEVEL_KEY)
    ceiling = document.get("max_num_seqs", document.get("max_num_seqs_per_instance"))
    ceiling = _positive_int(ceiling, "max_num_seqs")
    if admitted > ceiling:
        raise FlightWidthError(
            f"admitted_level {admitted} is above the profile ceiling max_num_seqs {ceiling}; the engine would queue or preempt"
        )
    instances = document.get("instances")
    if not isinstance(instances, (list, tuple)) or not instances:
        raise FlightWidthError("deployment profile declares no instances; flight width needs instances x admitted_level")
    return len(instances) * admitted

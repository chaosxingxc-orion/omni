"""Content-bound verification context for historical TF-GRPO imports.

The context is deliberately process-local.  It reuses an already parsed,
content-pinned JSON object while retaining the file identity and stat receipt
needed to invalidate the entry when the source is replaced or removed.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .tfgrpo_learning import LearningError


def _stat_receipt(path: Path) -> tuple[int, int, int, int, int]:
    try:
        stat = path.stat()
    except FileNotFoundError as exc:
        raise LearningError("verified import source disappeared") from exc
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


@dataclass
class _Entry:
    path: Path
    digest: str
    stat: tuple[int, int, int, int, int]
    raw: bytes
    document: Any


class VerifiedImportContext:
    """Reuse verified import inputs without weakening custody checks.

    The cache is scoped to this object and must be explicitly passed through a
    call chain.  It is never serialized or shared across processes.  A cache
    hit is accepted only while the path stat receipt is unchanged; a changed
    receipt causes a fresh byte read and SHA-256 check against the pinned hash.
    """

    def __init__(self, *, repo_root: Path | str | None = None,
                 data_root: Path | str | None = None):
        self.repo_root = Path(repo_root).resolve() if repo_root is not None else None
        self.data_root = Path(data_root).resolve() if data_root is not None else None
        self._entries: dict[tuple[str, str, str], _Entry] = {}

    def _root(self, kind: str) -> Path:
        if kind == "repository" and self.repo_root is not None:
            return self.repo_root
        if kind == "data" and self.data_root is not None:
            return self.data_root
        raise LearningError(f"verified import root unavailable: {kind}")

    def read(self, pin: Mapping[str, object], *, kind: str = "data") -> tuple[Path, Any]:
        if not isinstance(pin, Mapping) or set(pin) - {"root", "relpath", "sha256"} or set(pin) < {"relpath", "sha256"}:
            raise LearningError("exact verified import pin required")
        relpath, digest = pin["relpath"], pin["sha256"]
        if not isinstance(relpath, str) or not isinstance(digest, str):
            raise LearningError("verified import pin fields must be strings")
        root = self._root(kind)
        path = (root / relpath).resolve()
        if not path.is_relative_to(root):
            raise LearningError("verified import pin escaped root")
        key = (kind, relpath, digest)
        stat = _stat_receipt(path)
        entry = self._entries.get(key)
        if entry is not None and entry.path == path and entry.stat == stat:
            # Some filesystems can preserve a coarse timestamp for a rapid
            # same-size replacement.  Recheck the content digest on every hit;
            # the expensive JSON parse remains cached, while custody stays
            # fail-closed.
            try:
                current = path.read_bytes()
            except FileNotFoundError as exc:
                raise LearningError("verified import source disappeared") from exc
            if hashlib.sha256(current).hexdigest() != digest:
                raise LearningError("verified import source changed")
            return path, copy.deepcopy(entry.document)
        try:
            raw = path.read_bytes()
        except FileNotFoundError as exc:
            raise LearningError("verified import source disappeared") from exc
        actual = hashlib.sha256(raw).hexdigest()
        if actual != digest:
            raise LearningError("verified import source changed")
        try:
            document = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LearningError("verified import source is not valid JSON") from exc
        self._entries[key] = _Entry(path=path, digest=digest, stat=_stat_receipt(path),
                                    raw=raw, document=document)
        return path, copy.deepcopy(document)

    def clear(self) -> None:
        """Drop process-local entries; useful after a controlled lifecycle boundary."""
        self._entries.clear()


__all__ = ["VerifiedImportContext"]

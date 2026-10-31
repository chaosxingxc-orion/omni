"""Durable exclusive publication for Linux experiment checkpoints and evidence."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def publish_new(path: Path, data: bytes) -> None:
    """Publish complete bytes, refusing an existing destination even when identical.

    A process interruption before link leaves only a disposable hidden temporary file.
    After link, readers see all bytes. Callers decide whether an existing result is an
    idempotent retry or conflicting evidence; this function never replaces it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.unlink(temporary)

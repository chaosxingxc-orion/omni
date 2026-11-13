"""Durable local/shared receipts and conservative process observations (Linux)."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import socket
import time
import uuid
from pathlib import Path

from .config import ConfigError, canonical, within


def now():
    return time.time()


def read_json(filename, default=None):
    try:
        with open(filename, encoding="utf-8") as stream:
            return json.load(stream)
    except FileNotFoundError:
        return default


def boot_id():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def proc_details(pid):
    # comm may contain spaces and parentheses. The last ')' ends field 2.
    fields = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()
    return {"start_ticks": int(fields[19]), "process_state": fields[0]}


def identity(pid=None):
    pid = os.getpid() if pid is None else pid
    return {"pid": pid, "start_ticks": proc_details(pid)["start_ticks"],
            "boot_id": boot_id(), "host": socket.gethostname()}


def observe(receipt):
    """Never use a recorded RUNNING state as proof of a currently live process."""
    if not receipt or not all(k in receipt for k in ("pid", "start_ticks", "boot_id", "host")):
        return {"state": "unknown", "reason": "no complete process identity"}
    try:
        if receipt["host"] != socket.gethostname() or receipt["boot_id"] != boot_id():
            return {"state": "unknown", "reason": "different host or boot; /proc not inspected"}
        actual = proc_details(receipt["pid"])
        if actual["start_ticks"] != receipt["start_ticks"]:
            return {"state": "absent", "reason": "pid reused"}
        if actual["process_state"] in ("Z", "X"):
            return {"state": "absent", "reason": "process exited (zombie/dead)"}
        return {"state": "alive", "process_state": actual["process_state"]}
    except FileNotFoundError:
        return {"state": "absent", "reason": "process absent; exit outcome remains unknown"}
    except (OSError, ValueError, IndexError) as error:
        return {"state": "unknown", "reason": str(error)}


class Store:
    """All mutations are confined to one resolved run_root; no local fallback."""

    def __init__(self, root):
        self.root = Path(root).resolve()

    def checked(self, filename):
        return Path(within(filename, self.root, "state write"))

    def mkdir(self, directory):
        directory = self.checked(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.checked(directory)
        return directory

    def write(self, filename, value, exclusive=False, raw=False):
        filename = self.checked(filename)
        self.mkdir(filename.parent)
        temporary = filename.parent / f".{filename.name}.{uuid.uuid4().hex}.tmp"
        data = value if raw else canonical(value) + b"\n"
        try:
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            if exclusive:
                os.link(temporary, filename)
                temporary.unlink()
            else:
                os.replace(temporary, filename)
            directory_fd = os.open(filename.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)

    def lock(self, filename, blocking=False):
        filename = self.checked(filename)
        self.mkdir(filename.parent)
        fd = os.open(filename, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BaseException:
            os.close(fd)
            raise
        return fd

    def log(self, filename):
        filename = self.checked(filename)
        self.mkdir(filename.parent)
        fd = os.open(filename, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        return os.fdopen(fd, "ab", buffering=0)


def stat_key(filename):
    stat = Path(filename).stat()
    if not Path(filename).is_file():
        raise ConfigError(f"artifact must be a regular file: {filename}")
    return {"device": stat.st_dev, "inode": stat.st_ino, "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns}


def file_hash(filename):
    result = hashlib.sha256()
    with open(filename, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def verified_file(filename, expected, cached=None, immutable=False):
    before = stat_key(filename)
    # Coarse filesystem timestamps can hide same-size writes. Only large assets
    # with an explicit immutable publication contract may reuse a stat receipt.
    reused = bool(immutable and before["size"] > 1024 * 1024 and cached and cached.get("stat") == before and
                  (expected is None or cached.get("sha256") == expected.lower()))
    checksum = cached["sha256"] if reused else file_hash(filename)
    if stat_key(filename) != before:
        raise ConfigError(f"artifact changed during verification: {filename}")
    if expected is not None and checksum != expected.lower():
        raise ConfigError(f"artifact sha256 mismatch: {filename}")
    return {"path": str(filename), "sha256": checksum, "stat": before, "cache_reused": reused,
            "immutable": immutable}

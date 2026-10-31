"""Verify a sealed generation release; wait on Linux file events, not model polls."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import select
import math


def verify_release(path, *, data, expected):
    document = json.loads(path.read_bytes())
    required = {"schema": "saea-generation-released-v1", "generation_sealed": True,
                "generation_lease_released": True, "scoring_complete": False,
                "consumer_requires_own_lease_and_readiness": True, **expected}
    if any(type(document.get(k)) is not type(v) or document[k] != v for k, v in required.items()):
        raise ValueError("generation release identity or state differs")
    if any(type(document.get(key)) not in (int, float) or not math.isfinite(document[key])
           or document[key] < 0 for key in ('released_at_unix', 'campaign_seconds_to_release')):
        raise ValueError("generation release timing differs")
    root = (data / "runs" / expected["run_id"] / expected["attempt_id"]).resolve()
    if not root.is_relative_to(data.resolve() / "runs"):
        raise ValueError("generation release escapes owned runs")
    filenames = {"generation_manifest": expected["run_id"] + ".generation-manifest.json",
                 "score_ledger_cells": "score-ledger-cells.json", "score_jobs": "score-jobs.jsonl"}
    if set(document.get("artifacts", {})) != set(filenames):
        raise ValueError("generation release needs all sealed inputs")
    seal = document["seal_receipt"]
    for key, filename in filenames.items():
        pin = document["artifacts"][key]
        target = Path(pin["path"])
        if target.is_symlink() or target.resolve() != root / filename:
            raise ValueError("generation release input path differs")
        if seal.get(key + "_path") != str(target) or seal.get(key + "_sha256") != pin["sha256"]:
            raise ValueError("generation release seal differs")
        with target.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != pin["sha256"]:
            raise ValueError("generation release sealed input changed")
    return document


class DirectoryEvents:
    """Watch the existing queue directory before checking for a new run folder.

    A timeout checks the SAME child every600s; it never restarts a job. Kernel
    notifications wake immediately when the producer publishes the release.
    """
    def __init__(self, directory):
        self.lib = ctypes.CDLL(None, use_errno=True)
        self.fd = self.lib.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
        if self.fd < 0:
            raise OSError(ctypes.get_errno(), "inotify_init1")
        try:
            self.add(directory)
        except BaseException:
            self.close()
            raise

    def add(self, directory):
        # IN_CLOSE_WRITE | IN_MOVED_TO | IN_CREATE
        if self.lib.inotify_add_watch(self.fd, os.fsencode(directory), 0x0188) < 0:
            raise OSError(ctypes.get_errno(), "inotify_add_watch")

    def wait(self, timeout=600):
        if select.select([self.fd], [], [], timeout)[0]:
            while True:
                try:
                    os.read(self.fd, 65536)
                except BlockingIOError:
                    break
            return True
        return False

    def close(self):
        os.close(self.fd)


def wait_release(child, path, *, data, expected, events_factory=DirectoryEvents):
    # The producer creates path.parent; its existing parent is watched first.
    events = events_factory(path.parent.parent)
    watched = False
    try:
        while True:
            if not watched and path.parent.exists():
                events.add(path.parent)
                watched = True
            if path.exists():
                return verify_release(path, data=data, expected=expected)
            code = child.poll()
            if code is not None:
                # Close the race with publication immediately before process exit.
                if path.exists():
                    return verify_release(path, data=data, expected=expected)
                raise RuntimeError("generation exited without release; preserve checkpoints: " + str(code))
            events.wait(600)
    finally:
        events.close()

"""Interruption and competing writers cannot publish partial checkpoints."""
import concurrent.futures
import os

import pytest

from speech_aware_evidence_acquisition.core import atomic_file


def test_publish_failure_preserves_absence_and_retry(tmp_path, monkeypatch):
    target = tmp_path / "state.json"
    link = os.link
    def fail(*args):
        raise OSError("interrupted")
    monkeypatch.setattr(atomic_file.os, "link", fail)
    with pytest.raises(OSError, match="interrupted"):
        atomic_file.publish_new(target, b'complete')
    assert not target.exists()
    assert not list(tmp_path.iterdir())
    monkeypatch.setattr(atomic_file.os, "link", link)
    atomic_file.publish_new(target, b'complete')
    with pytest.raises(FileExistsError):
        atomic_file.publish_new(target, b'other')
    assert target.read_bytes() == b'complete'


def test_competing_writers_publish_exactly_one_complete_payload(tmp_path):
    target = tmp_path / "checkpoint"
    payloads = [bytes([n]) * 10000 for n in range(8)]
    def publish(data):
        try:
            atomic_file.publish_new(target, data)
            return data
        except FileExistsError:
            return None
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        winners = [r for r in pool.map(publish, payloads) if r is not None]
    assert len(winners) == 1
    assert target.read_bytes() == winners[0]
    assert list(tmp_path.iterdir()) == [target]

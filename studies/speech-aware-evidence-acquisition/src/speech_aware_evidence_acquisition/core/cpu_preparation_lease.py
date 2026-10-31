"""Serialize CPU preparations independently of the answering-core generation lease.

The opt-in protocol pins an exclusive CPU partition and every allowed serving
profile. Existing preparation_lease_root retains kernel locking and failed-contact custody. This never grants model/tool authorization.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from .cpu_encoder_pool import check_cpu_inventory
from .deployment import load_deployment_profile, parse_cpuset


def validate_cpu_partition(encoder, protocol, source_root, *, affinity=None, device_root=Path('/dev')):
    execution = protocol.get("cpu_execution")
    if not isinstance(execution, dict) or set(execution) != {
        "schema", "cpuset", "serving_profiles", "driver_cpuset"
    } or execution["schema"] != "saea-isolated-cpu-preparation-v1":
        raise ValueError("explicit pinned CPU preparation partition required")
    check_cpu_inventory(encoder)
    if encoder["runtime"] != "cpu-float32" or not encoder["packages"]["torch"].endswith("+cpu"):
        raise ValueError("asynchronous preparation requires the CPU-only fp32 encoder")
    cores = parse_cpuset(execution["cpuset"])
    first = encoder["first_core"]
    worker_cores = set(range(first, first + encoder["workers"] * encoder["threads_per_worker"]))
    allowed = os.sched_getaffinity(0) if affinity is None else set(affinity)
    if cores != worker_cores or allowed != cores:
        raise ValueError("CPU partition must equal worker cores and container affinity")
    if cores & parse_cpuset(execution["driver_cpuset"]):
        raise ValueError("CPU partition overlaps inference/analysis drivers")
    if protocol["monitor_interval_seconds"] != 600:
        raise ValueError("asynchronous preparation requires600s monitoring")
    profiles = execution["serving_profiles"]
    expected = {f"qwen3-omni-real500-{topology}-s{seq}-b2048-v1"
                for topology, seq in (("2xtp4",96),("2xtp4",128),("1xtp8",256),("1xtp8",384))}
    if not isinstance(profiles, dict) or set(profiles) != expected:
        raise ValueError("pin all authorized serving profiles before CPU concurrency")
    for name, digest in profiles.items():
        path = Path(source_root) / "configs/deployment" / (name + ".json")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("concurrent serving profile hash changed")
        profile = load_deployment_profile(name, source_root)
        if max(cores) >= profile.host_cpu_count:
            raise ValueError("CPU partition exceeds host CPU count")
        if any(cores & parse_cpuset(instance.cpuset) for instance in profile.instances):
            raise ValueError("CPU partition overlaps an authorized serving instance")
    if any(device_root.glob("davinci*")) or (device_root / "devmm_svm").exists() or (device_root / "hisi_hdc").exists():
        raise ValueError("CPU preparation container must not expose NPU devices")
    return sorted(cores)


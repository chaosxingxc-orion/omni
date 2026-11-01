"""Shared read-only accelerator telemetry for serving and preparation.

``nvidia-smi`` on a GPU host; on an Ascend host, where that binary does not
exist, the same fields are read from ``npu-smi info``'s device table.
"""

import re
import subprocess

GPU_FIELDS = (
    "utilization_percent",
    "memory_used_mib",
    "power_watts",
    "pstate",
    "clocks_sm_mhz",
)
NVIDIA_SMI_QUERY = "utilization.gpu,memory.used,power.draw,pstate,clocks.sm"


def _number(text: object) -> float | None:
    try:
        return float(str(text))
    except (TypeError, ValueError):
        return None


def read_gpu(run=subprocess.run, *, include_capacity=False) -> dict[str, object] | None:
    """One nvidia-smi row, or None when the card cannot be read (a monitor
    that cannot see the GPU keeps sampling the server rather than dying)."""

    fields = GPU_FIELDS + (("memory_total_mib", "memory_free_mib") if include_capacity else ())
    query = NVIDIA_SMI_QUERY + (",memory.total,memory.free" if include_capacity else "")
    try:
        completed = run(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except OSError:
        # No nvidia-smi at all: an Ascend host reports through npu-smi.
        return read_npu(run, include_capacity=include_capacity)
    except subprocess.SubprocessError:
        return None
    rows = (completed.stdout or "").strip().splitlines()
    if getattr(completed, "returncode", 1) != 0 or not rows:
        return None
    parts = [part.strip() for part in rows[0].split(",")]
    if len(parts) != len(fields):
        return None
    sample = dict(zip(fields, parts))
    return {
        name: (sample[name] if name == "pstate" else _number(sample[name]))
        for name in fields
    }



def read_gpu_devices(run=subprocess.run, *, include_capacity=False) -> list[dict[str, object]]:
    """Read every visible accelerator row without collapsing to one device.

    NVIDIA hosts are parsed from every ``nvidia-smi`` CSV row.  Ascend hosts
    fall back to :func:`read_npu_devices`.  An unreadable command returns an
    empty list, allowing low-frequency monitors to keep running while making
    missing per-card evidence explicit.
    """

    fields = GPU_FIELDS + (("memory_total_mib", "memory_free_mib") if include_capacity else ())
    query = NVIDIA_SMI_QUERY + (",memory.total,memory.free" if include_capacity else "")
    try:
        completed = run(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except OSError:
        return read_npu_devices(run, include_capacity=include_capacity)
    except subprocess.SubprocessError:
        return []
    if getattr(completed, "returncode", 1) != 0:
        return []
    devices: list[dict[str, object]] = []
    for device_id, line in enumerate((completed.stdout or "").strip().splitlines()):
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != len(fields):
            continue
        raw = dict(zip(fields, parts))
        devices.append({
            "device_id": device_id,
            **{name: (raw[name] if name == "pstate" else _number(raw[name])) for name in fields},
        })
    return devices


_NPU_HEAD = re.compile(r"^\|\s*(\d+)\s+\S+\s*\|\s*(\S+)\s*\|\s*([\d.]+)\s+[\d.]+\s+")
_NPU_CHIP = re.compile(
    r"^\|\s*\d+\s*\|\s*[0-9A-Fa-f:.]+\s*\|\s*([\d.]+)\s+\d+\s*/\s*\d+\s+(\d+)\s*/\s*(\d+)\s*\|"
)


def parse_npu_smi_table(text: str) -> list[dict[str, object]]:
    """The per-device rows of ``npu-smi info``'s default table.

    Each device is two table lines: ``| id name | health | power temp
    hugepages |`` then ``| chip | bus-id | aicore% memory used/total
    hbm used/total |``. A device whose two lines do not both parse is
    dropped, so a reworded table yields no rows rather than wrong ones.
    """

    devices: list[dict[str, object]] = []
    head = None
    for line in text.splitlines():
        match = _NPU_HEAD.match(line)
        if match:
            head = match
            continue
        chip = _NPU_CHIP.match(line)
        if chip and head is not None:
            used, total = int(chip.group(2)), int(chip.group(3))
            devices.append({
                "npu_id": int(head.group(1)),
                "health": head.group(2),
                "power_watts": float(head.group(3)),
                "utilization_percent": float(chip.group(1)),
                "memory_used_mib": float(used),
                "memory_total_mib": float(total),
                "memory_free_mib": float(total - used),
            })
        head = None
    return devices



def read_npu_devices(run=subprocess.run, *, include_capacity=False) -> list[dict[str, object]]:
    """Return telemetry for every parsed Ascend device, preserving device id."""

    try:
        completed = run(["npu-smi", "info"], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return []
    if getattr(completed, "returncode", 1) != 0:
        return []
    devices = parse_npu_smi_table(completed.stdout or "")
    if include_capacity:
        return [dict(device) for device in devices]
    return [
        {key: value for key, value in device.items() if key not in {"memory_total_mib", "memory_free_mib"}}
        for device in devices
    ]


def read_npu(run=subprocess.run, *, include_capacity=False) -> dict[str, object] | None:
    """The ``read_gpu`` fields for the most memory-constrained visible NPU.

    The resource guard admits on free accelerator memory, so with several
    cards visible the one with the least free HBM is reported (the safe
    direction). ``pstate`` carries npu-smi's health string; ``clocks_sm_mhz``
    has no table column and is None. None when the table cannot be read.
    """

    try:
        completed = run(["npu-smi", "info"], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    if getattr(completed, "returncode", 1) != 0:
        return None
    devices = parse_npu_smi_table(completed.stdout or "")
    if not devices:
        return None
    device = min(devices, key=lambda row: (row["memory_free_mib"], row["npu_id"]))
    sample = {
        "utilization_percent": device["utilization_percent"],
        "memory_used_mib": device["memory_used_mib"],
        "power_watts": device["power_watts"],
        "pstate": device["health"],
        "clocks_sm_mhz": None,
    }
    if include_capacity:
        sample["memory_total_mib"] = device["memory_total_mib"]
        sample["memory_free_mib"] = device["memory_free_mib"]
    return sample

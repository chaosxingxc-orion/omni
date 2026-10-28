"""Ascend telemetry: ``npu-smi info`` read into the shared ``read_gpu`` fields.

The table below is an excerpt of a real ``npu-smi 26.0.rc1`` capture on an Atlas 910B3 host (2026-10-28), with device
1's HBM use raised so the most-constrained-card rule is observable.
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

from speech_aware_evidence_acquisition.core.session import telemetry

TABLE = """\
+------------------------------------------------------------------------------------------------------------------+
| npu-smi 26.0.rc1                            Version: 26.0.rc1                                                    |
+---------------------------+---------------+----------------------------------------------------------------------+
| NPU   Name                | Health        | Power(W)             Temp(C)                 Hugepages-Usage(page)   |
| Chip                      | Bus-Id        | AICore(%)            Memory-Usage(MB)        HBM-Usage(MB)           |
+===========================+===============+======================================================================+
| 0     910B3               | OK            | 94.3                 45                      0    / 0                |
| 0                         | 0000:C1:00.0  | 0                    0    / 0                58405/ 65536            |
+===========================+===============+======================================================================+
| 1     910B3               | OK            | 97.8                 33                      0    / 0                |
| 0                         | 0000:C2:00.0  | 0                    0    / 0                60000/ 65536            |
+===========================+===============+======================================================================+
| NPU     Chip              | Process id    | Process name       | Process memory(MB)    | Process id in container |
+===========================+===============+======================================================================+
| 0       0                 | 1703791       | vllm               | 106                   | NA                      |
| 0       0                 | 1708018       | VLLMWorker_DP      | 54983                 | NA                      |
+===========================+===============+======================================================================+
"""


def _run(stdout=TABLE, returncode=0, nvidia=False):
    seen = []

    def run(argv, **kwargs):
        seen.append(argv)
        if argv[0] == "nvidia-smi" and not nvidia:
            raise FileNotFoundError("nvidia-smi")
        return SimpleNamespace(returncode=returncode, stdout=stdout)

    return run, seen


def test_the_device_table_parses_every_card_and_skips_process_rows():
    rows = telemetry.parse_npu_smi_table(TABLE)
    assert [row["npu_id"] for row in rows] == [0, 1]
    assert rows[0] == {
        "npu_id": 0, "health": "OK", "power_watts": 94.3, "utilization_percent": 0.0,
        "memory_used_mib": 58405.0, "memory_total_mib": 65536.0, "memory_free_mib": 7131.0,
    }


def test_without_nvidia_smi_the_most_constrained_npu_is_reported():
    run, seen = _run()
    sample = telemetry.read_gpu(run, include_capacity=True)
    assert [argv[0] for argv in seen] == ["nvidia-smi", "npu-smi"]
    assert sample == {
        "utilization_percent": 0.0, "memory_used_mib": 60000.0, "power_watts": 97.8, "pstate": "OK",
        "clocks_sm_mhz": None, "memory_total_mib": 65536.0, "memory_free_mib": 5536.0,
    }
    assert set(telemetry.read_gpu(run)) == set(telemetry.GPU_FIELDS)


def test_a_gpu_host_never_consults_npu_smi():
    run, seen = _run(stdout="90, 20000, 95, P2, 180", nvidia=True)
    assert telemetry.read_gpu(run)["memory_used_mib"] == 20000
    assert [argv[0] for argv in seen] == ["nvidia-smi"]


def test_an_unreadable_npu_table_is_no_reading_not_a_wrong_one():
    assert telemetry.read_gpu(_run(stdout="npu-smi: device busy")[0]) is None
    assert telemetry.read_gpu(_run(returncode=1)[0]) is None
    reworded = TABLE.replace("58405/ 65536", "58405 of 65536").replace("60000/ 65536", "60000 of 65536")
    assert telemetry.read_gpu(_run(stdout=reworded)[0]) is None

    def neither(argv, **kwargs):
        raise FileNotFoundError(argv[0])

    assert telemetry.read_gpu(neither) is None

    def hung(argv, **kwargs):
        if argv[0] == "nvidia-smi":
            raise FileNotFoundError(argv[0])
        raise subprocess.TimeoutExpired(argv, 15)

    assert telemetry.read_gpu(hung) is None

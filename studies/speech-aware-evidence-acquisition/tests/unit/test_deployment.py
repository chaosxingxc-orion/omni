"""The deployment layer (``core/deployment.py``): profile schema, card refusal, per-instance argv, recorded facts.

``BUSY_181`` is a real ``npu-smi 26.0.rc1`` capture from host 181 (A800I-03, 2026-10-28) while another job held every
card; ``IDLE`` is the same table with the HBM readings lowered to an idle level and the process rows replaced by
npu-smi's "no running processes" line.
"""

from __future__ import annotations

import copy
import json
import shlex

import pytest

from speech_aware_evidence_acquisition.core import deployment as dep
from speech_aware_evidence_acquisition.core.session import telemetry
from speech_aware_evidence_acquisition.core.session.serving import ServingSessionError

PROFILE_ID = "qwen3-omni-thinking-bf16-181-4xtp2-v1"
_BUS = ("C1", "C2", "81", "82", "01", "02", "41", "42")


def _device_rows(hbm: list[int]) -> str:
    rows = []
    for card, (bus, used) in enumerate(zip(_BUS, hbm)):
        rows.append(f"| {card}     910B4-1             | OK            | 92.3                 30                      0    / 0                |")
        rows.append(f"| 0                         | 0000:{bus}:00.0  | 0                    0    / 0                {used:5d}/ 65536            |")
        rows.append("+===========================+===============+======================================================================+")
    return "\n".join(rows)


_HEAD = """\
+------------------------------------------------------------------------------------------------------------------+
| npu-smi 26.0.rc1                            Version: 26.0.rc1                                                    |
+---------------------------+---------------+----------------------------------------------------------------------+
| NPU   Name                | Health        | Power(W)             Temp(C)                 Hugepages-Usage(page)   |
| Chip                      | Bus-Id        | AICore(%)            Memory-Usage(MB)        HBM-Usage(MB)           |
+===========================+===============+======================================================================+
"""
_PROC_HEAD = """
+---------------------------+---------------+----------------------------------------------------------------------+
| NPU     Chip              | Process id    | Process name       | Process memory(MB)    | Process id in container |
+===========================+===============+======================================================================+
"""
BUSY_181 = _HEAD + _device_rows([62846, 62788, 62787, 62788, 62787, 62785, 62786, 62785]) + _PROC_HEAD + "\n".join(
    [f"| 0       0                 | 2979702       | vllm               | 106                   | NA                      |"]
    + [f"| {card}       0                 | {2987175 + card}       | VLLMWorker_DP      | 59439                 | NA                      |"
       for card in range(8)]
) + "\n"
IDLE = _HEAD + _device_rows([3380] * 8) + _PROC_HEAD + "\n".join(
    f"| No running processes found in NPU {card}                                                            |" for card in range(8)
) + "\n"


@pytest.fixture(scope="module")
def profile():
    return dep.load_deployment_profile(PROFILE_ID)


@pytest.fixture(scope="module")
def fragment(profile):
    return dep.model_fragment_values(profile)


def _document():
    return json.loads(dep.resolve_profile_path(PROFILE_ID).read_text(encoding="utf-8"))


# --- profile schema ---------------------------------------------------------------------------------------------------


def test_the_181_profile_is_four_tp2_instances_on_disjoint_cards_ports_and_cores(profile):
    assert profile.profile_id == PROFILE_ID and profile.host == "A800I-03"
    assert [i.name for i in profile.instances] == [f"saea-omni-i{k}" for k in range(4)]
    assert [i.cards for i in profile.instances] == [(0, 1), (2, 3), (4, 5), (6, 7)]
    assert [i.port for i in profile.instances] == [8200, 8201, 8202, 8203]
    assert (profile.tensor_parallel_size, profile.max_num_seqs, profile.gpu_memory_utilization) == (2, 32, 0.92)
    cpus = [dep.parse_cpuset(i.cpuset) for i in profile.instances]
    # Each instance owns its cards' NUMA node (32 cores); nodes 1/3/5/7 stay free for the harness and CPU encoders.
    assert [len(c) for c in cpus] == [32] * 4 and len(frozenset().union(*cpus)) == 128


def test_every_committed_profile_validates_and_is_named_by_its_file():
    paths = sorted(dep.REPO_ROOT.joinpath(*dep.DEPLOYMENT_CONFIG_DIR).glob("*.json"))
    assert len(paths) >= 2
    for path in paths:
        assert dep.load_deployment_profile(path).profile_id == path.stem
        assert path.read_text(encoding="utf-8") == json.dumps(json.loads(path.read_text()), indent=2, sort_keys=True) + "\n"


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d.update(extra=1), "unknown keys"),
        (lambda d: d.pop("basis"), "missing keys"),
        (lambda d: d.update(schema="v0"), "schema"),
        (lambda d: d["instances"][1].update(cards=[1, 2]), "reuses card"),
        (lambda d: d["instances"][1].update(port=8200), "reuses port"),
        (lambda d: d["instances"][1].update(name="saea-omni-i0"), "reuses name"),
        (lambda d: d["instances"][1].update(cpuset="190-200"), "reuses cpu"),
        (lambda d: d["instances"][0].update(cards=[0]), "tensor_parallel_size"),
        (lambda d: d["instances"][0].update(cards=[1, 0]), "strictly increasing"),
        (lambda d: d["instances"][0].update(cards=[7, 8]), "cards"),
        (lambda d: d["instances"][0].update(cpuset="250-260"), "host_cpu_count"),
        (lambda d: d["instances"][0].update(cpuset="9-3"), "reversed"),
        (lambda d: d["instances"][0].update(name="omni-i0"), "name"),
        (lambda d: d["instances"][0]["env"].update(HF_HUB_OFFLINE="0"), "common environment"),
        (lambda d: d["instances"][0].update(server_args=["--max-model-len", "1"]), "exactly the keys"),
        (lambda d: d.update(gpu_memory_utilization=1.2), "gpu_memory_utilization"),
        (lambda d: d.update(max_num_seqs=0), "max_num_seqs"),
        (lambda d: d.update(model_relpath="/data/qwen"), "plain path relative"),
        (lambda d: d.update(model_relpath="models/../../x"), "plain path relative"),
        (lambda d: d.update(model_relpath="."), "plain path relative"),
        (lambda d: d.update(data_mount="mnt/omni"), "absolute path"),
        (lambda d: d.update(data_mount="/mnt/cephfs/xingchao/../x"), "absolute path"),
        (lambda d: d.update(instances=[]), "non-empty"),
    ],
)
def test_a_malformed_profile_is_refused(mutate, message):
    document = _document()
    mutate(document)
    with pytest.raises(dep.DeploymentError, match=message):
        dep.validate_deployment_profile(document)


def test_a_profile_whose_id_differs_from_its_file_is_refused(tmp_path):
    document = _document()
    path = tmp_path / "other-name.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(dep.DeploymentError, match="declares profile_id"):
        dep.load_deployment_profile(path)


def test_the_profile_hash_covers_every_recorded_field():
    document = _document()
    base = dep.validate_deployment_profile(document).sha256
    document["max_num_seqs"] = 24
    assert dep.validate_deployment_profile(document).sha256 != base


# --- card allocation -------------------------------------------------------------------------------------------------


def test_idle_cards_are_allocated_all_or_nothing(profile):
    allocation = dep.allocate_from_npu_smi(profile, IDLE)
    assert allocation == {f"saea-omni-i{k}": (2 * k, 2 * k + 1) for k in range(4)}


def test_the_busy_181_table_refuses_every_card_with_every_reason(profile):
    assert dep.npu_smi_process_cards(BUSY_181) == frozenset(range(8))
    with pytest.raises(dep.CardRefusal) as caught:
        dep.allocate_from_npu_smi(profile, BUSY_181)
    assert sorted(caught.value.refusals) == list(range(8))
    assert caught.value.refusals[0] == ("HBM used 62846.0 MiB > 4096 MiB", "holds a process")


def test_one_busy_card_refuses_the_whole_profile_but_not_an_instance_without_it(profile):
    rows = telemetry.parse_npu_smi_table(IDLE)
    rows[5]["memory_used_mib"] = 40000.0
    with pytest.raises(dep.CardRefusal, match="NPU 5: HBM used 40000.0") as caught:
        dep.allocate_cards(profile, rows)
    assert list(caught.value.refusals) == [5]
    assert dep.allocate_cards(profile, rows, instances=profile.instances[:2]) == {
        "saea-omni-i0": (0, 1), "saea-omni-i1": (2, 3)
    }


def test_a_process_on_a_low_memory_card_still_refuses_it(profile):
    rows = telemetry.parse_npu_smi_table(IDLE)
    with pytest.raises(dep.CardRefusal, match="NPU 3: holds a process"):
        dep.allocate_cards(profile, rows, process_cards={3})


def test_absent_unhealthy_and_unreadable_cards_are_refused(profile):
    rows = [row for row in telemetry.parse_npu_smi_table(IDLE) if row["npu_id"] != 7]
    rows[0]["health"] = "Warning"
    with pytest.raises(dep.CardRefusal) as caught:
        dep.allocate_cards(profile, rows)
    assert caught.value.refusals == {0: ("health 'Warning'",), 7: ("absent from the npu-smi table",)}
    with pytest.raises(dep.CardRefusal) as caught:
        dep.allocate_from_npu_smi(profile, "npu-smi: device busy")
    assert sorted(caught.value.refusals) == list(range(8))


def test_an_experiment_declares_cards_and_gets_whole_instances(profile):
    assert [i.name for i in dep.instances_for_cards(profile, 2)] == ["saea-omni-i0"]
    assert len(dep.instances_for_cards(profile, 8)) == 4
    for bad in (1, 3, 0, 10, True):
        with pytest.raises(dep.DeploymentError):
            dep.instances_for_cards(profile, bad)


# --- argv --------------------------------------------------------------------------------------------------------------


def test_the_serve_argv_keeps_the_protocol_window_and_takes_only_max_num_seqs_from_the_profile(profile, fragment):
    argv = dep.instance_server_argv(profile, profile.instances[2], fragment)
    assert argv[:2] == ["serve", "/mnt/cephfs/xingchao/omni/models/qwen3-omni-30b-a3b-thinking"]
    pairs = dict(zip(argv[2::2], argv[3::2]))
    assert pairs["--max-model-len"] == "24576"
    assert pairs["--max-num-seqs"] == "32"
    assert pairs["--port"] == "8202" and pairs["--host"] == "127.0.0.1"
    assert pairs["--tensor-parallel-size"] == "2" and pairs["--gpu-memory-utilization"] == "0.92"
    assert pairs["--served-model-name"] == fragment["served_model_name"]
    assert argv.count("--max-num-seqs") == 1 and argv.count("--max-model-len") == 1
    # Every other fragment flag rides through untouched and in order.
    rest = [token for token in fragment["server_args"]]
    rest[rest.index("--max-num-seqs") + 1] = "32"
    assert argv[-len(rest):] == rest
    assert "id_slot" not in " ".join(argv) and "--max-num-batched-tokens" not in argv


def test_a_fragment_carrying_a_deployment_flag_or_llama_geometry_is_refused(profile, fragment):
    for extra in (["--tensor-parallel-size", "4"], ["--port=9000"], ["--gpu-memory-utilization", "0.5"]):
        bad = dict(fragment, server_args=[*fragment["server_args"], *extra])
        with pytest.raises(dep.DeploymentError, match="deployment flags"):
            dep.instance_server_argv(profile, profile.instances[0], bad)
    with pytest.raises(ServingSessionError):
        dep.instance_server_argv(profile, profile.instances[0], dict(fragment, server_args=[*fragment["server_args"], "-np", "4"]))


def test_max_num_batched_tokens_is_a_deployment_flag_when_declared(fragment):
    document = _document()
    document["max_num_batched_tokens"] = 8192
    profile = dep.validate_deployment_profile(document)
    argv = dep.instance_server_argv(profile, profile.instances[0], fragment)
    assert argv[argv.index("--max-num-batched-tokens") + 1] == "8192"


def test_the_docker_command_mounts_only_its_own_cards_and_is_resident(profile, fragment):
    argv = dep.docker_run_argv(profile, profile.instances[1], fragment)
    assert argv[:5] == ["docker", "run", "--detach", "--name", "saea-omni-i1"]
    assert "--rm" not in argv
    devices = [argv[k + 1] for k, token in enumerate(argv) if token == "--device"]
    assert devices == ["/dev/davinci_manager", "/dev/devmm_svm", "/dev/hisi_hdc", "/dev/davinci2", "/dev/davinci3"]
    pairs = dict(zip(argv[::1], argv[1::1]))
    assert pairs["--restart"] == "unless-stopped" and pairs["--network"] == "host"
    assert pairs["--cpuset-cpus"] == "128-159" and pairs["--entrypoint"] == "vllm"
    assert "/mnt/cephfs/xingchao/omni:/mnt/cephfs/xingchao/omni" in argv
    assert f"{dep.LABEL_PROFILE}={PROFILE_ID}" in argv and f"{dep.LABEL_INSTANCE}=1" in argv
    assert "HCCL_IF_BASE_PORT=61200" in argv and "HF_HUB_OFFLINE=1" in argv
    image = argv.index("saea-base:20261027")
    assert argv[image + 1:] == dep.instance_server_argv(profile, profile.instances[1], fragment)


def test_the_data_root_is_one_overridable_path(fragment):
    moved = dep.load_deployment_profile(PROFILE_ID, data_root="/srv/data")
    assert (moved.data_mount, moved.model_path) == ("/srv/data", "/srv/data/models/qwen3-omni-30b-a3b-thinking")
    assert moved.sha256 == dep.load_deployment_profile(PROFILE_ID).sha256
    argv = dep.docker_run_argv(moved, moved.instances[0], fragment)
    assert "/srv/data:/srv/data" in argv and "/srv/data/models/qwen3-omni-30b-a3b-thinking" in argv
    assert not any("cephfs" in token for token in argv)
    assert dep.deployment_record(moved)["data_mount"] == "/srv/data"
    with pytest.raises(dep.DeploymentError, match="absolute path"):
        dep.load_deployment_profile(PROFILE_ID, data_root="relative/root")


# --- record and CLI ---------------------------------------------------------------------------------------------------


def test_the_deployment_record_is_the_unhashed_deployment_facts(profile):
    record = dep.deployment_record(profile)
    assert record["schema"] == dep.DEPLOYMENT_RECORD_SCHEMA and record["hashed"] is False
    assert (record["instance_count"], record["tensor_parallel_size"], record["max_num_seqs_per_instance"]) == (4, 2, 32)
    assert record["card_count"] == 8 and record["protocol_window_tokens"] == 24576
    assert [i["base_url"] for i in record["instances"]] == [f"http://127.0.0.1:{8200 + k}" for k in range(4)]
    assert record["profile_sha256"] == profile.sha256
    assert json.loads(json.dumps(record, sort_keys=True)) == record
    assert len({i["server_argv_sha256"] for i in record["instances"]}) == 4


def test_plan_prints_one_parseable_docker_command_per_instance(capsys):
    assert dep._main(["plan", PROFILE_ID]) == 0
    out = capsys.readouterr().out
    commands = [shlex.split(line) for line in out.splitlines() if line.startswith("docker run")]
    assert [command[4] for command in commands] == [f"saea-omni-i{k}" for k in range(4)]


def test_emit_refuses_without_a_card_check_and_on_busy_cards(tmp_path, capsysbinary):
    with pytest.raises(dep.DeploymentError, match="requires --npu-smi-file"):
        dep._main(["emit", PROFILE_ID])
    busy = tmp_path / "busy.txt"
    busy.write_text(BUSY_181, encoding="utf-8")
    with pytest.raises(dep.CardRefusal):
        dep._main(["emit", PROFILE_ID, "--npu-smi-file", str(busy)])
    assert capsysbinary.readouterr().out == b""
    idle = tmp_path / "idle.txt"
    idle.write_text(IDLE, encoding="utf-8")
    assert dep._main(["emit", PROFILE_ID, "--npu-smi-file", str(idle)]) == 0
    tokens = capsysbinary.readouterr().out.split(b"\0")
    assert tokens[-1] == b""
    names, cursor = [], 0
    while cursor < len(tokens) - 1:
        names.append(tokens[cursor].decode())
        count = int(tokens[cursor + 1])
        assert tokens[cursor + 2:cursor + 4] == [b"docker", b"run"]
        cursor += 2 + count
    assert names == [f"saea-omni-i{k}" for k in range(4)]


def test_the_fixture_tables_parse_like_the_real_capture():
    assert [row["memory_used_mib"] for row in telemetry.parse_npu_smi_table(BUSY_181)][:2] == [62846.0, 62788.0]
    assert dep.npu_smi_process_cards(IDLE) == frozenset()
    assert len(telemetry.parse_npu_smi_table(copy.copy(IDLE))) == 8

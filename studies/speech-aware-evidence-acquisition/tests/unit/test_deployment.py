"""The deployment layer (``core/deployment.py``): profile schema, card refusal, per-instance argv, recorded facts.

``BUSY_181`` is a real ``npu-smi 26.0.rc1`` capture from host 181 (A800I-03, 2026-10-28) while another job held every
card; ``IDLE`` is the same table with the HBM readings lowered to an idle level and the process rows replaced by
npu-smi's "no running processes" line.
"""

from __future__ import annotations

import copy
import json
import os
import re
import shlex
from pathlib import Path

import pytest

from speech_aware_evidence_acquisition.contracts.gate_runtime_vllm import (
    VLLM_ENTRYPOINT,
    protocol_sha256,
    split_vllm_server_argv,
)
from speech_aware_evidence_acquisition.core import deployment as dep
from speech_aware_evidence_acquisition.core.session import telemetry
from speech_aware_evidence_acquisition.core.session.serving import ServingSessionError

PROFILE_ID = "qwen3-omni-thinking-bf16-181-4xtp2-v1"
NIC = "enp61s0f2"
SCRIPTS = Path(os.environ.get("OMNI_ROOT", "/mnt/cephfs/xingchao/omni")) / "work" / "scripts"
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
    assert (profile.tensor_parallel_size, profile.max_num_seqs, profile.gpu_memory_utilization) == (2, 64, 0.92)
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
        (lambda d: d["instances"][0]["env"].update(VLLM_USE_V1="0"), "deployment variables"),
        (lambda d: d["instances"][0]["env"].update(PYTHONHASHSEED="0"), "deployment variables"),
        (lambda d: d["instances"][0]["env"].update(HCCL_SOCKET_IFNAME="eth0"), "--nic"),
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
    assert pairs["--max-num-seqs"] == "64"
    assert pairs["--port"] == "8202" and pairs["--host"] == "127.0.0.1"
    assert pairs["--tensor-parallel-size"] == "2" and pairs["--gpu-memory-utilization"] == "0.92"
    assert pairs["--served-model-name"] == fragment["served_model_name"]
    assert argv.count("--max-num-seqs") == 1 and argv.count("--max-model-len") == 1
    # Every fragment flag rides through untouched and in order: the hashed fragment holds no deployment knob.
    rest = [token for token in fragment["server_args"]]
    assert "--max-num-seqs" not in rest
    assert argv[-len(rest):] == rest
    assert "id_slot" not in " ".join(argv) and "--max-num-batched-tokens" not in argv


def test_a_fragment_carrying_a_deployment_flag_or_llama_geometry_is_refused(profile, fragment):
    for extra in (["--tensor-parallel-size", "4"], ["--port=9000"], ["--gpu-memory-utilization", "0.5"],
                  ["--max-num-seqs", "4"], ["--max-num-seqs=4"]):
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
    argv = dep.docker_run_argv(profile, profile.instances[1], fragment, nic=NIC)
    assert argv[:5] == ["docker", "run", "--detach", "--name", "saea-omni-i1"]
    assert "--rm" not in argv
    devices = [argv[k + 1] for k, token in enumerate(argv) if token == "--device"]
    assert devices == ["/dev/davinci_manager", "/dev/devmm_svm", "/dev/hisi_hdc", "/dev/davinci2", "/dev/davinci3"]
    pairs = dict(zip(argv[::1], argv[1::1]))
    assert pairs["--restart"] == "no" == dep.RESTART_POLICY and pairs["--network"] == "host"
    assert pairs["--cpuset-cpus"] == "128-159"
    assert pairs["--entrypoint"] == dep.CONTAINER_INTERPRETER == "/usr/local/python3.12.13/bin/python3"
    assert "/mnt/cephfs/xingchao/omni:/mnt/cephfs/xingchao/omni" in argv
    assert f"{dep.LABEL_PROFILE}={PROFILE_ID}" in argv and f"{dep.LABEL_INSTANCE}=1" in argv
    assert "HCCL_IF_BASE_PORT=61200" in argv and "HF_HUB_OFFLINE=1" in argv
    image = argv.index("saea-base:20261027")
    assert argv[image + 1:] == dep.instance_container_args(profile, profile.instances[1], fragment)
    assert argv[image + 1:image + 4] == ["-m", "vllm.entrypoints.cli.main", "serve"]
    assert argv[image + 4:] == dep.instance_server_argv(profile, profile.instances[1], fragment)[1:]


def test_the_live_argv_is_the_gate_entrypoint_form(profile, fragment):
    """The container's live command line is [entrypoint, *args after the image token]; the receipt gate parses it."""

    for instance in profile.instances:
        argv = dep.docker_run_argv(profile, instance, fragment, nic=NIC)
        image = argv.index(profile.image)
        live = [argv[argv.index("--entrypoint") + 1], *argv[image + 1:]]
        assert tuple(live[1:4]) == tuple(VLLM_ENTRYPOINT)
        parsed = split_vllm_server_argv(live)
        assert parsed.argv0 == dep.CONTAINER_INTERPRETER and parsed.model_dir == profile.model_path
        assert parsed.served_model_name == fragment["served_model_name"]
        assert parsed.deployment["--port"] == str(instance.port)
        assert parsed.deployment["--tensor-parallel-size"] == str(profile.tensor_parallel_size)


def _flag_values(argv, flag):
    return [argv[k + 1] for k, token in enumerate(argv) if token == flag]


def test_the_socket_ifnames_are_the_nic_and_the_nic_is_required(profile, fragment):
    argv = dep.docker_run_argv(profile, profile.instances[0], fragment, nic=NIC)
    env = _flag_values(argv, "-e")
    for key in dep.NIC_ENV_KEYS:
        assert f"{key}={NIC}" in env
    assert dep.NIC_ENV_KEYS == ("HCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME", "TP_SOCKET_IFNAME")
    for bad in (None, "", "   ", "a b", "eth0;rm", "x" * 16, "-eth0", 3):
        with pytest.raises(dep.DeploymentError, match="NIC"):
            dep.docker_run_argv(profile, profile.instances[0], fragment, nic=bad)
    with pytest.raises(TypeError):  # keyword-only, no default: a caller cannot forget it
        dep.docker_run_argv(profile, profile.instances[0], fragment)


def test_a_profile_cannot_override_the_nic_or_common_environment(profile, fragment):
    for key in (*dep.NIC_ENV_KEYS, "HF_HUB_OFFLINE"):
        instance = dep.InstanceSpec(0, "saea-x", (0, 1), 9000, "0-3", ((key, "other"),))
        with pytest.raises(dep.DeploymentError, match="owned by the deployment layer"):
            dep.docker_run_argv(profile, instance, fragment, nic=NIC)


def _script_docker_block(name):
    path = SCRIPTS / name
    if not path.is_file():
        pytest.skip(f"{path} is not on this host")
    text = path.read_text(encoding="utf-8")
    block = text[text.index("docker run -d"):]
    block = block[:block.index("> /dev/null")]
    return text, block


@pytest.mark.parametrize("script", ["npu_serve.sh", "npu_serve.sh.s10-next"])
def test_the_emitted_env_and_mounts_equal_the_sidecar_serve_script(profile, fragment, script):
    """``work/scripts/npu_serve.sh`` is the launcher known to bring up multi-card servers: same env names, same
    read-only mounts (the profile adds only its own per-instance keys and the data root)."""

    text, block = _script_docker_block(script)
    script_env = set(re.findall(r"-e \"?(\w+)=", text))
    script_mounts = set(re.findall(r"-v (\S+):\1:ro", block))
    argv = dep.docker_run_argv(profile, profile.instances[0], fragment, nic=NIC)
    emitted_env = {token.split("=")[0] for token in _flag_values(argv, "-e")}
    emitted_mounts = {token[:-3].split(":")[0] for token in _flag_values(argv, "-v") if token.endswith(":ro")}
    assert emitted_mounts == script_mounts == set(dep.ASCEND_READONLY_MOUNTS)
    assert set(dep.NIC_ENV_KEYS) <= script_env <= emitted_env  # the script's HCCL_IF_BASE_PORT is the profile's env
    assert emitted_env - script_env == {"HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"}
    assert "ip -o -4 addr show scope global | awk '{print $2; exit}'" in text


def test_the_s10_next_serve_script_uses_the_gate_entrypoint_form():
    # The sidecar is promoted over npu_serve.sh by a later mv; prefer it while it exists, else the promoted script.
    name = "npu_serve.sh.s10-next" if (SCRIPTS / "npu_serve.sh.s10-next").is_file() else "npu_serve.sh"
    text, block = _script_docker_block(name)
    assert f"--entrypoint {dep.CONTAINER_INTERPRETER} " in block
    assert '"$IMAGE" -m vllm.entrypoints.cli.main serve "$@"' in block
    assert "--entrypoint vllm" not in text
    thinking = text[text.index("  thinking)"):text.index("  embed)")]
    assert '--gpu-memory-utilization "${GPU_MEM_UTIL:-0.92}"' in thinking
    assert "GPU_MEM_UTIL" in text[:text.index("set -euo pipefail")]


def test_the_host_launcher_derives_the_nic_as_the_serve_script_does():
    path = SCRIPTS / "npu_deploy.sh"
    if not path.is_file():
        pytest.skip(f"{path} is not on this host")
    text = path.read_text(encoding="utf-8")
    assert "ip -o -4 addr show scope global | awk '{print $2; exit}'" in text
    assert '--nic "$NIC"' in text and "refused: no global IPv4 NIC" in text


def test_the_data_root_is_one_overridable_path(fragment):
    moved = dep.load_deployment_profile(PROFILE_ID, data_root="/srv/data")
    assert (moved.data_mount, moved.model_path) == ("/srv/data", "/srv/data/models/qwen3-omni-30b-a3b-thinking")
    assert moved.sha256 == dep.load_deployment_profile(PROFILE_ID).sha256
    argv = dep.docker_run_argv(moved, moved.instances[0], fragment, nic=NIC)
    assert "/srv/data:/srv/data" in argv and "/srv/data/models/qwen3-omni-30b-a3b-thinking" in argv
    assert not any("cephfs" in token for token in argv)
    assert dep.deployment_record(moved)["data_mount"] == "/srv/data"
    with pytest.raises(dep.DeploymentError, match="absolute path"):
        dep.load_deployment_profile(PROFILE_ID, data_root="relative/root")


# --- record and CLI ---------------------------------------------------------------------------------------------------


def test_the_deployment_record_is_the_unhashed_deployment_facts(profile):
    record = dep.deployment_record(profile)
    assert record["schema"] == dep.DEPLOYMENT_RECORD_SCHEMA and record["hashed"] is False
    assert (record["instance_count"], record["tensor_parallel_size"], record["max_num_seqs_per_instance"]) == (4, 2, 64)
    assert record["card_count"] == 8 and record["protocol_window_tokens"] == 24576
    assert [i["base_url"] for i in record["instances"]] == [f"http://127.0.0.1:{8200 + k}" for k in range(4)]
    assert record["profile_sha256"] == profile.sha256
    assert json.loads(json.dumps(record, sort_keys=True)) == record
    assert len({i["server_argv_sha256"] for i in record["instances"]}) == 4
    assert record["nic"] is None  # a profile-only read records no NIC; plan/emit never run without one


def test_the_deployment_record_carries_the_nic_and_the_nic_moves_no_protocol_hash(profile):
    record = dep.deployment_record(profile, nic=NIC)
    assert record["nic"] == NIC and json.loads(json.dumps(record, sort_keys=True)) == record
    plain = dep.deployment_record(profile)
    assert {k: v for k, v in record.items() if k not in {"nic", "instances"}} == {
        k: v for k, v in plain.items() if k not in {"nic", "instances"}}
    for with_nic, without in zip(record["instances"], plain["instances"]):
        assert with_nic["server_argv_sha256"] == without["server_argv_sha256"]  # the NIC is not a server flag
        assert with_nic["env"] == without["env"]  # per-instance env stays the profile's own; the NIC is one host fact
    with pytest.raises(dep.DeploymentError, match="NIC"):
        dep.deployment_record(profile, nic="bad nic")
    # The NIC is a deployment fact: the profile hash does not see it, and a receipt carries it in its deployment
    # block, which protocol_sha256 excludes.
    assert dep.load_deployment_profile(PROFILE_ID).sha256 == profile.sha256
    receipt = {"schema": "x", "protocol": {"served_model_name": "m", "server_args": ["--max-model-len", "24576"]},
               "deployment": {"instances": [{"max_num_seqs": 64, "env": {}}]}}
    moved = json.loads(json.dumps(receipt))
    moved["deployment"]["instances"][0]["env"] = dict.fromkeys(dep.NIC_ENV_KEYS, NIC)
    assert protocol_sha256(moved) == protocol_sha256(receipt)


def test_plan_prints_one_parseable_docker_command_per_instance(capsys):
    assert dep._main(["plan", PROFILE_ID, "--nic", NIC]) == 0
    out = capsys.readouterr().out
    assert f"HCCL_SOCKET_IFNAME={NIC}" in out
    commands = [shlex.split(line) for line in out.splitlines() if line.startswith("docker run")]
    assert [command[4] for command in commands] == [f"saea-omni-i{k}" for k in range(4)]


def test_plan_and_emit_refuse_without_a_nic(tmp_path, capsysbinary):
    idle = tmp_path / "idle.txt"
    idle.write_text(IDLE, encoding="utf-8")
    for command in (["plan", PROFILE_ID], ["emit", PROFILE_ID, "--npu-smi-file", str(idle)],
                    ["plan", PROFILE_ID, "--nic", ""], ["emit", PROFILE_ID, "--npu-smi-file", str(idle), "--nic", "a b"]):
        with pytest.raises(dep.DeploymentError, match="NIC"):
            dep._main(command)
    assert capsysbinary.readouterr().out == b""
    assert dep._main(["record", PROFILE_ID, "--nic", NIC]) == 0
    assert json.loads(capsysbinary.readouterr().out)["nic"] == NIC


def test_emit_refuses_without_a_card_check_and_on_busy_cards(tmp_path, capsysbinary):
    with pytest.raises(dep.DeploymentError, match="requires --npu-smi-file"):
        dep._main(["emit", PROFILE_ID, "--nic", NIC])
    busy = tmp_path / "busy.txt"
    busy.write_text(BUSY_181, encoding="utf-8")
    with pytest.raises(dep.CardRefusal):
        dep._main(["emit", PROFILE_ID, "--npu-smi-file", str(busy), "--nic", NIC])
    assert capsysbinary.readouterr().out == b""
    idle = tmp_path / "idle.txt"
    idle.write_text(IDLE, encoding="utf-8")
    assert dep._main(["emit", PROFILE_ID, "--npu-smi-file", str(idle), "--nic", NIC]) == 0
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


def test_mm_processor_cache_type_is_emitted_from_the_profile_only_when_set(fragment):
    document = _document()
    assert "mm_processor_cache_type" not in document
    plain = dep.instance_server_argv(dep.validate_deployment_profile(document), dep.validate_deployment_profile(document).instances[0], fragment)
    assert "--mm-processor-cache-type" not in plain
    document["mm_processor_cache_type"] = "shm"
    profile = dep.validate_deployment_profile(document)
    argv = dep.instance_server_argv(profile, profile.instances[0], fragment)
    assert argv[argv.index("--mm-processor-cache-type") + 1] == "shm"
    assert split_vllm_server_argv(["/p", *VLLM_ENTRYPOINT, *argv[1:]]).deployment["--mm-processor-cache-type"] == "shm"
    assert "--mm-processor-cache-type" in dep.DEPLOYMENT_ONLY_FLAGS


def test_a_recorded_run_never_restarts_itself_and_the_record_says_so(profile, fragment):
    for instance in profile.instances:
        argv = dep.docker_run_argv(profile, instance, fragment, nic=NIC)
        assert argv.count("--restart") == 1 and argv[argv.index("--restart") + 1] == "no"
        assert "unless-stopped" not in argv and "always" not in argv
    assert dep.deployment_record(profile, nic=NIC)["restart_policy"] == "no"


def test_admitted_level_is_an_optional_deployment_only_profile_field():
    document = _document()
    base = dep.validate_deployment_profile(document)
    assert base.admitted_level is None
    document["admitted_level"] = 16
    profile = dep.validate_deployment_profile(document)
    assert profile.admitted_level == 16 and profile.sha256 != base.sha256
    fragment = dep.model_fragment_values(profile)
    # never emitted to the argv: the argv is byte-identical with and without it
    assert dep.instance_server_argv(profile, profile.instances[0], fragment) == dep.instance_server_argv(
        base, base.instances[0], fragment
    )
    assert dep.deployment_record(profile)["admitted_level"] == 16
    assert dep.deployment_record(base)["admitted_level"] is None
    for bad in (0, -1, True, 1.5, "8", base.max_num_seqs + 1):
        with pytest.raises(dep.DeploymentError, match="admitted_level"):
            dep.validate_deployment_profile(dict(document, admitted_level=bad))

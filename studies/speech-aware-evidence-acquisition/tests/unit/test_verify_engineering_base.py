"""The engineering-base acceptance test (``scripts/serving/verify_engineering_base.py``) against recorded fixtures.

No NPU, no server, no network, no docker: every host, docker, HTTP and file read goes through the script's injectable
``Readers`` interface, and :class:`FixtureReaders` below answers from in-memory documents shaped like the real ones
(``docker inspect``, vLLM's ``/tokenize`` and ``/v1/models``, the 2026-10-29 startup log lines, a complete runtime receipt
built by ``_vllm_world.VllmWorld``). The all-pass world is built from the real 4 x TP2 deployment profile and the real
model fragment of the study, so the argv the fixture "containers" run is the argv ``core.deployment`` derives.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from _vllm_world import PINNED_ENV, VllmWorld, sha
from speech_aware_evidence_acquisition.contracts.gate_runtime_vllm import (
    classify_server_flags,
    instance_argv,
    protocol_sha256,
)
from speech_aware_evidence_acquisition.core import deployment as dep

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "serving" / "verify_engineering_base.py"
WRAPPER = Path("/mnt/cephfs/xingchao/omni/work/scripts/npu_verify_base.sh")
LIVE_LOG = Path("/mnt/cephfs/xingchao/omni/work/logs/saea-serve-thinking-20261029T063554.log")
_spec = importlib.util.spec_from_file_location("verify_engineering_base", SCRIPT)
veb = importlib.util.module_from_spec(_spec)
sys.modules["verify_engineering_base"] = veb
_spec.loader.exec_module(veb)

PROFILE_ID = "qwen3-omni-thinking-bf16-181-4xtp2-v1"
PROFILE_PATH = REPO / "configs" / "deployment" / f"{PROFILE_ID}.json"
PROFILE = dep.validate_deployment_profile(json.loads(PROFILE_PATH.read_text(encoding="utf-8")), source=str(PROFILE_PATH))
FRAGMENT_PATH = REPO / "configs" / "model" / f"{PROFILE.model_fragment}.json"
FRAGMENT = json.loads(FRAGMENT_PATH.read_text(encoding="utf-8"))
SERVED = FRAGMENT["served_model_name"]
HOSTNAME = "A800I-03"
NIC = "enp61s0f2"
IMAGE_ID = "sha256:" + "a1" * 32
BASE_LOCAL_ID = "4d683043e4bd"
LOCK_KEY = "qwen3-omni-30b-a3b-thinking-bf16"
LOCK_PATH = "/virtual/base-lock-v1.json"
RECEIPT_PATH = "/virtual/runtime-vllm-bf16-thinking-npu.json"
HOST_FILES = {
    "/usr/local/Ascend/driver/version.info": b"Version=26.0.rc1\nascendhal_version=7.35.23\n",
    "/etc/ascend_install.info": b"Driver_Install_Status=complete\n",
}
KV_TOKENS, KV_CONCURRENCY, BLOCK = 564224, 22.96, 128
PROMPT = (
    "<|im_start|>system\nYou label spoken requests.<|im_end|>\n<|im_start|>user\n"
    "<|audio_start|><|audio_pad|><|audio_end|>Name the intent.<|im_end|>\n<|im_start|>assistant\n"
)
CLOSED_THINK = "<think>\n\n</think>\n\n"


def encode_byte_level(text: str) -> str:
    table = veb.bytes_to_unicode()
    return "".join(table[byte] for byte in text.encode("utf-8"))


class FixtureReaders(veb.Readers):
    """Recorded answers. Every call is logged by method name; anything not recorded is an unreadable input."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.files: dict[str, bytes] = {}
        self.host_files: dict[str, bytes] = {}
        self.realpaths: dict[str, str] = {}
        self.host: dict | None = None
        self.images: dict[str, dict] = {}
        self.containers: dict[str, dict] = {}
        self.logs: dict[str, list[str]] = {}
        self.cmdlines: dict[str, list[str]] = {}
        self.versions: dict[str, str | None] = {}
        self.version_interpreters: set[str | None] = set()
        self.python = "3.12.13"
        self.execution: str | Exception = "saea-container"
        self.runtime_files_check = None
        self.get: dict[str, tuple[int, str]] = {}
        self.post_handler = None

    def _log(self, name: str) -> None:
        self.calls.append(name)

    def read_bytes(self, path):
        self._log("read_bytes")
        if path not in self.files:
            raise veb.ReadError(f"fixture has no file {path}")
        return self.files[path]

    def read_host_bytes(self, path):
        self._log("read_host_bytes")
        if path not in self.host_files:
            raise veb.ReadError(f"fixture has no host file {path}")
        return self.host_files[path]

    def realpath(self, path):
        self._log("realpath")
        if path not in self.realpaths:
            raise veb.ReadError(f"fixture has no path {path}")
        return self.realpaths[path]

    def list_files(self, root):
        self._log("list_files")
        prefix = root.rstrip("/") + "/"
        names = sorted(path[len(prefix):] for path in self.files if path.startswith(prefix))
        if not names:
            raise veb.ReadError(f"fixture has no directory {root}")
        dots = sorted({name.split("/")[0] for name in names if name.startswith(".")})
        return [name for name in names if not name.startswith(".")], dots

    def sha256_files(self, paths):
        self._log("sha256_files")
        out = {}
        for path in paths:
            if path not in self.files:
                raise veb.ReadError(f"fixture cannot hash {path}")
            out[path] = (len(self.files[path]), hashlib.sha256(self.files[path]).hexdigest())
        return out

    def host_facts(self):
        self._log("host_facts")
        if self.host is None:
            raise veb.ReadError("fixture has no host facts")
        return copy.deepcopy(self.host)

    def docker_image_inspect(self, ref):
        self._log("docker_image_inspect")
        if ref not in self.images:
            raise veb.ReadError(f"fixture has no image {ref}")
        return copy.deepcopy(self.images[ref])

    def docker_container_inspect(self, name):
        self._log("docker_container_inspect")
        if name not in self.containers:
            raise veb.ReadError(f"fixture has no container {name}")
        return copy.deepcopy(self.containers[name])

    def container_log_lines(self, name):
        self._log("container_log_lines")
        if name not in self.logs:
            raise veb.ReadError(f"fixture has no log for {name}")
        return list(self.logs[name])

    def container_cmdline(self, name):
        self._log("container_cmdline")
        if name not in self.cmdlines:
            raise veb.ReadError(f"fixture has no cmdline for {name}")
        return list(self.cmdlines[name])

    def dist_version(self, name, interpreter=None):
        self._log("dist_version")
        self.version_interpreters.add(interpreter)
        return self.versions.get(name)

    def python_version(self):
        self._log("python_version")
        return self.python

    def assert_execution_environment(self):
        self._log("assert_execution_environment")
        if isinstance(self.execution, Exception):
            raise self.execution
        return self.execution

    def verify_runtime_files(self, receipt):
        self._log("verify_runtime_files")
        self.runtime_files_check(receipt)

    def http_get(self, url):
        self._log("http_get")
        if url not in self.get:
            raise veb.ReadError(f"fixture cannot reach {url}")
        return self.get[url]

    def http_post_json(self, url, body):
        self._log("http_post_json")
        return self.post_handler(url, body)


class World:
    """One fully consistent deployment: readers, a base-lock printed from them, and the knobs the tests turn."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp = Path(tmp_path)
        self.vw = VllmWorld(self.tmp / "vw")
        self.readers = FixtureReaders()
        self.think_in_true_mode = False
        self.audio_spans = 1
        self.count_skew = 0
        self.model_files = {
            "config.json": b'{"model_type": "qwen3_omni_moe"}\n',
            "chat_template.json": b'{"chat_template": "{{ messages }}"}\n',
            "tokenizer_config.json": b'{"add_bos_token": false}\n',
            "model-00001-of-00002.safetensors": b"BF16-shard-one" * 64,
            "model-00002-of-00002.safetensors": b"BF16-shard-two" * 32,
        }
        self._build_readers()
        self.lock_dict = self.print_lock()

    # -- construction ------------------------------------------------------------------------------------------------
    def _build_readers(self) -> None:
        rd = self.readers
        rd.files[str(PROFILE_PATH)] = PROFILE_PATH.read_bytes()
        rd.files[str(FRAGMENT_PATH)] = FRAGMENT_PATH.read_bytes()
        for name, raw in self.model_files.items():
            rd.files[f"{PROFILE.model_path}/{name}"] = raw
        rd.files[f"{PROFILE.model_path}/.hfd/verified.tsv"] = b"bookkeeping\n"
        rd.files[f"{PROFILE.data_mount}/models/enc-view/model.safetensors"] = b"encoder-weights" * 16
        rd.files[f"{PROFILE.data_mount}/models/enc-view/config.json"] = b"{}\n"
        rd.host_files.update(HOST_FILES)
        rd.files["/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/ascend_toolkit_install.info"] = b"version=8.3\n"
        interpreter = "/usr/local/python3.12.13/bin/python3.12"
        rd.files[interpreter] = b"ELF-python-stub"
        rd.realpaths = {path: interpreter for path in veb.DEFAULT_INTERPRETERS}
        rd.versions = {"vllm": "0.30.0", "vllm-ascend": "0.30.0rc2", "torch": "2.10.0", "torch_npu": "2.10.0.post7",
                       "transformers": "5.14.1", "triton": None}
        rd.host = {"hostname": HOSTNAME, "cpu_count": 256, "card_count": 8,
                   "global_nics": ["docker0", NIC, "br-d3b20b1f3e3f"]}
        rd.images = {PROFILE.image: {"Id": IMAGE_ID, "RepoDigests": [], "Created": "2026-10-27T12:00:00Z"},
                     veb.DEFAULT_BASE_IMAGE_REF: {"Id": "sha256:" + BASE_LOCAL_ID + "0" * 52}}
        rd.runtime_files_check = veb.verify_vllm_runtime_files
        self._build_receipt()
        for instance in PROFILE.instances:
            self._build_instance(instance)
        rd.post_handler = self._post

    def _build_receipt(self) -> None:
        rd, vw = self.readers, self.vw
        inst0 = PROFILE.instances[0]
        argv = dep.instance_server_argv(PROFILE, inst0, FRAGMENT)
        served, tokens, _ = classify_server_flags(argv[2:])
        env = dict(PINNED_ENV, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")  # per-instance HCCL keys live in deployment.env
        receipt = vw.receipt(served_model_name=served, server_args=list(tokens), env=env)
        receipt["model_lock_key"] = LOCK_KEY
        receipt["container"] = {"image_ref": PROFILE.image, "image_id": IMAGE_ID}
        receipt["model"] = {
            "model_dir": PROFILE.model_path,
            "files": [{"path": f"{PROFILE.model_relpath}/{name}", "bytes": len(raw), "sha256": sha(raw)}
                      for name, raw in sorted(self.model_files.items())],
            "other_entries": [".hfd"],
        }
        receipt["deployment"] = {"instances": [
            {"instance_id": f"i{inst.index}", "host": "127.0.0.1", "port": inst.port, "device_ids": list(inst.cards),
             "tensor_parallel_size": PROFILE.tensor_parallel_size, "max_num_seqs": PROFILE.max_num_seqs,
             "gpu_memory_utilization": PROFILE.gpu_memory_utilization, "max_num_batched_tokens": None,
             "env": {**dict(inst.env), **dict.fromkeys(dep.NIC_ENV_KEYS, NIC)}}
            for inst in PROFILE.instances]}
        receipt["protocol_sha256"] = protocol_sha256(receipt)
        self.receipt = receipt
        rd.files[RECEIPT_PATH] = json.dumps(receipt).encode("utf-8")

    def _build_instance(self, inst) -> None:
        rd = self.readers
        env = dict(PINNED_ENV, HCCL_SOCKET_IFNAME=NIC, GLOO_SOCKET_IFNAME=NIC, TP_SOCKET_IFNAME=NIC,
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HOSTNAME="npu", **dict(inst.env),
                   ASCEND_RT_VISIBLE_DEVICES=veb.device_env_value(inst.cards))
        devices = [{"PathOnHost": p} for p in dep.ASCEND_CONTROL_DEVICES] + [
            {"PathOnHost": f"/dev/davinci{card}"} for card in inst.cards]
        rd.containers[inst.name] = {
            "Name": f"/{inst.name}", "Image": IMAGE_ID, "Path": dep.CONTAINER_INTERPRETER,
            "Args": dep.instance_container_args(PROFILE, inst, FRAGMENT), "RestartCount": 0,
            "State": {"Status": "running", "Running": True},
            "Config": {"Env": [f"{k}={v}" for k, v in env.items()],
                       "Labels": {dep.LABEL_PROFILE: PROFILE.profile_id, dep.LABEL_PROFILE_SHA256: PROFILE.sha256,
                                  dep.LABEL_INSTANCE: str(inst.index)}},
            "HostConfig": {"CpusetCpus": inst.cpuset, "NetworkMode": "host", "ShmSize": 16 * 1024 ** 3,
                           "Devices": devices},
            "Mounts": [{"Destination": mount, "RW": False} for mount in dep.ASCEND_READONLY_MOUNTS] + [
                       {"Destination": PROFILE.data_mount, "RW": True}],
        }
        rd.logs[inst.name] = [
            "(APIServer pid=1) INFO 10-29 06:36:54 [utils.py:1461] Block size is set to 128 if prefix cache or chunked "
            "prefill is enabled.",
            "(EngineCore pid=118) INFO 10-29 06:37:22 [core.py:123] Initializing a V1 LLM engine (v0.30.0) with config: "
            "model='x', seed=0, enable_prefix_caching=True, enable_chunked_prefill=True, compilation_config={'mode': 3, "
            "'cudagraph_mode': <CUDAGraphMode.FULL_AND_PIECEWISE: (2, 1)>, 'cudagraph_num_of_warmups': 1, "
            "'cudagraph_capture_sizes': [1, 2, 4], 'cudagraph_copy_inputs': False}",
            "(Worker_TP0 pid=137) INFO 10-29 06:39:24 [worker.py:650] Available KV cache memory: 25.83 GiB",
            f"(EngineCore pid=118) INFO 10-29 06:39:24 [kv_cache_utils.py:2395] NPU KV cache size: {KV_TOKENS:,} tokens, "
            f"Maximum concurrency for 24,576 tokens per request: {KV_CONCURRENCY}x",
        ]
        ri = next(i for i in self.receipt["deployment"]["instances"] if i["port"] == inst.port)
        rd.cmdlines[inst.name] = list(instance_argv(self.receipt, ri))
        base = PROFILE.base_url(inst)
        rd.get[f"{base}/health"] = (200, "")
        rd.get[f"{base}/v1/models"] = (200, json.dumps({"data": [{"id": SERVED, "max_model_len": 24576}]}))
        rd.get[f"{base}/metrics"] = (200, "# HELP\n" + f'vllm:cache_config_info{{block_size="{BLOCK}",num_gpu_blocks="'
                                      f'{KV_TOKENS // BLOCK}",cache_dtype="auto"}} 1.0\nvllm:num_preemptions_total 0.0\n')

    def _prompt(self, thinking: bool) -> str:
        text = PROMPT
        if self.audio_spans != 1:
            extra = "<|audio_start|><|audio_pad|><|audio_end|>" * (self.audio_spans - 1)
            text = text.replace("<|audio_start|>", extra + "<|audio_start|>", 1)
        if thinking:
            return text + ("<think>\n" if self.think_in_true_mode else "")
        return text + CLOSED_THINK

    def _post(self, url: str, body: dict):
        thinking = body["chat_template_kwargs"]["enable_thinking"]
        text = self._prompt(thinking)
        if url.endswith("/tokenize"):
            if body.get("return_token_strs"):
                return 200, {"count": len(text), "token_strs": [encode_byte_level(ch) for ch in text]}
            return 200, {"count": len(text), "tokens": list(range(len(text)))}
        return 200, {"usage": {"prompt_tokens": len(self._prompt(True)) + self.count_skew}}

    # -- running -------------------------------------------------------------------------------------------------------
    def print_lock(self) -> dict:
        out = self.tmp / "printed-lock.json"
        code = veb.main(["--profile", str(PROFILE_PATH), "--print-observed", "--receipt", RECEIPT_PATH,
                         "--rehash-weights", "--encoder", "enc-view=models/enc-view", "--out", str(out),
                         "--repo-root", str(REPO)], readers=self.readers)
        assert code == 0, out.read_text() if out.exists() else "no output"
        lock = json.loads(out.read_text(encoding="utf-8"))
        lock["weights"]["lock_key"] = LOCK_KEY
        self.readers.files[LOCK_PATH] = json.dumps(lock).encode("utf-8")
        return lock

    def set_lock(self, lock: dict) -> None:
        self.lock_dict = lock
        self.readers.files[LOCK_PATH] = json.dumps(lock).encode("utf-8")

    def verify(self, *extra: str, clock: datetime | None = None, readers=None):
        out = self.tmp / f"report-{len(list(self.tmp.glob('report-*.json')))}.json"
        when = clock or datetime(2026, 10, 29, 12, 0, 0, tzinfo=timezone.utc)
        code = veb.main(["--profile", str(PROFILE_PATH), "--base-lock", LOCK_PATH, "--out", str(out),
                         "--repo-root", str(REPO), *extra], readers=readers or self.readers, clock=lambda: when)
        return code, (json.loads(out.read_text(encoding="utf-8")) if out.exists() else None), out


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


# ---------------------------------------------------------------------------------------------------------------------
# All-pass world and the observed values it prints.
# ---------------------------------------------------------------------------------------------------------------------


def test_all_pass_with_every_check_selected(world):
    code, report, _ = world.verify("--rehash-weights")
    assert report["schema"] == "saea-engineering-base-verification-v1"
    assert {name: entry["verdict"] for name, entry in report["checks"].items()} == {name: "pass" for name in veb.CHECK_NAMES}, {
        name: entry["failures"] for name, entry in report["checks"].items() if entry["failures"]}
    assert (code, report["verdict"], report["failed_checks"]) == (0, "pass", [])
    assert report["selected_checks"] == list(veb.CHECK_NAMES) and report["skipped_checks"] == {}


def test_default_selection_skips_weights_and_still_passes(world):
    code, report, _ = world.verify()
    assert code == 0 and "weights" not in report["checks"]
    assert report["skipped_checks"] == {"weights": "--rehash-weights not given"}
    assert "weights" not in report["selected_checks"]


def test_only_selects_a_subset(world):
    code, report, _ = world.verify("--only", "image,kv")
    assert code == 0 and sorted(report["checks"]) == ["image", "kv"]


def test_unknown_check_name_is_a_usage_error(world):
    code, report, _ = world.verify("--only", "image,nonsense")
    assert code == 2 and report is None


def test_report_keys_and_observed_values(world):
    _, report, _ = world.verify("--rehash-weights")
    assert sorted(report) == ["checks", "failed_checks", "generated_at", "inputs", "schema", "selected_checks",
                              "skipped_checks", "verdict"]
    for entry in report["checks"].values():
        assert sorted(entry) == ["claims_checked", "details", "facts", "failures", "verdict"]
    assert report["checks"]["image"]["facts"]["image"]["id"] == IMAGE_ID
    assert report["checks"]["image"]["facts"]["image"]["base"]["local_id"] == BASE_LOCAL_ID
    kv = report["checks"]["kv"]["facts"]["profiles"][PROFILE_ID]["instances"]["saea-omni-i0"]["engine"]
    assert kv == {"kv_tokens": KV_TOKENS, "max_concurrency": KV_CONCURRENCY, "block_size": 128, "prefix_caching": True,
                  "chunked_prefill": True, "cudagraph_mode": "FULL_AND_PIECEWISE", "cudagraph_capture_sizes": [1, 2, 4],
                  "execute_model_timeout_seconds": None}
    assert report["checks"]["env"]["facts"]["hosts"][HOSTNAME]["global_nic"] == NIC
    tails = report["checks"]["template"]["facts"]["template"]["tails"]
    assert tails["thinking_true"].endswith("<|im_start|>assistant\n") and tails["thinking_false"].endswith(CLOSED_THINK)
    assert report["checks"]["template"]["facts"]["template"]["audio_spans"] == {"thinking_true": [1, 1], "thinking_false": [1, 1]}
    assert report["inputs"]["profile_id"] == PROFILE_ID and len(report["inputs"]["base_lock_sha256"]) == 64


def test_print_observed_has_the_documented_lock_shape(world):
    lock = world.lock_dict
    assert lock["schema"] == veb.LOCK_SCHEMA
    assert sorted(lock) == ["cann", "encoders", "execution", "hosts", "image", "network_env", "profiles", "protocol",
                            "python", "receipt", "schema", "template", "versions", "weights"]
    assert lock["image"]["base"] == {"ref": veb.DEFAULT_BASE_IMAGE_REF, "local_id": BASE_LOCAL_ID}
    assert lock["protocol"]["server_args"][:2] == ["--max-model-len", "24576"]
    assert lock["protocol"]["decoding"]["max_tokens"] == 4096 and lock["protocol"]["decoding"]["seed"] == 20260913
    assert lock["protocol"]["protocol_sha256"] == world.receipt["protocol_sha256"]
    assert sorted(lock["weights"]["file_names"]) == sorted(world.model_files)
    assert lock["weights"]["other_entries"] == [".hfd"]
    assert sorted(lock["encoders"]["enc-view"]["file_names"]) == ["config.json", "model.safetensors"]
    assert lock["receipt"]["path"] == RECEIPT_PATH
    assert sorted(lock["profiles"][PROFILE_ID]["instances"]) == [i.name for i in PROFILE.instances]
    assert lock["hosts"][HOSTNAME]["cpu_count"] == 256
    assert lock["network_env"]["ro_mounts"] == sorted(dep.ASCEND_READONLY_MOUNTS)
    assert lock["template"]["tokenizer_config_has_chat_template"] is False
    assert sorted(lock["cann"]["image_files"]) == list(veb.DEFAULT_IMAGE_CANN_FILES)


def test_print_observed_without_receipt_or_rehash_skips_those_parts(world, tmp_path):
    out = tmp_path / "lock2.json"
    code = veb.main(["--profile", str(PROFILE_PATH), "--print-observed", "--out", str(out), "--repo-root", str(REPO)],
                    readers=world.readers)
    lock = json.loads(out.read_text(encoding="utf-8"))
    assert code == 0 and "weights" not in lock and "receipt" not in lock and "protocol_sha256" not in lock["protocol"]


def test_print_observed_exits_nonzero_when_the_deployment_is_broken(world, tmp_path):
    world.readers.cmdlines.clear()
    world.readers.containers[PROFILE.instances[0].name]["HostConfig"]["CpusetCpus"] = "0-3"
    code = veb.main(["--profile", str(PROFILE_PATH), "--print-observed", "--out", str(tmp_path / "x.json"),
                     "--repo-root", str(REPO)], readers=world.readers)
    assert code == 1


# ---------------------------------------------------------------------------------------------------------------------
# One wrong value per check.
# ---------------------------------------------------------------------------------------------------------------------


def _image(w):
    w.readers.images[PROFILE.image]["Id"] = "sha256:" + "ff" * 32


def _versions(w):
    w.readers.versions["torch_npu"] = "2.10.0.post6"


def _cann(w):
    w.readers.host_files["/usr/local/Ascend/driver/version.info"] = b"Version=25.0.rc1\n"


def _weights(w):
    path = f"{PROFILE.model_path}/model-00002-of-00002.safetensors"
    w.readers.files[path] = w.readers.files[path][:-1] + b"X"


def _endpoints(w):
    w.readers.get[f"{PROFILE.base_url(PROFILE.instances[1])}/health"] = (503, "")


def _protocol(w):
    args = w.readers.containers[PROFILE.instances[2].name]["Args"]
    args[args.index("--dtype") + 1] = "float16"


def _deployment(w):
    w.readers.containers[PROFILE.instances[0].name]["HostConfig"]["CpusetCpus"] = "0-3"


def _env(w):
    container = w.readers.containers[PROFILE.instances[3].name]
    container["Config"]["Env"] = [e for e in container["Config"]["Env"] if not e.startswith("HCCL_SOCKET_IFNAME=")] + [
        "HCCL_SOCKET_IFNAME=docker0"]


def _kv(w):
    name = PROFILE.instances[0].name
    w.readers.logs[name] = [line.replace(f"{KV_TOKENS:,}", f"{int(KV_TOKENS * 0.95):,}") for line in w.readers.logs[name]]


def _template(w):
    w.think_in_true_mode = True


def _receipt(w):
    tree = next(iter(w.vw.trees.values()))
    (tree / "__init__.py").write_bytes(b"# tampered\n")


MUTATIONS = {
    "image": _image, "versions": _versions, "cann": _cann, "weights": _weights, "endpoints": _endpoints,
    "protocol": _protocol, "deployment": _deployment, "env": _env, "kv": _kv, "template": _template,
    "receipt": _receipt,
}


def test_there_is_one_mutation_per_check():
    assert sorted(MUTATIONS) == sorted(veb.CHECK_NAMES) and len(veb.CHECK_NAMES) == 11


@pytest.mark.parametrize("check", veb.CHECK_NAMES)
def test_a_wrong_value_fails_its_check_and_the_overall_verdict(world, check):
    MUTATIONS[check](world)
    code, report, _ = world.verify("--rehash-weights")
    assert report["checks"][check]["verdict"] == "fail", report["checks"][check]
    assert report["checks"][check]["failures"]
    assert check in report["failed_checks"] and report["verdict"] == "fail" and code == 1


def test_a_changed_lock_value_fails_with_a_message_naming_the_leaf(world):
    lock = copy.deepcopy(world.lock_dict)
    lock["versions"]["transformers"] = "5.14.0"
    world.set_lock(lock)
    _, report, _ = world.verify("--only", "versions")
    assert any("versions > transformers" in line and "'5.14.0'" in line for line in report["checks"]["versions"]["failures"])


def test_kv_tolerance_is_one_percent(world):
    lock = copy.deepcopy(world.lock_dict)
    engine = lock["profiles"][PROFILE_ID]["instances"]["saea-omni-i0"]["engine"]
    engine["kv_tokens"] = int(KV_TOKENS * 1.005)
    world.set_lock(lock)
    assert world.verify("--only", "kv")[0] == 0
    engine["kv_tokens"] = int(KV_TOKENS * 1.02)
    world.set_lock(lock)
    assert world.verify("--only", "kv")[0] == 1


@pytest.mark.parametrize("knob", ["spans", "count"])
def test_template_engine_count_and_audio_span_mismatches_fail(world, knob):
    if knob == "spans":
        world.audio_spans = 2
    else:
        world.count_skew = 1
    code, report, _ = world.verify("--only", "template")
    assert code == 1 and report["checks"]["template"]["verdict"] == "fail"


def test_template_decodes_byte_level_forms_before_comparing(world):
    _, report, _ = world.verify("--only", "template")
    entry = report["checks"]["template"]["details"]["instances"]["saea-omni-i0"]
    assert "Ċ" in entry["raw_tails"]["thinking_true"] and "\n" not in entry["raw_tails"]["thinking_true"]
    assert entry["tails"]["thinking_true"].endswith("<|im_start|>assistant\n")
    assert entry["tokenize_count"] == entry["engine_prompt_tokens"]


def test_byte_level_decoder_inverts_gpt2_bytes_to_unicode_for_every_byte():
    table = veb.bytes_to_unicode()
    assert len(table) == 256 and len(set(table.values())) == 256
    assert veb.decode_byte_level("".join(table[b] for b in range(0x20, 0x7F))) == "".join(map(chr, range(0x20, 0x7F)))
    assert veb.decode_byte_level("ameĠtheĠintent.<|im_end|>Ċ<|im_start|>assistantĊ") == "ame the intent.<|im_end|>\n<|im_start|>assistant\n"
    assert veb.decode_byte_level("_end|>Ċ<|im_start|>assistantĊ<think>ĊĊ</think>ĊĊ").endswith(CLOSED_THINK)
    assert veb.decode_byte_level("caf" + table[0xC3] + table[0xA9]) == "café"


# ---------------------------------------------------------------------------------------------------------------------
# Fail closed on anything unreadable.
# ---------------------------------------------------------------------------------------------------------------------


def test_unreadable_container_facts_fail_every_check_that_needs_them(world):
    del world.readers.containers[PROFILE.instances[1].name]
    code, report, _ = world.verify("--rehash-weights")
    failed = set(report["failed_checks"])
    assert code == 1 and {"image", "endpoints", "protocol", "deployment", "env", "kv", "receipt"} <= failed
    assert all("input unreadable or invalid: ReadError" in report["checks"][name]["failures"][0] for name in ("image", "env"))
    assert report["checks"]["versions"]["verdict"] == "pass"


@pytest.mark.parametrize("where", ["lock_missing", "lock_garbage", "lock_wrong_schema"])
def test_unreadable_base_lock_fails_every_selected_check(world, where):
    if where == "lock_missing":
        del world.readers.files[LOCK_PATH]
    elif where == "lock_garbage":
        world.readers.files[LOCK_PATH] = b"{not json"
    else:
        world.readers.files[LOCK_PATH] = json.dumps({"schema": "other"}).encode()
    code, report, _ = world.verify()
    assert code == 1 and report["failed_checks"] == sorted(report["selected_checks"])
    assert all("base-lock unreadable or invalid" in entry["failures"][0] for entry in report["checks"].values())
    assert report["verdict"] == "fail"


def test_a_lock_without_a_compared_leaf_fails_closed(world):
    lock = copy.deepcopy(world.lock_dict)
    del lock["image"]["created"]
    world.set_lock(lock)
    _, report, _ = world.verify("--only", "image")
    assert report["checks"]["image"]["verdict"] == "fail"
    assert "base-lock lacks image > created" in report["checks"]["image"]["failures"]


def test_a_lock_without_a_whole_section_fails_closed(world):
    lock = copy.deepcopy(world.lock_dict)
    del lock["versions"]
    world.set_lock(lock)
    _, report, _ = world.verify("--only", "versions")
    assert report["checks"]["versions"]["verdict"] == "fail" and "LockMissing" in report["checks"]["versions"]["failures"][0]


def test_unreadable_profile_fails_everything(world):
    del world.readers.files[str(PROFILE_PATH)]
    code, report, _ = world.verify()
    assert code == 1 and all("deployment profile unreadable" in e["failures"][0] for e in report["checks"].values())


def test_weights_named_in_only_without_rehash_fails_closed(world):
    code, report, _ = world.verify("--only", "weights")
    assert code == 1 and "--rehash-weights" in report["checks"]["weights"]["failures"][0]


def test_unreadable_weight_file_listing_fails_weights(world):
    for path in [p for p in world.readers.files if p.startswith(PROFILE.model_path + "/")]:
        del world.readers.files[path]
    _, report, _ = world.verify("--only", "weights", "--rehash-weights")
    assert report["checks"]["weights"]["verdict"] == "fail"


def test_extra_and_missing_weight_files_fail_the_exact_set(world):
    world.readers.files[f"{PROFILE.model_path}/added-tokenizer.json"] = b"{}"
    _, report, _ = world.verify("--only", "weights", "--rehash-weights")
    assert any("file_names" in line for line in report["checks"]["weights"]["failures"])


def test_unreadable_receipt_fails_protocol_and_receipt(world):
    del world.readers.files[RECEIPT_PATH]
    _, report, _ = world.verify()
    assert {"protocol", "receipt"} <= set(report["failed_checks"])


def test_unreadable_http_endpoint_fails_closed(world):
    del world.readers.get[f"{PROFILE.base_url(PROFILE.instances[0])}/v1/models"]
    _, report, _ = world.verify("--only", "endpoints")
    assert report["checks"]["endpoints"]["verdict"] == "fail"


def test_unreadable_cann_file_fails_cann(world):
    del world.readers.host_files["/etc/ascend_install.info"]
    _, report, _ = world.verify("--only", "cann")
    assert report["checks"]["cann"]["verdict"] == "fail"
    assert any("unreadable CANN file /etc/ascend_install.info" in line for line in report["checks"]["cann"]["failures"])


def test_execution_environment_refusal_fails_versions(world):
    world.readers.execution = RuntimeError("not the saea-base container")
    _, report, _ = world.verify("--only", "versions")
    assert report["checks"]["versions"]["verdict"] == "fail"


def test_receipt_cmdline_in_the_wrong_shape_fails_receipt(world):
    name = PROFILE.instances[0].name
    world.readers.cmdlines[name] = ["/usr/local/python3.12.13/bin/python3", "/usr/local/python3.12.13/bin/vllm", "serve",
                                    PROFILE.model_path]
    _, report, _ = world.verify("--only", "receipt")
    assert report["checks"]["receipt"]["verdict"] == "fail"
    assert any(name in line for line in report["checks"]["receipt"]["failures"])


def test_live_vllm_args_accept_only_the_gate_entrypoint_form():
    inspect = {"Path": dep.CONTAINER_INTERPRETER, "Args": ["-m", "vllm.entrypoints.cli.main", "serve", "/m", "--port", "1"]}
    assert veb.live_vllm_args(inspect) == ("/m", ["--port", "1"])
    for bad in (
        {"Path": "vllm", "Args": ["serve", "/m", "--port", "1"]},  # the retired --entrypoint vllm form
        {"Path": dep.CONTAINER_INTERPRETER, "Args": ["serve", "/m", "--port", "1"]},
        {"Path": "/usr/bin/python3", "Args": inspect["Args"]},
        {"Path": dep.CONTAINER_INTERPRETER, "Args": ["-m", "vllm.entrypoints.cli.main", "serve"]},
        {"Path": dep.CONTAINER_INTERPRETER},
    ):
        with pytest.raises(veb.ReadError):
            veb.live_vllm_args(bad)


def test_receipt_missing_device_group_variable_fails_receipt(world):
    container = world.readers.containers[PROFILE.instances[0].name]
    container["Config"]["Env"] = [e for e in container["Config"]["Env"] if not e.startswith("ASCEND_RT_VISIBLE_DEVICES=")]
    _, report, _ = world.verify("--only", "receipt")
    assert report["checks"]["receipt"]["verdict"] == "fail"


# ---------------------------------------------------------------------------------------------------------------------
# Determinism, and the no-container-control rail.
# ---------------------------------------------------------------------------------------------------------------------


def test_two_runs_are_byte_identical_apart_from_generated_at(world):
    _, first, first_path = world.verify("--rehash-weights", clock=datetime(2026, 10, 29, 12, 0, 0, tzinfo=timezone.utc))
    _, second, second_path = world.verify("--rehash-weights", clock=datetime(2026, 10, 30, 7, 5, 9, tzinfo=timezone.utc))
    assert first["generated_at"] != second["generated_at"]
    strip = lambda path: re.sub(r'"generated_at": "[^"]*"', '"generated_at": "-"', path.read_text(encoding="utf-8"))
    assert strip(first_path) == strip(second_path)
    text = first_path.read_text(encoding="utf-8")
    assert json.dumps(json.loads(text), indent=2, sort_keys=True) + "\n" == text
    assert len(re.findall(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", re.sub(r'"created": "[^"]*"', "", text))) == 1


def test_a_failing_run_is_deterministic_too(world):
    _kv(world)
    runs = [world.verify(clock=datetime(2026, 10, 29, 1, 1, 1, tzinfo=timezone.utc))[2].read_text() for _ in range(2)]
    assert runs[0] == runs[1]


def test_the_reader_interface_cannot_control_containers():
    assert set(veb.READER_METHODS) >= {"docker_image_inspect", "docker_container_inspect", "container_log_lines",
                                       "container_cmdline", "http_get", "http_post_json"}
    for method in veb.READER_METHODS:
        assert not set(method.split("_")) & set(veb.FORBIDDEN_READER_WORDS), method
    assert not [m for m in dir(veb.LiveReaders) if not m.startswith("_") and m not in veb.READER_METHODS]


def test_a_full_run_only_calls_the_read_interface(world):
    world.readers.calls.clear()
    world.verify("--rehash-weights")
    called = set(world.readers.calls)
    assert called and called <= set(veb.READER_METHODS)
    assert "http_post_json" in called and "container_cmdline" in called


def test_script_source_has_no_process_or_container_control():
    source = SCRIPT.read_text(encoding="utf-8")
    for needle in ("subprocess", "os.system", "Popen", "docker.sock", "os.exec", "os.kill", "signal."):
        assert needle not in source, needle
    assert not re.search(r"\bdocker\s+(run|start|stop|rm|kill|restart|exec|create|pause)\b", source)


@pytest.mark.skipif(not WRAPPER.exists(), reason="host wrapper is outside the study tree")
def test_wrapper_runs_one_deviceless_socketless_container_and_only_reads_docker():
    lines = [line for line in WRAPPER.read_text(encoding="utf-8").splitlines() if not line.lstrip().startswith("#")]
    text = "\n".join(lines)
    verbs = re.findall(r"\bdocker\s+((?:image\s+)?\w+)", text)
    assert set(verbs) <= {"image inspect", "inspect", "logs", "run"}, verbs
    assert verbs.count("run") == 1
    assert "--device" not in text and "docker.sock" not in text and "--privileged" not in text
    assert "-I" not in text.split("verify_engineering_base.py")[1]
    assert re.search(r'-v "\$OMNI_ROOT:\$OMNI_ROOT:ro"', text)


# ---------------------------------------------------------------------------------------------------------------------
# Parsers against real evidence.
# ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.skipif(not LIVE_LOG.exists(), reason="2026-10-29 serve log not on this machine")
def test_engine_log_parser_reads_the_real_2026_10_29_startup_log():
    lines = LIVE_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
    parsed = veb.parse_engine_log([l for l in lines if re.search(r"KV cache|V1 LLM engine|Block size", l)])
    assert (parsed["kv_tokens"], parsed["max_concurrency"], parsed["window_tokens"]) == (564224, 22.96, 24576)
    assert (parsed["block_size"], parsed["prefix_caching"], parsed["chunked_prefill"]) == (128, True, True)
    assert parsed["cudagraph_mode"] == "FULL_AND_PIECEWISE" and parsed["cudagraph_capture_sizes"] == [1, 2, 4]
    assert parsed["available_kv_gib"] == 25.83


def test_select_global_nic_skips_bridges_and_virtual_pairs():
    assert veb.select_global_nic(["docker0", "br-d3b20b1f3e3f", "veth12ab", "enp61s0f2"]) == "enp61s0f2"
    assert veb.select_global_nic(["docker0", "lo"]) is None


def test_cache_config_info_labels_are_parsed():
    labels = veb.parse_cache_config_info('x 1\nvllm:cache_config_info{block_size="128",num_gpu_blocks="4408"} 1.0\n')
    assert labels == {"block_size": "128", "num_gpu_blocks": "4408"}


def test_the_versions_check_reads_the_serving_interpreter_not_the_verifiers_own(world):
    assert veb.SERVING_INTERPRETER == "/usr/local/python3.12.13/bin/python3"
    assert veb.SERVING_INTERPRETER in veb.DEFAULT_INTERPRETERS
    _, report, _ = world.verify("--only", "versions")
    assert report["checks"]["versions"]["verdict"] == "pass"
    assert world.readers.version_interpreters == {veb.SERVING_INTERPRETER}


def test_live_readers_read_package_metadata_of_another_interpreters_site_packages(tmp_path):
    prefix = tmp_path / "py"
    (prefix / "bin").mkdir(parents=True)
    interpreter = prefix / "bin" / "python3"
    interpreter.write_bytes(b"ELF-stub")
    site = prefix / "lib" / "python3.12" / "site-packages"
    for dist, version in (("vllm_ascend", "0.30.0rc2"), ("torch_npu", "2.10.0.post7")):
        info = site / f"{dist}-{version}.dist-info"
        info.mkdir(parents=True)
        (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {dist.replace('_', '-')}\nVersion: {version}\n")
    readers = veb.LiveReaders(None)
    assert readers.dist_version("vllm-ascend", str(interpreter)) == "0.30.0rc2"
    assert readers.dist_version("torch_npu", str(interpreter)) == "2.10.0.post7"
    assert readers.dist_version("vllm", str(interpreter)) is None  # absent there, whatever the verifier's own has
    with pytest.raises(veb.ReadError):
        readers.dist_version("vllm", str(tmp_path / "missing" / "bin" / "python3"))
    from importlib import metadata

    assert readers.dist_version("pytest") == metadata.version("pytest")  # the verifier's own interpreter, in process

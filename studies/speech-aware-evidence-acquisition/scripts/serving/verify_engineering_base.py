#!/usr/bin/env python3
"""Acceptance test of the locked engineering base (npu-vllm-port, plan 2026-10-29 section 1.2).

    python scripts/serving/verify_engineering_base.py --profile configs/deployment/<id>.json \
        --base-lock <base-lock-v1.json> --out report.json [--only image,kv,...] [--rehash-weights]
    python scripts/serving/verify_engineering_base.py --profile ... --print-observed [--receipt <runtime receipt>] \
        [--encoder NAME=LOCAL_SUBDIR ...] [--rehash-weights] [--out base-lock.json]

It ATTACHES to the resident deployment. It never starts, stops, restarts or execs into a container, never signals a
process, and writes nothing but its own ``--out`` file. Every host, docker, HTTP and file read goes through one small
injectable :class:`Readers` object, so the tests feed recorded fixtures; the production :class:`LiveReaders` reads the
local file system, loopback HTTP, ``importlib.metadata`` and a directory of host facts that
``work/scripts/npu_verify_base.sh`` collected with read-only docker commands (the verifier itself needs no docker
socket). The only request that reaches a model is ONE synthetic one-token chat per instance in the ``template`` check
(plan check 10), built from a generated tone; no dataset row is read.

Output (schema ``saea-engineering-base-verification-v1``, ``sort_keys``): ``generated_at`` is the ONLY timestamp, and
nothing in a check carries a clock value or a ``/metrics`` counter, so two runs on an untouched deployment are
byte-identical apart from ``generated_at``. Exit 0 only if every SELECTED check passes (1 otherwise, 2 for a usage or
output error). ANY unreadable or invalid input fails its check closed; the report still records what was observed.

Checks (plan 1.2 numbering): 1 image, 2 versions, 3 cann, 4 weights (only with ``--rehash-weights``; otherwise it is
listed under ``skipped_checks`` and, if named in ``--only``, it FAILS), 5 endpoints, 6 protocol, 7 deployment, 8 env,
9 kv, 10 template, 11 receipt.

Base-lock record (``saea-engineering-base-lock-v1``) -- the minimal machine-readable format this tool compares against.
Every leaf is compared for equality (``kv_tokens`` and ``max_concurrency`` within 1 %); ``--print-observed`` writes the
same shape from a live run::

    schema                                   "saea-engineering-base-lock-v1"
    image.ref / id / repo_digests[] / created                      L1  (docker image inspect of the profile image)
    image.base.ref / local_id                                      L2  (12-hex local id of the base image)
    python.version                                                 L3  (running interpreter)
    python.interpreters.<path>.realpath / sha256                   L3  (exe sha256 of each listed interpreter)
    versions.<dist>                                                L4  (importlib.metadata version; null = absent)
    cann.image_files.<path>.sha256 / bytes                         L5  (files inside the verifier image)
    hosts.<hostname>.cann_files.<host path>.sha256 / bytes         L5  (host files, read through --host-root)
    hosts.<hostname>.card_count / cpu_count / global_nic           L6, L12
    weights.local_subdir / file_names[] / other_entries[] / payload_size_bytes /
        files.<relative name>.size_bytes / sha256                  L7  (+ free provenance: lock_key, hf_id, revision)
    encoders.<name>.<same keys as weights>                         L8
    protocol.served_model_name / server_args[] / protocol_sha256   L9
    protocol.decoding.{temperature,seed,max_tokens,reasoning_budget_tokens,answer_reserve_tokens,
        forced_close_message_sha256,reasoning_context_policy,timeout_seconds,enable_thinking}   L10
    profiles.<profile_id>.profile_sha256                           L11
    profiles.<profile_id>.instances.<name>.{cards,cpuset,port,tensor_parallel_size,max_num_seqs,
        gpu_memory_utilization,max_num_batched_tokens,server_argv_sha256}                       L11
    profiles.<profile_id>.instances.<name>.engine.{kv_tokens,max_concurrency,block_size,prefix_caching,
        chunked_prefill,cudagraph_mode,cudagraph_capture_sizes,execute_model_timeout_seconds}   L13
    network_env.network_mode / ro_mounts[]                         L12
    template.chat_template_sha256 / tokenizer_config_has_chat_template /
        tails.{thinking_true,thinking_false} / audio_spans.{thinking_true,thinking_false}       L14
    execution.environment                                          L15
    receipt.path                                                   absolute path of the runtime receipt (input)

Standard library plus the study's own contract and deployment modules; run it inside ``saea-base``.
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
for _entry in (str(_REPO / "src"), str(_HERE)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

import synthetic_sweep as sweep  # noqa: E402  (synthetic_wav; shares the sweep's synthetic content rules)
from speech_aware_evidence_acquisition.contracts.gate_runtime_vllm import (  # noqa: E402
    DEVICE_ENV_KEY,
    VLLM_ENTRYPOINT,
    assert_argv_matches_instance,
    classify_server_flags,
    deployment_environment,
    device_env_value,
    instance_deployment_env,
    instance_for_endpoint,
    pinned_environment,
    protocol_sha256,
    validate_vllm_receipt_document,
    verify_vllm_runtime_files,
)
from speech_aware_evidence_acquisition.core import deployment as dep  # noqa: E402

SCHEMA = "saea-engineering-base-verification-v1"
LOCK_SCHEMA = "saea-engineering-base-lock-v1"
CHECK_NAMES = (
    "image", "versions", "cann", "weights", "endpoints", "protocol", "deployment", "env", "kv", "template", "receipt",
)
DEFAULT_BASE_IMAGE_REF = "quay.io/ascend/vllm-ascend:v0.30.0rc2"
DEFAULT_INTERPRETERS = ("/opt/saea-venv/bin/python", "/usr/local/python3.12.13/bin/python3")
DEFAULT_DISTS = ("vllm", "vllm-ascend", "torch", "torch_npu", "transformers", "triton")
REQUIRED_DISTS = ("vllm", "vllm-ascend", "torch", "torch_npu", "transformers")
DEFAULT_HOST_CANN_FILES = (
    "/usr/local/Ascend/driver/version.info",
    "/etc/ascend_install.info",
    "/usr/local/Ascend/ascend-toolkit/8.2.RC1/aarch64-linux/ascend_toolkit_install.info",
)
DEFAULT_IMAGE_CANN_FILES = ("/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/ascend_toolkit_install.info",)
INTERFACE_VARS = ("HCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME", "TP_SOCKET_IFNAME")
#: Interfaces that are never "the NIC holding the node's global IPv4": container bridges, virtual pairs, loopback.
_VIRTUAL_NIC = re.compile(r"lo|docker\d*|br-[0-9a-f]+|veth\S*|virbr\S*|cni\S*|flannel\S*|cali\S*")
KV_REL_TOL = 0.01
TAIL_CHARS = 48
SMOKE_SYSTEM = "You label spoken requests."
SMOKE_USER = "Name the intent."
TAIL_TRUE_SUFFIX = "<|im_start|>assistant\n"
TAIL_FALSE_SUFFIX = "<think>\n\n</think>\n\n"
#: Method names no reader may ever have (the "never starts or stops a container" rail, asserted by the tests).
FORBIDDEN_READER_WORDS = ("start", "stop", "run", "kill", "restart", "rm", "exec", "create", "pause", "remove")


class ReadError(RuntimeError):
    """An input could not be read or is not what it must be; the check that needed it fails closed."""


class LockMissing(KeyError):
    """The base-lock record does not carry a leaf this check compares."""


# ---------------------------------------------------------------------------
# The injectable reader interface.
# ---------------------------------------------------------------------------


class Readers:
    """Every external read the verifier performs. There is deliberately no method that changes anything.

    ``docker_*`` and ``container_*`` return what read-only docker commands printed on the host (``docker image inspect``,
    ``docker inspect``, filtered ``docker logs``, ``/proc/<pid>/cmdline``). The base class implements only the
    derived helpers; subclasses implement the rest.
    """

    def read_bytes(self, path: str) -> bytes:
        raise NotImplementedError

    def read_host_bytes(self, path: str) -> bytes:
        raise NotImplementedError

    def realpath(self, path: str) -> str:
        raise NotImplementedError

    def list_files(self, root: str) -> tuple[list[str], list[str]]:
        """``(relative file paths, top-level dot-directories)`` below ``root``; a symlink or odd entry is a ReadError."""
        raise NotImplementedError

    def sha256_files(self, paths: Sequence[str]) -> dict[str, tuple[int, str]]:
        raise NotImplementedError

    def host_facts(self) -> dict[str, Any]:
        raise NotImplementedError

    def docker_image_inspect(self, ref: str) -> dict[str, Any]:
        raise NotImplementedError

    def docker_container_inspect(self, name: str) -> dict[str, Any]:
        raise NotImplementedError

    def container_log_lines(self, name: str) -> list[str]:
        raise NotImplementedError

    def container_cmdline(self, name: str) -> list[str]:
        raise NotImplementedError

    def dist_version(self, name: str) -> str | None:
        raise NotImplementedError

    def python_version(self) -> str:
        raise NotImplementedError

    def assert_execution_environment(self) -> str:
        raise NotImplementedError

    def verify_runtime_files(self, receipt: Mapping[str, Any]) -> None:
        raise NotImplementedError

    def http_get(self, url: str) -> tuple[int, str]:
        raise NotImplementedError

    def http_post_json(self, url: str, body: Mapping[str, Any]) -> tuple[int, Any]:
        raise NotImplementedError

    def read_text(self, path: str) -> str:
        try:
            return self.read_bytes(path).decode("utf-8")
        except UnicodeError as error:
            raise ReadError(f"{path} is not UTF-8: {error}") from error


READER_METHODS = tuple(
    sorted(name for name in dir(Readers) if not name.startswith("_") and callable(getattr(Readers, name)))
)


def _sha16(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class LiveReaders(Readers):
    """Production readers. Local files, loopback HTTP, ``importlib.metadata`` and a host-facts directory.

    The facts directory (written by ``npu_verify_base.sh``): ``host.json`` (hostname, cpu_count, card_count,
    global_nics), ``image-<sha256(ref)[:16]>.json``, ``container-<name>.json`` (``docker inspect``),
    ``logs-<name>.txt`` (the startup lines of ``docker logs``) and ``cmdline-<name>.bin`` (raw ``/proc/<pid>/cmdline``).
    """

    def __init__(self, facts_dir: str | None, host_root: str | None = None, *, http_timeout: float = 120.0,
                 workers: int = 16) -> None:
        self.facts_dir, self.host_root = facts_dir, host_root or ""
        self.http_timeout, self.workers = http_timeout, workers

    # -- files -------------------------------------------------------------
    @staticmethod
    def _read(path: str) -> bytes:
        try:
            return Path(path).read_bytes()
        except OSError as error:
            raise ReadError(f"cannot read {path}: {error}") from error

    def read_bytes(self, path: str) -> bytes:
        return self._read(path)

    def read_host_bytes(self, path: str) -> bytes:
        return self._read(self.host_root.rstrip("/") + path)

    def realpath(self, path: str) -> str:
        if not os.path.exists(path):
            raise ReadError(f"{path} does not exist")
        return os.path.realpath(path)

    def list_files(self, root: str) -> tuple[list[str], list[str]]:
        files: list[str] = []
        dots: list[str] = []

        def refuse(error: OSError) -> None:
            raise ReadError(f"cannot enumerate {root}: {error}")

        top = Path(root)
        if top.is_symlink() or not top.is_dir():
            raise ReadError(f"{root} is not a real directory")
        for directory, subdirs, names in os.walk(root, onerror=refuse, followlinks=False):
            here = Path(directory)
            if here == top:
                dots = sorted(name for name in subdirs if name.startswith("."))
                subdirs[:] = sorted(name for name in subdirs if not name.startswith("."))
            else:
                subdirs.sort()
            for name in sorted(subdirs):
                if (here / name).is_symlink():
                    raise ReadError(f"{here / name} is a symlink")
            for name in sorted(names):
                entry = here / name
                if entry.is_symlink() or not entry.is_file():
                    raise ReadError(f"{entry} is not a regular file")
                files.append(entry.relative_to(top).as_posix())
        return sorted(files), dots

    def sha256_files(self, paths: Sequence[str]) -> dict[str, tuple[int, str]]:
        def one(path: str) -> tuple[str, tuple[int, str]]:
            digest, size = hashlib.sha256(), 0
            try:
                with open(path, "rb") as handle:
                    for block in iter(lambda: handle.read(8 << 20), b""):
                        digest.update(block)
                        size += len(block)
            except OSError as error:
                raise ReadError(f"cannot hash {path}: {error}") from error
            return path, (size, digest.hexdigest())

        with ThreadPoolExecutor(max_workers=max(1, min(self.workers, len(paths) or 1))) as pool:
            return dict(pool.map(one, list(paths)))

    # -- host facts (docker-socket free) --------------------------------------
    def _fact(self, name: str) -> bytes:
        if not self.facts_dir:
            raise ReadError("no --facts-dir: host and docker facts were not collected")
        return self._read(str(Path(self.facts_dir) / name))

    def _fact_json(self, name: str) -> Any:
        try:
            return json.loads(self._fact(name).decode("utf-8"))
        except (UnicodeError, ValueError) as error:
            raise ReadError(f"host fact {name} is not JSON: {error}") from error

    def host_facts(self) -> dict[str, Any]:
        facts = self._fact_json("host.json")
        if not isinstance(facts, dict):
            raise ReadError("host.json must be an object")
        return facts

    @staticmethod
    def _first(document: Any, what: str) -> dict[str, Any]:
        item = document[0] if isinstance(document, list) and document else document
        if not isinstance(item, dict):
            raise ReadError(f"{what} is not a docker inspect document")
        return item

    def docker_image_inspect(self, ref: str) -> dict[str, Any]:
        return self._first(self._fact_json(f"image-{_sha16(ref)}.json"), f"image {ref}")

    def docker_container_inspect(self, name: str) -> dict[str, Any]:
        return self._first(self._fact_json(f"container-{name}.json"), f"container {name}")

    def container_log_lines(self, name: str) -> list[str]:
        return self._fact(f"logs-{name}.txt").decode("utf-8", "replace").splitlines()

    def container_cmdline(self, name: str) -> list[str]:
        raw = self._fact(f"cmdline-{name}.bin")
        return [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]

    # -- python environment ----------------------------------------------------
    def dist_version(self, name: str) -> str | None:
        from importlib import metadata

        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None

    def python_version(self) -> str:
        import platform

        return platform.python_version()

    def assert_execution_environment(self) -> str:
        from speech_aware_evidence_acquisition.contracts.plan import assert_execution_environment

        return assert_execution_environment()

    def verify_runtime_files(self, receipt: Mapping[str, Any]) -> None:
        verify_vllm_runtime_files(receipt)

    # -- loopback HTTP -----------------------------------------------------------
    def _open(self, request: Any) -> tuple[int, str]:
        import urllib.error
        import urllib.request

        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(request, timeout=self.http_timeout) as response:
                return response.status, response.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as error:
            return error.code, error.read().decode("utf-8", "replace")
        except (urllib.error.URLError, OSError, ValueError) as error:
            raise ReadError(f"cannot reach the endpoint: {error}") from error

    def http_get(self, url: str) -> tuple[int, str]:
        import urllib.request

        return self._open(urllib.request.Request(url, method="GET"))

    def http_post_json(self, url: str, body: Mapping[str, Any]) -> tuple[int, Any]:
        import urllib.request

        request = urllib.request.Request(
            url, data=json.dumps(body).encode("utf-8"), method="POST", headers={"Content-Type": "application/json"}
        )
        status, text = self._open(request)
        try:
            return status, json.loads(text)
        except ValueError:
            return status, {"raw": text[:400]}


# ---------------------------------------------------------------------------
# Claims: observed facts in base-lock shape.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Claim:
    path: tuple[str, ...]
    value: Any
    rel_tol: float | None = None


@dataclass
class Result:
    details: dict[str, Any] = field(default_factory=dict)
    claims: list[Claim] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    def claim(self, *path: str, value: Any, rel_tol: float | None = None) -> None:
        self.claims.append(Claim(tuple(path), value, rel_tol))

    def require(self, ok: bool, message: str) -> bool:
        if not ok:
            self.failures.append(message)
        return bool(ok)


class ClaimConflict(ValueError):
    pass


def build_nested(claims: Sequence[Claim]) -> dict[str, Any]:
    """The base-lock-shaped document of a list of claims; two claims may agree but never disagree."""

    root: dict[str, Any] = {}
    for claim in claims:
        node = root
        for key in claim.path[:-1]:
            node = node.setdefault(key, {})
            if not isinstance(node, dict):
                raise ClaimConflict(f"{' > '.join(claim.path)} sits below a leaf")
        leaf = claim.path[-1]
        if leaf in node and node[leaf] != claim.value:
            raise ClaimConflict(f"conflicting observations for {' > '.join(claim.path)}: {node[leaf]!r} vs {claim.value!r}")
        node[leaf] = copy.deepcopy(claim.value)
    return root


def lock_lookup(lock: Mapping[str, Any], path: Sequence[str]) -> Any:
    node: Any = lock
    for key in path:
        if not isinstance(node, Mapping) or key not in node:
            raise LockMissing(" > ".join(path))
        node = node[key]
    return node


def _short(value: Any, limit: int = 160) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def values_equal(expected: Any, observed: Any, rel_tol: float | None) -> bool:
    numbers = (int, float)
    if (
        rel_tol is not None
        and isinstance(expected, numbers) and not isinstance(expected, bool)
        and isinstance(observed, numbers) and not isinstance(observed, bool)
    ):
        return abs(observed - expected) <= rel_tol * abs(expected)
    return expected == observed


# ---------------------------------------------------------------------------
# Context shared by the checks.
# ---------------------------------------------------------------------------


@dataclass
class Context:
    readers: Readers
    profile: dep.DeploymentProfile
    repo_root: Path
    lock: Mapping[str, Any] | None
    rehash: bool = False
    receipt_arg: str | None = None
    encoders_arg: Mapping[str, str] = field(default_factory=dict)
    _cache: dict[str, Any] = field(default_factory=dict)

    def cached(self, key: str, make: Callable[[], Any]) -> Any:
        if key not in self._cache:
            self._cache[key] = make()
        return self._cache[key]

    @property
    def printing(self) -> bool:
        return self.lock is None

    def lock_keys(self, path: Sequence[str], default: Sequence[str]) -> list[str]:
        """Names a check should observe: the lock's own keys when verifying, the defaults when printing."""
        if self.lock is None:
            return list(default)
        node = lock_lookup(self.lock, path)
        if not isinstance(node, Mapping):
            raise ReadError(f"base-lock {' > '.join(path)} must be an object")
        return sorted(node)

    def lock_value(self, path: Sequence[str], default: Any) -> Any:
        if self.lock is None:
            return default
        return lock_lookup(self.lock, path)

    @property
    def fragment(self) -> Mapping[str, Any]:
        def load() -> Mapping[str, Any]:
            path = self.repo_root / "configs" / "model" / f"{self.profile.model_fragment}.json"
            try:
                values = json.loads(self.readers.read_text(str(path)))
            except ValueError as error:
                raise ReadError(f"model fragment {path} is not JSON: {error}") from error
            if not isinstance(values, dict) or values.get("transport") != "vllm":
                raise ReadError(f"model fragment {path} is not a vllm fragment")
            if not isinstance(values.get("served_model_name"), str) or not values["served_model_name"]:
                raise ReadError(f"model fragment {path} declares no served_model_name")
            return values

        return self.cached("fragment", load)

    def inspect(self, name: str) -> dict[str, Any]:
        return self.cached(f"inspect:{name}", lambda: self.readers.docker_container_inspect(name))

    def host(self) -> dict[str, Any]:
        facts = self.cached("host", self.readers.host_facts)
        if not isinstance(facts.get("hostname"), str) or not facts["hostname"]:
            raise ReadError("host facts lack a hostname")
        return facts

    def receipt_path(self) -> str | None:
        if self.receipt_arg:
            return self.receipt_arg
        if self.lock is None:
            return None
        return str(lock_lookup(self.lock, ("receipt", "path")))

    @property
    def receipt(self) -> Mapping[str, Any]:
        def load() -> Mapping[str, Any]:
            path = self.receipt_path()
            if not path:
                raise ReadError("no runtime receipt path (base-lock receipt.path or --receipt)")
            try:
                document = json.loads(self.readers.read_text(path))
            except ValueError as error:
                raise ReadError(f"runtime receipt {path} is not JSON: {error}") from error
            if not isinstance(document, dict):
                raise ReadError(f"runtime receipt {path} must be a JSON object")
            return document

        return self.cached("receipt", load)

    def expected_argv(self, instance: dep.InstanceSpec) -> list[str]:
        return dep.instance_server_argv(self.profile, instance, self.fragment)


def container_env(inspect: Mapping[str, Any]) -> dict[str, str]:
    env: dict[str, str] = {}
    for item in (inspect.get("Config") or {}).get("Env") or []:
        key, separator, value = str(item).partition("=")
        if not separator:
            raise ReadError(f"container environment entry {item!r} has no '='")
        env[key] = value
    return env


def live_vllm_args(inspect: Mapping[str, Any]) -> tuple[str, list[str]]:
    """``(model dir, flag tokens)`` of a container started with ``--entrypoint <python> <image> -m
    vllm.entrypoints.cli.main serve <model> <flags>`` (``dep.CONTAINER_INTERPRETER``, the gate's ``VLLM_ENTRYPOINT``)."""

    path, args = inspect.get("Path"), inspect.get("Args")
    head = len(VLLM_ENTRYPOINT)
    if not isinstance(path, str) or not isinstance(args, list) or len(args) < head + 1:
        raise ReadError("container inspect lacks Path/Args")
    if path != dep.CONTAINER_INTERPRETER or tuple(str(token) for token in args[:head]) != tuple(VLLM_ENTRYPOINT):
        raise ReadError(
            f"container does not run '{dep.CONTAINER_INTERPRETER} {' '.join(VLLM_ENTRYPOINT)}' "
            f"(Path {path!r}, Args[:{head}] {args[:head]!r})"
        )
    return str(args[head]), [str(token) for token in args[head + 1:]]


def instance_base_url(ctx: Context, instance: dep.InstanceSpec) -> str:
    return ctx.profile.base_url(instance)


# ---------------------------------------------------------------------------
# Byte-level BPE surface forms (GPT-2 bytes_to_unicode and its inverse).
# ---------------------------------------------------------------------------


def bytes_to_unicode() -> dict[int, str]:
    kept = list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    chars, extra = list(kept), 0
    for byte in range(256):
        if byte not in kept:
            kept.append(byte)
            chars.append(256 + extra)
            extra += 1
    return {byte: chr(char) for byte, char in zip(kept, chars)}


_UNICODE_TO_BYTE = {char: byte for byte, char in bytes_to_unicode().items()}


def decode_byte_level(text: str) -> str:
    """Token surface forms back to text: space is U+0120, newline U+010A; a character outside the map stands for itself."""

    raw = bytearray()
    for char in text:
        if char in _UNICODE_TO_BYTE:
            raw.append(_UNICODE_TO_BYTE[char])
        else:
            raw.extend(char.encode("utf-8"))
    return raw.decode("utf-8", "replace")


# ---------------------------------------------------------------------------
# Check 1: image.
# ---------------------------------------------------------------------------


def check_image(ctx: Context) -> Result:
    r, readers = Result(), ctx.readers
    info = readers.docker_image_inspect(ctx.profile.image)
    image_id, created = info.get("Id"), info.get("Created")
    if not isinstance(image_id, str) or not isinstance(created, str):
        raise ReadError("docker image inspect lacks Id/Created")
    r.claim("image", "ref", value=ctx.profile.image)
    r.claim("image", "id", value=image_id)
    r.claim("image", "repo_digests", value=sorted(info.get("RepoDigests") or []))
    r.claim("image", "created", value=created)
    base_ref = ctx.lock_value(("image", "base", "ref"), DEFAULT_BASE_IMAGE_REF)
    base_id = readers.docker_image_inspect(str(base_ref)).get("Id")
    if not isinstance(base_id, str):
        raise ReadError(f"docker image inspect of {base_ref} lacks Id")
    r.claim("image", "base", "ref", value=base_ref)
    r.claim("image", "base", "local_id", value=base_id.removeprefix("sha256:")[:12])
    containers: dict[str, Any] = {}
    for instance in ctx.profile.instances:
        running = ctx.inspect(instance.name).get("Image")
        containers[instance.name] = {"image_id": running}
        r.require(running == image_id, f"container {instance.name} runs image {running!r}, not {image_id!r}")
    r.details["containers"] = containers
    return r


# ---------------------------------------------------------------------------
# Check 2: versions (interpreters, serving stack, execution authority).
# ---------------------------------------------------------------------------


def check_versions(ctx: Context) -> Result:
    r, readers = Result(), ctx.readers
    r.claim("python", "version", value=readers.python_version())
    for path in ctx.lock_keys(("python", "interpreters"), DEFAULT_INTERPRETERS):
        real = readers.realpath(path)
        size, sha = readers.sha256_files([real])[real]
        r.claim("python", "interpreters", path, "realpath", value=real)
        r.claim("python", "interpreters", path, "sha256", value=sha)
    dists = ctx.lock_keys(("versions",), DEFAULT_DISTS)
    r.require(set(REQUIRED_DISTS) <= set(dists), f"base-lock does not pin the required packages {sorted(set(REQUIRED_DISTS) - set(dists))}")
    for dist in dists:
        version = readers.dist_version(dist)
        r.claim("versions", dist, value=version)
        r.require(version is not None or dist not in REQUIRED_DISTS, f"required package {dist} is not installed")
    r.claim("execution", "environment", value=readers.assert_execution_environment())
    return r


# ---------------------------------------------------------------------------
# Check 3: cann (driver and toolkit files).
# ---------------------------------------------------------------------------


def _version_lines(raw: bytes) -> list[str]:
    lines = raw.decode("utf-8", "replace").splitlines()
    return [line for line in lines if line.lower().startswith(("version=", "package_name=", "innerversion="))][:4]


def check_cann(ctx: Context) -> Result:
    r, readers = Result(), ctx.readers
    hostname = ctx.host()["hostname"]
    r.require(hostname == ctx.profile.host, f"this host is {hostname!r} but the profile is for {ctx.profile.host!r}")
    versions: dict[str, Any] = {}
    skipped: list[str] = []
    plans = (
        (("hosts", hostname, "cann_files"), DEFAULT_HOST_CANN_FILES, readers.read_host_bytes),
        (("cann", "image_files"), DEFAULT_IMAGE_CANN_FILES, readers.read_bytes),
    )
    for prefix, defaults, read in plans:
        for path in ctx.lock_keys(prefix, defaults):
            try:
                raw = read(path)
            except ReadError as error:
                if ctx.printing:
                    skipped.append(path)
                    continue
                r.failures.append(f"unreadable CANN file {path}: {error}")
                continue
            r.claim(*prefix, path, "sha256", value=hashlib.sha256(raw).hexdigest())
            r.claim(*prefix, path, "bytes", value=len(raw))
            versions[path] = _version_lines(raw)
    r.details["version_lines"] = versions
    if skipped:
        r.details["skipped_unreadable_defaults"] = sorted(skipped)
    return r


# ---------------------------------------------------------------------------
# Check 4: weights (expensive; --rehash-weights).
# ---------------------------------------------------------------------------


def check_weights(ctx: Context) -> Result:
    r, readers = Result(), ctx.readers
    if not ctx.rehash:
        raise ReadError("the weights check re-hashes every file and runs only with --rehash-weights")
    if ctx.lock is None:
        targets = [(("weights",), ctx.profile.model_relpath)]
        targets += [(("encoders", name), subdir) for name, subdir in sorted(ctx.encoders_arg.items())]
    else:
        targets = [(("weights",), str(lock_lookup(ctx.lock, ("weights", "local_subdir"))))]
        encoders = ctx.lock.get("encoders")
        if isinstance(encoders, Mapping):
            targets += [(("encoders", name), str(lock_lookup(ctx.lock, ("encoders", name, "local_subdir"))))
                        for name in sorted(encoders)]
        r.require(
            targets[0][1] == ctx.profile.model_relpath,
            f"base-lock weights.local_subdir {targets[0][1]!r} is not the profile's model_relpath {ctx.profile.model_relpath!r}",
        )
    summary: dict[str, Any] = {}
    for prefix, subdir in targets:
        root = str(PurePosixPath(ctx.profile.data_mount) / subdir)
        files, dots = readers.list_files(root)
        hashed = readers.sha256_files([f"{root}/{name}" for name in files])
        r.claim(*prefix, "local_subdir", value=subdir)
        r.claim(*prefix, "file_names", value=list(files))
        r.claim(*prefix, "other_entries", value=list(dots))
        total = 0
        for name in files:
            size, sha = hashed[f"{root}/{name}"]
            total += size
            r.claim(*prefix, "files", name, "size_bytes", value=size)
            r.claim(*prefix, "files", name, "sha256", value=sha)
        r.claim(*prefix, "payload_size_bytes", value=total)
        summary["/".join(prefix)] = {"root": root, "files": len(files), "bytes": total}
    r.details["rehashed"] = summary
    return r


# ---------------------------------------------------------------------------
# Check 5: endpoints.
# ---------------------------------------------------------------------------


def declared_window(server_args: Sequence[str]) -> int | None:
    tokens = list(server_args)
    if "--max-model-len" in tokens and tokens.index("--max-model-len") + 1 < len(tokens):
        value = tokens[tokens.index("--max-model-len") + 1]
        return int(value) if value.isdigit() else None
    return None


def check_endpoints(ctx: Context) -> Result:
    r, readers = Result(), ctx.readers
    served = str(ctx.fragment["served_model_name"])
    window = declared_window([str(token) for token in ctx.fragment.get("server_args") or []])
    r.claim("protocol", "served_model_name", value=served)
    seen: dict[str, Any] = {}
    for instance in ctx.profile.instances:
        state = ctx.inspect(instance.name)
        info = state.get("State") or {}
        entry: dict[str, Any] = {
            "container_status": info.get("Status"), "restart_count": state.get("RestartCount"),
        }
        r.require(info.get("Running") is True and info.get("Status") == "running", f"{instance.name}: container is not running ({info.get('Status')!r})")
        r.require(state.get("RestartCount") == 0, f"{instance.name}: container restarted {state.get('RestartCount')!r} time(s)")
        base = instance_base_url(ctx, instance)
        status, _ = readers.http_get(base + "/health")
        entry["health_status"] = status
        r.require(status == 200, f"{instance.name}: /health answered {status}")
        status, text = readers.http_get(base + "/v1/models")
        entry["models_status"] = status
        listed: list[str] = []
        max_len = None
        if status == 200:
            try:
                data = json.loads(text).get("data") or []
            except (ValueError, AttributeError):
                data = []
            listed = sorted(str(item.get("id")) for item in data if isinstance(item, dict))
            max_len = next((item.get("max_model_len") for item in data if isinstance(item, dict) and item.get("id") == served), None)
        entry["models"], entry["max_model_len"] = listed, max_len
        r.require(status == 200 and served in listed, f"{instance.name}: /v1/models (HTTP {status}) does not list {served!r}")
        r.require(window is None or max_len == window, f"{instance.name}: served max_model_len {max_len!r} is not the declared window {window!r}")
        seen[instance.name] = entry
    r.details["instances"] = seen
    return r


# ---------------------------------------------------------------------------
# Check 6: protocol.
# ---------------------------------------------------------------------------

DECODING_KEYS = (
    "reasoning_budget_tokens", "answer_reserve_tokens", "forced_close_message_sha256", "reasoning_context_policy",
    "timeout_seconds", "enable_thinking",
)


def fragment_decoding(fragment: Mapping[str, Any]) -> dict[str, Any]:
    params = fragment["decoding_params"]
    decoding = {key: params[key] for key in ("temperature", "seed", "max_tokens")}
    decoding.update({key: fragment[key] for key in DECODING_KEYS})
    return decoding


def check_protocol(ctx: Context) -> Result:
    r = Result()
    fragment = ctx.fragment
    served_expected, tokens_expected, _ = classify_server_flags(ctx.expected_argv(ctx.profile.instances[0])[2:])
    r.claim("protocol", "served_model_name", value=served_expected)
    r.claim("protocol", "server_args", value=list(tokens_expected))
    r.claim("protocol", "decoding", value=fragment_decoding(fragment))
    live: dict[str, Any] = {}
    for instance in ctx.profile.instances:
        model_dir, flags = live_vllm_args(ctx.inspect(instance.name))
        served, tokens, _deployment = classify_server_flags(flags)
        live[instance.name] = {"model_dir": model_dir, "served_model_name": served, "server_args": list(tokens)}
        r.require(list(tokens) == list(tokens_expected), f"{instance.name}: live protocol flags {list(tokens)} differ from the fragment's {list(tokens_expected)}")
        r.require(served == served_expected, f"{instance.name}: serves {served!r}, not {served_expected!r}")
        r.require(model_dir == ctx.profile.model_path, f"{instance.name}: serves {model_dir!r}, not {ctx.profile.model_path!r}")
    r.require(len({json.dumps(item, sort_keys=True) for item in live.values()}) == 1, "instances do not share one protocol subset")
    r.details["instances"] = live
    if ctx.printing and not ctx.receipt_path():
        r.details["receipt"] = "no --receipt given; protocol_sha256 not observed"
        return r
    receipt = ctx.receipt
    recomputed = protocol_sha256(receipt)
    r.claim("protocol", "protocol_sha256", value=recomputed)
    r.require(recomputed == receipt.get("protocol_sha256"), f"receipt protocol_sha256 {receipt.get('protocol_sha256')!r} is not the recomputed {recomputed!r}")
    protocol = receipt.get("protocol") or {}
    r.require(list(protocol.get("server_args") or []) == list(tokens_expected), "receipt protocol.server_args differ from the live protocol flags")
    r.require(protocol.get("served_model_name") == served_expected, "receipt served_model_name differs from the live one")
    r.details["receipt_protocol_sha256"] = recomputed
    return r


# ---------------------------------------------------------------------------
# Check 7: deployment (and host identity).
# ---------------------------------------------------------------------------


def _shm_bytes(text: str) -> int:
    return int(text[:-1]) * (1024 ** 2 if text[-1] == "m" else 1024 ** 3)


def check_deployment(ctx: Context) -> Result:
    r, profile = Result(), ctx.profile
    host = ctx.host()
    hostname = host["hostname"]
    r.require(hostname == profile.host, f"this host is {hostname!r} but the profile is for {profile.host!r}")
    r.require(host.get("card_count") == profile.host_card_count, f"host has {host.get('card_count')!r} cards, profile says {profile.host_card_count}")
    r.require(host.get("cpu_count") == profile.host_cpu_count, f"host has {host.get('cpu_count')!r} cores, profile says {profile.host_cpu_count}")
    r.claim("hosts", hostname, "card_count", value=host.get("card_count"))
    r.claim("hosts", hostname, "cpu_count", value=host.get("cpu_count"))
    r.claim("profiles", profile.profile_id, "profile_sha256", value=profile.sha256)
    base = ("profiles", profile.profile_id, "instances")
    seen_cards: dict[int, str] = {}
    seen_ports: dict[int, str] = {}
    seen_cpus: dict[int, str] = {}
    observed: dict[str, Any] = {}
    for instance in profile.instances:
        state = ctx.inspect(instance.name)
        r.require(str(state.get("Name", "")).lstrip("/") == instance.name, f"{instance.name}: container is named {state.get('Name')!r}")
        host_config = state.get("HostConfig") or {}
        devices = sorted(
            int(m.group(1)) for item in host_config.get("Devices") or []
            if (m := re.fullmatch(r"/dev/davinci(\d+)", str(item.get("PathOnHost"))))
        )
        cpuset = host_config.get("CpusetCpus")
        model_dir, flags = live_vllm_args(state)
        _served, _tokens, deployment = classify_server_flags(flags)
        labels = (state.get("Config") or {}).get("Labels") or {}
        argv = [state.get("Path"), *state["Args"]] if isinstance(state.get("Args"), list) else []
        expected_argv = [dep.CONTAINER_INTERPRETER, *dep.instance_container_args(ctx.profile, instance, ctx.fragment)]
        batched = deployment.get("--max-num-batched-tokens")
        facts = {
            "cards": devices, "cpuset": cpuset, "port": int(deployment.get("--port", -1)),
            "tensor_parallel_size": int(deployment.get("--tensor-parallel-size", -1)),
            "max_num_seqs": int(deployment.get("--max-num-seqs", -1)),
            "gpu_memory_utilization": float(deployment.get("--gpu-memory-utilization", -1)),
            "max_num_batched_tokens": int(batched) if batched is not None else None,
            "server_argv_sha256": hashlib.sha256(json.dumps(["serve", model_dir, *flags]).encode("utf-8")).hexdigest(),
        }
        for key, value in facts.items():
            r.claim(*base, instance.name, key, value=value)
        wanted = {
            "cards": list(instance.cards), "cpuset": instance.cpuset, "port": instance.port,
            "tensor_parallel_size": profile.tensor_parallel_size, "max_num_seqs": profile.max_num_seqs,
            "gpu_memory_utilization": profile.gpu_memory_utilization,
            "max_num_batched_tokens": profile.max_num_batched_tokens,
        }
        for key, value in wanted.items():
            r.require(facts[key] == value, f"{instance.name}: {key} is {facts[key]!r}, the profile says {value!r}")
        r.require(deployment.get("--host") == profile.bind_host, f"{instance.name}: --host is {deployment.get('--host')!r}, profile bind_host {profile.bind_host!r}")
        r.require(argv == expected_argv, f"{instance.name}: live argv differs from the argv derived from profile + fragment")
        r.require(
            labels.get(dep.LABEL_PROFILE) == profile.profile_id and labels.get(dep.LABEL_PROFILE_SHA256) == profile.sha256
            and labels.get(dep.LABEL_INSTANCE) == str(instance.index),
            f"{instance.name}: deployment labels {labels!r} do not name this profile and instance",
        )
        for card in devices:
            r.require(card not in seen_cards, f"{instance.name}: card {card} is also held by {seen_cards.get(card)}")
            seen_cards[card] = instance.name
        r.require(facts["port"] not in seen_ports, f"{instance.name}: port {facts['port']} is also used by {seen_ports.get(facts['port'])}")
        seen_ports[facts["port"]] = instance.name
        for cpu in sorted(dep.parse_cpuset(cpuset)) if isinstance(cpuset, str) else []:
            r.require(cpu not in seen_cpus, f"{instance.name}: cpu {cpu} is also pinned by {seen_cpus.get(cpu)}")
            seen_cpus[cpu] = instance.name
        observed[instance.name] = {**facts, "model_dir": model_dir, "labels": labels}
    r.details["instances"] = observed
    return r


# ---------------------------------------------------------------------------
# Check 8: env (HCCL interface, network, mounts).
# ---------------------------------------------------------------------------


def select_global_nic(nics: Sequence[str]) -> str | None:
    return next((name for name in nics if not _VIRTUAL_NIC.fullmatch(name)), None)


def check_env(ctx: Context) -> Result:
    r, profile = Result(), ctx.profile
    host = ctx.host()
    nic = select_global_nic(host.get("global_nics") or [])
    r.require(nic is not None, f"host reports no physical global-scope interface ({host.get('global_nics')!r})")
    r.claim("hosts", host["hostname"], "global_nic", value=nic)
    seen: dict[str, Any] = {}
    for instance in profile.instances:
        state = ctx.inspect(instance.name)
        env = container_env(state)
        host_config = state.get("HostConfig") or {}
        mounts = state.get("Mounts") or []
        ro_mounts = sorted(str(m.get("Destination")) for m in mounts if m.get("RW") is False)
        entry = {
            "interface_vars": {key: env.get(key) for key in INTERFACE_VARS},
            "network_mode": host_config.get("NetworkMode"), "shm_bytes": host_config.get("ShmSize"),
            "ro_mounts": ro_mounts, "owned_env": {key: env.get(key) for key, _ in instance.env},
        }
        seen[instance.name] = entry
        for key in INTERFACE_VARS:
            r.require(env.get(key) == nic, f"{instance.name}: {key} is {env.get(key)!r}, the NIC with the global IPv4 is {nic!r}")
        for key, value in instance.env:
            r.require(env.get(key) == value, f"{instance.name}: {key} is {env.get(key)!r}, the profile says {value!r}")
        r.require(host_config.get("NetworkMode") == "host", f"{instance.name}: network mode is {host_config.get('NetworkMode')!r}, not 'host'")
        r.require(host_config.get("ShmSize") == _shm_bytes(profile.shm_size), f"{instance.name}: shm is {host_config.get('ShmSize')!r} bytes, profile {profile.shm_size}")
        r.claim("network_env", "network_mode", value=host_config.get("NetworkMode"))
        r.claim("network_env", "ro_mounts", value=ro_mounts)
    r.details["instances"] = seen
    return r


# ---------------------------------------------------------------------------
# Check 9: kv.
# ---------------------------------------------------------------------------

_KV_SIZE = re.compile(r"KV cache size: ([\d,]+) tokens, Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x")
_PREFIX = re.compile(r"enable_prefix_caching=(True|False)")
_CHUNKED = re.compile(r"enable_chunked_prefill=(True|False)")
_GRAPH_MODE = re.compile(r"'cudagraph_mode': <CUDAGraphMode\.(\w+):")
_GRAPH_SIZES = re.compile(r"'cudagraph_capture_sizes': \[([\d,\s]*)\]")
_BLOCK = re.compile(r"Block size is set to (\d+)")
_AVAILABLE = re.compile(r"Available KV cache memory: ([\d.]+) GiB")


def parse_engine_log(lines: Sequence[str]) -> dict[str, Any]:
    """The last occurrence of each engine-derived fact in a vLLM startup log (None where absent)."""

    out: dict[str, Any] = {
        "kv_tokens": None, "max_concurrency": None, "window_tokens": None, "block_size": None, "prefix_caching": None,
        "chunked_prefill": None, "cudagraph_mode": None, "cudagraph_capture_sizes": None, "available_kv_gib": None,
    }
    for line in lines:
        if m := _KV_SIZE.search(line):
            out["kv_tokens"], out["window_tokens"] = int(m.group(1).replace(",", "")), int(m.group(2).replace(",", ""))
            out["max_concurrency"] = float(m.group(3))
        if m := _PREFIX.search(line):
            out["prefix_caching"] = m.group(1) == "True"
        if m := _CHUNKED.search(line):
            out["chunked_prefill"] = m.group(1) == "True"
        if m := _GRAPH_MODE.search(line):
            out["cudagraph_mode"] = m.group(1)
        if m := _GRAPH_SIZES.search(line):
            out["cudagraph_capture_sizes"] = [int(x) for x in m.group(1).replace(" ", "").split(",") if x]
        if m := _BLOCK.search(line):
            out["block_size"] = int(m.group(1))
        if m := _AVAILABLE.search(line):
            out["available_kv_gib"] = float(m.group(1))
    return out


def parse_cache_config_info(text: str) -> dict[str, str] | None:
    for line in text.splitlines():
        if line.startswith("vllm:cache_config_info"):
            return dict(re.findall(r'(\w+)="([^"]*)"', line))
    return None


def check_kv(ctx: Context) -> Result:
    r, readers = Result(), ctx.readers
    base = ("profiles", ctx.profile.profile_id, "instances")
    seen: dict[str, Any] = {}
    for instance in ctx.profile.instances:
        parsed = parse_engine_log(readers.container_log_lines(instance.name))
        env = container_env(ctx.inspect(instance.name))
        timeout = env.get("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS")
        parsed["execute_model_timeout_seconds"] = int(timeout) if timeout and timeout.isdigit() else None
        status, text = readers.http_get(instance_base_url(ctx, instance) + "/metrics")
        info = parse_cache_config_info(text) if status == 200 else None
        if info and info.get("num_gpu_blocks", "").isdigit() and info.get("block_size", "").isdigit():
            metrics_blocks, metrics_block_size = int(info["num_gpu_blocks"]), int(info["block_size"])
            from_metrics = metrics_blocks * metrics_block_size
            parsed["metrics_kv_tokens"] = from_metrics
            if parsed["kv_tokens"] is None:
                parsed["kv_tokens"] = from_metrics
            r.require(abs(from_metrics - parsed["kv_tokens"]) <= KV_REL_TOL * parsed["kv_tokens"], f"{instance.name}: /metrics says {from_metrics} KV tokens, the startup log {parsed['kv_tokens']}")
            if parsed["block_size"] is None:
                parsed["block_size"] = metrics_block_size
            r.require(parsed["block_size"] == metrics_block_size, f"{instance.name}: /metrics block size {metrics_block_size} differs from the log's {parsed['block_size']}")
        parsed["metrics_status"] = status
        for key in ("kv_tokens", "max_concurrency", "block_size", "prefix_caching", "chunked_prefill", "cudagraph_mode", "cudagraph_capture_sizes"):
            r.require(parsed[key] is not None, f"{instance.name}: the startup log and /metrics do not show {key}")
        for key, tol in (("kv_tokens", KV_REL_TOL), ("max_concurrency", KV_REL_TOL), ("block_size", None), ("prefix_caching", None),
                         ("chunked_prefill", None), ("cudagraph_mode", None), ("cudagraph_capture_sizes", None),
                         ("execute_model_timeout_seconds", None)):
            r.claim(*base, instance.name, "engine", key, value=parsed[key], rel_tol=tol)
        seen[instance.name] = parsed
    r.details["instances"] = seen
    return r


# ---------------------------------------------------------------------------
# Check 10: template.
# ---------------------------------------------------------------------------


def smoke_messages() -> list[dict[str, Any]]:
    wav = sweep.synthetic_wav("verify-engineering-base", 2.0)
    audio = {"type": "input_audio", "input_audio": {"data": base64.b64encode(wav).decode("ascii"), "format": "wav"}}
    return [{"role": "system", "content": SMOKE_SYSTEM},
            {"role": "user", "content": [audio, {"type": "text", "text": SMOKE_USER}]}]


def check_template(ctx: Context) -> Result:
    r, readers = Result(), ctx.readers
    model_dir = ctx.profile.model_path
    template = f"{model_dir}/chat_template.json"
    r.claim("template", "chat_template_sha256", value=readers.sha256_files([template])[template][1])
    try:
        config = json.loads(readers.read_text(f"{model_dir}/tokenizer_config.json"))
    except ValueError as error:
        raise ReadError(f"tokenizer_config.json is not JSON: {error}") from error
    r.claim("template", "tokenizer_config_has_chat_template", value="chat_template" in config)
    served = str(ctx.fragment["served_model_name"])
    messages = smoke_messages()
    seen: dict[str, Any] = {}
    for instance in ctx.profile.instances:
        base = instance_base_url(ctx, instance)
        entry: dict[str, Any] = {"raw_tails": {}, "tails": {}, "audio_spans": {}}
        for mode in (True, False):
            key = "thinking_true" if mode else "thinking_false"
            body = {"model": served, "messages": messages, "chat_template_kwargs": {"enable_thinking": mode},
                    "add_generation_prompt": True, "add_special_tokens": False, "return_token_strs": True}
            status, response = readers.http_post_json(base + "/tokenize", body)
            strs = response.get("token_strs") if isinstance(response, dict) else None
            r.require(status == 200 and isinstance(strs, list), f"{instance.name}: /tokenize ({key}) answered HTTP {status} without token_strs")
            joined = "".join(strs or [])
            text = decode_byte_level(joined)
            spans = [text.count("<|audio_start|>"), text.count("<|audio_end|>")]
            entry["raw_tails"][key], entry["tails"][key], entry["audio_spans"][key] = joined[-TAIL_CHARS:], text[-TAIL_CHARS:], spans
            r.claim("template", "tails", key, value=text[-TAIL_CHARS:])
            r.claim("template", "audio_spans", key, value=spans)
            r.require(spans == [1, 1], f"{instance.name}: {key} prompt has audio spans {spans}, expected exactly one")
        true_tail, false_tail = entry["tails"]["thinking_true"], entry["tails"]["thinking_false"]
        r.require(true_tail.endswith(TAIL_TRUE_SUFFIX) and "<think>" not in true_tail, f"{instance.name}: thinking=true tail {true_tail!r} must end at the assistant turn with no <think>")
        r.require(false_tail.endswith(TAIL_FALSE_SUFFIX), f"{instance.name}: thinking=false tail {false_tail!r} must end in the empty think block")
        count_body = {"model": served, "messages": messages, "chat_template_kwargs": {"enable_thinking": True},
                      "add_generation_prompt": True, "add_special_tokens": False}
        status, counted = readers.http_post_json(base + "/tokenize", count_body)
        measured = counted.get("count") if isinstance(counted, dict) else None
        chat = {"model": served, "messages": messages, "temperature": 0, "seed": 20260913, "max_tokens": 1,
                "thinking_token_budget": 1, "chat_template_kwargs": {"enable_thinking": True}}
        c_status, completion = readers.http_post_json(base + "/v1/chat/completions", chat)
        engine = (completion.get("usage") or {}).get("prompt_tokens") if isinstance(completion, dict) else None
        entry["tokenize_count"], entry["engine_prompt_tokens"] = measured, engine
        r.require(status == 200 and c_status == 200 and isinstance(measured, int) and measured == engine,
                  f"{instance.name}: /tokenize count {measured!r} (HTTP {status}) != engine prompt_tokens {engine!r} (HTTP {c_status})")
        seen[instance.name] = entry
    r.details["instances"] = seen
    return r


# ---------------------------------------------------------------------------
# Check 11: receipt.
# ---------------------------------------------------------------------------


def check_receipt(ctx: Context) -> Result:
    r, readers, profile = Result(), ctx.readers, ctx.profile
    path = ctx.receipt_path()
    receipt = ctx.receipt
    if ctx.printing or not ctx.receipt_arg:
        r.claim("receipt", "path", value=path)
    validate_vllm_receipt_document(receipt)
    readers.verify_runtime_files(receipt)
    image_id = readers.docker_image_inspect(profile.image).get("Id")
    container = receipt["container"]
    r.require(container["image_ref"] == profile.image, f"receipt image_ref {container['image_ref']!r} is not the profile's {profile.image!r}")
    r.require(container["image_id"] == image_id, f"receipt image_id {container['image_id']!r} is not the live image {image_id!r}")
    if ctx.lock is not None:
        locked = lock_lookup(ctx.lock, ("weights", "files"))
        modelled = {entry["path"].rsplit("/", 1)[-1]: entry["sha256"] for entry in receipt["model"]["files"]}
        r.require(set(modelled) == set(locked), f"receipt model files differ from the base-lock weights: {sorted(set(modelled) ^ set(locked))}")
        for name, sha in sorted(modelled.items()):
            r.require(name not in locked or locked[name].get("sha256") == sha, f"receipt sha256 of {name} differs from the base-lock")
        key = ctx.lock_value(("weights", "lock_key"), None)
        r.require(key is None or key == receipt["model_lock_key"], f"receipt model_lock_key {receipt['model_lock_key']!r} differs from the base-lock's {key!r}")
    r.require(receipt["model"]["model_dir"] == profile.model_path, f"receipt model_dir {receipt['model']['model_dir']!r} is not the profile's {profile.model_path!r}")
    per: dict[str, Any] = {}
    for instance in profile.instances:
        entry: dict[str, Any] = {}
        per[instance.name] = entry
        try:
            ri = instance_for_endpoint(receipt, profile.bind_host, instance.port)
            entry["receipt_instance_id"] = ri["instance_id"]
            expected = {"device_ids": list(instance.cards), "tensor_parallel_size": profile.tensor_parallel_size,
                        "max_num_seqs": profile.max_num_seqs, "gpu_memory_utilization": profile.gpu_memory_utilization,
                        "max_num_batched_tokens": profile.max_num_batched_tokens}
            for key, value in expected.items():
                r.require(ri[key] == value, f"{instance.name}: receipt {key} is {ri[key]!r}, the profile says {value!r}")
            cmdline = readers.container_cmdline(instance.name)
            entry["cmdline_head"] = cmdline[:4]
            assert_argv_matches_instance(cmdline, receipt, ri)
            env = container_env(ctx.inspect(instance.name))
            live_deployment, wanted_deployment = deployment_environment(env), instance_deployment_env(ri)
            r.require(live_deployment == wanted_deployment, f"{instance.name}: live deployment environment {live_deployment!r} differs from the receipt instance's {wanted_deployment!r}")
            live_pinned, wanted_pinned = pinned_environment(env), dict(receipt["protocol"]["env"])
            r.require(live_pinned == wanted_pinned, f"{instance.name}: pinned environment differs from the receipt at {sorted(set(live_pinned) ^ set(wanted_pinned)) or sorted(k for k in wanted_pinned if live_pinned.get(k) != wanted_pinned[k])}")
            devices = device_env_value(ri["device_ids"])
            r.require(env.get(DEVICE_ENV_KEY) == devices, f"{instance.name}: {DEVICE_ENV_KEY} is {env.get(DEVICE_ENV_KEY)!r}, the receipt's device group is {devices!r}")
        except Exception as error:  # noqa: BLE001  (GateClosed, ReadError, KeyError: this instance cannot be proven)
            r.failures.append(f"{instance.name}: {type(error).__name__}: {error}")
    r.details["instances"] = per
    r.details["live_process_identity"] = (
        "argv from /proc/<pid>/cmdline and the container environment are compared with the receipt; the namespace, "
        "exe, maps and socket-ownership proofs of core.session.vllm_identity need the serving container's own /proc "
        "and are NOT run from this separate container"
    )
    return r


CHECK_FUNCTIONS: dict[str, Callable[[Context], Result]] = {
    "image": check_image, "versions": check_versions, "cann": check_cann, "weights": check_weights,
    "endpoints": check_endpoints, "protocol": check_protocol, "deployment": check_deployment, "env": check_env,
    "kv": check_kv, "template": check_template, "receipt": check_receipt,
}


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------


def run_check(name: str, ctx: Context) -> tuple[dict[str, Any], list[Claim]]:
    """One check's report entry and its claims. ANY exception is an unreadable or invalid input: the check fails."""

    try:
        result = CHECK_FUNCTIONS[name](ctx)
        facts = build_nested(result.claims)
    except Exception as error:  # noqa: BLE001  (fail closed on every kind of unreadable input)
        message = f"input unreadable or invalid: {type(error).__name__}: {error}"
        return {"verdict": "fail", "failures": [message], "facts": {}, "details": {}, "claims_checked": 0}, []
    failures = list(result.failures)
    checked = 0
    if ctx.lock is not None:
        for claim in result.claims:
            checked += 1
            try:
                expected = lock_lookup(ctx.lock, claim.path)
            except LockMissing as error:
                failures.append(f"base-lock lacks {error.args[0]}")
                continue
            if not values_equal(expected, claim.value, claim.rel_tol):
                failures.append(f"{' > '.join(claim.path)}: base-lock says {_short(expected)}, observed {_short(claim.value)}")
    entry = {"verdict": "pass" if not failures else "fail", "failures": failures, "facts": facts,
             "details": result.details, "claims_checked": checked}
    return entry, result.claims


def select_checks(only: Sequence[str] | None, rehash: bool, printing: bool, have_receipt: bool) -> tuple[list[str], dict[str, str]]:
    if only:
        unknown = sorted(set(only) - set(CHECK_NAMES))
        if unknown:
            raise ValueError(f"unknown check(s) {unknown}; choose from {list(CHECK_NAMES)}")
        return [name for name in CHECK_NAMES if name in only], {}
    selected, skipped = [], {}
    for name in CHECK_NAMES:
        if name == "weights" and not rehash:
            skipped[name] = "--rehash-weights not given"
        elif name == "receipt" and printing and not have_receipt:
            skipped[name] = "--print-observed without --receipt"
        else:
            selected.append(name)
    return selected, skipped


def _canonical(document: Any) -> str:
    return json.dumps(document, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--profile", required=True, help="deployment profile JSON (configs/deployment/<id>.json)")
    p.add_argument("--base-lock", help="base-lock record (saea-engineering-base-lock-v1)")
    p.add_argument("--out", help="verification report path (required unless --print-observed)")
    p.add_argument("--only", help="comma-separated subset of checks: " + ",".join(CHECK_NAMES))
    p.add_argument("--rehash-weights", action="store_true", help="enable the expensive weights check")
    p.add_argument("--print-observed", action="store_true", help="write the observed facts L1-L15 as a base-lock record")
    p.add_argument("--receipt", help="runtime receipt path (overrides base-lock receipt.path)")
    p.add_argument("--encoder", action="append", default=[], metavar="NAME=LOCAL_SUBDIR", help="--print-observed: encoder weights to hash")
    p.add_argument("--repo-root", default=str(_REPO))
    p.add_argument("--data-root", help="override the profile's data_mount")
    p.add_argument("--facts-dir", default="/facts", help="host facts written by npu_verify_base.sh")
    p.add_argument("--host-root", default="", help="prefix under which the host's /usr/local/Ascend and /etc are mounted read-only")
    return p


def main(argv: Sequence[str] | None = None, *, readers: Readers | None = None,
         clock: Callable[[], datetime] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.print_observed and not (args.base_lock and args.out):
        print("usage error: --base-lock and --out are required unless --print-observed", file=sys.stderr)
        return 2
    readers = readers or LiveReaders(args.facts_dir, args.host_root)
    now = (clock or (lambda: datetime.now(timezone.utc)))()
    only = [name for name in (args.only or "").split(",") if name] or None
    try:
        selected, skipped = select_checks(only, args.rehash_weights, args.print_observed, bool(args.receipt))
    except ValueError as error:
        print(f"usage error: {error}", file=sys.stderr)
        return 2
    inputs: dict[str, Any] = {"profile_path": args.profile, "base_lock_path": args.base_lock, "repo_root": args.repo_root}
    setup_error: str | None = None
    profile: dep.DeploymentProfile | None = None
    lock: dict[str, Any] | None = None
    try:
        document = json.loads(readers.read_text(args.profile))
        profile = dep.validate_deployment_profile(document, source=args.profile, data_root=args.data_root)
        if profile.profile_id != Path(args.profile).stem:
            raise ReadError(f"profile_id {profile.profile_id!r} is not the file stem of {args.profile}")
        inputs.update(profile_id=profile.profile_id, profile_sha256=profile.sha256, data_mount=profile.data_mount)
    except Exception as error:  # noqa: BLE001
        setup_error = f"deployment profile unreadable or invalid: {type(error).__name__}: {error}"
    if not args.print_observed and setup_error is None:
        try:
            raw = readers.read_bytes(args.base_lock)
            lock = json.loads(raw.decode("utf-8"))
            inputs["base_lock_sha256"] = hashlib.sha256(raw).hexdigest()
            if not isinstance(lock, dict) or lock.get("schema") != LOCK_SCHEMA:
                raise ReadError(f"base-lock schema must be {LOCK_SCHEMA!r}")
            inputs["base_lock_schema"] = lock["schema"]
        except Exception as error:  # noqa: BLE001
            setup_error, lock = f"base-lock unreadable or invalid: {type(error).__name__}: {error}", None
    checks: dict[str, Any] = {}
    claims: list[Claim] = []
    if setup_error is not None:
        for name in selected:
            checks[name] = {"verdict": "fail", "failures": [setup_error], "facts": {}, "details": {}, "claims_checked": 0}
    else:
        assert profile is not None
        encoders = dict(item.split("=", 1) for item in args.encoder if "=" in item)
        ctx = Context(readers, profile, Path(args.repo_root), lock, bool(args.rehash_weights), args.receipt, encoders)
        for name in selected:
            checks[name], found = run_check(name, ctx)
            claims.extend(found)
    failed = sorted(name for name, entry in checks.items() if entry["verdict"] != "pass")
    if args.print_observed:
        try:
            observed = build_nested(claims)
        except ClaimConflict as error:
            failed.append(f"claims: {error}")
            observed = {}
        observed = {"schema": LOCK_SCHEMA, **observed}
        text = _canonical(observed)
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
        else:
            sys.stdout.write(text)
        for name in failed:
            print(f"observation problem in {name}: {checks.get(name, {}).get('failures')}", file=sys.stderr)
        return 1 if failed else 0
    report = {
        "schema": SCHEMA, "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "inputs": inputs,
        "selected_checks": selected, "skipped_checks": skipped, "checks": checks,
        "failed_checks": failed, "verdict": "pass" if selected and not failed else "fail",
    }
    try:
        Path(args.out).write_text(_canonical(report), encoding="utf-8")
    except OSError as error:
        print(f"cannot write {args.out}: {error}", file=sys.stderr)
        return 2
    print(f"{report['verdict'].upper()}: {len(selected) - len(failed)}/{len(selected)} checks passed"
          + (f"; failed: {', '.join(failed)}" if failed else "") + f"; report {args.out}")
    return 0 if report["verdict"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())

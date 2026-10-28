"""The vLLM runtime receipt (owner decision 9.3, 2026-10-28): its schema, its checks, and the gate mixin.

Owner, 9.3: approved, "preferably compatible via an adapter pattern". This module is the receipt half of the vLLM
runtime-identity adapter (``contracts.runtime_identity``). It sits next to the llama.cpp receipt checks in
``gate_runtime.py`` and does not change them. The llama gate keeps its exact key set and refuses this schema. The vLLM
gate accepts only this schema.

What the receipt binds
======================

A vLLM server is not a single binary. It is a Python interpreter running source trees, and its behaviour also depends
on the environment the interpreter was started with. The receipt therefore binds every input that decides what answers
a request, and the gate re-measures each one from bytes on every model touch:

* ``interpreter``: the real executable file (no symlinks) and its sha256. On the image this is a ~10 KB stub.
* ``runtime_libraries``: the shared libraries the stub runs (``libpython3.12.so.1.0``). Re-hashed every time, like the
  llama.cpp ``build/bin`` set.
* ``site_packages`` and ``import_path_files``: every ``*.pth``, ``__editable__*.py``, ``sitecustomize.py`` and
  ``usercustomize.py`` file in the interpreter's site directory, as an exact set. These files decide where ``import
  vllm`` resolves and can run code at start-up.
* ``source_trees``: the ``vllm`` and ``vllm_ascend`` package directories, as an exact-set manifest (relative path,
  size, sha256 of every regular file; ``__pycache__`` directories excluded). The gate walks and re-hashes both trees.
  A symlink or an unreadable entry inside a tree is refused, not skipped.
* ``packages``: name, version and the sha256 of ``METADATA`` and ``RECORD`` for vllm, vllm-ascend and the libraries
  they run on (torch, torch_npu, transformers). For torch and the other non-tree packages this binds the installer's
  manifest, not each installed file.
* ``model``: the served model directory and every regular file in it (weights, config, tokenizer, templates) by size
  and sha256, against the umbrella lock entry. The directory must hold exactly these files; any other entry must be a
  directory named in ``other_entries``. vLLM reads the whole directory, so an added ``tokenizer.json`` would change
  tokenisation. Without this exact-set check that addition would go unnoticed.
* ``container``: the image reference and image id (``sha256:...``). The process inside the container cannot see the
  image id. The producer records it from ``docker image inspect`` on the host. The gate binds the image's content
  through the items above, not through the id.
* ``protocol``: the served model name, the interpreter argv[0], the fixed entry point
  ``-m vllm.entrypoints.cli.main serve``, the protocol server flags, the working directory and the behaviour-relevant
  environment (``PINNED_ENV_PREFIXES`` / ``PINNED_ENV_NAMES``).
* ``deployment``: one entry per serving instance (owner input 2026-10-28: serving is multi-instance). Each entry holds
  the loopback host, port, device ids, tensor-parallel size, ``--max-num-seqs``, ``--gpu-memory-utilization`` and
  optional ``--max-num-batched-tokens``.

``protocol_sha256`` is the canonical hash of everything except ``deployment``. A change of cards, ports, TP or
per-instance concurrency changes only ``deployment``, so it does not look like a protocol change. Any other change
moves the protocol hash.

Argv model
==========

An instance's command line is::

    <argv0> -m vllm.entrypoints.cli.main serve <model_dir> --served-model-name <name> <flag groups...>

A flag group is one ``--flag`` token and the values after it. Groups whose flag is in ``DEPLOYMENT_FLAGS`` are
deployment groups. Each takes exactly one value and may appear only once. ``--host``, ``--port``,
``--tensor-parallel-size``, ``--max-num-seqs`` and ``--gpu-memory-utilization`` are required. Every other group is a
protocol group, compared in order with ``protocol.server_args``. ``--flag=value`` spellings and short options are
refused, because they would let a flag change its class unseen. Flags that load a file or code the receipt does not
pin (``REFUSED_SERVER_FLAGS``) are also refused, as is any value that names a filesystem path.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import speech_aware_evidence_acquisition.contracts as _contracts_pkg

from .errors import GateClosed
from .identifiers import _SHA256, VLLM_RUNTIME_RECEIPT_SCHEMA

#: The one accepted entry point after argv[0].
VLLM_ENTRYPOINT: tuple[str, ...] = ("-m", "vllm.entrypoints.cli.main", "serve")
SERVED_MODEL_NAME_FLAG = "--served-model-name"
#: Deployment flags: they say where and how wide an instance runs, never what it computes.
DEPLOYMENT_FLAGS: Mapping[str, str] = {
    "--host": "host",
    "--port": "port",
    "--tensor-parallel-size": "tensor_parallel_size",
    "--max-num-seqs": "max_num_seqs",
    "--gpu-memory-utilization": "gpu_memory_utilization",
    "--max-num-batched-tokens": "max_num_batched_tokens",
}
REQUIRED_DEPLOYMENT_FLAGS = frozenset(set(DEPLOYMENT_FLAGS) - {"--max-num-batched-tokens"})
#: Flags that make the server read a file, a model or code the receipt does not pin.
REFUSED_SERVER_FLAGS = frozenset(
    {
        "--model",
        "--tokenizer",
        "--revision",
        "--tokenizer-revision",
        "--code-revision",
        "--chat-template",
        "--hf-config-path",
        "--config",
        "--download-dir",
        "--tool-parser-plugin",
        "--middleware",
        "--lora-modules",
        "--allowed-local-media-path",
        "--trust-remote-code",
        "--api-key",
    }
)
#: Environment variables the live process must carry exactly as the receipt pins them (prefix or exact name).
PINNED_ENV_PREFIXES: tuple[str, ...] = (
    "PYTHON",
    "VLLM_",
    "LD_",
    "ASCEND_",
    "HCCL_",
    "ATB_",
    "LCCL_",
    "TORCH",
    "OMP_",
    "MKL_",
    "TASK_QUEUE_",
    "HF_",
    "TRANSFORMERS_",
    "TOKENIZERS_",
)
PINNED_ENV_NAMES = frozenset({"PATH", "SOC_VERSION", "CANN_VERSION"})
#: The per-instance device-visibility variable: deployment, derived from ``device_ids``.
DEVICE_ENV_KEY = "ASCEND_RT_VISIBLE_DEVICES"
#: The user site directory must be off, so ``~/.local`` cannot shadow a pinned package.
REQUIRED_ENV = {"PYTHONNOUSERSITE": "1"}
#: The packages a vLLM receipt must bind (normalized names).
REQUIRED_PACKAGES = frozenset({"vllm", "vllm-ascend"})
REQUIRED_SOURCE_TREES = frozenset({"vllm", "vllm_ascend"})
_INSTANCE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")
_LOOPBACK_HOSTS = ("127.0.0.1", "::1")

RECEIPT_KEYS = frozenset(
    {
        "schema",
        "study",
        "model_lock_key",
        "container",
        "interpreter",
        "runtime_libraries",
        "site_packages",
        "import_path_files",
        "source_trees",
        "packages",
        "model",
        "protocol",
        "deployment",
        "protocol_sha256",
    }
)
_FILE_KEYS = frozenset({"path", "bytes", "sha256"})
_PROTOCOL_KEYS = frozenset({"served_model_name", "argv0", "entrypoint", "server_args", "cwd", "env"})
_INSTANCE_KEYS = frozenset(
    {
        "instance_id",
        "host",
        "port",
        "device_ids",
        "tensor_parallel_size",
        "max_num_seqs",
        "gpu_memory_utilization",
        "max_num_batched_tokens",
    }
)


def _fail(message: str) -> GateClosed:
    return GateClosed(f"vllm runtime receipt: {message}")


# ---------------------------------------------------------------------------
# Hashing helpers (shared with the producer and the live-process check).
# ---------------------------------------------------------------------------


def canonical_sha256(document: object) -> str:
    raw = json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def protocol_sha256(receipt: Mapping[str, object]) -> str:
    """The protocol identity: the canonical hash of every receipt field except ``deployment`` and itself."""

    return canonical_sha256({key: value for key, value in receipt.items() if key not in {"deployment", "protocol_sha256"}})


def file_entry(path: Path) -> dict[str, object]:
    """``{path, bytes, sha256}`` for one absolute regular file, measured from its bytes."""

    size, sha = _contracts_pkg._measure_file(Path(path), str(path))
    return {"path": str(path), "bytes": size, "sha256": sha}


def source_tree_manifest(root: Path) -> dict[str, object]:
    """Exact-set manifest of one package directory: ``{files, bytes, manifest_sha256}``.

    Every regular file below ``root`` counts, except files under a ``__pycache__`` directory. The manifest hash covers
    ``relpath NUL size NUL sha256 LF`` lines in sorted relpath order. A symlink, an unreadable directory or any entry
    that is neither a file nor a directory is refused. A tree the checker cannot fully read cannot be proven.
    """

    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise _fail(f"source tree {root} is not a real directory")
    lines: list[str] = []
    total = 0

    def refuse(error: OSError) -> None:
        raise _fail(f"source tree {root} cannot be fully read: {error}")

    for directory, subdirs, files in os.walk(root, onerror=refuse, followlinks=False):
        base = Path(directory)
        for name in list(subdirs):
            entry = base / name
            if entry.is_symlink():
                raise _fail(f"source tree {root} contains a symlink {entry}")
            if name == "__pycache__":
                subdirs.remove(name)
                continue
            if not os.access(entry, os.R_OK | os.X_OK):
                raise _fail(f"source tree {root} contains an unreadable directory {entry}")
        subdirs.sort()
        for name in sorted(files):
            entry = base / name
            if entry.is_symlink() or not entry.is_file():
                raise _fail(f"source tree {root} contains a non-regular entry {entry}")
            size, sha = _contracts_pkg._measure_file(entry, f"source tree file {entry}")
            lines.append(f"{entry.relative_to(root).as_posix()}\0{size}\0{sha}\n")
            total += size
    if not lines:
        raise _fail(f"source tree {root} holds no files")
    lines.sort()
    return {
        "files": len(lines),
        "bytes": total,
        "manifest_sha256": hashlib.sha256("".join(lines).encode("utf-8")).hexdigest(),
    }


def import_path_file_names(site_packages: Path) -> list[Path]:
    """The files in a site directory that decide import resolution or run code at interpreter start-up."""

    try:
        entries = sorted(Path(site_packages).iterdir())
    except OSError as error:
        raise _fail(f"cannot enumerate site directory {site_packages}: {error}") from error
    chosen = []
    for entry in entries:
        name = entry.name
        if (
            name.endswith(".pth")
            or (name.startswith("__editable__") and name.endswith(".py"))
            or name in {"sitecustomize.py", "usercustomize.py"}
        ):
            if entry.is_symlink() or not entry.is_file():
                raise _fail(f"import-path entry {entry} is not a regular file")
            chosen.append(entry)
    return chosen


def normalize_package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def read_metadata_identity(metadata: Path) -> tuple[str, str]:
    """``(Name, Version)`` from one ``METADATA`` file's header block."""

    try:
        text = Path(metadata).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise _fail(f"cannot read {metadata}: {error}") from error
    name = version = None
    for line in text.splitlines():
        if not line.strip():
            break
        if line.startswith("Name:") and name is None:
            name = line.split(":", 1)[1].strip()
        elif line.startswith("Version:") and version is None:
            version = line.split(":", 1)[1].strip()
    if not name or not version:
        raise _fail(f"{metadata} lacks a Name/Version header")
    return name, version


# ---------------------------------------------------------------------------
# Argv and environment.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VllmServerArgv:
    """One instance command line, split into protocol and deployment parts (module docstring, "Argv model")."""

    argv0: str
    model_dir: str
    served_model_name: str
    protocol_args: tuple[str, ...]
    deployment: Mapping[str, str]


def _looks_like_path(value: str) -> bool:
    return value.startswith(("/", "./", "../", "~"))


def _flag_groups(tokens: Sequence[str]) -> list[list[str]]:
    groups: list[list[str]] = []
    for token in tokens:
        if token.startswith("--"):
            if "=" in token:
                raise _fail(f"--flag=value spellings are refused: {token!r}")
            groups.append([token])
        elif re.match(r"-[A-Za-z]", token):
            raise _fail(f"short options are refused (one spelling per flag): {token!r}")
        elif not groups:
            raise _fail(f"argv token {token!r} is not preceded by a flag")
        else:
            groups[-1].append(token)
    return groups


def classify_server_flags(tokens: Sequence[str]) -> tuple[str | None, tuple[str, ...], dict[str, str]]:
    """``(served model name, protocol tokens, deployment flags)`` of a flag list (module docstring, "Argv model")."""

    served: str | None = None
    deployment: dict[str, str] = {}
    protocol: list[str] = []
    for group in _flag_groups([str(token) for token in tokens]):
        flag, values = group[0], group[1:]
        if flag in REFUSED_SERVER_FLAGS:
            raise _fail(f"flag {flag} loads a file, model or code the receipt does not pin")
        if flag == SERVED_MODEL_NAME_FLAG:
            if served is not None or len(values) != 1 or not values[0]:
                raise _fail(f"{SERVED_MODEL_NAME_FLAG} must appear once with one value")
            served = values[0]
            continue
        if flag in DEPLOYMENT_FLAGS:
            if flag in deployment or len(values) != 1:
                raise _fail(f"deployment flag {flag} must appear once with one value")
            deployment[flag] = values[0]
            continue
        for value in values:
            if _looks_like_path(value):
                raise _fail(f"flag {flag} names a filesystem path {value!r} the receipt does not pin")
        protocol.extend(group)
    return served, tuple(protocol), deployment


def split_vllm_server_argv(argv: Sequence[str]) -> VllmServerArgv:
    """Parse one instance argv; refuse anything outside the argv model."""

    argv = [str(token) for token in argv]
    if len(argv) < 5 or tuple(argv[1:4]) != VLLM_ENTRYPOINT:
        raise _fail(f"argv must start with <interpreter> {' '.join(VLLM_ENTRYPOINT)} <model_dir>; got {argv[:5]!r}")
    model_dir = argv[4]
    if not model_dir.startswith("/"):
        raise _fail(f"the served model directory must be an absolute path, got {model_dir!r}")
    served, protocol, deployment = classify_server_flags(argv[5:])
    if served is None:
        raise _fail(f"argv lacks {SERVED_MODEL_NAME_FLAG}")
    missing = sorted(REQUIRED_DEPLOYMENT_FLAGS - set(deployment))
    if missing:
        raise _fail(f"argv lacks the deployment flags {missing}")
    return VllmServerArgv(
        argv0=argv[0], model_dir=model_dir, served_model_name=served, protocol_args=protocol, deployment=deployment
    )


def pinned_environment(environ: Mapping[str, str]) -> dict[str, str]:
    """The behaviour-relevant part of an environment (module docstring), without the device key."""

    return {
        key: value
        for key, value in environ.items()
        if key != DEVICE_ENV_KEY and (key.startswith(PINNED_ENV_PREFIXES) or key in PINNED_ENV_NAMES)
    }


def device_env_value(device_ids: Sequence[int]) -> str:
    return ",".join(str(int(device)) for device in device_ids)


def instance_argv(receipt: Mapping[str, object], instance: Mapping[str, object]) -> tuple[str, ...]:
    """The canonical command line of one receipt instance (what a spawn runs)."""

    protocol = receipt["protocol"]
    model = receipt["model"]
    deployment = [
        "--host", str(instance["host"]),
        "--port", str(instance["port"]),
        "--tensor-parallel-size", str(instance["tensor_parallel_size"]),
        "--max-num-seqs", str(instance["max_num_seqs"]),
        "--gpu-memory-utilization", str(instance["gpu_memory_utilization"]),
    ]
    if instance.get("max_num_batched_tokens") is not None:
        deployment += ["--max-num-batched-tokens", str(instance["max_num_batched_tokens"])]
    return (
        str(protocol["argv0"]),
        *VLLM_ENTRYPOINT,
        str(model["model_dir"]),
        SERVED_MODEL_NAME_FLAG,
        str(protocol["served_model_name"]),
        *[str(token) for token in protocol["server_args"]],
        *deployment,
    )


def _int_value(value: str, flag: str) -> int:
    if not re.fullmatch(r"[0-9]+", value):
        raise _fail(f"{flag} value {value!r} is not a non-negative integer")
    return int(value)


def assert_argv_matches_instance(
    argv: Sequence[str], receipt: Mapping[str, object], instance: Mapping[str, object]
) -> VllmServerArgv:
    """Refuse unless ``argv`` is the receipt's protocol plus exactly this instance's deployment."""

    parsed = split_vllm_server_argv(argv)
    protocol = receipt["protocol"]
    interpreter = receipt["interpreter"]["path"]
    try:
        argv0_resolved = str(Path(parsed.argv0).resolve(strict=True))
    except OSError as error:
        raise _fail(f"argv[0] {parsed.argv0!r} does not resolve: {error}") from error
    if parsed.argv0 != protocol["argv0"] or argv0_resolved != interpreter:
        raise _fail(
            f"argv[0] {parsed.argv0!r} is not the pinned interpreter invocation {protocol['argv0']!r} -> {interpreter}"
        )
    if parsed.model_dir != receipt["model"]["model_dir"]:
        raise _fail(f"argv serves {parsed.model_dir!r}, not the pinned model directory {receipt['model']['model_dir']!r}")
    if parsed.served_model_name != protocol["served_model_name"]:
        raise _fail(f"argv serves the model name {parsed.served_model_name!r}, not {protocol['served_model_name']!r}")
    if list(parsed.protocol_args) != list(protocol["server_args"]):
        raise _fail(
            f"argv protocol flags {list(parsed.protocol_args)} differ from the receipt's {list(protocol['server_args'])}"
        )
    deployment = parsed.deployment
    observed = {
        "host": deployment["--host"],
        "port": _int_value(deployment["--port"], "--port"),
        "tensor_parallel_size": _int_value(deployment["--tensor-parallel-size"], "--tensor-parallel-size"),
        "max_num_seqs": _int_value(deployment["--max-num-seqs"], "--max-num-seqs"),
        "max_num_batched_tokens": (
            _int_value(deployment["--max-num-batched-tokens"], "--max-num-batched-tokens")
            if "--max-num-batched-tokens" in deployment
            else None
        ),
    }
    for key, value in observed.items():
        if instance[key] != value:
            raise _fail(f"instance {instance['instance_id']!r} {key} is {instance[key]!r} but argv says {value!r}")
    try:
        utilization = float(deployment["--gpu-memory-utilization"])
    except ValueError as error:
        raise _fail("--gpu-memory-utilization is not a number") from error
    if utilization != float(instance["gpu_memory_utilization"]):
        raise _fail(
            f"instance {instance['instance_id']!r} gpu_memory_utilization is {instance['gpu_memory_utilization']!r} "
            f"but argv says {deployment['--gpu-memory-utilization']!r}"
        )
    return parsed


def deployment_instances(receipt: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    """Every instance this receipt deploys, in receipt order.

    The list a multi-endpoint transport routes over (coordinator design input, 2026-10-28): each entry carries one
    instance's recorded deployment facts -- host, port, device ids, tensor-parallel size, ``--max-num-seqs``,
    ``--gpu-memory-utilization`` -- which are deliberately OUTSIDE ``protocol_sha256``, so adding, moving or resizing
    an instance never looks like a change to what the model computes.
    """

    deployment = receipt["deployment"]
    return tuple(deployment["instances"])


def instance_by_id(receipt: Mapping[str, object], instance_id: str) -> Mapping[str, object]:
    """The one instance named ``instance_id`` (the stable key a router may hold instead of an endpoint)."""

    matches = [instance for instance in deployment_instances(receipt) if instance["instance_id"] == instance_id]
    if len(matches) != 1:
        raise _fail(f"no single deployment instance is named {instance_id!r}")
    return matches[0]


def instance_for_endpoint(receipt: Mapping[str, object], host: str, port: int) -> Mapping[str, object]:
    """The one deployment instance serving ``host:port``."""

    matches = [
        instance
        for instance in receipt["deployment"]["instances"]
        if instance["host"] == host and instance["port"] == port
    ]
    if len(matches) != 1:
        raise _fail(f"no single deployment instance serves {host}:{port}")
    return matches[0]


# ---------------------------------------------------------------------------
# Structural validation (no filesystem access).
# ---------------------------------------------------------------------------


def _absolute(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("/") or "\\" in value:
        raise _fail(f"{label} must be an absolute POSIX path, got {value!r}")
    if any(part in (".", "..") for part in value.split("/")[1:]) or "//" in value or (value != "/" and value.endswith("/")):
        raise _fail(f"{label} must be a normalized absolute path, got {value!r}")
    return value


def _file_record(entry: object, label: str, *, absolute: bool = True) -> Mapping[str, object]:
    if not isinstance(entry, Mapping) or set(entry) != _FILE_KEYS:
        raise _fail(f"{label} must have exact path/bytes/sha256")
    if absolute:
        _absolute(entry["path"], f"{label}.path")
    size, sha = entry["bytes"], entry["sha256"]
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise _fail(f"{label}.bytes must be a non-negative integer")
    if not isinstance(sha, str) or _SHA256.fullmatch(sha) is None:
        raise _fail(f"{label}.sha256 must be a 64-hex sha256")
    return entry


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise _fail(f"{label} must be a positive integer, got {value!r}")
    return value


def validate_vllm_receipt_document(receipt: object) -> Mapping[str, object]:
    """Structure, types and internal consistency of one vLLM runtime receipt. Reads no file."""

    if not isinstance(receipt, Mapping):
        raise _fail("must be a JSON object")
    if receipt.get("schema") != VLLM_RUNTIME_RECEIPT_SCHEMA:
        raise _fail(f"schema must be {VLLM_RUNTIME_RECEIPT_SCHEMA!r}, found {receipt.get('schema')!r}")
    if receipt.get("study") != _contracts_pkg.STUDY_SLUG:
        raise _fail(f"study must be {_contracts_pkg.STUDY_SLUG!r}, found {receipt.get('study')!r}")
    if set(receipt) != RECEIPT_KEYS:
        raise _fail(f"must have exact keys {sorted(RECEIPT_KEYS)}, found {sorted(receipt)}")
    if not isinstance(receipt["model_lock_key"], str) or not receipt["model_lock_key"]:
        raise _fail("model_lock_key must be a non-empty string")

    container = receipt["container"]
    if not isinstance(container, Mapping) or set(container) != {"image_ref", "image_id"}:
        raise _fail("container must have exact image_ref/image_id")
    if not isinstance(container["image_ref"], str) or not container["image_ref"].strip():
        raise _fail("container.image_ref must be a non-empty string")
    if not isinstance(container["image_id"], str) or _IMAGE_ID.fullmatch(container["image_id"]) is None:
        raise _fail("container.image_id must be 'sha256:' plus 64 hex digits")

    _file_record(receipt["interpreter"], "interpreter")
    libraries = receipt["runtime_libraries"]
    if not isinstance(libraries, list) or not libraries:
        raise _fail("runtime_libraries must be a non-empty list")
    for index, entry in enumerate(libraries):
        _file_record(entry, f"runtime_libraries[{index}]")
    if len({entry["path"] for entry in libraries}) != len(libraries):
        raise _fail("runtime_libraries duplicates a path")

    site = _absolute(receipt["site_packages"], "site_packages")
    import_files = receipt["import_path_files"]
    if not isinstance(import_files, list):
        raise _fail("import_path_files must be a list")
    for index, entry in enumerate(import_files):
        _file_record(entry, f"import_path_files[{index}]")
        if str(Path(entry["path"]).parent) != site:
            raise _fail(f"import_path_files[{index}] is outside site_packages")

    trees = receipt["source_trees"]
    if not isinstance(trees, list) or not trees:
        raise _fail("source_trees must be a non-empty list")
    names = set()
    roots = set()
    for index, tree in enumerate(trees):
        if not isinstance(tree, Mapping) or set(tree) != {"package", "root", "files", "bytes", "manifest_sha256"}:
            raise _fail(f"source_trees[{index}] must have exact package/root/files/bytes/manifest_sha256")
        if not isinstance(tree["package"], str) or not tree["package"]:
            raise _fail(f"source_trees[{index}].package must be a non-empty string")
        _absolute(tree["root"], f"source_trees[{index}].root")
        _positive_int(tree["files"], f"source_trees[{index}].files")
        if isinstance(tree["bytes"], bool) or not isinstance(tree["bytes"], int) or tree["bytes"] < 0:
            raise _fail(f"source_trees[{index}].bytes must be a non-negative integer")
        if not isinstance(tree["manifest_sha256"], str) or _SHA256.fullmatch(tree["manifest_sha256"]) is None:
            raise _fail(f"source_trees[{index}].manifest_sha256 must be a 64-hex sha256")
        names.add(tree["package"])
        roots.add(tree["root"])
    if len(names) != len(trees) or len(roots) != len(trees):
        raise _fail("source_trees duplicates a package or root")
    if not REQUIRED_SOURCE_TREES <= names:
        raise _fail(f"source_trees must include {sorted(REQUIRED_SOURCE_TREES)}")

    packages = receipt["packages"]
    if not isinstance(packages, list) or not packages:
        raise _fail("packages must be a non-empty list")
    package_names = set()
    for index, package in enumerate(packages):
        keys = {"name", "version", "dist_info", "metadata_sha256", "record_sha256"}
        if not isinstance(package, Mapping) or set(package) != keys:
            raise _fail(f"packages[{index}] must have exact keys {sorted(keys)}")
        for key in ("name", "version"):
            if not isinstance(package[key], str) or not package[key]:
                raise _fail(f"packages[{index}].{key} must be a non-empty string")
        _absolute(package["dist_info"], f"packages[{index}].dist_info")
        for key in ("metadata_sha256", "record_sha256"):
            if not isinstance(package[key], str) or _SHA256.fullmatch(package[key]) is None:
                raise _fail(f"packages[{index}].{key} must be a 64-hex sha256")
        package_names.add(normalize_package_name(package["name"]))
    if len(package_names) != len(packages):
        raise _fail("packages duplicates a package")
    if not REQUIRED_PACKAGES <= package_names:
        raise _fail(f"packages must include {sorted(REQUIRED_PACKAGES)}")

    model = receipt["model"]
    if not isinstance(model, Mapping) or set(model) != {"model_dir", "files", "other_entries"}:
        raise _fail("model must have exact model_dir/files/other_entries")
    _absolute(model["model_dir"], "model.model_dir")
    files = model["files"]
    if not isinstance(files, list) or not files:
        raise _fail("model.files must be a non-empty list")
    for index, entry in enumerate(files):
        _file_record(entry, f"model.files[{index}]", absolute=False)
    others = model["other_entries"]
    if not isinstance(others, list) or not all(isinstance(name, str) and name and "/" not in name for name in others):
        raise _fail("model.other_entries must be a list of plain entry names")

    protocol = receipt["protocol"]
    if not isinstance(protocol, Mapping) or set(protocol) != _PROTOCOL_KEYS:
        raise _fail(f"protocol must have exact keys {sorted(_PROTOCOL_KEYS)}")
    if not isinstance(protocol["served_model_name"], str) or not protocol["served_model_name"]:
        raise _fail("protocol.served_model_name must be a non-empty string")
    _absolute(protocol["argv0"], "protocol.argv0")
    if list(protocol["entrypoint"]) != list(VLLM_ENTRYPOINT) or not isinstance(protocol["entrypoint"], list):
        raise _fail(f"protocol.entrypoint must be {list(VLLM_ENTRYPOINT)}")
    server_args = protocol["server_args"]
    if not isinstance(server_args, list) or not all(isinstance(token, str) and token for token in server_args):
        raise _fail("protocol.server_args must be a list of non-empty strings")
    _absolute(protocol["cwd"], "protocol.cwd")
    env = protocol["env"]
    if not isinstance(env, Mapping) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
        raise _fail("protocol.env must map strings to strings")
    if dict(env) != pinned_environment(env):
        raise _fail("protocol.env may hold only the pinned behaviour-relevant variables (no device key)")
    for key, value in REQUIRED_ENV.items():
        if env.get(key) != value:
            raise _fail(f"protocol.env must set {key}={value}")

    deployment = receipt["deployment"]
    if not isinstance(deployment, Mapping) or set(deployment) != {"instances"}:
        raise _fail("deployment must have exactly an instances list")
    instances = deployment["instances"]
    if not isinstance(instances, list) or not instances:
        raise _fail("deployment.instances must be a non-empty list")
    seen_ids: set[str] = set()
    seen_endpoints: set[tuple[str, int]] = set()
    seen_devices: set[int] = set()
    for index, instance in enumerate(instances):
        label = f"deployment.instances[{index}]"
        if not isinstance(instance, Mapping) or set(instance) != _INSTANCE_KEYS:
            raise _fail(f"{label} must have exact keys {sorted(_INSTANCE_KEYS)}")
        instance_id = instance["instance_id"]
        if not isinstance(instance_id, str) or _INSTANCE_ID.fullmatch(instance_id) is None or instance_id in seen_ids:
            raise _fail(f"{label}.instance_id must be a unique identifier")
        seen_ids.add(instance_id)
        if instance["host"] not in _LOOPBACK_HOSTS:
            raise _fail(f"{label}.host must be a loopback literal {list(_LOOPBACK_HOSTS)}")
        port = instance["port"]
        if isinstance(port, bool) or not isinstance(port, int) or not 0 < port < 65536:
            raise _fail(f"{label}.port must be an integer in 1..65535")
        if (instance["host"], port) in seen_endpoints:
            raise _fail(f"{label} reuses the endpoint {instance['host']}:{port}")
        seen_endpoints.add((instance["host"], port))
        tensor_parallel = _positive_int(instance["tensor_parallel_size"], f"{label}.tensor_parallel_size")
        devices = instance["device_ids"]
        if (
            not isinstance(devices, list)
            or not devices
            or not all(isinstance(d, int) and not isinstance(d, bool) and d >= 0 for d in devices)
            or len(set(devices)) != len(devices)
        ):
            raise _fail(f"{label}.device_ids must be unique non-negative integers")
        if len(devices) != tensor_parallel:
            raise _fail(f"{label} has {len(devices)} devices for tensor-parallel size {tensor_parallel}")
        if seen_devices & set(devices):
            raise _fail(f"{label} shares devices with another instance")
        seen_devices |= set(devices)
        _positive_int(instance["max_num_seqs"], f"{label}.max_num_seqs")
        utilization = instance["gpu_memory_utilization"]
        if isinstance(utilization, bool) or not isinstance(utilization, (int, float)) or not 0 < utilization <= 1:
            raise _fail(f"{label}.gpu_memory_utilization must be a number in (0, 1]")
        if instance["max_num_batched_tokens"] is not None:
            _positive_int(instance["max_num_batched_tokens"], f"{label}.max_num_batched_tokens")
    served, protocol_tokens, deployment_flags = classify_server_flags(server_args)
    if served is not None or deployment_flags or list(protocol_tokens) != list(server_args):
        raise _fail(
            "protocol.server_args must hold only protocol flag groups: the served model name and the deployment "
            "flags have their own fields"
        )

    recorded = receipt["protocol_sha256"]
    if not isinstance(recorded, str) or _SHA256.fullmatch(recorded) is None or recorded != protocol_sha256(receipt):
        raise _fail("protocol_sha256 does not equal the canonical hash of the protocol fields")
    return receipt


# ---------------------------------------------------------------------------
# On-disk re-verification (gate time).
# ---------------------------------------------------------------------------


def _verify_real_file(entry: Mapping[str, object], label: str) -> None:
    path = Path(str(entry["path"]))
    if path.is_symlink():
        raise _fail(f"{label} {path} must name the real file, not a symlink")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise _fail(f"{label} {path} does not exist: {error}") from error
    if str(resolved) != str(path) or not resolved.is_file():
        raise _fail(f"{label} {path} is not a canonical regular file (resolves to {resolved})")
    size, sha = _contracts_pkg._measure_file(resolved, label)
    if size != entry["bytes"] or sha != entry["sha256"]:
        raise _fail(f"{label} {path} drifted from the receipt; model touch refused")


def verify_vllm_runtime_files(receipt: Mapping[str, object]) -> None:
    """Re-measure interpreter, libraries, import-path files, source trees and package metadata."""

    _verify_real_file(receipt["interpreter"], "interpreter")
    argv0 = Path(str(receipt["protocol"]["argv0"]))
    try:
        argv0_target = argv0.resolve(strict=True)
    except OSError as error:
        raise _fail(f"protocol.argv0 {argv0} does not resolve: {error}") from error
    if str(argv0_target) != str(receipt["interpreter"]["path"]):
        raise _fail(f"protocol.argv0 {argv0} resolves to {argv0_target}, not the pinned interpreter")
    for index, entry in enumerate(receipt["runtime_libraries"]):
        _verify_real_file(entry, f"runtime_libraries[{index}]")
    site = Path(str(receipt["site_packages"]))
    declared = {str(entry["path"]) for entry in receipt["import_path_files"]}
    actual = {str(path) for path in import_path_file_names(site)}
    if declared != actual:
        raise _fail(f"import-path files drifted: {sorted(declared ^ actual)} are unlisted or missing")
    for index, entry in enumerate(receipt["import_path_files"]):
        _verify_real_file(entry, f"import_path_files[{index}]")
    for tree in receipt["source_trees"]:
        measured = source_tree_manifest(Path(str(tree["root"])))
        recorded = {key: tree[key] for key in ("files", "bytes", "manifest_sha256")}
        if measured != recorded:
            raise _fail(
                f"source tree {tree['package']} at {tree['root']} drifted from the receipt "
                f"({measured['files']} files, manifest {measured['manifest_sha256'][:12]}...); model touch refused"
            )
    for package in receipt["packages"]:
        dist_info = Path(str(package["dist_info"]))
        for key, filename in (("metadata_sha256", "METADATA"), ("record_sha256", "RECORD")):
            path = dist_info / filename
            if path.is_symlink() or not path.is_file():
                raise _fail(f"package {package['name']} lacks {path}")
            _size, sha = _contracts_pkg._measure_file(path, f"package {package['name']} {filename}")
            if sha != package[key]:
                raise _fail(f"package {package['name']} {filename} drifted from the receipt")
        name, version = read_metadata_identity(dist_info / "METADATA")
        if normalize_package_name(name) != normalize_package_name(str(package["name"])) or version != package["version"]:
            raise _fail(f"package {package['name']} metadata says {name} {version}, not {package['version']}")


def verify_model_directory(receipt: Mapping[str, object], materialization_root: Path, subdir: str) -> None:
    """The served model directory is the lock's directory and holds exactly the receipt's files."""

    model = receipt["model"]
    expected_dir = Path(materialization_root) / subdir
    if str(model["model_dir"]) != str(expected_dir):
        raise _fail(f"model.model_dir {model['model_dir']!r} is not the lock's directory {str(expected_dir)!r}")
    try:
        entries = list(expected_dir.iterdir())
    except OSError as error:
        raise _fail(f"cannot enumerate the served model directory {expected_dir}: {error}") from error
    files = set()
    others = set()
    for entry in entries:
        if entry.is_symlink():
            raise _fail(f"the served model directory holds a symlink {entry.name!r}")
        if entry.is_file():
            files.add(entry.name)
        elif entry.is_dir():
            others.add(entry.name)
        else:
            raise _fail(f"the served model directory holds a non-file entry {entry.name!r}")
    declared = set()
    for entry in model["files"]:
        relative = str(entry["path"])
        if relative.rsplit("/", 1)[0] != subdir:
            raise _fail(f"model file {relative!r} is not directly in the lock's directory {subdir!r}")
        declared.add(relative.rsplit("/", 1)[1])
    if files != declared:
        raise _fail(f"served model directory files drifted: {sorted(files ^ declared)} are unlisted or missing")
    if others != set(model["other_entries"]):
        raise _fail(f"served model directory entries drifted: {sorted(others ^ set(model['other_entries']))}")


class _VllmRuntimeValidationMixin:
    """``_validate_vllm_runtime``: the gate-time check of a vLLM runtime receipt (module docstring)."""

    def _validate_vllm_runtime(
        self,
        lock: Mapping[str, object] | None = None,
        *,
        full_hash: bool | None = None,
        diagnostics: dict[str, object] | None = None,
    ) -> None:
        runtime = self._load(self.runtime_receipt, "runtime receipt")
        validate_vllm_receipt_document(runtime)
        if runtime["model_lock_key"] != self.expected_model_lock_key:
            raise GateClosed(
                f"runtime receipt model_lock_key {runtime['model_lock_key']!r} does not match "
                f"the model this run declares ({self.expected_model_lock_key!r})"
            )
        lock_entry = self._lock_model_entry(lock)
        subdir = str(lock_entry["local_subdir"]).rstrip("/")
        files = runtime["model"]["files"]
        if len(files) != lock_entry["files"]:
            raise GateClosed(
                f"runtime receipt declares {len(files)} model files but the lock freezes "
                f"{lock_entry['files']}; extra or missing model files are refused"
            )
        verify_model_directory(runtime, Path(self.model_root or self.data_root), subdir)
        # The per-file cheap/full identity loop is the llama path's own, reused unchanged: it is generic over
        # {path, bytes, sha256} entries under the lock's subdir and enforces the lock's byte total.
        self._verify_gguf_files(files, lock_entry, subdir, full_hash=full_hash, diagnostics=diagnostics)
        verify_vllm_runtime_files(runtime)
        if diagnostics is not None:
            diagnostics["vllm_protocol_sha256"] = runtime["protocol_sha256"]
            diagnostics["vllm_instances"] = [instance["instance_id"] for instance in runtime["deployment"]["instances"]]


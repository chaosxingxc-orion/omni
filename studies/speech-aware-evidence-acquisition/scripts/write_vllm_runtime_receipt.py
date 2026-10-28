#!/usr/bin/env python3
"""Write the vLLM runtime receipt (``saea-runtime-receipt-vllm-v1``) from the RUNNING servers it describes.

Owner decision 9.3 (npu-vllm-port, 2026-10-28): "a new vLLM runtime-receipt schema (served weights by file hash,
server argv, package versions, image digest) sits beside the llama one, without loosening it", reached through an
adapter seam (``contracts.runtime_identity`` / ``core.session.runtime_identity``).

Why it is observed, not declared
================================

The llama receipt pins one ~18 KB binary, so it can be written from a build. vLLM's identity is an interpreter, two
editable source trees, the installed package metadata, a pinned environment and a container image -- none of which a
human can transcribe correctly. So this tool reads it from ``/proc`` of the SERVERS THEMSELVES: argv, cwd, the
behaviour-relevant environment, the executable image and the mapped interpreter runtime. The served weights are
re-hashed from disk. Nothing is taken on trust, and nothing is written until the finished document passes BOTH the
gate-time validator (``contracts.gate_runtime_vllm``) and the per-instance live check
(``core.session.vllm_identity.verify_vllm_process_identity``) that every later model touch will re-run.

Protocol vs deployment (coordinator design input, 2026-10-28)
=============================================================

Serving is N instances (e.g. 4 x tensor-parallel 2 on 8 cards), each with its own port and device group. The document
therefore records WHAT each instance computes once (``protocol``: served model name, the interpreter invocation, the
decoding and reasoning flags, cwd, pinned environment -- hashed into ``protocol_sha256``) and WHERE/HOW WIDE each one
runs per instance (``deployment.instances``: host, port, device ids, tensor-parallel size, ``--max-num-seqs``,
``--gpu-memory-utilization``, ``--max-num-batched-tokens`` -- deliberately outside that hash). Every instance named on
the command line must share one protocol; a difference is refused rather than averaged. Adding, moving or resizing an
instance therefore changes the deployment half only, and never looks like a change to what the model computes.

The container image id cannot be proven from inside the container (``/proc/self/mountinfo`` names overlay layers, not
an image digest), so it is ATTESTED by the operator from the host (``docker image inspect --format '{{.Id}}'``) and
recorded; the image's CONTENT is what gets re-verified -- the interpreter, the runtime libraries, the import-path
files, both source trees and every package's METADATA/RECORD, all re-hashed at every gate open.

Usage (inside the serving container, as the uid that runs the servers)::

    scripts/write_vllm_runtime_receipt.py \\
        --instance tp2-a=12345 --instance tp2-b=12456 \\
        --model-lock-key qwen3-omni-30b-a3b-thinking-bf16 \\
        --materialization-root "$SPEECHRL_DATA_DIR" \\
        --image-ref saea-base:20261027 --image-id sha256:<64 hex> \\
        --output docs/receipts/runtime-vllm-bf16-thinking-npu.json

Zero model contacts, zero network, and no process is signalled: every pid named is only READ.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from speech_aware_evidence_acquisition.contracts import STUDY_SLUG  # noqa: E402
from speech_aware_evidence_acquisition.contracts.errors import GateClosed  # noqa: E402
from speech_aware_evidence_acquisition.contracts.gate_runtime_vllm import (  # noqa: E402
    DEVICE_ENV_KEY,
    VLLM_RUNTIME_RECEIPT_SCHEMA,
    device_env_value,
    file_entry,
    import_path_file_names,
    normalize_package_name,
    pinned_environment,
    protocol_sha256,
    read_metadata_identity,
    source_tree_manifest,
    split_vllm_server_argv,
    validate_vllm_receipt_document,
    verify_model_directory,
    verify_vllm_runtime_files,
)
from speech_aware_evidence_acquisition.core.session import SessionError  # noqa: E402
from speech_aware_evidence_acquisition.core.session.vllm_identity import (  # noqa: E402
    verify_vllm_process_identity,
)

DEFAULT_SOURCE_TREES = ("vllm=/vllm-workspace/vllm/vllm", "vllm_ascend=/vllm-workspace/vllm-ascend/vllm_ascend")
DEFAULT_PACKAGES = ("vllm", "vllm_ascend", "torch", "torch_npu", "transformers")
DEFAULT_SITE_PACKAGES = "/usr/local/python3.12.13/lib/python3.12/site-packages"
_LIBPYTHON = re.compile(r"libpython[0-9.]*\.so")
_INSTANCE_SPEC = re.compile(r"(?P<id>[A-Za-z0-9][A-Za-z0-9._-]{0,63})=(?P<pid>[0-9]+)\Z")


class ProducerError(RuntimeError):
    """The receipt cannot be written as asked; nothing is left behind."""


# ---------------------------------------------------------------------------
# Live observation.
# ---------------------------------------------------------------------------


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        raise ProducerError(f"cannot read {path}: {error}") from error


def _readlink(path: Path) -> str:
    try:
        return os.readlink(path)
    except OSError as error:
        raise ProducerError(f"cannot read the symlink {path}: {error}") from error


def observe_process(proc_root: Path, pid: int) -> dict[str, object]:
    """Argv, cwd, environment, executable image and mapped libpython of one running server."""

    process = Path(proc_root) / str(pid)
    raw_argv = (process / "cmdline").read_bytes() if (process / "cmdline").exists() else b""
    if not raw_argv:
        raise ProducerError(f"pid {pid} has no readable command line at {process}/cmdline")
    argv = tuple(part.decode("utf-8") for part in raw_argv.split(b"\0") if part)
    environ: dict[str, str] = {}
    for part in (process / "environ").read_bytes().split(b"\0"):
        if not part:
            continue
        key, separator, value = part.decode("utf-8").partition("=")
        if not separator or not key:
            raise ProducerError(f"pid {pid} has an unparseable environment entry")
        environ[key] = value
    libraries: list[str] = []
    for line in _read_text(process / "maps").splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) < 6:
            continue
        name = fields[5].strip()
        if name.startswith("/") and not name.endswith(" (deleted)") and _LIBPYTHON.search(Path(name).name):
            if name not in libraries:
                libraries.append(name)
    return {
        "pid": pid,
        "argv": argv,
        "environ": environ,
        "cwd": _readlink(process / "cwd"),
        "exe": _readlink(process / "exe"),
        "libpython": libraries,
    }


def protocol_of(observation: dict[str, object]) -> dict[str, object]:
    """The protocol half one running server declares (module docstring, "Protocol vs deployment")."""

    try:
        parsed = split_vllm_server_argv(observation["argv"])
    except GateClosed as error:
        raise ProducerError(f"pid {observation['pid']} is not a pinnable vLLM server: {error}") from error
    return {
        "served_model_name": parsed.served_model_name,
        "argv0": parsed.argv0,
        "entrypoint": list(("-m", "vllm.entrypoints.cli.main", "serve")),
        "server_args": list(parsed.protocol_args),
        "cwd": str(observation["cwd"]),
        "env": pinned_environment(observation["environ"]),
        # Not part of the receipt's protocol section: carried here for the deployment/model fields below.
        "_model_dir": parsed.model_dir,
        "_deployment": dict(parsed.deployment),
    }


def deployment_of(observation: dict[str, object], instance_id: str, deployment: dict[str, str]) -> dict[str, object]:
    """One instance's recorded deployment facts: endpoint, device group, width and memory budget."""

    devices_text = observation["environ"].get(DEVICE_ENV_KEY)
    if not devices_text:
        raise ProducerError(
            f"pid {observation['pid']} does not set {DEVICE_ENV_KEY}; an instance's device group must be explicit, "
            "never inherited from whatever the driver happens to expose"
        )
    try:
        devices = [int(part) for part in devices_text.split(",") if part != ""]
    except ValueError as error:
        raise ProducerError(f"pid {observation['pid']} has an unparseable {DEVICE_ENV_KEY}={devices_text!r}") from error
    if device_env_value(devices) != devices_text:
        raise ProducerError(
            f"pid {observation['pid']} spells {DEVICE_ENV_KEY}={devices_text!r} in a form this study does not pin "
            f"(expected {device_env_value(devices)!r})"
        )
    batched = deployment.get("--max-num-batched-tokens")
    return {
        "instance_id": instance_id,
        "host": deployment["--host"],
        "port": int(deployment["--port"]),
        "device_ids": devices,
        "tensor_parallel_size": int(deployment["--tensor-parallel-size"]),
        "max_num_seqs": int(deployment["--max-num-seqs"]),
        "gpu_memory_utilization": float(deployment["--gpu-memory-utilization"]),
        "max_num_batched_tokens": int(batched) if batched is not None else None,
    }


# ---------------------------------------------------------------------------
# Static halves: packages, trees, weights.
# ---------------------------------------------------------------------------


def package_entry(site_packages: Path, requested: str) -> dict[str, object]:
    """One installed package's pinned metadata identity (``dist-info`` METADATA + RECORD hashes)."""

    normalized = normalize_package_name(requested)
    candidates = sorted(
        entry
        for entry in Path(site_packages).glob("*.dist-info")
        if normalize_package_name(entry.name.split("-")[0]) == normalized
    )
    if len(candidates) != 1:
        raise ProducerError(
            f"package {requested!r} resolves to {len(candidates)} dist-info directories under {site_packages}; "
            "exactly one installation may answer for a pinned package"
        )
    dist_info = candidates[0]
    name, version = read_metadata_identity(dist_info / "METADATA")
    metadata = file_entry(dist_info / "METADATA")
    record = file_entry(dist_info / "RECORD")
    return {
        "name": name,
        "version": version,
        "dist_info": str(dist_info),
        "metadata_sha256": metadata["sha256"],
        "record_sha256": record["sha256"],
    }


def model_section(model_dir: Path, materialization_root: Path) -> tuple[dict[str, object], str]:
    """``(model section, lock subdir)``: every served weight file re-hashed, every other entry named."""

    model_dir = Path(model_dir)
    root = Path(materialization_root)
    try:
        subdir = str(model_dir.relative_to(root))
    except ValueError as error:
        raise ProducerError(
            f"the served model directory {model_dir} is not under the materialization root {root}; the lock's "
            "relative paths could not describe it"
        ) from error
    files: list[dict[str, object]] = []
    others: list[str] = []
    for entry in sorted(model_dir.iterdir()):
        if entry.is_symlink():
            raise ProducerError(f"the served model directory holds a symlink {entry.name!r}; refused")
        if entry.is_file():
            measured = file_entry(entry)
            files.append({"path": f"{subdir}/{entry.name}", "bytes": measured["bytes"], "sha256": measured["sha256"]})
        elif entry.is_dir():
            others.append(entry.name)
        else:
            raise ProducerError(f"the served model directory holds a non-file entry {entry.name!r}; refused")
    if not files:
        raise ProducerError(f"the served model directory {model_dir} holds no files")
    return {"model_dir": str(model_dir), "files": files, "other_entries": sorted(others)}, subdir


# ---------------------------------------------------------------------------
# Document assembly.
# ---------------------------------------------------------------------------


def build_receipt(
    *,
    instances: list[tuple[str, int]],
    model_lock_key: str,
    materialization_root: Path,
    image_ref: str,
    image_id: str,
    site_packages: Path,
    source_trees: list[tuple[str, str]],
    packages: list[str],
    runtime_libraries: list[str],
    proc_root: Path,
) -> dict[str, object]:
    """Observe every instance, assemble the document, then re-verify it exactly as the gate will."""

    observations = [(instance_id, observe_process(proc_root, pid)) for instance_id, pid in instances]
    protocols = [(instance_id, protocol_of(observation)) for instance_id, observation in observations]
    first_id, first = protocols[0]
    for instance_id, protocol in protocols[1:]:
        drift = sorted(key for key in first if key != "_deployment" and protocol[key] != first[key])
        if drift:
            raise ProducerError(
                f"instance {instance_id!r} does not share instance {first_id!r}'s protocol: {drift} differ. One "
                "receipt describes ONE protocol served by N deployments; two protocols need two receipts"
            )
    interpreter_path = str(observations[0][1]["exe"])
    for instance_id, observation in observations[1:]:
        if str(observation["exe"]) != interpreter_path:
            raise ProducerError(
                f"instance {instance_id!r} runs the executable image {observation['exe']!r}, not {interpreter_path!r}"
            )
    libraries = list(runtime_libraries)
    if not libraries:
        for _instance_id, observation in observations:
            for name in observation["libpython"]:
                if name not in libraries:
                    libraries.append(name)
    if not libraries:
        raise ProducerError(
            "no interpreter runtime library (libpython*.so) is mapped by any named server; pass --runtime-library "
            "explicitly for a statically linked interpreter"
        )

    model, subdir = model_section(Path(str(first["_model_dir"])), materialization_root)
    receipt: dict[str, object] = {
        "schema": VLLM_RUNTIME_RECEIPT_SCHEMA,
        "study": STUDY_SLUG,
        "model_lock_key": model_lock_key,
        "container": {"image_ref": image_ref, "image_id": image_id},
        "interpreter": file_entry(Path(interpreter_path)),
        "runtime_libraries": [file_entry(Path(path)) for path in sorted(libraries)],
        "site_packages": str(site_packages),
        "import_path_files": [file_entry(path) for path in import_path_file_names(Path(site_packages))],
        "source_trees": [
            {"package": package, "root": root, **source_tree_manifest(Path(root))}
            for package, root in sorted(source_trees)
        ],
        "packages": [package_entry(Path(site_packages), name) for name in sorted(set(packages))],
        "model": model,
        "protocol": {key: value for key, value in first.items() if not key.startswith("_")},
        "deployment": {
            "instances": [
                deployment_of(observation, instance_id, dict(protocol["_deployment"]))
                for (instance_id, observation), (_same_id, protocol) in zip(observations, protocols)
            ]
        },
        "protocol_sha256": "0" * 64,
    }
    receipt["protocol_sha256"] = protocol_sha256(receipt)

    # The three checks every later model touch runs, run here so a receipt is never written unverifiable.
    validate_vllm_receipt_document(receipt)
    verify_vllm_runtime_files(receipt)
    verify_model_directory(receipt, Path(materialization_root), subdir)
    for (instance_id, observation), instance in zip(observations, receipt["deployment"]["instances"]):
        verify_vllm_process_identity(
            int(observation["pid"]),
            receipt,
            instance,
            host=str(instance["host"]),
            port=int(instance["port"]),
            argv=observation["argv"],
            proc_root=proc_root,
        )
    return receipt


def write_document(path: Path, receipt: dict[str, object]) -> None:
    """Create-only write: a runtime receipt is never silently replaced."""

    raw = json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as error:
        raise ProducerError(
            f"{path} already exists; a runtime receipt is replaced by a reviewed commit, never by this tool"
        ) from error
    except OSError as error:
        raise ProducerError(f"cannot create {path}: {error}") from error
    with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(raw)


def parse_instance(spec: str) -> tuple[str, int]:
    match = _INSTANCE_SPEC.fullmatch(spec)
    if match is None:
        raise argparse.ArgumentTypeError(f"--instance must be <instance-id>=<pid>, got {spec!r}")
    return match.group("id"), int(match.group("pid"))


def parse_source_tree(spec: str) -> tuple[str, str]:
    package, separator, root = spec.partition("=")
    if not separator or not package or not root.startswith("/"):
        raise argparse.ArgumentTypeError(f"--source-tree must be <package>=<absolute root>, got {spec!r}")
    return package, root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--instance", action="append", required=True, type=parse_instance, metavar="ID=PID",
        help="one running server: a stable instance id and its pid (repeat for N instances)",
    )
    parser.add_argument("--model-lock-key", required=True, help="the umbrella-lock entry the served weights come from")
    parser.add_argument(
        "--materialization-root", required=True, type=Path,
        help="the root the lock's relative model paths are under (the fragment's model_materialization_root)",
    )
    parser.add_argument("--image-ref", required=True, help="the container image tag the servers and harness run in")
    parser.add_argument(
        "--image-id", required=True,
        help="that image's id, attested from the host: docker image inspect --format '{{.Id}}' <ref>",
    )
    parser.add_argument("--site-packages", type=Path, default=Path(DEFAULT_SITE_PACKAGES))
    parser.add_argument("--source-tree", action="append", type=parse_source_tree, metavar="PACKAGE=ROOT")
    parser.add_argument("--package", action="append", metavar="NAME")
    parser.add_argument(
        "--runtime-library", action="append", metavar="PATH",
        help="pin this library explicitly instead of the libpython the servers have mapped",
    )
    parser.add_argument("--proc-root", type=Path, default=Path("/proc"), help="for tests against a fake /proc")
    parser.add_argument("--output", required=True, type=Path, help="where to create the receipt (never overwritten)")
    args = parser.parse_args(argv)

    source_trees = args.source_tree or [parse_source_tree(spec) for spec in DEFAULT_SOURCE_TREES]
    packages = args.package or list(DEFAULT_PACKAGES)
    seen = {instance_id for instance_id, _pid in args.instance}
    if len(seen) != len(args.instance):
        raise ProducerError("each --instance needs its own instance id")
    try:
        receipt = build_receipt(
            instances=args.instance,
            model_lock_key=args.model_lock_key,
            materialization_root=args.materialization_root,
            image_ref=args.image_ref,
            image_id=args.image_id,
            site_packages=args.site_packages,
            source_trees=source_trees,
            packages=packages,
            runtime_libraries=args.runtime_library or [],
            proc_root=args.proc_root,
        )
    except (ProducerError, GateClosed, SessionError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    write_document(args.output, receipt)
    instances = ", ".join(
        f"{instance['instance_id']}@{instance['host']}:{instance['port']} cards {instance['device_ids']}"
        for instance in receipt["deployment"]["instances"]
    )
    print(f"wrote {args.output}")
    print(f"  protocol_sha256 {receipt['protocol_sha256']}")
    print(f"  served          {receipt['protocol']['served_model_name']} from {receipt['model']['model_dir']}")
    print(f"  weights         {len(receipt['model']['files'])} files")
    print(f"  instances       {instances}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ProducerError as error:  # argument-level refusals
        print(f"refused: {error}", file=sys.stderr)
        raise SystemExit(2) from error

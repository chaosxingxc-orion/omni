"""The deployment layer: how many resident vLLM instances serve the protocol, on which cards, and nothing else.

Owner decision (npu-vllm-port, 2026-10-28). The serving facts split in two:

protocol    (hashed) the weights, the 24,576-token per-sequence window, the decoding parameters and the reasoning
            budget -- everything the model fragment's ``server_args``/``decoding_params`` say about WHAT is computed;
deployment  (recorded in the run manifest and the runtime receipt, NOT hashed) the instance count, the tensor-parallel
            size per instance, ``--max-num-seqs`` per instance, ``--gpu-memory-utilization``, the card assignment, the
            ports and the container names -- everything about WHERE and HOW MANY AT ONCE.

A deployment change must therefore never look like a protocol change, and a protocol change must never hide inside a
deployment profile. This module is the one place the two meet:

* :func:`load_deployment_profile` reads ``configs/deployment/<profile-id>.json`` (schema
  :data:`DEPLOYMENT_PROFILE_SCHEMA`) and refuses anything it does not recognise: unknown keys, overlapping cards,
  ports, names or cpusets, a tensor-parallel size that differs from an instance's card count, a model path that is not
  a plain path relative to the data root. The profile names ONE absolute path, ``data_mount`` (the data root, mounted
  at the same path in every container); the weights are ``data_mount/model_relpath``, and an operator moving the data
  root overrides it in one place (``data_root=`` / ``--data-root``) without editing the profile. ``host`` is the
  ``hostname`` the launcher must be running on. The profile carries no free-form server flags, so it cannot smuggle a
  protocol flag in.
* :func:`allocate_cards` takes ``npu-smi info`` rows parsed by the study's own
  :func:`~.session.telemetry.parse_npu_smi_table` (plus the process rows :func:`npu_smi_process_cards` reads) and
  REFUSES every card that is absent from the table, not healthy, above the profile's busy threshold, or holding a
  process. A refusal names every refused card and why; nothing is ever started on a card it could not prove free.
* :func:`instance_server_argv` derives one instance's ``vllm serve`` argv: the deployment flags this module owns,
  then the fragment's ``server_args`` passed through :func:`~.session.serving.merge_server_args` -- the SAME merge a
  serving session and a flight use -- with only ``--max-num-seqs`` taken from the profile. ``--max-model-len`` stays
  the fragment's own window (it is protocol), and a fragment that carries a deployment flag is refused (one owner per
  fact).
* :func:`deployment_record` yields the canonical recorded facts for a manifest or receipt.

vLLM has no slots: nothing here reads or writes llama.cpp ``id_slot``. Per-trajectory affinity across instances is the
client's routing decision (instance ``slot mod instance_count``), not a server flag.

Run inside the ``saea-base`` image (``python -m speech_aware_evidence_acquisition.core.deployment``): ``plan`` prints
the exact docker commands, ``emit`` writes them NUL-delimited for the host launcher (``work/scripts/npu_deploy.sh``)
after the card check passes, ``instances`` lists name/port/cards, ``record`` prints :func:`deployment_record`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence

from .config import read_fragment
from .session.serving import declared_server_geometry, merge_server_args
from .session.telemetry import parse_npu_smi_table

DEPLOYMENT_PROFILE_SCHEMA = "saea-deployment-profile-v1"
DEPLOYMENT_RECORD_SCHEMA = "saea-deployment-record-v1"
#: ``<repo>/configs/deployment``; NOT one of :func:`~.config.compose`'s fragment kinds -- a profile never enters a
#: composed config, so it can never enter the protocol hash.
DEPLOYMENT_CONFIG_DIR = ("configs", "deployment")
REPO_ROOT = Path(__file__).resolve().parents[3]

_PROFILE_KEYS = frozenset({
    "schema", "profile_id", "host", "image", "model_fragment", "model_relpath", "bind_host", "data_mount",
    "tensor_parallel_size", "max_num_seqs", "gpu_memory_utilization", "max_num_batched_tokens",
    "busy_threshold_mib", "host_card_count", "host_cpu_count", "shm_size", "instances", "basis",
})
_INSTANCE_KEYS = frozenset({"name", "cards", "port", "cpuset", "env"})
_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9.-]*")
_NAME_PATTERN = re.compile(r"saea-[a-z0-9][a-z0-9-]*")
_ENV_NAME_PATTERN = re.compile(r"[A-Z][A-Z0-9_]*")
_SHM_PATTERN = re.compile(r"[1-9][0-9]*[mg]")

#: Flags this module owns. A model fragment that carries one is refused: the fact would have two owners, and a
#: deployment value would silently enter the protocol hash (or a protocol value would be overridden per instance).
DEPLOYMENT_ONLY_FLAGS = frozenset({
    "--served-model-name", "--host", "--port", "--tensor-parallel-size", "-tp", "--pipeline-parallel-size", "-pp",
    "--data-parallel-size", "-dp", "--gpu-memory-utilization", "--max-num-batched-tokens",
})
#: The Ascend control devices every NPU container needs beside its own ``/dev/davinci<N>`` cards (as ``npu_serve.sh``).
ASCEND_CONTROL_DEVICES = ("/dev/davinci_manager", "/dev/devmm_svm", "/dev/hisi_hdc")
#: Host paths mounted read-only so ``npu-smi`` inside the container reads exactly the mounted cards.
ASCEND_READONLY_MOUNTS = ("/usr/local/Ascend/driver", "/usr/local/sbin/npu-smi")
#: Environment every instance gets: the NPU hosts have no internet, so a hub lookup must fail at once, not hang.
COMMON_ENV = (("HF_HUB_OFFLINE", "1"), ("TRANSFORMERS_OFFLINE", "1"))
#: docker's json-file log grows without bound for a resident container; cap it (``docker logs`` keeps the tail).
LOG_OPTS = ("max-size=200m", "max-file=5")
LABEL_PROFILE = "saea.deployment.profile"
LABEL_PROFILE_SHA256 = "saea.deployment.profile_sha256"
LABEL_INSTANCE = "saea.deployment.instance"


class DeploymentError(ValueError):
    """A deployment profile, card allocation or derived argv the study refuses."""


class CardRefusal(DeploymentError):
    """Some requested card is not provably free. ``refusals`` maps each refused card to its reasons."""

    def __init__(self, message: str, refusals: Mapping[int, Sequence[str]]) -> None:
        super().__init__(message)
        self.refusals = {int(card): tuple(reasons) for card, reasons in refusals.items()}


@dataclass(frozen=True)
class InstanceSpec:
    index: int
    name: str
    cards: tuple[int, ...]
    port: int
    cpuset: str
    env: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class DeploymentProfile:
    profile_id: str
    host: str
    image: str
    model_fragment: str
    model_relpath: str
    model_path: str
    bind_host: str
    data_mount: str
    tensor_parallel_size: int
    max_num_seqs: int
    gpu_memory_utilization: float
    max_num_batched_tokens: int | None
    busy_threshold_mib: int
    host_card_count: int
    host_cpu_count: int
    shm_size: str
    instances: tuple[InstanceSpec, ...]
    basis: str
    sha256: str

    def base_url(self, instance: InstanceSpec) -> str:
        return f"http://{self.bind_host}:{instance.port}"

    @property
    def cards(self) -> tuple[int, ...]:
        return tuple(card for instance in self.instances for card in instance.cards)


# ---------------------------------------------------------------------------
# Profile schema.
# ---------------------------------------------------------------------------


def _canonical_sha256(document: object) -> str:
    blob = json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _int(value: object, field: str, *, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum or (maximum is not None and value > maximum):
        bound = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
        raise DeploymentError(f"deployment profile {field} must be an integer {bound}; got {value!r}")
    return value


def _str(value: object, field: str, pattern: re.Pattern[str] | None = None) -> str:
    if not isinstance(value, str) or not value.strip() or (pattern is not None and not pattern.fullmatch(value)):
        raise DeploymentError(f"deployment profile {field} is not a valid string: {value!r}")
    return value


def parse_cpuset(text: object) -> frozenset[int]:
    """The CPUs of a docker ``--cpuset-cpus`` string (``"0-63"``, ``"0-31,64-95"``); refuses empty, reversed or
    malformed ranges rather than guessing."""

    if not isinstance(text, str) or not re.fullmatch(r"\d+(-\d+)?(,\d+(-\d+)?)*", text):
        raise DeploymentError(f"cpuset {text!r} is not a docker --cpuset-cpus list")
    cpus: set[int] = set()
    for part in text.split(","):
        low, _, high = part.partition("-")
        first, last = int(low), int(high or low)
        if last < first:
            raise DeploymentError(f"cpuset {text!r} has a reversed range {part!r}")
        cpus.update(range(first, last + 1))
    return frozenset(cpus)


def _absolute(value: object, field: str) -> PurePosixPath:
    path = PurePosixPath(_str(value, field))
    if not path.is_absolute() or ".." in path.parts:
        raise DeploymentError(f"deployment profile {field} must be an absolute path without '..': {value!r}")
    return path


def validate_deployment_profile(
    document: object, *, source: str = "?", data_root: str | Path | None = None
) -> DeploymentProfile:
    """The typed profile, or :class:`DeploymentError` naming the first violated rule (module docstring).

    ``data_root`` replaces the profile's ``data_mount`` (the one absolute path) when the data root has moved; the
    profile's ``sha256`` stays the hash of the document as written, and the record carries the mount actually used.
    """

    if not isinstance(document, dict):
        raise DeploymentError(f"deployment profile {source} must be a JSON object")
    unknown, missing = sorted(set(document) - _PROFILE_KEYS), sorted(_PROFILE_KEYS - set(document))
    if unknown or missing:
        raise DeploymentError(f"deployment profile {source}: unknown keys {unknown}, missing keys {missing}")
    if document["schema"] != DEPLOYMENT_PROFILE_SCHEMA:
        raise DeploymentError(f"deployment profile {source} schema must be {DEPLOYMENT_PROFILE_SCHEMA!r}")
    tp = _int(document["tensor_parallel_size"], "tensor_parallel_size", minimum=1)
    host_cards = _int(document["host_card_count"], "host_card_count", minimum=1)
    host_cpus = _int(document["host_cpu_count"], "host_cpu_count", minimum=1)
    utilization = document["gpu_memory_utilization"]
    if isinstance(utilization, bool) or not isinstance(utilization, (int, float)) or not 0 < utilization <= 1:
        raise DeploymentError(f"deployment profile gpu_memory_utilization must be in (0, 1]; got {utilization!r}")
    batched = document["max_num_batched_tokens"]
    if batched is not None:
        batched = _int(batched, "max_num_batched_tokens", minimum=1)
    data_mount = _absolute(document["data_mount"] if data_root is None else str(data_root), "data_mount")
    relpath = PurePosixPath(_str(document["model_relpath"], "model_relpath"))
    if relpath.is_absolute() or ".." in relpath.parts or not relpath.parts:
        raise DeploymentError(f"model_relpath must be a plain path relative to the data root: {relpath}")
    raw_instances = document["instances"]
    if not isinstance(raw_instances, list) or not raw_instances:
        raise DeploymentError("deployment profile instances must be a non-empty list")
    instances: list[InstanceSpec] = []
    seen_cards: set[int] = set()
    seen_cpus: set[int] = set()
    seen_ports: set[int] = set()
    seen_names: set[str] = set()
    for index, raw in enumerate(raw_instances):
        where = f"instances[{index}]"
        if not isinstance(raw, dict) or set(raw) != _INSTANCE_KEYS:
            raise DeploymentError(f"deployment profile {where} must have exactly the keys {sorted(_INSTANCE_KEYS)}")
        name = _str(raw["name"], f"{where}.name", _NAME_PATTERN)
        port = _int(raw["port"], f"{where}.port", minimum=1024, maximum=65535)
        cards = raw["cards"]
        if not isinstance(cards, list) or not cards:
            raise DeploymentError(f"deployment profile {where}.cards must be a non-empty list")
        cards = tuple(_int(card, f"{where}.cards", minimum=0, maximum=host_cards - 1) for card in cards)
        if len(cards) != tp:
            raise DeploymentError(f"{where} has {len(cards)} card(s) but tensor_parallel_size is {tp}")
        if list(cards) != sorted(set(cards)):
            raise DeploymentError(f"{where}.cards must be strictly increasing: {list(cards)}")
        cpus = parse_cpuset(raw["cpuset"])
        if max(cpus) >= host_cpus:
            raise DeploymentError(f"{where}.cpuset {raw['cpuset']!r} exceeds host_cpu_count {host_cpus}")
        env = raw["env"]
        if not isinstance(env, dict) or not all(
            isinstance(key, str) and _ENV_NAME_PATTERN.fullmatch(key) and isinstance(value, str)
            for key, value in env.items()
        ):
            raise DeploymentError(f"{where}.env must map UPPER_CASE names to strings")
        if set(env) & {key for key, _ in COMMON_ENV}:
            raise DeploymentError(f"{where}.env must not override the common environment {[k for k, _ in COMMON_ENV]}")
        for kind, value, seen in (("card", set(cards), seen_cards), ("cpu", set(cpus), seen_cpus),
                                  ("port", {port}, seen_ports), ("name", {name}, seen_names)):
            overlap = sorted(value & seen)
            if overlap:
                raise DeploymentError(f"{where} reuses {kind}(s) {overlap} of an earlier instance")
            seen.update(value)
        instances.append(InstanceSpec(index, name, cards, port, raw["cpuset"], tuple(sorted(env.items()))))
    return DeploymentProfile(
        profile_id=_str(document["profile_id"], "profile_id", _ID_PATTERN),
        host=_str(document["host"], "host"),
        image=_str(document["image"], "image"),
        model_fragment=_str(document["model_fragment"], "model_fragment"),
        model_relpath=str(relpath),
        model_path=str(data_mount / relpath),
        bind_host=_str(document["bind_host"], "bind_host"),
        data_mount=str(data_mount),
        tensor_parallel_size=tp,
        max_num_seqs=_int(document["max_num_seqs"], "max_num_seqs", minimum=1),
        gpu_memory_utilization=float(utilization),
        max_num_batched_tokens=batched,
        busy_threshold_mib=_int(document["busy_threshold_mib"], "busy_threshold_mib", minimum=0),
        host_card_count=host_cards,
        host_cpu_count=host_cpus,
        shm_size=_str(document["shm_size"], "shm_size", _SHM_PATTERN),
        instances=tuple(instances),
        basis=_str(document["basis"], "basis"),
        sha256=_canonical_sha256(document),
    )


def resolve_profile_path(profile: str | Path, repo_root: Path = REPO_ROOT) -> Path:
    """A path to an existing file, or a bare profile id resolved under ``configs/deployment``."""

    candidate = Path(profile)
    if candidate.suffix == ".json" or candidate.is_file():
        return candidate
    if not _ID_PATTERN.fullmatch(str(profile)):
        raise DeploymentError(f"not a deployment profile id or path: {profile!r}")
    return Path(repo_root).joinpath(*DEPLOYMENT_CONFIG_DIR, f"{profile}.json")


def load_deployment_profile(
    profile: str | Path, repo_root: Path = REPO_ROOT, *, data_root: str | Path | None = None
) -> DeploymentProfile:
    """Read and validate one profile; its ``profile_id`` must equal its file stem (one name, one file)."""

    path = resolve_profile_path(profile, repo_root)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise DeploymentError(f"cannot read deployment profile {path}: {error}") from error
    loaded = validate_deployment_profile(document, source=str(path), data_root=data_root)
    if loaded.profile_id != path.stem:
        raise DeploymentError(f"deployment profile {path} declares profile_id {loaded.profile_id!r}")
    return loaded


# ---------------------------------------------------------------------------
# Card allocation.
# ---------------------------------------------------------------------------

_NPU_PROCESS_ROW = re.compile(r"^\|\s*(\d+)\s+\d+\s*\|\s*(\d+)\s*\|\s*\S")


def npu_smi_process_cards(text: str) -> frozenset[int]:
    """The NPU ids that hold at least one process in ``npu-smi info``'s process table (the section after the
    ``Process id`` header). A card with a resident process is busy even when its HBM reading is low."""

    cards: set[int] = set()
    in_processes = False
    for line in text.splitlines():
        if "Process id" in line:
            in_processes = True
            continue
        match = _NPU_PROCESS_ROW.match(line) if in_processes else None
        if match:
            cards.add(int(match.group(1)))
    return frozenset(cards)


def card_refusals(
    cards: Iterable[int],
    rows: Sequence[Mapping[str, object]],
    *,
    busy_threshold_mib: float,
    process_cards: Iterable[int] = (),
) -> dict[int, list[str]]:
    """Every requested card that is not provably free, with all of its reasons (empty when all are free)."""

    by_id = {int(row["npu_id"]): row for row in rows}  # type: ignore[call-overload]
    holding = set(process_cards)
    refusals: dict[int, list[str]] = {}
    for card in cards:
        row = by_id.get(card)
        if row is None:
            refusals[card] = ["absent from the npu-smi table"]
            continue
        reasons = []
        if row.get("health") != "OK":
            reasons.append(f"health {row.get('health')!r}")
        used = row.get("memory_used_mib")
        if not isinstance(used, (int, float)) or used > busy_threshold_mib:
            reasons.append(f"HBM used {used} MiB > {busy_threshold_mib} MiB")
        if card in holding:
            reasons.append("holds a process")
        if reasons:
            refusals[card] = reasons
    return refusals


def allocate_cards(
    profile: DeploymentProfile,
    rows: Sequence[Mapping[str, object]],
    *,
    process_cards: Iterable[int] = (),
    instances: Sequence[InstanceSpec] | None = None,
) -> dict[str, tuple[int, ...]]:
    """``{instance name: cards}`` for ``instances`` (default: all of the profile's), or :class:`CardRefusal` listing
    every refused card. All-or-nothing: one busy card refuses the whole allocation, so a half-deployed profile never
    exists by accident."""

    chosen = tuple(profile.instances if instances is None else instances)
    wanted = [card for instance in chosen for card in instance.cards]
    refusals = card_refusals(
        wanted, rows, busy_threshold_mib=profile.busy_threshold_mib, process_cards=process_cards
    )
    if refusals:
        detail = "; ".join(f"NPU {card}: {', '.join(reasons)}" for card, reasons in sorted(refusals.items()))
        raise CardRefusal(f"refused: {len(refusals)} card(s) not free for {profile.profile_id}: {detail}", refusals)
    return {instance.name: instance.cards for instance in chosen}


def allocate_from_npu_smi(profile: DeploymentProfile, text: str, **kwargs) -> dict[str, tuple[int, ...]]:
    """:func:`allocate_cards` straight from ``npu-smi info`` text. An unparseable table is a refusal of every card."""

    return allocate_cards(profile, parse_npu_smi_table(text), process_cards=npu_smi_process_cards(text), **kwargs)


def instances_for_cards(profile: DeploymentProfile, cards_wanted: int) -> tuple[InstanceSpec, ...]:
    """The first instances that together hold exactly ``cards_wanted`` cards (an experiment declares cards: smoke 2,
    flight 8). Refused when the count is not a whole number of instances or exceeds the deployment."""

    tp = profile.tensor_parallel_size
    if isinstance(cards_wanted, bool) or not isinstance(cards_wanted, int) or cards_wanted <= 0 or cards_wanted % tp:
        raise DeploymentError(f"{cards_wanted!r} card(s) is not a whole number of tensor-parallel-{tp} instances")
    count = cards_wanted // tp
    if count > len(profile.instances):
        raise DeploymentError(
            f"{cards_wanted} card(s) requested but {profile.profile_id} deploys {len(profile.cards)}"
        )
    return profile.instances[:count]


# ---------------------------------------------------------------------------
# argv derivation.
# ---------------------------------------------------------------------------


def model_fragment_values(profile: DeploymentProfile, repo_root: Path = REPO_ROOT) -> dict[str, object]:
    """The profile's model fragment, read exactly as :func:`~.config.compose` reads it; it must be a vLLM fragment."""

    values = read_fragment(Path(repo_root), "model", profile.model_fragment)
    if values.get("transport") != "vllm":
        raise DeploymentError(f"model fragment {profile.model_fragment!r} is not a vllm fragment")
    if not isinstance(values.get("served_model_name"), str) or not values["served_model_name"]:
        raise DeploymentError(f"model fragment {profile.model_fragment!r} declares no served_model_name")
    return values


def instance_server_argv(
    profile: DeploymentProfile, instance: InstanceSpec, fragment: Mapping[str, object]
) -> list[str]:
    """The arguments after ``vllm`` for one instance: ``serve <model>``, the deployment flags, then the fragment's
    protocol ``server_args`` with ``--max-num-seqs`` set from the profile (``--max-model-len`` untouched)."""

    server_args = fragment.get("server_args") or []
    flags = {str(token).split("=", 1)[0] for token in server_args}  # type: ignore[union-attr]
    owned = sorted(DEPLOYMENT_ONLY_FLAGS & flags)
    if owned:
        raise DeploymentError(f"model fragment {profile.model_fragment!r} carries deployment flags {owned}")
    _, window = declared_server_geometry(server_args, transport="vllm", model_fragment=profile.model_fragment)
    protocol = merge_server_args(
        server_args, slots=profile.max_num_seqs, context_tokens=profile.max_num_seqs * window, transport="vllm"
    )
    deployment = [
        "--served-model-name", str(fragment["served_model_name"]),
        "--host", profile.bind_host,
        "--port", str(instance.port),
        "--tensor-parallel-size", str(profile.tensor_parallel_size),
        "--gpu-memory-utilization", repr(profile.gpu_memory_utilization),
    ]
    if profile.max_num_batched_tokens is not None:
        deployment += ["--max-num-batched-tokens", str(profile.max_num_batched_tokens)]
    return ["serve", profile.model_path, *deployment, *protocol]


def docker_run_argv(
    profile: DeploymentProfile, instance: InstanceSpec, fragment: Mapping[str, object]
) -> list[str]:
    """The full detached, named, resident ``docker run`` for one instance: only its own cards, host network, its own
    cpuset, the data root bind-mounted at the same path, ``--restart unless-stopped``, deployment labels so ``down``
    can prove a container is ours before stopping it."""

    argv = [
        "docker", "run", "--detach", "--name", instance.name,
        "--label", f"{LABEL_PROFILE}={profile.profile_id}",
        "--label", f"{LABEL_PROFILE_SHA256}={profile.sha256}",
        "--label", f"{LABEL_INSTANCE}={instance.index}",
        "--restart", "unless-stopped", "--network", "host", "--shm-size", profile.shm_size,
        "--cpuset-cpus", instance.cpuset,
    ]
    for option in LOG_OPTS:
        argv += ["--log-opt", option]
    for device in (*ASCEND_CONTROL_DEVICES, *(f"/dev/davinci{card}" for card in instance.cards)):
        argv += ["--device", device]
    for mount in ASCEND_READONLY_MOUNTS:
        argv += ["-v", f"{mount}:{mount}:ro"]
    argv += ["-v", f"{profile.data_mount}:{profile.data_mount}"]
    for key, value in (*COMMON_ENV, *instance.env):
        argv += ["-e", f"{key}={value}"]
    argv += ["--entrypoint", "vllm", profile.image]
    return argv + instance_server_argv(profile, instance, fragment)


def deployment_record(profile: DeploymentProfile, repo_root: Path = REPO_ROOT) -> dict[str, object]:
    """The canonical recorded (NOT hashed) deployment facts for a run manifest or runtime receipt.

    ``server_argv_sha256`` pins each instance's exact ``vllm serve`` argv so a receipt can tell two deployments of the
    same profile apart if the fragment changed under it; ``protocol_window_tokens`` is copied from the fragment only so
    a reader can see the deployment did not change it.
    """

    fragment = model_fragment_values(profile, repo_root)
    _, window = declared_server_geometry(
        fragment.get("server_args") or [], transport="vllm", model_fragment=profile.model_fragment
    )
    return {
        "schema": DEPLOYMENT_RECORD_SCHEMA,
        "hashed": False,
        "profile_id": profile.profile_id,
        "profile_sha256": profile.sha256,
        "host": profile.host,
        "image": profile.image,
        "model_fragment": profile.model_fragment,
        "served_model_name": fragment["served_model_name"],
        "data_mount": profile.data_mount,
        "model_relpath": profile.model_relpath,
        "model_path": profile.model_path,
        "protocol_window_tokens": window,
        "instance_count": len(profile.instances),
        "tensor_parallel_size": profile.tensor_parallel_size,
        "max_num_seqs_per_instance": profile.max_num_seqs,
        "gpu_memory_utilization": profile.gpu_memory_utilization,
        "max_num_batched_tokens": profile.max_num_batched_tokens,
        "card_count": len(profile.cards),
        "instances": [
            {
                "index": instance.index,
                "name": instance.name,
                "cards": list(instance.cards),
                "port": instance.port,
                "base_url": profile.base_url(instance),
                "cpuset": instance.cpuset,
                "env": dict(instance.env),
                "server_argv_sha256": hashlib.sha256(
                    json.dumps(instance_server_argv(profile, instance, fragment)).encode("utf-8")
                ).hexdigest(),
            }
            for instance in profile.instances
        ],
    }


# ---------------------------------------------------------------------------
# CLI (run inside the saea-base image; the host launcher executes what it prints).
# ---------------------------------------------------------------------------


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m speech_aware_evidence_acquisition.core.deployment")
    parser.add_argument("command", choices=("plan", "emit", "instances", "record"))
    parser.add_argument("profile", help="profile id under configs/deployment, or a path to a profile JSON")
    parser.add_argument("--npu-smi-file", help="captured `npu-smi info` text ('-' for stdin); emit requires it")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--data-root", help="override the profile's data_mount (the data root moved)")
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root)
    profile = load_deployment_profile(args.profile, repo_root, data_root=args.data_root)
    table = None
    if args.npu_smi_file:
        table = sys.stdin.read() if args.npu_smi_file == "-" else Path(args.npu_smi_file).read_text(encoding="utf-8")
    if args.command == "record":
        print(json.dumps(deployment_record(profile, repo_root), indent=2, sort_keys=True))
        return 0
    if args.command == "instances":
        for instance in profile.instances:
            print("\t".join([instance.name, str(instance.port), ",".join(map(str, instance.cards)),
                             profile.base_url(instance)]))
        return 0
    fragment = model_fragment_values(profile, repo_root)
    commands = [(instance, docker_run_argv(profile, instance, fragment)) for instance in profile.instances]
    if args.command == "emit":
        if table is None:
            raise DeploymentError("emit requires --npu-smi-file: nothing is started without a card check")
        allocate_from_npu_smi(profile, table)
        out = sys.stdout.buffer
        for instance, command in commands:
            for token in (instance.name, str(len(command)), *command):
                if "\0" in token:
                    raise DeploymentError(f"argv token contains NUL: {token!r}")
                out.write(token.encode("utf-8") + b"\0")
        out.flush()
        return 0
    print(f"# deployment profile {profile.profile_id} (sha256 {profile.sha256})")
    print(f"# host {profile.host}; data root {profile.data_mount}; weights {profile.model_path}")
    print(f"# {len(profile.instances)} instance(s) x TP{profile.tensor_parallel_size}; "
          f"max-num-seqs {profile.max_num_seqs}/instance; gpu-memory-utilization {profile.gpu_memory_utilization}")
    if table is not None:
        try:
            allocate_from_npu_smi(profile, table)
            print("# card check: every card is free")
        except CardRefusal as refusal:
            print(f"# card check: {refusal}")
    for instance, command in commands:
        print(f"\n# {instance.name}: cards {list(instance.cards)} port {instance.port} cpuset {instance.cpuset}")
        print(shlex.join(command))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(_main())
    except DeploymentError as error:
        sys.exit(str(error))

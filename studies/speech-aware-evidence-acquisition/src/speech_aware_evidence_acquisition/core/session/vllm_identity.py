"""The ``vllm`` runtime identity, live half (owner decision 9.2, 2026-10-28: the server runs in the SAME container as
the harness, and the pinned "binary" is the interpreter plus the vLLM / vllm-ascend source trees plus the image).

What this module is for
=======================

The llama path proves, on every send, that a LIVE process with the pinned identity owns the listening socket
(``core.session.assert_session_alive``: receipt agreement, pid liveness, ``/proc/<pid>/exe`` re-hashed against the
pinned ~18 KB binary, and address-aware socket ownership). vLLM has no such binary: the executable image is a generic
Python interpreter, and the code that actually serves the model is a pair of editable source trees plus the pinned
container image. Re-hashing ``/proc/<pid>/exe`` alone would prove only "some Python is running".

So the same question is answered from the same ground truth, at the same strictness, with the parts the pinned identity
actually has (``contracts.gate_runtime_vllm`` re-verified every one of them from bytes when the gate opened):

``/proc/<pid>/stat``     the pid is a live process, not a zombie.
``/proc/<pid>/ns/{mnt,net}``  it shares this process's mount and network namespaces -- i.e. it is in THIS container, so
                         the files the gate hashed are the files it loaded, and the ``/proc/net/tcp`` this module reads
                         is its own network. This is the check that makes "same container as the harness" provable
                         rather than assumed.
``/proc/<pid>/exe``      the executable image is the pinned interpreter file, re-hashed right now (never path-equality).
``/proc/<pid>/cmdline``  the live command line is EXACTLY the receipt's protocol plus this instance's deployment flags
                         (``assert_argv_matches_instance``) and exactly what the session receipt recorded.
``/proc/<pid>/cwd``      the working directory is the pinned one (the entrypoint is ``-m``: cwd is import path).
``/proc/<pid>/environ``  every behaviour-relevant variable equals the receipt's pinned environment, the device
                         variable equals this instance's own device group, and the per-instance deployment variables
                         (``DEPLOYMENT_ENV_KEYS``, e.g. the HCCL base port) equal this instance's own recorded
                         ``env`` -- exactly, so a missing, extra or different one is refused.
``/proc/<pid>/maps``     every pinned runtime library is mapped, no OTHER libpython is mapped (a second interpreter
                         runtime would make the re-hashed ``exe`` meaningless), and nothing mapped under a pinned root
                         was deleted from disk after being loaded.
``/proc/net/tcp``(6) + ``/proc/<pid>/fd``  the pid owns the LISTEN socket on this instance's exact (address, port) --
                         the llama path's own address-aware proof, ported to a caller-supplied ``/proc`` root.

Namespace policy (optional ``shared-pid-host``; default ``same-container``, unchanged): the receipt's
``deployment.namespace_policy``, set from the deployment profile, replaces the namespace line above by: same host, pid
visible, pinned executable and command line, and an endpoint that answers ``/v1/models`` with the recorded model.

Multi-instance (coordinator design input, 2026-10-28)
=====================================================

Serving is N instances (e.g. 4 x tensor-parallel 2 on 8 cards), each with its own port and device group. Nothing here
assumes one server or one socket: every check takes ONE session receipt and resolves its endpoint to ONE receipt
instance (``instance_for_endpoint``), and :func:`assert_vllm_sessions_alive` verifies a list of sessions and refuses a
list that reuses an endpoint or a pid. A second instance plugs in as one more entry in the receipt's
``deployment.instances`` and one more session receipt; the protocol half of the receipt (and its ``protocol_sha256``)
does not move, so a deployment change never looks like a protocol change.

Session receipts
================

A vLLM session is recorded in the UNCHANGED ``SessionReceipt`` v1 schema (role ``legacy``): ``binary_path`` /
``binary_sha256`` are the pinned interpreter (relative-path spelling, as the schema requires), ``argv`` is the live
command line, and ``model_files`` is the runtime receipt's served-weight set. A receipt whose binary or model identity
disagrees with the runtime receipt is worthless paper here exactly as it is on the llama path.
"""

from __future__ import annotations

import hashlib
import os
import re
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Callable, Mapping, Sequence

from ...contracts import STUDY_SLUG, assert_execution_environment
from ...contracts.errors import GateClosed
from ...contracts.gate_runtime_vllm import (
    DEFAULT_NAMESPACE_POLICY,
    DEVICE_ENV_KEY,
    NAMESPACE_POLICY_SHARED_PID_HOST,
    SERVED_MODEL_NAME_FLAG,
    assert_argv_matches_instance,
    classify_server_flags,
    device_env_value,
    deployment_environment,
    harness_hostname,
    instance_argv,
    instance_deployment_env,
    instance_for_endpoint,
    namespace_policy,
    pinned_environment,
    validate_vllm_receipt_document,
)
from . import SessionError, _bracketed, require_loopback
from .process import _linux_process_state_char, _pid_alive, _pid_is_defunct
from .receipt import (
    SESSION_RECEIPT_SCHEMA,
    ModelFileRef,
    SessionReceipt,
    write_session_receipt,
)
from .sockets import _SOCKET_FD_PATTERN, _loopback_family_and_address, _proc_net_listen_entries

DEFAULT_PROC_ROOT = Path("/proc")
_LIBPYTHON = re.compile(r"libpython[0-9.]*\.so")


def _proc_root(proc_root: "Path | None") -> Path:
    """The ``/proc`` to read. Resolved at CALL time: ``None`` means this module's :data:`DEFAULT_PROC_ROOT`.

    Default-at-call-time rather than a default ARGUMENT value so the seam's own fixed signatures
    (``core.session.runtime_identity.RuntimeIdentity.assert_session_alive`` takes no ``proc_root``) can still be
    exercised against a fake ``/proc`` in a test, by redirecting the module attribute. Production callers pass
    nothing and read the real one.
    """

    return DEFAULT_PROC_ROOT if proc_root is None else Path(proc_root)


def _refuse(message: str) -> SessionError:
    return SessionError(f"{message}; refused (I5)")


def _as_session_error(label: str, error: GateClosed) -> SessionError:
    """A gate-level identity refusal is a session refusal here; the message is carried through verbatim."""

    return SessionError(f"{label}: {error}")


# ---------------------------------------------------------------------------
# Declared agreement (no live process involved).
# ---------------------------------------------------------------------------


def interpreter_identity(runtime_receipt: Mapping[str, object]) -> tuple[str, str, int]:
    """``(absolute interpreter path, sha256, bytes)`` the receipt pins, or a refusal if it pins none."""

    interpreter = runtime_receipt.get("interpreter")
    if not isinstance(interpreter, Mapping):
        raise _refuse("vllm runtime receipt lacks an interpreter record")
    path, sha, size = interpreter.get("path"), interpreter.get("sha256"), interpreter.get("bytes")
    if not isinstance(path, str) or not path.startswith("/"):
        raise _refuse("vllm runtime receipt interpreter.path must be an absolute path")
    if not isinstance(sha, str) or len(sha) != 64:
        raise _refuse("vllm runtime receipt interpreter.sha256 must be a sha256")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise _refuse("vllm runtime receipt interpreter.bytes must be a byte count")
    return path, sha, size


def pinned_model_files(runtime_receipt: Mapping[str, object]) -> tuple[ModelFileRef, ...]:
    """The served-weight set, as ``ModelFileRef`` (relative paths, exactly as the runtime receipt records them)."""

    model = runtime_receipt.get("model")
    files = model.get("files") if isinstance(model, Mapping) else None
    if not isinstance(files, list) or not files:
        raise _refuse("vllm runtime receipt lacks a non-empty model.files list")
    refs = []
    for entry in files:
        if not isinstance(entry, Mapping):
            raise _refuse("vllm runtime receipt model.files entries must be mappings")
        refs.append(ModelFileRef(path=str(entry.get("path")), sha256=str(entry.get("sha256"))).validate())
    return tuple(refs)


def verify_vllm_session(
    session: SessionReceipt, runtime_receipt: Mapping[str, object]
) -> SessionReceipt:
    """The declared half: this session names the receipt's interpreter, weights, endpoint and command line.

    The vLLM counterpart of ``verify_against_runtime_receipt`` -- and, like it, the rail that refuses a hand-built
    receipt that never went through :func:`start_vllm_session` / :func:`attach_vllm_session`.
    """

    session.validate()
    # Structure first, so every field read below exists and no direct caller (a stop, an attach, a test) can reach
    # the live checks with a document that is not a vLLM receipt at all. Pure structure: no filesystem access, and
    # the gate's own byte-level re-verification is what proves the content.
    try:
        validate_vllm_receipt_document(runtime_receipt)
    except GateClosed as error:
        raise _as_session_error(f"session {session.session_id!r}", error) from error
    if session.schema != SESSION_RECEIPT_SCHEMA or session.role != "legacy":
        raise _refuse(
            f"a vllm session is recorded in {SESSION_RECEIPT_SCHEMA!r} with role 'legacy' "
            f"(the two-stage speech roles are llama.cpp feature-cache roles); got "
            f"{session.schema!r}/{session.role!r}"
        )
    path, sha, _size = interpreter_identity(runtime_receipt)
    pinned_relpath = path.lstrip("/")
    if session.binary_path != pinned_relpath or session.binary_sha256 != sha:
        raise _refuse(
            f"session {session.session_id!r} binary identity {session.binary_path!r}:"
            f"{session.binary_sha256[:12]}... does not match the pinned interpreter "
            f"{pinned_relpath!r}:{sha[:12]}..."
        )
    pinned = {(entry.path, entry.sha256) for entry in pinned_model_files(runtime_receipt)}
    if {(entry.path, entry.sha256) for entry in session.model_files} != pinned:
        raise _refuse(
            f"session {session.session_id!r} model file set does not match the vllm runtime receipt's served weights"
        )
    try:
        instance = instance_for_endpoint(runtime_receipt, session.host, session.port)
        assert_argv_matches_instance(session.argv, runtime_receipt, instance)
    except GateClosed as error:
        raise _as_session_error(f"session {session.session_id!r}", error) from error
    return session


def session_instance(
    session: SessionReceipt, runtime_receipt: Mapping[str, object]
) -> Mapping[str, object]:
    """The one receipt instance this session's endpoint belongs to."""

    try:
        return instance_for_endpoint(runtime_receipt, session.host, session.port)
    except GateClosed as error:
        raise _as_session_error(f"session {session.session_id!r}", error) from error


# ---------------------------------------------------------------------------
# Live process identity, read from a (possibly injected) /proc root.
# ---------------------------------------------------------------------------


def _read(path: Path, label: str, pid: int) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise _refuse(f"cannot read {label} of pid {pid} at {path}: {error}") from error


def _readlink(path: Path, label: str, pid: int) -> str:
    try:
        return os.readlink(path)
    except OSError as error:
        raise _refuse(f"cannot read {label} of pid {pid} at {path}: {error}") from error


def _live_argv(proc_root: Path, pid: int) -> tuple[str, ...]:
    raw = _read(Path(proc_root) / str(pid) / "cmdline", "cmdline", pid)
    try:
        return tuple(part.decode("utf-8") for part in raw.split(b"\0") if part)
    except UnicodeError as error:
        raise _refuse(f"pid {pid} has a non-UTF-8 command line") from error


def _live_environ(proc_root: Path, pid: int) -> dict[str, str]:
    raw = _read(Path(proc_root) / str(pid) / "environ", "environ", pid)
    environ: dict[str, str] = {}
    for part in raw.split(b"\0"):
        if not part:
            continue
        try:
            text = part.decode("utf-8")
        except UnicodeError as error:
            raise _refuse(f"pid {pid} has a non-UTF-8 environment entry") from error
        key, separator, value = text.partition("=")
        if not separator or not key:
            raise _refuse(f"pid {pid} has an unparseable environment entry {text[:32]!r}")
        if key in environ:
            raise _refuse(f"pid {pid} declares the environment variable {key!r} twice")
        environ[key] = value
    return environ


def _mapped_paths(proc_root: Path, pid: int) -> tuple[list[str], list[str]]:
    """``(mapped file paths, deleted mapped file paths)`` from ``/proc/<pid>/maps``."""

    raw = _read(Path(proc_root) / str(pid) / "maps", "maps", pid)
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeError as error:
        raise _refuse(f"pid {pid} has a non-UTF-8 memory map") from error
    mapped: list[str] = []
    deleted: list[str] = []
    for line in text.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) < 6:
            continue
        name = fields[5].strip()
        if not name.startswith("/"):
            continue  # [heap], [stack], anonymous shared memory: not a pinned file
        if name.endswith(" (deleted)"):
            deleted.append(name[: -len(" (deleted)")])
        else:
            mapped.append(name)
    return mapped, deleted


#: Seconds the ``shared-pid-host`` policy waits for the endpoint to answer ``/v1/models``.
NAMESPACE_PROBE_TIMEOUT_SECONDS = 5.0


def _http_get_json(url: str, timeout_seconds: float) -> object:
    """The decoded JSON body of a loopback GET. A module seam: tests replace :data:`http_get_json`."""

    import json
    import urllib.request

    with urllib.request.urlopen(url, timeout=timeout_seconds) as response:  # loopback only (require_loopback)
        return json.loads(response.read().decode("utf-8"))


http_get_json: Callable[[str, float], object] = _http_get_json


def current_hostname(proc_root: "Path | None" = None) -> str:
    """The harness's own hostname: ``<proc_root>/sys/kernel/hostname``, else ``uname`` ``nodename``."""

    try:
        text = (_proc_root(proc_root) / "sys" / "kernel" / "hostname").read_text(encoding="utf-8").strip()
    except OSError:
        text = ""
    return text or os.uname().nodename


def _verify_same_container(proc_root: Path, pid: int) -> None:
    for namespace in ("mnt", "net"):
        ours = _readlink(Path(proc_root) / "self" / "ns" / namespace, f"own {namespace} namespace", os.getpid())
        theirs = _readlink(Path(proc_root) / str(pid) / "ns" / namespace, f"{namespace} namespace", pid)
        if ours != theirs:
            raise _refuse(
                f"pid {pid} is in the {namespace} namespace {theirs!r}, not this process's {ours!r} -- the harness "
                "and the server must share one container, so the files the gate hashed are the files it loaded"
            )


def _verify_shared_pid_host(
    proc_root: Path,
    pid: int,
    runtime_receipt: Mapping[str, object],
    instance: Mapping[str, object],
    host: str,
    port: int,
) -> None:
    """The relaxed policy (plan Decision 3, option a): servers in their own containers with ``--pid host``.

    Mount and network namespaces are NOT compared. What replaces them, each refusal naming the instance and the
    failed condition: the harness is on the host the receipt was written on; the server pid is visible here and its
    executable and command line are the receipt instance's; the endpoint answers ``/v1/models`` with the recorded
    model. The image, the argv and the environment are pinned per instance by the receipt (re-checked by the live
    checks that follow), so a separate mount namespace cannot hide a different binary.
    """

    label = f"instance {instance['instance_id']!r} (pid {pid}, policy {NAMESPACE_POLICY_SHARED_PID_HOST!r})"

    def fail(condition: str, detail: str) -> SessionError:
        return _refuse(f"{label}: condition {condition!r} failed: {detail}")

    recorded = harness_hostname(runtime_receipt)
    if recorded is None:
        raise fail("same-host", "the runtime receipt records no deployment.harness_hostname")
    here = current_hostname(proc_root)
    if here != recorded:
        raise fail("same-host", f"this harness is on host {here!r} but the receipt was written on {recorded!r}")

    process = Path(proc_root) / str(pid)
    if not process.is_dir():
        raise fail(
            "pid-visible",
            f"{process} does not exist; the servers must run with --pid host and the harness must see the host's pids",
        )
    interpreter, _sha, _size = interpreter_identity(runtime_receipt)
    try:
        target = os.readlink(process / "exe")
    except OSError as error:
        raise fail("executable-image", f"cannot read {process / 'exe'}: {error}") from error
    if target != interpreter:
        raise fail("executable-image", f"the server runs {target!r}, not the pinned interpreter {interpreter!r}")
    try:
        assert_argv_matches_instance(_live_argv(proc_root, pid), runtime_receipt, instance)
    except (GateClosed, SessionError) as error:
        raise fail("command-line", str(error)) from error

    url = f"http://{_bracketed(host)}:{port}/v1/models"
    served = runtime_receipt["protocol"]["served_model_name"]
    try:
        body = http_get_json(url, NAMESPACE_PROBE_TIMEOUT_SECONDS)
    except Exception as error:  # noqa: BLE001 - any transport failure is the same refusal
        raise fail("endpoint-reachable", f"GET {url} failed: {error}") from error
    entries = body.get("data") if isinstance(body, Mapping) else None
    ids = [entry.get("id") for entry in entries if isinstance(entry, Mapping)] if isinstance(entries, list) else []
    if served not in ids:
        raise fail("endpoint-reachable", f"GET {url} lists models {ids}, not the recorded {served!r}")


def _verify_namespaces(
    proc_root: Path,
    pid: int,
    runtime_receipt: Mapping[str, object] | None = None,
    instance: Mapping[str, object] | None = None,
    host: str | None = None,
    port: int | None = None,
) -> None:
    """The pid is where the receipt's namespace policy says it must be.

    ``same-container`` (default; owner decision 9.2): the pid shares this process's mount and network namespaces, i.e.
    it is in THIS container. ``shared-pid-host`` (``deployment.namespace_policy`` of the receipt): see
    :func:`_verify_shared_pid_host`. A call without a receipt is the default policy.
    """

    try:
        policy = namespace_policy(runtime_receipt) if runtime_receipt is not None else DEFAULT_NAMESPACE_POLICY
    except GateClosed as error:
        raise _as_session_error(f"pid {pid}", error) from error
    if policy == NAMESPACE_POLICY_SHARED_PID_HOST:
        assert instance is not None and host is not None and port is not None
        _verify_shared_pid_host(Path(proc_root), pid, runtime_receipt, instance, host, port)
        return
    _verify_same_container(Path(proc_root), pid)


def _verify_executable_image(proc_root: Path, pid: int, path: str, sha: str, size: int) -> None:
    target = _readlink(Path(proc_root) / str(pid) / "exe", "executable image", pid)
    if target.endswith(" (deleted)"):
        raise _refuse(
            f"pid {pid}'s executable image {target!r} was deleted from disk after exec; identity cannot be proven "
            "against a file that is no longer there"
        )
    if target != path:
        raise _refuse(f"pid {pid} is running {target!r}, not the pinned interpreter {path!r}")
    image = Path(path)
    try:
        measured_size = image.stat().st_size
        digest = hashlib.sha256()
        with image.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise _refuse(f"cannot re-hash pid {pid}'s executable image {image}: {error}") from error
    if measured_size != size or digest.hexdigest() != sha:
        raise _refuse(
            f"pid {pid}'s executable image {image} hashes to {digest.hexdigest()[:12]}... ({measured_size} bytes) "
            f"but the runtime receipt pins {sha[:12]}... ({size} bytes)"
        )


def _verify_maps(proc_root: Path, pid: int, runtime_receipt: Mapping[str, object], interpreter: str) -> None:
    libraries = runtime_receipt.get("runtime_libraries")
    if not isinstance(libraries, list) or not libraries:
        raise _refuse("vllm runtime receipt lacks a non-empty runtime_libraries list")
    pinned_libraries = {str(entry["path"]) for entry in libraries if isinstance(entry, Mapping)}
    mapped, deleted = _mapped_paths(proc_root, pid)
    missing = sorted(pinned_libraries - set(mapped))
    if missing:
        raise _refuse(f"pid {pid} has not mapped the pinned runtime libraries {missing}")
    foreign = sorted(
        {
            name
            for name in mapped
            if name not in pinned_libraries and _LIBPYTHON.search(Path(name).name) is not None
        }
    )
    if foreign:
        raise _refuse(
            f"pid {pid} has mapped the unpinned interpreter runtime {foreign} -- a second libpython would make the "
            "re-hashed executable image meaningless"
        )
    roots = {interpreter, *pinned_libraries}
    for key in ("site_packages",):
        value = runtime_receipt.get(key)
        if isinstance(value, str):
            roots.add(value)
    for tree in runtime_receipt.get("source_trees") or ():
        if isinstance(tree, Mapping) and isinstance(tree.get("root"), str):
            roots.add(tree["root"])
    model = runtime_receipt.get("model")
    if isinstance(model, Mapping) and isinstance(model.get("model_dir"), str):
        roots.add(model["model_dir"])
    stale = sorted(
        {
            name
            for name in deleted
            if any(name == root or name.startswith(root.rstrip("/") + "/") for root in roots)
        }
    )
    if stale:
        raise _refuse(
            f"pid {pid} holds deleted mappings under the pinned roots {stale}; the bytes it is running are not the "
            "bytes the gate re-verified"
        )


def _verify_owns_listening_socket(proc_root: Path, pid: int, host: str, port: int) -> None:
    """The llama path's address-aware socket-ownership proof, read from ``proc_root``.

    A wildcard bind is never accepted as owning a specific loopback address, exactly as in
    ``core.session._posix_verify_pid_owns_listening_socket`` (see its own docstring for why).
    """

    family, expected_address = _loopback_family_and_address(host)
    name = "tcp" if family == socket.AF_INET else "tcp6"
    path = Path(proc_root) / "net" / name
    entries = _proc_net_listen_entries(path, family)
    inodes = {
        inode for address, entry_port, inode in entries if entry_port == port and address == expected_address
    }
    if not inodes:
        raise _refuse(
            f"no LISTEN socket found for {expected_address}:{port} ({path}); socket ownership must be provable"
        )
    fd_dir = Path(proc_root) / str(pid) / "fd"
    try:
        fds = list(fd_dir.iterdir())
    except OSError as error:
        raise _refuse(
            f"cannot enumerate open file descriptors for pid {pid} ({fd_dir}): {error}; socket ownership must be "
            "provable"
        ) from error
    for entry in fds:
        try:
            target = os.readlink(entry)
        except OSError:
            continue  # the fd raced away between listdir and readlink -- not fatal
        match = _SOCKET_FD_PATTERN.match(target)
        if match and int(match.group(1)) in inodes:
            return
    raise _refuse(
        f"pid {pid} does not own the listening socket on {expected_address}:{port} -- interpreter and source-tree "
        "identity alone do not prove that this process is the one answering this endpoint"
    )


def verify_vllm_process_identity(
    pid: int,
    runtime_receipt: Mapping[str, object],
    instance: Mapping[str, object],
    *,
    host: str,
    port: int,
    argv: Sequence[str] | None = None,
    proc_root: Path | None = None,
) -> tuple[str, ...]:
    """Every live check of one serving process, against one receipt instance. Returns its live command line.

    Takes a pid and an instance rather than a session receipt, so the receipt PRODUCER can run exactly the checks the
    gate will later run, before it writes anything.
    """

    proc_root = _proc_root(proc_root)
    state = _linux_process_state_char(pid, proc_root)
    if state is None:
        raise _refuse(f"pid {pid} is not a live process ({proc_root}/{pid}/stat cannot be read)")
    if state == "Z":
        raise _refuse(f"pid {pid} has exited and is awaiting reaping (state Z); it serves nothing")
    _verify_namespaces(proc_root, pid, runtime_receipt, instance, host, port)
    interpreter, sha, size = interpreter_identity(runtime_receipt)
    _verify_executable_image(proc_root, pid, interpreter, sha, size)
    live_argv = _live_argv(proc_root, pid)
    if argv is not None and tuple(str(token) for token in argv) != live_argv:
        raise _refuse(
            f"pid {pid}'s live command line {list(live_argv)} is not the one recorded for it {[str(t) for t in argv]}"
        )
    try:
        assert_argv_matches_instance(live_argv, runtime_receipt, instance)
    except GateClosed as error:
        raise _as_session_error(f"live command line of pid {pid}", error) from error
    protocol = runtime_receipt["protocol"]
    cwd = _readlink(Path(proc_root) / str(pid) / "cwd", "working directory", pid)
    if cwd != str(protocol["cwd"]):
        raise _refuse(f"pid {pid} runs in {cwd!r}, not the pinned working directory {protocol['cwd']!r}")
    environ = _live_environ(proc_root, pid)
    observed = pinned_environment(environ)
    expected = dict(protocol["env"])
    if observed != expected:
        drift = sorted(set(observed) ^ set(expected)) or sorted(
            key for key in expected if observed.get(key) != expected[key]
        )
        raise _refuse(
            f"pid {pid}'s behaviour-relevant environment differs from the receipt's at {drift}"
        )
    live_deployment = deployment_environment(environ)
    recorded_deployment = instance_deployment_env(instance)
    if live_deployment != recorded_deployment:
        drift = sorted(set(live_deployment) ^ set(recorded_deployment)) or sorted(
            key for key in recorded_deployment if live_deployment.get(key) != recorded_deployment[key]
        )
        raise _refuse(
            f"pid {pid}'s deployment environment {live_deployment} differs from instance "
            f"{instance['instance_id']!r}'s recorded {recorded_deployment} at {drift}"
        )
    devices = device_env_value(instance["device_ids"])
    if environ.get(DEVICE_ENV_KEY) != devices:
        raise _refuse(
            f"pid {pid} has {DEVICE_ENV_KEY}={environ.get(DEVICE_ENV_KEY)!r}, not instance "
            f"{instance['instance_id']!r}'s own device group {devices!r}"
        )
    _verify_maps(proc_root, pid, runtime_receipt, interpreter)
    _verify_owns_listening_socket(proc_root, pid, host, port)
    return live_argv


def assert_vllm_session_alive(
    session: SessionReceipt,
    runtime_receipt: Mapping[str, object],
    runtime_root: Path | None = None,
    *,
    proc_root: Path | None = None,
) -> None:
    """The full vLLM I5 re-check for ONE endpoint: declared agreement, then every live check above.

    Called from the same three independent sites the llama check is (transport construction AND every send, the
    adapter's pre-request call, and attach/stop), through ``core.session.runtime_identity``. Nothing is cached:
    every call re-reads ``/proc`` and re-hashes the interpreter.

    ``runtime_root`` is accepted for one seam signature with the llama check and is deliberately unused: a vLLM
    receipt pins absolute paths (the interpreter, the libraries, the source trees and the model directory all live
    outside the study tree, in the container image or on the shared filesystem), so there is no repo-relative root to
    resolve against -- and resolving one would be a second, weaker spelling of paths the gate already re-verified.
    """

    verify_vllm_session(session, runtime_receipt)
    instance = session_instance(session, runtime_receipt)
    verify_vllm_process_identity(
        session.pid,
        runtime_receipt,
        instance,
        host=session.host,
        port=session.port,
        argv=session.argv,
        proc_root=_proc_root(proc_root),
    )


def assert_vllm_sessions_alive(
    sessions: Sequence[SessionReceipt],
    runtime_receipt: Mapping[str, object],
    runtime_root: Path | None = None,
    *,
    proc_root: Path | None = None,
) -> None:
    """The multi-instance re-check: every session, each against its own receipt instance.

    Two sessions that claim one endpoint, or one pid, are refused before any of them is verified -- either would mean
    the caller cannot know which instance it is talking to (coordinator design input, 2026-10-28).
    """

    if not sessions:
        raise _refuse("a vllm serving set must hold at least one session")
    endpoints: set[tuple[str, int]] = set()
    pids: set[int] = set()
    for session in sessions:
        session.validate()
        endpoint = (session.host, session.port)
        if endpoint in endpoints:
            raise _refuse(f"two vllm sessions claim the endpoint {session.host}:{session.port}")
        if session.pid in pids:
            raise _refuse(f"two vllm sessions claim pid {session.pid}")
        endpoints.add(endpoint)
        pids.add(session.pid)
    for session in sessions:
        assert_vllm_session_alive(session, runtime_receipt, runtime_root, proc_root=_proc_root(proc_root))


# ---------------------------------------------------------------------------
# Lifecycle: spawn, attach, stop.
# ---------------------------------------------------------------------------


def assert_extra_argv_matches_instance(
    extra_argv: Sequence[str], runtime_receipt: Mapping[str, object], instance: Mapping[str, object]
) -> None:
    """The serving geometry a flight hands ``start_session`` must be exactly this instance's, protocol and deployment.

    ``core.session.serving.merge_server_args`` hands the fragment's ``server_args`` with the two geometry flags of the
    ``vllm`` dialect (``--max-num-seqs``, ``--max-model-len``) set from the resolved geometry. ``--max-model-len`` is
    protocol and must already be in the receipt's ``protocol.server_args``; ``--max-num-seqs`` is deployment and must
    equal this instance's. A fragment whose geometry drifted from the receipt is refused here, before any spawn.
    """

    try:
        served, protocol_tokens, deployment = classify_server_flags(extra_argv)
    except GateClosed as error:
        raise _as_session_error("serving geometry", error) from error
    if served is not None:
        raise _refuse(
            f"the serving geometry must not set {SERVED_MODEL_NAME_FLAG}: the served model name is the receipt's"
        )
    expected = [str(token) for token in runtime_receipt["protocol"]["server_args"]]
    if list(protocol_tokens) != expected:
        raise _refuse(
            f"the serving geometry's protocol flags {list(protocol_tokens)} differ from the vllm runtime receipt's "
            f"{expected}"
        )
    from ...contracts.gate_runtime_vllm import DEPLOYMENT_FLAGS

    for flag, value in deployment.items():
        field = DEPLOYMENT_FLAGS[flag]
        if str(instance[field]) != str(value):
            raise _refuse(
                f"the serving geometry says {flag} {value!r} but instance {instance['instance_id']!r} is deployed "
                f"with {field}={instance[field]!r}"
            )


def start_vllm_session(
    *,
    runtime_root: Path | None = None,
    data_root: Path | None = None,
    runtime_receipt: Mapping[str, object],
    host: str,
    port: int,
    session_id: str,
    extra_argv: Sequence[str] = (),
    extra_env: Mapping[str, str] | None = None,
    startup_timeout_seconds: float | None = None,
    startup_max_bytes: int | None = None,
    receipt_path: Path | None = None,
    popen: Callable[..., "subprocess.Popen[bytes]"] = subprocess.Popen,
    now: Callable[[], float] = time.time,
    ready_timeout_seconds: float | None = None,
    ready_poll_interval_seconds: float | None = None,
    http_get_status: Callable[[str, float], int] | None = None,
    server_log_path: Path | None = None,
    role: str = "legacy",
    model_root: Path | None = None,
    proc_root: Path | None = None,
) -> SessionReceipt:
    """Spawn ONE receipt instance and return the session receipt that binds this run to the process it started.

    Keyword-for-keyword the llama ``start_session`` signature (the flight's ``session_starter`` seam calls one of
    them), so a caller never learns which backend it drove. What differs is only what is spawned and what is proven:

    - the argv is NOT assembled from the caller's flags: it is the receipt instance's canonical command line
      (``instance_argv``), and the caller's serving geometry must already equal it
      (:func:`assert_extra_argv_matches_instance`);
    - the environment is the receipt's pinned environment plus this instance's device group and its recorded
      deployment variables, NOT the harness's
      inherited environment with overrides -- the whole point of the pinned environment is that it is not an
      operator-shell accident;
    - the interpreter is re-hashed immediately before the spawn (the llama path re-hashes its binary at the same
      point, for the same reason: a file swapped between gate-open and spawn must not be executed);
    - the spawned process is then put through the full live check, and killed if it fails it.
    """

    assert_execution_environment()
    if role != "legacy":
        raise _refuse(
            f"a vllm session has no two-stage speech-feature role; got {role!r} (those roles are llama.cpp "
            "feature-cache roles)"
        )
    require_loopback(f"http://{_bracketed(host)}:{port}")
    try:
        instance = instance_for_endpoint(runtime_receipt, host, port)
    except GateClosed as error:
        raise _as_session_error("spawn", error) from error
    assert_extra_argv_matches_instance(extra_argv, runtime_receipt, instance)
    interpreter, sha, size = interpreter_identity(runtime_receipt)
    image = Path(interpreter)
    try:
        measured = image.stat().st_size
        digest = hashlib.sha256()
        with image.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise _refuse(f"cannot re-hash the pinned interpreter {image} before spawning it: {error}") from error
    if measured != size or digest.hexdigest() != sha:
        raise _refuse(
            f"the pinned interpreter {image} hashes to {digest.hexdigest()[:12]}... but the runtime receipt pins "
            f"{sha[:12]}...; a file swapped between gate-open and spawn is never executed"
        )
    protocol = runtime_receipt["protocol"]
    env = {str(key): str(value) for key, value in protocol["env"].items()}
    env[DEVICE_ENV_KEY] = device_env_value(instance["device_ids"])
    env.update(instance_deployment_env(instance))
    if extra_env:
        for key, value in extra_env.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise _refuse(f"extra_env entries must be str->str, got {key!r}={value!r}")
            if env.get(key, value) != value:
                raise _refuse(
                    f"extra_env sets {key}={value!r} but the vllm runtime receipt pins {key}={env[key]!r}; serving "
                    "environment is receipt identity, never a caller override"
                )
            env[key] = value
    argv = list(instance_argv(runtime_receipt, instance))

    # Imported here, not at module import time: ``lifecycle`` imports this package, which imports this module.
    from .lifecycle import (
        DEFAULT_READY_POLL_INTERVAL_SECONDS,
        DEFAULT_READY_TIMEOUT_SECONDS,
        DEFAULT_STARTUP_MAX_BYTES,
        DEFAULT_STARTUP_TIMEOUT_SECONDS,
        _ServerLogWriter,
        _capture_startup_stdout,
        _http_get_status,
        _iso,
        _wait_until_ready,
    )

    startup_timeout = DEFAULT_STARTUP_TIMEOUT_SECONDS if startup_timeout_seconds is None else startup_timeout_seconds
    startup_bytes = DEFAULT_STARTUP_MAX_BYTES if startup_max_bytes is None else startup_max_bytes
    ready_timeout = DEFAULT_READY_TIMEOUT_SECONDS if ready_timeout_seconds is None else ready_timeout_seconds
    ready_poll = (
        DEFAULT_READY_POLL_INTERVAL_SECONDS if ready_poll_interval_seconds is None else ready_poll_interval_seconds
    )
    started_wall = now()
    process = popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(protocol["cwd"]),
        env=env,
    )
    log_writer = _ServerLogWriter(Path(server_log_path) if server_log_path is not None else None)
    stdout_bytes = _capture_startup_stdout(
        process, max_bytes=startup_bytes, timeout_seconds=startup_timeout, on_chunk=log_writer.write
    )
    try:
        if ready_timeout > 0:
            _wait_until_ready(
                f"http://{_bracketed(host)}:{port}/health",
                is_process_alive=lambda: process.poll() is None,
                http_get_status=http_get_status or _http_get_status,
                timeout_seconds=ready_timeout,
                poll_interval_seconds=ready_poll,
                startup_stdout_tail=stdout_bytes,
            )
        live_argv = verify_vllm_process_identity(
            process.pid,
            runtime_receipt,
            instance,
            host=host,
            port=port,
            argv=argv,
            proc_root=_proc_root(proc_root),
        )
    except BaseException:
        # Only ever this process's OWN child, and only after it failed the check it was spawned to pass.
        try:
            os.kill(process.pid, getattr(signal, "SIGTERM", signal.SIGTERM))
        except OSError:
            pass
        raise
    receipt = SessionReceipt(
        schema=SESSION_RECEIPT_SCHEMA,
        study=STUDY_SLUG,
        session_id=session_id,
        binary_path=interpreter.lstrip("/"),
        binary_sha256=sha,
        argv=tuple(live_argv),
        pid=process.pid,
        host=host,
        port=port,
        model_files=pinned_model_files(runtime_receipt),
        started_at=_iso(started_wall),
        startup_stdout_sha256=hashlib.sha256(stdout_bytes).hexdigest(),
        server_log_path=str(server_log_path) if server_log_path is not None else None,
    ).validate()
    verify_vllm_session(receipt, runtime_receipt)
    if receipt_path is not None:
        write_session_receipt(receipt_path, receipt)
    return receipt


def attach_vllm_session(
    receipt_path: Path,
    *,
    runtime_receipt: Mapping[str, object],
    runtime_root: Path | None = None,
    proc_root: Path | None = None,
) -> SessionReceipt:
    """Bind to an already-resident vLLM instance through an on-disk session receipt, after re-proving all of it."""

    from .receipt import read_session_receipt

    receipt = read_session_receipt(receipt_path)
    require_loopback(f"http://{_bracketed(receipt.host)}:{receipt.port}")
    assert_vllm_session_alive(receipt, runtime_receipt, runtime_root, proc_root=_proc_root(proc_root))
    return receipt


def stop_vllm_session(
    session: SessionReceipt,
    *,
    runtime_receipt: Mapping[str, object],
    runtime_root: Path | None = None,
    timeout_seconds: float = 30.0,
    poll_interval_seconds: float = 0.1,
    force_after_timeout: bool = False,
    receipt_path: Path | None = None,
    proc_root: Path | None = None,
) -> dict[str, object]:
    """Stop only the instance proven by ``session``, and write the same stop receipt the llama path writes.

    Identity, process image, environment, command line and socket ownership are rechecked immediately before SIGTERM,
    and again (the process-identity half) before any optional SIGKILL -- a timeout never widens the target. A pid that
    has already exited still has its DECLARED runtime identity verified, and the outcome is ``ALREADY_EXITED``.
    """

    from .lifecycle import (
        _canonical_document_sha256,
        _iso,
        _reap_stopped_child,
        _write_stop_receipt,
    )
    from .process import PROCESS_STATE_DEFUNCT, PROCESS_STATE_GONE

    session.validate()
    proc_root = _proc_root(proc_root)
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or timeout_seconds <= 0
        or isinstance(poll_interval_seconds, bool)
        or not isinstance(poll_interval_seconds, (int, float))
        or poll_interval_seconds <= 0
    ):
        raise _refuse("session stop timeout and poll interval must be positive numbers")
    if not isinstance(force_after_timeout, bool):
        raise _refuse("force_after_timeout must be boolean")

    def finish(termination_signal: str, forced: bool, final_state: str) -> dict[str, object]:
        document: dict[str, object] = {
            "schema": "saea-session-stop-receipt-v1",
            "study": STUDY_SLUG,
            "session_id": session.session_id,
            "session_role": session.role,
            "session_receipt_sha256": _canonical_document_sha256(session.as_json()),
            "runtime_receipt_sha256": _canonical_document_sha256(runtime_receipt),
            "pid": session.pid,
            "termination_signal": termination_signal,
            "forced": forced,
            "stopped_at": _iso(time.time()),
            "final_process_state": final_state,
        }
        if receipt_path is not None:
            _write_stop_receipt(receipt_path, document)
        return document

    verify_vllm_session(session, runtime_receipt)
    instance = session_instance(session, runtime_receipt)
    defunct = _pid_is_defunct(session.pid, Path(proc_root))
    if defunct or not _pid_alive(session.pid):
        reaped = _reap_stopped_child(session.pid)
        return finish(
            "ALREADY_EXITED", False, PROCESS_STATE_DEFUNCT if defunct and not reaped else PROCESS_STATE_GONE
        )
    try:
        assert_vllm_session_alive(session, runtime_receipt, runtime_root, proc_root=proc_root)
    except SessionError:
        if _pid_alive(session.pid):
            raise
        return finish("ALREADY_EXITED", False, PROCESS_STATE_GONE)
    try:
        os.kill(session.pid, signal.SIGTERM)
    except OSError as error:
        raise _refuse(f"cannot signal receipt-bound session pid {session.pid}: {error}") from error
    termination_signal = "SIGTERM"
    forced = False

    def wait_until_stopped(seconds: float) -> bool:
        deadline = time.monotonic() + seconds
        while True:
            if (
                _reap_stopped_child(session.pid)
                or not _pid_alive(session.pid)
                or _pid_is_defunct(session.pid, Path(proc_root))
            ):
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(float(poll_interval_seconds), remaining))

    stopped = wait_until_stopped(float(timeout_seconds))
    if not stopped and force_after_timeout:
        verify_vllm_process_identity(
            session.pid,
            runtime_receipt,
            instance,
            host=session.host,
            port=session.port,
            argv=session.argv,
            proc_root=proc_root,
        )
        kill_signal = getattr(signal, "SIGKILL", signal.SIGTERM)
        try:
            os.kill(session.pid, kill_signal)
        except OSError as error:
            raise _refuse(f"cannot force-stop receipt-bound session pid {session.pid}: {error}") from error
        termination_signal = "SIGKILL" if kill_signal != signal.SIGTERM else "SIGTERM"
        forced = True
        stopped = wait_until_stopped(min(float(timeout_seconds), 10.0))
    if not stopped:
        raise _refuse(
            f"receipt-bound session pid {session.pid} did not stop within {timeout_seconds} seconds"
        )
    return finish(
        termination_signal,
        forced,
        PROCESS_STATE_DEFUNCT if _pid_is_defunct(session.pid, Path(proc_root)) else PROCESS_STATE_GONE,
    )

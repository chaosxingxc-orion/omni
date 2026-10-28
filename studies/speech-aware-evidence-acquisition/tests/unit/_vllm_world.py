"""A complete, internally consistent vLLM runtime-identity world in ``tmp_path``, plus a fake ``/proc`` that satisfies
it -- the fixture the vLLM half of the runtime-identity seam is tested against (owner decisions 9.2/9.3, 2026-10-28).

Built, never hand-written: the receipt is assembled from the files this module actually creates (interpreter, libpython,
site-packages import-path files, two source trees, two dist-infos, the served weights) and the fake ``/proc`` entries
are derived from the receipt's own canonical per-instance command line, so a world that needs tampering to fail a
check is a world where that check genuinely has something to catch. Every test below then breaks exactly one thing.

No NPU, no server, no network: the "servers" are directories, and the only live pid any test reads is a fake one.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import socket
import struct
from pathlib import Path

from speech_aware_evidence_acquisition.contracts import STUDY_SLUG, FrozenCoreGate
from speech_aware_evidence_acquisition.contracts.gate_runtime_vllm import (
    DEVICE_ENV_KEY,
    VLLM_RUNTIME_RECEIPT_SCHEMA,
    device_env_value,
    file_entry,
    import_path_file_names,
    instance_argv,
    protocol_sha256,
    source_tree_manifest,
)
from speech_aware_evidence_acquisition.core.session import SESSION_RECEIPT_SCHEMA, ModelFileRef, SessionReceipt

MODEL_LOCK_KEY = "qwen3-omni-30b-a3b-thinking-bf16"
SUBDIR = f"models/{MODEL_LOCK_KEY}"
SERVED = "qwen3-omni-30b-a3b-thinking"
IMAGE_REF = "saea-base:20261027"
IMAGE_ID = "sha256:" + "4" * 64
PROTOCOL_ARGS = [
    "--max-model-len", "24576",
    "--dtype", "bfloat16",
    "--reasoning-parser", "qwen3",
    "--enable-auto-tool-choice",
    "--tool-call-parser", "hermes",
    "--seed", "0",
]
PINNED_ENV = {
    "PATH": "/usr/local/bin:/usr/bin",
    "PYTHONNOUSERSITE": "1",
    "PYTHONPATH": "/usr/local/Ascend/ascend-toolkit/latest/python/site-packages",
    "LD_LIBRARY_PATH": "/usr/local/Ascend/driver/lib64",
    "OMP_NUM_THREADS": "1",
    "TASK_QUEUE_ENABLE": "2",
    "VLLM_USE_V1": "1",
}
#: Variables a real server also carries and the receipt deliberately does NOT pin (not behaviour-relevant).
UNPINNED_ENV = {"HOSTNAME": "npu-181", "TERM": "dumb"}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _tcp_row(index: int, address: str, port: int, inode: int) -> str:
    packed = socket.inet_pton(socket.AF_INET, address)
    local = f"{struct.unpack('<I', packed)[0]:08X}:{port:04X}"
    return (
        f"{index:4d}: {local} 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1008        0 {inode} 1 x 100 0"
    )


class VllmWorld:
    """The whole pinned world: a container-shaped file tree, a model directory, a receipt, and a fake ``/proc``."""

    def __init__(self, tmp_path: Path) -> None:
        self.root = Path(tmp_path).resolve()
        self.materialization_root = self.root / "cephfs"
        self.model_dir = self.materialization_root / SUBDIR
        self.model_dir.mkdir(parents=True)
        self.model_files = {
            "config.json": b'{"model_type": "qwen3_omni_moe"}\n',
            "model-00001-of-00002.safetensors": b"BF16-fake-shard-one" * 32,
            "model-00002-of-00002.safetensors": b"BF16-fake-shard-two" * 16,
            "tokenizer.json": b'{"version": "1.0"}\n',
        }
        for name, content in self.model_files.items():
            (self.model_dir / name).write_bytes(content)
        (self.model_dir / ".hfd").mkdir()

        # The container's pinned Python: a real interpreter FILE (never executed by any test) and its runtime library.
        self.interpreter = self.root / "usr/local/python3.12.13/bin/python3.12"
        self.interpreter.parent.mkdir(parents=True)
        self.interpreter.write_bytes(b"ELF-fake-python-stub" * 8)
        self.libpython = self.root / "usr/local/python3.12.13/lib/libpython3.12.so.1.0"
        self.libpython.parent.mkdir(parents=True, exist_ok=True)
        self.libpython.write_bytes(b"ELF-fake-libpython" * 64)
        self.site_packages = self.root / "usr/local/python3.12.13/lib/python3.12/site-packages"
        self.site_packages.mkdir(parents=True)
        (self.site_packages / "__editable__.vllm-0.30.0.pth").write_bytes(b"import __editable___vllm_finder\n")
        (self.site_packages / "__editable___vllm_finder.py").write_bytes(b"MAPPING = {'vllm': '/vllm-workspace'}\n")
        (self.site_packages / "distutils-precedence.pth").write_bytes(b"import os\n")

        # The two editable source trees that actually serve the model.
        self.trees = {}
        for package, relative in (
            ("vllm", "vllm-workspace/vllm/vllm"),
            ("vllm_ascend", "vllm-workspace/vllm-ascend/vllm_ascend"),
        ):
            tree = self.root / relative
            (tree / "entrypoints").mkdir(parents=True)
            (tree / "__init__.py").write_bytes(f"# {package}\n".encode("utf-8"))
            (tree / "entrypoints" / "cli.py").write_bytes(b"def main():\n    return 0\n")
            (tree / "__pycache__").mkdir()
            (tree / "__pycache__" / "junk.pyc").write_bytes(b"not part of identity")
            self.trees[package] = tree

        # Installed package metadata.
        self.packages = {}
        for name, version in (("vllm", "0.30.0+empty"), ("vllm_ascend", "0.30.0rc2")):
            dist_info = self.site_packages / f"{name}-{version}.dist-info"
            dist_info.mkdir()
            (dist_info / "METADATA").write_text(
                f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n\nfake\n", encoding="utf-8"
            )
            (dist_info / "RECORD").write_text(f"{name}/__init__.py,,\n", encoding="utf-8")
            self.packages[name] = (dist_info, version)

        self.cwd = self.root / "vllm-workspace"
        self.instances = [
            {
                "instance_id": "tp2-a",
                "host": "127.0.0.1",
                "port": 8200,
                "device_ids": [0, 1],
                "tensor_parallel_size": 2,
                "max_num_seqs": 8,
                "gpu_memory_utilization": 0.9,
                "max_num_batched_tokens": None,
            },
            {
                "instance_id": "tp2-b",
                "host": "127.0.0.1",
                "port": 8201,
                "device_ids": [2, 3],
                "tensor_parallel_size": 2,
                "max_num_seqs": 8,
                "gpu_memory_utilization": 0.9,
                "max_num_batched_tokens": None,
            },
        ]
        self._receipt = self._build_receipt()

        self.proc_root = self.root / "proc"
        (self.proc_root / "net").mkdir(parents=True)
        for namespace in ("mnt", "net"):
            target = self.proc_root / "self" / "ns"
            target.mkdir(parents=True, exist_ok=True)
            os.symlink(f"{namespace}:[4026531840]", target / namespace)
        self._listeners: list[tuple[str, int, int]] = []
        self._write_tcp_table()
        (self.root / "receipts").mkdir()
        self.receipt_path = self.root / "receipts" / "runtime-vllm.json"
        self.lock_path = self.root / "receipts" / "datasets.lock.json"

    # -- the receipt --------------------------------------------------------
    def _build_receipt(self) -> dict:
        receipt = {
            "schema": VLLM_RUNTIME_RECEIPT_SCHEMA,
            "study": STUDY_SLUG,
            "model_lock_key": MODEL_LOCK_KEY,
            "container": {"image_ref": IMAGE_REF, "image_id": IMAGE_ID},
            "interpreter": file_entry(self.interpreter),
            "runtime_libraries": [file_entry(self.libpython)],
            "site_packages": str(self.site_packages),
            "import_path_files": [file_entry(path) for path in import_path_file_names(self.site_packages)],
            "source_trees": [
                {"package": package, "root": str(root), **source_tree_manifest(root)}
                for package, root in sorted(self.trees.items())
            ],
            "packages": [
                {
                    "name": name,
                    "version": version,
                    "dist_info": str(dist_info),
                    "metadata_sha256": sha((dist_info / "METADATA").read_bytes()),
                    "record_sha256": sha((dist_info / "RECORD").read_bytes()),
                }
                for name, (dist_info, version) in sorted(self.packages.items())
            ],
            "model": {
                "model_dir": str(self.model_dir),
                "files": [
                    {"path": f"{SUBDIR}/{name}", "bytes": len(content), "sha256": sha(content)}
                    for name, content in sorted(self.model_files.items())
                ],
                "other_entries": [".hfd"],
            },
            "protocol": {
                "served_model_name": SERVED,
                "argv0": str(self.interpreter),
                "entrypoint": ["-m", "vllm.entrypoints.cli.main", "serve"],
                "server_args": list(PROTOCOL_ARGS),
                "cwd": str(self.cwd),
                "env": dict(PINNED_ENV),
            },
            "deployment": {"instances": copy.deepcopy(self.instances)},
            "protocol_sha256": "0" * 64,
        }
        receipt["protocol_sha256"] = protocol_sha256(receipt)
        return receipt

    def receipt(self, **protocol_overrides: object) -> dict:
        """A fresh copy of the receipt; ``protocol_overrides`` replace protocol fields and re-hash it."""

        receipt = copy.deepcopy(self._receipt)
        if protocol_overrides:
            receipt["protocol"].update(protocol_overrides)
            receipt["protocol_sha256"] = protocol_sha256(receipt)
        return receipt

    @property
    def lock(self) -> dict:
        files = self._receipt["model"]["files"]
        return {
            "models": [
                {
                    "name": MODEL_LOCK_KEY,
                    "local_subdir": SUBDIR,
                    "files": len(files),
                    "size_bytes": sum(entry["bytes"] for entry in files),
                }
            ],
            "datasets": [],
        }

    def gate(self, receipt: dict | None = None, **overrides: object) -> FrozenCoreGate:
        """A gate bound to this world's receipt and lock, with ``transport='vllm'``."""

        document = self._receipt if receipt is None else receipt
        self.receipt_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
        self.lock_path.write_text(json.dumps(self.lock), encoding="utf-8")
        fields: dict = dict(
            e0_receipt=self.root / "e0-closure.json",
            runtime_receipt=self.receipt_path,
            data_root=self.materialization_root,
            lock_path=self.lock_path,
            runtime_root=self.root / "runtime",
            exposure_ledger=self.root / "exposure-ledger.md",
            source_root=self.root,
            expected_model_lock_key=MODEL_LOCK_KEY,
            transport="vllm",
        )
        fields.update(overrides)
        return FrozenCoreGate(**fields)

    # -- the fake /proc -----------------------------------------------------
    def argv_for(self, instance_id: str, receipt: dict | None = None) -> list[str]:
        document = self._receipt if receipt is None else receipt
        instance = next(i for i in document["deployment"]["instances"] if i["instance_id"] == instance_id)
        return list(instance_argv(document, instance))

    def _write_tcp_table(self) -> None:
        header = (
            "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode"
        )
        rows = [_tcp_row(index, address, port, inode) for index, (address, port, inode) in enumerate(self._listeners)]
        (self.proc_root / "net" / "tcp").write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
        (self.proc_root / "net" / "tcp6").write_text(header + "\n", encoding="utf-8")

    def spawn(
        self,
        instance_id: str,
        *,
        pid: int,
        receipt: dict | None = None,
        argv: list[str] | None = None,
        environ: dict[str, str] | None = None,
        cwd: Path | None = None,
        exe: Path | None = None,
        state: str = "S",
        maps: list[str] | None = None,
        namespaces: dict[str, str] | None = None,
        listening: bool = True,
        owns_socket: bool = True,
    ) -> int:
        """Create a fake ``/proc/<pid>`` for one instance; every argument defaults to what the receipt pins."""

        document = self._receipt if receipt is None else receipt
        instance = next(i for i in document["deployment"]["instances"] if i["instance_id"] == instance_id)
        process = self.proc_root / str(pid)
        (process / "fd").mkdir(parents=True)
        (process / "ns").mkdir()
        # /proc/<pid>/stat: field 2 is a parenthesised comm that may itself hold spaces, field 3 is the state.
        (process / "stat").write_text(
            f"{pid} (python3.12 (serve)) {state} 1" + " 0" * 49 + "\n", encoding="utf-8"
        )
        command = list(argv) if argv is not None else self.argv_for(instance_id, document)
        (process / "cmdline").write_bytes(b"\0".join(part.encode("utf-8") for part in command) + b"\0")
        live_env = dict(PINNED_ENV if environ is None else environ)
        live_env.setdefault(DEVICE_ENV_KEY, device_env_value(instance["device_ids"]))
        live_env.update(UNPINNED_ENV)
        (process / "environ").write_bytes(
            b"\0".join(f"{key}={value}".encode("utf-8") for key, value in live_env.items()) + b"\0"
        )
        mapped = maps if maps is not None else [str(self.interpreter), str(self.libpython), "[heap]"]
        (process / "maps").write_text(
            "".join(
                f"7f0000{index:03d}000-7f0000{index:03d}fff r-xp 00000000 fd:01 {1000 + index} {name}\n"
                for index, name in enumerate(mapped)
            ),
            encoding="utf-8",
        )
        os.symlink(str(exe if exe is not None else self.interpreter), process / "exe")
        os.symlink(str(cwd if cwd is not None else self.cwd), process / "cwd")
        for namespace, target in (namespaces or {"mnt": "mnt:[4026531840]", "net": "net:[4026531840]"}).items():
            os.symlink(target, process / "ns" / namespace)
        inode = 500000 + pid
        if listening:
            self._listeners.append((instance["host"], int(instance["port"]), inode))
            self._write_tcp_table()
        if owns_socket:
            os.symlink(f"socket:[{inode}]", process / "fd" / "3")
        else:
            os.symlink(f"socket:[{inode + 1}]", process / "fd" / "3")
        return pid

    def session(
        self,
        instance_id: str,
        pid: int,
        *,
        receipt: dict | None = None,
        session_id: str | None = None,
        **overrides: object,
    ) -> SessionReceipt:
        """The session receipt ``start_vllm_session`` would have written for one fake instance."""

        document = self._receipt if receipt is None else receipt
        instance = next(i for i in document["deployment"]["instances"] if i["instance_id"] == instance_id)
        fields: dict = dict(
            schema=SESSION_RECEIPT_SCHEMA,
            study=STUDY_SLUG,
            session_id=session_id or f"vllm-{instance_id}",
            binary_path=str(self.interpreter).lstrip("/"),
            binary_sha256=document["interpreter"]["sha256"],
            argv=tuple(self.argv_for(instance_id, document)),
            pid=pid,
            host=instance["host"],
            port=int(instance["port"]),
            model_files=tuple(
                ModelFileRef(path=entry["path"], sha256=entry["sha256"]) for entry in document["model"]["files"]
            ),
            started_at="2026-10-28T00:00:00+00:00",
            startup_stdout_sha256=sha(b""),
        )
        fields.update(overrides)
        return SessionReceipt(**fields)

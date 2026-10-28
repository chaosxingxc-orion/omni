"""The runtime-identity seam, receipt half (owner decision 9.3, 2026-10-28: "preferably compatible via an adapter").

A serving backend's runtime identity has two halves:

1. what its committed runtime receipt must say, re-verified from bytes when the gate opens (this module), and
2. how a live process proves, on every send, that it is that runtime (``core.session.runtime_identity``).

The gate, the transport and the session lifecycle call this one seam. They do not branch on the backend. The adapter
is chosen by the model fragment's ``transport`` field. ``contracts`` must never import ``core`` (``contracts/__init__``
docstring), so the seam is split across the two layers by that rule. This module holds the interface and the receipt
validators. ``core.session.runtime_identity`` extends each one with its live-process half and registers the full
adapter under the same transport name.

Adapters:

``llama-server``  ``FrozenCoreGate._validate_runtime`` (``gate_runtime.py``), called exactly as before. The llama.cpp
                  receipt schemas (v2/v3) are not changed.
``vllm``          ``FrozenCoreGate._validate_vllm_runtime`` (``gate_runtime_vllm.py``), schema
                  ``saea-runtime-receipt-vllm-v1``.

A transport with no registered adapter closes the gate, because nothing could re-verify its runtime.
"""

from __future__ import annotations

from typing import Mapping

from .errors import GateClosed
from .identifiers import RUNTIME_RECEIPT_SCHEMA, RUNTIME_RECEIPT_SCHEMA_V3, VLLM_RUNTIME_RECEIPT_SCHEMA

LLAMA_SERVER_TRANSPORT = "llama-server"
VLLM_TRANSPORT = "vllm"


class RuntimeReceiptIdentity:
    """Receipt half of one serving backend's runtime identity."""

    #: The model fragment ``transport`` value this adapter serves.
    transport: str = ""
    #: The runtime-receipt schemas this adapter accepts (and no other adapter accepts).
    receipt_schemas: frozenset[str] = frozenset()

    def validate_runtime_receipt(
        self,
        gate: object,
        lock: Mapping[str, object] | None,
        *,
        full_hash: bool | None = None,
        diagnostics: dict[str, object] | None = None,
    ) -> None:
        """Re-verify ``gate.runtime_receipt`` from bytes; raise :class:`GateClosed` on any drift."""

        raise NotImplementedError

    def owns_receipt(self, runtime_receipt: Mapping[str, object]) -> bool:
        return runtime_receipt.get("schema") in self.receipt_schemas


class LlamaServerReceiptIdentity(RuntimeReceiptIdentity):
    transport = LLAMA_SERVER_TRANSPORT
    receipt_schemas = frozenset({RUNTIME_RECEIPT_SCHEMA, RUNTIME_RECEIPT_SCHEMA_V3})

    def validate_runtime_receipt(self, gate, lock, *, full_hash=None, diagnostics=None) -> None:
        # The historical check, called exactly as ``assert_model_touch_allowed`` always called it.
        gate._validate_runtime(lock, full_hash=full_hash, diagnostics=diagnostics)


class VllmReceiptIdentity(RuntimeReceiptIdentity):
    transport = VLLM_TRANSPORT
    receipt_schemas = frozenset({VLLM_RUNTIME_RECEIPT_SCHEMA})

    def validate_runtime_receipt(self, gate, lock, *, full_hash=None, diagnostics=None) -> None:
        gate._validate_vllm_runtime(lock, full_hash=full_hash, diagnostics=diagnostics)


RUNTIME_RECEIPT_IDENTITIES: Mapping[str, RuntimeReceiptIdentity] = {
    LLAMA_SERVER_TRANSPORT: LlamaServerReceiptIdentity(),
    VLLM_TRANSPORT: VllmReceiptIdentity(),
}


def runtime_receipt_identity(transport: object) -> RuntimeReceiptIdentity:
    """The receipt half of the adapter for one fragment ``transport``; unknown transports close the gate."""

    identity = RUNTIME_RECEIPT_IDENTITIES.get(transport) if isinstance(transport, str) else None
    if identity is None:
        raise GateClosed(
            f"no runtime identity is registered for transport {transport!r} "
            f"(registered: {sorted(RUNTIME_RECEIPT_IDENTITIES)}); model touch refused"
        )
    return identity

"""SLURP lock ``v2.2-npu-port-r01c``: the port lock with the minus-R01 exemplar-store pins, for the T construction cells.

Plan ``docs/superpowers/plans/2026-10-30-tfrl-three-way-comparison-and-improvement-plan.md`` section 13 ruling 3 (c) and lock plan 3.14 (addendum item 6). The three T
construction cells (smoke, H1, H2) sample on the construction column R01C, so they pin the ``size-pool-annotated-exemplars-MINUS-R01`` train-exemplars store (the
lock's own pool contains the R01C rows, which the runner's knowledge-source guard refuses). Lock ``v2.2-npu-port`` carries the lock pool, so those cells need one more layer.

``v2.2-npu-port-r01c`` = the ``v2.2-npu-port`` layer with EXACTLY FOUR leaves moved (``R01C_LEAVES``):

* ``knowledge_stores.train-exemplars.release.{relpath,sha256}`` and ``knowledge_stores.train-exemplars.snapshot.{relpath,sha256}``.

The values are read from the committed build receipt of the minus-R01 store (``R01C_RECEIPT_RELPATH``: ``release_artifact_relpath`` / ``release_file_sha256`` and
``snapshot_artifact_relpath`` / ``snapshot_sha256``); nothing is typed here. The facet-vector source, the roster pins and every other leaf stay the port lock's. The layer is
fail-closed: a missing or incomplete receipt, or a pin equal to the port lock's value (a pin that did not move), is refused. Only a block that names this lock in its ``lock``
key reaches it; the CURRENT lock stays v2.3.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Callable, Mapping

from . import slurp_module_lock_v2_2 as lock_v2_2
from . import slurp_module_lock_v2_2_npu_port as lock_port

LOCK_V2_2_NPU_PORT_R01C = "v2.2-npu-port-r01c"
#: The committed build receipt of the minus-R01 store, relative to the study repo root.
R01C_RECEIPT_RELPATH = "docs/receipts/knowledge/slurp-kb-size-pool-annotated-exemplars-minus-r01.json"

POOL_PATH = ("knowledge_stores", "train-exemplars")
#: ``(leaf path into the layer, receipt key)`` -- the four leaves and where each value is read from.
R01C_SOURCES: tuple[tuple[tuple[str, ...], str], ...] = (
    (POOL_PATH + ("release", "relpath"), "release_artifact_relpath"),
    (POOL_PATH + ("release", "sha256"), "release_file_sha256"),
    (POOL_PATH + ("snapshot", "relpath"), "snapshot_artifact_relpath"),
    (POOL_PATH + ("snapshot", "sha256"), "snapshot_sha256"),
)
R01C_LEAVES: tuple[tuple[str, ...], ...] = tuple(path for path, _ in R01C_SOURCES)

#: ``{lock version: the deliver normaliser id its layer carries}`` -- the r01c lock keeps v2.2's v4g.
LOCK_DELIVER_NORMALISERS: Mapping[str, str] = {
    **lock_port.LOCK_DELIVER_NORMALISERS,
    LOCK_V2_2_NPU_PORT_R01C: lock_v2_2.LOCK_V2_2_DELIVER_NORMALISER,
}


def lock_deliver_normalisers() -> Mapping[str, Callable[[], str]]:
    """The ``CarrierHooks.lock_deliver_normalisers`` table, v2.0 through v2.3, ``v2.2-npu-port`` and ``v2.2-npu-port-r01c``."""

    return {lock: (lambda name=name: name) for lock, name in LOCK_DELIVER_NORMALISERS.items()}


def r01c_pins(repo_root: Path) -> Mapping[tuple[str, ...], str]:
    """``{leaf path: value}`` read from the minus-R01 build receipt; refused when the receipt is missing or a key is absent or not a non-empty string."""

    from ..consumption_module import ConsumptionModuleConfigError  # noqa: PLC0415

    path = Path(repo_root) / R01C_RECEIPT_RELPATH
    if not path.is_file():
        raise ConsumptionModuleConfigError(f"lock {LOCK_V2_2_NPU_PORT_R01C}: the minus-R01 build receipt {R01C_RECEIPT_RELPATH} is missing")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for leaf, key in R01C_SOURCES:
        value = receipt.get(key) if isinstance(receipt, dict) else None
        if not isinstance(value, str) or not value:
            raise ConsumptionModuleConfigError(f"lock {LOCK_V2_2_NPU_PORT_R01C}: {R01C_RECEIPT_RELPATH} carries no usable {key!r}")
        out[leaf] = value
    return out


def layer_from_port(port_layer: Mapping[str, object], repo_root: Path) -> Mapping[str, object]:
    """The ``v2.2-npu-port`` layer with the four :data:`R01C_LEAVES` set to the minus-R01 pins. Fail-closed on a missing leaf and on a pin equal to the port lock's value."""

    from ..consumption_module import ConsumptionModuleConfigError  # noqa: PLC0415

    pins = r01c_pins(repo_root)
    layer = copy.deepcopy(dict(port_layer))
    for path in R01C_LEAVES:
        name = ".".join(path)
        try:
            holder = lock_port._at(layer, path[:-1])  # noqa: SLF001
            old = holder[path[-1]]
        except KeyError:
            raise ConsumptionModuleConfigError(f"lock {LOCK_V2_2_NPU_PORT_R01C}: leaf {name} is missing from the {lock_port.LOCK_V2_2_NPU_PORT} layer") from None
        if old == pins[path]:
            raise ConsumptionModuleConfigError(f"lock {LOCK_V2_2_NPU_PORT_R01C}: {name}: the minus-R01 pin {pins[path]!r} equals the {lock_port.LOCK_V2_2_NPU_PORT} value")
        holder[path[-1]] = pins[path]
    return layer

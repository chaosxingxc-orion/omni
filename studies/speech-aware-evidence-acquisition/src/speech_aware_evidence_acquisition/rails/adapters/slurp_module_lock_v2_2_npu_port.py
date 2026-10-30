"""SLURP lock ``v2.2-npu-port``: lock v2.2 with the NPU port's re-authored knowledge-store pins.

Source of the decision: ``docs/results/2026-10-30-badcase-verify-declared-module-l338-port.md`` (analyst note) and
``docs/superpowers/plans/2026-10-29-lock-engineering-base-and-efficiency-plan.md`` section 3.14 (planner decision). The port cells
(``l338_port_*``, ``fold_a_port_*``, the T cells) pin stores that the NPU port re-authored (decided differences 8.4 and 8.6, owner decision
9.5 of ``docs/superpowers/specs/2026-10-28-npu-vllm-port-design.md``), while lock ``v2.2`` carries the HISTORICAL 268-B store pins. A
config that declares a consumption-module block must carry exactly the blocks the block compiles to
(``consumption_module.verify_declared_module``), so every port cell was refused at load time. This lock is the missing layer: not a new
policy, only the port store pins.

``v2.2-npu-port`` = the v2.2 layer with EXACTLY TEN LEAVES moved (``PORT_LEAVES``), each taken from
``scripts/flights/nx/l338_port_pins.json`` (the single source of the port retrieval pins; nothing is typed here):

* ``knowledge_stores.intent-boundaries.release.{relpath,sha256}`` and ``.snapshot.{relpath,sha256}`` (the re-authored forced-causal
  intent-boundaries store, ``intent-boundaries-npu-port-v1``);
* ``knowledge_stores.slot-contracts.release.{relpath,sha256}`` and ``.snapshot.{relpath,sha256}`` (``slot-contracts-npu-port-v2``);
* ``knowledge_stores.train-exemplars.phase_supply.solver.facet_source.sha256`` and ``.tool_contact_ledger_row`` (the regenerated kus3
  facet vectors, ledger row ``SAEA-SLURP-MF7B-NEMOTRON-EMBEDDING-NPU-PORT-V1``; the relpath is the same on both sides).

Everything else is v2.2 byte for byte: the deliver normaliser stays v4g, the roster pins stay exempt
(``slurp_module_hooks.ROSTER_PIN_PATHS``), and the module's components still write their own pin sites (the centroid facet source, the
agreement source, ``k``) exactly as under v2.2. The layer is fail-closed on the pins file: a missing file, a missing leaf, or a pin equal
to the v2.2 value (a port pin that did not move) is refused. The CURRENT lock stays v2.3. A block selects this lock only through its
``lock`` key. T construction cells (minus-R01 exemplar store) are NOT covered by this lock; they select ``v2.2-npu-port-r01c`` (see ``slurp_module_lock_v2_2_npu_port_r01c``).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Callable, Mapping

from . import slurp_module_lock_v2_2 as lock_v2_2
from . import slurp_module_lock_v2_3 as lock_v2_3

LOCK_V2_2_NPU_PORT = "v2.2-npu-port"
#: The pins file the ten leaves are read from, relative to the study repo root.
PORT_PINS_RELPATH = "scripts/flights/nx/l338_port_pins.json"

_STORES = "knowledge_stores"
_FACET = (_STORES, "train-exemplars", "phase_supply", "solver", "facet_source")
#: The ten leaves, as paths into a lock layer (and, identically, into the pins file: its ``knowledge_stores`` key is the layer's).
PORT_LEAVES: tuple[tuple[str, ...], ...] = (
    (_STORES, "intent-boundaries", "release", "relpath"),
    (_STORES, "intent-boundaries", "release", "sha256"),
    (_STORES, "intent-boundaries", "snapshot", "relpath"),
    (_STORES, "intent-boundaries", "snapshot", "sha256"),
    (_STORES, "slot-contracts", "release", "relpath"),
    (_STORES, "slot-contracts", "release", "sha256"),
    (_STORES, "slot-contracts", "snapshot", "relpath"),
    (_STORES, "slot-contracts", "snapshot", "sha256"),
    _FACET + ("sha256",),
    _FACET + ("tool_contact_ledger_row",),
)

#: ``{lock version: the deliver normaliser id its layer carries}`` -- the port lock keeps v2.2's v4g.
LOCK_DELIVER_NORMALISERS: Mapping[str, str] = {
    **lock_v2_3.LOCK_DELIVER_NORMALISERS,
    LOCK_V2_2_NPU_PORT: lock_v2_2.LOCK_V2_2_DELIVER_NORMALISER,
}


def lock_deliver_normalisers() -> Mapping[str, Callable[[], str]]:
    """The ``CarrierHooks.lock_deliver_normalisers`` table, v2.0 through v2.3 and ``v2.2-npu-port``."""

    return {lock: (lambda name=name: name) for lock, name in LOCK_DELIVER_NORMALISERS.items()}


def _at(node: object, path: tuple[str, ...]) -> object:
    for key in path:
        if not isinstance(node, dict) or key not in node:
            raise KeyError(".".join(path))
        node = node[key]
    return node


def port_pins(repo_root: Path) -> Mapping[str, object]:
    """The ``knowledge_stores`` mapping of the port pins file; refused when the file is missing or has no such mapping."""

    from ..consumption_module import ConsumptionModuleConfigError  # noqa: PLC0415

    path = Path(repo_root) / PORT_PINS_RELPATH
    if not path.is_file():
        raise ConsumptionModuleConfigError(f"lock {LOCK_V2_2_NPU_PORT}: the port pins file {PORT_PINS_RELPATH} is missing")
    stores = json.loads(path.read_text(encoding="utf-8")).get(_STORES)
    if not isinstance(stores, dict):
        raise ConsumptionModuleConfigError(f"lock {LOCK_V2_2_NPU_PORT}: {PORT_PINS_RELPATH} carries no {_STORES} mapping")
    return stores


def layer_from_v2_2(v2_2_layer: Mapping[str, object], repo_root: Path) -> Mapping[str, object]:
    """The v2.2 layer with the ten :data:`PORT_LEAVES` set to the port pins. Fail-closed on a missing leaf (either side) and on a port pin that
    equals the v2.2 value."""

    from ..consumption_module import ConsumptionModuleConfigError  # noqa: PLC0415

    pins = {_STORES: port_pins(repo_root)}
    layer = copy.deepcopy(dict(v2_2_layer))
    for path in PORT_LEAVES:
        name = ".".join(path)
        try:
            old, new = _at(layer, path), _at(pins, path)
        except KeyError as error:
            raise ConsumptionModuleConfigError(f"lock {LOCK_V2_2_NPU_PORT}: leaf {error.args[0]} is missing from the v2.2 layer or from {PORT_PINS_RELPATH}") from None
        if not isinstance(new, str) or not new or old == new:
            raise ConsumptionModuleConfigError(
                f"lock {LOCK_V2_2_NPU_PORT}: {name}: the port pin {new!r} is empty, not a string, or equal to the v2.2 value")
        _at(layer, path[:-1])[path[-1]] = new
    return layer

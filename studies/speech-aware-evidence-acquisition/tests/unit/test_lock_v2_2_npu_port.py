"""Lock ``v2.2-npu-port`` (plan 3.14, 2026-10-30): registry, fail-closed layer, loader acceptance of the port cells, writer self-check.

The port cells pin the stores the NPU port re-authored while lock v2.2 carries the historical 268-B store pins, so ``verify_declared_module`` (the config loader's call) refused
every port cell. The lock layer ``v2.2-npu-port`` is v2.2 with exactly ten leaves read from ``scripts/flights/nx/l338_port_pins.json``. Model-free: no contact of any kind.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
for _extra in (REPO / "scripts" / "flights" / "nx", REPO / "src"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import fold_a_port_derive_cells as fdc  # noqa: E402
import l338_port_derive_cells as pdc  # noqa: E402
import l338_port_family as pf  # noqa: E402

from speech_aware_evidence_acquisition.rails import consumption_module as cm  # noqa: E402
from speech_aware_evidence_acquisition.rails.adapters import slurp_module_hooks as smh  # noqa: E402
from speech_aware_evidence_acquisition.rails.adapters import slurp_module_lock_v2_2_npu_port as port  # noqa: E402

LOCK = "v2.2-npu-port"
V4G = "saea-deliver-normalise-v4g"
PINS = json.loads((REPO / "scripts" / "flights" / "nx" / "l338_port_pins.json").read_text(encoding="utf-8"))
#: The ten leaves, dotted (the analyst note's table, bad-case 2026-10-30 section 2).
TEN = sorted([
    "knowledge_stores.intent-boundaries.release.relpath", "knowledge_stores.intent-boundaries.release.sha256",
    "knowledge_stores.intent-boundaries.snapshot.relpath", "knowledge_stores.intent-boundaries.snapshot.sha256",
    "knowledge_stores.slot-contracts.release.relpath", "knowledge_stores.slot-contracts.release.sha256",
    "knowledge_stores.slot-contracts.snapshot.relpath", "knowledge_stores.slot-contracts.snapshot.sha256",
    "knowledge_stores.train-exemplars.phase_supply.solver.facet_source.sha256",
    "knowledge_stores.train-exemplars.phase_supply.solver.facet_source.tool_contact_ledger_row",
])
OLD_RELEASE_SHA = "2e51c14b4f1c3a122968bc0fd7809523470aad80923ff87c1ce9992ed8d3beb4"   # v2.2 intent-boundaries release (historical 268-B)
PORT_BASELINES = (
    "slurp-native-stage3-l338-lever-column-fold-b-baseline-raw-port-v1",
    "slurp-native-stage3-l338-lever-column-fold-b-smoke-4-baseline-raw-port-v1",
    "slurp-native-stage3-l338-lever-column-fold-a-baseline-raw-port-v1",
)
HISTORICAL_BASELINES = (
    "slurp-native-stage3-l338-lever-column-fold-b-baseline-v1",
    "slurp-native-stage3-l338-lever-column-fold-b-smoke-4-baseline-v1",
)


def _hooks(root: Path = REPO):
    return smh.carrier_module_hooks(repo_root=root)


def _json(layer) -> dict:
    return json.loads(json.dumps(dict(layer)))


def _baseline(name: str) -> dict:
    return json.loads((REPO / "configs" / "baseline" / f"{name}.json").read_text(encoding="utf-8"))


def _verify(doc: dict) -> str:
    return cm.verify_declared_module(doc["native_episode"]["consumption_module"], doc, repo_root=REPO)


# ---- the registry ------------------------------------------------------------------------------------------------------------------------------------------------


def test_the_port_lock_is_registered_and_is_not_the_current_lock():
    hooks = _hooks()
    assert LOCK == port.LOCK_V2_2_NPU_PORT and LOCK in hooks.lock_versions
    assert sorted(hooks.lock_versions) == ["v2.0", "v2.1", "v2.2", "v2.2-npu-port", "v2.3"]
    assert smh.CURRENT_LOCK == hooks.current_lock == "v2.3"
    assert {k: v() for k, v in hooks.lock_deliver_normalisers.items()}[LOCK] == V4G
    assert _json(hooks.lock_versions[LOCK]())["native_episode"]["deliver_normaliser"] == {"normaliser": V4G}


def test_the_port_lock_is_v2_2_with_exactly_the_ten_leaves_moved_to_the_pins():
    hooks = _hooks()
    v22, port_layer = _json(hooks.lock_versions["v2.2"]()), _json(hooks.lock_versions[LOCK]())
    assert sorted(cm.differing_paths(v22, port_layer)) == TEN and len(TEN) == 10
    assert len(port.PORT_LEAVES) == 10 and sorted(".".join(p) for p in port.PORT_LEAVES) == TEN
    leaves = cm._leaf_paths(port_layer)
    pin_leaves = cm._leaf_paths({"knowledge_stores": PINS["knowledge_stores"]})
    for leaf in TEN:
        assert leaves[leaf] == pin_leaves[leaf], leaf        # read from the pins file, never typed
        assert cm._leaf_paths(v22)[leaf] != leaves[leaf], leaf
    assert cm._leaf_paths(v22)["knowledge_stores.intent-boundaries.release.sha256"] == OLD_RELEASE_SHA
    # the facet-source relpath is the same on both sides; the roster pins are untouched
    assert leaves["knowledge_stores.train-exemplars.phase_supply.solver.facet_source.relpath"] == "vectors/slurp-kus3/facet-vectors.jsonl"
    assert port_layer["knowledge_stores"]["train-exemplars"]["release"] == v22["knowledge_stores"]["train-exemplars"]["release"]


def test_the_port_lock_layer_is_fail_closed_on_the_pins_file(tmp_path):
    layer_v22 = _hooks().lock_versions["v2.2"]()
    with pytest.raises(cm.ConsumptionModuleConfigError, match="pins file .* is missing"):
        port.layer_from_v2_2(layer_v22, tmp_path)
    # a pin equal to the v2.2 value is not a port pin
    pins_dir = tmp_path / "scripts" / "flights" / "nx"
    pins_dir.mkdir(parents=True)
    stale = copy.deepcopy(PINS)
    stale["knowledge_stores"]["intent-boundaries"]["release"]["sha256"] = OLD_RELEASE_SHA
    (pins_dir / "l338_port_pins.json").write_text(json.dumps(stale), encoding="utf-8")
    with pytest.raises(cm.ConsumptionModuleConfigError, match="intent-boundaries.release.sha256"):
        port.layer_from_v2_2(layer_v22, tmp_path)
    # a missing leaf is refused
    broken = copy.deepcopy(PINS)
    del broken["knowledge_stores"]["slot-contracts"]["snapshot"]
    (pins_dir / "l338_port_pins.json").write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(cm.ConsumptionModuleConfigError, match="slot-contracts.snapshot"):
        port.layer_from_v2_2(layer_v22, tmp_path)


@pytest.mark.parametrize("lock, ok", [("v2.2-npu-port", True), ("v2.2", True), ("v2.3", True), ("v2.2-", False), ("v2.2-Npu", False), ("2.2-npu-port", False), ("v2.2--x", False)])
def test_the_lock_key_grammar_admits_a_lowercase_suffix(lock, ok):
    block = copy.deepcopy(_baseline(PORT_BASELINES[0])["native_episode"]["consumption_module"])
    block["lock"] = lock
    if ok:
        cm.parse_consumption_module_block(block, repo_root=REPO)
    else:
        with pytest.raises(cm.ConsumptionModuleConfigError, match="must match"):
            cm.parse_consumption_module_block(block, repo_root=REPO)


# ---- the loader's check on the real cells -----------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", PORT_BASELINES)
def test_the_port_cells_pass_verify_declared_module_and_name_the_port_lock(name):
    doc = _baseline(name)
    assert doc["native_episode"]["consumption_module"]["lock"] == LOCK
    assert _verify(doc)


@pytest.mark.parametrize("name", HISTORICAL_BASELINES)
def test_the_historical_fold_b_cells_still_pass_on_v2_2(name):
    doc = _baseline(name)
    assert doc["native_episode"]["consumption_module"]["lock"] == "v2.2"
    assert _verify(doc)


@pytest.mark.parametrize("name", PORT_BASELINES)
def test_a_port_cell_on_the_old_lock_is_refused_at_the_ten_leaves(name):
    doc = copy.deepcopy(_baseline(name))
    doc["native_episode"]["consumption_module"]["lock"] = "v2.2"
    with pytest.raises(cm.ConsumptionModuleConfigError, match="differ from what its consumption_module"):
        _verify(doc)
    block = doc["native_episode"]["consumption_module"]
    composed = cm.compile_module(cm.parse_consumption_module_block(block, repo_root=REPO), _hooks(), repo_root=REPO)
    assert sorted(cm.differing_paths(composed.values_on_roster_of(doc), cm.policy_owned_values(doc))) == TEN


# ---- the writer self-check ----------------------------------------------------------------------------------------------------------------------------------------


def test_the_self_check_passes_the_committed_port_cells_and_names_the_report():
    docs = {name: _baseline(name) for name in PORT_BASELINES[:2]}
    experiments = {pf.CELLS[k]["cell"]: json.loads((REPO / "configs" / "experiment" / f"{pf.CELLS[k]['cell']}.json").read_text(encoding="utf-8")) for k in pf.ALL_KEYS}
    assert pdc.module_self_check(REPO, docs, experiments) == {name: "ok" for name in PORT_BASELINES[:2]}


def test_the_self_check_refuses_a_deliberately_wrong_lock_and_reports_an_exempt_one():
    wrong = copy.deepcopy(_baseline(PORT_BASELINES[0]))
    wrong["native_episode"]["consumption_module"]["lock"] = "v2.2"
    with pytest.raises(ValueError, match="verify_declared_module refuses .*nothing was written"):
        pdc.module_self_check(REPO, {PORT_BASELINES[0]: wrong}, {})
    # the experiment path: the emitted experiment names a baseline that is checked too
    experiment = json.loads((REPO / "configs" / "experiment" / f"{pf.CELLS['arm']['cell']}.json").read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="refuses"):
        pdc.module_self_check(REPO, {PORT_BASELINES[0]: wrong}, {pf.CELLS["arm"]["cell"]: experiment})
    report = pdc.module_self_check(REPO, {PORT_BASELINES[0]: wrong}, {}, exempt=(PORT_BASELINES[0],))
    assert report[PORT_BASELINES[0]].startswith("exempt: ")
    # a baseline that declares no module block is not checked
    assert pdc.module_self_check(REPO, {"plain": {"native_episode": {}}}, {}) == {"plain": "ok"}


def test_the_composed_check_exits_on_a_refused_config():
    doc = copy.deepcopy(_baseline(PORT_BASELINES[0]))
    pdc.verify_composed_module(REPO, doc, "port")
    doc["native_episode"]["consumption_module"]["lock"] = "v2.2"
    with pytest.raises(SystemExit, match="refused by verify_declared_module"):
        pdc.verify_composed_module(REPO, doc, "port")
    pdc.verify_composed_module(REPO, {"native_episode": {}}, "no module block")


def test_the_l338_derive_refuses_to_run_on_the_old_lock(monkeypatch):
    monkeypatch.setattr(pdc, "PORT_MODULE_LOCK", "v2.2")
    with pytest.raises(ValueError, match="verify_declared_module refuses"):
        pdc.derive(REPO, dry_run=True)


def test_the_fold_a_derive_refuses_to_run_on_the_old_lock(monkeypatch):
    monkeypatch.setattr(pdc, "PORT_MODULE_LOCK", "v2.2")
    with pytest.raises(ValueError, match="verify_declared_module refuses"):
        fdc.derive(REPO, dry_run=True)


# ---- re-mint determinism and the committed bytes -----------------------------------------------------------------------------------------------------------------


def _dump(doc: dict) -> str:
    return json.dumps(doc, ensure_ascii=True, indent=2, sort_keys=True) + "\n"


@pytest.mark.parametrize("key", pf.ALL_KEYS)
def test_the_l338_derive_is_deterministic_and_reproduces_the_committed_configs(key):
    spec = pf.CELLS[key]
    first = pdc.port_baseline(key, REPO, module_lock=pdc.PORT_MODULE_LOCK)
    second = pdc.port_baseline(key, REPO, module_lock=pdc.PORT_MODULE_LOCK)
    assert _dump(first) == _dump(second)
    assert _dump(first) == (REPO / "configs" / "baseline" / f"{spec['baseline']}.json").read_text(encoding="utf-8")
    exp = pdc.port_experiment(key, REPO)
    assert _dump(exp) == (REPO / "configs" / "experiment" / f"{spec['cell']}.json").read_text(encoding="utf-8")
    # the default keeps the historical block's lock (the T derive is switched in its own step)
    assert pdc.port_baseline(key, REPO)["native_episode"]["consumption_module"]["lock"] == "v2.2"


def test_the_fold_a_derive_is_deterministic_and_reproduces_the_committed_configs():
    first, second = fdc.fold_a_baseline(REPO), fdc.fold_a_baseline(REPO)
    assert _dump(first) == _dump(second)
    assert first["native_episode"]["consumption_module"]["lock"] == LOCK
    name = fdc.fa.CELLS["arm"]["baseline"]
    assert _dump(first) == (REPO / "configs" / "baseline" / f"{name}.json").read_text(encoding="utf-8")

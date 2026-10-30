"""The T cell of the three-way comparison (TFRL-v2 on the NPU port, iteration 1): ids, caps, programs, derive, writer, package.

Five model cells (the R01C construction smoke and its two interleaved halves, the Fold B smoke and the Fold B selection flight) are derived by
``scripts/flights/nx/tfrl_v2_port_derive_cells.py`` from the L338 port cell and written up by ``tfrl_v2_port_write_registration.py`` (reached through
``l338_port_write_registration.py --family tfrl-v2``). Every pin that does not exist yet is the zero sentinel. Nothing here contacts a model, a server or an encoder; every derive and
writer run goes into a throwaway root. The byte pins (``GOLDEN``) are the package and configs written at ``COMMIT`` / ``AT``; a change of the L338 port family moves them on purpose.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
for _extra in (REPO / "scripts" / "flights" / "nx", REPO / "scripts" / "flights" / "direct", REPO / "scripts" / "flights" / "tfrl", REPO / "tests" / "unit", REPO / "src"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import l338_family as fam  # noqa: E402
import l338_port_family as pf  # noqa: E402
import l338_port_write_registration as lw  # noqa: E402
import tfrl_v2_port_derive_cells as tdc  # noqa: E402
import tfrl_v2_port_family as tf  # noqa: E402
import tfrl_v2_port_write_registration as tw  # noqa: E402
import tfrl_v2_sampler as smp  # noqa: E402
from test_l338_port_family_switch import DATA_ROOT, make_root, needs_data  # noqa: E402

from speech_aware_evidence_acquisition.core.config import compose, protocol_hash  # noqa: E402
from speech_aware_evidence_acquisition.core.preparation_fit import native_contact_required_cap  # noqa: E402
from speech_aware_evidence_acquisition.rails import consumption_module as cm  # noqa: E402
from speech_aware_evidence_acquisition.rails.adapters.slurp_module_hooks import carrier_module_hooks  # noqa: E402

CJK = re.compile("[\\u4e00-\\u9fff]")
ZERO = "0" * 64
COMMIT = "28bf3176ba9798208ae1e1e22ecf6412efa91755"
AT = "2026-10-30T00:00:00+00:00"
#: composed-config hash / protocol hash of each T cell at the default fragment.
HASHES = {
    "csmoke": ("c656076e3a2ffa162a5f96188b3b4f6f4d6a8ad93cc6c003b9a6b8fa0f36122c", "6fa9cf6da9020031d460a35cc269feb08c970465e0ae0c85f2d1a15a268cf75c"),
    "ch1": ("86125d03d68c4cefb01cf3b1031842509036e80eeeb4f8d694b74ee1cbf40a85", "36f08eeb02f8f5e020876277b2a72ad8dbbb550d3b0dd0f3ce8c820a2b5b67da"),
    "ch2": ("f1a9749d0ec04e665b5e6c5599c2ae803f94a599be531072bb2554050f70fabb", "16db3d67fba5fdaaca8aa7a214753c3ce41af824515d72f3dacafefd846d5259"),
    "fsmoke": ("b6669e39e1248fb98da80d6dc4e82ffc81e0d4bd535dd9293b27f0e5a45c9907", "3bb9257e69948172a2e60512987d2e8297833fec6b075e8d8e541fee7a108db3"),
    "fb": ("9edcbb4f2337fac6468b2ec22d2c0282274268e6d7bc7e2ae35937f7a7a30c18", "edcffc2564f3a6a2913d2eafca300c9df84928b131ee57707e6dca38aa1b4082"),
}
CAPS = {"csmoke": (56, 28, 239), "ch1": (4382, 2191, 18474), "ch2": (4368, 2184, 19703), "fsmoke": (80, 40, 204), "fb": (21600, 10800, 63096)}
#: sha256 of every file the writer emits for the default fragment at COMMIT / AT (the five baselines, five experiments, two campaigns, the package and the preregistration).
GOLDEN = {
    "configs/baseline/slurp-native-stage3-tfrl-v2-construction-r01c-i1-h1-baseline-raw-port-v1.json": "5011a02c319efa5473c55290227e67a1188b85349da6fecdc26dc798fa1fc773",
    "configs/baseline/slurp-native-stage3-tfrl-v2-construction-r01c-i1-h2-baseline-raw-port-v1.json": "d5ca613c33efcfcacf21aae8ee07208f99ddf8700d7a12bb9a0f49c5ffc4244b",
    "configs/baseline/slurp-native-stage3-tfrl-v2-construction-r01c-smoke-4-baseline-raw-port-v1.json": "3aaf6a2995797472946676d24fa7204132686d9d461389b4ba2d8de82907bf3d",
    "configs/baseline/slurp-native-stage3-tfrl-v2-fold-b-i1-baseline-raw-port-v1.json": "719573e3f37d63654e49570ea8df1a6ef933812cdcdba5cd064bb5bc6bd7b5b7",
    "configs/baseline/slurp-native-stage3-tfrl-v2-fold-b-smoke-4-baseline-raw-port-v1.json": "02d19f826476ff0d0b5896eada0febb57ccfc7f35d4c6790c9e163a38b8625fc",
    "configs/campaign/slurp-native-stage3-tfrl-v2-construction-r01c-i1-npu-port-v1.json": "13377668cfa68ca7a7d9336a06646120f8a1279882da3802daa3750be4633e23",
    "configs/campaign/slurp-native-stage3-tfrl-v2-fold-b-i1-npu-port-v1.json": "d2b3d507921ffabf8826940d107d9eba4c8e28cba736562892853b9a45aa94b3",
    "configs/experiment/slurp-native-stage3-tfrl-v2-construction-r01c-i1-h1-npu-port-v1.json": "f2cbf84afe6997d6fb47da09f62e1456afa1eeb7e13ea36ad5f39cd5fb735680",
    "configs/experiment/slurp-native-stage3-tfrl-v2-construction-r01c-i1-h2-npu-port-v1.json": "6d580004d62723b9c3b762078e58afc8461e38b4cc738547fab7beb1bf798ea4",
    "configs/experiment/slurp-native-stage3-tfrl-v2-construction-r01c-smoke-4-npu-port-v1.json": "83483b2db9bdd2dc00efeb86e8c90c137485a6d931839aa041ea8ea99b5bac3c",
    "configs/experiment/slurp-native-stage3-tfrl-v2-fold-b-i1-npu-port-v1.json": "6d67464d551405b3b3507d9ca6c5c2435fc455a9af2115824673224642cd6b4a",
    "configs/experiment/slurp-native-stage3-tfrl-v2-fold-b-smoke-4-npu-port-v1.json": "a430b03bd8a0a76381bb3be0910621bc9ed8b0509a6cddbcd817cd990c05ac1a",
    "docs/readiness/2026-10-30-slurp-tfrl-v2-fold-b-npu-port-v1/construction-i1-h1-fit-plan-registered-v1.json": "eb4c6d6f4464b735bd7968f9ea8c787dfb0e57e6539a9cbdb8127c6be6a5e568",
    "docs/readiness/2026-10-30-slurp-tfrl-v2-fold-b-npu-port-v1/construction-i1-h1-generation-plan-registered-v1.json": "c1851576231def2f753bcce70ca800b2981e9f5584b6c4dabfde1ecf046647d2",
    "docs/readiness/2026-10-30-slurp-tfrl-v2-fold-b-npu-port-v1/construction-i1-h2-fit-plan-registered-v1.json": "fb43e99d559b3eae291100c6235e91de2aef7544877ad097c261cc1edb9091b4",
    "docs/readiness/2026-10-30-slurp-tfrl-v2-fold-b-npu-port-v1/construction-i1-h2-generation-plan-registered-v1.json": "efc9ec2a42fc8aad82cd9f8f92d0da06ddb2d4f734d606d0829c83cb46682006",
    "docs/readiness/2026-10-30-slurp-tfrl-v2-fold-b-npu-port-v1/construction-smoke-fit-plan-registered-v1.json": "389c3f33a2f90e9146c1965a8d518f20399bdd47fb21e5f8fee1558da5b22e21",
    "docs/readiness/2026-10-30-slurp-tfrl-v2-fold-b-npu-port-v1/construction-smoke-generation-plan-registered-v1.json": "40fc2cca3444eba3725b3ec3f9b7252d9b423e781452f9472d85d391902b85de",
    "docs/readiness/2026-10-30-slurp-tfrl-v2-fold-b-npu-port-v1/countersign-row-texts-v1.json": "a5644190f19c802de4fddea46157e57a7aa7ae08865249ac4ff84a93c12115e1",
    "docs/readiness/2026-10-30-slurp-tfrl-v2-fold-b-npu-port-v1/draft-ledger-rows.md": "a0107e9af83b77d65e0ee9e882e02fcda8d75a861dde2af2efea2e5a790ce96a",
    "docs/readiness/2026-10-30-slurp-tfrl-v2-fold-b-npu-port-v1/foldb-i1-fit-plan-registered-v1.json": "17c0fd04ab52c77b39a091d854b7a598cd4a8fd2d60daae23430d7e8530d442c",
    "docs/readiness/2026-10-30-slurp-tfrl-v2-fold-b-npu-port-v1/foldb-i1-generation-plan-registered-v1.json": "9c7c141c7dd3aa7fa7881e30faadcc82fb358bba5e37d636dc61036b3d4c6fe2",
    "docs/readiness/2026-10-30-slurp-tfrl-v2-fold-b-npu-port-v1/foldb-smoke-fit-plan-registered-v1.json": "f18c50078dcfd2637ace76429c6a1f0c2d2d571426045ce2e710afaac810d8e9",
    "docs/readiness/2026-10-30-slurp-tfrl-v2-fold-b-npu-port-v1/foldb-smoke-generation-plan-registered-v1.json": "100a33795193b2946edb2b0654fad87468dcea6bcb9f1473881979b2ba9a56b1",
    "docs/readiness/2026-10-30-slurp-tfrl-v2-fold-b-npu-port-v1-preregistration.md": "2b691f65e92748e2c57df2d8cc671f0f52c195344cf9c23f65aeae8c55152fb0",
    "docs/readiness/2026-10-30-slurp-tfrl-v2-fold-b-npu-port-v1/protocol-registered-v1.json": "9ff2c8e3f154179a363178fee36512e9cf8a40d48c5e1d1b7bbc8c6e67c96e00",
    "docs/readiness/2026-10-30-slurp-tfrl-v2-fold-b-npu-port-v1/README.md": "7739897df69c268d48a5e5d85a61cc33cf06382c095d33f7e631d4df8b0870d9",
    "docs/readiness/2026-10-30-slurp-tfrl-v2-fold-b-npu-port-v1/registered-facts-v1.json": "be34ea2dacf271f4d26faf452769dc5b6d2bd97cd29c52790860f217a3cccf84",
    "docs/readiness/2026-10-30-slurp-tfrl-v2-fold-b-npu-port-v1/root-admission-draft-v1.json": "001bb9b987ca7e8f514aa41272bca0d3a81f9d06b654a4dab0ab1b82322aa058",
    "docs/readiness/2026-10-30-slurp-tfrl-v2-fold-b-npu-port-v1/signoff-conditions-construction-v1.json": "aac5f0d52d2eeb665bce9e709ee1842dac81d399a383d2f2bdc24302df9e9a01",
    "docs/readiness/2026-10-30-slurp-tfrl-v2-fold-b-npu-port-v1/signoff-conditions-fold-b-v1.json": "e00755ca504dd7156ebb0d0d6ab74ecdd94466324ddee435b3264c87dec9ec57",
}
T_FILE_MARK = "tfrl-v2"
PKG_DIR = Path(tf.PACKAGE).name
PREREG_NAME = Path(tf.PREREG).name


@pytest.fixture(autouse=True)
def _restore_default_fragment():
    pf.select_model_fragment("default")
    tf.select_model_fragment()
    yield
    pf.select_model_fragment("default")
    tf.select_model_fragment()


def _t_config_names() -> list[str]:
    names = []
    for key in tf.ALL_KEYS:
        names += [f"{tf.CELLS[key]['baseline']}.json", f"{tf.CELLS[key]['cell']}.json"]
    names += [f"{name}.json" for name in tdc.t_campaigns()]
    return names


def make_t_root(dst: Path) -> Path:
    """``make_root`` with every T path a real copy or absent, so no derive or writer run can write through a symlink into the study tree."""

    root = make_root(dst, packages=(), copy_config_files=_t_config_names())
    for child in (root / "docs" / "readiness").iterdir():
        if child.name in (PKG_DIR, PREREG_NAME) and child.is_symlink():
            child.unlink()
    return root


@pytest.fixture(scope="module")
def derived_root(tmp_path_factory):
    pf.select_model_fragment("default")
    root = make_t_root(tmp_path_factory.mktemp("tfrl_v2") / "root")
    tdc.derive(root)
    return root


@pytest.fixture(scope="module")
def written_root(tmp_path_factory):
    if not (DATA_ROOT / "prepared").is_dir():
        pytest.skip("the data root with the port preparation receipts is not mounted")
    pf.select_model_fragment("default")
    root = make_t_root(tmp_path_factory.mktemp("tfrl_v2_w") / "root")
    result = tw.write_package(root, COMMIT, AT, DATA_ROOT)
    return root, result


def _load(root: Path, kind: str, name: str) -> dict:
    return json.loads((root / "configs" / kind / f"{name}.json").read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _leaves(node, prefix=""):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _leaves(value, f"{prefix}.{key}" if prefix else key)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _leaves(value, f"{prefix}[{index}]")
    else:
        yield prefix, node


# ---- ids, rosters, caps -------------------------------------------------------------------------------------------------------------------------


def test_ids_follow_the_port_rule_and_match_the_sampler():
    assert tf.ALL_KEYS == ("csmoke", "ch1", "ch2", "fsmoke", "fb") and tf.ITERATION == 1
    assert tf.CELL_ID == "SAEA-SLURP-TFRL-V2-FOLD-B-NPU-PORT-V1" and tf.DATE == "2026-10-30"
    assert tf.DECISION_ID == "slurp-tfrl-v2-npu-port-admission-v1" and tf.DECISION_RELPATH == "coordination/slurp-tfrl-v2-npu-port-v1/root-admission-v1.json"
    assert tf.PACKAGE == "docs/readiness/2026-10-30-slurp-tfrl-v2-fold-b-npu-port-v1"
    runs = {k: tf.CELLS[k]["run"] for k in tf.ALL_KEYS}
    assert runs == {"csmoke": "SAEA-SLURP-TFRL-V2-CONSTRUCTION-R01C-SMOKE-4-NPU-PORT-V1", "ch1": "SAEA-SLURP-TFRL-V2-CONSTRUCTION-R01C-I1-H1-NPU-PORT-V1",
                    "ch2": "SAEA-SLURP-TFRL-V2-CONSTRUCTION-R01C-I1-H2-NPU-PORT-V1", "fsmoke": "SAEA-SLURP-TFRL-V2-FOLD-B-SMOKE-4-NPU-PORT-V1",
                    "fb": "SAEA-SLURP-TFRL-V2-FOLD-B-I1-NPU-PORT-V1"}
    for key in tf.ALL_KEYS:
        spec = tf.CELLS[key]
        assert spec["fit_run"] == spec["run"].replace("-NPU-PORT-V1", "-FIT-NPU-PORT-V1")
        assert spec["fit_attempt"] == spec["attempt"].replace("-generation-", "-fit-")
        assert spec["cell"].startswith("slurp-native-stage3-tfrl-v2-") and spec["cell"].endswith("-npu-port-v1") and spec["baseline"].endswith("-baseline-raw-port-v1")
    # the halves' ids are the committed sampler's own, so the run list, the trace READ id and the cell agree
    for key, half in (("ch1", "R01C-H1"), ("ch2", "R01C-H2")):
        assert tf.CELLS[key]["run"] == smp.run_id_for(half, 1) and tf.CELLS[key]["attempt"] == smp.attempt_id_for(half, 1)
    assert tf.trace_read_id(1, "H1") == "SAEA-SLURP-TFRL-V2-FOLD-B-READ-NPU-PORT-V1-I1-H1" == tf.CELLS["ch1"]["trace_read_id"]
    assert tf.CELLS["ch2"]["trace_read_id"].endswith("-I1-H2") and tf.CELLS["fb"]["trace_read_id"] is None
    with pytest.raises(ValueError):
        tf.trace_read_id(1, "H3")
    assert tf.LOOP_ID == smp.cm.LOOP_ID == "SAEA-SLURP-TFRL-V2-NPU-PORT-V1"
    assert len(set(tf.eligible_run_ids())) == 10 and len(set(tf.eligible_attempt_ids())) == 10


def test_rosters_rows_and_halves_are_the_real_identities():
    assert [tf.CELLS[k]["rows"] for k in tf.ALL_KEYS] == [4, 313, 312, 4, 1080]
    assert tf.CELLS["ch1"]["split_identity_hash"].startswith("1a88c40b") and tf.CELLS["ch2"]["split_identity_hash"].startswith("c7b16e13")
    assert tf.CELLS["csmoke"]["split_identity_hash"].startswith("e9dcea53") and tf.CELLS["fsmoke"]["split_identity_hash"].startswith("bdd025a1")
    assert tf.CELLS["fb"]["split_identity_hash"] == "4407794c4a5d2e8883d93fdf242deb33c0987405d62e05a7d8416f5251a3484a"
    parent = [i["sample_id"] for i in json.loads((REPO / tf.R01C_ROSTER).read_text(encoding="utf-8"))["items"]]
    h1 = [i["sample_id"] for i in tf._roster_doc("ch1")["items"]]
    h2 = [i["sample_id"] for i in tf._roster_doc("ch2")["items"]]
    assert h1 == parent[0::2] and h2 == parent[1::2] and len(parent) == 625
    assert tf.CELLS["ch1"]["dataset"] == tf.CELLS["csmoke"]["dataset"] == "slurp-speech-native-stage3-construction-r01c" and tf.CELLS["fb"]["dataset"] == "slurp-speech-native-evo-fold-b-1080"


def test_caps_follow_the_port_formulas_and_the_gate_helpers():
    assert {k: (tf.generation_calls(k), tf.fit_calls(k), tf.audio_seconds(k)) for k in tf.ALL_KEYS} == CAPS
    for key in tf.ALL_KEYS:
        rows, c = tf.CELLS[key]["rows"], tf.contacts_per_row(key)
        assert c == (7 if tf.kind_of(key) == "construction" else 10)
        assert tf.generation_calls(key) == rows * 2 * c == 2 * tf.program_calls(key)
        assert tf.fit_calls(key) == native_contact_required_cap(rows * c, feature_packets=False) == rows * c * pf.RAW_FIT_ENDPOINTS_PER_CONTACT
        assert tf.audio_seconds(key) == math.ceil(tf.roster_audio(key) * 2 * c - 1e-9)
    assert (tf.cell_total(0), tf.cell_total(1), tf.cell_total(2)) == (30486, 15243, 101716)
    assert tf.contacts("ch1") == (1, 6) and tf.contacts("fb") == (4, 6)
    assert (tf.mix_rows("ch1"), tf.mix_rows("fb")) == (3.6, 5.04)
    assert tf.program_id("ch1") == "phase-native-module-v1-c7" and tf.program_id("fb") == "phase-native-module-v1-c10"


def test_card_hours_are_provisional_formulas_never_numbers():
    assert tf.plan_gpu_hours_placeholder("csmoke") == tf.plan_gpu_hours_placeholder("fsmoke") == 0.5
    assert (tf.plan_gpu_hours_placeholder("ch1"), tf.plan_gpu_hours_placeholder("ch2"), tf.plan_gpu_hours_placeholder("fb")) == (12.6, 12.5, 60.5)
    assert tf.card_hours(1080, 5.04, lock_rows_per_minute_per_instance=fam.ROWS_PER_MINUTE_FULL, tensor_parallel_size=2) == pytest.approx(121.0)
    for key in tf.ALL_KEYS:
        assert tf.ledger_gpu_cell(key).startswith("PROVISIONAL") and "must be" not in tf.ledger_gpu_cell(key)
        assert tf.card_hour_formula(key).startswith("PROVISIONAL")
    with pytest.raises(ValueError):
        tf.card_hours(10, 1.0, lock_rows_per_minute_per_instance=0, tensor_parallel_size=1)


# ---- the context fit and the thinking-budget switch -----------------------------------------------------------------------------------------------


def test_default_fits_and_the_switch_does_not():
    table = tf.fit_table()
    assert table["default"] == {"a_bound": 55, "t_bound": 47, "base_prompt_tokens": 4723, "estimate_tokens": 15768, "solver_prompt_limit_tokens": 17406, "fits": True,
                                "headroom_tokens": 1638, "largest_fitting_bound": 53}
    assert table["budget6144-max7168"]["t_bound"] == 43 and table["budget6144-max7168"]["estimate_tokens"] == 14828 and not table["budget6144-max7168"]["fits"]
    assert table["budget6144-max7168"]["largest_fitting_bound"] == 40 and table["budget6144-max7168"]["solver_prompt_limit_tokens"] == 14334
    assert tf.t_bound() == 47 and tf.a_bound() == 55 and tf.fold_b_fit_base_tokens() == 2000 + fam.DELIVERING_TEMPLATE_TOKENS + 2000 + 400
    tf.require_fit()                                                                    # the default composes
    pf.select_model_fragment("budget6144-max7168")
    assert (tf.a_bound(), tf.t_bound(), tf.largest_fitting_bound()) == (51, 43, 40)
    with pytest.raises(SystemExit, match="largest fitting T bound is 40 items"):
        tf.require_fit()


def test_derive_and_writer_refuse_under_the_switch_and_write_nothing(tmp_path):
    root = make_t_root(tmp_path / "root")
    before = sorted(p.name for p in (root / "configs" / "baseline").iterdir())
    pf.select_model_fragment("budget6144-max7168")
    with pytest.raises(SystemExit, match="largest fitting T bound is 40"):
        tdc.derive(root)
    with pytest.raises(SystemExit, match="largest fitting T bound is 40"):
        tw.write_package(root, COMMIT, AT, DATA_ROOT, derive=False)
    assert sorted(p.name for p in (root / "configs" / "baseline").iterdir()) == before
    assert not (root / "docs" / "readiness" / PKG_DIR).exists()


def test_the_largest_fitting_bound_composes_under_the_switch(tmp_path, monkeypatch):
    root = make_t_root(tmp_path / "root")
    pf.select_model_fragment("budget6144-max7168")
    monkeypatch.setattr(tf, "EXEMPLAR_ITEM_EQUIVALENT", 11)                              # 51 - 11 = 40, the largest fitting bound
    assert tf.t_bound() == tf.largest_fitting_bound() == 40
    tf.require_fit()
    base = tdc.t_baseline("fb", root)
    frame = base["native_episode"]["consumption_module"]["sampling_frame"]
    assert frame["max_supplied_items"] == 40 and frame["fit_estimate"]["completion_ceiling"] == 7168 and base["policy"] == "phase-native-module-v1-c10"
    assert base["tfrl_v2"]["supply_bound_items"] == 40 and base["tfrl_v2"]["a_supply_bound_items"] == 51
    monkeypatch.setattr(tf, "EXEMPLAR_ITEM_EQUIVALENT", 10)                              # 41 items: one past the bound, refused
    with pytest.raises(SystemExit):
        tf.require_fit()


# ---- the programs, by the real module compiler ---------------------------------------------------------------------------------------------------


def _hooks(root: Path):
    return carrier_module_hooks(repo_root=root)


def test_construction_group_compiles_to_c7_and_the_alternatives_to_c10_and_c9(tmp_path):
    import l338_port_derive_cells as ldc  # noqa: PLC0415

    root = make_t_root(tmp_path / "root")
    block = ldc.port_baseline("arm", root)["native_episode"]["consumption_module"]
    compiled = cm.compile_module(tdc.construction_block(block), _hooks(root), repo_root=root)
    assert (compiled.program_id, compiled.program_ceiling) == ("phase-native-module-v1-c7", 7)
    decoding = tdc.construction_block(block)["columns"]
    assert [c["column_id"] for c in decoding] == ["s1", "s2", "s3", "s4"] and all(c["kind"] == "decoding" and c["node"] == "solver" for c in decoding)
    tops = [c["decoding"]["top_p"] for c in decoding]
    assert tops == [0.8, 0.9, 0.95, 1.0] and len(set(tops)) == 4 and {c["decoding"]["temperature"] for c in decoding} == {0.7}
    assert "sampling_frame" not in tdc.construction_block(block) and "construction_roster" not in tdc.construction_block(block)
    alt = cm.compile_module(tdc.construction_block(block, "perception-sampling"), _hooks(root), repo_root=root)
    assert (alt.program_id, alt.program_ceiling) == ("phase-native-module-v1-c10", 10) and tf.contacts("ch1", "perception-sampling") == (3, 7)
    n4 = tdc.construction_block(block, "perception-sampling")
    n4["columns"][0]["solver_samples"]["n"] = 4
    assert cm.compile_module(n4, _hooks(root), repo_root=root).program_ceiling == 9
    with pytest.raises(ValueError):
        tdc.construction_block(block, "no-such-group")


def test_the_solver_decoding_group_refuses_a_seed_override_and_equal_overrides(tmp_path):
    import l338_port_derive_cells as ldc  # noqa: PLC0415

    root = make_t_root(tmp_path / "root")
    block = ldc.port_baseline("arm", root)["native_episode"]["consumption_module"]
    seeded = tdc.construction_block(block)
    seeded["columns"][0]["decoding"]["seed"] = 7
    with pytest.raises(Exception):
        cm.compile_module(seeded, _hooks(root), repo_root=root)
    equal = tdc.construction_block(block)
    equal["columns"][1]["decoding"] = dict(equal["columns"][0]["decoding"])
    with pytest.raises(Exception):
        cm.compile_module(equal, _hooks(root), repo_root=root)


def test_fold_b_column_is_one_sampling_vote_and_compiles_to_c10(tmp_path):
    import l338_port_derive_cells as ldc  # noqa: PLC0415

    root = make_t_root(tmp_path / "root")
    block = tdc.fold_b_block(ldc.port_baseline("arm", root)["native_episode"]["consumption_module"])
    assert len(block["columns"]) == 1 and block["columns"][0]["column_id"] == "t-v2"
    votes = block["columns"][0]["solver_samples"]
    assert votes["n"] == 3 and votes["decoding"] == {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
    assert votes["commit"] == {"policy": "saea-commit-plurality-support-v1", "min_support": 2, "similarity_threshold": 0.8}
    assert block["sampling_frame"]["max_supplied_items"] == 47 and block["sampling_frame"]["fit_estimate"]["base_prompt_tokens"] == 4723
    compiled = cm.compile_module(block, _hooks(root), repo_root=root)
    assert (compiled.program_id, compiled.program_ceiling) == ("phase-native-module-v1-c10", 10)
    assert 2 + 2 * (4 - 1) + (3 - 1) == 10 == tf.contacts_per_row("fb")


# ---- the derive -----------------------------------------------------------------------------------------------------------------------------------


def test_derive_writes_exactly_the_t_configs_as_real_files(derived_root):
    for key in tf.ALL_KEYS:
        for kind, name in (("baseline", tf.CELLS[key]["baseline"]), ("experiment", tf.CELLS[key]["cell"])):
            path = derived_root / "configs" / kind / f"{name}.json"
            assert path.is_file() and not path.is_symlink(), name
            assert not CJK.search(path.read_text(encoding="utf-8"))
    for name in tdc.t_campaigns():
        assert (derived_root / "configs" / "campaign" / f"{name}.json").is_file()
    campaigns = tdc.t_campaigns()
    assert [c["cell"] for c in campaigns["slurp-native-stage3-tfrl-v2-construction-r01c-i1-npu-port-v1"]["cells"]] == [tf.CELLS["ch1"]["cell"], tf.CELLS["ch2"]["cell"]]
    assert [c["cell"] for c in campaigns["slurp-native-stage3-tfrl-v2-fold-b-i1-npu-port-v1"]["cells"]] == [tf.CELLS["fb"]["cell"]]


def test_derive_is_deterministic_and_a_fixed_point(derived_root, tmp_path):
    before = {(kind, p.name): _sha(p) for kind in ("baseline", "experiment", "campaign") for p in (derived_root / "configs" / kind).iterdir() if T_FILE_MARK in p.name}
    result = tdc.derive(derived_root)
    after = {(kind, p.name): _sha(p) for kind in ("baseline", "experiment", "campaign") for p in (derived_root / "configs" / kind).iterdir() if T_FILE_MARK in p.name}
    assert before == after and len(after) == 12
    for key in tf.ALL_KEYS:
        assert result[key]["versus_disk"] == {"baseline": {"moved": []}, "experiment": {"moved": []}}


@pytest.mark.parametrize("key", tf.ALL_KEYS)
def test_cell_composes_with_the_registered_program_caps_and_hashes(derived_root, key):
    spec = tf.CELLS[key]
    exp = _load(derived_root, "experiment", spec["cell"])
    cell = exp["cells"][0]
    composed = compose(derived_root, cell["model_fragment"], cell["dataset_fragment"], cell["baseline_fragment"], spec["cell"])
    values = composed.values
    assert values["policy"] == tf.program_id(key) and values["arm"] == "raw-speech-agent-native-tools" and "speech_packet_promotion_receipt" not in values
    assert values["native_contact_fit"] == {"run_id": spec["fit_run"], "attempt_id": spec["fit_attempt"], "contact_cap": CAPS[key][1], "max_speech_packets": 1}
    assert (cell["run_id"], cell["attempt_id"], cell["model_fragment"], cell["dataset_fragment"]) == (spec["run"], spec["attempt"], pf.MODEL_FRAGMENT, spec["dataset"])
    assert (composed.config_hash, protocol_hash(composed.config_hash, values["arm"], exp["population"]["split_identity_hash"])) == HASHES[key]
    budget = exp["family_budget"]
    assert (budget["generation_calls_cap"], budget["fit_endpoint_calls_cap"], budget["repeated_speech_audio_seconds_cap"]) == CAPS[key]
    assert exp["execution_plan"]["planned_model_calls"] == CAPS[key][0] and exp["generation_budget"]["cap"] == CAPS[key][0]
    assert exp["population"]["count"] == spec["rows"] and exp["population"]["roster_receipt"] == spec["roster"] and exp["population"]["split_identity_hash"] == spec["split_identity_hash"]
    assert exp["populations"][0]["ids"] == [i["sample_id"] for i in tf._roster_doc(key)["items"]] and exp["populations"][0]["dataset_fragment"] == spec["dataset"]
    assert cell["readiness"]["state"] == "held" and exp["service_session"] == pf.SERVICE_SESSION and "tfrl_v2" in values


def test_construction_baselines_pin_the_minus_r01_pool_and_pending_r01c_artifacts(derived_root):
    for key in tf.CONSTRUCTION_KEYS:
        base = _load(derived_root, "baseline", tf.CELLS[key]["baseline"])
        module = base["native_episode"]["consumption_module"]
        assert "construction_roster" not in module and "sampling_frame" not in module and module["decoding_frame"]["twin"] is True
        roles = base["native_episode"]["roles"]
        assert "sampling_variants" not in roles["perception"]
        assert [a["arm_id"] for a in roles["solver"]["decoding_variants"]["arms"]] == ["s1", "s2", "s3", "s4", "aa", "lock"]
        pool = base["knowledge_stores"]["train-exemplars"]
        assert pool["release"]["relpath"].endswith("size-pool-annotated-exemplars-minus-r01/release.json")
        assert pool["release"]["sha256"] == "83620eb0b02f12424a00cd5572904f4df48a49c329b1b0e1a9321d4b9ec14fd0"
        assert pool["snapshot"]["sha256"] == "6eaa03aea96a1928dd1ac29ecc71c3955c136498c9f8114266eaa830d1207c63"
        perception = base["knowledge_stores"]["intent-boundaries"]["phase_supply"]["perception"]
        for source in (pool["phase_supply"]["solver"]["query_source"], perception["query_source"], perception["agreement_source"]):
            assert source["sha256"] == ZERO and "r01c" in source["relpath"].lower() and "-npu-port-v1" in source["relpath"].lower()
        assert "FOLD-B" not in json.dumps(base["knowledge_stores"]) and "fold-b" not in json.dumps(base["knowledge_stores"])
        assert module["components"][3]["params"]["agreement_source"] == perception["agreement_source"]
        # nothing of the Fold B cell leaks into a construction cell outside the declared prose leaves
        leaks = [p for p, v in _leaves(base) if isinstance(v, str) and re.search(r"fold[- ]b", v, re.I) and not p.startswith(("what_this_is", "tfrl_v2"))]
        assert leaks == [], leaks
        assert base["what_the_train_exemplars_store_is"] == tdc.POOL_TEXT
        receipts = {r["relpath"]: r["sha256"] for r in base["preparation_source_receipts"]}
        assert not any("FOLD-B" in rel or "filler-fold-b" in rel or "evo-fold-b" in rel for rel in receipts)
        assert sum(1 for sha in receipts.values() if sha == ZERO) == 4
        assert receipts[tf.CELLS[key]["roster"]] == _sha(REPO / tf.CELLS[key]["roster"])
        assert "prepared_phase_supply" not in base
    halves = _load(derived_root, "baseline", tf.CELLS["ch1"]["baseline"])
    assert {r["relpath"] for r in halves["preparation_source_receipts"]} >= {tf.R01C_ROSTER, tf.HALVES_SPLIT_RECEIPT}


def test_fold_b_baselines_carry_the_t_arm_table_bound_and_the_tfrl_block(derived_root):
    for key in tf.FOLD_B_KEYS:
        base = _load(derived_root, "baseline", tf.CELLS[key]["baseline"])
        module = base["native_episode"]["consumption_module"]
        assert [c["column_id"] for c in module["columns"]] == ["t-v2"] and module["sampling_frame"]["replicates"] == ["lock-rep2"]
        assert module["sampling_frame"]["max_supplied_items"] == 47 and module["sampling_frame"]["fit_estimate"]["base_prompt_tokens"] == 4723
        arms = base["native_episode"]["roles"]["perception"]["sampling_variants"]
        assert [a["arm_id"] for a in arms["arms"]] == ["t-v2", "aa", "lock-rep2", "lock"] and arms["max_supplied_items"] == 47
        block = base["tfrl_v2"]
        assert block["supply_bound_items"] == 47 == block["a_supply_bound_items"] - 8 and block["exemplar_policy"] == "replaces supplied items"
        assert block["caps"] == {"exemplar_block_tokens": 2000, "rule_bank_tokens": 400}
        assert block["context_fit"]["estimated_prompt_tokens_at_bound"] == 15768 and block["context_fit"]["largest_fitting_bound"] == 53
        assert block["vote_harness"]["n"] == 3 and block["vote_harness"]["commit"]["policy"] == "saea-commit-plurality-support-v1"
        assert block["vote_harness"]["planned_commit_policy"] == "saea-commit-vote-v1"
        for name, store in block["banks"].items():
            assert store["release"]["sha256"] == ZERO and store["snapshot"]["sha256"] == ZERO and store["store_role"] in ("tfrl-exemplar-bank", "tfrl-rule-bank")
        assert {item["id"] for item in block["rails_pending"]} >= {"bank-store-roles", "vote-commit-policy", "exemplar-block", "top-rule-per-vote", "stage3-algorithm-label"}
        assert "tfrl-exemplar-bank" not in base["knowledge_stores"] and "prepared_phase_supply" not in base
        assert "cross_fit" in block and "union" in block
    fold = _load(derived_root, "baseline", tf.CELLS["fb"]["baseline"])
    ref = _load(derived_root, "baseline", pf.CELLS["arm"]["baseline"])
    assert fold["knowledge_stores"] == ref["knowledge_stores"] and fold["preparation_source_receipts"] == ref["preparation_source_receipts"]


def test_construction_tfrl_block_is_seed_only_with_no_bank_consumed(derived_root):
    base = _load(derived_root, "baseline", tf.CELLS["ch2"]["baseline"])
    block = base["tfrl_v2"]
    assert block["kind"] == "trajectory-sampling" and block["mode"] == "seed-only" and block["k_trajectories"] == 4 and block["trajectory_arms"] == ["s1", "s2", "s3", "s4"]
    assert block["half"] == "R01C-H2" and block["other_half"] == "R01C-H1" and block["banks_consumed"] is None and block["rule_bank_i0"]["consumed_by_this_cell"] is False
    assert block["trace_read_id"] == tf.trace_read_id(1, "H2") and {i["id"] for i in block["rails_pending"]} >= {"construction-variant-port-names", "solver-decoding-seed-override"}
    smoke = _load(derived_root, "baseline", tf.CELLS["csmoke"]["baseline"])["tfrl_v2"]
    assert smoke["half"] is None and smoke["other_half"] is None and smoke["trace_read_id"] is None


def test_experiments_carry_pending_requirements_labels_and_texts(derived_root):
    for key in tf.ALL_KEYS:
        exp = _load(derived_root, "experiment", tf.CELLS[key]["cell"])
        reqs = exp["cells"][0]["readiness"]["requirements"]
        assert reqs[0]["relpath"] == tf.DECISION_RELPATH and reqs[0]["sha256"] == ZERO and reqs[0]["expected_decision_id"] == tf.DECISION_ID
        assert reqs[2]["relpath"] == tf.CELLS[key]["supply_relpath"] and reqs[2]["sha256"] == ZERO
        if tf.kind_of(key) == "construction":
            assert reqs[1]["sha256"] == ZERO and "AGREEMENT-R01C-NPU-PORT-V1" in reqs[1]["relpath"]
        else:
            assert reqs[1]["sha256"] == "db4614a3f8f644c8514cd76bf596446cc9b7043eb222a97973eb8b807f113918"
        label = exp["stage3_algorithm"]
        assert label["id"] == ("tfgrpo-semantic-advantage-v1" if key == "fb" else "shared-rollout") and label["note"]
        assert exp["purpose"].startswith(tf.CELLS[key]["run"]) and tf.PACKAGE in exp["purpose"]
        assert "PROVISIONAL" in exp["generation_budget"]["definition"] and f"program {tf.program_id(key)}" in exp["generation_budget"]["definition"]
    fb = _load(derived_root, "experiment", tf.CELLS["fb"]["cell"])
    assert fb["core_read_design"]["arms"] == {arm: tf.CELLS["fb"]["run"] for arm in tf.FOLD_B_ARM_ORDER}
    assert fb["split_id"] == "SLURP-EVO-FOLD-B-V2" and _load(derived_root, "experiment", tf.CELLS["ch1"]["cell"])["split_id"] == "STAGE3-CONSTRUCTION-R01C"


def test_no_stale_arm_table_fact_or_geometry_survives(derived_root):
    for key in tf.ALL_KEYS:
        for kind, name in (("baseline", tf.CELLS[key]["baseline"]), ("experiment", tf.CELLS[key]["cell"])):
            doc = _load(derived_root, kind, name)
            assert tdc.pdc.scan_geometry(doc) == []
            assert [p for p, v in _leaves(doc) if isinstance(v, str) and tdc.T_STALE.search(v)] == [], (kind, name)


# ---- the writer and the package -------------------------------------------------------------------------------------------------------------------


@needs_data
def test_writer_files_and_bytes_are_pinned(written_root):
    root, result = written_root
    emitted = {}
    for rel in GOLDEN:
        path = root / rel
        assert path.is_file() and not path.is_symlink(), rel
        emitted[rel] = _sha(path)
    assert emitted == GOLDEN
    package = {p.name for p in (root / tf.PACKAGE).iterdir()}
    assert package == {Path(r).name for r in GOLDEN if r.startswith(tf.PACKAGE + "/")}
    assert result["package"] == tf.PACKAGE and result["ledger_appended"] is None and "__prereg__" in result["shas"]


@needs_data
def test_writer_is_deterministic_and_never_touches_the_study_tree(written_root, tmp_path):
    root, result = written_root
    tracked = [REPO / rel for rel in GOLDEN if (REPO / rel).is_file()]
    before = {p: _sha(p) for p in tracked}
    again = tw.write_package(root, COMMIT, None, DATA_ROOT)                             # no stamp: the one already in the facts
    assert again["shas"] == result["shas"]
    assert {p: _sha(p) for p in tracked} == before
    assert json.loads((root / tf.PACKAGE / "registered-facts-v1.json").read_text(encoding="utf-8"))["registered_at_utc"] == AT


@needs_data
def test_facts_state_identity_caps_pins_and_the_budget_outcome(written_root):
    root, _ = written_root
    facts = json.loads((root / tf.PACKAGE / "registered-facts-v1.json").read_text(encoding="utf-8"))
    assert facts["status"] == "DRAFT-HELD" and facts["study_commit_at_registration"] == COMMIT and facts["registered_at_utc"] == AT
    assert {k: (v["composed_config_hash"], v["protocol_hash"]) for k, v in facts["identity"].items()} == {k: tuple(v) for k, v in HASHES.items()}
    assert {k: (v["generation_calls"], v["fit_endpoint_calls"], v["speech_audio_seconds"]) for k, v in facts["accounting"]["caps"].items()} == CAPS
    assert facts["accounting"]["family_total_iteration_1"] == {"generation_calls": 30486, "fit_endpoint_calls": 15243, "speech_audio_seconds": 101716}
    assert facts["budget_switch"]["per_switch"]["budget6144-max7168"]["largest_fitting_bound"] == 40 and facts["budget_switch"]["per_switch"]["default"]["fits"] is True
    assert facts["accounting"]["construction_group"]["chosen"] == "solver-decoding" and facts["accounting"]["construction_group"]["alternative_program"] == "phase-native-module-v1-c10"
    r01c = facts["r01c"]
    assert r01c["rows"] == 625 and r01c["halves"]["R01C-H1"]["count"] == 313 and r01c["halves"]["R01C-H2"]["count"] == 312 and r01c["audio_pins"]["matching"] == 625
    assert r01c["halves"]["R01C-H1"]["split_identity_hash"] == tf.CELLS["ch1"]["split_identity_hash"] and r01c["halves_split_receipt"]["split_mode"] == "interleave"
    assert facts["pins"]["fold_b_audio_pins"]["matching"] == 1080 and "supply" not in facts["pins"]["fold_b_preparations_reused"]
    pending = facts["pending_pins"]
    assert set(pending["banks"]) == {"exemplar_bank", "rule_bank"} and pending["decision_receipt"] == tf.DECISION_RELPATH and set(pending["supplies"]) == set(tf.ALL_KEYS)
    assert all(v["sha256"] is None for v in pending["r01c_port_artifacts"].values())
    assert facts["acceptance_and_kill_rules"]["kill_and_stop"][0].startswith("iteration 1 construction gain < +1 pp")
    assert {o["id"] for o in facts["owner_items"]} >= {"O-switch-bound", "O-construction-group", "O-commit-policy"} and facts["findings"] and facts["open_questions"]


@needs_data
def test_measurement_blocks_never_claim_a_benefit_and_pass_the_floor_gate_shape(written_root):
    root, _ = written_root
    blocks = json.loads((root / tf.PACKAGE / "registered-facts-v1.json").read_text(encoding="utf-8"))["stage3_measurement"]
    runs = {tf.CELLS[k]["run"]: k for k in tf.ALL_KEYS}
    assert set(blocks) == set(runs)
    designs = {runs[run]: block["measurement_design"] for run, block in blocks.items()}
    assert designs == {"csmoke": "form-proof", "fsmoke": "form-proof", "ch1": "material-source", "ch2": "material-source", "fb": "same-flight-node-column"}
    assert all(block["claims_benefit"] is False for block in blocks.values())
    fb = blocks[tf.CELLS["fb"]["run"]]
    assert fb["template_variants"] == {"node": "perception", "candidate_arms": 1, "twin": True} and fb["primary_metric_key"] == "slu" and fb["rows"] == 1080
    import importlib  # noqa: PLC0415

    checks = str(REPO / "scripts" / "checks")
    if checks not in sys.path:
        sys.path.insert(0, checks)
    checker = importlib.import_module("stage3_effect_floor_check")
    for run, block in blocks.items():
        parsed = checker._parse_declaration(block, run, None)
        assert parsed.problems == [], (run, parsed.problems)
        assert parsed.design == block["measurement_design"] and parsed.claims_benefit is False


@needs_data
def test_ledger_rows_are_unconsumed_drafts_with_provisional_cells(written_root):
    root, _ = written_root
    text = (root / tf.PACKAGE / "draft-ledger-rows.md").read_text(encoding="utf-8")
    rows = [[c.strip() for c in line.strip().strip("|").split(" | ")] for line in text.splitlines() if line.startswith("| 2026-10-30")]
    assert len(rows) == 20 and all(len(r) == 15 for r in rows) and all(r[8] == "no" for r in rows)
    by_run = {r[1]: r for r in rows}
    assert tw.tdc.R01C_ENCODE_RUN in by_run and by_run[tw.tdc.R01C_ENCODE_RUN][9:12] == ["625", "2727", "2.0"]
    assert by_run[tw.REFLECTOR_RUN][9] == "36" and by_run[tw.REFLECTOR_FIT_RUN][9] == "37"
    for key in tf.ALL_KEYS:
        gen, fit = by_run[tf.CELLS[key]["run"]], by_run[tf.CELLS[key]["fit_run"]]
        assert (gen[9], fit[9], gen[10]) == (str(CAPS[key][0]), str(CAPS[key][1]), str(CAPS[key][2])) and gen[11].startswith("PROVISIONAL") and fit[11].startswith("PROVISIONAL")
        assert gen[7] == tf.CELLS[key]["split_identity_hash"] and gen[3].startswith("DRAFT -- REGISTERED") and gen[14].startswith("DRAFT at study commit")
    halves = by_run[tw.HALVES_RUN]
    assert "1a88c40bfec99edd9f6f38ce93a346dd0fb046cba493d6f2558f2f1620310b14" in halves[14] and "c7b16e132f43cf32bf7aa492b7724e535669fe33b2254a76e214386c5299cd72" in halves[14]
    assert halves[14].startswith("RUN BEFORE THIS ROW WAS WRITTEN") and by_run[tw.CREDIT_RUN][9] == "0"
    assert tf.PREP_LEDGER_MARK in text and tf.LEDGER_MARK in text


@needs_data
def test_ledger_append_is_idempotent_and_preserves_the_ledger(tmp_path):
    root = make_t_root(tmp_path / "root")
    ledger = root / "docs" / "exposure-ledger.md"
    original = ledger.read_text(encoding="utf-8")
    first = tw.write_package(root, COMMIT, AT, DATA_ROOT, append_ledger=True)
    assert first["ledger_appended"] is True
    appended = ledger.read_text(encoding="utf-8")
    assert appended.startswith(original.rstrip("\n")) and tf.LEDGER_MARK in appended and tf.PREP_LEDGER_MARK in appended
    assert tw.write_package(root, COMMIT, AT, DATA_ROOT, append_ledger=True)["ledger_appended"] is False
    assert ledger.read_text(encoding="utf-8") == appended


@needs_data
def test_admission_draft_and_countersign_texts_are_held(written_root):
    root, _ = written_root
    draft = json.loads((root / tf.PACKAGE / "root-admission-draft-v1.json").read_text(encoding="utf-8"))
    assert draft["status"] == "DRAFT-NOT-MINTED" and draft["decision_id"] == tf.DECISION_ID and draft["recorded_at_utc"] is None
    assert draft["eligible_run_ids"] == tf.eligible_run_ids() + [tw.REFLECTOR_RUN, tw.REFLECTOR_FIT_RUN]
    assert len(draft["eligible_attempt_ids"]) == 12 and draft["population"]["fold_a_contact_authorized"] is False and draft["population"]["test_contact_authorized"] is False
    signs = json.loads((root / tf.PACKAGE / "countersign-row-texts-v1.json").read_text(encoding="utf-8"))
    assert set(signs["rows"]) == set(draft["eligible_run_ids"]) and signs["decision_receipt"] == tf.DECISION_RELPATH
    for name in ("signoff-conditions-construction-v1.json", "signoff-conditions-fold-b-v1.json"):
        doc = json.loads((root / tf.PACKAGE / name).read_text(encoding="utf-8"))
        assert doc["package"] == tf.PKG_NAME and doc["table_relpath"].startswith(f"analysis/{tf.READ_RUN_ID}/")
    assert json.loads((root / tf.PACKAGE / "signoff-conditions-construction-v1.json").read_text(encoding="utf-8"))["expected_arms"] == sorted(tf.CONSTRUCTION_ARM_ORDER)
    assert json.loads((root / tf.PACKAGE / "signoff-conditions-fold-b-v1.json").read_text(encoding="utf-8"))["expected_arms"] == sorted(tf.FOLD_B_ARM_ORDER)


@needs_data
def test_plans_register_caps_arms_and_the_fit_context(written_root):
    root, _ = written_root
    for key in tf.ALL_KEYS:
        gen = json.loads((root / tf.PACKAGE / tf.plan_file_name(key, "generation")).read_text(encoding="utf-8"))
        fit = json.loads((root / tf.PACKAGE / tf.plan_file_name(key, "fit")).read_text(encoding="utf-8"))
        assert gen["execution_plan"]["planned_model_calls"] == CAPS[key][0] and fit["execution_plan"]["planned_model_calls"] == CAPS[key][1]
        assert gen["execution_plan"]["run_id"] == tf.CELLS[key]["run"] and fit["execution_plan"]["run_id"] == tf.CELLS[key]["fit_run"]
        assert gen["arm_order"] == list(tf.CONSTRUCTION_ARM_ORDER if tf.kind_of(key) == "construction" else tf.FOLD_B_ARM_ORDER)
        assert gen["status"] == lw.STATUS and gen["composed_config_hash"] == HASHES[key][0] and fit["context_fit"]["ok"] is True
        assert gen["prepared_phase_supply"]["sha256"] == "MINTED_AT_THE_FLIP" and gen["accelerator_bound"]["status"] == "PROVISIONAL"
        if tf.CELLS[key]["trace_read_id"]:
            assert gen["trace_read_id"] == tf.CELLS[key]["trace_read_id"]


@needs_data
def test_package_is_english_only(written_root):
    root, _ = written_root
    files = [p for p in (root / tf.PACKAGE).iterdir()] + [root / tf.PREREG]
    files += [REPO / "scripts" / "flights" / "nx" / name for name in ("tfrl_v2_port_family.py", "tfrl_v2_port_derive_cells.py", "tfrl_v2_port_write_registration.py")]
    files += [Path(__file__)]
    for path in files:
        assert not CJK.search(path.read_text(encoding="utf-8")), path
    assert re.search(r"\bnp4\b|-np 4|--max-num-seqs", (root / tf.PACKAGE / "README.md").read_text(encoding="utf-8")) is None


@needs_data
def test_dispatcher_reaches_the_t_writer_with_the_family_switch(tmp_path, capsys):
    root = make_t_root(tmp_path / "root")
    assert lw.family_module("tfrl-v2") is tw and "tfrl-v2" in lw.FAMILIES
    code = lw.main(["--family", "tfrl-v2", "--study-commit", COMMIT, "--data-root", str(DATA_ROOT), "--registered-at", AT, "--root", str(root), "--no-derive"])
    assert code == 0 or code is None
    out = json.loads(capsys.readouterr().out)
    assert out["package"] == tf.PACKAGE and out["ledger_appended"] is None
    pf.select_model_fragment("default")
    with pytest.raises(SystemExit, match="largest fitting T bound is 40"):
        lw.main(["--family", "tfrl-v2", "--model-fragment", "budget6144-max7168", "--study-commit", COMMIT, "--data-root", str(DATA_ROOT), "--root", str(root), "--no-derive"])

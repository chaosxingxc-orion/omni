"""The L338 port writer's family switch (``--family``), its thinking-budget switch (``--model-fragment``) and the c17 ledger successor block.

* ``--family l338-port`` (the default) must write the SAME bytes it wrote before the switches existed: the eleven writer-emitted files of the committed port package, the
  preregistration and the two port cells' configs are pinned by sha256 against a run into a throwaway root (``GOLDEN``), and the committed package equals that run except for the two
  files that quote the derive's ``versus_committed.moved`` lists (a record of the config state the derive found when the package was last minted).
* ``--model-fragment budget6144-max7168`` swaps the fragment line, lowers the supply bound 55 -> 51, sets the Solver prompt limit 14,334 and the completion ceiling 7,168, and keeps the
  latency rule (the fragment's ``timeout_seconds``) at 735 s; the composed-config and protocol hashes it produces for the two port cells are pinned (``SWITCH_HASHES``). It is a writer
  option for a throwaway root: nothing here writes a committed package.
* ``--render-ledger-successor`` renders the successor block of the c17 re-mint without touching the exposure ledger.

Every root here is a throwaway: read-only trees are symlinks, every path a derive or writer may write is a copy.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
for _extra in (REPO / "scripts" / "flights" / "nx", REPO / "scripts" / "flights" / "direct", REPO / "scripts" / "flights" / "tfrl", REPO / "src"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import l338_port_family as pf  # noqa: E402
import l338_port_write_registration as w  # noqa: E402

CJK = re.compile("[\u4e00-\u9fff]")
PORT_PKG_NAME = "2026-10-28-slurp-l338-lever-column-fold-b-npu-port-v1"
FOLD_A_PKG_NAME = "2026-10-30-slurp-l338-fold-a-confirmation-npu-port-v1"
#: The commit the committed port package was minted at; the writer stamps it into the files.
MINT_COMMIT = "890643114fc1f2d851737e1202f7e32e4bba2ea0"
DATA_ROOT = Path(os.environ.get("SAEA_DATA_ROOT") or os.environ.get("SPEECHRL_DATA_DIR") or "/nonexistent-data-root")
needs_data = pytest.mark.skipif(not (DATA_ROOT / "prepared").is_dir(), reason="the data root with the port preparation receipts is not mounted")

#: sha256 of the writer's output for ``--family l338-port`` (default fragment) at MINT_COMMIT, produced before the family and budget switches existed and reproduced after.
GOLDEN = {
    "README.md": "f1c016f071e8f3dd7e4d33e9feeeb2ae9f0d2f1673fce2f9d52dcb92e1538ac2",
    "countersign-row-texts-v1.json": "47f0b84c6e8110ecc636ce1e5ade040f68120a68662f02692283a0dfdd81e67b",
    "draft-ledger-rows.md": "47a841e702524af94a35fb414e480bf5c771dd73f3d83b639f11a017cb41a2f1",
    "fit-plan-registered-v1.json": "532a23148e92aa7cdd9c8d919c7a08cd2497dafb1dd2c77b67602be0eeed7f93",
    "generation-plan-registered-v1.json": "dc33b4c2907dcd364531ef45d0eb2cb6b8465b6ce669b8de276c6b6a0d8b9251",
    "protocol-registered-v1.json": "b5f22dedbcbeda8af808d468908741a8d057190d67694a1bde29db3622c5dfa6",
    "registered-facts-v1.json": "27523efa04e05995a47d26f4b60b3dcc45f800d8aff26ec0e4ef05c0d9ad940c",
    "root-admission-draft-v1.json": "42561d25d5bf654b96ea58e0487cd715a38dbdb5effcfb88407d19eeea5977da",
    "signoff-conditions-v1.json": "dead4b432b86c8693b98e92fbb0ceff03ce41d16657440ed0bdff812e9277109",
    "smoke-fit-plan-registered-v1.json": "ab5ce60fd02c7ad902a0b196208cd22ad9fb58bc70da6fe9e8be4e5bc163989b",
    "smoke-generation-plan-registered-v1.json": "882aa23789701f07cc6ecf9fc9eaf002452651ecd1e107e93c925e84fb086382",
}
GOLDEN_PREREG = "35202fec4284cc7e5da9ef2836436b756b35c3139fe7f006580a0404d1364ed0"
#: composed-config hash / protocol hash of the two port cells (``smoke`` and ``arm``) by fragment choice.
DEFAULT_HASHES = {"arm": ("15fc16341ac7b5c2053d718b7f2e5507eb63e0b733c09070203fb7592186b748", "22430cf316a58db45de1a64f935c57a5538e2f099e0669b9593e0b2d966363a8"),
                  "smoke": ("a45c48ef8972e67adc891bfcee9b2230bc8f4805445c1d30743aed071dc79c32", "19ad703be5892a730cc7244a29b1b93f808201d07cd25fea9f4ad90fe8f888b3")}
SWITCH_HASHES = {"arm": ("ba3f28f919db05d9c8d7ec524c71982bcf8c499eb2ca4d771c9267c58fa7cb0b", "fdadfbf50737a03ea30b3e68b06142991d3fa28a39f6775b51e0c7759ac5ef92"),
                 "smoke": ("30a8c03f44f8930603344d8e419f969226a20c00eb275262c6e487e3e7b95d85", "d85ab5269b014cdeac4bc6da4eaad0ea3e34a8058a0aedba774292068ab6fef6")}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_root(dst: Path, packages=(PORT_PKG_NAME,), copy_config_files=()) -> Path:
    """A throwaway study root: every read-only path is a symlink; the configs a derive rewrites, the package directories, the preregistrations and the ledger are copies."""

    dst.mkdir(parents=True)
    (dst / "configs").mkdir()
    for child in (REPO / "configs").iterdir():
        if child.name in ("baseline", "experiment", "campaign"):
            (dst / "configs" / child.name).mkdir()
            for item in child.iterdir():
                target = dst / "configs" / child.name / item.name
                if item.name in copy_config_files or ("l338" in item.name and "port" in item.name):
                    shutil.copy2(item, target)
                else:
                    os.symlink(item, target)
        else:
            os.symlink(child, dst / "configs" / child.name)
    (dst / "docs" / "readiness").mkdir(parents=True)
    for child in (REPO / "docs").iterdir():
        if child.name == "readiness":
            continue
        if child.name == "exposure-ledger.md":
            shutil.copy2(child, dst / "docs" / child.name)
        else:
            os.symlink(child, dst / "docs" / child.name)
    prefixes = tuple(f"{name}-" for name in (PORT_PKG_NAME, FOLD_A_PKG_NAME))
    for child in (REPO / "docs" / "readiness").iterdir():
        if child.name in packages:
            shutil.copytree(child, dst / "docs" / "readiness" / child.name)
        elif child.name.startswith(prefixes):
            shutil.copy2(child, dst / "docs" / "readiness" / child.name)
        else:
            os.symlink(child, dst / "docs" / "readiness" / child.name)
    (dst / "scripts" / "flights" / "nx").mkdir(parents=True)
    for name in ("l338_port_pins.json", "fold_a_port_pins.json"):
        shutil.copy2(REPO / "scripts" / "flights" / "nx" / name, dst / "scripts" / "flights" / "nx" / name)
    return dst


@pytest.fixture(autouse=True)
def _restore_default_fragment():
    yield
    pf.select_model_fragment("default")


def _run(root: Path, *, choice: str = "default", **kwargs) -> dict:
    pf.select_model_fragment(choice)
    return w.write_package(root, MINT_COMMIT, None, DATA_ROOT, **kwargs)


def _hashes(root: Path) -> dict:
    out = {}
    for key in ("smoke", "arm"):
        facts = json.loads((root / "docs" / "readiness" / PORT_PKG_NAME / "registered-facts-v1.json").read_text(encoding="utf-8"))
        out[key] = (facts["identity"][key]["composed_config_hash"], facts["identity"][key]["protocol_hash"])
    return out


# ---- the unchanged port package (snapshot) ---------------------------------------------------------------------------------------------------


@needs_data
def test_default_writer_output_is_byte_identical_to_the_pre_switch_snapshot(tmp_path):
    root = make_root(tmp_path / "root")
    result = _run(root)
    pkg = root / "docs" / "readiness" / PORT_PKG_NAME
    assert {name: _sha(pkg / name) for name in GOLDEN} == GOLDEN
    assert _sha(root / "docs" / "readiness" / f"{PORT_PKG_NAME}-preregistration.md") == GOLDEN_PREREG
    assert {k: v for k, v in result["shas"].items() if k in GOLDEN} == GOLDEN
    assert _hashes(root) == DEFAULT_HASHES
    # the two port cells' configs are rewritten identically (the derive is idempotent on the committed tree)
    for kind, names in (("baseline", ("slurp-native-stage3-l338-lever-column-fold-b-baseline-raw-port-v1", "slurp-native-stage3-l338-lever-column-fold-b-smoke-4-baseline-raw-port-v1")),
                        ("experiment", ("slurp-native-stage3-l338-lever-column-fold-b-npu-port-v1", "slurp-native-stage3-l338-lever-column-fold-b-smoke-4-npu-port-v1"))):
        for name in names:
            assert (root / "configs" / kind / f"{name}.json").read_bytes() == (REPO / "configs" / kind / f"{name}.json").read_bytes(), name


@needs_data
def test_committed_port_package_is_the_writer_output_except_the_recorded_moved_lists(tmp_path):
    """The committed package was minted from a tree whose configs were one step behind, so ``derive.*.versus_committed.*.moved`` (and the admission draft's sha of the facts file)
    differ; every other byte is the writer's."""

    root = make_root(tmp_path / "root")
    _run(root)
    new, old = root / "docs" / "readiness" / PORT_PKG_NAME, REPO / "docs" / "readiness" / PORT_PKG_NAME
    for name in sorted(GOLDEN):
        if name in ("registered-facts-v1.json", "root-admission-draft-v1.json"):
            continue
        assert (new / name).read_bytes() == (old / name).read_bytes(), name
    a, b = (json.loads((d / "registered-facts-v1.json").read_text(encoding="utf-8")) for d in (new, old))
    for facts in (a, b):
        for cell in facts["derive"].values():
            for kind in cell["versus_committed"].values():
                kind["moved"] = []
    assert a == b
    na, nb = (json.loads((d / "root-admission-draft-v1.json").read_text(encoding="utf-8")) for d in (new, old))
    for doc in (na, nb):
        doc["antecedents"]["registered_facts"]["sha256"] = "-"
    assert na == nb


@needs_data
def test_family_names_and_default_fragment(tmp_path):
    assert w.FAMILIES == ("l338-port", "tfrl-v2", "fold-a-confirmation")
    assert pf.active_switch() == "default" and pf.MODEL_FRAGMENT == pf.DEFAULT_MODEL_FRAGMENT
    assert pf.budget_view() == {"switch": "default", "model_fragment": pf.DEFAULT_MODEL_FRAGMENT, "thinking_budget_tokens": 3072, "completion_cap": 4096,
                                "prompt_limit_tokens": 17406, "max_supplied_items": 55, "timeout_seconds": 420}
    with pytest.raises(ValueError, match="--model-fragment"):
        pf.select_model_fragment("budget9999")
    assert w.family_module("l338-port") is sys.modules[w.__name__]


# ---- the thinking-budget switch --------------------------------------------------------------------------------------------------------------


def test_budget6144_view_is_derived_not_typed():
    view = pf.select_model_fragment("budget6144-max7168")
    assert view == {"switch": "budget6144-max7168", "model_fragment": "bf16-thinking-vllm-slurp-module-fold-b-seq4-len24k-budget6144-max7168-port-v1",
                    "thinking_budget_tokens": 6144, "completion_cap": 7168, "prompt_limit_tokens": 14334, "max_supplied_items": 51, "timeout_seconds": 735}
    assert pf.MODEL_FRAGMENT == view["model_fragment"] and all(spec["model"] == view["model_fragment"] for spec in pf.CELLS.values())
    assert "budget 6144, cap 7168" in pf.switch_text("lock's per-node thinking fields (budget 3072, cap 4096; the")
    assert "thinking budget 6144 and max_tokens 7168" in pf.switch_text("lock's thinking budget 3072 and max_tokens 4096, its sample")
    assert "ceiling" not in pf.model_text() and "max_tokens 7168 (reasoning budget 6144" in pf.model_text()
    pf.select_model_fragment("default")
    assert pf.switch_text("budget 3072, cap 4096") == "budget 3072, cap 4096" and "max_tokens 4096 (reasoning budget 3072" in pf.model_text()


@needs_data
def test_budget6144_switch_hash_table_and_effects(tmp_path):
    root = make_root(tmp_path / "root")
    _run(root, choice="budget6144-max7168")
    assert _hashes(root) == SWITCH_HASHES
    pkg = root / "docs" / "readiness" / PORT_PKG_NAME
    facts = json.loads((pkg / "registered-facts-v1.json").read_text(encoding="utf-8"))
    record = facts["model_fragment_switch"]
    assert (record["max_supplied_items"], record["solver_prompt_limit_tokens"], record["latency_rule_timeout_seconds"], record["latency_rule_scaled_from"]) == (51, 14334, 735, 420)
    assert record["model_fragment"].endswith("budget6144-max7168-port-v1") and "WRITER OPTION ONLY" in record["status"]
    assert facts["context_fit"]["max_supplied_items"] == 51 and facts["context_fit"]["prompt_limit_tokens"] == 14334 and facts["context_fit"]["ok"] is True
    assert facts["context_fit"]["estimate_at_bound_tokens"] == 14308 and facts["context_fit"]["completion_cap"] == 7168
    assert facts["thinking_knobs"]["thinking_budget_tokens"] == 6144 and facts["thinking_knobs"]["completion_cap"] == 7168
    for key in ("smoke", "arm"):
        assert facts["identity"][key]["model_fragment"] == pf.BUDGET6144_MODEL_FRAGMENT
    # the caps do not depend on the budget; the fragment line and the bound are what moved
    assert facts["accounting"]["caps"]["arm"] == {"fit_endpoint_calls": 18360, "generation_calls": 36720, "speech_audio_seconds": 107264}
    arm_baseline = json.loads((root / "configs/baseline/slurp-native-stage3-l338-lever-column-fold-b-baseline-raw-port-v1.json").read_text(encoding="utf-8"))
    frame = arm_baseline["native_episode"]["consumption_module"]["sampling_frame"]
    assert frame["max_supplied_items"] == 51 and frame["fit_estimate"]["completion_ceiling"] == 7168
    sampling = arm_baseline["native_episode"]["roles"]["perception"]["sampling_variants"]
    assert sampling["max_supplied_items"] == 51 and sampling["fit_estimate"]["completion_ceiling"] == 7168
    assert "budget 6144, cap 7168" in arm_baseline["what_this_is"] and "budget 3072" not in arm_baseline["what_this_is"]
    # the committed tree was not touched
    committed = json.loads((REPO / "configs/baseline/slurp-native-stage3-l338-lever-column-fold-b-baseline-raw-port-v1.json").read_text(encoding="utf-8"))
    assert committed["native_episode"]["consumption_module"]["sampling_frame"]["max_supplied_items"] == 55
    assert (REPO / "docs" / "readiness" / PORT_PKG_NAME / "registered-facts-v1.json").read_bytes() != (pkg / "registered-facts-v1.json").read_bytes()


def test_switch_fragment_exists_and_carries_the_latency_rule():
    fragment = json.loads((REPO / "configs" / "model" / f"{pf.BUDGET6144_MODEL_FRAGMENT}.json").read_text(encoding="utf-8"))
    assert fragment["timeout_seconds"] == 735 == 420 * 7168 // 4096
    assert fragment["reasoning_budget_tokens"] == 6144 and fragment["decoding_params"]["max_tokens"] == 7168


# ---- the c17 ledger successor ------------------------------------------------------------------------------------------------------------------


@needs_data
def test_ledger_successor_block_is_rendered_and_the_ledger_is_untouched(tmp_path):
    root = make_root(tmp_path / "root")
    ledger = root / "docs" / "exposure-ledger.md"
    before = _sha(ledger)
    rel = f"docs/readiness/{PORT_PKG_NAME}/ledger-successor-draft-c17.md"
    result = _run(root, successor_path=rel)
    assert _sha(ledger) == before == _sha(REPO / "docs" / "exposure-ledger.md")
    text = (root / rel).read_text(encoding="utf-8")
    assert result["ledger_successor"]["sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert text.startswith(w.LEDGER_SUCCESSOR_MARK) and "SUCCESSOR NOTE (c17" in text
    # the stale c15 figures of the committed rows -> the package's c17 figures
    for needle in ("calls 120 -> 136", "calls 240 -> 68", "calls 32400 -> 36,720", "calls 64800 -> 18,360", "speech-seconds 94644 -> 107,264", "speech-seconds 306 -> 347"):
        assert needle in text, needle
    rows = [line for line in text.splitlines() if line.startswith("| 20")]
    assert len(rows) == 4 and all(" | no | " in row for row in rows)
    for key, (config, protocol) in DEFAULT_HASHES.items():
        assert config in text and protocol in text
    assert not CJK.search(text)
    # idempotent: a second render is the same bytes
    again = _run(make_root(tmp_path / "root2"), successor_path=rel)
    assert again["ledger_successor"]["sha256"] == result["ledger_successor"]["sha256"]


# ---- the dispatcher ------------------------------------------------------------------------------------------------------------------------------


@needs_data
def test_cli_dispatch_matches_write_package_and_refuses_misuse(tmp_path, capsys):
    root = make_root(tmp_path / "root")
    assert w.main(["--study-commit", MINT_COMMIT, "--data-root", str(DATA_ROOT), "--root", str(root)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert {k: v for k, v in printed["shas"].items() if k in GOLDEN} == GOLDEN
    with pytest.raises(SystemExit):
        w.main(["--family", "no-such-family", "--study-commit", MINT_COMMIT, "--data-root", str(DATA_ROOT), "--root", str(root)])
    with pytest.raises(SystemExit):
        w.main(["--family", "fold-a-confirmation", "--render-ledger-successor", "x.md", "--study-commit", MINT_COMMIT, "--data-root", str(DATA_ROOT), "--root", str(root)])


def test_no_cjk_in_the_switch_sources():
    for rel in ("scripts/flights/nx/l338_port_family.py", "scripts/flights/nx/l338_port_derive_cells.py", "scripts/flights/nx/l338_port_write_registration.py",
                "tests/unit/test_l338_port_family_switch.py"):
        assert not CJK.search((REPO / rel).read_text(encoding="utf-8")), rel

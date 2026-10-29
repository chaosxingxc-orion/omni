"""Guard G: the DRAFT port registration package must quote the live composed config hashes of both port cells.

Commit db798a83 moved ``--max-num-seqs`` out of the hashed port model fragment (fragment hash df95b65b -> a35b2740), so the composed and protocol hashes
of the two port cells moved and ``protocol-registered-v1.json`` was stale until the registration writer was re-run (stream S13). Every contact path recomputes the
protocol hash and refuses a mismatch; this module keeps any future drift visible: the package must quote the live composed hash of both cells, and the mint
entry point refuses a stale package (fail closed), covered below.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
for _extra in (REPO / "scripts" / "flights" / "nx", REPO / "scripts" / "flights" / "direct", REPO / "src"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import l338_port_family as pf  # noqa: E402
import l338_port_mint_receipt as mint  # noqa: E402

from speech_aware_evidence_acquisition.core.config import compose  # noqa: E402

REGISTERED = REPO / pf.PACKAGE / "protocol-registered-v1.json"
#: The composed hashes the package quoted before db798a83 moved the fragment (flight, smoke): what a stale package looks like.
STALE_HASHES = {
    "arm": "21bf1af0b91a47d2a9870aaffa0d619f1555bc82834e87b3c51196f019477d40",
    "smoke": "14d06e305218ce8060c3a17897cd3841d45fea7df50d371876697294fb5df714",
}


def _live_hash(root: Path, key: str) -> str:
    spec = pf.CELLS[key]
    cell = json.loads((root / "configs" / "experiment" / f"{spec['cell']}.json").read_text(encoding="utf-8"))["cells"][0]
    return compose(root, cell["model_fragment"], cell["dataset_fragment"], cell["baseline_fragment"], spec["cell"]).config_hash


@pytest.mark.parametrize("key", list(pf.ALL_KEYS))
def test_registered_package_quotes_the_live_composed_hash(key):
    registered = json.loads(REGISTERED.read_text(encoding="utf-8"))["cells"][key]["composed_config_hash"]
    assert _live_hash(REPO, key) == registered


def _fixture_root(tmp_path: Path, *, matching: bool) -> Path:
    """A minimal study root: the real configs (symlinked) plus a copy of the package whose quoted hashes are live (matching) or the pre-db798a83 ones (stale)."""

    for name in ("configs", "src"):
        (tmp_path / name).symlink_to(REPO / name, target_is_directory=True)
    package = tmp_path / pf.PACKAGE
    package.mkdir(parents=True)
    doc = json.loads(REGISTERED.read_text(encoding="utf-8"))
    for key in pf.ALL_KEYS:
        doc["cells"][key]["composed_config_hash"] = _live_hash(REPO, key) if matching else STALE_HASHES[key]
    (package / REGISTERED.name).write_text(json.dumps(doc), encoding="utf-8")
    return tmp_path


def test_matching_package_passes_the_guard(tmp_path):
    assert mint.stale_cells(_fixture_root(tmp_path, matching=True)) == []


def test_mint_refuses_a_stale_package_before_any_mint_call(tmp_path, monkeypatch):
    root = _fixture_root(tmp_path, matching=False)
    package = root / pf.PACKAGE
    shutil.copy(REPO / pf.PACKAGE / "root-admission-draft-v1.json", package / "root-admission-draft-v1.json")
    monkeypatch.setenv("SAEA_STUDY_ROOT", str(root))
    called = []
    monkeypatch.setattr(mint.base, "mint", lambda *a, **k: called.append(1))
    with pytest.raises(SystemExit, match="STALE"):
        mint.main(["--dry"])
    assert not called

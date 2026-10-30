"""Shared identities of the Fold A CONFIRMATION twin of the L338 port flight (plan 2026-10-30 section 8 R2 / M2: a lever is accepted only if its Fold A lower CI95 > 0).

The cell is the NPU-port L338 flight (``l338_port_family``: the c17 arm table l1-support / l2-union / aa / lock-rep2 / lock, the registered caps, the raw-audio FIT rule)
moved onto TRAIN Fold A (``SLURP-EVO-FOLD-A-V2``, 1,080 rows, discovery role) by ONE rule: every Fold B token of an identity becomes its Fold A token
(``FOLD-B`` -> ``FOLD-A``, ``fold-b`` -> ``fold-a``, ``Fold B`` -> ``Fold A``). That reproduces, name for name, the Fold A preparation ids of the draft
(``docs/readiness/2026-10-30-three-way-fold-b-npu-port-v1/fold-a-confirmation-fragment-draft-v1.md`` section 3: FA1 ``SAEA-SLURP-TFRL-FOLD-A-SPEECH-QUERY-ENCODING-NPU-PORT-V1``, FA2
``...-TRAIN-FOLD-A-ALIAS-...``, FA3 ``SAEA-SLURP-FOLD-A-INPUTS-PREPARE-NPU-PORT-V1``, FA4 ``...-AGREEMENT-FOLD-A-...``). One cell, no smoke (draft section 4: the Fold A runs reuse the
programs and harness bytes that were smoked on Fold B). The Fold A artefacts FA1-FA4 and the cell's first-phase supply do not exist yet: every sha256 that pins one of them is the
zero sentinel (PENDING) until ``fold_a_port_pins.json`` carries the real value. Model-free; nothing here authorises a contact.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import l338_family as fam  # noqa: E402
import l338_port_family as pf  # noqa: E402

REPO_ROOT = pf.REPO_ROOT
ZERO = "0" * 64
PENDING = "PENDING"
TOKENS = (("FOLD-B", "FOLD-A"), ("fold-b", "fold-a"), ("Fold B", "Fold A"))
PINS_PATH = HERE / "fold_a_port_pins.json"


#: Names that contain a Fold B token without being Fold B identities: the serving model fragments (their names say "fold-b" for the module geometry) and the deployment-neutral
#: service session (the Fold A flight is served by the SAME resident deployment). The swap leaves them untouched.
PROTECTED = (pf.DEFAULT_MODEL_FRAGMENT, pf.BUDGET6144_MODEL_FRAGMENT, pf.SERVICE_SESSION,
             "serving-slurp-thinking-module-fold-b-np4-budget3072-max4096-v1")


def _mask(text: str) -> str:
    for index, name in enumerate(PROTECTED):
        text = text.replace(name, f"\x00{index}\x00")
    return text


def _unmask(text: str) -> str:
    for index, name in enumerate(PROTECTED):
        text = text.replace(f"\x00{index}\x00", name)
    return text


def swap(text: str) -> str:
    """Every Fold B token of ``text`` as its Fold A token (protected names untouched)."""

    text = _mask(text)
    for old, new in TOKENS:
        text = text.replace(old, new)
    return _unmask(text)


def has_fold_b(text: str) -> bool:
    text = _mask(text)
    return any(old in text for old, _ in TOKENS)


DATE = "2026-10-30"
PKG_NAME = "slurp-l338-fold-a-confirmation-npu-port-v1"                 # the package directory suffix
PACKAGE = f"docs/readiness/{DATE}-{PKG_NAME}"
PREREG = f"{PACKAGE}-preregistration.md"
COORD_NAME = swap(pf.PKG_NAME)                                          # slurp-l338-lever-column-fold-a-npu-port-v1 (the id family of the run ids)
DECISION_ID = swap(pf.DECISION_ID)
DECISION_RELPATH = swap(pf.DECISION_RELPATH)
DESIGN_DOC = pf.DESIGN_DOC
OWNER_DECISIONS = pf.OWNER_DECISIONS
HISTORICAL_PACKAGE = pf.PACKAGE                                         # the Fold B port package this is the confirmation twin of
HISTORICAL_DECISION_ID = pf.DECISION_ID
TOOL_MINT = pf.TOOL_MINT
LEDGER_MARK = ("## Fold A confirmation of the L338 NPU port: the lever column on the BF16 / vLLM-Ascend Thinking core, TRAIN Fold A (one pass of the c17 arm table; "
               "no smoke)")
PREP_LEDGER_MARK = "**NPU PORT FOLD A PREPARATIONS FOR THE L338 FOLD A CONFIRMATION CELL (DRAFT TEXT; NOT YET RUN).**"

ROSTER_PATH = swap(fam.ROSTER["arm"])                                   # docs/receipts/rosters/slurp-evo-fold-a-v2.json
DATASET_FRAGMENT = swap("slurp-speech-native-evo-fold-b-1080")
SPLIT_ID = "SLURP-EVO-FOLD-A-V2"
CARRIER_TEXT = "slurp TRAIN Fold A SLURP-EVO-FOLD-A-V2 1080"
INHERITED = ("TRAIN Fold A discovery rows already exposed by prior SLURP cells (VER-1 / VER-1c, ROLL-2, the Fold A encoder pass and the historical Fold A direct and "
             "lever cells included); this cell opens no split. Prior exposure is inherited in full.")


def _roster() -> dict:
    return json.loads((REPO_ROOT / ROSTER_PATH).read_text(encoding="utf-8"))


ROSTER = _roster()
if ROSTER["roster_id"] != SPLIT_ID or len(ROSTER["items"]) != fam.ROWS:
    raise ValueError("the Fold A roster is not SLURP-EVO-FOLD-A-V2 with the family's row count")
SPLIT_IDENTITY_HASH = ROSTER["split_identity_hash"]
ROSTER_AUDIO_SECONDS = sum(float(item["audio_seconds"]) for item in ROSTER["items"])


def load_pins() -> dict:
    """``{relpath: sha256}`` of the Fold A artefacts the preparations FA1-FA4 and the cell's supply have sealed (empty until they are run)."""

    doc = json.loads(PINS_PATH.read_text(encoding="utf-8")) if PINS_PATH.is_file() else {"pins": {}}
    pins = doc.get("pins", {})
    if not all(isinstance(k, str) and isinstance(v, str) and len(v) == 64 for k, v in pins.items()):
        raise ValueError("fold_a_port_pins.json: pins must map a relpath to a 64-hex sha256")
    return pins


def sha_for(relpath: str, repo_root: Path = REPO_ROOT) -> str:
    """The sha256 that pins Fold A ``relpath``: the real file digest of a committed repository file (the roster), a value of the pins file, else the zero sentinel."""

    import hashlib  # noqa: PLC0415

    path = Path(repo_root) / relpath
    if relpath.startswith("docs/") and path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    return load_pins().get(relpath, ZERO)


def _spec() -> dict:
    hist = pf.CELLS["arm"]
    run = swap(hist["run"])
    return {
        "key": "arm", "role": f"lever-column flight on TRAIN Fold A ({pf.K} arms in one flight; the Fold A confirmation pass)", "rows": hist["rows"],
        "roster": "arm", "scoring_jobs": hist["scoring_jobs"],
        "historical_cell": hist["cell"], "historical_baseline": hist["baseline"], "historical_run": hist["run"], "historical_fit_run": hist["fit_run"],
        "cell": swap(hist["cell"]), "baseline": swap(hist["baseline"]), "run": run, "fit_run": swap(hist["fit_run"]), "attempt": swap(hist["attempt"]),
        "fit_attempt": swap(hist["fit_attempt"]), "model": pf.MODEL_FRAGMENT, "session": pf.SERVICE_SESSION, "prefix": hist["prefix"],
        "supply_relpath": f"prepared/{run}/phase-supply-v1.json", "plan_gpu_hours_placeholder": hist["plan_gpu_hours_placeholder"],
        "historical_role": hist["role"],
    }


CELLS = {"arm": _spec()}
ALL_KEYS = ("arm",)
COMPARATOR_CELL = swap("slurp-native-stage3-l338-lever-column-fold-b-comparator-npu-port-v1")
COMPARATOR_RUN = swap("SAEA-SLURP-STAGE3-L338-LEVER-COLUMN-FOLD-B-COMPARATOR-NPU-PORT-V1")

# the READ of the confirmation (the Fold B port reader/recipe re-pointed at Fold A is a later item; ids only here)
READ_RUN_ID = swap(pf.READ_RUN_ID)
ANALYSIS_ROOT = f"analysis/{READ_RUN_ID}"
RECIPE_RELPATH = f"{PACKAGE}/analysis-recipe-v1.json"
SIGNOFF_CONDITIONS_RELPATH = f"{PACKAGE}/signoff-conditions-v1.json"

#: Audio ceiling: ceil(roster audio x (contacts + fallback contacts per row)); the Fold A roster is 3,084.5284375 s.
AUDIO_FACTOR = pf.CONTACTS_PER_ROW + pf.FALLBACK_CONTACTS_PER_ROW_CEILING


def audio_seconds(key: str = "arm") -> int:
    return math.ceil(ROSTER_AUDIO_SECONDS * AUDIO_FACTOR - 1e-9)


def eligible_run_ids() -> list[str]:
    return [CELLS["arm"]["run"], CELLS["arm"]["fit_run"]]


def eligible_attempt_ids() -> list[str]:
    return [CELLS["arm"]["attempt"], CELLS["arm"]["fit_attempt"]]


def signoff_conditions() -> dict:
    """No smoke is registered for Fold A (the harness bytes were smoked on Fold B), so there is no machine sign-off table: the document says so and keeps the Fold B
    port conditions by reference."""

    return {"schema": "saea-smoke-signoff-conditions-v1", "package": PKG_NAME, "applicable": False,
            "_note": ("The Fold A confirmation cell has no smoke: the programs and harness bytes were smoked and signed off on Fold B "
                      f"({pf.PACKAGE}/signoff-conditions-v1.json). There is no decode-cost table to read for this package; the Fold B sign-off is the antecedent."),
            "expected_arms": sorted(pf.ARM_ORDER), "antecedent": f"{pf.PACKAGE}/signoff-conditions-v1.json"}


# ---- the caps, restated against the gates (as l338_port_family does at import) ----------------------------------------------------------------
EXPECTED_CAPS = (36720, 18360, math.ceil(ROSTER_AUDIO_SECONDS * 34 - 1e-9))
if (pf.generation_calls("arm"), pf.fit_calls("arm"), audio_seconds()) != EXPECTED_CAPS or EXPECTED_CAPS[2] != 104874:
    raise ValueError(f"the Fold A confirmation caps moved: {(pf.generation_calls('arm'), pf.fit_calls('arm'), audio_seconds())}")
if CELLS["arm"]["run"] != "SAEA-SLURP-STAGE3-L338-LEVER-COLUMN-FOLD-A-NPU-PORT-V1":
    raise ValueError("the Fold A id rule no longer reproduces the Fold A run id")

__all__ = [name for name in globals() if not name.startswith("_")]

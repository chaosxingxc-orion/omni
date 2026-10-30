"""The TFRL-v2 consumption rail: the declaration a cell pins, the refusals a flight start applies, and the pure composition of what the Solver is given.

Plan ``docs/superpowers/plans/2026-10-30-tfrl-three-way-comparison-and-improvement-plan.md`` section 9 ruling 4 and section 11 (addenda 1 and 2): in the T arm an
exemplar block of at most 2,000 tokens REPLACES supplied items (the cell's supply bound is A's bound minus 8) and a rule bank of at most 400 tokens is Solver
instruction text. Both come from a bank the distiller wrote (``scripts/flights/nx/tfrl_v2_distiller.py``) and released; a flight may consume it only when

* the bank receipt's content hash equals the hash PINNED in the module declaration (the zero sentinel is a PENDING pin and is refused);
* the receipt's token counts came from the vLLM tokenize measure (``estimate`` is refused) and its exemplars quote the resolved supplied-item text;
* none of the receipt's source halves is the consuming roster (cross-fit, plan section 9 ruling 3), and no source row or split identity is the consumer's;
* every source half lies in the declared construction scope (the R01C halves): a test-split (T0) row, or any other roster, is never a bank source.

This module is MODEL-FREE and PURE: it contacts nothing, reads only the files it is handed, and changes no rendering path. It is reached from the module
declaration (``rails.consumption_module``: the optional block key ``tfrl_v2_bank``, parsed here by :func:`parse_bank_declaration`) and from a flight start
(:func:`verify_for_flight`, called from ``core.flight.runtime.FlightRuntime._prepare_native_phase_runtime`` with the flight's own row ids; it returns ``None`` for a cell with no frame).
``rails/runner.py`` calls :func:`compose_solver_messages` for a Solver contact only when the cell's module block carries the frame AND the bank passed :func:`verify_for_flight`
(plan section 13 ruling 3 (a)); without the frame no composition function is reachable from a request, so every existing label's bytes are exactly what they were.

Message composition (:func:`compose_solver_messages`) follows the runner's own message shape (``system`` text message, ``user`` content list): the rule bank is a
pure SUFFIX of the system text (the pattern of the knowledge index) and the exemplar block is one trailing text part without a ``knowledge_role`` (so the SUPPLY
accounting, which reads ``key`` / ``value`` parts only, does not see it). With no bank (both texts empty) the messages are returned unchanged.

``used_basis`` is read from the rollout trace (``contracts.rollout_trace``): :func:`used_basis_summary` counts, per basis, the supply items of the Solver contacts
of a flight, and refuses a line that does not carry the closed field.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..contracts import rollout_trace as rt
from .errors import RailsError

ALGORITHM_ID = "tfrl-v2"
DECLARATION_SCHEMA = "saea-tfrl-v2-bank-consumption-v1"
BANK_RECEIPT_SCHEMA = "saea-tfrl-v2-bank-receipt-v1"
#: The block key inside ``native_episode.consumption_module`` that carries the declaration.
BLOCK_KEY = "tfrl_v2_bank"
ZERO_SHA256 = "0" * 64

EXEMPLAR_BLOCK_TOKEN_CAP = 2000
RULE_BANK_TOKEN_CAP = 400
#: Plan section 9 ruling 4: the block of at most 2,000 tokens is about 8 supplied items' worth.
EXEMPLAR_ITEM_EQUIVALENT = 8
MEASURE_VLLM = "vllm-tokenize"
MEASURE_ESTIMATE = "estimate"
EXEMPLAR_TEXT_RESOLVED = "resolved"
RULE_DELIVERY = "solver-instruction"
USED_BASIS_SOURCE = "trace"

KIND_HALF = "half"
KIND_UNION = "union"
CONSUMER_FOLD_B = "FOLD-B"
CONSUMER_FOLD_A = "FOLD-A"
HALF_IDS = ("R01C-H1", "R01C-H2")
UNION_CONSUMERS = (CONSUMER_FOLD_B, CONSUMER_FOLD_A)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_TOP_KEYS = frozenset({
    "schema", "algorithm_id", "iteration", "consumer", "bank_kind", "bank_receipt", "measure", "exemplar_block", "rule_bank", "source_scope", "used_basis",
})
_RECEIPT_KEYS = frozenset({"relpath", "sha256"})
_EXEMPLAR_KEYS = frozenset({"token_cap", "replaces_supplied_items", "text"})
_RULE_KEYS = frozenset({"token_cap", "delivery"})


class TfrlV2BankError(RailsError):
    """A TFRL-v2 bank declaration is malformed. Raised at config load, before any contact."""


class TfrlV2BankRefusal(TfrlV2BankError):
    """A bank may not be consumed by this flight. Raised at flight start, before any contact."""


# ---------------------------------------------------------------------------
# hashing
# ---------------------------------------------------------------------------


def canonical_json(obj: Any) -> str:
    """Compact, key-sorted, UTF-8 JSON text: the form every bank content hash is taken over (equal to ``tfrl_v2_common.canonical_json``)."""

    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_sha256(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def receipt_sha256(receipt: Mapping[str, Any]) -> str:
    """The BANK RECEIPT HASH: the content hash of the receipt document. It covers the file hashes of both banks, the sources, the measure and the counts, so
    pinning it pins the whole release."""

    return content_sha256(receipt)


# ---------------------------------------------------------------------------
# the declaration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BankDeclaration:
    """The parsed ``tfrl_v2_bank`` frame of a module block."""

    iteration: int
    consumer: str
    bank_kind: str
    receipt_relpath: str
    receipt_sha256: str
    measure: str
    exemplar_token_cap: int
    exemplar_replaces_items: int
    rule_token_cap: int
    source_scope: tuple[tuple[str, str], ...]

    @property
    def pending(self) -> bool:
        return self.receipt_sha256 == ZERO_SHA256

    @property
    def scope(self) -> dict[str, str]:
        return dict(self.source_scope)


def _hex(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise TfrlV2BankError(f"{BLOCK_KEY}.{field} must be a lowercase 64-hex sha256 (the zero sentinel marks a PENDING pin)")
    return value


def _exact(raw: object, keys: frozenset[str], field: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != keys:
        raise TfrlV2BankError(f"{field} must be an object with exactly the keys {sorted(keys)}")
    return raw


def parse_bank_declaration(raw: object) -> BankDeclaration:
    """Validate one ``tfrl_v2_bank`` frame, fail-closed.

    Refused: unknown or missing keys; a schema or algorithm id other than the registered ones; a cap above the plan's (2,000 exemplar tokens, 400 rule tokens) or
    below one; ``replaces_supplied_items`` other than 8; a measure other than ``vllm-tokenize`` (an estimate-measured bank can never be pinned); exemplar text other
    than ``resolved``; a consumer that does not fit the bank kind; a source scope that is not the two construction halves with 64-hex split identities."""

    top = _exact(raw, _TOP_KEYS, BLOCK_KEY)
    if top["schema"] != DECLARATION_SCHEMA:
        raise TfrlV2BankError(f"{BLOCK_KEY}.schema must be {DECLARATION_SCHEMA!r}, got {top['schema']!r}")
    if top["algorithm_id"] != ALGORITHM_ID:
        raise TfrlV2BankError(f"{BLOCK_KEY}.algorithm_id must be {ALGORITHM_ID!r}, got {top['algorithm_id']!r}")
    iteration = top["iteration"]
    if isinstance(iteration, bool) or not isinstance(iteration, int) or not 1 <= iteration <= 3:
        raise TfrlV2BankError(f"{BLOCK_KEY}.iteration must be an integer in 1..3 (the loop runs at most 3 iterations)")
    kind, consumer = top["bank_kind"], top["consumer"]
    if kind == KIND_HALF:
        if consumer not in HALF_IDS:
            raise TfrlV2BankError(f"{BLOCK_KEY}: a half bank is consumed by a construction half {list(HALF_IDS)}, got consumer {consumer!r}")
    elif kind == KIND_UNION:
        if consumer not in UNION_CONSUMERS:
            raise TfrlV2BankError(f"{BLOCK_KEY}: the union bank is consumed by {list(UNION_CONSUMERS)} only, got consumer {consumer!r}")
    else:
        raise TfrlV2BankError(f"{BLOCK_KEY}.bank_kind must be {KIND_HALF!r} or {KIND_UNION!r}, got {kind!r}")
    receipt = _exact(top["bank_receipt"], _RECEIPT_KEYS, f"{BLOCK_KEY}.bank_receipt")
    relpath = receipt["relpath"]
    if (not isinstance(relpath, str) or not relpath or "\\" in relpath or relpath.startswith("/")
            or any(part in ("", ".", "..") for part in relpath.split("/"))):
        raise TfrlV2BankError(f"{BLOCK_KEY}.bank_receipt.relpath must be a data-root-relative POSIX path, got {relpath!r}")
    digest = _hex(receipt["sha256"], "bank_receipt.sha256")
    if top["measure"] != MEASURE_VLLM:
        raise TfrlV2BankError(
            f"{BLOCK_KEY}.measure must be {MEASURE_VLLM!r}, got {top['measure']!r}: a bank whose token counts are estimates cannot be pinned"
        )
    exemplar = _exact(top["exemplar_block"], _EXEMPLAR_KEYS, f"{BLOCK_KEY}.exemplar_block")
    cap = exemplar["token_cap"]
    if isinstance(cap, bool) or not isinstance(cap, int) or not 1 <= cap <= EXEMPLAR_BLOCK_TOKEN_CAP:
        raise TfrlV2BankError(f"{BLOCK_KEY}.exemplar_block.token_cap must be an integer in 1..{EXEMPLAR_BLOCK_TOKEN_CAP}, got {cap!r}")
    if exemplar["replaces_supplied_items"] != EXEMPLAR_ITEM_EQUIVALENT or isinstance(exemplar["replaces_supplied_items"], bool):
        raise TfrlV2BankError(f"{BLOCK_KEY}.exemplar_block.replaces_supplied_items must be {EXEMPLAR_ITEM_EQUIVALENT} (exemplars replace supplied items, plan section 9 ruling 4)")
    if exemplar["text"] != EXEMPLAR_TEXT_RESOLVED:
        raise TfrlV2BankError(f"{BLOCK_KEY}.exemplar_block.text must be {EXEMPLAR_TEXT_RESOLVED!r}")
    rules = _exact(top["rule_bank"], _RULE_KEYS, f"{BLOCK_KEY}.rule_bank")
    rule_cap = rules["token_cap"]
    if isinstance(rule_cap, bool) or not isinstance(rule_cap, int) or not 1 <= rule_cap <= RULE_BANK_TOKEN_CAP:
        raise TfrlV2BankError(f"{BLOCK_KEY}.rule_bank.token_cap must be an integer in 1..{RULE_BANK_TOKEN_CAP}, got {rule_cap!r}")
    if rules["delivery"] != RULE_DELIVERY:
        raise TfrlV2BankError(f"{BLOCK_KEY}.rule_bank.delivery must be {RULE_DELIVERY!r} (the rule bank is Solver instruction text)")
    scope = top["source_scope"]
    if not isinstance(scope, Mapping) or sorted(scope) != sorted(HALF_IDS):
        raise TfrlV2BankError(f"{BLOCK_KEY}.source_scope must map exactly the construction halves {list(HALF_IDS)} to their split identities")
    pairs = tuple((half, _hex(scope[half], f"source_scope.{half}")) for half in HALF_IDS)
    if top["used_basis"] != USED_BASIS_SOURCE:
        raise TfrlV2BankError(f"{BLOCK_KEY}.used_basis must be {USED_BASIS_SOURCE!r} (the basis of a used flag is read from the rollout trace)")
    return BankDeclaration(
        iteration=iteration, consumer=consumer, bank_kind=kind, receipt_relpath=relpath, receipt_sha256=digest, measure=top["measure"],
        exemplar_token_cap=cap, exemplar_replaces_items=EXEMPLAR_ITEM_EQUIVALENT, rule_token_cap=rule_cap, source_scope=pairs,
    )


def declaration_block(
    *, iteration: int, consumer: str, bank_kind: str, receipt_relpath: str, receipt_sha: str = ZERO_SHA256, source_scope: Mapping[str, str],
    exemplar_token_cap: int = EXEMPLAR_BLOCK_TOKEN_CAP, rule_token_cap: int = RULE_BANK_TOKEN_CAP,
) -> dict[str, Any]:
    """The ``tfrl_v2_bank`` frame as a plain dict (the form a cell's module block carries); the result always parses."""

    block = {
        "schema": DECLARATION_SCHEMA, "algorithm_id": ALGORITHM_ID, "iteration": iteration, "consumer": consumer, "bank_kind": bank_kind,
        "bank_receipt": {"relpath": receipt_relpath, "sha256": receipt_sha}, "measure": MEASURE_VLLM,
        "exemplar_block": {"token_cap": exemplar_token_cap, "replaces_supplied_items": EXEMPLAR_ITEM_EQUIVALENT, "text": EXEMPLAR_TEXT_RESOLVED},
        "rule_bank": {"token_cap": rule_token_cap, "delivery": RULE_DELIVERY}, "source_scope": dict(source_scope), "used_basis": USED_BASIS_SOURCE,
    }
    parse_bank_declaration(block)
    return block


# ---------------------------------------------------------------------------
# the refusals applied at a flight start
# ---------------------------------------------------------------------------


def verify_bank(
    decl: BankDeclaration, receipt: Mapping[str, Any], *, consuming_id: str,
    consuming_row_ids: Sequence[str] | None = None, consuming_split_identity: str | None = None,
) -> str:
    """Refuse, by name, a bank the flight may not consume; return the receipt hash when it may.

    Checked in this order (the first failure is the one raised): the receipt schema; the pin (PENDING or different from the receipt's content hash); the measure
    (``estimate``); the exemplar text; the iteration and the bank kind; the source scope (a source outside the declared construction halves, e.g. a test-split T0
    roster, or a split identity that is not the pinned one); the cross-fit (a source half that is the consuming roster, a source split identity equal to the
    consumer's, a consuming row among the source rows); the caps."""

    if not isinstance(receipt, Mapping) or receipt.get("schema") != BANK_RECEIPT_SCHEMA:
        raise TfrlV2BankRefusal(f"tfrl-v2 bank refused: the receipt schema is not {BANK_RECEIPT_SCHEMA!r}")
    actual = receipt_sha256(receipt)
    if decl.pending:
        raise TfrlV2BankRefusal(
            "tfrl-v2 bank refused: the bank receipt hash pinned in the module declaration is the PENDING zero sentinel; release the bank and pin "
            f"its receipt hash {actual} before a flight"
        )
    if actual != decl.receipt_sha256:
        raise TfrlV2BankRefusal(f"tfrl-v2 bank refused: the bank receipt hash {actual} differs from the hash {decl.receipt_sha256} pinned in the module declaration")
    measure = receipt.get("measure")
    if measure != decl.measure:
        raise TfrlV2BankRefusal(f"tfrl-v2 bank refused: the bank's token counts were produced by measure {measure!r}, not {decl.measure!r}")
    if receipt.get("exemplar_text") != EXEMPLAR_TEXT_RESOLVED:
        raise TfrlV2BankRefusal(f"tfrl-v2 bank refused: the bank's exemplars do not carry the resolved supplied-item text (exemplar_text must be {EXEMPLAR_TEXT_RESOLVED!r})")
    if receipt.get("iteration") != decl.iteration:
        raise TfrlV2BankRefusal(f"tfrl-v2 bank refused: the bank is of iteration {receipt.get('iteration')!r}, the declaration is of iteration {decl.iteration}")
    if receipt.get("kind") != decl.bank_kind:
        raise TfrlV2BankRefusal(f"tfrl-v2 bank refused: the bank is a {receipt.get('kind')!r} bank, the declaration consumes a {decl.bank_kind!r} bank")
    if consuming_id != decl.consumer:
        raise TfrlV2BankRefusal(f"tfrl-v2 bank refused: the consuming roster is {consuming_id!r}, the declaration names consumer {decl.consumer!r}")
    sources = receipt.get("source_halves")
    if not isinstance(sources, list) or not sources:
        raise TfrlV2BankRefusal("tfrl-v2 bank refused: the bank receipt names no source_halves")
    scope = decl.scope
    identities = receipt.get("source_split_identities") or {}
    for source in sources:
        if source not in scope:
            raise TfrlV2BankRefusal(
                f"tfrl-v2 bank refused: bank source {source!r} is outside the declared construction scope {sorted(scope)}; "
                "test-split (T0) rows and every other roster are never bank sources"
            )
        if identities.get(source) != scope[source]:
            raise TfrlV2BankRefusal(f"tfrl-v2 bank refused: the split identity of bank source {source} is not the pinned construction identity")
    if consuming_id in sources:
        raise TfrlV2BankRefusal(f"tfrl-v2 bank refused: the bank was distilled from {sources}, which includes the consuming roster {consuming_id}")
    if consuming_split_identity is not None and consuming_split_identity in set(identities.values()):
        raise TfrlV2BankRefusal("tfrl-v2 bank refused: the consuming split identity equals a source split identity of the bank")
    if consuming_row_ids is not None:
        overlap = sorted(set(consuming_row_ids) & set(receipt.get("source_row_ids") or []))
        if overlap:
            raise TfrlV2BankRefusal(f"tfrl-v2 bank refused: {len(overlap)} consuming rows are source rows of the bank (first: {overlap[0]})")
    exemplar_tokens = (receipt.get("exemplar_bank") or {}).get("max_block_tokens")
    if not isinstance(exemplar_tokens, int) or exemplar_tokens > decl.exemplar_token_cap:
        raise TfrlV2BankRefusal(f"tfrl-v2 bank refused: the largest exemplar block is {exemplar_tokens!r} tokens, over the declared cap {decl.exemplar_token_cap}")
    rule_tokens = (receipt.get("rule_bank") or {}).get("rendered_tokens")
    if not isinstance(rule_tokens, int) or rule_tokens > decl.rule_token_cap:
        raise TfrlV2BankRefusal(f"tfrl-v2 bank refused: the rendered rule bank is {rule_tokens!r} tokens, over the declared cap {decl.rule_token_cap}")
    return actual


@dataclass(frozen=True)
class LoadedBank:
    """A verified bank: the two bank documents and the receipt hash that was pinned."""

    receipt: Mapping[str, Any]
    receipt_sha256: str
    exemplar_bank: Mapping[str, Any]
    rule_bank: Mapping[str, Any]

    def exemplar_block_text(self, scenario: str | None) -> str:
        """The rendered block of the scenario the Perception node predicted, or the empty string when the bank has none for it."""

        block = (self.exemplar_bank.get("by_scenario") or {}).get(scenario) if scenario else None
        return "" if not block else str(block["block_text"])

    def scenarios(self) -> tuple[str, ...]:
        return tuple(sorted(self.exemplar_bank.get("by_scenario") or {}))

    def exemplar_block_for_intent(self, intent_id: str | None) -> str:
        """The block of the scenario ``intent_id`` (``<scenario>_<action>``) belongs to, or the empty string when the intent is unknown or the bank has no block for it."""

        return self.exemplar_block_text(scenario_of_intent(intent_id, self.scenarios()))

    @property
    def source_uri(self) -> str:
        """The ``source_uri`` the exemplar part carries (the raw-native request refuses a text part without one); it names the pinned receipt."""

        return f"{BANK_SOURCE_SCHEME}{self.receipt_sha256}/exemplar-block"

    def rule_text(self) -> str:
        return str(self.rule_bank.get("rendered_text") or "")


def verify_bank_dir(
    decl: BankDeclaration, data_root: Path, *, consuming_id: str,
    consuming_row_ids: Sequence[str] | None = None, consuming_split_identity: str | None = None,
) -> LoadedBank:
    """Load the receipt at ``data_root / decl.receipt_relpath``, re-hash every bank file against it, then apply :func:`verify_bank`."""

    path = Path(data_root) / decl.receipt_relpath
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise TfrlV2BankRefusal(f"tfrl-v2 bank refused: the bank receipt {decl.receipt_relpath} cannot be read ({error.__class__.__name__})") from error
    digest = verify_bank(decl, receipt, consuming_id=consuming_id, consuming_row_ids=consuming_row_ids, consuming_split_identity=consuming_split_identity)
    banks: list[Mapping[str, Any]] = []
    for key in ("exemplar_bank", "rule_bank"):
        name = (receipt.get(key) or {}).get("file")
        pinned = ((receipt.get("files") or {}).get(name) or {}).get("sha256")
        try:
            raw = (path.parent / str(name)).read_bytes()
        except OSError as error:
            raise TfrlV2BankRefusal(f"tfrl-v2 bank refused: the bank file {name!r} cannot be read ({error.__class__.__name__})") from error
        if hashlib.sha256(raw).hexdigest() != pinned:
            raise TfrlV2BankRefusal(f"tfrl-v2 bank refused: the file hash of {name} differs from the receipt")
        bank = json.loads(raw.decode("utf-8"))
        if content_sha256(bank) != receipt[key].get("content_sha256"):
            raise TfrlV2BankRefusal(f"tfrl-v2 bank refused: the content hash of {name} differs from the receipt")
        banks.append(bank)
    return LoadedBank(receipt=receipt, receipt_sha256=digest, exemplar_bank=banks[0], rule_bank=banks[1])


def declaration_of_values(values: Mapping[str, Any]) -> BankDeclaration | None:
    """The bank declaration of a composed config, or ``None`` when its module block carries no ``tfrl_v2_bank`` frame (every cell written before TFRL-v2)."""

    episode = values.get("native_episode") if isinstance(values, Mapping) else None
    block = episode.get("consumption_module") if isinstance(episode, Mapping) else None
    if not isinstance(block, Mapping) or BLOCK_KEY not in block:
        return None
    return parse_bank_declaration(block[BLOCK_KEY])


def verify_for_flight(
    values: Mapping[str, Any], data_root: Path, *, split_identity_hash: str, consuming_row_ids: Sequence[str] | None = None,
) -> LoadedBank | None:
    """The flight-start entry: refuse, before the first contact, a cell whose declared bank may not be consumed; ``None`` for a cell with no declaration.

    The consuming roster is the declared consumer; its identity is the flight's own split identity. A half consumer must BE the half it names (the flight's
    identity equals the pinned half identity); a Fold B / Fold A consumer must carry an identity that is none of the bank's source identities.

    ROW DISJOINTNESS (plan section 13 ruling 3 (d)): a cell that declares a bank must hand over the flight's own row ids (``consuming_row_ids``); they are re-checked
    against the receipt's ``source_row_ids`` here, so a bank whose contributing rows include a consuming row is refused even when the split identities differ. A declared
    cell with no row ids (``None`` or empty) is refused: the disjointness cannot be re-checked, and an unchecked bank is never consumed."""

    decl = declaration_of_values(values)
    if decl is None:
        return None
    if not consuming_row_ids:
        raise TfrlV2BankRefusal(
            "tfrl-v2 bank refused: the flight handed over no consuming row ids, so row disjointness from the bank's source_row_ids cannot be re-checked"
        )
    scope = decl.scope
    if decl.consumer in scope and scope[decl.consumer] != split_identity_hash:
        raise TfrlV2BankRefusal(
            f"tfrl-v2 bank refused: the cell declares consumer {decl.consumer} but its split identity is not the pinned identity of that half"
        )
    return verify_bank_dir(decl, data_root, consuming_id=decl.consumer, consuming_row_ids=consuming_row_ids, consuming_split_identity=split_identity_hash)


# ---------------------------------------------------------------------------
# composition (pure; reachable only from a cell that declares the frame)
# ---------------------------------------------------------------------------

RULES_SEPARATOR = "\n\n"
EXEMPLAR_EVIDENCE_ID = "tfrl-v2/exemplar-block"


def compose_solver_messages(
    messages: Sequence[Mapping[str, Any]], *, exemplar_block_text: str = "", rule_text: str = "", exemplar_source_uri: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The Solver's two messages with the bank added; the input is not modified.

    ``messages`` is the runner's pair: ``{"role": "system", "content": <text>}`` and ``{"role": "user", "content": [<parts>]}``. The rule text is appended to the
    system text as a suffix; the exemplar block is appended as one text part with no ``knowledge_role`` (and, when ``exemplar_source_uri`` is given, that ``source_uri``,
    which the raw-audio request requires of every text part). With both texts empty the result equals the input."""

    if len(messages) != 2 or messages[0].get("role") != "system" or messages[1].get("role") != "user":
        raise TfrlV2BankError("compose_solver_messages needs exactly the system and user messages")
    system, user = copy.deepcopy(dict(messages[0])), copy.deepcopy(dict(messages[1]))
    if not isinstance(system.get("content"), str) or not isinstance(user.get("content"), list):
        raise TfrlV2BankError("the system content must be text and the user content a list of parts")
    if rule_text:
        system["content"] = f"{system['content']}{RULES_SEPARATOR}{rule_text}"
    if exemplar_block_text:
        part = {"type": "text", "text": exemplar_block_text, "evidence_id": EXEMPLAR_EVIDENCE_ID}
        if exemplar_source_uri is not None:
            part["source_uri"] = exemplar_source_uri
        user["content"].append(part)
    return system, user


# ---------------------------------------------------------------------------
# the runner's seams (pure helpers; rails/runner.py applies them behind the declaration)
# ---------------------------------------------------------------------------

BANK_SOURCE_SCHEME = "tfrl-v2-bank://"


def scenario_of_intent(intent_id: object, scenarios: Sequence[str]) -> str | None:
    """The scenario an intent id ``<scenario>_<action>`` belongs to: the LONGEST bank scenario ``s`` with ``intent_id`` starting ``s + "_"``; ``None`` when none does."""

    if not isinstance(intent_id, str) or not intent_id:
        return None
    best = None
    for scenario in scenarios:
        if intent_id.startswith(f"{scenario}_") and (best is None or len(scenario) > len(best)):
            best = scenario
    return best


def predicted_intent_id(states: Sequence[Any], perception_state_kind: str) -> str | None:
    """The top candidate's intent id of the LATEST Perception state (``state_kind`` equal to ``perception_state_kind``) whose content is the ``{"candidates": [{"intent_id": ...}]}``
    document; ``None`` when there is no such state or it does not parse (the contact then gets no exemplar block, and the rule bank still applies)."""

    for state in reversed(list(states)):
        if getattr(state, "state_kind", None) != perception_state_kind:
            continue
        try:
            document = json.loads(getattr(state, "content", ""))
        except (TypeError, ValueError):
            return None
        rows = document.get("candidates") if isinstance(document, Mapping) else None
        if isinstance(rows, list) and rows and isinstance(rows[0], Mapping) and isinstance(rows[0].get("intent_id"), str):
            return rows[0]["intent_id"]
        return None
    return None


_FRAME_KEYS = ("decoding_frame", "sampling_frame", "supply_frame")


def supply_bound_of_block(block: Mapping[str, Any]) -> int | None:
    """The supplied-item bound the module block's arm frame declares (``max_supplied_items`` of its decoding / sampling frame); ``None`` when the block declares none.

    In a T cell this is A's bound minus the 8 items the exemplar block replaces (plan section 9 ruling 4): 43 where A's is 51."""

    for key in _FRAME_KEYS:
        frame = block.get(key)
        if isinstance(frame, Mapping):
            bound = frame.get("max_supplied_items")
            if isinstance(bound, int) and not isinstance(bound, bool) and bound > 0:
                return bound
    return None


def declaration_of_block_json(block_json: str | None) -> tuple[BankDeclaration, int | None] | None:
    """``(declaration, supply bound)`` of a config's canonical module block text, or ``None`` when the block (or the text) carries no ``tfrl_v2_bank`` frame."""

    if not block_json:
        return None
    block = json.loads(block_json)
    if not isinstance(block, Mapping) or BLOCK_KEY not in block:
        return None
    return parse_bank_declaration(block[BLOCK_KEY]), supply_bound_of_block(block)


# ---------------------------------------------------------------------------
# used_basis, read from the trace
# ---------------------------------------------------------------------------


def used_basis_summary(contact_lines: Sequence[Mapping[str, Any]], *, node: str = "solver") -> dict[str, Any]:
    """Count the supply items of the ``node`` contacts of a flight by ``used_basis`` (the closed field of ``saea-rollout-trace-v1``) and by ``used``.

    Refuses a line whose supply items do not carry the closed field (a legacy line is ``unknown``, not ``none``)."""

    counts = {basis: 0 for basis in rt.USED_BASES}
    used = {"true": 0, "false": 0, "null": 0}
    contacts = 0
    for line in contact_lines:
        if line.get("kind") != rt.KIND_CONTACT or line.get("node") != node:
            continue
        contacts += 1
        for item in line.get("supply_items") or []:
            basis = item.get("used_basis") if isinstance(item, Mapping) else None
            if basis not in counts:
                raise TfrlV2BankError(f"a {node} contact carries a supply item without a closed used_basis (one of {list(rt.USED_BASES)})")
            counts[basis] += 1
            used["null" if item["used"] is None else "true" if item["used"] else "false"] += 1
    items = sum(counts.values())
    judged = items - counts[rt.USED_BASIS_NONE]
    return {"contacts": contacts, "supply_items": items, "by_basis": counts, "by_used": used, "judged_items": judged,
            "judged_share": round(judged / items, 6) if items else 0.0, "source": USED_BASIS_SOURCE}


__all__ = [
    "ALGORITHM_ID", "BANK_RECEIPT_SCHEMA", "BLOCK_KEY", "BankDeclaration", "CONSUMER_FOLD_A", "CONSUMER_FOLD_B", "DECLARATION_SCHEMA", "EXEMPLAR_BLOCK_TOKEN_CAP",
    "EXEMPLAR_ITEM_EQUIVALENT", "HALF_IDS", "KIND_HALF", "KIND_UNION", "LoadedBank", "MEASURE_ESTIMATE", "MEASURE_VLLM", "RULE_BANK_TOKEN_CAP", "TfrlV2BankError",
    "TfrlV2BankRefusal", "ZERO_SHA256", "canonical_json", "compose_solver_messages", "content_sha256", "declaration_block", "parse_bank_declaration",
    "declaration_of_values", "declaration_of_block_json", "predicted_intent_id", "scenario_of_intent", "supply_bound_of_block", "BANK_SOURCE_SCHEME", "receipt_sha256", "used_basis_summary", "verify_bank", "verify_bank_dir", "verify_for_flight",
]

"""TFRL-v2 item-text resolver, token measures, exemplar text, the union bank and the measured-bank gate (task T3b; model-free, no card).

Proved here, on fake trace lines, a fake tokenizer client and (when the data root is present) the real knowledge stores:

* the resolver maps a supplied item id to the exact text the Solver read, rendered as the transport renders it, with a content hash per item; an
  unknown id is refused; the format string is pinned against the transport source; the real stores resolve a real id (skipped without a data root);
* the token measure is injectable: the vLLM tokenize measure goes through the FIT gate's function and caches; the caps (2,000 per exemplar block,
  400 for the rule bank) are applied with whatever measure is given; the bank receipt names the measure;
* the exemplar block of one fake row quotes the supplied item text and carries the item hashes; distillation is deterministic; an unknown id refuses
  the distillation;
* the sampler's plan builder refuses a bank counted with an estimate and a bank without resolved text;
* the union bank: deterministic order, per-scenario cap kept, token cap re-measured, content hash, both source halves recorded, round trip through
  ``load_bank``; the guard accepts it for Fold B rows and refuses it for either half; refusals for mismatched halves.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
NX = REPO / "scripts" / "flights" / "nx"
for _p in (str(REPO / "src"), str(REPO / "scripts" / "analysis"), str(NX), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import tfrl_v2_common as cm  # noqa: E402
import tfrl_v2_distiller as dd  # noqa: E402
import tfrl_v2_sampler as sp  # noqa: E402
import tfrl_v2_text as tx  # noqa: E402
import test_tfrl_v2_distiller as base  # noqa: E402  (fixtures only; used as a namespace so its tests are not collected twice)

H1, H2 = cm.HALF_H1, cm.HALF_H2


class FakeTokenizer:
    """The contract of the vLLM chat-form POST /tokenize: ``{"count": n, "tokens": [n ids]}``; here ``cpt`` characters per token plus a 7-token wrapper."""

    def __init__(self, cpt=4):
        self.cpt = cpt
        self.requests = []

    def request_json(self, method, path, payload):
        self.requests.append((method, path, payload))
        assert (method, path) == ("POST", "/tokenize")
        text = payload["messages"][0]["content"][0]["text"]
        n = -(-len(text) // self.cpt) + 7
        return {"count": n, "tokens": list(range(n))}


def vllm(cpt=4):
    return tx.VllmTokenizeMeasure(FakeTokenizer(cpt))


def resolver_for(fixtures):
    ids = {item["item_id"] for fx in fixtures for c in fx.contacts for item in c["supply_items"]}
    return tx.ItemTextResolver({i: f"{i} | text of {i} | date=monday" for i in sorted(ids)}), ids


def roster_of(fx, half):
    return {"roster_id": cm.HALF_ROSTER_ID[half], "split_identity_hash": ("a" if half == H1 else "b") * 64,
            "items": [{"sample_id": r, "audio_seconds": 2.0} for r in fx.rows]}


def distil_half(fx, half, *, resolver, measure, iteration=1):
    return dd.distil(roster_of(fx, half), half, iteration, fx.contacts, fx.scores, resolver=resolver, measure=measure)


def second_half_fixture(offset=100, filler="tuesday"):
    fx = base.Fixture()
    for i in range(4):
        base.date_row(fx, offset + i, filler=filler)
    return fx


# ---- the resolver ---------------------------------------------------------------------------------------------------------------------------


def test_resolver_returns_the_exact_text_with_a_hash_and_the_transport_line():
    resolver = tx.ItemTextResolver({"train-exemplars::value:1": "wake me up at nine am on friday | alarm_set | date=friday; time=nine am"})
    item = resolver.resolve("train-exemplars::value:1")
    assert item.content == "wake me up at nine am on friday | alarm_set | date=friday; time=nine am"
    assert item.rendered == "[evidence train-exemplars::value:1] wake me up at nine am on friday | alarm_set | date=friday; time=nine am"
    assert item.content_sha256 == tx.content_sha256_text(item.content) and len(item.content_sha256) == 64
    assert "train-exemplars::value:1" in resolver and len(resolver) == 1


def test_resolver_refuses_unknown_ids_and_empty_stores():
    resolver = tx.ItemTextResolver({"a::b": "text"})
    with pytest.raises(tx.ItemTextError, match="unknown supplied item id"):
        resolver.resolve("a::missing")
    with pytest.raises(tx.ItemTextError, match="no item text"):
        tx.ItemTextResolver({})
    with pytest.raises(tx.ItemTextError, match="empty"):
        tx.ItemTextResolver({"a": ""})


def test_resolver_identity_is_order_independent_and_carries_no_text():
    one = tx.ItemTextResolver({"a::1": "alpha", "a::2": "beta"})
    two = tx.ItemTextResolver({"a::2": "beta", "a::1": "alpha"})
    other = tx.ItemTextResolver({"a::1": "alpha", "a::2": "BETA"})
    assert one.identity() == two.identity() and one.identity()["values_index_sha256"] != other.identity()["values_index_sha256"]
    identity = one.identity()
    assert identity["rank_label"] == "none" and identity["renderer"] == "[evidence <evidence_id>] <content>" and identity["values"] == 2
    assert "alpha" not in json.dumps(identity)


def test_the_rendered_line_is_the_transport_format():
    transport = (REPO / "src" / "speech_aware_evidence_acquisition" / "core" / "model" / "transport.py").read_text(encoding="utf-8")
    assert "f\"[evidence {item['evidence_id']}] {item['content']}\"" in transport
    assert tx.render_evidence_line("x::1", "c") == "[evidence x::1] c"


DATA_ROOT = Path(os.environ.get("SPEECHRL_DATA_DIR", "/nonexistent"))


@pytest.mark.skipif(not (DATA_ROOT / "knowledge" / "slurp" / "saea-kb-distmodel-v1" / "size-pool-annotated-exemplars" / "snapshot.json").is_file(),
                    reason="the data root with the knowledge stores is not mounted")
def test_the_registered_stores_resolve_a_real_item():
    resolver = tx.ItemTextResolver.from_pins(DATA_ROOT)
    item = resolver.resolve("train-exemplars::value:1")
    assert item.content == "wake me up at nine am on friday | alarm_set | date=friday; time=nine am"
    assert item.rendered.startswith("[evidence train-exemplars::value:1] wake me up")
    identity = resolver.identity()
    assert identity["source"]["pins"] == "scripts/flights/nx/l338_port_pins.json" and identity["values"] > 9000
    assert sorted(identity["source"]["stores"]) == ["intent-boundaries", "slot-contracts", "train-exemplars"]
    with pytest.raises(tx.ItemTextError):
        resolver.resolve("train-exemplars::value:no-such-item")


def test_pipe_form_usage_rule():
    answer = base.ans("alarm", "set", [("date", "next Friday")])
    content = "wake me up on friday | alarm_set | date=friday; time=nine am"
    assert tx.pipe_item_fields(content) == ("alarm_set", ["friday", "nine am"])
    assert tx.pipe_item_used(content, answer) is True                                            # same intent
    assert tx.pipe_item_used("remind me | reminder_add | date=friday", answer) is True           # slot value occurs in a filler
    assert tx.pipe_item_used("play jazz | music_play | genre=jazz", answer) is False
    assert tx.pipe_item_used("Slot date.\nIntent bindings", answer) is None                      # not the pipe form
    items = [{"item_id": "a", "used": None}, {"item_id": "b", "used": False}, {"item_id": "c", "used": True}, {"item_id": "d", "used": None}]
    resolver = tx.ItemTextResolver({"a": content, "b": content, "c": "x", "d": "play jazz | music_play | genre=jazz"})
    assert tx.used_items(items, answer, resolver, 6) == [{"item_id": "a", "basis": "content-pipe-form"}, {"item_id": "c", "basis": "trace"}]
    assert tx.used_items(items, answer, None, 6) == [{"item_id": "c", "basis": "trace"}]
    assert tx.used_items(items, answer, resolver, 1) == [{"item_id": "a", "basis": "content-pipe-form"}]


# ---- the token measures ------------------------------------------------------------------------------------------------------------------


def test_vllm_measure_uses_the_fit_gate_function_and_caches():
    client = FakeTokenizer()
    measure = tx.VllmTokenizeMeasure(client, model="m")
    assert measure.name == "vllm-tokenize" == cm.MEASURE_VLLM
    assert measure.count("abcdefgh") == 2 + 7 and measure.count("abcdefgh") == 9 and measure.count("") == 0
    assert measure.calls == 1 and len(client.requests) == 1                                       # cached; the empty text is not sent
    method, path, payload = client.requests[0]
    assert (method, path) == ("POST", "/tokenize")
    assert payload["add_generation_prompt"] is True and payload["add_special_tokens"] is False and payload["model"] == "m"
    assert payload["messages"] == [{"role": "user", "content": [{"type": "text", "text": "abcdefgh"}]}]
    assert measure.detail()["wrapper_tokens_included"] is True


def test_vllm_measure_refuses_an_inconsistent_reply_and_a_client_without_request_json():
    class Bad:
        def request_json(self, method, path, payload):
            return {"count": 3, "tokens": [1]}

    with pytest.raises(tx.MeasureError):
        tx.VllmTokenizeMeasure(Bad()).count("text")
    with pytest.raises(tx.MeasureError):
        tx.VllmTokenizeMeasure(object())
    assert tx.EstimateMeasure().name == "estimate" == cm.MEASURE_ESTIMATE and tx.EstimateMeasure().count("x" * 7) == 2


def test_caps_are_applied_with_the_injected_measure_and_the_receipt_names_it():
    fx = base.Fixture()
    for i in range(6):
        fx.add(f"r{i:03d}", 0.5, [(1.0 - i * 0.01, True, base.ans("alarm", "set", [("note", "y" * 300 + str(i))]), ()), (0.5, True, base.ans(), ()),
                                  (0.5, True, base.ans(), ()), (0.25, True, base.ans(), ())])
    resolver, _ = resolver_for([fx])
    coarse = distil_half(fx, H1, resolver=resolver, measure=vllm(cpt=1))        # one character per token: the 2,000 cap bites early
    fine = distil_half(fx, H1, resolver=resolver, measure=vllm(cpt=8))
    n_coarse = len(coarse["exemplar_bank"]["by_scenario"]["alarm"]["entries"])
    n_fine = len(fine["exemplar_bank"]["by_scenario"]["alarm"]["entries"])
    assert n_coarse < n_fine == 6
    for result, measure in ((coarse, vllm(cpt=1)), (fine, vllm(cpt=8))):
        block = result["exemplar_bank"]["by_scenario"]["alarm"]
        assert block["block_tokens"] == measure.count(block["block_text"]) <= cm.EXEMPLAR_BLOCK_TOKEN_CAP
        assert result["exemplar_bank"]["token_measure"] == "vllm-tokenize"
        core = result["receipt_core"]
        assert core["measure"] == core["exemplar_bank"]["measure"] == core["rule_bank"]["measure"] == "vllm-tokenize"
    estimate = distil_half(fx, H1, resolver=resolver, measure=None)
    assert estimate["receipt_core"]["measure"] == "estimate" and estimate["exemplar_bank"]["token_measure"] == "estimate"


def test_rule_bank_cap_is_applied_with_the_injected_measure():
    fx = base.Fixture()
    for slot in range(10):
        for i in range(3):
            fx.add(f"r{slot:02d}{i}", 0.5, [(1.0, True, base.ans("alarm", "set", [(f"type{slot:02d}", "v")]), ()), (0.5, True, base.ans("alarm", "set"), ()),
                                           (0.5, True, base.ans("alarm", "set"), ()), (0.25, True, base.ans("alarm", "set"), ())])
    resolver, _ = resolver_for([fx])
    tight = distil_half(fx, H1, resolver=resolver, measure=vllm(cpt=1))["rule_bank"]
    loose = distil_half(fx, H1, resolver=resolver, measure=vllm(cpt=16))["rule_bank"]
    assert len(tight["rules"]) < len(loose["rules"]) and tight["rendered_tokens"] <= cm.RULE_BANK_TOKEN_CAP == 400
    assert loose["rendered_tokens"] <= 400 and tight["token_measure"] == "vllm-tokenize"


# ---- the exemplar block ------------------------------------------------------------------------------------------------------------------


def one_row_fixture():
    fx = base.Fixture()
    fx.add("r000", 0.5, [(1.0, True, base.ans("alarm", "set", [("date", "friday"), ("time", "nine am")]), ("train-exemplars::value:1", "slot-contracts::value:slot:date")),
                         (0.5, True, base.ans("alarm", "set"), ()), (0.5, True, base.ans("alarm", "set"), ()), (0.0, False, base.ans("alarm", "query"), ())])
    return fx


def test_the_exemplar_block_of_one_fake_row_quotes_the_supplied_item_text():
    fx = one_row_fixture()
    resolver = tx.ItemTextResolver({"train-exemplars::value:1": "wake me up at nine am on friday | alarm_set | date=friday; time=nine am",
                                    "slot-contracts::value:slot:date": "Slot date.\nAdmissible formats:\n- the spoken day", "unused-a": "unused text"})
    result = distil_half(fx, H1, resolver=resolver, measure=vllm())
    block = result["exemplar_bank"]["by_scenario"]["alarm"]
    assert block["block_text"] == (
        "Worked examples for alarm requests:\n"
        "- worked example\n"
        "[evidence train-exemplars::value:1] wake me up at nine am on friday | alarm_set | date=friday; time=nine am\n"
        "[evidence slot-contracts::value:slot:date] Slot date.\nAdmissible formats:\n- the spoken day\n"
        "answer: intent: alarm_set; entities: date=friday; time=nine am")
    entry = block["entries"][0]
    assert [i["item_id"] for i in entry["evidence_items"]] == ["train-exemplars::value:1", "slot-contracts::value:slot:date"]
    for item in entry["evidence_items"]:
        assert item["content_sha256"] == resolver.resolve(item["item_id"]).content_sha256 and item["used_basis"] == "trace"
    assert result["exemplar_bank"]["exemplar_text"] == "resolved"
    assert result["receipt_core"]["exemplar_text"] == "resolved" and result["receipt_core"]["exemplar_text_source"] == resolver.identity()
    assert "unused text" not in json.dumps(result["exemplar_bank"])                                 # only the used items are quoted


def test_a_row_without_a_used_item_says_so_and_the_content_rule_fills_unjudged_items():
    fx = base.Fixture()
    fx.add("r000", 0.5, [(1.0, True, base.ans("alarm", "set", [("date", "friday")]), ()), (0.5, True, base.ans(), ()), (0.5, True, base.ans(), ()), (0.0, False, base.ans(), ())])
    resolver = tx.ItemTextResolver({"unused-a": "x"})
    text = distil_half(fx, H1, resolver=resolver, measure=vllm())["exemplar_bank"]["by_scenario"]["alarm"]["entries"][0]["text"]
    assert text == "- worked example\n(no supplied item was used)\nanswer: intent: alarm_set; entities: date=friday"
    # trace could not judge the item (used null) on a pipe-form store: the content rule judges it
    contact = base.contact("r000", "s1", base.ans("alarm", "set", [("date", "friday")]))
    contact = {**contact, "supply_items": [{"item_id": "p1", "rank": 1, "score": 0.5, "used": None}]}
    fx.contacts = [contact if (c["row_id"], c["arm"]) == ("r000", "s1") else c for c in fx.contacts]
    resolver = tx.ItemTextResolver({"p1": "wake me up | alarm_set | date=friday"})
    entry = distil_half(fx, H1, resolver=resolver, measure=vllm())["exemplar_bank"]["by_scenario"]["alarm"]["entries"][0]
    assert entry["evidence_item_ids"] == ["p1"] and entry["evidence_items"][0]["used_basis"] == "content-pipe-form"


def test_an_unknown_item_id_refuses_the_distillation():
    fx = one_row_fixture()
    resolver = tx.ItemTextResolver({"train-exemplars::value:1": "text"})                           # the second used id is not a store value
    with pytest.raises(dd.DistillError, match="unknown supplied item id"):
        distil_half(fx, H1, resolver=resolver, measure=vllm())


def test_resolved_distillation_is_deterministic(tmp_path):
    fixtures = [base.base_fixture(), base.base_fixture()]
    resolver, _ = resolver_for(fixtures)
    receipts = [dd.write_outputs(distil_half(fx, H1, resolver=resolver, measure=vllm()), tmp_path / name) for fx, name in zip(fixtures, "ab")]
    assert receipts[0] == receipts[1]
    for name in sorted(p.name for p in (tmp_path / "a").iterdir()):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


# ---- the measured-bank gate of the plan builder --------------------------------------------------------------------------------------------


def other_half_roster():
    return {"roster_id": cm.HALF_ROSTER_ID[H2], "split_identity_hash": "d" * 64, "items": [{"sample_id": "h2-row", "audio_seconds": 1.0}]}


def test_the_plan_builder_refuses_estimate_measured_and_ids_only_banks(tmp_path):
    fx = base.base_fixture()
    resolver, _ = resolver_for([fx])
    estimate = dd.write_outputs(distil_half(fx, H1, resolver=resolver, measure=None), tmp_path / "estimate")
    ids_only = dd.write_outputs(fx.distil(measure=vllm()), tmp_path / "ids")
    good = dd.write_outputs(distil_half(fx, H1, resolver=resolver, measure=vllm()), tmp_path / "good")
    assert (estimate["measure"], ids_only["measure"], good["measure"]) == ("estimate", "vllm-tokenize", "vllm-tokenize")
    with pytest.raises(sp.SamplerError, match="'estimate', not 'vllm-tokenize'"):
        sp.build_plan(other_half_roster(), H2, 2, bank_receipt=estimate)
    with pytest.raises(sp.SamplerError, match="resolved supplied-item text"):
        sp.build_plan(other_half_roster(), H2, 2, bank_receipt=ids_only)
    plan = sp.build_plan(other_half_roster(), H2, 2, bank_receipt=good)
    assert plan["banks"]["bank_measure"] == "vllm-tokenize" and plan["banks"]["bank_source_halves"] == [H1]
    assert sp.build_plan(other_half_roster(), H2, 1)["banks"]["bank_measure"] is None


# ---- the union bank ------------------------------------------------------------------------------------------------------------------------


def half_banks(tmp_path, measure_factory=vllm):
    h1, h2 = base.Fixture(), second_half_fixture()
    for i in range(4):
        base.date_row(h1, i, filler="monday")
    h1.add("r050", 1.0, [(1.0, True, base.ans(), ())] * 4)             # a row the lock already gets right: in the half, never a source row
    resolver, _ = resolver_for([h1, h2])
    out = {}
    for half, fx in ((H1, h1), (H2, h2)):
        dd.write_outputs(distil_half(fx, half, resolver=resolver, measure=measure_factory()), tmp_path / half)
        out[half] = dd.load_bank(tmp_path / half / "bank-receipt.json")
    return out, (h1, h2), resolver


def test_union_bank_merges_both_halves_deterministically(tmp_path):
    banks, _, _ = half_banks(tmp_path)
    first = dd.union_banks(banks[H1], banks[H2], measure=vllm())
    second = dd.union_banks(banks[H1], banks[H2], measure=vllm())
    assert cm.canonical_json(first) == cm.canonical_json(second)
    receipt = first["receipt_core"]
    assert receipt["kind"] == "union" and receipt["source_halves"] == [H1, H2] and receipt["measure"] == "vllm-tokenize"
    assert receipt["source_split_identities"] == {H1: "a" * 64, H2: "b" * 64}
    assert [s["half"] for s in receipt["sources"]] == [H1, H2]
    assert [s["exemplar_bank_content_sha256"] for s in receipt["sources"]] == [banks[h][0]["exemplar_bank"]["content_sha256"] for h in (H1, H2)]
    entries = first["exemplar_bank"]["by_scenario"]["alarm"]["entries"]
    assert [e["source_row_id"] for e in entries] == sorted(e["source_row_id"] for e in entries)             # equal reward and F1: ranked by row id
    assert {e["source_half"] for e in entries} == {H1, H2}
    assert set(receipt["source_row_ids"]) == {e["source_row_id"] for e in entries} | {r for rule in first["rule_bank"]["rules"] for r in rule["support_rows"]}
    rule = next(r for r in first["rule_bank"]["rules"] if r["candidate_key"] == "slot_missing|alarm|date")
    assert rule["support_count"] == 8 and rule["support_rows"] == sorted(rule["support_rows"]) and len(rule["pair_ids"]) == 8
    assert receipt["exemplar_bank"]["content_sha256"] == cm.content_sha256(first["exemplar_bank"])


def test_union_ranks_by_reward_and_keeps_the_per_scenario_cap(tmp_path):
    h1, h2 = base.Fixture(), base.Fixture()
    for i in range(8):
        base.date_row(h1, i, filler=f"day{i}")
    for i in range(8):
        base.date_row(h2, 100 + i, filler=f"night{i}")
    h2.add("r999", 0.0, [(1.0, True, base.ans("alarm", "set", [("date", "sunday")]), ()), (0.5, True, base.ans(), ()), (0.5, True, base.ans(), ()), (0.0, False, base.ans(), ())])
    resolver, _ = resolver_for([h1, h2])
    banks = {}
    for half, fx in ((H1, h1), (H2, h2)):
        dd.write_outputs(distil_half(fx, half, resolver=resolver, measure=vllm()), tmp_path / half)
        banks[half] = dd.load_bank(tmp_path / half / "bank-receipt.json")
    union = dd.union_banks(banks[H1], banks[H2], measure=vllm())
    entries = union["exemplar_bank"]["by_scenario"]["alarm"]["entries"]
    assert len(entries) == cm.MAX_EXEMPLARS_PER_SCENARIO == 8
    assert entries[0]["source_row_id"] == "r999" and entries[0]["reward"] == 1.0                      # the largest reward leads
    assert union["receipt_core"]["counts"]["exemplar_entries_in"] == 16
    assert union["receipt_core"]["counts"]["dropped"] == {"scenario-count-cap": 8}


def test_union_remeasures_the_merged_block_against_the_token_cap(tmp_path):
    def long_row(fx, name, filler):
        fx.add(name, 0.5, [(1.0, True, base.ans("alarm", "set", [("note", filler * 1400 + name)]), ()), (0.5, True, base.ans(), ()), (0.5, True, base.ans(), ()),
                           (0.25, True, base.ans(), ())])

    h1, h2 = base.Fixture(), base.Fixture()
    for i in range(3):
        long_row(h1, f"a{i}", "p")
        long_row(h2, f"b{i}", "q")
    resolver, _ = resolver_for([h1, h2])
    banks = {}
    for half, fx in ((H1, h1), (H2, h2)):
        dd.write_outputs(distil_half(fx, half, resolver=resolver, measure=vllm()), tmp_path / half)
        banks[half] = dd.load_bank(tmp_path / half / "bank-receipt.json")
        assert len(banks[half][1]["by_scenario"]["alarm"]["entries"]) == 3
    union = dd.union_banks(banks[H1], banks[H2], measure=vllm())
    block = union["exemplar_bank"]["by_scenario"]["alarm"]
    assert 3 < len(block["entries"]) < 6 and block["block_tokens"] <= cm.EXEMPLAR_BLOCK_TOKEN_CAP
    assert union["receipt_core"]["counts"]["dropped"] == {"token-cap": 6 - len(block["entries"])}


def test_union_round_trip_write_once_and_cli(tmp_path, capsys):
    banks, _, _ = half_banks(tmp_path)
    union = dd.union_banks(banks[H1], banks[H2], measure=vllm())
    receipt = dd.write_union_outputs(union, tmp_path / "union")
    assert dd.write_union_outputs(union, tmp_path / "union") == receipt                      # same inputs again: no-op
    loaded, exemplars, rules = dd.load_bank(tmp_path / "union" / "bank-receipt.json")
    assert loaded == receipt and cm.content_sha256(exemplars) == receipt["exemplar_bank"]["content_sha256"] and cm.content_sha256(rules) == receipt["rule_bank"]["content_sha256"]
    (tmp_path / "union" / "rule-bank.json").write_bytes(b"{}\n")
    with pytest.raises(dd.DistillError, match="refusing to overwrite"):
        dd.write_union_outputs(union, tmp_path / "union")
    # the command line (estimate measure only): halves distilled with the estimate measure
    est, _, _ = half_banks(tmp_path / "est", measure_factory=tx.EstimateMeasure)
    argv = ["union", "--h1-receipt", str(tmp_path / "est" / H1 / "bank-receipt.json"), "--h2-receipt", str(tmp_path / "est" / H2 / "bank-receipt.json"),
            "--out-dir", str(tmp_path / "est-union")]
    assert dd.main(argv) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["kind"] == "union" and summary["measure"] == "estimate" and summary["source_halves"] == [H1, H2]
    assert dd.main(argv) == 0
    assert dd.main(argv[:-2] + ["--out-dir", str(tmp_path / "x"), "--measure", "vllm-tokenize"]) == 2        # the command line cannot reach a tokenizer


def test_union_guard_accepts_fold_b_and_refuses_either_half(tmp_path):
    banks, (h1, h2), _ = half_banks(tmp_path)
    receipt = dd.write_union_outputs(dd.union_banks(banks[H1], banks[H2], measure=vllm()), tmp_path / "union")
    fold_b = ["fb-row-1", "fb-row-2"]
    cm.check_union_for_fold_b(receipt, fold_b, half_row_ids={H1: h1.rows, H2: h2.rows}, fold_b_split_identity="e" * 64)
    cm.check_cross_fit(receipt, cm.FOLD_B_CONSUMER, consuming_row_ids=fold_b)
    for half in cm.HALVES:
        with pytest.raises(cm.CrossFitError, match="consuming half"):
            cm.check_cross_fit(receipt, half)
        with pytest.raises(sp.SamplerError, match="cross-fit guard"):                                 # and the plan builder refuses it for a construction half
            sp.build_plan({"roster_id": cm.HALF_ROSTER_ID[half], "split_identity_hash": "f" * 64, "items": [{"sample_id": "z", "audio_seconds": 1.0}]},
                          half, 2, bank_receipt=receipt)
    with pytest.raises(cm.CrossFitError, match="source rows"):                                       # Fold B rows must not be source rows
        cm.check_union_for_fold_b(receipt, [receipt["source_row_ids"][0]])
    with pytest.raises(cm.CrossFitError, match="rows of a construction half"):                       # nor any row of either half
        cm.check_union_for_fold_b(receipt, ["r050"], half_row_ids={H1: h1.rows, H2: h2.rows})
    with pytest.raises(cm.CrossFitError, match="split identity"):
        cm.check_union_for_fold_b(receipt, fold_b, fold_b_split_identity=receipt["source_split_identities"][H1])
    with pytest.raises(cm.CrossFitError, match="not one"):                                           # a single-half bank is not a Fold B bank
        cm.check_union_for_fold_b(banks[H1][0], fold_b)


def test_union_for_fold_b_refuses_an_estimate_measured_union(tmp_path):
    banks, (h1, h2), _ = half_banks(tmp_path, measure_factory=tx.EstimateMeasure)
    receipt = dd.write_union_outputs(dd.union_banks(banks[H1], banks[H2], measure=tx.EstimateMeasure()), tmp_path / "union")
    assert receipt["measure"] == "estimate"
    with pytest.raises(cm.BankNotMeasuredError, match="estimate"):
        cm.check_union_for_fold_b(receipt, ["fb-row"])
    cm.check_union_for_fold_b(receipt, ["fb-row"], require_measured=False)


def test_union_refuses_mismatched_inputs(tmp_path):
    banks, _, resolver = half_banks(tmp_path)
    with pytest.raises(dd.DistillError, match="not the single-half receipt of R01C-H2"):
        dd.union_banks(banks[H1], banks[H1], measure=vllm())                                           # the same half twice
    with pytest.raises(dd.DistillError, match="re-measured with the same one"):
        dd.union_banks(banks[H1], banks[H2], measure=tx.EstimateMeasure())
    other, _, _ = half_banks(tmp_path / "estimate", measure_factory=tx.EstimateMeasure)
    with pytest.raises(dd.DistillError, match="differ in measure"):
        dd.union_banks(banks[H1], other[H2], measure=vllm())
    ids_only = tmp_path / "ids"
    h2 = second_half_fixture()
    dd.write_outputs(dd.distil(roster_of(h2, H2), H2, 1, h2.contacts, h2.scores, measure=vllm()), ids_only)
    with pytest.raises(dd.DistillError, match="differ in exemplar_text"):
        dd.union_banks(banks[H1], dd.load_bank(ids_only / "bank-receipt.json"), measure=vllm())
    iteration_two = tmp_path / "it2"
    dd.write_outputs(dd.distil(roster_of(h2, H2), H2, 2, h2.contacts, h2.scores, resolver=resolver, measure=vllm()), iteration_two)
    with pytest.raises(dd.DistillError, match="different iterations"):
        dd.union_banks(banks[H1], dd.load_bank(iteration_two / "bank-receipt.json"), measure=vllm())


def test_union_refuses_shared_source_rows(tmp_path):
    h1 = base.Fixture()
    for i in range(4):
        base.date_row(h1, i)
    resolver, _ = resolver_for([h1])
    dd.write_outputs(distil_half(h1, H1, resolver=resolver, measure=vllm()), tmp_path / H1)
    dd.write_outputs(distil_half(h1, H2, resolver=resolver, measure=vllm()), tmp_path / H2)           # the same rows distilled as "H2"
    with pytest.raises(dd.DistillError, match="share source rows"):
        dd.union_banks(dd.load_bank(tmp_path / H1 / "bank-receipt.json"), dd.load_bank(tmp_path / H2 / "bank-receipt.json"), measure=vllm())


# ---- static --------------------------------------------------------------------------------------------------------------------------------


def test_new_modules_construct_no_client_and_are_ascii():
    for name in ("tfrl_v2_text.py", "tfrl_v2_loop_adapters.py"):
        text = (NX / name).read_text(encoding="utf-8")
        for needle in ("import socket", "import requests", "import httpx", "import urllib", "import http", "openai", "anthropic", "aiohttp", "subprocess"):
            assert needle not in text, (name, needle)
        assert all(ord(ch) < 128 for ch in text), name
    assert all(ord(ch) < 128 for ch in Path(__file__).read_text(encoding="utf-8"))

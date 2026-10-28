# Root review of the NPU-port SLURP semantic stores (re-authored)

Date: 2026-10-28. Branch `npu-vllm-port`. Design: `docs/superpowers/specs/2026-10-28-npu-vllm-port-design.md`.
Zero model, tokenizer, encoder, scorer, GPU/NPU and tool contacts. No utterance text is reproduced in this note.

Three stores are prepared on the port data root: the two that Fold B
(`slurp-native-stage3-l338-lever-column-fold-b-baseline-v1`) binds and the slot v1 store that its completion pins name.
Nothing is admitted and nothing is scheduled.

## 1. Why the cards were re-authored

The authored source files were never shipped with the study. The historical data root only ever held
`prepared/SAEA-SLURP-SEMANTIC-AUTHORING-PREPARE-V{1,2}/authored-*.json`, and the repository records only their
structure: SHA-256 pins, store measurements, the contrast-card receipt and the 2026-09-13 and 2026-09-17 root
reviews. The card text cannot be recovered. The owner chose to re-author it following the documented procedure.

**Authorship.** The cards were written by root, meaning the agent session operating this port. Root read the
governed TRAIN support directly. Qwen3-Omni Thinking BF16 was not used, and neither was any other model, encoder
or NPU. The authoring script holds every card sentence as literal data, with nothing generated. It is published
beside its outputs (`author_semantic_knowledge_npu_port_v1.py`, SHA-256 `43acda62…`).

## 2. Inputs and rules

- **Support:** `prepared/SAEA-SLURP-SEMANTIC-AUTHORING-PREPARE-V1/support.json` (SHA-256 `8a5cd536…`).
  - It was rebuilt by `2026-10-28-slurp-semantic-authoring-registration-npu-port-v1.json`.
  - The pool and the three construction rosters are byte-identical to the historical pins; only provenance
    paths differ.
  - The support holds 474 intent-support utterances and 803 slot-support utterances.
  - The per-type counts reproduce the 2026-09-17 review exactly: date 171 mentions over 40 intents, time 131,
    media_type 46, transport_type 37, business_type 33, order_type 15, relation 48, event_name 88.
- **Rules carried over unchanged from 2026-09-13:**
  - Cards explain semantics, ambiguity, inclusion and exclusion. They do not copy TRAIN.
  - Co-occurrence is evidence, never a required-slot schema.
  - No DEV, test, anchor-output or scorer content is read.
- **Slot v2 card selection.** The eight revised slot cards are the ones the 2026-09-17 revision named:
  date, time, media_type, transport_type, business_type, order_type, relation and event_name.
  - That selection was made historically from the A3 coverage audit. This port reuses the selection; it does
    not re-read the audit.
  - Every revised sentence is derived from the support pool.
- **Structure matched to the receipts:**
  - Each intent card has exactly one confusable exclusion, naming the neighbour given in
    `docs/receipts/knowledge/slurp-intent-contrast-cards-v1.json`.
  - One shared uncertainty sentence appears on every intent card.
  - A shared limitation appears on every card, plus card-specific limitations on the same 14 intents.
  - Slot bindings equal exactly the observed intent/type pairs in the support.
  - Each slot v2 revision adds one role cue; seven add an emit-when-spoken sentence; time adds a spoken-form cue
    and a format, with 0 digit and 0 colon spans among its 131 mentions.
- **Source support is computed, not typed.**
  - Intent cards cite their own support group plus the neighbour group.
  - Slot cards cite the support rows that carry the type.

## 3. Validation (in memory, then registered preparation)

Each source was compiled with the study builders before preparation:

- the permitted construction roster was used;
- the Fold A, Fold B and DEV-1000 rosters were forbidden;
- the facet/value material was containment-checked against perception_v3, planner_v1, reviewer_v1, solver_v2,
  context_v3 and the rendered `context_v3 + slurp-label-inventory` contract.

There were zero collisions. `prepare-semantic-knowledge` then produced the same content hashes.

| store | values | chars (port / historical) | words (port / historical) | snapshot content hash |
| --- | --- | --- | --- | --- |
| intent-boundaries-npu-port-v1 | 60 | 45,641 / 42,133 | 7,228 / 5,740 | `cb6ef966…` |
| slot-contracts-npu-port-v1 | 54 | 67,081 / 80,049 | 11,255 / 11,647 | `2ac1666b…` |
| slot-contracts-npu-port-v2 | 54 | 70,228 / 83,351 | 11,727 / 12,137 | `e035c1b8…` |

Intent section characters, port against historical:

| section | port | historical |
| --- | --- | --- |
| activation | 4,289 | 4,244 |
| exclusions | 7,933 | 7,301 |
| uncertainty | 13,140 | 11,940 |
| limitations | 13,591 | 12,362 |

Slot v2 section characters, port against historical:

| section | port | historical |
| --- | --- | --- |
| bindings | 18,976 | 24,877 |
| cues | 8,564 | 12,928 |
| formats | 3,347 | 5,352 |
| missing | 8,697 | 8,513 |
| ambiguity | 5,507 | 6,997 |
| limitations | 15,441 | 16,725 |

The slot cards are about 16% shorter. The difference is in phrasing (role, cue and format sentences), not in
coverage: every observed pair is bound. They were not padded toward a byte target.

## 4. Declared deviation

The port stores have the historical structure, rules and support. Their text, however, is new. Every
knowledge-bound Fold B number from the port is therefore measured against re-authored knowledge, not the sealed
cards, and has to be reported that way. Naming also differs:

- The port intent source `authored-intent-boundaries-v1.json` plays the role of the historical
  `authored-intent-boundaries-v2.json`. That is the source of registration V2, built under split identity
  `f76a7d00…` into `intent-boundaries-v1`.
- The port namespaces carry `npu-port` so that they can never be mistaken for the historical stores.

## 5. Pins

| registration | snapshot.json | release.json | completion.json |
| --- | --- | --- | --- |
| `2026-10-28-slurp-intent-boundaries-preparation-registration-npu-port-v1.json` | `08976279…` | `22242e83…` | `7b7e4744…` |
| `2026-10-28-slurp-slot-contracts-preparation-registration-npu-port-v1.json` | `0650b5db…` | `19723a2a…` | `89493781…` |
| `2026-10-28-slurp-slot-contracts-preparation-registration-npu-port-v2.json` | `557080ad…` | `2f11a748…` | `479e9496…` |

The authored sources in `prepared/SAEA-SLURP-SEMANTIC-AUTHORING-NPU-PORT-V1/` have these SHA-256 hashes:

| source | SHA-256 |
| --- | --- |
| intent | `a174189d…` |
| slot v1 | `dd546edf…` |
| slot v2 | `8994f773…` |
| CHANGES.json | `570be21c…` |

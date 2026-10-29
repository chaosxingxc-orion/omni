# English successors for owner quotations inside hash-pinned documents (2026-10-29)

Date: 2026-10-29. Readiness note, docs only; it grants no contact, no authority, and admits nothing.

Policy (study `CLAUDE.md`, first principle 2): non-English text in a mutable document is repaired in place; bytes that a
receipt, registration or admission draft pins by SHA-256 are never rewritten. The three documents below still carry the
owner's original Chinese wording because another committed record pins their exact bytes. This note is their English
successor: the English text here, together with the unchanged English surrounding text in each source, is the working
text. The original wording is never passed to a model.

| Source (sha256 at commit 9db58c1c) | Pinned by | Lines | English working text |
| --- | --- | --- | --- |
| `docs/readiness/2026-10-10-slurp-nx-r6-node-local-then-cascade-design-v1.md` (`8d30e09570500bf144d43e254fff812dd80d9d81f53f704a93b906c24b8147bd`) | `docs/readiness/2026-10-10-slurp-nx-r6-perception-fold-b-v1/root-admission-draft-v1.json` | 10-13, 19 | Decision 245, 2026-10-10: "Optimise each node on its own first, then optimise the cascade. During the node-local phase, ignore the connections between nodes. During the cascade phase, the upstream node must work out how to pass more effective information to the downstream node; what matters most there is optimising the communication template: define what the elements most useful to the downstream judgement look like. The upstream may also return several options together with their confidence for the downstream node to choose from, or the downstream node treats the task as a re-ranking task. Note: the most important thing is to provide stable and additional information gain." Line 19: the owner wrote "execution degree", read as "confidence". |
| `docs/results/2026-09-17-slurp-node-context-audit-v1.md` (`627d2b7a7766de36ba552fc69b5a246db492d2b1506f81185143c3122065ba57`) | `docs/receipts/agent-base-lock-v2-draft.json` | 22 | Owner, 2026-09-17: "I want a unified handling policy for the context in front of every agent node: do not hand the model's raw output to the next node unprocessed. For a node's context you may provide templates, and where necessary we can add one more model call to make the model's context cleaner, so that the model performs better on the different tasks." |
| `docs/results/2026-09-30-slurp-v2-1-lock-v1.md` (`e346b29fe6bc75a6786fea6ce1dff63f5133533e043bac75cf966cef45044264`) | `docs/readiness/2026-09-30-slurp-stage3-module-comp1b-dev2000-added-v1/root-admission-draft-v1.json` | 8 | Owner, 2026-09-29: "Decision 204: agreed." |

Other sealed carriers of Chinese wording and their existing English successors:

- `docs/exposure-ledger.md` historical rows 892-985 (2026-08-11/12): `docs/handoff/historical-exposure-english-successor-2026-10-09.json`
  (full-line successors keyed by line SHA-256) and `docs/handoff/2026-10-03-historical-ledger-english-quote-successors.md`. The two
  `docs/readiness/2026-09-12-slurp-whisper-*/exposure-ledger.md` snapshots are byte copies of that ledger pinned by their registrations.
- `docs/receipts/owner/*.json`: each owner-words field is paired with an `english_translation` field in the same receipt; the receipts are
  pinned by SHA-256 in configs and tests and stay byte-identical.
- `docs/checks/**` and `reference/w1-snapshot/**`: sealed by `MANIFEST.sha256` / `SHA256SUMS`; the Chinese there is model output
  (ASR hypotheses), a prompt-card row, or a historical code comment, not working text.

Everything else in the mutable tree was translated in place on 2026-10-29 (commit following this note). Code and tests keep
CJK punctuation sets and non-Latin fixtures because they are functional inputs, not prose.

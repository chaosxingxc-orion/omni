# NPU-port lock entries v1 (owner decision 9.4, 2026-10-28)

Owner decision 9.4 of the NPU/vLLM port (`docs/superpowers/specs/2026-10-28-npu-vllm-port-design.md`) was approved
on 2026-10-28. The owner replied "go ahead" (translated from the owner's Chinese message) and asked the agent to draft
the entries. This record registers four assets in the umbrella `docs/datasets.lock.json`: the BF16 Thinking model and
the three vLLM serving views of the retrieval encoders. Registration only: no download, no model load, no forward, no
corpus contact.

The umbrella lock is not under version control on the Ascend hosts. The dev umbrella root is a plain copy of a
pristine umbrella clone. This directory is therefore the reviewable, reproducible form of the edit:

| file | content |
|---|---|
| `entries.json` | the exact new entries: the profile, the four `models` entries and the `amendments` record (sha256 `c3478c92b14ac753e39054aad83437a0c1d30cfdbb1d3115ef411111f006890c`) |
| `apply_amendment.py` | applies `entries.json` to the pristine lock; refuses any other input, any changed `entries.json`, any name or profile that already exists, and any result other than the reviewed bytes |

Neither file names a host path. Lock entries are relative to the data root (`SPEECHRL_DATA_DIR`), or to the port
fragment's `model_materialization_root` for the BF16 model. The script takes the lock path as an argument.

## Lock identity

| | sha256 | bytes |
|---|---|---|
| before (pristine; also kept as `datasets.lock.json.pre-npu-port-v1` beside the live lock) | `eb39a1caa93605dabc25e8eb692d0d306b8a4aa572157449736cd921d286966d` | 223,435 |
| after (the live dev-umbrella lock) | `61f3d111b7bed7dfc191358f2ce557efd824795cc3cdbb28f32761d8b2ee20df` | 268,539 |

Reproduce. `PRISTINE` is a pristine umbrella clone and `UMBRELLA` the amended umbrella root. The first command writes
a new file and refuses to overwrite an existing one:

```bash
python3 -I docs/readiness/2026-10-28-npu-port-lock-entries-v1/apply_amendment.py \
  --lock "$PRISTINE/docs/datasets.lock.json" --output /tmp/datasets.lock.amended.json
python3 -I docs/readiness/2026-10-28-npu-port-lock-entries-v1/apply_amendment.py \
  --lock "$UMBRELLA/docs/datasets.lock.json" --check-amended
```

The edit is add-only. It does three things:

- appends the four entries to `models`;
- inserts the amendment first in `amendments`, which is ordered newest first;
- adds the profile `speech-aware-npu-port` to `profiles`.

No existing entry, profile, amendment or total changed. Removing the three additions from the amended lock gives the
pristine document exactly. The lock round-trips byte-exactly through `json.dumps(indent=2, ensure_ascii=False)`
plus a trailing newline, and the script relies on this.

## Entries

| name | local_subdir | source | files | bytes | revision / identity |
|---|---|---|---|---|---|
| `qwen3-omni-30b-a3b-thinking-bf16` | `models/qwen3-omni-30b-a3b-thinking` | hf `Qwen/Qwen3-Omni-30B-A3B-Thinking` | 26 | 63,447,489,451 (59.1G) | `2f443cfc4c54b14a815c0e2bb9a9d6cbcd9a748b` |
| `omni-embed-nemotron-3b-causal-v1` | `models/serving-views/omni-embed-nemotron-3b-causal-v1` | derived (`core/vllm_serving_view.py`) from `omni-embed-nemotron-3b` | 14 | 9,423,118,096 (8.8G) | receipt `a3ddac0294a4d346dcdf5fed39226d6ce2508f3a28a8e70afbbc773d80b6cd81` |
| `lco-embedding-omni-3b-2605-v1` | `models/serving-views/lco-embedding-omni-3b-2605-v1` | derived from `LCO-Embedding/LCO-Embedding-Omni-3B-2605` @ `5f6b5329` | 14 | 9,423,121,465 (8.8G) | receipt `4c95f853cdb7e9279f267c9d11f53e819484905dc0c9d195605e0effe8e33f91` |
| `lco-embedding-omni-7b-v1` | `models/serving-views/lco-embedding-omni-7b-v1` | derived from `LCO-Embedding/LCO-Embedding-Omni-7B` @ `108f6f1a` | 16 | 17,879,801,172 (16.7G) | receipt `c77fd25b103c189db61624010db4313c93140c6bffbec587ae333c72e1ce9135` |

All four entries follow the conventions of the existing post-baseline model entries (`whisper-large-v3`,
`qwen3-embedding-0.6b-gguf`):

- `source.files` lists every governed file with `size_bytes` and `sha256`;
- integer `files` and `size_bytes` are the totals the gate's model-entry lookup requires;
- `lifecycle`, `profiles`, `license`, `verification`, `note`, `research_role`, `claim_limit` and
  `acquisition_receipt` (this file) are recorded.

The BF16 entry is `AUXILIARY` with an `hf` source and a 40-hex revision, as the umbrella validator requires. The views
are `DERIVED` with a `derived` source naming the builder and its arguments. Each view also records:

- `revision`: the receipt hash, which is the identity the pooling client pins;
- `derived_from`: the source checkpoint and every source file it consumed;
- `dependencies: ["omni-embed-nemotron-3b"]`, for the Nemotron view only.

The views are also members of `speech-aware-tools`.

The BF16 entry governs all 26 published files, so it has byte closure with the revision. vLLM reads 24 of them; README.md
and .gitattributes are governed only for closure. The downloader bookkeeping under `.hfd/` is not governed.

## How the fingerprints were measured (2026-10-28)

1. **Hashing.** Every file of the four assets, and of the three view sources, was hashed with `sha256sum` on host
   `npu`, pinned to cores 64-127 (`taskset -c 64-127`, 16 parallel readers). The per-file lists are in the data root's
   `work/scratch/lock/` (`*.sha256`, `*.sizes.tsv`); the entries carry the same values.
2. **BF16 check against the official API.** The official huggingface.co model API was read at the pinned revision. The
   revision is ungated and public, with 26 siblings.
   - The local file set and sizes equal the sibling inventory.
   - The 16 safetensors shards match their LFS sha256 oids.
   - The 10 non-LFS files match their Git blob SHA-1.
   - This agrees with the download-time verification of 2026-10-27 (`VERIFY ... files=26 bad=0`).
3. **Views against their receipts.** Every file of each view equals the `view_sha256` and `view_bytes` in its
   `serving_view.json`. Each directory holds exactly the receipt's files plus the receipt.
4. **View sources.** The source files each view consumed, re-hashed on disk, equal the receipt's `source_sha256` and
   `source_bytes`.
   - For the two LCO views, these files also match the official API at the pinned revisions: LFS oids for the
     shards, Git blob SHA-1 for the rest.
   - The Nemotron source is the ModelScope master copy. It is content-fingerprinted only, so no upstream per-file
     hash exists to compare against.
5. **Umbrella inventory.** The umbrella's own `scripts/data/asset_lock.py` (umbrella commit `70b4bb5`) checked the
   amended lock:
   - `validate`: OK, 133 assets.
   - `inventory --full --fail-on-drift` for the four entries, run on host npu (cores 64-127): COMPLETE=4, drift=0.

The entries were assembled by a scratch builder (`work/scratch/lock/build_entries.py` in the data root). It asserts
every check above before writing anything.

## Consequences and limits

- **Registrations that pin the old lock.** Two 2026-10-28 registrations pin the pre-amendment lock bytes (`eb39a1ca`):
  - `2026-10-28-slurp-tfrl-fold-b-speech-query-encoding-registration-npu-port-v1.json`;
  - `2026-10-28-slurp-semantic-authoring-registration-npu-port-v1.json`.

  Both runs are sealed, and a preparation attempt is never replayed. A fresh attempt under either registration would
  now refuse the live lock, because the bytes differ. The pinned bytes are preserved as `datasets.lock.json.pre-npu-port-v1`.
- **No serving identity yet.** The entries admit bytes only. The served identity (the vLLM runtime receipt) is owner
  decision 9.3, which is still open. Every corpus contact stays behind its own exposure-ledger row.
- **Rebuilding a view.** Each view receipt (`serving_view.json`, part of the asset bytes) records its source as an
  absolute `source_dir`. A rebuild reproduces the receipt hash only from a source at that same absolute path. Moving
  the data root does not change any registered byte, but a view rebuilt after the move is a new view with a new
  receipt hash.

#!/usr/bin/env bash
# Rebuild the SAEA serving core's exact Q4_K_M GGUF from its documented source.
#
# docs/receipts/runtime-q4km.json pins the core by sha256 and core/session/receipt.py's
# verify_against_runtime_receipt() refuses any session whose model files differ, so the
# reproduction cannot use ggml-org's published Q4_K_M (d9e28765...). The lock records the real
# file as a local requant: llama-quantize --allow-requantize Q4_K_M over the Q8_0, at llama.cpp
# fdbd6abee20e408de21e90ca77a24cd50a6ea073. Both inputs are verified here before use, and the
# product is accepted only if it hashes to the pinned 0751c279... -- a near miss is still a miss.
#
# Resumable and idempotent: an already-correct output short-circuits, a wrong one is removed.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
export DL_LOG=${DL_LOG:-$(dirname "$HERE")/dl_logs/q4km-requant.log}
source "$HERE/_daemonize.sh"

DR=${SPEECHRL_DATA_DIR:-/mnt/cephfs/xingchao/omni}
SRC_DIR=$DR/models/qwen3-omni-30b-a3b-instruct-gguf
OUT_DIR=$DR/models/qwen3-omni-30b-a3b-instruct-gguf-q4km
QUANTIZE=${LLAMA_QUANTIZE:-$DR/src/llama.cpp/build/bin/llama-quantize}

SRC=$SRC_DIR/Qwen3-Omni-30B-A3B-Instruct-Q8_0.gguf
MMPROJ=$SRC_DIR/mmproj-Qwen3-Omni-30B-A3B-Instruct-bf16.gguf
OUT=$OUT_DIR/Qwen3-Omni-30B-A3B-Instruct-Q4_K_M.gguf
OUT_MMPROJ=$OUT_DIR/mmproj-Qwen3-Omni-30B-A3B-Instruct-bf16.gguf

SRC_SHA=8a50e5a7d29ae6a28fea9ca45e3bb0a142e76ec07e6787a7703cd498eb08ffaa
MMPROJ_SHA=f0dfe825fb692d426362b1ac79678fc08daa4758f7151526cad110515f122883
WANT_SHA=0751c279498785c0b07130ae7748038d1e2cfd04617928e4557063807f98066d
WANT_BYTES=18557053952

check() { # path expected_sha label
  [ -f "$1" ] || { echo "missing $3: $1" >&2; return 1; }
  local got; got=$(sha256sum "$1" | cut -d' ' -f1)
  [ "$got" = "$2" ] && { echo "ok   $3 sha256=$got"; return 0; }
  echo "BAD  $3 sha256=$got want=$2" >&2; return 1
}

mkdir -p "$OUT_DIR"

if [ -f "$OUT" ] && check "$OUT" "$WANT_SHA" "Q4_K_M (existing)" 2>/dev/null; then
  echo "requant already complete"
else
  check "$SRC" "$SRC_SHA" "Q8_0 source" || { echo "requant source not ready" >&2; exit 1; }
  [ -x "$QUANTIZE" ] || { echo "llama-quantize not built at $QUANTIZE" >&2; exit 1; }
  echo "build: $("$QUANTIZE" --version 2>&1 | head -2 | tr '\n' ' ')"
  rm -f "$OUT"
  # --allow-requantize: the input is already Q8_0, which llama-quantize otherwise refuses.
  "$QUANTIZE" --allow-requantize "$SRC" "$OUT" Q4_K_M || { echo "llama-quantize failed" >&2; exit 1; }
  size=$(stat -c %s "$OUT")
  echo "produced $OUT bytes=$size (want $WANT_BYTES)"
  if ! check "$OUT" "$WANT_SHA" "Q4_K_M (requant)"; then
    echo "REQUANT_MISMATCH: this build does not reproduce the pinned core byte-for-byte." >&2
    echo "The tensor data may still be identical (the lock records a tensor-for-tensor match" >&2
    echo "against ggml-org's published Q4_K_M); differences here would be GGUF metadata." >&2
    echo "Compare with: $QUANTIZE --dry-run, and do NOT claim an exact-identity reproduction." >&2
    exit 2
  fi
fi

# The q4km asset carries its own copy of the same bf16 mmproj, as the lock records.
if [ ! -f "$OUT_MMPROJ" ] || ! check "$OUT_MMPROJ" "$MMPROJ_SHA" "mmproj (copy)" 2>/dev/null; then
  check "$MMPROJ" "$MMPROJ_SHA" "mmproj source" || exit 1
  cp -f "$MMPROJ" "$OUT_MMPROJ.incoming" && mv -f "$OUT_MMPROJ.incoming" "$OUT_MMPROJ"
  check "$OUT_MMPROJ" "$MMPROJ_SHA" "mmproj (copy)" || exit 1
fi
echo "Q4KM_REQUANT_DONE"

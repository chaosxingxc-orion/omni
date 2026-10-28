#!/usr/bin/env bash
# SLURP, laid out exactly like exploring-l4-intelligence fetch-assets.sh fetch_slurp() with
# SPEECHRL_DATA_DIR=$DR: pinned git checkout at $DR/repos/slurp, Zenodo 4274930 tarballs fetched
# (aria2c -x16 -s16 -j4 pattern, here via mt_download.py) and extracted into $DR/datasets/slurp,
# plus the governed annotation copy at datasets/slurp/metadata recorded in datasets.lock.json.
# Resumable (.part segments, md5 + extraction markers). Runs detached by default (_daemonize.sh).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/_daemonize.sh"
DR=${SPEECHRL_DATA_DIR:-/mnt/cephfs/xingchao/omni}
REV=8eb16545762be97ace75334109d73824217311f1
REPO="$DR/repos/slurp" AUDIO="$DR/datasets/slurp"

[ -d "$REPO/.git" ] || git clone https://github.com/pswietojanski/slurp.git "$REPO"
git -C "$REPO" checkout -q "$REV"
echo "repos/slurp at $(git -C "$REPO" rev-parse HEAD)"

mkdir -p "$AUDIO" "$DR/manifests"
MAN="$DR/manifests/slurp.links.txt"
for f in slurp_real.tar.gz slurp_synth.tar.gz; do
  printf 'https://zenodo.org/api/records/4274930/files/%s/content\n  out=%s\n  split=16\n' "$f" "$f"
done > "$MAN"
python3 -I "$HERE/mt_download.py" -i "$MAN" -d "$AUDIO" -j 4 -s 16

declare -A MD5=([slurp_real.tar.gz]=9efc0f058ced47bf5131c7cb2cade513 [slurp_synth.tar.gz]=6186b4d2475aa8020afe746096e30b4f)
for f in slurp_real.tar.gz slurp_synth.tar.gz; do
  [ -f "$AUDIO/$f.md5ok" ] || {
    got=$(md5sum "$AUDIO/$f" | cut -d' ' -f1)
    [ "$got" = "${MD5[$f]}" ] || { echo "MD5 MISMATCH $f: $got (deleting for re-download)"; rm -f "$AUDIO/$f"; exit 1; }
    touch "$AUDIO/$f.md5ok"; echo "md5 ok $f"; }
  [ -f "$AUDIO/$f.extracted" ] || { tar -xzf "$AUDIO/$f" -C "$AUDIO" && touch "$AUDIO/$f.extracted"; }
done
echo "slurp_real: $(find "$AUDIO/slurp_real" -name '*.flac' | wc -l) flac (lock: 72395)"
echo "slurp_synth: $(find "$AUDIO/slurp_synth" -name '*.flac' | wc -l) flac (lock: 69257)"

# governed annotation copy (datasets.lock.json slurp.annotations), sha256-checked against the lock
mkdir -p "$AUDIO/metadata"
for f in train.jsonl devel.jsonl test.jsonl train_synthetic.jsonl metadata.json; do
  cp "$REPO/dataset/slurp/$f" "$AUDIO/metadata/$f"
done
cp "$REPO/dataset/lm/lm.txt" "$AUDIO/metadata/lm.txt"
cp "$REPO/LICENSE.txt" "$AUDIO/metadata/LICENSE.txt"
(cd "$AUDIO/metadata" && sha256sum -c - <<'EOF'
4a05a3de6c7f028aff298c4a90196e6a3fecd1938d96fb1c125d6fd10d5e301e  train.jsonl
00d346b183ecc2d160853cd3dc20e8ccd207d9e1809cf6e2c05182076bd0ecc9  devel.jsonl
fe3449af69b42fda7163345482556066c35a80f7e552a6d67e212a0e2f0783cc  test.jsonl
8394ac2573e1d72b3b736ceff8b342e7a6569243db84295aa568766aef0035db  train_synthetic.jsonl
be7943d576c0b79b34ae32796e5c9474bd3d4d123bb35e38f73497f4a429fb83  metadata.json
295a336c38481b954e23ba42c81b944d48119af7d013eccff456338dd619d754  lm.txt
97547dd6236684935697b26629fb6e70b3be4ed535c2d85fdc053a6389192849  LICENSE.txt
EOF
)
echo SLURP_DONE

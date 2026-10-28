#!/usr/bin/env bash
# Show download progress for every asset under SPEECHRL_DATA_DIR.
DR=${SPEECHRL_DATA_DIR:-/mnt/cephfs/xingchao/omni}
LOGS=/home/xingchao/workspace/omni/dl_logs
echo "=== queue"
tail -n 4 "$LOGS/queue.log" 2>/dev/null
echo "=== per-asset (last lines)"
for f in "$LOGS"/*.log; do
  case $(basename "$f") in queue.log) continue;; esac
  echo "-- $(basename "$f")"
  grep -E 'progress|FAILED|SUMMARY|VERIFY|complete|DONE|MISMATCH|missing blobs|md5 ok' "$f" | tail -n 2
done
echo "=== sizes (target in parentheses)"
printf '%-42s %8s  %s\n' "models/qwen3-omni-30b-a3b-instruct" \
  "$(du -sh "$DR/models/qwen3-omni-30b-a3b-instruct" 2>/dev/null | cut -f1)" \
  "(70.5G)$([ -f "$DR/models/qwen3-omni-30b-a3b-instruct/.hfd/speechrl-complete-revision" ] && echo ' VERIFIED')"
printf '%-42s %8s  %s\n' "datasets/slurp" "$(du -sh "$DR/datasets/slurp" 2>/dev/null | cut -f1)" "(6.8G tar + 13G extracted)"
printf '%-42s %8s  %s\n' "datasets/earnings21-22" "$(du -sh "$DR/datasets/earnings21-22" 2>/dev/null | cut -f1)" "(~2.9G)"
printf '%-42s %8s  %s\n' "datasets/audio2tool" "$(du -sh "$DR/datasets/audio2tool" 2>/dev/null | cut -f1)" "(10.4G / 71441 files)"
printf '%-42s %8s\n' "repos/slurp (git)" "$(du -sh "$DR/repos/slurp" 2>/dev/null | cut -f1)"
echo "=== running"
pgrep -af 'run_queue|fetch_hf|fetch_slurp|fetch_earnings|clone_saea|mt_download|hf_complete|verify_hf|git-lfs' \
  | grep -v pgrep | awk '{print $1, $NF}' | sed 's#/home/xingchao/workspace/omni/##' | head

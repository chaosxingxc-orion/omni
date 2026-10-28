#!/usr/bin/env bash
# READ-ONLY watch: report whether host 181's NPU cards are free. Starts and stops nothing.
# usage: bash watch_181_cards.sh   -> prints one status line, exit 0 when every card is idle.
set -uo pipefail
read -r -d '' PROBE <<'EOF'
busy=$(npu-smi info 2>/dev/null | awk '/Process id/{f=1;next} f&&/^\| [0-7] /{print $0}' | wc -l)
hbm=$(npu-smi info 2>/dev/null | grep -oE '[0-9]+/ *65536' | awk -F/ '{s+=$1} END{print s+0}')
names=$(docker ps --format '{{.Names}}' | paste -sd, -)
echo "busy_card_processes=$busy hbm_used_mb_total=$hbm containers=${names:-none}"
EOF
line=$(ssh -o BatchMode=yes npu2 "$PROBE" 2>&1 | tail -1)
echo "[$(date +%FT%T)] 181 $line"
case "$line" in
  *"busy_card_processes=0 "*) exit 0 ;;
  *) exit 1 ;;
esac

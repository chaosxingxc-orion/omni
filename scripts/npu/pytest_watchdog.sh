#!/usr/bin/env bash
# Kill pytest-xdist workers in container $1 whose CPU time has not advanced for 180 s; stop when $2 shows PYTEST_EXIT.
C=$1 LOG=$2
declare -A last since
while ! grep -q PYTEST_EXIT= "$LOG"; do
  now=$(date +%s)
  while read -r pid cpu; do
    if [ "${last[$pid]:-}" = "$cpu" ]; then
      if [ $((now - ${since[$pid]})) -ge 180 ]; then
        echo "$(date +%T) killing stalled worker $pid (cpu $cpu)"; docker exec "$C" kill -TERM "$pid"; unset "last[$pid]"
      fi
    else
      last[$pid]=$cpu; since[$pid]=$now
    fi
  done < <(docker exec "$C" sh -c 'for p in $(pgrep -f "exec\(eval"); do n=$(ls /proc/$p/task | wc -l); [ $n -gt 3 ] && echo $p $(awk "{print \$14+\$15}" /proc/$p/stat); done' 2>/dev/null)
  sleep 20
done
echo "$(date +%T) suite finished"

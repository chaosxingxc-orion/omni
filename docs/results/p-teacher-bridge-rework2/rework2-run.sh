#!/usr/bin/env bash
set -euo pipefail
base=/home/xingchao/workspace/omni
rel=repro/codex-subprocess/p-teacher-candidate-bridge
study=repro/worktrees/p-teacher-candidate-bridge/studies/speech-aware-evidence-acquisition
remote=/var/lib/omni-local-state/private-analysis/p-teacher-rework2-$(date -u +%Y%m%dT%H%M%S)-$$
name=omni-p-teacher-rework2-$(date -u +%Y%m%d%H%M%S)-$$
ssh npu2 "sudo -n mkdir -p '$remote/stage' '$remote/output' && sudo -n chmod 700 '$remote' && sudo -n chmod 777 '$remote/output'"
tar -C "$base" -cf - "$study/src" "$study/scripts" "$study/configs" "$study/tests/unit/test_p_teacher_parent_bridge.py" \
  repro/codex-subprocess/p-retrieval-supply-analysis/inputs/parent.baseline.json \
  repro/codex-subprocess/p-stage-evidence/private/teacher-review-v2/gepa-P-consumption/prepared \
  repro/codex-subprocess/p-stage-evidence/private/rework3-teacher-review-v3/rework3-reflexion-P-consumption/rework3-prepared \
  repro/codex-subprocess/p-stage-evidence/private/rework3-teacher-review-v3/rework3-ace-P-consumption/rework3-prepared \
  repro/codex-subprocess/p-stage-evidence/private/rework3-teacher-review-v3/rework3-source.P-only.json \
  repro/p-stage-gepa-offline-review-v1/private/context.snapshot.json \
  repro/p-stage-gepa-offline-review-v1/private/tokenizer-identity.snapshot.json \
  "$rel/rework2-gepa3-config.json" "$rel/rework2-reflexion-c1-config.json" "$rel/rework2-reflexion-c2-config.json" |
  ssh npu2 "sudo -n tar -xf - -C '$remote/stage' && sudo -n chmod -R a+rX '$remote/stage'"
ssh npu2 "sudo -n docker run -d --name '$name' --network none --read-only --cpuset-cpus 76-77 --memory 8g --pids-limit 128 --cap-drop ALL --security-opt no-new-privileges --tmpfs /tmp:rw,nosuid,nodev,size=2g -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=1 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -v '$remote/stage/repro:/home/xingchao/workspace/omni/repro:ro' -v '$remote/output:/result' -v /mnt/cephfs/xingchao/omni/models/qwen3-omni-30b-a3b-thinking:/actor-tokenizer:ro --entrypoint bash sha256:7e60208fb45e8af3f1ab777cf24df7a7a415d8d84e2972ee76def4daf5c0f026 -c 'p=/home/xingchao/workspace/omni/repro; s=\$p/worktrees/p-teacher-candidate-bridge/studies/speech-aware-evidence-acquisition/scripts/analysis/p_teacher_bridge.py; for c in gepa3 reflexion-c1 reflexion-c2; do /opt/saea-venv/bin/python -B \"\$s\" \"\$p/codex-subprocess/p-teacher-candidate-bridge/rework2-\$c-config.json\" || exit; done; export BRIDGE_REAL_INPUT=\$p/codex-subprocess/p-retrieval-supply-analysis/inputs/parent.baseline.json BRIDGE_REAL_OUTPUT=/result BRIDGE_CONFIG=\$p/codex-subprocess/p-teacher-candidate-bridge/rework2-gepa3-config.json; /opt/saea-venv/bin/python -B -m pytest -q \$p/worktrees/p-teacher-candidate-bridge/studies/speech-aware-evidence-acquisition/tests/unit/test_p_teacher_parent_bridge.py'"
status=$(ssh npu2 "sudo -n docker wait '$name'")
ssh npu2 "sudo -n docker logs '$name'"
if [[ $status != 0 ]]; then echo "container failed: $status; scratch: $remote" >&2; exit 1; fi
out="$base/$rel/rework2-output-$(date -u +%Y%m%dT%H%M%S)-$$"
mkdir -m 700 "$out"
ssh npu2 "sudo -n tar -C '$remote/output' -cf - gepa3 reflexion-c1 reflexion-c2" | tar -xmf - -C "$out"
ssh npu2 "sudo -n chmod -R go-rwx '$remote/stage' '$remote/output'"
printf '输出：%s\n远端隔离副本：%s\n' "$out" "$remote"

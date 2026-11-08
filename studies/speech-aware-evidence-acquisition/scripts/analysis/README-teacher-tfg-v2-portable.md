# Teacher TFG v2 CPU portable entry

This entry runs the existing `teacher_tfg_v2_cpu.py` operator. It requires Docker, Python 3 on the host, the pinned SAEA image, and the original compatible input formats and dimensions. The example is a template, not an executable dataset. Paths are resolved relative to the config file when not absolute. Replace every path and SHA-256 with the actual input file and its digest; `sha256sum FILE` computes a digest. The gold path and digest must be configured before rank, but gold itself need not be present until score.

From the study directory:

```bash
cp configs/analysis/teacher-tfg-v2-portable.example.json /path/to/my-config.json
# Edit all six input paths and hashes in /path/to/my-config.json.
mkdir -p /path/to/output
python3 scripts/analysis/run_teacher_tfg_v2_portable.py rank \
  --config /path/to/my-config.json --output /path/to/output --name my-tfg-rank-001
# After rank exits successfully and rank-freeze.json exists:
python3 scripts/analysis/run_teacher_tfg_v2_portable.py score \
  --config /path/to/my-config.json --output /path/to/output --name my-tfg-score-001
```

Use the same config and output for both phases. The operator checks the canonical container config hash and frozen file hashes before scoring; do not edit the config between phases. Choose a unique container name each time. The command runs in the foreground, returns Docker's status, and removes the container on exit. Schedule it independently of an SSH session for a long run. It does not overwrite an existing rank freeze or score; use a fresh output directory for a separate run. Output must not contain any configured input, including gold. Inputs are individual read-only file mounts; rank does not mount gold. Do not put unrelated private data in the output directory.

Defaults: fixed image `sha256:7e60208fb45e8af3f1ab777cf24df7a7a415d8d84e2972ee76def4daf5c0f026`, 4 CPUs, 4 threads, 4g memory, caller UID:GID. Override with `--image`, `--cpus`, `--threads`, `--memory`, and `--user UID:GID`. Ensure the chosen user can read inputs and write output. Docker access and image availability are host prerequisites. The rank/score operator is specific to the original 8554-member, 1080-query, 60-label data contract; it does not make the example paths or data public.

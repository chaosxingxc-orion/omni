#!/usr/bin/env bash
# Run only on npu2; context.py must have prepared an immutable Git-object context.
set -euo pipefail
release_root=${1:?absolute owned build directory}
release_tag=${2:?unique release tag}
[[ "$release_root" == /var/tmp/omni-unified-release-* ]] || exit 2
[[ "$release_tag" =~ ^saea-unified-release:[a-zA-Z0-9_.-]+$ ]] || exit 2
mkdir "$release_root/launch-custody"
parent=sha256:7e60208fb45e8af3f1ab777cf24df7a7a415d8d84e2972ee76def4daf5c0f026
docker image inspect "$parent" >/dev/null
# Unique name and O_EXCL directory preserve an unknown launch rather than retrying.
docker run -d --name "$(basename "$release_root")" --network none --cpuset-cpus 240-243 --memory 1g --blkio-weight 100 \
 --mount type=bind,src=/usr/bin/docker,dst=/usr/bin/docker,readonly \
 --mount type=bind,src=/var/run/docker.sock,dst=/var/run/docker.sock \
 --mount "type=bind,src=$release_root,dst=/build" \
 --entrypoint /bin/bash "$parent" -c \
 'docker build --network none --cpuset-cpus 240-243 --memory 8g --pull=false --iidfile /build/image-id.txt -t "$1" /build/context > /build/build.log 2>&1; code=$?; printf "%s\n" "$code" > /build/exit-code.txt; exit "$code"' build "$release_tag" | tee "$release_root/launch-custody/container-id.txt"

#!/usr/bin/env bash
# Run on an authorised build host; context.py prepares the Git-object context.
set -euo pipefail
release_root=${1:?absolute owned build directory}
release_tag=${2:?unique release tag}
shared_root=${OMNI_SHARED_ROOT:?set the owned shared project root}
release_root=$(realpath -e -- "$release_root")
shared_root=$(realpath -e -- "$shared_root")
[[ "$shared_root" != / && "$release_root" == "$shared_root/"* ]] || exit 2
[[ "$release_tag" =~ ^saea-unified-release:[a-zA-Z0-9_.-]+$ ]] || exit 2
build_cpus=${OMNI_BUILD_CPUS:?set allocated CPU list}
build_memory=${OMNI_BUILD_MEMORY:-8g}
[[ "$build_cpus" =~ ^[0-9]+([,-][0-9]+)*$ ]] || exit 2
[[ "$build_memory" =~ ^[1-9][0-9]*[mMgG]$ ]] || exit 2
mkdir "$release_root/launch-custody"
parent=sha256:7e60208fb45e8af3f1ab777cf24df7a7a415d8d84e2972ee76def4daf5c0f026
docker image inspect "$parent" >/dev/null
# Unique name and O_EXCL directory preserve an unknown launch rather than retrying.
docker run -d --name "$(basename "$release_root")" --hostname saea-release-builder --network none --cpuset-cpus "$build_cpus" --memory 1g --blkio-weight 100 \
 --mount type=bind,src=/usr/bin/docker,dst=/usr/bin/docker,readonly \
 --mount type=bind,src=/var/run/docker.sock,dst=/var/run/docker.sock \
 --mount "type=bind,src=$release_root,dst=$release_root" \
 --entrypoint /bin/bash "$parent" -c \
 'docker build --network none --cpuset-cpus "$2" --memory "$3" --pull=false --iidfile "$4/image-id.txt" -t "$1" "$4/context" > "$4/build.log" 2>&1; code=$?; printf "%s\n" "$code" > "$4/exit-code.txt"; exit "$code"' build "$release_tag" "$build_cpus" "$build_memory" "$release_root" | tee "$release_root/launch-custody/container-id.txt"

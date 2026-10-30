"""Tests of the generalised port encoder launcher (port_encoder_prep.sh) and its config writer (port_write_runtime_config.py).

Hermetic: a throwaway study tree, data root and stub `docker`; no container, no encoder, no card, no real data. Run in the saea-base container:
  bash port_py.sh -B -m pytest -q -p no:cacheprovider /mnt/cephfs/xingchao/omni/work/scripts/test_port_encoder_prep.py
"""
import hashlib
import importlib.util
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SH = HERE / "port_encoder_prep.sh"
PY = HERE / "port_write_runtime_config.py"
FOLD_B_REG = "docs/readiness/2026-10-28-slurp-tfrl-fold-b-speech-query-encoding-registration-npu-port-v1.json"
FOLD_A_REG = "docs/readiness/2026-10-30-slurp-tfrl-fold-a-speech-query-encoding-registration-npu-port-v1.json"
RUN_B = "SAEA-SLURP-TFRL-FOLD-B-SPEECH-QUERY-ENCODING-NPU-PORT-V1"
RUN_A = "SAEA-SLURP-TFRL-FOLD-A-SPEECH-QUERY-ENCODING-NPU-PORT-V1"
CJK = re.compile("[\u4e00-\u9fff]")
LOCK = {"path": "/lock/datasets.lock.json", "sha256": "ab" * 32}

spec = importlib.util.spec_from_file_location("port_write_runtime_config", PY)
cfg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfg)


def sh(command, cwd=None, env=None, check=True):
    return subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True, check=check)


@pytest.fixture()
def world(tmp_path):
    """tmp OMNI root + tmp study tree (a clean git repo) with a Fold A and a Fold B registration, protocols and the pinned CPU encoder inventory (cores 128-191)."""
    data = tmp_path / "omni"
    (data / "coordination").mkdir(parents=True)
    (data / "work" / "logs").mkdir(parents=True)
    tree = tmp_path / "umbrella"
    study = tree / "studies" / "speech-aware-evidence-acquisition"
    inventory = {"schema": "saea-cpu-encoder-inventory-v1", "workers": 4, "threads_per_worker": 16, "first_core": 128, "attention_geometry": "causal"}
    (data / "coordination" / "inv.json").write_text(json.dumps(inventory))
    for fold, run, reg in (("a", RUN_A, FOLD_A_REG), ("b", RUN_B, FOLD_B_REG)):
        protocol_rel = f"configs/preparation/p-{fold}.json"
        protocol = json.dumps({"encoder": {"path": "${SAEA_DATA_ROOT}/coordination/inv.json", "sha256": "00"}, "run": fold}, sort_keys=True).encode()
        (study / protocol_rel).parent.mkdir(parents=True, exist_ok=True)
        (study / protocol_rel).write_bytes(protocol)
        plan = {"run_id": run, "protocol_hash": hashlib.sha256(protocol).hexdigest()}
        (study / reg).parent.mkdir(parents=True, exist_ok=True)
        (study / reg).write_text(json.dumps({"umbrella_lock": LOCK, "plans": [{
            "execution_plan": plan, "session_id": f"sess-{fold}", "protocol": {"relpath": protocol_rel, "sha256": plan["protocol_hash"]}}]}))
    git = ["git", "-c", "user.email=t@t", "-c", "user.name=t", "-C", str(study)]
    sh(["git", "init", "-q", str(study)])
    sh(git + ["add", "-A"])
    sh(git + ["commit", "-q", "-m", "fixture"])
    return {"data": data, "tree": tree, "study": study, "head": sh(git + ["rev-parse", "HEAD"]).stdout.strip()}


def env_for(world, **extra):
    env = dict(os.environ, TREE=str(world["tree"]), OMNI_ROOT=str(world["data"]), DRY_RUN_PYTHON=sys.executable)
    env.pop("PORT_ENCODER_CPUSET", None)
    env.update(extra)
    return env


def listing(world):
    return sorted(str(p) for p in world["data"].rglob("*"))


def test_dry_run_fold_a_on_cores_184_223_prints_the_chain_and_flags_the_uncovered_pool(world):
    before = listing(world)
    out = sh(["bash", str(SH), "fold-a", "184-223", "--dry-run"], env=env_for(world)).stdout
    assert out.startswith("DRY RUN")
    assert f"container saea-port-encoder-fold-a  cpuset 184-223" in out
    assert out.count("--cpuset-cpus 184-223") == 2  # the config step and the encoder step
    assert "--cpuset-cpus 128-191" not in out
    assert f"port_write_runtime_config.py '{world['study']}' 'fold-a'" in out and "--cpuset '184-223'" in out
    assert "-m speech_aware_evidence_acquisition.core.speech_preparation_runtime" in out
    assert RUN_A in out and "cores 128-191 (64 cores); cpuset 184-223 does NOT cover it" in out
    assert "a real launch would be REFUSED" in out
    assert listing(world) == before  # nothing written: no log, no runtime config


def test_dry_run_with_a_covering_cpuset_is_ok(world):
    out = sh(["bash", str(SH), "fold-a", "100-199", "--dry-run"], env=env_for(world)).stdout
    assert "covers it" in out and "preflight verdict: OK" in out


def test_defaults_are_the_historical_fold_b_values(world):
    out = sh(["bash", str(SH), "fold-b", "--dry-run"], env=env_for(world)).stdout
    assert "container saea-port-encoder-fold-b  cpuset 128-191" in out
    assert out.count("--cpuset-cpus 128-191") == 2 and "preflight verdict: OK" in out
    # the env default moves it, an explicit argument beats the env
    assert "cpuset 0-63" in sh(["bash", str(SH), "fold-b", "--dry-run"], env=env_for(world, PORT_ENCODER_CPUSET="0-63")).stdout
    assert "cpuset 5-9" in sh(["bash", str(SH), "fold-b", "5-9", "--dry-run"], env=env_for(world, PORT_ENCODER_CPUSET="0-63")).stdout


def test_usage_errors(world):
    assert sh(["bash", str(SH)], env=env_for(world), check=False).returncode == 2
    assert sh(["bash", str(SH), "fold-b", "--bogus"], env=env_for(world), check=False).returncode == 2


def _stub_docker(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "docker.log"
    stub = bin_dir / "docker"
    body = ('#!/usr/bin/env bash\n'
            'if [ "$1" = ps ]; then exit 0; fi\n'
            'printf "%s\\n" "$*" >> LOGFILE\n'
            'case " $* " in *" --name "*) exit 0 ;; esac\n'
            'echo "/cfg/runtime-config.json deadbeef"\n')
    stub.write_text(body.replace("LOGFILE", f'"{log}"'))
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    return bin_dir, log


def _launch(world, tmp_path, *args):
    bin_dir, log = _stub_docker(tmp_path)
    env = env_for(world, PATH=f"{bin_dir}:{os.environ['PATH']}")
    out = sh(["bash", str(SH), *args], env=env).stdout
    for _ in range(50):  # the encoder docker call is backgrounded
        if log.exists() and len(log.read_text().splitlines()) >= 2:
            break
        time.sleep(0.1)
    return out, log.read_text().splitlines()


def test_default_fold_b_launch_issues_the_historical_docker_argv(world, tmp_path):
    out, calls = _launch(world, tmp_path, "fold-b")
    study, data = world["study"], world["data"]
    common = (f"--user 1008:1008 -e HOME=/tmp -e USER=saea -e LOGNAME=saea -e TORCHINDUCTOR_CACHE_DIR=/tmp/inductor -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 --cpuset-cpus 128-191 "
              f"-v {data}:{data} -w {study} -e PYTHONPATH={study}/src -e PYTHONDONTWRITEBYTECODE=1 -e SPEECHRL_DATA_DIR={data} -e SAEA_DATA_ROOT={data} "
              "--entrypoint bash saea-base:20261027")
    assert len(calls) == 2
    assert calls[0] == (f"run --rm {common} -c git config --global --add safe.directory '*' && /opt/saea-venv/bin/python {data}/work/scripts/port_write_runtime_config.py "
                        f"'{study}' 'fold-b' {data} --cpuset '128-191'")
    assert calls[1].startswith(f"run --rm --name saea-port-encoder-fold-b {common} -c git config --global --add safe.directory '*' && exec /opt/saea-venv/bin/python -m "
                               "speech_aware_evidence_acquisition.core.speech_preparation_runtime")
    assert f"--config '/cfg/runtime-config.json' --config-sha256 deadbeef --data-root {data} --source-root '{study}'" in calls[1]
    assert "launched saea-port-encoder-fold-b" in out
    logs = list((data / "work" / "logs").glob("port-encoder-fold-b-*.log"))
    assert len(logs) == 1 and logs[0].read_text().startswith("config /cfg/runtime-config.json sha256 deadbeef")


def test_fold_a_launch_passes_the_fold_and_cpuset_through(world, tmp_path):
    _, calls = _launch(world, tmp_path, "fold-a", "184-223")
    assert all("--cpuset-cpus 184-223" in call for call in calls)
    assert "'fold-a'" in calls[0] and "--cpuset '184-223'" in calls[0]
    assert "--name saea-port-encoder-fold-a" in calls[1]


def test_config_writer_fold_b_default_is_unchanged_and_fold_a_reads_its_own_registration(world):
    study, data = world["study"], world["data"]
    for which, run, session in (("fold-b", RUN_B, "sess-b"), ("fold-a", RUN_A, "sess-a")):
        result = sh([sys.executable, str(PY), str(study), which, str(data)])
        path, sha = result.stdout.split()
        raw = Path(path).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == sha
        assert Path(path).name == f"{run}-runtime-config-{world['head'][:12]}.json"
        config = json.loads(raw)
        assert set(config) == {"plan", "protocol", "session_id", "study_head", "umbrella_lock"}
        assert config["plan"]["run_id"] == run and config["session_id"] == session and config["study_head"] == world["head"] and config["umbrella_lock"] == LOCK
        assert raw == (json.dumps(config, indent=2, sort_keys=True) + "\n").encode()
        assert sh([sys.executable, str(PY), str(study), which, str(data)]).stdout == result.stdout  # idempotent create-only


def test_config_writer_refuses_a_cpuset_that_misses_the_pool_before_writing(world):
    study, data = world["study"], world["data"]
    result = sh([sys.executable, str(PY), str(study), "fold-a", str(data), "--cpuset", "184-223"], check=False)
    assert result.returncode != 0 and "does NOT cover it" in result.stderr
    assert not list((data / "coordination").glob("*runtime-config*"))
    ok = sh([sys.executable, str(PY), str(study), "fold-a", str(data), "--cpuset", "0-255"])
    assert ok.returncode == 0 and list((data / "coordination").glob("*runtime-config*"))


def test_check_only_exit_codes_and_no_side_effects(world):
    study, data = world["study"], world["data"]
    (study / "dirty.txt").write_text("x")  # a dirty checkout does not matter for --check-only, and nothing is written
    covered = sh([sys.executable, str(PY), str(study), "fold-b", str(data), "--cpuset", "128-191", "--check-only"], check=False)
    missed = sh([sys.executable, str(PY), str(study), "fold-b", str(data), "--cpuset", "184-223", "--check-only"], check=False)
    none = sh([sys.executable, str(PY), str(study), "fold-b", str(data), "--check-only"], check=False)
    assert (covered.returncode, missed.returncode, none.returncode) == (0, 3, 0)
    assert not list((data / "coordination").glob("*runtime-config*"))


def test_cpuset_parsing_and_pool_cores():
    assert cfg.parse_cpuset("0-3,8") == {0, 1, 2, 3, 8}
    assert cfg.parse_cpuset("184-223") == set(range(184, 224))
    assert cfg.pool_cores({"first_core": 128, "workers": 4, "threads_per_worker": 16}) == set(range(128, 192))
    with pytest.raises(ValueError):
        cfg.parse_cpuset("1,,2")


def test_scripts_are_english_only():
    for path in (SH, PY, Path(__file__)):
        assert not CJK.search(path.read_text(encoding="utf-8")), path

"""CPU checks for environment custody and the no-compute probe boundary."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import pytest

ROOT=Path(__file__).resolve().parents[2]/'scripts/environment/verl_omni'
spec=importlib.util.spec_from_file_location('omni_env_prepare',ROOT/'prepare.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_context_verify_rejects_changed_and_extra_files(tmp_path):
    (tmp_path/'source').write_text('fixed')
    (tmp_path/'source-lock.json').write_text(json.dumps({'files':m.inventory(tmp_path)}))
    assert m.verify(tmp_path)['verified']
    (tmp_path/'source').write_text('changed')
    with pytest.raises(ValueError):m.verify(tmp_path)
    (tmp_path/'source').write_text('fixed')
    (tmp_path/'extra').write_text('unexpected')
    with pytest.raises(ValueError):m.verify(tmp_path)

def test_snapshot_refuses_symlink_inputs(tmp_path):
    (tmp_path/'target').write_text('content')
    (tmp_path/'link').symlink_to(tmp_path/'target')
    with pytest.raises(ValueError,match='symlink'):m.inventory(tmp_path)

@pytest.mark.parametrize('fault',['floating_image','foreign_model','foreign_output','foreign_name'])
def test_probe_launcher_refuses_invalid_boundary(fault):
    args=['--image','sha256:'+'a'*64,'--model','/mnt/cephfs/xingchao/omni/models/model',
          '--out','/mnt/cephfs/xingchao/omni/work/probe-fixture-unused','--name','omni-verl-fixture']
    key,value={'floating_image':('--image','latest'),'foreign_model':('--model','/tmp/model'),
               'foreign_output':('--out','/tmp/output'),'foreign_name':('--name','foreign')}[fault]
    args[args.index(key)+1]=value
    result=subprocess.run([sys.executable,str(ROOT/'launch_probe.py'),*args],capture_output=True,text=True)
    assert result.returncode==2

def test_probe_dry_run_maps_no_compute_card_and_creates_no_output():
    out=Path('/mnt/cephfs/xingchao/omni/work/probe-fixture-dryrun-unused')
    assert not out.exists()
    result=subprocess.run([sys.executable,str(ROOT/'launch_probe.py'),'--image','sha256:'+'a'*64,
        '--model','/mnt/cephfs/xingchao/omni/models/model','--out',str(out)],capture_output=True,text=True,check=True)
    assert '--network=none' in result.stdout and '--device=/dev/davinci_manager' in result.stdout
    assert '--device=/dev/davinci0' not in result.stdout and '--rm' not in result.stdout
    assert not out.exists()

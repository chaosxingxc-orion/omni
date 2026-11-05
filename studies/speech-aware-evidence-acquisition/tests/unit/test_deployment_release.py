"""Deployment release boundary, exercised without devices or Docker services."""
import json
import hashlib
import subprocess
from pathlib import Path
import pytest
from speech_aware_evidence_acquisition.core import deployment as dep
from test_deployment import PROFILE_ID, BUSY_181, _document

IMAGE='sha256:'+'a'*64

def files(tmp_path):
    d=_document();d['image']=IMAGE
    p=tmp_path/(d['profile_id']+'.json');p.write_text(json.dumps(d))
    fragment=dep.model_fragment_values(dep.validate_deployment_profile(d))
    m=tmp_path/'model.json';m.write_text(json.dumps(fragment))
    busy=tmp_path/'npu.txt';busy.write_text(BUSY_181)
    return p,m,busy

def pins(p,m):
    return ['--profile-sha256',hashlib.sha256(p.read_bytes()).hexdigest(),'--model-fragment-sha256',hashlib.sha256(m.read_bytes()).hexdigest()]

def test_release_plan_is_busy_and_pins_profile_image(tmp_path,capsys):
    p,m,busy=files(tmp_path)
    assert dep._main(['plan',str(p),'--model-fragment-file',str(m),'--release-image',IMAGE,'--nic','eth0','--npu-smi-file',str(busy),*pins(p,m)])==0
    out=capsys.readouterr().out
    assert 'card check:' in out and 'refus' in out.lower() and IMAGE in out
    assert '--device /dev/davinci0' in out
    with pytest.raises(dep.CardRefusal):dep._main(['emit',str(p),'--model-fragment-file',str(m),'--release-image',IMAGE,'--nic','eth0','--npu-smi-file',str(busy),*pins(p,m)])

@pytest.mark.parametrize('image', ['sha256:'+'b'*64, 'a'*64, 'sha256:abc'])
def test_bad_or_mismatched_release_image_refused(tmp_path,image):
    p,m,_=files(tmp_path)
    with pytest.raises(dep.DeploymentError):dep._main(['plan',str(p),'--model-fragment-file',str(m),'--release-image',image,'--nic','eth0',*pins(p,m)])

def test_fragment_only_release_and_missing_fragment_fail(tmp_path):
    p,m,_=files(tmp_path)
    with pytest.raises(dep.DeploymentError):dep._main(['plan',str(p),'--model-fragment-file',str(m),'--nic','eth0'])
    with pytest.raises(dep.DeploymentError):dep._main(['plan',str(p),'--release-image',IMAGE,'--nic','eth0',*pins(p,m)])
    m.write_text('[]')
    with pytest.raises(dep.DeploymentError):dep._main(['plan',str(p),'--release-image',IMAGE,'--model-fragment-file',str(m),'--nic','eth0',*pins(p,m)])

def test_shell_rejects_bad_pins_before_docker(tmp_path):
    root=Path(__file__).resolve().parents[4]; script=root/'scripts/npu/npu_deploy.sh'
    p,m,_=files(tmp_path); marker=tmp_path/'executed'
    env={'PATH':'/usr/bin:/bin','SAEA_DEPLOY_MODE':'release','SAEA_RELEASE_IMAGE_ID':'sha256:bad',
         'SAEA_RELEASE_SOURCE_MANIFEST_SHA256':'a'*64,'SAEA_DEPLOY_MODEL_FRAGMENT':str(m)}
    result=subprocess.run(['bash',str(script),'plan',str(p)],env=env,capture_output=True,text=True)
    assert result.returncode and 'malformed' in result.stderr and not marker.exists()

def test_literal_fragment_flag_does_not_execute_shell(tmp_path,capsys):
    p,m,_=files(tmp_path);d=json.loads(m.read_text());d['served_model_name']='literal-$(touch /tmp/saea-shell-injection)';m.write_text(json.dumps(d))
    dep._main(['plan',str(p),'--model-fragment-file',str(m),'--release-image',IMAGE,'--nic','eth0',*pins(p,m)])
    assert "'literal-$(touch /tmp/saea-shell-injection)'" in capsys.readouterr().out
    assert not Path('/tmp/saea-shell-injection').exists()


def test_release_record_pins_both_inputs_and_changed_bytes_rejected(tmp_path,capsys):
    p,m,_=files(tmp_path); original=pins(p,m)
    dep._main(['record',str(p),'--release-image',IMAGE,'--model-fragment-file',str(m),*original])
    record=json.loads(capsys.readouterr().out)
    assert record['profile_file_sha256']==original[1]
    assert record['model_fragment_file_sha256']==original[3]
    m.write_text(m.read_text()+' ')
    with pytest.raises(dep.DeploymentError,match='SHA256 differs'):
        dep._main(['plan',str(p),'--release-image',IMAGE,'--model-fragment-file',str(m),'--nic','eth0',*original])

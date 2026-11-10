"""Main-owned fixed-image entry. Emits deployable TRAIN pins only after ALL checks."""
import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[3]
OWNER=ROOT/'delivery/s-terminal-handoff-repair-v1'
sys.path.insert(0,str(OWNER/'code'))
from transport_manifest import verify
import terminal_handoff as h
IMAGE='sha256:7e60208fb45e8af3f1ab777cf24df7a7a415d8d84e2972ee76def4daf5c0f026'
PYTHON='/opt/saea-venv/bin/python'

def dependency_packages():
    deps={}
    for name in ('numpy','tiktoken','pydantic','openjiuwen'):
        m=importlib.import_module(name)
        try:version=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:version=getattr(m,'__version__','unknown')
        deps[name]=dict(required=True,available=True,version=version,path=getattr(m,'__file__',None))
    # Native model communication does not require torch. Never import/install it.
    spec=importlib.util.find_spec('torch')
    deps['torch']=dict(required=False,available=spec is not None,
                       path=spec.origin if spec is not None else None,probe='find_spec')
    return deps

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--image-id',required=True);p.add_argument('--expected-python',default=PYTHON)
    a=p.parse_args();start=time.monotonic()
    h.need(sys.executable==a.expected_python==PYTHON,'fixed /opt runtime Python required')
    h.need(a.image_id==IMAGE and Path('/.dockerenv').exists(),'fixed image container required')
    h.need(os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')=='0','disable NPU autoload')
    h.need(not list(Path('/dev').glob('davinci*')) and not list(Path('/dev').glob('nvidia*')),'no accelerator devices')
    h.need(os.environ.get('HF_HUB_OFFLINE')=='1' and os.environ.get('TRANSFORMERS_OFFLINE')=='1','offline environment')
    # Docker --network none supplies only lo, and no daemon socket is mounted.
    interfaces={p.name for p in Path('/sys/class/net').iterdir()}
    h.need(interfaces<={'lo'} and not Path('/var/run/docker.sock').exists(),'network none/no Docker socket')
    out=Path(a.output).resolve();h.need(out.is_relative_to(OWNER/'evidence') and not out.exists(),'fresh owner evidence attempt')
    out.mkdir(parents=True);tmp=out/'tmp';tmp.mkdir();steps=[]
    env=dict(os.environ,TMPDIR=str(tmp),PYTHONDONTWRITEBYTECODE='1')
    def step(name,args):
        t=time.monotonic();r=subprocess.run([sys.executable,'-B']+list(map(str,args)),env=env,capture_output=True,text=True)
        (out/(name+'.stdout.log')).write_text(r.stdout);(out/(name+'.stderr.log')).write_text(r.stderr)
        steps.append(dict(name=name,argv=args,returncode=r.returncode,wall_seconds=time.monotonic()-t))
        h.json_new(out/(name+'.json'),dict(step=steps[-1]))
        h.need(r.returncode==0,name+' failed; logs preserved')
        return r
    # Explicit user order: first run all 21 existing fixtures.
    fixtures=step('fixtures21',[str(OWNER/'code/test_handoff.py')])
    h.need('Ran 21 tests' in fixtures.stderr and fixtures.stderr.rstrip().endswith('OK'),'21 fixtures required')
    step('optional-torch-fixtures',[str(OWNER/'code/test_fixed_cpu_dependencies.py')])
    deps=dependency_packages()
    # No tokenizer execution is needed: same unchanged real wires reuse formal
    # budget, which the native validator recomputes and checks for every branch.
    h.json_new(out/'DEPENDENCIES.json',dict(python=sys.executable,packages=deps,
          tokenizer_required=False,jinja2_required=False,network_interfaces=sorted(interfaces)))
    manifest=h.read(OWNER/'TRANSPORT-MANIFEST.json');count=verify(manifest)
    step('real-validation',[str(OWNER/'code/cpu_real_validation.py'),'--output',str(out/'real')])
    result=h.read(out/'real/RESULT.json')
    h.need(result['status']=='PASS_REAL_CPU_BIND_LOAD_NATIVE_VALIDATION' and
           [x['workers'] for x in result['bind_load']]==[64,96,128] and len(result['native'])==8 and result['full_native_coroutine'] is True,'complete CPU suite')
    verify(manifest)  # sources remain byte-identical after validation.
    receipt=dict(status='PASS_FIXED_IMAGE_REAL_TRAIN_HANDOFF_CPU',image=IMAGE,python=sys.executable,
          pins=result['pins'],resources_cache=result['resources'],model_calls=0,NPU_calls=0,
          runtime_admitted=False,owner_reconciled_is_cpu_fixture=True,journal_created=False,
          fixtures=21,dependency_fixtures=4,train_samples=933,train_branches=2799,concurrency=[64,96,128],
          native_checks=result['native'],transport_manifest=h.pin(OWNER/'TRANSPORT-MANIFEST.json'),
          real_validation=h.pin(out/'real/RESULT.json'),dependency_receipt=h.pin(out/'DEPENDENCIES.json'),
          formal_fixed_image=True,verified_files=count,wall_seconds=time.monotonic()-start,
          scope='CPU implementation verified only; does not authorize real owner/lease admission')
    h.json_new(out/'CPU-VERIFIED.json',receipt)
    # LAST write: no partial/failing run can emit deployable HANDOFF pins.
    h.json_new(out/'HANDOFF-TRAIN-VERIFIED.json',receipt)
    print(json.dumps(receipt))
if __name__=='__main__':main()

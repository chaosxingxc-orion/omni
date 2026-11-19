"""Replay actual retained witness through closed locations; reject mutation."""
import argparse, importlib.util, json, sys
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--source',required=True)
p.add_argument('--returned',required=True)
p.add_argument('--output',required=True)
p.add_argument('--dependency-returned',action='append',default=[])
a=p.parse_args()
from dependencies import load, ROOT
from transport import restore_locations
from run_package import read
for directory in a.dependency_returned:restore_locations(Path(directory))
restore_locations(Path(a.returned))
result=read(Path(a.returned)/'RETURN.json')
raw=read(result['raw_ref'])
candidate=load(Path(a.source),'candidate_policy_identity')
binding=candidate.behavior_binding(raw['resident_identity'],raw)
assert binding==raw['behavior_binding']
saved={n:sys.modules.pop(n,None) for n in ('io_utils','protocol','lineage')}
sys.path.insert(0,str(ROOT/'policy-iteration-successor'))
try:
 protocol=load(ROOT/'policy-iteration-successor/protocol.py','closed_witness_protocol')
 io=sys.modules['io_utils']
 candidate.bind_readonly_protocol(protocol,io)
 ref=raw['endpoint_witness_ref']
 value=protocol.doc(ref)
 assert protocol.save(ref['path'],value)==ref
 for altered in (dict(ref,sha256='0'*64),dict(ref,bytes=ref['bytes']+1)):
  try:protocol.raw(altered)
  except ValueError:pass
  else:raise AssertionError('changed reference accepted')
 try:protocol.save(ref['path'],dict(value,passed=False))
 except ValueError:pass
 else:raise AssertionError('changed witness accepted')
 try:protocol.save(ref['path']+'.absent',value)
 except FileNotFoundError:pass
 else:raise AssertionError('verification created a missing witness')
finally:
 sys.path.pop(0)
 for name,old in saved.items():
  sys.modules.pop(name,None)
  if old is not None:sys.modules[name]=old
out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
report=dict(passed=True,checks=6,actual_closed_witness=True,model_calls=0,
            numerical_passed=result['passed'],optimizer_steps=result['optimizer_steps'])
import policy_identity
policy_identity.behavior_binding=candidate.behavior_binding
from parity_gate import verify
from dependencies import sha,HERE
try:
 verify(result['raw_ref'],config=read(result['runtime_ref']),policy_version=raw['policy_version'],
        source_pins_sha256=sha(HERE/'SOURCE-PINS.json'),identity=raw['resident_identity'])
except ValueError as error:
 assert str(error)=='real DP8 R/C/P failed; original thresholds retained',str(error)
 report.update(checks=7,actual_verifier_reached_numeric_rejection=True)
else:raise AssertionError('known numerical failure incorrectly accepted')
(out/'RESULT.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))

"""Bind the actual P iteration to scoped diagnostic dispatch and storage."""
import argparse,copy,json
from pathlib import Path
from run_package import read,pin,write
from configure_release import bind_scoped_diagnostic_sampling,capture_capacity
from diagnostic_storage import plan
p=argparse.ArgumentParser()
for n in ('iteration','spec','previous_spec','output'):p.add_argument('--'+n,required=True)
a=p.parse_args();c=copy.deepcopy(read(a.iteration));package=copy.deepcopy(read(c['training_package_template']))
c['finite_diagnostic_config_ref']=pin(a.previous_spec)
before=plan(c['diagnostic_logits_capture'],capture_capacity(c,c['diagnostic_logits_capture']),package['diagnostic_selection']['max_records'],4)
c['finite_diagnostic_config_ref']=pin(a.spec);bind_scoped_diagnostic_sampling(c,package)
assert c['diagnostic_target_nodes']==['P']
assert c['diagnostic_capture_budgets']['base_parity']==dict(tokens_upper=417792,requests_upper=16)
assert c['diagnostic_logits_capture']['selection']['inputs_per_node']==dict(P=1,S1=1,S2=0,D=0)
assert package['budget']['diagnostic_token_upper']==1114112
actual=plan(c['diagnostic_logits_capture'],capture_capacity(c,c['diagnostic_logits_capture']),package['diagnostic_selection']['max_records'],4)
assert all(actual['roles'][r]<before['roles'][r] for r in actual['roles'])
bad=copy.deepcopy(c);bad['diagnostic_target_nodes']=['D']
try:bind_scoped_diagnostic_sampling(bad,copy.deepcopy(package))
except ValueError:pass
else:raise AssertionError('missing training target accepted')
write(Path(a.output)/'RESULT.json',dict(passed=True,checks=6,dispatch=False,optimizer_updates=0,
    before=before,after=actual,requests=c['diagnostic_capture_budgets'],
    note='retained causal-control gate unchanged; actual fresh samples still required'))
print('PASS 6 actual P configuration and capacity checks')

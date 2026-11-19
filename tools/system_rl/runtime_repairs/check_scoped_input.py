"""Use actual pinned zero-update P input to derive independent P/S1 evidence."""
import argparse,copy
from pathlib import Path
from run_package import read,pin,write
from dependencies import register_locations,ROOT,VIRTUAL_ROOT
from next_input import diagnostic_input
p=argparse.ArgumentParser()
for n in ('cursor','output'):p.add_argument('--'+n,required=True)
a=p.parse_args();state=read(a.cursor);source=read(state['input_ref']);out=Path(a.output)
# Existing learning source is not rewritten by evidence derivation.
old=read(source['learning_config_ref']);before=read(source['clock_state_ref'])
ref=diagnostic_input(state['input_ref'],None,source['policy']['current_ref'],state['witness_ref'],out,
    root_seed=47101,sampling_round=0,cohort_sequence=1,evidence_nodes=['P','S1'])
v=read(ref);learning=read(v['learning_config_ref']);phase=learning['phases'][0]
assert phase['target_nodes']==['P','S1'] and phase['trainable_mask']==dict(P=True,S1=True,S2=False,D=False)
assert not v['native']['enabled'] and not v['authorization']['native_training']
assert v['diagnostic_evidence_only']['trainable'] is False
assert old==read(source['learning_config_ref']) and before==read(source['clock_state_ref'])
assert old['phases'][0]['target_nodes']==['P']
write(out/'RESULT.json',dict(passed=True,checks=5,model_calls=0,optimizer_updates=0,input_ref=ref))
print('PASS 5 actual input scope and training isolation checks')

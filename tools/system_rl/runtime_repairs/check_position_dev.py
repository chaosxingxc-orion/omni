"""Exercise the original paired DEV gate, not business quality evidence."""
import argparse,json
from pathlib import Path
from unittest.mock import patch
import business_gate as gate
from run_package import write,pin
from dependencies import HERE
from storage import environment

def run(root):
    root.mkdir(parents=True,exist_ok=False)
    environment(root,0)
    cfg={'bootstrap':{'replicates':100,'seed':13,'confidence':.95},'phase_acceptance':{'P':dict(
        target='P',record_metric_path=['node_statistics','P','coverage_at3'],failure_policy='stop_preserve',
        system_minimum_deltas={'slu_f1':0.},minimum_target_gain=0.,minimum_target_lower_bound=0.)}}
    gate.bind_p_phase(cfg,json.loads((HERE/'learning.template.json').read_bytes()))
    assert cfg['phase_acceptance']['P']['record_metric_path'][-1]=='position_reward_v1'
    system=write(root/'SYSTEM.json',dict(fixture=False,quality_claim_allowed=True,recordings=8690,
        candidate={'valid_answer_rate':1.},paired={'slu_f1':{'delta':0.}}))
    refs=[]
    for j in range(3):
        p=root/f'rows-{j}.jsonl'
        with p.open('x') as out:
            for i in range(8690):
                candidates=['b','c','a'] if i else ['b','c'];selected=(candidates if j==0 else ['a']) if i else ['b']
                if j==2 and i==1:selected=['b']
                hit=int('a' in selected)
                call=dict(labels={'intent':'a'},output={'selected':selected},online={'retrieved_candidates':candidates},call_ids=['held'])
                stat=dict(mask=1,complete=True,coverage_at3=hit,reward_calls=[call])
                out.write(json.dumps(dict(sample_id=str(i),source_id=str(i%16),reference_id=str(i),gold_sha256='test',
                    statistics={'joint_exact':0,'valid_answer':1},counts={'original':1},node_statistics={'P':stat}))+'\n')
        refs.append(write(root/f'report-{j}.json',dict(evaluation_complete=True,denominator=8690,records=pin(p))))
    with patch.object(gate,'assess',return_value=(system,{})):
        for j in [1,2]:
            _,result=gate.assess_phase(refs[0],refs[j],cfg,'P',root/f'quality-{j}')
            assert result['checks']['target_gain'] and result['checks']['target_paired_lower_bound']
            assert result['P_coverage_at3']==dict(denominator=8690,baseline_hits=8689,candidate_hits=8690-j,delta=(1-j)/8690)
            assert result['passed']==(j==1)
    write(root/'RESULT.json',dict(passed=True,fixture=True,actual_dev_gain_claim=False,model_calls=0,
        checks=['rank3 to rank1 passes with flat coverage','missing upstream gold retained in denominator',
                'one new miss vetoes promotion despite positive rank gain']))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)

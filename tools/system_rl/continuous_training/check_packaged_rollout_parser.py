"""Consume real rollout replies using just the parser member of the release."""
import argparse,hashlib,json,os,sys
from pathlib import Path
def run(a):
    parents=a.release/'tree/parents';sys.path.insert(0,str(parents/'fresh-cohort-successor'))
    import rails
    os.environ.pop('OMNI_SCREENED_P_PARSER_CONFIG_REF',None)
    member=json.loads(a.policy.read_bytes())['screened_p_parser_member'];parse=rails.node_p_parser(dict(screened_p_parser_member=member))
    count=0;records=set()
    for path in sorted((a.cohort/'groups').glob('*/GENERATED.json')):
        group=json.loads(path.read_bytes())
        for ep in group['episodes']:
            call,=ep['calls'];packet=call['packet'];reply=call['reply'];result=parse(packet,reply)
            if [item['candidate_id'] for item in result['candidates']]!=call['output']['selected'] or result['status']!=call['parse_diagnostic']['status']:
                raise ValueError('original ranked output or status differs')
            count+=1;records.add(packet['sample_id'])
    try:rails.node_p_parser(dict(screened_p_parser_member=dict(member,sha256='0'*64)))
    except ValueError:pass
    else:raise ValueError('incorrect source accepted')
    print(json.dumps(dict(groups=len(records),episodes=count,all_rankings_and_status_equal=True,
        rejected_wrong_parser=True,parser_sha256=member['sha256'],npu=False)))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('release','policy','cohort'):p.add_argument('--'+k,type=Path,required=True)
    run(p.parse_args())

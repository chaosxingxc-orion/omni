from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/analysis'))
import resume_a0_resident_repin as m


@pytest.mark.parametrize('fault',[None,'decoding','ready','receipt'])
def test_partial_refresh_preserves_science_and_held_state(fault):
    old={'runtime_receipt':'old.json','temperature':0.7}
    model={**old,'runtime_receipt':m.original.RECEIPT}
    spec={'runtime_receipt':m.original.RECEIPT,'service_session':m.original.SESSION,
          'resident_refresh':{'runtime_receipt_sha256':'a'*64}}
    exp={'service_session':m.original.SESSION,'cells':[{'run_id':m.RUN,'readiness':{'state':'held'}}]}
    if fault=='decoding':model['temperature']=1
    elif fault=='ready':exp['cells'][0]['readiness']['state']='ready'
    elif fault=='receipt':spec['resident_refresh']['runtime_receipt_sha256']='b'*64
    if fault:
        with pytest.raises(ValueError):m.validate_partial(old,model,spec,exp,'a'*64)
    else:m.validate_partial(old,model,spec,exp,'a'*64)

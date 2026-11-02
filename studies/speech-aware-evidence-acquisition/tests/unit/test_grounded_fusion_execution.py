import copy
import json
import pytest
from speech_aware_evidence_acquisition.rails import grounded_fusion_execution as m
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import (
    ExperienceLearner, empty_experience, sha, UnknownLearningContact,
)


def fixture(tmp_path):
    custody = {'unknown_contacts':0,
               'original_total_usage':{'calls':312,'prompt_tokens':1000,'completion_tokens':200}}
    binding = {'schema':m.SCHEMA, 'model_id':'model','learning_config_sha256':'a'*64,
               'source_custody_sha256':sha(custody), 'parent':empty_experience('model'),
               'learned_query_ids':['slurp:train:q'], 'identity':'final',
               'messages':[{'role':'system','content':'learn'},{'role':'user','content':'evidence'}],
               'bounds':{'max_items':2,'max_characters':100,'max_operations':2},
               'rollout_kind':'shared-perception-adaptation','reward_mode':'intent-exact-slot-f1-v1'}
    calls, fits = [], []
    def call(request):
        calls.append(request)
        return {'model_id':'model','receipt_id':'r1',
                'text':'[{"operation":"ADD","id":null,"content":"conditional rule"}]',
                'usage':{'calls':1,'prompt_tokens':20,'completion_tokens':10}}
    def learner(call=call):
        return ExperienceLearner(tmp_path,model_id='model',metered_call=call,
            learning_config_sha256='a'*64,max_calls=1,preflight=lambda r:fits.append(r))
    verify = lambda:copy.deepcopy(custody)
    validate = lambda text:{'operations':json.loads(text),'semantic_review_required':True}
    return learner,binding,verify,validate,calls,fits


def test_known_result_reused_with_same_candidate_cost_and_parent(tmp_path):
    make,binding,verify,validate,calls,fits=fixture(tmp_path)
    first=m.execute_fusion(make(),binding,verify_source=verify,validate_output=validate)
    second=m.execute_fusion(make(),binding,verify_source=verify,validate_output=validate)
    assert first == second and len(calls)==len(fits)==1
    assert first['historical_fusion_custody']['original_total_usage']['calls']==312
    assert first['learning_usage']['calls']==1 and first['semantic_review_required']
    assert first['deployment_accepted'] is False and first['parent_sha256']==binding['parent']['sha256']


def test_invalid_observed_output_is_retained_never_resampled(tmp_path):
    make,binding,verify,_,calls,fits=fixture(tmp_path)
    def reject(text):raise ValueError('unsupported references')
    for _ in range(2):
        with pytest.raises(ValueError,match='unsupported'):
            m.execute_fusion(make(),binding,verify_source=verify,validate_output=reject)
    assert len(calls)==len(fits)==1 and not (tmp_path/'candidate.json').exists()


def test_unknown_result_blocks_resend(tmp_path):
    make,binding,verify,validate,calls,fits=fixture(tmp_path)
    def unknown(request):raise OSError('connection lost')
    with pytest.raises(OSError):
        m.execute_fusion(make(unknown),binding,verify_source=verify,validate_output=validate)
    with pytest.raises(UnknownLearningContact):
        m.execute_fusion(make(),binding,verify_source=verify,validate_output=validate)
    assert not calls and len(fits)==1


@pytest.mark.parametrize('fault',['source','model','ids','bounds'])
def test_source_or_binding_errors_before_any_spend(tmp_path,fault):
    make,binding,verify,validate,calls,fits=fixture(tmp_path)
    if fault=='source':binding['source_custody_sha256']='b'*64
    if fault=='model':binding['model_id']='another'
    if fault=='ids':binding['learned_query_ids']=['slurp:test:q']
    if fault=='bounds':binding['bounds']['max_items']=True
    with pytest.raises(ValueError):
        m.execute_fusion(make(),binding,verify_source=verify,validate_output=validate)
    assert not calls and not fits


def test_explicit_empty_update_is_a_retained_rejected_candidate(tmp_path):
    make,binding,verify,_,calls,fits=fixture(tmp_path)
    result=m.execute_fusion(make(),binding,verify_source=verify,
        validate_output=lambda text:{'operations':[{'operation':'NONE','id':None,'content':''}],
                                    'disposition':'all-rejected'})
    assert result['empty_candidate'] and result['items']=={}
    assert result['deployment_accepted'] is False and len(calls)==1

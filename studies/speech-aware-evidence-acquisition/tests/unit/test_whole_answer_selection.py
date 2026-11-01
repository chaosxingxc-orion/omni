import copy
import json
import pytest
from speech_aware_evidence_acquisition.rails import whole_answer_selection as m


def text(value='one'):
    return json.dumps({'scenario':'alarm','action':'set','entities':[{'type':'time','filler':value}]})


def contact(label, value):
    return {'agent_contact_id':label,'request_id':label+'-request','request_sha256':'a'*64,
            'response_sha256':'b'*64,'response_text_sha256':m.text_sha(value)}


def args():
    return dict(a_text=text(), b_text=text('two'), a_contact=contact('a',text()), b_contact=contact('b',text('two')))


def test_source_bound_whole_answer_and_replay():
    judge='{"choice":"B"}'
    out,receipt=m.select_observed(**args(),judge_text=judge,judge_contact=contact('judge',judge),judge_finish_reason='stop')
    assert out==text('two') and receipt['selected_contact']==args()['b_contact']
    assert m.verify_receipt(receipt,a_text=text(),b_text=text('two'),judge_text=judge)==out
    altered=copy.deepcopy(receipt);altered['selected']='A'
    with pytest.raises(ValueError):m.verify_receipt(altered,a_text=text(),b_text=text('two'),judge_text=judge)


@pytest.mark.parametrize('fault',['missing','missing_contact','wrong_hash','same_contact'])
def test_unknown_or_unbound_outcome_never_falls_back(fault):
    judge='{"choice":"B"}';kwargs=dict(judge_text=judge,judge_contact=contact('judge',judge),judge_finish_reason='stop')
    if fault=='missing':kwargs['judge_text']=None
    elif fault=='missing_contact':kwargs['judge_contact']=None
    elif fault=='wrong_hash':kwargs['judge_contact']['response_text_sha256']='0'*64
    else:kwargs['judge_contact']['agent_contact_id']='a'
    with pytest.raises(ValueError):m.select_observed(**args(),**kwargs)


@pytest.mark.parametrize('finish',['length','content_filter',None])
def test_observed_nonstop_choice_falls_back(finish):
    judge='{"choice":"B"}'
    out,r=m.select_observed(**args(),judge_text=judge,judge_contact=contact('judge',judge),judge_finish_reason=finish)
    assert out==text() and r['reason']=='non-stop-selection-fallback'


def test_equal_candidates_have_no_judge_contact():
    values=args();values['b_text']=text();values['b_contact']=contact('b',text())
    out,r=m.select_observed(**values)
    assert out==text() and r['additional_contacts']==0 and r['adjudicator'] is None


def test_prompt_uses_original_packet_and_union_of_typed_evidence():
    speech={'type':'audio','url':'same-recording'}
    evidence={'type':'text','text':'evidence A','evidence_id':'v1','knowledge_role':'value','source_uri':'test:v1'}
    a=[{'role':'system','content':'contract'},{'role':'user','content':[speech,evidence]}]
    b=copy.deepcopy(a);b[1]['content'].append({**evidence,'evidence_id':'v2','text':'evidence B'})
    before=copy.deepcopy((a,b))
    messages,data=m.compose_messages(a,b,a_text=text(),b_text=text('two'),perception_text='shared note')
    assert (a,b)==before
    parts=messages[1]['content'];assert parts.count(speech)==1
    assert [p['evidence_id'] for p in parts if 'evidence_id' in p]==['v1','v2']
    assert data['candidates'][0]['evidence_ids']==['v1']
    assert data['candidates'][1]['evidence_ids']==['v1','v2']
    b[1]['content'][0]['url']='different'
    with pytest.raises(ValueError):m.compose_messages(a,b,a_text=text(),b_text=text('two'),perception_text='note')


def test_compiler_and_loader_count_conditional_fifth_contact(tmp_path):
    from test_tfrl_odd40_candidates import m as candidates, documents, ROOT
    from speech_aware_evidence_acquisition.rails.controllers import default_phase_program
    from speech_aware_evidence_acquisition.rails.native_episode_config import load_native_episode_config
    base,_=candidates.derive(ROOT,documents(tmp_path)[0])
    block=copy.deepcopy(base['native_episode']['consumption_module']);block['answer_selection']=m.POLICY_ID
    compiled=candidates.cm.compile_module(block,candidates.carrier_module_hooks(repo_root=ROOT),repo_root=ROOT)
    values=compiled.apply_to(base);values['native_episode']['consumption_module']=block
    parsed=load_native_episode_config(values,default_phase_program(values['policy']))
    assert parsed.max_contacts==compiled.program_ceiling==5
    assert parsed.answer_selection==m.POLICY_ID


def test_full_method_derivation_preserves_candidates(tmp_path):
    from test_tfrl_odd40_candidates import m as candidates, documents, ROOT
    source=documents(tmp_path)[0];before=copy.deepcopy(source)
    base,proof=candidates.derive_method(ROOT,source)
    assert source==before
    assert proof['method_program_ceiling']==5 and proof['full_method_budget_compiled']
    assert proof['generation_authorized'] is False
    assert base['native_episode']['answer_selection']==m.POLICY_ID
    assert base['phase_material']==source['phase_material']
    assert base['knowledge_stores']==source['knowledge_stores']


@pytest.mark.parametrize('fault',['policy','group','prompt','ceiling'])
def test_selection_runtime_refuses_unbound_config(tmp_path,fault):
    from dataclasses import replace
    from test_tfrl_odd40_candidates import parsed_candidate
    from speech_aware_evidence_acquisition.rails.native_episode_config import validate_prompt_contact_limits
    config=replace(parsed_candidate(tmp_path),answer_selection=m.POLICY_ID,max_contacts=5)
    if fault=='policy':config=replace(config,answer_selection='other')
    elif fault=='prompt':config=replace(config,prompt_contact_limits=None)
    elif fault=='ceiling':config=replace(config,max_contacts=4)
    else:config=replace(config,roles=tuple(replace(role,supply_variants=None) if role.role_kind=='solver' else role for role in config.roles))
    with pytest.raises(ValueError):validate_prompt_contact_limits(config)

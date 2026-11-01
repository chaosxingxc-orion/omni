import copy
import sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/analysis'))
import tfrl_odd40_candidates as m
from test_tfrl_current_banked_cells import documents


def test_real_compiler_preserves_control_and_removes_consumed_bindings(tmp_path):
    original = documents(tmp_path)[0]
    saved = copy.deepcopy(original)
    base, proof = m.derive(ROOT, original)
    assert original == saved
    assert proof['candidate_program_ceiling'] == 4
    assert proof['full_method_budget_compiled'] is False
    assert proof['generation_authorized'] is False
    assert 'native_contact_fit' not in base and 'prepared_phase_supply' not in base
    assert base['knowledge_stores'] == original['knowledge_stores']
    solver = base['native_episode']['roles']['solver']
    arms = solver['supply_variants']['arms']
    assert [a['arm_id'] for a in arms] == ['b','aa','lock']
    assert arms[0]['overrides'][m.ODD40_OVERRIDE_PATH] == m.ODD40_OVERRIDE_VALUE
    assert all('overrides' not in a for a in arms[1:])
    assert base['phase_material'] == original['phase_material']
    assert base['native_episode']['prompt_contact_limits'] == original['native_episode']['prompt_contact_limits']


@pytest.mark.parametrize('fault',['source_policy','bank','columns','k'])
def test_rejects_drifted_source(tmp_path,fault):
    source = documents(tmp_path)[0]
    if fault=='source_policy': source['policy']='unknown'
    elif fault=='bank': source['native_episode']['consumption_module']['tfrl_v2_bank']={}
    elif fault=='columns': source['native_episode']['consumption_module']['columns'].pop()
    else: source['knowledge_stores']['train-exemplars']['phase_supply']['solver']['k']=19
    with pytest.raises((ValueError, m.cm.ConsumptionModuleConfigError)): m.derive(ROOT,source)


@pytest.mark.parametrize('fault',['policy','count','twin','bound'])
def test_legacy_prompt_extension_rejects_other_interventions(tmp_path,fault):
    base,_ = m.derive(ROOT,documents(tmp_path)[0])
    block = copy.deepcopy(base['native_episode']['consumption_module'])
    if fault=='policy': block['columns'][0]['overrides'][m.ODD40_OVERRIDE_PATH]='saea-fuse-supplyfilter-v1'
    elif fault=='count': block['columns'][0]['overrides']['knowledge_stores.train-exemplars.phase_supply.solver.k']=30
    elif fault=='twin': block['supply_frame']['twin']=False
    else: block['columns'][0]['max_supplied_items']+=1
    with pytest.raises(m.cm.ConsumptionModuleConfigError):
        m.cm.compile_module(block,m.carrier_module_hooks(repo_root=ROOT),repo_root=ROOT)

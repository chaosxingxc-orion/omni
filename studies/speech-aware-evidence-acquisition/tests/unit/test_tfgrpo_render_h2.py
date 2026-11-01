import copy
import json
from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/analysis'))
import tfgrpo_render_h2_paired as m
import tfgrpo_render_h2_read as reader
from test_tfgrpo_consumption import release
from test_tfgrpo_h2_read import inputs


def test_three_arm_statistics_preserve_primary_and_representation_contrasts():
    r,items=inputs();r['compact']=copy.deepcopy(r['learned'])
    out=reader.statistics(r,items,draws=20)
    assert out['compact_vs_learned']['metrics']['slu_f1']['point_pp']==0
    assert out['compact_vs_lock']['numeric_gate_passed']
    assert out['primary_contrast']=='compact_vs_lock'
    assert not out['candidate_accepted'] and not out['latency_causal_claim']
    del r['learned']
    with pytest.raises(ValueError):reader.statistics(r,items,draws=20)


def test_compile_three_arms_preserves_perception_supply_and_delivery(release,monkeypatch):
    _,data,params=release
    roster=json.loads((ROOT/m.ROSTER).read_bytes())
    params['consumer_ids']=[r['sample_id'] for r in roster['items']]
    original=json.loads((ROOT/'configs/baseline/slurp-tfrl-hardset-a0-observation-r1-v1.json').read_bytes())
    params['base_template']['relpath']=original['phase_material']['solver']['instruction_template']
    compact=copy.deepcopy(params)
    compact.update(rendering='short-display-id-v1',render_manifest={'relpath':'probes/short.render.json','sha256':'a'*64},
                   template={'relpath':'probes/short.txt','sha256':'b'*64})
    monkeypatch.setattr(m.consumption,'verify',lambda *a,**kw:None)
    kwargs=dict(fit_estimate={'slot_tokens':49152,'completion_ceiling':4096,'base_prompt_tokens':9600,'tokens_per_supplied_item':120},
                base_template_tokens=208,learned_template_tokens=5850,compact_template_tokens=2385)
    out=m.derive(ROOT,data,params,compact,**kwargs)
    base=out['baseline']
    assert base['knowledge_stores']==original['knowledge_stores']
    assert base['phase_material']==original['phase_material']
    assert base['native_episode']['roles']['perception']==original['native_episode']['roles']['perception']
    arms=base['native_episode']['roles']['solver']['template_variants']['arms']
    assert [a['arm_id'] for a in arms]==['learned','compact','lock']
    assert [a['arm_id'] for a in arms if a.get('deliver')]==['lock']
    assert base['native_contact_fit']['contact_cap']==1248
    assert out['experiment']['execution_plan']['planned_model_calls']==2496
    assert base['native_episode']['consumption_module']['template_frame']['max_supplied_items']==56
    assert out['evaluation']['normal_generation_calls']==1248
    assert not out['evaluation']['latency_causal_claim']
    from speech_aware_evidence_acquisition.core.flight.native_recovery import native_contact_custody_enabled
    assert native_contact_custody_enabled({**base,'transport':'vllm','carrier_lock_key':'slurp'})
    with pytest.raises(ValueError):m.derive(ROOT,data,params,compact,**{**kwargs,'compact_template_tokens':5850})
    compact['candidate']['sha256']='f'*64
    with pytest.raises(ValueError):m.derive(ROOT,data,params,compact,**kwargs)

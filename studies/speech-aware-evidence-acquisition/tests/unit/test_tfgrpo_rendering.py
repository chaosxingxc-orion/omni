import copy
import json
import pytest
from speech_aware_evidence_acquisition.rails import tfgrpo_rendering as r
from speech_aware_evidence_acquisition.rails import tfgrpo_consumption as c
from speech_aware_evidence_acquisition.rails.tfgrpo_learning import sha
from test_tfgrpo_consumption import candidate,release,TARGET,SOURCE


def test_display_only_preserves_values_and_identity():
    value=candidate();before=copy.deepcopy(value)
    text,manifest=r.render(value,TARGET)
    assert text=='[1]. '+value['items']['E1']
    assert manifest['mapping'][0]['stable_id']=='E1'
    assert manifest['candidate_sha256']==value['sha256'] and value==before
    with pytest.raises(ValueError):r.render(value,SOURCE)
    value['items']['E2']='See E1 for details';value['sha256']=sha({k:v for k,v in value.items() if k!='sha256'})
    with pytest.raises(ValueError,match='references'):r.render(value,TARGET)


def test_short_prepare_pins_mapping_and_rejects_tamper(release):
    repo,data,params=release
    short=c.prepare(candidate_path=data/'candidate.json',source_manifest=data/'manifest.json',repo=repo,data=data,
        base_template='probes/base.txt',template='probes/short.txt',consumer_ids=[TARGET],model_id='model',rendering=r.SHORT)
    assert c.verify(short,repo=repo,data=data,model_id='model',consumer_ids=[TARGET])==candidate()['sha256']
    assert c.verify(params,repo=repo,data=data,model_id='model',consumer_ids=[TARGET])==candidate()['sha256']
    path=repo/'configs'/short['render_manifest']['relpath']
    doc=json.loads(path.read_bytes());doc['mapping'][0]['stable_id']='different'
    path.write_text(json.dumps(doc))
    short['render_manifest']['sha256']=c.digest(path)
    with pytest.raises(ValueError,match='map differs'):c.verify(short,repo=repo,data=data,model_id='model',consumer_ids=[TARGET])


@pytest.mark.parametrize('fault',['mode','missing_map','different_candidate','recursive','target'])
def test_three_arm_declaration_refuses_unbound_variations(release,fault):
    _,_,params=release;short=copy.deepcopy(params)
    short.update(rendering=r.SHORT,render_manifest={'relpath':'probes/map.json','sha256':'a'*64})
    parent={**copy.deepcopy(params),'target_arm':'learned','compact_params':short}
    c.validate_params(parent)
    if fault=='mode':short['rendering']='other'
    elif fault=='missing_map':del short['render_manifest']
    elif fault=='different_candidate':short['candidate']['sha256']='0'*64
    elif fault=='recursive':short['compact_params']={}
    else:short['target_arm']='learned'
    with pytest.raises(ValueError):c.validate_params(parent)

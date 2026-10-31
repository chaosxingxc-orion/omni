import json
import pytest
pytest.importorskip('openjiuwen')
from test_llama_server_transport import _spawn_live_matching_process
from test_tfrl_seed_execution import group
from test_decoding_variants_live import _decoding_harness,_requests,_run,seam
from speech_aware_evidence_acquisition.rails.trajectory_seeds import resolve
from speech_aware_evidence_acquisition.rails.native_episode_receipt import validate_supply_variant_receipt


def test_real_shared_runner_delivers_seed_only_candidates_and_valid_receipt(tmp_path):
    p,b,g=group();h=_decoding_harness(tmp_path,g,rows=2)
    summary,close=_run(h)
    assert close['request_count']==14
    requests=_requests(h)
    for result in summary.results:
        row=[r for r in requests if r['sample_id']==result.sample_id]
        assert len(row)==7
        candidates,twin,lock=row[1:5],row[5],row[6]
        for k,request in enumerate(candidates):
            override=resolve(g.arm(f's{k+1}').decoding,result.sample_id)
            assert request['decoding_params']=={**lock['decoding_params'],**override}
            for key in ('speech_feature_parts','speech_packet_sha256','supplied_evidence','task_instruction'):
                assert request[key]==lock[key]
        normalized=[{k:v for k,v in r['decoding_params'].items() if k!='seed'} for r in candidates]
        assert all(x==normalized[0] for x in normalized)
        assert len({r['decoding_params']['seed'] for r in candidates})==4
        assert twin['decoding_params']==lock['decoding_params']
        validate_supply_variant_receipt(result.supply_variants,arm_ids=g.arm_ids,delivered_arm_id='lock',
            phase_id='solver',retrieved_value_ids=result.final_query_state.retrieved_value_ids,arm_visits='first')


def test_request_seam_resolves_recipe_and_fallback_keeps_seed(seam):
    make,request=seam;p,b,g=group()
    adapter=make(solver_arm_decoding={a.arm_id:a.decoding for a in g.arms if a.decoding},solver_arm_decoding_role_id='solver')
    req=request('solver','s1','seed-c1')
    body=adapter.prepare(req).rendered_body()['decoding_params']
    assert body['seed']==resolve(g.arm('s1').decoding,'s1')['seed']
    assert 'trajectory_index' not in body and 'trajectory_seed_base' not in body

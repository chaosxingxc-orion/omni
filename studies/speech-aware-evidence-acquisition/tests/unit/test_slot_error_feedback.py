import copy
from speech_aware_evidence_acquisition.rails.slot_error_feedback import group_error_feedback


def fixture():
    ref={'scenario':'play','action':'music','entities':[
        {'type':'artist_name','filler':'artist'}, {'type':'song_name','filler':'song'}]}
    pred={**copy.deepcopy(ref),'entities':[{'type':'artist_name','filler':'artist'}]}
    return {'group_sha256':'a'*64,'evidence':{'reference':ref,'rollouts':[
        {'trajectory_id':'t0','prediction_observed':True,'scored_prediction':pred}]}}


def test_missing_song_not_erased_by_better_partial_reward():
    group=fixture();before=copy.deepcopy(group)
    d=group_error_feedback(group);row=d['trajectories'][0]
    assert row['missing']==[{'type':'song_name','filler':'song','count':1}]
    assert row['intent_exact'] and not row['complete_frame_exact'] and not row['extra']
    assert not d['official_score_recomputed'] and group==before


def test_duplicate_slots_count_as_extra_and_wrong_boundary_kept():
    g=fixture();p=g['evidence']['rollouts'][0]['scored_prediction']
    p['entities'] += [{'type':'artist_name','filler':'artist'},{'type':'song_name','filler':'the song'}]
    r=group_error_feedback(g)['trajectories'][0]
    assert r['extra']==[{'type':'artist_name','filler':'artist','count':1},
                        {'type':'song_name','filler':'the song','count':1}]
    assert r['missing']==[{'type':'song_name','filler':'song','count':1}]


def test_unavailable_is_not_empty_prediction():
    g=fixture();g['evidence']['rollouts'][0]['scored_prediction']=None
    r=group_error_feedback(g)['trajectories'][0]
    assert r['available'] is False and 'missing' not in r and 'complete_frame_exact' not in r


def test_empty_slots_and_wrong_intent_are_separate():
    g=fixture();g['evidence']['reference']['entities']=[]
    p=g['evidence']['rollouts'][0]['scored_prediction'];p['entities']=[];p['action']='query'
    r=group_error_feedback(g)['trajectories'][0]
    assert not r['missing'] and not r['extra'] and not r['intent_exact'] and not r['complete_frame_exact']

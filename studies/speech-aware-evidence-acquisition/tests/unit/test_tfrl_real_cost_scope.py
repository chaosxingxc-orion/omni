from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/flights/nx'))
from tfrl_current_seed_cells import discard_synthetic_cost_forecasts


def test_real_plan_keeps_exact_counts_without_synthetic_forecast():
    source={'rows':313,'k':4,'contacts_per_row':7,'includes_twin_and_lock':True,
        'calls':2191,'calls_with_fallback_ceiling':4382,'audio_seconds':123,
        'tokens':9579052,'host_tok_s':1422.59,'host_wall_hours':1.87,
        'card_hours':14.96,'median_completion_tokens':4372,'margin':1.3}
    plan={'budget':{'program_c7':source},'row_plan':[{'row_id':'retained'}]}
    result=discard_synthetic_cost_forecasts(plan)
    b=result['budget']['program_c7']
    assert b['calls']==2191 and b['calls_with_fallback_ceiling']==4382
    assert b['audio_seconds']==123 and b['rows']==313
    assert not b['cost_forecast_available']
    assert not ({'tokens','host_tok_s','host_wall_hours','card_hours','median_completion_tokens','margin'} & b.keys())
    assert result['row_plan']==[{'row_id':'retained'}]

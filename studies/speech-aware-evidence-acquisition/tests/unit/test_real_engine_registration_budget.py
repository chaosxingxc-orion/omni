import math
from speech_aware_evidence_acquisition.contracts import ExecutionPlan


def test_real_audio_budget_rounds_up_to_integral_contract():
    seconds=4*1473.5844375
    plan=ExecutionPlan(run_id='SAEA-REAL-BUDGET-TEST',attempt_id='attempt-1',carrier_lock_key='slurp',split_identity_hash='a'*64,protocol_hash='b'*64,execution_profile='bounded-discovery-probe',split_role='discovery',planned_model_calls=2000,planned_gpu_hours=16.,planned_speech_audio_seconds=math.ceil(seconds))
    assert plan.validate().planned_speech_audio_seconds==5895

P position reward candidate

The scorer is the original unified-learning/scoring.py module, published here as p_position_scoring.py for review. It is not a standalone trainer. Apply it within the existing unified learning namespace. The regression test requires the original fixtures, Clock, kernel, and learning configuration.

New definition: component-definitions-v3-p-position. Select metric_weights.P = {"position_reward_v1": 1.0}. Correct rank 1/2/3 scores 1/0.75/0.5; absent scores 0. Historical component-definitions-v2 continues to use coverage_at3 only. Candidate P_ranking auxiliary coefficients are zero. SFT is not implemented in this change.

Validation: 53 tests passed against source and sealed candidate in the fixed CPU image, including actual scorer-to-GRPO evaluation. No optimizer update or DEV improvement is claimed.

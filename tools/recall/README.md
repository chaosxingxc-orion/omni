# Incremental fixed-K insertion screening

`IncrementalTopK` uses frozen score arrays, source groups, per-query excluded groups, legal labels and fixed description ranks. Call `reset(current_source_orders)` after each accepted insertion, then `trial(bank_index)` to screen real TRAIN insertions. `primary_metric` is `question` by default; `union` is also supported. Acceptance and candidate proposal stay in the caller. It does not encode data or call models.

The retained K+1 prefix gives exact question/union hit deltas, label-occurrence changes and repair margins. Score ties use original index ascending. Source exclusion and source deduplication precede TopK; legal label filtering follows it. `verify_reference` compares the prefixes to an existing `source_order/retrieve` implementation and checks aliases, ties and legal masks. Measured scope and limits: [first-pass results](../../docs/results/Recall-Minimum-Question-First-Pass.md).

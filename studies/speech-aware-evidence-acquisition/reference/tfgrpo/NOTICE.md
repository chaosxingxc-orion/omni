# Training-Free GRPO attribution

The prompt strings in `rails/tfgrpo_prompts.py` are copied from TencentCloudADP/youtu-agent, `utu/prompts/practice/experience.yaml`, at commit `c2caa539f4c95ae1c39ed24dc8a99cb3651e1d5d` (main integration), under the included MIT license. The original YAML blocks are represented as Python strings without a template-engine dependency. Source SHA256 is embedded beside the strings.

Paper: https://arxiv.org/abs/2510.08191. Paper branch: https://github.com/TencentCloudADP/youtu-agent/tree/training_free_GRPO. The pinned main integration is distinct from the paper-branch revision.

The local adapter follows summary, group advantage, group update and batch update stages from `utu/practice/experience_updater.py`. It injects an externally metered same-model callable, uses binary rewards, refuses malformed/unknown outcomes instead of upstream retry/skip behavior, preserves stable IDs and publishes immutable candidate versions. UPDATE and NONE are the upstream spellings of Modify and Keep. Explicit learning/evaluation context purpose and strict operation/size validation are local adaptations. No claim of a reproduced empirical result is made.


## Integration contract and current limits

`ExperienceLearner.learn` accepts verified learning groups with source model, policy and parent-experience identities. This adapter checks identity equality; the caller must verify the supplied trajectory, reward and source receipt contents against sealed artifacts before invoking it. The supplied `metered_call` must use the existing registered frozen-core runtime and return its actual receipt identity and usage. There is deliberately no HTTP client, registration bypass or automatic execution entry point.

Each batch has a process lock, immutable input binding, intent-before-send and observed-response checkpoint. Restart reuses observed responses; an intent without a response is UNKNOWN and cannot be retried automatically. Invalid observed model output is retained and not resampled. `accounting()` includes attempted contacts, known costs and unknown reservations even for failed batches. Memory candidate publication is atomic. A hard character/item bound is an engineering limit, not a tokenizer measurement; production FIT still enforces exact token/context limits.

Candidates require source-leakage review, independent development acceptance and ordinary release registration before deployment. `experience_context` provides the actual next-query text. Its default held-out-evaluation mode excludes queries that contributed to the candidate (including ancestors). Explicit learning mode allows the original method's repeated training queries and records reuse in subsequent batch results; such queries must never be reported as held-out. Cross-fold evaluation ownership remains the orchestrator's responsibility. Same-policy, binary-reward groups are the baseline; shared-Perception data must use the explicit adaptation label. Real rollout orchestration, metered runtime binding, registered release and development acceptance remain outside this module and are not claimed complete.

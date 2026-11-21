"""Diagnostic-only V1 processor: choose sealed actions *after* raw logprobs."""
import torch
from vllm.v1.sample.logits_processor.builtin import process_dict_updates
from vllm.v1.sample.logits_processor.interface import LogitsProcessor


class ReplayActions(LogitsProcessor):
    @classmethod
    def validate_params(cls, params):
        extra = params.extra_args or {}
        path = extra.get('decode_action_ids')
        if (set(extra) != {'decode_action_ids'} or type(path) is not list or not path or
                len(path) != params.max_tokens or any(type(t) is not int or t < 0 for t in path)
                or params.logprobs != 2 or params.n != 1):
            raise ValueError('decode replay requires explicit actions, matching length and raw top-1')

    def __init__(self, vllm_config, device, is_pin_memory):
        if vllm_config.model_config.logprobs_mode != 'raw_logprobs' or vllm_config.speculative_config:
            raise ValueError('raw logprobs and ordinary autoregressive decode required')
        self.rows = {}

    def update_state(self, batch_update):
        def new(params, prompt, output):
            self.validate_params(params)
            return tuple(params.extra_args['decode_action_ids']), output
        process_dict_updates(self.rows, batch_update, new)

    def is_argmax_invariant(self):
        return False

    def apply(self, logits):
        if len(self.rows) != logits.shape[0] or set(self.rows) != set(range(len(logits))):
            raise ValueError('incomplete replay roster')
        chosen = []
        for i in range(len(logits)):
            path, output = self.rows[i]
            if len(output)>=len(path) or tuple(output)!=path[:len(output)] or path[len(output)]>=logits.shape[1]:
                raise ValueError('replay history or vocabulary mismatch')
            chosen.append(path[len(output)])
        # Sampler.forward has already computed raw_logprobs from the real logits.
        result = torch.full_like(logits, -float('inf'))
        result.scatter_(1, torch.tensor(chosen,device=logits.device)[:,None], 0.)
        return result

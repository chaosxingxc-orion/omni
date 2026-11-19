"""Bounded full-group operands, preserving the actual expert GEMM shape.

The HF observation is at the grouped-linear boundary; vLLM observes the actual
mm call. Neither boundary is described as an internal fused-kernel trace.
"""
def validate_policy(value, *, max_experts=2):
    if not (isinstance(value, list) and len(value) <= max_experts
            and value == sorted(set(value))
            and all(type(e) is int and 0 <= e < 128 for e in value)):
        raise ValueError('too many or invalid full projection experts')
    return value


def layout(value):
    return dict(shape=list(value.shape), stride=list(value.stride()),
                storage_offset=value.storage_offset(), dtype=str(value.dtype),
                device=str(value.device))


class Capture:
    def __init__(self, session, experts, *, max_bytes=24*1024**2,
                 max_experts=2, namespace='expert_down_full'):
        self.session = session
        self.experts = validate_policy(experts, max_experts=max_experts)
        self.namespace = namespace
        self.audit = dict(experts=self.experts, calls={}, bytes=0,
                          max_bytes=max_bytes, extra_compute=0, extra_collectives=0,
                          production_acceptance=False)
        session.audit[namespace] = self.audit

    def record(self, expert, phase, start, x, weight, output, *, routes=None,
               boundary='actual torch.mm'):
        if expert not in self.experts:
            return
        key = str(expert)
        calls = self.audit['calls']
        if phase == 'before':
            if key in calls or len(x) == 0:
                raise ValueError('one nonempty full-group down call per expert')
            # Check the entire group budget before recording any part of it.
            tensors = [x, weight, output] + ([] if routes is None else [routes])
            size = sum(t.numel()*t.element_size() for t in tensors)
            if self.audit['bytes']+size > self.audit['max_bytes']:
                raise ValueError('full down operand budget exceeded')
            calls[key] = dict(start=start, rows=len(x), boundary=boundary,
                              input=layout(x), weight=layout(weight), output=layout(output),
                              complete=False, reserved_bytes=size)
            self.audit['bytes'] += size
            for name, value in [('input', x), ('weight', weight), ('routes', routes)]:
                if value is not None:
                    self.session.tensor(f'{self.namespace}.{expert}.{name}', value)
        elif phase == 'after':
            if key not in calls or calls[key]['complete']:
                raise ValueError('full down output without one matching input')
            if (calls[key]['start'] != start or calls[key]['rows'] != len(x)
                    or calls[key]['output'] != layout(output)):
                raise ValueError('full down output contract changed')
            self.session.tensor(f'{self.namespace}.{expert}.output', output)
            calls[key]['complete'] = True
        else:
            raise ValueError('unknown full down observation phase')

    def validate(self):
        if (set(self.audit['calls']) != {str(e) for e in self.experts}
                or not all(c['complete'] for c in self.audit['calls'].values())):
            raise ValueError('incomplete full down expert observation')

"""Exercise the paired BF16 projection's input and trainable adapter gradients."""
def check(project, device, width=4096):
    import torch
    from torch.nn.functional import linear

    # A private generator keeps the model RNG and restored training state intact.
    generator = torch.Generator(device='cpu').manual_seed(971)
    def sample(*shape):
        return torch.randn(shape, generator=generator).bfloat16().to(device)
    x = sample(1, 129, width).requires_grad_()
    master = sample(8, width).float().requires_grad_()
    dy = sample(1, 129, 8)
    with torch.enable_grad():
        result = []
        for fn in (linear, project):
            y = fn(x, master.bfloat16())
            result.append(torch.autograd.grad(y, (x, master), grad_outputs=dy))
    report = {}
    for name, expected, actual in zip(('input','fp32_master'), *result, strict=True):
        if not expected.isfinite().all() or not expected.any() or not torch.equal(expected, actual):
            raise ValueError('LoRA A failed native gradient parity: ' + name)
        report[name] = dict(elements=actual.numel(),dtype=str(actual.dtype),differences=0)
    return report

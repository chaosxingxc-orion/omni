"""Scoped native audio convolution dispatch; preserve language NPU settings.

This fixes the measured internal-format dispatch mismatch. It does not replace
the audio tower, change weights/dtypes or claim end-to-end numerical acceptance.
The original eager synchronous worker owns execution and installation.
"""
from contextlib import contextmanager
from functools import wraps


@contextmanager
def native_conv_context():
    import torch_npu
    option=torch_npu._C._npu_getOption('ALLOW_INTERNAL_FORMAT')
    if option not in (b'enable',b'disable'):
        raise ValueError('explicit internal-format setting required')
    torch_npu._C._npu_setOption({'ALLOW_INTERNAL_FORMAT':'disable'})
    try:
        yield
    finally:
        torch_npu._C._npu_setOption({'ALLOW_INTERNAL_FORMAT':option.decode()})


def attach(tower):
    import torch
    # Single measured earliest operator; downstream convolutions are not yet
    # proved to need this change and retain their original dispatch.
    names=('conv2d1',)
    modules=[getattr(tower,name) for name in names]
    if any(not isinstance(m,torch.nn.Conv2d) for m in modules):
        raise ValueError('native audio conv2d1 required')
    if any(getattr(m,'_omni_native_conv_context',False) for m in modules):
        raise ValueError('audio convolution precision already owned')
    undo=[];audit=dict(calls={name:0 for name in names},restored=False)
    for name,module in zip(names,modules):
        original=module.forward;present='forward' in vars(module);old=vars(module).get('forward')
        def wrap(original,name):
            @wraps(original)
            def forward(*args,**kwargs):
                with native_conv_context():
                    result=original(*args,**kwargs)
                audit['calls'][name]+=1
                return result
            return forward
        installed=wrap(original,name);module.forward=installed;module._omni_native_conv_context=True
        undo.append((module,present,old,installed))
    def restore():
        if audit['restored']:return
        for module,present,old,installed in reversed(undo):
            if vars(module).get('forward') is not installed:
                raise ValueError('audio convolution owner changed')
            if present:module.forward=old
            else:del module.forward
            del module._omni_native_conv_context
        audit['restored']=True
    return audit,restore

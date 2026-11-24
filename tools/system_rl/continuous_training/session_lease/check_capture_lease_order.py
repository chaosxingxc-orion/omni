"""Exercise a pre-contact SSH failure and a lost HTTP reply using the production wrapper."""
import argparse
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch


def run(parents,code):
    sys.path[:0]=[str(parents/'fresh-cohort-successor'),str(parents/'collection-dispatch-integration'),
                  str(parents/'collection-scale-successor/code')]
    from importlib.util import spec_from_file_location,module_from_spec
    spec=spec_from_file_location('tested_capture_execution',code)
    mod=module_from_spec(spec);spec.loader.exec_module(mod)
    Registry,Accounted,Fence,Contact=mod.execution_modules()
    calls=[]
    class Ledger:
        def reserve_request(self,*args):calls.append('reserve')
        def observed(self,rid):calls.append('observed')
    class Target:
        def submit(self,*args):
            calls.append('http')
            if fail_http:raise ConnectionError('after HTTP entry')
            return 'response'
    from contextlib import contextmanager
    class Guard:
        @contextmanager
        def admission(self):
            calls.append('probe')
            # Use the exception class bound to the actual ContactFence.
            if fail_probe:raise Contact.submit.__globals__['LeaseLost']('before HTTP')
            yield
        def check_local(self):calls.append('check')
    lease=mod.CaptureLease({'resources':{'rank':'a'},'budget':{}},Path('.'))
    lease.Accounted=Accounted;lease.ContactFence=Contact;lease.registry=Ledger();lease.fence=Guard()
    with patch.object(mod,'request_cap',return_value=1):
        for fail_probe,fail_http,expected in [(True,False,['probe']),
             (False,True,['probe','check','reserve','http']),
             (False,False,['probe','check','reserve','http','observed'])]:
            calls.clear()
            try:ans=lease.transport(Target()).submit('r',{})
            except Exception as e:
                assert type(e).__name__==('NotExecuted' if fail_probe else 'ConnectionError'),e
            else:assert not(fail_probe or fail_http) and ans=='response'
            assert calls==expected,calls
    print('pre-contact failure: zero reservations; lost reply: retained reservation; success: accounted once')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('parents',type=Path);p.add_argument('source',type=Path)
    a=p.parse_args();run(a.parents,a.source)

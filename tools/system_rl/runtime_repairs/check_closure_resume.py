"""Exercise the sealed CPU recovery entry without dispatching any request."""
import argparse
import json
import runpy
import sys
from pathlib import Path
from unittest.mock import patch

p=argparse.ArgumentParser()
for key in ('release','config','output'):p.add_argument('--'+key,required=True)
a=p.parse_args()
import dependencies
import iteration
from run_package import write
original=dependencies.HERE
entry=Path(a.release)/'tree/parents/jev-system-rl/implementation/resume_closed.py'
with patch.object(iteration,'run',return_value={'dispatch':False}) as run:
    sys.argv=[str(entry),'--release',a.release,'--config',a.config]
    runpy.run_path(str(entry),run_name='__main__')
    run.assert_called_once_with(a.config)
assert dependencies.HERE==original
assert Path(sys.modules['transport'].__file__).parent==entry.parent
assert Path(iteration.__file__).parent==original
write(a.output,dict(passed=True,checks=['sealed transport loaded','original iteration executed',
    'original numerical namespace preserved'],model_calls=0))

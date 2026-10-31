"""CPU subprocess fault injection for queue supervisor recovery."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

import card_queue as cq
import card_queue_guardian as guardian


class GuardianTests(unittest.TestCase):
    def test_corrupt_state_refuses_restart(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'state.json'
            p.write_text('{')
            with self.assertRaises(cq.QueueError): guardian.recovery_action(p)

    def test_terminal_states_are_not_resumed(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'state.json'
            for state in ('done', 'failed', 'stopped'):
                p.write_text(json.dumps({'schema': cq.STATE_SCHEMA, 'state': state}))
                self.assertEqual(guardian.recovery_action(p), 'terminal')

    def test_hangup_and_killed_runner_do_not_duplicate_job(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); qdir=root/'q'; qdir.mkdir(); trace=root/'trace'
            job_script=root/'job.py'
            job_script.write_text('import sys,time\nfrom pathlib import Path\np=Path(sys.argv[1])\nwith p.open("a") as f: f.write(sys.argv[2]+"\\n")\ntime.sleep(float(sys.argv[3]))\nprint("COMPLETE",flush=True)\n')
            jobs=[]
            for name,delay in [('one',2),('two',0.1)]:
                jobs.append(dict(id=name,cmd=[sys.executable,str(job_script),str(trace),name,str(delay)],
                    cwd=d,log=str(root/(name+'.log')),timeout_s=20,done_marker='^COMPLETE$',on_fail='stop'))
            queue=root/'queue.json';queue.write_text(json.dumps(dict(schema=cq.QUEUE_SCHEMA,jobs=jobs)))
            proc=subprocess.Popen([sys.executable,guardian.__file__,'--queue',str(queue),'--queue-dir',str(qdir),
                '--poll-s','0.05','--restart-delay','0.05'],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,start_new_session=True)
            try:
                deadline=time.monotonic()+15
                while time.monotonic()<deadline:
                    state=cq.read_json(str(qdir/'state.json')) or {}
                    if state.get('jobs',{}).get('one',{}).get('pid') and trace.exists(): break
                    time.sleep(0.05)
                else: self.fail('first job did not start')
                worker=state['jobs']['one']['pid']
                os.kill(proc.pid,signal.SIGHUP)
                os.kill(state['runner_pid'],signal.SIGHUP)
                time.sleep(0.1)
                self.assertIsNone(proc.poll())
                self.assertTrue(cq.proc_alive(worker))
                os.kill(state['runner_pid'],signal.SIGKILL)
                self.assertEqual(proc.wait(timeout=20),0)
                self.assertEqual(trace.read_text().splitlines(),['one','two'])
                self.assertEqual(cq.read_json(str(qdir/'state.json'))['state'],'done')
                self.assertEqual(cq.read_json(str(qdir/'guardian.json'))['restarts'],1)
            finally:
                if proc.poll() is None: proc.terminate();proc.wait(timeout=5)
                state=cq.read_json(str(qdir/'state.json')) or {}
                for rec in state.get('jobs',{}).values():
                    if cq.proc_alive(rec.get('pid'),rec.get('token')): cq.killpg(rec['pid'],signal.SIGKILL)
                if cq.proc_alive(state.get('runner_pid')): os.kill(state['runner_pid'],signal.SIGTERM)


if __name__=='__main__': unittest.main()

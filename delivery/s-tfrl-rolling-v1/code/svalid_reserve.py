"""Evaluation-only original1080 C/D reserve; admission gated by actual shortage."""
import fcntl,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'delivery/s-tfrl-rolling-v1'
sys.path.insert(0,str(ROOT/'delivery/s-single-intent-fewshot-v1/code'))
from common import read,save,lines
import producer

def main():
    lock=(OUT/'svalid-producer.guard').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    window=read(OUT/'REQUESTS/QUEUE-DEFICIT.json')['window']
    save(OUT/'SVALID-PRODUCER-START.json',dict(pid=os.getpid(),ppid=os.getppid(),at=time.time(),agent=False,
        evaluation_only=True,teacher_update_allowed=False,admission='only after preparedTRAIN committed and measured coverage shortage'))
    control=dict(task_id='train-s_valid-single-v1',prepared='delivery/s-tfrl-rolling-v1/train-supply/train-s_valid-single-v1',
        config='delivery/s-tfrl-rolling-v1/train-supply/train-s_valid-single-v1/single_intent_no_examples/native-UNBOUND.json',
        runner='delivery/s-tfrl-rolling-v1/code/run_n.py',bind_script='delivery/s-tfrl-rolling-v1/code/bind_n.py',
        preferred_lane='C',parent_version='prepared-v5-dynamic/single_intent_no_examples',method='prelocked_validation_C',
        purpose='independent S_valid1080 evaluation only; no teacher or tuning',evaluation_only=True,teacher_update_allowed=False)
    few=control|dict(task_id='train-s_valid-single-fewshot-v1',prepared='delivery/s-tfrl-rolling-v1/train-paired-fewshot/train-s_valid-single-fewshot-v1',
        config='delivery/s-tfrl-rolling-v1/train-paired-fewshot/train-s_valid-single-fewshot-v1/single_intent_fewshot/native-UNBOUND.json',
        control_prepared=control['prepared'],preferred_lane='D',method='prelocked_validation_D_score_top2',parent_version=control['task_id'])
    while not (ROOT/control['prepared']/'CPU-PREPARED.json').exists():time.sleep(10)
    # Strong downstream isolation is checked by run_n and independent scoring.
    for row in lines(ROOT/control['prepared']/'offline/gold.private.jsonl'):
        assert row['evaluation_only'] and row['teacher_update_allowed'] is False
    # Reuse accepted immutable receipts; restarting a watcher must not rerun budgets.
    for task in (control,few):
        receipt=OUT/'evidence'/task['task_id']
        assert (receipt/'CPU-VERIFIED.json').exists()
        cfg=read(ROOT/task['config'])
        assert read(receipt/'budget.json')['manifest_sha256']==cfg['manifest_sha256']
    save(OUT/'SVALID-CPU-RESERVE.json',dict(at=time.time(),tasks=[control,few],samples=1080,branches=6480,
        status='FIXED_IMAGE_VERIFIED_UNPUBLISHED_UNTIL_SHORTAGE',evaluation_only=True,teacher_update_allowed=False))
    required={t['task_id'] for t in read(OUT/'config/train-paired-fewshot-tasks.json')['tasks']}
    while time.time()<window['end_epoch']:
        published={t['task_id'] for t in read(OUT/'READY-TASKS.json')['tasks']}
        reserve=read(ROOT/'delivery/s-single-intent-execution-v1/rolling-state/RESERVE.json')
        fresh=0<=time.time()-reserve['at']<=900
        shortage=fresh and reserve['estimated_queue_wall_hours']<max(0,(window['end_epoch']-time.time())/3600)
        if required<=published and shortage:
            for task in (control,few):producer.publish(task)
            save(OUT/'SVALID-ADMISSION-BASIS.json',dict(at=time.time(),reserve=reserve,all_TRAIN_paired_READY=True,
                evaluation_only=True,teacher_update_allowed=False));return
        save(OUT/'SVALID-WAITING.json',dict(at=time.time(),shortage=shortage,reserve_fresh=fresh,reserve_at=reserve['at'],TRAIN_paired_missing=sorted(required-published),
            status='CPU_VERIFIED_WAITING_SCIENTIFIC_ADMISSION',evaluation_only=True))
        time.sleep(600)

if __name__=='__main__':main()

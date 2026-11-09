"""Prepare a configured CPU batch ahead of NPU admission; reuse completed work."""
import argparse
import concurrent.futures
import fcntl
import json
import time
import traceback
from pathlib import Path
import producer
from common import read, save

def prepare_one(task):
    name = task['task_id']
    guard = producer.OUT / 'evidence' / name / 'prepare-plan.guard'
    guard.parent.mkdir(parents=True, exist_ok=True)
    with guard.open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        published = {t['task_id']: t for t in read(producer.OUT / 'READY-TASKS.json')['tasks']}
        if name in published:
            return dict(task_id=name, status='EXISTING_PUBLICATION_PRESERVED', published_status=published[name].get('status'))
        prior = producer.OUT / 'evidence' / name / 'CPU-PLAN-FAILED.json'
        if prior.exists():
            return dict(task_id=name, status='FAILED_ATTEMPT_REQUIRES_REVIEW')
        try:
            producer.prepare(task)
            producer.verify(task)
            producer.publish(task)
            return dict(task_id=name, status='READY')
        except Exception as error:
            record = dict(task_id=name, at=time.time(), status='FAILED', error=str(error), traceback=traceback.format_exc(), model_calls=0)
            save(prior, record)
            return record

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    path = Path(args.plan).resolve()
    plan = read(path)
    tasks = plan['tasks']
    ids = [t['task_id'] for t in tasks]
    assert len(ids) == len(set(ids)), 'duplicate logical task in CPU plan'
    for task in tasks:
        assert Path(task['task_id']).name == task['task_id'], 'task id must be a basename'
        for key in ('config', 'prepared'):
            assert (producer.ROOT / task[key]).resolve().is_relative_to(producer.ROOT), 'path outside workspace'
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(2, max(1, plan.get('cpu_workers', 2)))) as pool:
        results = list(pool.map(prepare_one, tasks))
    result = dict(at=time.time(), plan=str(path.relative_to(producer.ROOT)), results=results, NPU_calls=0, teacher_calls=0)
    output = producer.ROOT / plan['receipt']
    assert output.resolve().is_relative_to(producer.ROOT)
    save(output, result)
    print(json.dumps(result))

if __name__ == '__main__':
    main()

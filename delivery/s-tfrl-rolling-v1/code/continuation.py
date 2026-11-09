"""Detached CPU continuation, bounded fixed jobs and terminal feedback until deadline."""
import fcntl,json,os,subprocess,sys,time,traceback
from pathlib import Path
import producer as p
from common import read,save,sha,lines

def train_task(name):
    return dict(task_id=name,prepared='delivery/s-tfrl-rolling-v1/train-supply/'+name,
        config='delivery/s-tfrl-rolling-v1/train-supply/'+name+'/single_intent_no_examples/native-UNBOUND.json',
        runner='delivery/s-tfrl-rolling-v1/code/run_n.py',bind_script='delivery/s-tfrl-rolling-v1/code/bind_n.py',
        preferred_lane='B',parent_version='prepared-v5-dynamic/single_intent_no_examples',
        method='fixed_TRAIN_single_projection',purpose='new S error coverage; reviewed_hold never teacher update')

def source_packets(folder):
    path=folder/'source-proof.private.jsonl';rows=lines(path)
    if all('source-packets' in r['original_packet'] for r in rows):return
    for row in rows:
        src=p.ROOT/row['original_packet'];dst=folder/'source-packets'/(sha(row['sample_id'].encode())+'.private.json')
        dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():os.link(src,dst)
        row['original_source_packet']=row['original_packet'];row['original_packet']=str(dst.relative_to(p.ROOT))
    save(path,rows,jsonl=True)

def main():
    lock=(p.OUT/'continuation.guard').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    window=read(p.OUT/'REQUESTS/QUEUE-DEFICIT.json')['window']
    save(p.OUT/'CONTINUATION-START.json',dict(pid=os.getpid(),ppid=os.getppid(),at=time.time(),window=window,
        agent=False,poll_seconds=600,service_actions=False,model_resend=False))
    tasks=[t for t in read(p.OUT/'PRODUCTION-PLAN.json')['tasks'] if t['task_id'].startswith('grpo-')]
    tasks.append(read(p.OUT/'config/train-s-fit-task.json'))
    parent=p.OUT/'train-supply/train-eligible_remainder-single-v1'
    source_packets(parent)
    shardcfg=p.OUT/'config/train-remainder-shards.json'
    if not shardcfg.exists():subprocess.run([sys.executable,'-B',str(p.OUT/'code/shard_train.py')],check=True)
    tasks.extend(read(shardcfg)['tasks'])
    plan=read(p.OUT/'PRODUCTION-PLAN.json');known={t['task_id'] for t in plan['tasks']}
    plan['tasks'].extend(t for t in tasks if t['task_id'] not in known)
    actual_requests=sum(read(p.ROOT/t['config'])['expected_rows'] for t in plan['tasks'] if (p.ROOT/t['config']).exists())
    plan.update(at=time.time(),materialized_requests=actual_requests,estimated_lane_hours_materialized_only=actual_requests*4/3600,
        coverage_12h='ESTIMATE_ONLY; no extra seeds; original4814 TRAIN and eight-seed reserve; actual throughput may need more informative supply')
    save(p.OUT/'PRODUCTION-PLAN.json',plan)
    completed=set();failed=set();feedback_done=False
    while time.time()<window['end_epoch']:
        progress=False
        for task in tasks:
            tid=task['task_id']
            if tid in completed or tid in failed:continue
            folder=p.ROOT/task['prepared']
            if not (folder/'CPU-PREPARED.json').exists():continue
            if (p.OUT/'evidence'/tid/'CPU-VERIFIED.json').exists():
                if tid not in {t['task_id'] for t in read(p.OUT/'READY-TASKS.json')['tasks']}:p.publish(task)
                completed.add(tid);continue
            try:
                if 'dataset_partition' in read(p.ROOT/task['config']):source_packets(folder)
                receipt=p.OUT/'evidence'/tid
                if (receipt/'LAUNCH.json').exists() and not (receipt/'WAIT.json').exists():
                    cid=read(receipt/'LAUNCH.json')['stdout'].strip()
                    p.ssh(['sudo','-n','docker','wait',cid],capture_output=True,text=True)
                p.verify(task);p.publish(task);completed.add(tid);progress=True
            except Exception as e:
                failed.add(tid);save(p.OUT/'evidence'/tid/'CONTINUATION-FAILED.json',dict(at=time.time(),error=str(e),traceback=traceback.format_exc(),model_calls=0))
        feedback=p.OUT/'REQUESTS/GRPO-FEEDBACK-READY.json'
        if feedback.exists() and not feedback_done:
            d=read(feedback);data=read(p.ROOT/d['reflector_input'])
            save(p.OUT/'experience/actual-group-feedback.json',dict(at=time.time(),input=d,feedback=data,
                curator_status='ACTUAL_ADVANTAGES_READY_FOR_BOUNDED_EXPERIENCE_REVIEW',not_weight_training=True))
            feedback_done=True;progress=True
        save(p.OUT/'CPU-CONTINUATION-STATE.json',dict(at=time.time(),completed=sorted(completed),failed=sorted(failed),
            awaiting=[t['task_id'] for t in tasks if t['task_id'] not in completed|failed],
            feedback_consumed=feedback_done,status='ACTIVE_TO_WINDOW_END',window_end=window['end_epoch']))
        if not progress:time.sleep(min(600,max(0,window['end_epoch']-time.time())))
    save(p.OUT/'CPU-CONTINUATION-END.json',dict(at=time.time(),status='WINDOW_ENDED_CPU_NO_NEW_ADMISSION',completed=sorted(completed),failed=sorted(failed)))

if __name__=='__main__':main()

"""Event-driven live report owner; preserve legacy collector output as evidence."""
import ctypes,fcntl,json,os,select,struct,time
from pathlib import Path
R=Path(__file__).resolve().parents[3];D=R/'delivery/s-single-intent-execution-v1';O=D/'rolling-state';P=R/'delivery/s-tfrl-rolling-v1';MARK='<!-- rolling-four-lane-report-v1 -->'
def read(p,default=None):
 try:return json.loads(p.read_bytes())
 except (OSError,ValueError):return {} if default is None else default
def save(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.report-tmp')
 with tmp.open('w') as f:f.write(d if isinstance(d,str) else json.dumps(d,ensure_ascii=False,indent=2)+'\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def alive(pid):
 try:os.kill(int(pid),0);return True
 except (OSError,ValueError,TypeError):return False
def refresh():
 window=read(R/'delivery/s-rolling-12h-v1/WINDOW.json');reserve=read(O/'RESERVE.json');driver=read(O/'DRIVER-START.json');heartbeat=read(O/'HEARTBEAT.json');resource=read(O/'RESOURCE-LATEST.json');lanes={}
 for lane in 'ABCD':
  current=read(O/'lanes'/(lane+'-CURRENT.json'));live=read(O/'lanes'/(lane+'-LIVE.json'));launch=current.get('launch',{})
  lanes[lane]=dict(current=current.get('status'),task_id=launch.get('task_id','original-'+lane),container_id=launch.get('container_id'),pid=launch.get('pid'),endpoint=launch.get('runtime',{}).get('endpoint',{'C':'http://127.0.0.1:8204','D':'http://127.0.0.1:8206'}.get(lane)),receipt=str((O/'lanes'/(lane+'-CURRENT.json')).relative_to(R)),live=live,blocked=read(O/'lanes'/(lane+'-BLOCKED.json')))
 pids={n:read(O/f).get('pid') for n,f in [('driver','DRIVER-START.json'),('resource','RESOURCE-MONITOR-START.json'),('group_collector','GROUP-COLLECTOR-START.json')]};pids['producer_continuation']=read(P/'CONTINUATION-START.json').get('pid');pids['producer_paired']=read(P/'TRAIN-PAIRED-PRODUCER-START.json').get('pid');pids['producer_Svalid']=read(P/'SVALID-PRODUCER-START.json').get('pid')
 status='WINDOW_ADMISSION_CLOSED_DRAINING' if time.time()>=window['end_epoch'] else 'FOUR_LANE_SCHEDULER_ACTIVE' if alive(driver.get('pid')) and all(heartbeat.get('workers',{}).values()) and len(heartbeat.get('workers',{}))==4 else 'ATTENTION_CHECK_DRIVER_AND_LANES'
 known_tokens=sum(d['live'].get('usage',{}).get('total_tokens',0) for d in lanes.values());known_card_hours=sum(d['live'].get('card_hours',0) for d in lanes.values())
 ready=dict(known_token_cost=known_tokens,known_card_hours=known_card_hours,owner='rolling-four-lane-report-v1',status=status,at=time.time(),window=window,pids={k:dict(pid=v,alive=alive(v)) for k,v in pids.items()},heartbeat=heartbeat,lanes=lanes,reserve=reserve,resource_sample_at=resource.get('at'),cards=16,instances=4,TP=4,coverage_12h='NOT_YET_PROVEN',money=None,limits=['snapshot64running does not prove AIcore80%','CPU pending is not ready','12h window has not yet completed'])
 for filename in ['REPORT.md','READY.json']:
  path=D/filename
  if path.exists():
   content=path.read_text()
   owned=MARK in content if filename.endswith('.md') else read(path).get('owner')=='rolling-four-lane-report-v1'
   if not owned:
    history=O/'legacy-collector-reports';history.mkdir(exist_ok=True);save(history/(str(time.time_ns())+'-'+filename),content)
 stamp=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime());rows=[]
 for lane,d in lanes.items():
  l=d['live'];spb=l.get('seconds_per_branch');eta=l.get('remaining_seconds_estimate');q=reserve.get('lanes',{}).get(lane,{})
  rows.append(f"|{lane}|{d['task_id']}|{l.get('completed_branches','?')}/{l.get('total_branches','?')}|{l.get('completed_samples','?')}/{l.get('total_samples','?')}|{spb:.3f}"+f"|{eta/3600:.2f}|"+str(q.get('ready_reserved','?'))+' / '+str(q.get('cpu_preparing','?'))+'|'+str(l.get('scheduler',{}).get('running','?'))+'|'+str(l.get('at','?'))+'|' if spb and eta is not None else f"|{lane}|{d['task_id']}|待核|待核|待核|待核|待核|待核|待核|")
 text=MARK+'\n\n'+f'四lane滚动执行中；更新时间 {stamp}。唯一窗口 {window.get("start_utc")} 至 {window.get("end_utc")}，不重置，到期停止准入并drain在途。\n\n'
 text+=f'实际driver PID {driver.get("pid")}（live={alive(driver.get("pid"))}），四worker状态 {heartbeat.get("workers")}。原C/D保持在途，不重跑。16卡 / 4实例 / TP4；四路有任务不代表AIcore满算力。\n\n'
 text+='|lane|task|分支完成/总量|整样本完成/总量|秒/分支|估计剩余h|ready / CPU待处理|running快照|快照epoch|\n|---|---|---|---|---|---|---|---|---|\n'+'\n'.join(rows)+'\n\n'
 text+=f"ready {reserve.get('ready_tasks')}，估计 {reserve.get('estimated_ready_lane_hours',0):.2f} lane h；在途余量估计 {reserve.get('estimated_running_residual_lane_hours',0):.2f} lane h，合四路约 {reserve.get('estimated_queue_wall_hours',0):.2f} h。窗口剩余 {max(0,window['end_epoch']-time.time())/3600:.2f} h。CPU待处理 {reserve.get('cpu_preparing')} 项，仅预算验收后才能计ready；最短lane reserve {reserve.get('minimum_simultaneous_ready_reserve')} ready + {reserve.get('minimum_cpu_reserve')} CPU待处理。尚未完成或证明12h覆盖；时长为估计，不能当承诺。\n\n"
 spread=reserve.get('estimated_ready_lane_hours_range',[0,0]);text+=f"ready时长范围 {spread[0]:.2f}–{spread[1]:.2f} lane h（实测各lane速度，跨策略/96后继仍有不确定性），最快估计见上。\n\n"
 for lane in 'ABCD':
  ld=lanes[lane];text+=f"{lane} 容器 {ld['container_id']}，PID {ld['pid']}，回执 {ld['receipt']}；next staged={reserve.get('lanes',{}).get(lane,{}).get('next_staged')}。\n\n"
 text+=f"当前快照已观察tokens {known_tokens:,}、卡时 {known_card_hours:.3f}；未含所有历史失败/CPU总成本，金额未知。\n\n"
 text+=f"另有{reserve.get('conditional_cpu_verified',0)}个已固定镜像预算的条件CPU后备，共{reserve.get('conditional_cpu_verified_branches',0)}分支，尚未准入，不计ready；fresh容量不足时自动发布。\n\n"
 text+='执行状态、launch回执、历史成本在 rolling-state/；READY-TASKS只由producer写。CPU评分/归档独立接续；unknown不重发，known failure留证后需新attempt。typed/strict/official并报，all/eligible/hold/evaluation单列并隔离教师输入；货币成本未知，已用tokens/card_hours见各lane快照。\n\n'
 text+='detached进程 '+', '.join(k+'='+str(v) for k,v in pids.items())+'。固定镜像CPU评分集成回放PASS见 rolling-state/SCORE-EVALUATION-ISOLATION.json（工程fixture，非科学成绩）。资源证据见 rolling-state/RESOURCE-LATEST.json。公开push由主线程。\n'
 save(D/'READY.json',ready);save(D/'REPORT.md',text);save(O/'REPORT-MONITOR-HEARTBEAT.json',dict(pid=os.getpid(),at=time.time(),status=status))
lock=(O/'report-monitor.guard').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
save(O/'REPORT-MONITOR-START.json',dict(pid=os.getpid(),ppid=os.getppid(),at=time.time(),cadence=600,legacy_collectors_preserved=True))
libc=ctypes.CDLL(None);fd=libc.inotify_init1(0x800|0x80000)
for path in [O,O/'lanes',D]:libc.inotify_add_watch(fd,os.fsencode(path),0x80|0x100|0x8)
refresh();last=time.monotonic()
while time.time()<read(R/'delivery/s-rolling-12h-v1/WINDOW.json')['end_epoch']+7200:
 changed=False
 if select.select([fd],[],[],min(30,max(0,600-(time.monotonic()-last))))[0]:
  raw=os.read(fd,65536)
  names=[];offset=0
  while offset+16<=len(raw):
   _,_,_,length=struct.unpack_from('iIII',raw,offset);names.append(raw[offset+16:offset+16+length].rstrip(b'\0'));offset+=16+length
  changed=any(n in {b'RESERVE.json',b'HEARTBEAT.json',b'DRIVER-START.json',b'RESOURCE-LATEST.json'} or (len(n)>1 and n[:1] in [b'A',b'B',b'C',b'D'] and n[1:] in {b'-LIVE.json',b'-BLOCKED.json',b'-CURRENT.json'}) for n in names)
  if b'REPORT.md' in raw and MARK not in (D/'REPORT.md').read_text():changed=True
  if b'READY.json' in raw and read(D/'READY.json').get('owner')!='rolling-four-lane-report-v1':changed=True
 if changed or time.monotonic()-last>=600:refresh();last=time.monotonic()

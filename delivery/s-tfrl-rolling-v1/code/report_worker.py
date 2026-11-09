"""600s evidence/status rollup and ACE terminal feedback; no model or launch actions."""
import collections,fcntl,json,os,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'delivery/s-tfrl-rolling-v1'
sys.path.insert(0,str(ROOT/'delivery/s-single-intent-fewshot-v1/code'))
from common import read,lines,save,sha
E=ROOT/'delivery/s-single-intent-execution-v1/rolling-state'

def update():
    ready=read(OUT/'READY-TASKS.json');window=read(OUT/'REQUESTS/QUEUE-DEFICIT.json')['window']
    teacher=[]
    for folder in sorted((OUT/'teacher').glob('revision-*')):
        result=folder/'ACTUAL-COST.json';teacher.append(dict(revision=folder.name,
            status=read(folder/'COLLECTED.json') if (folder/'COLLECTED.json').exists() else 'STREAMING_OR_UNKNOWN_NO_RESEND',
            local_review=read(folder/'LOCAL-REVIEW.json') if (folder/'LOCAL-REVIEW.json').exists() else None,
            actual_cost=read(result) if result.exists() else None))
    terminal=[];consumer=collections.Counter();consumer_requests=collections.Counter()
    for path in sorted((E/'tasks').glob('*.json')):
        d=read(path)
        consumer[d['status']]+=1;consumer_requests[d['status']]+=d['task'].get('estimated_requests',0)
        if d['status'] in ('completed','failed','blocked'):
            terminal.append(dict(task_id=path.stem,status=d['status'],terminal=d.get('terminal'),score=d.get('score')))
    groups=E/'group-feedback/group-advantages.private.jsonl'
    if groups.exists() and not (OUT/'experience/curator-actual-grpo.json').exists():
        rows=lines(groups);assert len(rows)==545 and all(len(r['members'])==8 for r in rows)
        positive=[];negative=[]
        for r in sorted(rows,key=lambda r:(-r['normalized_std'],r['sample_id']))[:12]:
            members=sorted(r['members'],key=lambda m:(m['advantage'],m['seed']))
            if r['normalized_std']>0:
                negative.append(dict(sample_id=r['sample_id'],member=members[0],group_id=r['group_id']))
                positive.append(dict(sample_id=r['sample_id'],member=members[-1],group_id=r['group_id']))
        save(OUT/'experience/curator-actual-grpo.json',dict(schema='s-ace-curator-actual-feedback-v1',operation='UPDATE',
            entry_id='same-policy-variance-v1',parent_version='prepared-v5-dynamic/single_intent_fewshot',
            source=str(groups.relative_to(ROOT)),source_sha256=sha(groups.read_bytes()),
            applicable='same frozen eight-seed policy and fixed545 only',positive_advantages=positive,negative_advantages=negative,
            groups_with_variance=sum(r['normalized_std']>0 for r in rows),
            lesson='Actual within-policy positive/negative outcomes supersede untested causal claims; review entity differences before compiling a new mechanism.',
            counterevidence='Reward variance alone does not identify a causal prompt/card mechanism; no across-policy pooling.',
            rollback='retire any experience whose next complete paired evaluation has no net benefit',
            next_teacher_input='this entry plus parent policy, exact frozen rewards and repair/regression evidence',
            model_weights_updated=False,gain_proven=False))
    reserve=read(E/'RESERVE.json')
    state=dict(at=time.time(),window=window,ready_catalog_tasks=len(ready['tasks']),
        ready_requests=sum(t['estimated_requests'] for t in ready['tasks']),
        consumer_states=dict(consumer),consumer_requests=dict(consumer_requests),adopted_original_CD=reserve.get('adopted_running_tasks'),
        ready_catalog_is_not_pending=True,teacher=teacher,terminal_tasks=terminal,money=None,
        historical_same545_all_time_best='UNKNOWN',
        reserve=reserve,coverage='estimated inventory only; 12h effective four-lane utilization not yet proven')
    save(OUT/'STATUS.json',state)
    table=['|任务|样本|分支/请求|父版本|CPU验收|','|---|---:|---:|---|---|']
    table.extend('|'+str(t['task_id'])+'|'+str(t['samples'])+'|'+str(t['estimated_requests'])+'|'+t['parent_version']+'|PASS|' for t in ready['tasks'])
    costs=[]
    for t in teacher:
        result=(t.get('actual_cost') or {}).get('result',{})
        costs.append('- '+t['revision']+'：'+str(result.get('state','流式中或未知，禁止重发'))+'；'+str(result.get('usage',{}).get('total_tokens','未知'))+' tokens；'+str(result.get('elapsed_seconds','未知'))+' 秒。金额未知。')
    report='''# 12小时滚动S：CPU生产与经验反馈

状态原位更新，截止UTC '''+time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime())+'''。窗口2026-11-09 16:16:20至2026-11-10 04:16:20；尚未完成12小时。执行owner独占服务、guard和lane，本职责0次NPU launch、0次P重采、0次音频重编码、无spawn。

首发时区卡与相关性约束互补例均已READY且由执行owner接续。时区仅3个value语义变更，复用合法TRAIN。互补例保留第一邻居，第二例在前16及最优剩余分数−0.03内选真实组合互补，不使用query gold，不覆盖全部slots、不超过2例。后继紧凑JSON与缺省mention分别为单机制，真实效果未测。

目录catalog共'''+str(len(ready['tasks']))+'''项（累计准入清单，含已运行/完成，不能当pending）。执行consumer当前状态：'''+json.dumps(dict(consumer),ensure_ascii=False)+'''；各状态请求量：'''+json.dumps(dict(consumer_requests),ensure_ascii=False)+'''；原C/D adopted运行'''+str(reserve.get('adopted_running_tasks'))+'''，未必属于catalog。执行观察时间UTC '''+time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime(reserve['at']))+'''。

'''+ '\n'.join(table)+'''

所有READY来自固定镜像实际tokenizer预算。配对TRAIN已批准原4814逐ID/P三分支对照当前D动态top2，复用原TRAIN10000音频query向量和同8554key，无新embedding。S_fit fewshot3240分支已PASS，预算63793；其3212分支有真实例句。独立审计逐条去examples后packet/wire一致、query向量原jobs行号与audio SHA一致、源句组/录音alias排除PASS。remainder四片fewshot全部正式PASS并READY：933/934/933/934样本，2799/2802/2799/2802分支，预算64045/63722/63468/64080。五个配对任务完整14442分支已交接，TRAIN-PAIRED-BATCH-END.json记录passed5/failed0，旧失败回执保留。主已独立验收，配对不是随机采样或重采P。S_fit完整1080×3验收通过，max63144，validate_fit与无service/无guard的CPU-only bind回放PASS。remainder3734按原序连续四片933/934/933/934，全集不重抽；以各片正式回执为ready依据。两TRAIN集合与545交集0；S_fit reviewed_hold130、remainder750，保全部分母，teacher_update_allowed=false；S_valid1080不喂教师。source packet/P_response/P_wire/原始audio与transport音频分别验证，原packet随source-packets归档。

8seed只是一种策略的真实随机实现，temperature0.7、top_p0.9，实际wire去seed后1635×8逐项一致。typed-normalized主reward在采样前冻结，strict/official另报。完整已知8seed终态后才能545组优势，std0→0；不得跨策略混组，不把13080请求称13080独立样本。正式group反馈尚未完成时不能称训练成功/闭环收益。

教师v3真实返回1 proposal/3经验，因混旧batched sample545与新single branch部分诊断、陈旧计分和低信息casing全部拒绝准入。旧58394tokens/622.319s成本保留。v4是一次授权纠偏，按固定模板带版本、父、单位、arm、完整性、分母、time；同ID358局部协议表独立复算修130退1，C length126/D3，不能称完整语义效果。两例fewshot之外wire归一差分358/358一致，13个无例相同wire修2退1提示lane/time等限制。教师不得重复既有策略，允许0提案。v4真实HTTP200 stop，60297 tokens/484.75秒，1 proposal/3经验；唯一verbatim候选经MAIN-V4-REVIEW暂缓，无admitted candidate、无READY/NPU。该假设仅旧batched两例，不足外推当前全类型；不再教师API纠偏。v3/v4合118691tokens/1107.069秒，旧revision-v2成本另保。响应流fsync，0自动重试，unknown绝不重发；不从reasoning提策略。

'''+ '\n'.join(costs)+'''

基线严格118/545、F1 .291611185；surface177/.496671105；typed180/.507323569；官方条件SLU .568915020。归一化是改分，非模型提升。全历史同545 ID/mask最好仍UNKNOWN；已有FoldA1080参考.733139214不同ID/架构/口径，不能直接相减，未喂教师。合法TRAIN5598 binding缺失0，不能虚构根因。

CONTINUATION-START.json给真实后台PID；失败CPU attempt均留存，仅明确CPU错误修复后重验，不自动重发NPU/API。热文件在本机持久root盘、远端/srv，保fsync。已知旧CPU问题含模块导入、Python缺jinja2、N来源缺包和full_denominator误判；历史日志保留。生产计划含两TRAIN/四分片与固定8seed，不扩seed填时长。按4秒/分支库存时长仅粗估，不能保证四路12h，需根据实际有效吞吐与反馈继续有信息的候选。

进度/秒每条/卡数/实例/剩余/成本以执行owner rolling-state/RESERVE.json、STATUS.json及实际终态为准。运行4实例各TP4是执行方上报；本职责CPU镜像0卡。正式金额、全职责CPU累计成本与全历史最好未知。
'''
    stages=dict(group_collection=dict(expected_seeds=8,expected_groups=545,complete=groups.exists()),
        advantages=dict(actual=groups.exists()),
        next_teacher=dict(actual=any(t['revision'].startswith('revision-v5-') and t.get('actual_cost') for t in teacher),max_new_requests=1),
        experience=dict(actual=(OUT/'experience/curator-actual-grpo.json').exists(),gain_proven=False),
        candidate_evaluation=dict(actual=False,status='No complete next-round TFGRPO candidate evaluation evidenced'))
    state['feedback_stages']=stages
    save(OUT/'STATUS.json',state)
    save(OUT/'FEEDBACK-STAGES.json',stages)
    report+='\n反馈五阶段（actual依据真实artifact）：\n\n'+json.dumps(stages,ensure_ascii=False,indent=2)+'\n'
    report+='\nS_valid 原锁1080两策略各3240分支已固定image PASS，预算62503/63148；6480分支保持独立评价后备，禁止教师/调优。仅执行方最新RESERVE在900秒内且实际容量不足才自动发布；不因每lane CPU计数机械拆片。已读主MAIN-RESERVE-ACCEPTANCE。\n'
    report+='\nTFGRPO独立固定模板与通用操作DSL已落实，真实父dynamic fewshot，两例/cards/audio/其他wire保持；四必测路径及实际context join离线通过，见evidence/FEEDBACK-OFFLINE-ACCEPTANCE.json，0API/0NPU测试。操作白名单只限制内容，不证明因果或收益。最多一次新付费请求，持久quota与launch-intent防重，完整545×8实际反馈且非零优势触发，完整语义比较仅作可选补充。\n'
    report+='\n库存观测：'+str(reserve.get('ready_tasks'))+' pending任务 / '+str(reserve.get('ready_branches'))+'分支；四路剩余窗口'+str(round(max(0,(window['end_epoch']-time.time())/3600),3))+'小时，库存粗估'+str(reserve.get('estimated_queue_wall_hours'))+'小时。秒/分支范围'+str(reserve.get('seconds_per_branch_range'))+'；实例4×TP4合16卡由执行方上报，本职责0卡。估算不保证有效利用率。每lane详情和快照时间见STATUS.json reserve。\n'
    with (OUT/'REPORT.md').open('w') as file:
        file.write(report);file.flush();os.fsync(file.fileno())
    cfgpath=OUT/'CONFIG.json';cfg=read(cfgpath);cfg['at']=time.time()
    cfg['code_pins']={str(p.relative_to(OUT)):sha(p.read_bytes()) for p in (OUT/'code').glob('*.py')}
    cfg['configuration_pins']={str(p.relative_to(OUT)):sha(p.read_bytes()) for p in (OUT/'config').iterdir() if p.is_file()}
    cfg['teacher']['next_round']=dict(algorithm='training-free GRPO group-relative experience update',max_new_requests=1,actual_complete_groups_only=True,proposal_limit=2,independent_template='config/training-free-grpo-template-v1.md',bounded_operations='config/tfgrpo-operations-v1.json')
    save(cfgpath,cfg)
    cost=read(OUT/'COST.json');success=[]
    task_ids={t['task_id'] for t in ready['tasks']}
    task_ids.update(('train-s_valid-single-v1','train-s_valid-single-fewshot-v1'))
    for name in sorted(task_ids):
        receipt=OUT/'evidence'/name/'CPU-VERIFIED.json'
        if not receipt.exists():continue
        c=read(receipt);success.append(dict(task_id=name,samples=c.get('samples',545),branches=c.get('rows',c.get('branches',1635)),CPU_seconds=c.get('CPU_seconds'),wall_seconds=c.get('wall_seconds'),max_total_upper=c['max_total_upper']))
    cost.update(at=time.time(),formal_CPU_successes=success,success_CPU_seconds=sum(c['CPU_seconds'] or 0 for c in success),teachers=[dict(revision=t['revision'],actual=t['actual_cost']) for t in teacher if t['actual_cost']])
    cost['current_new_teacher_tokens']=sum(t['actual_cost'].get('result',{}).get('usage',{}).get('total_tokens',0) for t in teacher if t['actual_cost'])
    save(OUT/'COST.json',cost)

def main():
    guard=(OUT/'report-worker.guard').open('a+b');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(OUT/'REPORT-WORKER-START.json',dict(pid=os.getpid(),ppid=os.getppid(),at=time.time(),interval_seconds=600))
    end=read(OUT/'REQUESTS/QUEUE-DEFICIT.json')['window']['end_epoch']
    while True:
        try:update()
        except Exception as e:save(OUT/'REPORT-WORKER-ERROR.json',dict(error=str(e),traceback=traceback.format_exc(),at=time.time()))
        if time.time()>=end:break
        time.sleep(min(600,end-time.time()))

if __name__=='__main__':main()

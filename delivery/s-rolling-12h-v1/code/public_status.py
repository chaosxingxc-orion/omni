"""Render aggregate-only status from local receipts; never contact NPU hosts."""
import json
import re
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STATE = ROOT / 'delivery/s-single-intent-execution-v1/rolling-state'
OUT = ROOT / 'delivery/s-rolling-12h-v1/public'

def read(path):
    return json.loads(path.read_text()) if path.exists() else {}

def stamp(epoch):
    return datetime.fromtimestamp(epoch, timezone(timedelta(hours=8))).isoformat(timespec='seconds')

def render():
    window = read(ROOT / 'delivery/s-rolling-12h-v1/WINDOW.json')
    resource = read(STATE / 'RESOURCE-LATEST.json')
    reserve = read(STATE / 'RESERVE.json')
    lanes = {}
    for lane in 'ABCD':
        current = read(STATE / 'lanes' / (lane + '-CURRENT.json'))
        live = read(STATE / 'lanes' / (lane + '-LIVE.json'))
        launch = current.get('launch') or {}
        lanes[lane] = dict(task=launch.get('task_id'), state=current.get('status'), snapshot_at=live.get('at'),
            completed_samples=live.get('completed_samples'), total_samples=live.get('total_samples'),
            completed_branches=live.get('completed_branches'), total_branches=live.get('total_branches'),
            seconds_per_branch=live.get('seconds_per_branch'), remaining_seconds=live.get('remaining_seconds_estimate'),
            invalid=live.get('invalid'), tokens=live.get('usage', {}).get('total_tokens'),
            concurrency=launch.get('concurrency'), cards=4, client_inflight_or_unknown=live.get('inflight_or_unknown'))
    policy = read(STATE.parent / 'lane-policy.json').get('lanes', {})
    released = {lane: read(STATE / 'lanes' / (lane + '-RELEASED.json')) for lane in lanes}
    active_lanes = [lane for lane in lanes if not released[lane] and policy.get(lane, {}).get('admit_new_tasks', True)]
    for lane, receipt in released.items():
        if receipt:
            lanes[lane].update(state='released_by_user', cards=0, remaining_seconds=0)
    core = []
    for host in resource.get('hosts', {}).values():
        for line in host.get('npu_smi', '').splitlines():
            cells = [s.strip() for s in line.split('|')]
            if len(cells) > 3 and re.fullmatch(r'[0-9A-Fa-f]{4}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}[.][0-9]', cells[2]):
                core.append(int(cells[3].split()[0]))
        for lane, metrics in host.get('metrics', {}).items():
            if lane in lanes:
                lanes[lane]['server_running'] = metrics.get('running')
                lanes[lane]['server_waiting'] = metrics.get('waiting')
    for lane, receipt in released.items():
        if receipt:
            lanes[lane].update(server_running=None, server_waiting=None)
    ready = [read(p) for p in (STATE / 'tasks').glob('*.json') if read(p).get('status') == 'ready']
    branches = sum(s['task'].get('branches', 0) for s in ready)
    residual = sum(max(0, (d.get('total_branches') or 0) - (d.get('completed_branches') or 0)) for d in lanes.values() if d['state'] == 'running')
    groups = {k: [] for k in ('ready_dynamic_TRAIN', 'ready_evaluation_only', 'ready_hypotheses', 'deprioritized_no_examples')}
    for state in ready:
        task = state['task']
        name = task['task_id']
        if name in {'train-remainder-part1-v1', 'train-remainder-part2-v1', 'train-remainder-part3-v1', 'train-s_fit-single-v1'}:
            kind = 'deprioritized_no_examples'
        elif 's_valid' in name:
            kind = 'ready_evaluation_only'
        elif 'remainder' in name:
            kind = 'ready_dynamic_TRAIN'
        else:
            kind = 'ready_hypotheses'
        groups[kind].append(dict(task_id=name, branches=task.get('branches', 0)))
    counts = {k: sum(t['branches'] for t in v) for k, v in groups.items()}
    unknown_eta = [s['task']['task_id'] for s in ready if not s['task'].get('capacity_estimate_eligible', True)]
    unknown_eta_branches = sum(s['task'].get('branches', 0) for s in ready if s['task']['task_id'] in unknown_eta)
    useful = branches - counts['deprioritized_no_examples'] - unknown_eta_branches
    capacity = dict(at=time.time(), groups=groups, branch_counts=counts,
                    ready_useful_policy_and_eval_branches=useful, running_residual_branches=residual,
                    useful_plus_running_four_lane_hours={str(s): (useful + residual) * s / (3600 * max(1, len(active_lanes))) for s in (3, 4, 5)},
                    conditional_compact_CPU_branches=11202, conditional_is_ready=False,
                    compact_promotion_status="REJECTED_FULL545_ACCURACY_GATE",
                    unknown_eta_task_ids=unknown_eta, unknown_eta_branches=unknown_eta_branches,
                    no_filler_seeds=True, estimate_is_guarantee=False)
    scenarios = {str(s): (branches - unknown_eta_branches + residual) * s / (3600 * max(1, len(active_lanes))) for s in (3, 4, 5)}
    record = dict(schema='public-rolling-aggregate-v2', generated_at=stamp(time.time()),
        window=dict(start=stamp(window['start_epoch']), end=stamp(window['end_epoch']), duration_hours=12,
                    admission_policy='close at deadline; drain inflight'),
        cards=16, instances=4, TP=4, lanes=lanes,
        queue=dict(ready_tasks=len(ready), ready_branches=branches, running_residual_branches=residual,
                   remaining_window_hours=max(0, window['end_epoch']-time.time())/3600,
                   estimated_wall_hours_by_seconds_per_branch=scenarios,
                   estimate_is_guarantee=False, cpu_pending=reserve.get('cpu_preparing'),
                   preference_is_exclusive_reservation=False),
        resource=dict(snapshot_at=resource.get('at'), ai_core_percent=core,
                      mean_ai_core_percent=sum(core)/len(core) if core else None,
                      continuous_80_percent_proven=False),
        coverage_12h='NOT_YET_PROVEN', observed_handoff_gaps=True,
        monitoring_seconds=600, cpu_and_npu_async=True, unknown_request_replay=False,
        useful_capacity=capacity,
        cost_scope='tokens reflect current lane snapshots only; historical failures preserved privately; currency unknown')
    lines=['# 持续实验队列', '', '更新时间：'+record['generated_at'], '',
           '**固定窗口：'+record['window']['start']+' 至 '+record['window']['end']+'。** 到期停止新准入，在途请求排空；不再次重置窗口。', '',
           f'当前可继续调度实例 {len(active_lanes)} 个、每实例TP4。已释放实例不再派发任务；CPU准备、评分、归档与教师分析异步执行。', '',
           '|路|实验|完成整样本|完成分支|秒/分支|剩余分钟|服务 running / waiting|',
           '|---|---|---:|---:|---:|---:|---:|']
    for lane,d in lanes.items():
        speed='待估' if d['seconds_per_branch'] is None else format(d['seconds_per_branch'], '.2f')
        eta='待估' if d['remaining_seconds'] is None else format(d['remaining_seconds']/60, '.0f')
        lines.append(f"|{lane}|{d['task']}|{d['completed_samples']}/{d['total_samples']}|{d['completed_branches']}/{d['total_branches']}|{speed}|{eta}|{('已按用户要求释放' if d['state']=='released_by_user' else str(d.get('server_running'))+' / '+str(d.get('server_waiting')))}|")
    lines += ['', f'待执行 {len(ready)} 项、{branches:,} 分支；在途剩余 {residual:,} 分支。总库存含低价值对照，不用总数宣称有效负载保障；高价值库存估计见下文。窗口剩余 {record["queue"]["remaining_window_hours"]:.2f} 小时。吞吐加速会更快消耗库存，容量估计不能证明未来连续满载。', '',
              f'最新16卡瞬时AI Core：{core}；平均 {record["resource"]["mean_ai_core_percent"]}%。已有接续间隙，未证明持续80%以上，也未完成12小时验收。', '',
              '## 接续与学习', '',
              '- 当前固定545真实样本、P Top3共1635分支；八种子组保持相同策略与预算，不扩种子填时长。',
              '- seed03明确的调用前失败保留；retry2按显式映射收集，要求相同seed/config/pins、完整终态且unknown=0。',
              '- seed08在1103/1635分支后超时，保留1个未知请求，不重发、不拼成完整八种子组。seed03已完成545/1635，现七个成员完整；TFGRPO全组学习暂缺前提，独立ACE与TRAIN继续。',
              '- D路已完成外卖槽位修订37条/111分支，评分与归档保存后按用户要求停止引擎、释放4–7卡；lane-policy关闭后续准入。A/B/C继续。',
              '- B路高浪费无示例对照主动停止新准入并排空至891/2802分支，354非法；保留全部成本。已自动接续动态few-shot TRAIN，不能把部分对照算完整结果。',
              '- CPU采用prepare_plan.py --plan，复用已发布输入；后续优先运行真实根因策略，避免重复基线占位。',
              '- S_valid仅独立评价，禁止进入教师调优。保留竞争角色提示假设；关系方向探针缺乏可评分错例，已在启动前退役；单纯前移ambiguity字段的候选不准入。',
              '- 紧凑JSON在无示例父上显著减少非法输出，但精度仍低于动态few-shot；compact+fewshot组合已完成545/1635：全分支非法1，typed F1 0.57236、整样本正确217；strict F1 0.34971、整样本正确155。父版对应219/163条，组合未通过锁定精度门槛，不扩展TRAIN。',
              '- 原3734条TRAIN四片的compact+fewshot推广输入由独立CPU子进程提前准备；545组合已未通过精度门槛，11202分支不准入、不计READY；保留CPU准备产物。',
              '- 教师编译已修复：相同父策略与实际instruction复用原任务和结果，退役策略不再准入；固定镜像16项fixture通过。CPU教师worker已接管，付费quota及未知请求不重发规则不变。',
              '- 各路进度和资源为异步快照，表内时间不能视为同一时刻测量；新任务启动阶段的秒/分支不代表稳定吞吐。',
              '- 每600秒或明确终态更新；公开页面原位刷新。当前tokens仅为各路快照，全部历史失败成本另存，金额未知。', '']
    if capacity:
        useful = capacity.get('useful_plus_running_four_lane_hours', {})
        lines += ['## 有效库存与CPU补给', '',
                  f'有效待执行分支 {capacity.get("ready_useful_policy_and_eval_branches")}；低优先级高浪费对照另列 {capacity.get("branch_counts", {}).get("deprioritized_no_examples")} 分支。有效库存加在途按3/4/5秒估计 {useful.get("3", 0):.2f}/{useful.get("4", 0):.2f}/{useful.get("5", 0):.2f} 小时，尚不足以无条件覆盖剩余窗口。此段为此前四实例容量快照；D释放后不作当前三实例容量结论。', '',
                  '四片compact TRAIN已完成CPU准备，共3734样本/11202分支；组合精度门槛已失败，不能计READY。ACE三个可选槽卡片候选及共享父对照共648请求已完成正式CPU验收、预部署并发布；教师v8消费分析返回零策略；完整545回退与重复诊断已完成，通用跨类型去重缺乏依据。教师v9已返回并审核，原三提案不直接接受；独立咖啡长度删限策略正在CPU物化。TRAIN53候选与ACE条件扩展已准备；subset接续修复尚待最终固定镜像验证，不将准备任务算为正在推理，未知请求不重发。', '',
                  '知识卡全54张、402绑定及5598条合法TRAIN来源已审计；发现可选槽被绝对提取指令覆盖，以及无来源长度限制两个机制。它们是待检验因果假设。few-shot分差的高总体AUC主要重复父top1信息，当前没有足够证据部署新的选择器。', '']
    interval = read(ROOT / 'delivery/s-rolling-12h-v1/ENGINE-INTERVAL-LATEST.json')
    if interval:
        lines += ['## 推理效率分解', '', '最近两个后台快照间的计数差分；请求延迟与并行吞吐分开报告，不推断跨策略的并发因果收益。', '', '|路|区间秒/分支|生成token/s|队列秒/请求|prefill秒/请求|decode秒/请求|KV占比|', '|---|---:|---:|---:|---:|---:|---:|']
        def number(value):
            return '未知' if value is None else format(value, '.2f')
        for lane, metrics in interval['lanes'].items():
            phases = metrics.get('phases_seconds', {})
            values = [metrics.get('seconds_per_branch'), metrics.get('generation_tokens_per_second'), phases.get('queue'), phases.get('prefill'), phases.get('decode'), metrics.get('cache_usage')]
            lines.append('|'+lane+'|'+'|'.join(number(v) for v in values)+'|')
        lines += ['', '该区间无抢占，主要耗时是decode；CPU供给与引擎排队不是当前主要等待。prefix命中约3%—6%，但prefill仅约1秒，单独增加prefix复用难以解释大幅整体提速。AI Core为瞬时采样，不能据此声称持续80%以上。', '']
    recovery = read(ROOT / 'delivery/s-rolling-12h-v1/175-ARCHIVE-SPACE-RECOVERY.json')
    if recovery:
        lines += ['## 存储故障恢复', '',
                  '旧归档进程重复打包并将二进制归档留在Docker日志，导致一台主机Docker分区耗尽。已停止三个旧归档进程，回收2430个确认成功退出的重复归档容器，可用空间恢复约570GB；原始轨迹、模型镜像和在途推理保留。C/D监测容器已恢复，未重启推理。旧脚本修正一次性终态事件、600秒重试和禁用归档日志留存，旧进程保持停用。', '',
                  '详细结果见 [持续队列存储故障与恢复](S-Storage-And-Queue-Recovery)。', '']
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'TWELVE_HOUR-RUNTIME-RECEIPT.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
    (OUT/'S-Rolling-Experiment-Queue.md').write_text(chr(10).join(lines))
    print(json.dumps(dict(ready_tasks=len(ready), ready_branches=branches, scenarios=scenarios, window_end=record['window']['end'], mean_ai_core=record['resource']['mean_ai_core_percent'])))

if __name__ == '__main__':
    render()

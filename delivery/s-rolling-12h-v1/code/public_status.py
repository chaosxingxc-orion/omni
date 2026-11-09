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
    capacity = read(ROOT / 'delivery/s-rolling-12h-v1/USEFUL-QUEUE-CAPACITY.json')
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
    ready = [read(p) for p in (STATE / 'tasks').glob('*.json') if read(p).get('status') == 'ready']
    branches = sum(s['task'].get('branches', 0) for s in ready)
    residual = sum(max(0, (d.get('total_branches') or 0) - (d.get('completed_branches') or 0)) for d in lanes.values() if d['state'] == 'running')
    scenarios = {str(s): (branches + residual) * s / 14400 for s in (3, 4, 5)}
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
    lines=['# 四路持续实验队列', '', '更新时间：'+record['generated_at'], '',
           '**固定窗口：'+record['window']['start']+' 至 '+record['window']['end']+'。** 到期停止新准入，在途请求排空；不再次重置窗口。', '',
           '四实例、每实例 TP4，共16卡。NPU任务独立接续；CPU提前准备输入、预算和远端文件，评分、归档、教师分析异步执行。', '',
           '|路|实验|完成整样本|完成分支|秒/分支|剩余分钟|服务 running / waiting|',
           '|---|---|---:|---:|---:|---:|---:|']
    for lane,d in lanes.items():
        speed='待估' if d['seconds_per_branch'] is None else format(d['seconds_per_branch'], '.2f')
        eta='待估' if d['remaining_seconds'] is None else format(d['remaining_seconds']/60, '.0f')
        lines.append(f"|{lane}|{d['task']}|{d['completed_samples']}/{d['total_samples']}|{d['completed_branches']}/{d['total_branches']}|{speed}|{eta}|{d.get('server_running')} / {d.get('server_waiting')}|")
    lines += ['', f'待执行 {len(ready)} 项、{branches:,} 分支；在途剩余 {residual:,} 分支。按每分支3/4/5秒，四路容量分别估计 {scenarios["3"]:.2f}/{scenarios["4"]:.2f}/{scenarios["5"]:.2f} 小时；窗口剩余 {record["queue"]["remaining_window_hours"]:.2f} 小时。吞吐加速会更快消耗库存，容量估计不能证明未来连续满载。', '',
              f'最新16卡瞬时AI Core：{core}；平均 {record["resource"]["mean_ai_core_percent"]}%。已有接续间隙，未证明持续80%以上，也未完成12小时验收。', '',
              '## 接续与学习', '',
              '- 当前固定545真实样本、P Top3共1635分支；八种子组保持相同策略与预算，不扩种子填时长。',
              '- seed03明确的调用前失败保留；retry2按显式映射收集，要求相同seed/config/pins、完整终态且unknown=0。',
              '- seed08在1103/1635分支后超时，保留1个未知请求，不重发、不拼成完整八种子组。六个成员已完整，seed03仍在运行；TFGRPO全组学习暂缺前提，独立ACE与TRAIN继续。',
              '- D路旧进程已退出且引擎空闲后，持共享guard归档并清理失效flight元数据；未改未知请求journal。独立TRAIN后继已产生真实响应，96并发；没有重启模型服务。',
              '- B路高浪费无示例对照主动停止新准入并排空至891/2802分支，354非法；保留全部成本。已自动接续动态few-shot TRAIN，不能把部分对照算完整结果。',
              '- CPU采用prepare_plan.py --plan，复用已发布输入；后续优先运行真实根因策略，避免重复基线占位。',
              '- S_valid仅独立评价，禁止进入教师调优。保留竞争角色提示假设；关系方向探针缺乏可评分错例，已在启动前退役；单纯前移ambiguity字段的候选不准入。',
              '- 紧凑JSON在无示例父上显著减少非法输出，但精度仍低于动态few-shot；compact+fewshot组合已通过545/1635固定镜像CPU验收并加入A路后继，尚无组合推理收益。',
              '- 原3734条TRAIN四片的compact+fewshot推广输入由独立CPU子进程提前准备；仅在545组合精度、均值奖励与协议门槛通过且相应对照完整后准入，条件准备不计READY。',
              '- 教师编译已修复：相同父策略与实际instruction复用原任务和结果，退役策略不再准入；固定镜像16项fixture通过。CPU教师worker已接管，付费quota及未知请求不重发规则不变。',
              '- 各路进度和资源为异步快照，表内时间不能视为同一时刻测量；新任务启动阶段的秒/分支不代表稳定吞吐。',
              '- 每600秒或明确终态更新；公开页面原位刷新。当前tokens仅为各路快照，全部历史失败成本另存，金额未知。', '']
    if capacity:
        useful = capacity.get('useful_plus_running_four_lane_hours', {})
        lines += ['## 有效库存与CPU补给', '',
                  f'有效待执行分支 {capacity.get("ready_useful_policy_and_eval_branches")}；低优先级高浪费对照另列 {capacity.get("branch_counts", {}).get("deprioritized_no_examples")} 分支。有效库存加在途按3/4/5秒估计 {useful.get("3", 0):.2f}/{useful.get("4", 0):.2f}/{useful.get("5", 0):.2f} 小时，尚不足以无条件覆盖剩余窗口。此容量核算自带时间戳，不是实时保证。', '',
                  '四片compact TRAIN已完成CPU准备，共3734样本/11202分支；科学门槛未通过前不计READY。并行CPU任务正在补齐ACE小规模候选的运行物料、准备基于真实知识卡冲突的下一轮教师请求，并修复已复现的失败后接续故障。', '',
                  '知识卡全54张、402绑定及5598条合法TRAIN来源已审计；发现可选槽被绝对提取指令覆盖，以及无来源长度限制两个机制。它们是待检验因果假设。few-shot分差的高总体AUC主要重复父top1信息，当前没有足够证据部署新的选择器。', '']
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'TWELVE_HOUR-RUNTIME-RECEIPT.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
    (OUT/'S-Rolling-Experiment-Queue.md').write_text(chr(10).join(lines))
    print(json.dumps(dict(ready_tasks=len(ready), ready_branches=branches, scenarios=scenarios, window_end=record['window']['end'], mean_ai_core=record['resource']['mean_ai_core_percent'])))

if __name__ == '__main__':
    render()

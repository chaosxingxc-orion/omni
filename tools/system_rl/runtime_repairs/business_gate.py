"""Call MAIN's one official paired evaluator; DEV selects, frozen TEST confirms."""
import copy,json,math,subprocess,sys
from pathlib import Path
from dependencies import ROOT,require
from run_package import read,pin,write
PAIRED=ROOT/'jev-system-rl/paired-evaluation/paired.py'


def assess(baseline,candidate,selection,output,*,split='DEV',frozen_selection=None):
    root=Path(output);marker=root/'RESULT.json'
    config=dict(baseline=pin(baseline),candidate=pin(candidate),split=split,
        bootstrap=copy.deepcopy(selection['bootstrap']),quality=copy.deepcopy(selection['quality']),
        min_dev_absolute_gain=selection['quality']['min_dev_absolute_slu_gain'],output=str(marker))
    if split=='TEST':
        frozen=read(frozen_selection)
        require(frozen['selection_split']=='DEV' and not frozen['test_feedback'],'TEST requires prior frozen DEV selection')
    evaluator=PAIRED
    if 'reward_evaluation' in selection:
        binding=selection['reward_evaluation']
        for ref in read(binding['source_closure']):pin(ref)
        evaluator=Path(pin(binding['paired_module'])['path'])
        config['reward_evaluation']=dict(specification=binding['specification'])
        if split=='TEST':
            require(binding['frozen_dev_selection_ref']==pin(frozen_selection),'TEST reward freeze differs from controller selection')
            config['reward_evaluation']['frozen_dev_selection_ref']=pin(frozen_selection)
    config_ref=write(root/'CONFIG.json',config)
    if not marker.exists():subprocess.run([sys.executable,'-B',str(evaluator),'--config',config_ref['path']],check=True)
    result=read(marker)
    require(result['baseline_report']==pin(baseline) and result['candidate_report']==pin(candidate),'paired result inputs drift')
    require(result['business_acceptance']['test_selection_allowed'] is False,'TEST selection forbidden')
    acceptance=copy.deepcopy(result['business_acceptance'])
    if 'reward_evaluation' in selection:
        reward=result['reward_evaluation']
        acceptance['reward_evaluation']=reward
        require(reward['specification']==binding['specification'],'paired fixed reward specification drift')
        acceptance['dev_selection_eligible']=acceptance['dev_selection_eligible'] and reward['clear_dev_reward_gain']
        if split=='TEST':
            delta=reward['paired_delta'];low,high=reward['confidence_interval_95']
            require(all(type(x) in (int,float) and math.isfinite(x) for x in (delta,low,high)) and low<=high,
                    'TEST reward interval invalid')
            checks=dict(frozen_DEV=frozen.get('frozen') is True and frozen['reward_specification']==binding['specification'],
                complete_real_TEST=result['fixture'] is False and result['quality_claim_allowed'] and result['recordings']==13078,
                positive_paired_direction=delta>0,positive_CI_lower_bound=low>0,
                no_engineering_unknown=all(reward[k].get('engineering_unknown')==0 for k in ('baseline_counts','candidate_counts')))
            acceptance['frozen_test_reward_consistency']=dict(checks=checks,passed=all(checks.values()),
                specification=binding['specification'],frozen_dev_selection_ref=pin(frozen_selection),
                paired_delta=delta,confidence_interval_95=[low,high],test_selection_allowed=False)
            acceptance['test_consistent_positive_evidence']=bool(acceptance['test_consistent_positive_evidence'] and all(checks.values()))
    decision=write(root/'ACCEPTANCE.json',dict(result,business_acceptance=acceptance,paired_result_ref=pin(marker)))
    return decision,acceptance


def bind_p_phase(selection, learning):
    from node_difficulty import bind_objective, POSITION
    target=bind_objective({'node':'P'},learning)['metric']
    for spec in selection['phase_acceptance'].values():
        if spec['target']!='P':continue
        path=spec['record_metric_path']
        require(path in (['node_statistics','P','coverage_at3'],['node_statistics','P',target]),
                'unexpected P DEV objective')
        spec['record_metric_path']=['node_statistics','P',target]
        if target==POSITION:spec.setdefault('minimum_coverage_at3_delta',0.)


def assess_phase(baseline,candidate,selection,phase,output):
    """Node learning gate is independent of final SLU business promotion.

    Reuse the same paired source bootstrap on original scorer node statistics;
    never infer P/S1 improvement from D-only end-to-end counts.
    """
    import math
    from dependencies import load
    paired=load(PAIRED,'dp8_phase_original_paired')
    spec=selection['phase_acceptance'][phase]
    require(spec['failure_policy']=='stop_preserve','shared LoRA cannot roll back a single node')
    global_ref,_=assess(baseline,candidate,selection,Path(output)/'system')
    global_result=read(global_ref)
    coverage=None
    if spec['target']=='system':
        target=global_result
    else:
        reports=[read(baseline),read(candidate)]
        from node_sidecar import join
        tables=[join({k:v for k,v in r.items() if k!='reward_contributions_ref'},paired.load_records(r)) for r in reports]
        path=spec['record_metric_path']
        require(path and path[0]=='node_statistics','phase metric must be original node scorer statistic')
        ranking=path==['node_statistics','P','position_reward_v1']
        if ranking:
            from node_difficulty import position_from_call
            scope=load(ROOT/'fresh-cohort-successor/cohort_io.py','phase_score_scope')
            with scope.imports([ROOT/'unified-learning'],('kernel',)):
                score=load(ROOT/'unified-learning/scoring.py','phase_position_score').score_component
        def metric(row):
            node=row.get('node_statistics',{}).get(path[1],{})
            require(node.get('complete') is True,'selected node observation incomplete: '+path[1])
            require(node.get('mask')==1,'selected node gold mask missing: '+path[1])
            if ranking:return position_from_call(node,score,training=False)
            if len(path)>3:
                selected=node
                for key in path[2:-1]:
                    require(key in selected,'selected conditional metric absent');selected=selected[key]
                require(selected.get('complete') is True and selected.get('mask')==1,'selected conditional metric incomplete/masked')
            value=row
            for key in path:
                require(key in value,'full DEV export lacks original phase node statistic: '+'.'.join(path))
                value=value[key]
            require(type(value) in (int,float) and math.isfinite(value) and 0<=value<=1,'phase node statistic range')
            return value
        if len(path)>3:
            require(path[1]=='S2' and path[2]=='conditional_on_supply','unsupported conditional node statistic')
            fixed=read(spec['fixed_upstream_ref'])
            expected=fixed['record_upstream_identities']
            require(tables[0].keys()==tables[1].keys()==expected.keys(),'fixed upstream full population required')
            for sid in tables[0]:
                require(tables[0][sid].get('node_upstream_identity',{}).get('S2')==tables[1][sid].get('node_upstream_identity',{}).get('S2')==expected[sid],'conditional S2 upstream differs')
        converted=[]
        for table in tables:
            converted.append({sid:dict(row,counts=dict(node_score=metric(row))) for sid,row in table.items()})
        if ranking:
            hits=[sum(row['node_statistics']['P']['coverage_at3'] for row in table.values()) for table in tables]
            n=len(tables[0]);coverage=dict(denominator=n,baseline_hits=hits[0],candidate_hits=hits[1],delta=(hits[1]-hits[0])/n)
        class NodeMetric:
            def metrics(self,counts,n):return dict(slu_f1={'f1':counts['node_score']/n},intent_accuracy=0.)
        target=paired.calculate(*converted,NodeMetric(),**selection['bootstrap'])
    bounds=target['paired']['mean_reward' if spec['target']=='system' and 'reward_evaluation' in selection else 'slu_f1'];system=global_result
    checks=dict(real_full_DEV=system['fixture'] is False and system['quality_claim_allowed'] and system['recordings']==8690,
        target_gain=bounds['delta']>=spec['minimum_target_gain'],
        target_paired_lower_bound=bounds['low']>spec['minimum_target_lower_bound'],
        full_valid_answers=system['candidate']['valid_answer_rate']==1.)
    for name,floor in spec['system_minimum_deltas'].items():
        require(name in ('slu_f1','intent_accuracy','joint_exact') and floor<=0,'explicit system regression floor')
        checks[name+'_regression']=system['paired'][name]['delta']>=floor
    if coverage is not None:
        checks['P_coverage_at3']=coverage['delta']>=spec['minimum_coverage_at3_delta']
    checks={name:bool(value) for name,value in checks.items()}
    result=dict(schema='omni-phase-acceptance-v1',phase=phase,passed=all(checks.values()),checks=checks,
        baseline_ref=pin(baseline),candidate_ref=pin(candidate),system_paired_ref=global_ref,target_paired=target,
        specification=spec,threshold_status='configured_pending_real_measurement',final_business_promotion=False)
    if coverage is not None:result['P_coverage_at3']=coverage
    return write(Path(output)/'PHASE-QUALITY.json',result),result

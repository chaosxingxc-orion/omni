"""One-time import of a fully locked node input roster into the sampled TRAIN controller."""
from collections import Counter
import json,random
from pathlib import Path
from dependencies import require,local_path
from run_package import pin,read

def outcome_mix(sampling):
    import math
    mix=sampling['error_success']
    require(isinstance(mix,dict) and set(mix)=={'error','success'}
        and all(type(v) in (int,float) and math.isfinite(v) and 0<=v<=1 for v in mix.values())
        and math.isclose(sum(mix.values()),1,abs_tol=1e-12,rel_tol=0),'invalid configured screen outcome mix')
    return mix


def panel_binding(pool,sampling):
    require(pool['schema']=='node-panel-selection-v1' and pool['node']==sampling['node']=='S1'
        and pool['sealed'] is True and pool['synthetic_fixture'] is False and sampling==pool['sampling']
        and pool['split']==sampling['split']=='TRAIN' and pool['population_scope']=='frozen_recording_fold_panel'
        and sampling['schema']=='recording-fold-sampling-v1' and sampling['purpose']=='screen' and sampling['scope']=='node'
        and sampling['replacement']=='forbid' and sampling['format']=='separated-json-v1'
        and sampling['pool_version']==pool['version']==pool['inputs_ref']['sha256']
        and sampling['system_version']==pool['system_version'] and 0<sampling['sample_count']<=pool['population'],
        'frozen S1 panel identity differs')


def panel_selection(pool,sampling):
    """Read the already sampled rows in their sealed order; never sample their actions."""
    panel_binding(pool,sampling)
    roster=read(pool['inputs_ref']);evidence=read(roster['screen']);design=evidence['configuration']
    require(roster['schema']=='frozen-s1-screen-roster-v1' and roster['node']=='S1' and roster['split']=='TRAIN'
        and roster['fresh_actions'] is False and roster['frozen_membership'] is True
        and roster['system_version']==pool['system_version']==design['system_version']
        and roster['execution_version']==pool['execution_version']==design['execution_version'] and design['seed']==sampling['seed']
        and evidence['schema']=='s1-outcome-screen-v1' and evidence['panel_complete'] is True
        and roster['panel_size']==pool['population']==evidence['observed']==evidence['panel_size']
        and roster['folds']==evidence['training_folds']
        and len(evidence['selected'])==len(roster['draws'])==roster['count']==sampling['sample_count'],
        'incomplete or different frozen panel')
    seen=set(); cells=Counter();used=Counter();draws=[]
    for idx,(entry,s) in enumerate(zip(roster['draws'],evidence['selected'])):
        record=read(entry['record_ref']);origin=read(entry['recording_origin_ref']);obs=read(record['screen_observation_ref'])
        rid=entry['reference_id'];fold=entry['fold'];kind=entry['outcome'];domain=record['domain']
        require(rid==s['reference_id']==record['recording_ref_id']==origin['recording_ref_id']==obs['reference_id'] and rid not in seen
            and record['source']==origin['source']=='SLURP' and record['split']==origin['split']==obs['split']=='TRAIN'
            and record['source_id']==origin['source_id']==obs['source_id'] and domain==origin['domain']==obs['domain']
            and record['synthetic_fixture'] is False and obs['node']=='S1' and s['split']=='selection'
            and fold==s['fold']==obs['fold'] and fold in roster['folds'] and kind==s['outcome']==obs['outcome']
            and kind in design['cap_per_domain_fold_outcome'] and obs['source_component']==s['component']
            and obs['collection_ref']==s['seal'] and obs['response_ref']==s['receipt']['response_ref']
            and obs['source_wire']==s['source_wire'] and record['online']['node_input_ref']==obs['packet_ref']==s['packet']
            and record['outcome']==('success' if kind in ('correct_empty','correct_positive') else 'error'),
            'S1 selection order, observation or recording differs')
        seen.add(rid);key='/'.join(('selection',domain,fold,kind));n=evidence['counts'][key]
        k=min(design['cap_per_domain_fold_outcome'][kind],n)
        probs={name:entry[name] for name in ('fold_probability','panel_conditional_inclusion','outcome_conditional_inclusion')}
        require(all(value==s[name] and type(value) in (int,float) and 0<value<=1 for name,value in probs.items())
            and probs['outcome_conditional_inclusion']==k/n and used[key]<k,'invalid S1 stratum probability')
        # Outcome samples were then shuffled: their relative order within a cell
        # remains a uniform ordered sample without replacement from n candidates.
        j=used[key];q=1/(n-j);used[key]+=1;cells[fold+'/'+kind]+=1
        draws.append(dict(record=record,recording_origin_ref=entry['recording_origin_ref'],draw_index=idx,
            duplicate_occurrence=1,cell=dict(domain=domain,source='SLURP',fold=fold,outcome=kind,balance_key='fold_outcome'),
            cell_draw_index=j,conditional_source_probability=1.,conditional_within_cell_probability=q,conditional_record_probability=q,
            inclusion_probability=None,random_output_position_probability=None,
            unavailable_probability_reason='conditional on the assigned fold, selection partition and observed cell; whole-TRAIN marginal unknown',
            panel_probabilities=probs))
    require(dict(cells)==roster['cells'],'sealed S1 cells differ')
    return draws,dict(samples=len(draws),cells=dict(cells),inputs_ref=pool['inputs_ref'],selection_ref=roster['screen'],
        order='frozen outcome sampling order',population_scope=pool['population_scope'],fresh_actions=False)


def selection(pool,sampling):
    if pool.get('schema')=='node-panel-selection-v1':return panel_selection(pool,sampling)
    require(pool['schema']=='node-screen-selection-v1' and pool['sealed'] is True and not pool['synthetic_fixture']
        and pool['retrieval_population_complete'] is True and pool['official_train_count']>0
        and pool['node']==sampling['node']=='P' and sampling['scope']=='node' and sampling['split']=='TRAIN'
        and sampling['pool_version']==pool['version']==pin(pool['inputs_ref'])['sha256']
        and sampling['system_version']==pool['system_version'],'node screen version or scope differs')
    release=read(pool['inputs_ref']);screen=read(release['screen_config_ref']);cfg=screen['sampling']
    require(release['schema']=='closed-node-screen-inputs-v1' and release['preview'] is False
        and release['formal_sample_locked'] is True and release['screen_scope']=='P' and release['policy_ref']==pool['policy_ref']
        and release['policy']==read(pool['policy_ref'])['version']
        and release['rows']==len(release['draws'])==sampling['sample_count']
        and sampling==dict(cfg,pool_version=pool['version'],system_version=pool['system_version'])
        and cfg['replacement']=='forbid' and cfg['metric']=='valid_P_top3','official roster not locked or sampling differs')
    seal=read(release['complete_ref']);ranked=read(release['recall_ref'])
    require(seal['roster_sha256']==pin(release['selection_ref'])['sha256'] and seal['rows']==release['rows']
        and ranked['population']==pool['official_train_count'] and ranked['p_gate_hits']==release['admitted_population'],
        'screen or retrieval receipt differs')
    with Path(local_path(release['selection_ref']['path'])).open() as f:
        members={v['reference_id']:v for line in f for v in [json.loads(line)]}
    require(len(members)==release['rows'],'duplicate screen recording')
    # random.sample chooses an ordered sample without replacement from each
    # observed (domain,outcome) cell. Condition on that actual eligible pool;
    # dispatch stopped at quotas, so there is no whole-TRAIN inclusion estimate.
    populations=read(release['usage_ref'])['valid_cells']
    require(cfg['source_balance']['key']=='source' and cfg['source_balance']['mode']=='proportional',
            'unrecognized screen source design')
    if screen.get('selection_policy')=='cap_each_cell_at_available':
        targets=screen['requested_cells']
        require(set(populations)==set(targets) and seal['requested']==targets and
                all(type(target) is int and target>=0 and type(populations[cell]) is int and populations[cell]>=0
                    for cell,target in targets.items()),'invalid capped screen populations')
        expected={cell:min(target,populations[cell]) for cell,target in targets.items()}
        require(release['cells']=={cell:n for cell,n in expected.items() if n}
                and seal['deficits']=={cell:target-expected[cell] for cell,target in targets.items()
                                       if expected[cell]<target},
                'screen did not take the available samples up to each cap')
    else:
        require(set(populations)==set(release['cells']),'unrecognized screen populations')
    draws=[];cells=Counter();members_order=list(members)
    for descriptor in release['draws']:
        record=read(descriptor['record_ref']);origin=read(descriptor['recording_origin_ref'])
        rid=record['recording_ref_id'];z=members.pop(rid);o=read(record['screen_observation_ref'])
        require(o['reference_id']==rid==origin['recording_ref_id'] and record['source_id']==origin['source_id']==o['source_id']
            and record['domain']==z['domain']==o['domain'] and record['outcome']==z['outcome']==('success' if z['p_rank'] else 'error')
            and record['split']=='TRAIN' and not record['synthetic_fixture'] and o['response_gold_rank']==z['p_rank']
            and o['valid'] and o['node']=='P' and o['response_ref']['sha256']==z['response_sha256']
            and record['online']['node_input_ref']==o['packet_ref'] and o['policy_version']==release['policy'],
            'record membership or P observation differs')
        cell=record['domain']+'/'+record['outcome'];j=cells[cell]
        require(record['source']=='SLURP' and populations[cell]>=release['cells'][cell]>j and
                rid==members_order[len(draws)],'roster order or cell population differs')
        prob=1/(populations[cell]-j)
        draws.append(dict(record=record,recording_origin_ref=descriptor['recording_origin_ref'],duplicate_occurrence=1,
            cell=dict(outcome=record['outcome'],domain=record['domain'],source=record['source'],balance_key='source'),
            cell_draw_index=j,conditional_source_probability=1.,conditional_within_cell_probability=prob,
            conditional_record_probability=prob,inclusion_probability=None,random_output_position_probability=None,
            unavailable_probability_reason='conditioned on screened valid cell; whole-TRAIN inclusion unknown after priority and stopping'))
        cells[cell]+=1
    require(not members and dict(cells)==release['cells'],'unaccounted screen members')
    # Roster production groups rows by outcome/domain. Mix all four cells across
    # training windows with the frozen seed, preserving exact membership.
    random.Random(sampling['seed']).shuffle(draws)
    for i,d in enumerate(draws):d['draw_index']=i
    return draws,dict(samples=len(draws),cells=dict(cells),selection_ref=release['selection_ref'],
        inputs_ref=pool['inputs_ref'],order='seeded permutation of the locked roster',fresh_actions=False)


def phase_inputs(source,phase,pool):
    """bind_phase locks the NODE-POOL digest: it includes the actual screened input manifest."""
    release=pin(source['phase_frozen_upstream_ref'])
    panel=pool['schema']=='node-panel-selection-v1'
    if panel:panel_binding(pool,source['sampling'])
    require((panel or pool['schema']=='node-screen-selection-v1') and phase['rollout_scope']=='node'
        and phase['target_nodes']==[pool['node']]==(['S1'] if panel else ['P']) and source['screen_ref']==release
        and phase['frozen_upstream_release']==release['sha256']
        and pool['version']==pool['inputs_ref']['sha256']==source['sampling']['pool_version']
        and pool['system_version']==source['sampling']['system_version'],
        'node phase frozen input or system identity differs')
    return release


def panel_cost(pool,sampling,usage_ref):
    panel_binding(pool,sampling)
    r=read(pool['inputs_ref']);u=read(usage_ref)
    require(r['schema']=='frozen-s1-screen-roster-v1' and r['usage_ref']==usage_ref
        and r['panel_size']==pool['population']==u['requests']==len(u['calls'])
        and u['schema']=='s1-native-panel-usage-v1' and u['screen']==r['screen']
        and u['partition']==read(r['inputs'])['partition'] and r['frozen_membership'] is True,
        'native usage does not cover the complete sealed panel')
    seen=set();totals=Counter()
    for call in u['calls'].values():
        rid=call['reference_id'];require(rid not in seen,'duplicate billed recording');seen.add(rid)
        for k in ('prompt_tokens','completion_tokens','total_tokens'):
            v=call[k];require(type(v) is int and v>=0,'invalid native token count');totals[k]+=v
        require(call['total_tokens']==call['prompt_tokens']+call['completion_tokens'],'per-call token sum differs')
    require(all(u[k]==v for k,v in totals.items()),'panel token sum differs')
    return u['total_tokens']


def inherited_cost(answer,request_ref,request):
    if answer.get('schema')=='node-panel-data-result-v1':
        pool=read(answer['screen_ref']);sampling=read(answer['sampling_config_ref'])
        original=read(request['input_ref']);branch=read(original['branch_initial_ref'])
        witness=read(request['endpoint_witness_ref']);policy=read(request['policy_ref'])
        require(answer['complete'] and answer['request_ref']==request_ref and request['stage']=='screen'
            and answer['node']==pool['node']==branch['node'] and answer['population']==pool['population']
            and answer['population_scope']==pool['population_scope'] and answer['screen_scope']=='node'
            and answer['system_version']==pool['system_version']==branch['system_ref']['sha256']
            and answer['transport_config_ref']==original['capture']['transport_config_ref']
            and original['policy']['current_ref']==branch['policy_ref']==request['policy_ref']
            and witness['schema']=='resident-tp4-startup-observation-v1' and witness['ready'] is True
            and witness['adapter_sha256']==policy['assets'][0]['sha256'],'different screen request, policy or service')
        roster=read(pool['inputs_ref']);design=read(roster['screen'])['configuration']
        require(design['collections'],'missing measured S1 collection')
        calls=read(answer['usage_ref'])['calls'];billed=set()
        for ref in design['collections']:
            version=read(read(ref)['version'])
            # Only a dedicated panel collection issued by this DATA request can
            # be charged here. Historical replays lack its endpoint/policy binding.
            require(version['mode']=='live' and version['status']=='SEALED' and version['unresolved_requests']==0
                and version.get('panel_request_ref')==request_ref and version.get('endpoint_witness_ref')==request['endpoint_witness_ref']
                and version.get('policy_ref')==request['policy_ref'] and version['system_version']==pool['system_version']
                and version['execution_version']==pool['execution_version'],'collection lacks the actual DATA/service binding')
            from panel_receipts import collection_calls
            for call in collection_calls(version,local_path):
                env=call['receipt']['native_envelope'];rid=env['request_id']
                require(call['stage']=='S1' and call['status']=='known' and call['submits']==1 and rid in calls and rid not in billed
                    and calls[rid]['response_ref']==env['response_ref'],'unbilled or unresolved DATA call')
                billed.add(rid)
        require(billed==set(calls) and len(billed)<=request['budget']['requests_upper'],'panel billed population differs')
        return panel_cost(pool,sampling,answer['usage_ref'])
    program=read(answer['program_config_ref']);pool=read(answer['screen_ref']);release=read(pool['inputs_ref'])
    dataset=read(request['dataset_ref']);usage=read(answer['usage_ref']);original=read(request['input_ref']);cfg=read(release['screen_config_ref'])
    require(program['request_ref']==request_ref and pin(cfg['program'])==answer['program_config_ref'] and program['stages']==['TRAIN']
        and program['policy_iteration']['policy_ref']==pool['policy_ref']==request['policy_ref']
        and answer['transport_config_ref']==original['capture']['transport_config_ref']
        and program['inputs']['system_version']['sha256']==pool['system_version']==answer['system_version']
        and pool['schema']=='node-screen-selection-v1' and release['formal_sample_locked'] and not release['preview']
        and answer['complete'] and pool['retrieval_population_complete'] and release['official_train_count']==pool['official_train_count']==answer['population']==dataset['population'] and dataset['split']=='TRAIN' and not dataset['synthetic']
        and release['usage_ref']==answer['usage_ref'] and sum(usage['valid_cells'].values())==usage['statuses']['valid']
        and set(usage['statuses'])<={'valid','invalid'} and usage['statuses']['valid']>=release['rows']
        and usage['requests']==sum(usage['statuses'].values())<=request['budget']['requests_upper']
        and usage['total_tokens']==usage['prompt_tokens']+usage['completion_tokens']
        and all(type(usage[k]) is int and usage[k]>=0 for k in ('prompt_tokens','completion_tokens')), 'P screen receipt differs')
    return usage['total_tokens']

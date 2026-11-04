"""Parameterized same-P GEPA parent/child/lock workloads on shared Flight."""
import argparse
import copy
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/analysis')]
import tfrl_h1_observation as observation
from tfrl_hardset_a0_registration import pin,pinned,scientific_hash
from tfrl_hardset_a0_workload import read,raw,sha,confined
from speech_aware_evidence_acquisition.core.atomic_file import publish_new
KIND='gepa-solver-template-pair'
SCHEMA='saea-gepa-pair-workload-v1'


class PairWorkload:
    CONTACTS = 4
    ARMS = ['parent', 'child', 'lock']
    RESULT_SCHEMA = 'saea-gepa-pair-read-v1'
    LABEL = 'GEPA configured parent child comparison'
    PROGRAM = 'One shared P then verified current parent, mutation-derived child and original lock Solver.'

    def __init__(self, config):
        fields={'schema','name','run_id','sample_ids','runtime_binding','execution','material'}
        if not fields <= set(config) <= fields | {'evaluation_partition', 'context_adaptation'} or config['schema']!=SCHEMA:
            raise ValueError('exact parameterized paired workload config required')
        if config.get('evaluation_partition', 'feedback') not in ('feedback', 'selection'):
            raise ValueError('explicit feedback or selection partition required')
        if not re.fullmatch('[a-z0-9][a-z0-9-]*',config['name']) or not re.fullmatch('[A-Z0-9][A-Z0-9-]*',config['run_id']):
            raise ValueError('safe pair identities required')
        ids=config['sample_ids']
        if (not isinstance(ids,list) or not ids or len(ids)>80 or len(set(ids))!=len(ids)
                or any(not isinstance(s,str) or not s.startswith('slurp:train:') for s in ids)):
            raise ValueError('bounded unique ordered TRAIN samples required')
        execution=config['execution']
        if set(execution)!={'admitted_level'} or type(execution['admitted_level']) is not int or not 1<=execution['admitted_level']<=96:
            raise ValueError('explicit bounded client width required')
        self.config=copy.deepcopy(config);self.NAME=config['name'];self.RUN=config['run_id']
        self.context_adaptation()
        self.ROWS=len(ids);self.ROSTER='docs/receipts/rosters/'+self.NAME+'.json'
        self.AUTH='docs/decisions/Development-and-Runtime.md';self.KIND=KIND

    def context_adaptation(self, historical=None, current=None):
        """Explicit same-deployment comparison; never waive frozen model checks."""
        policy = self.config.get('context_adaptation')
        if policy is None:
            if historical is not None and current < historical:
                raise ValueError('paired deployment shrinks historical context without explicit adaptation')
            return None
        fields = {'historical_context_tokens', 'runtime_context_tokens', 'arms',
                  'fit_policy', 'comparison_scope'}
        if (not isinstance(policy, dict) or set(policy) != fields
                or any(type(policy[k]) is not int or not 1 <= policy[k] <= 65536
                       for k in ('historical_context_tokens', 'runtime_context_tokens'))
                or policy['arms'] != self.ARMS
                or policy['fit_policy'] != 'native-every-contact-no-truncation'
                or policy['comparison_scope'] != 'within-runtime-only'):
            raise ValueError('exact bounded context adaptation and all paired arms required')
        if historical is not None and (historical != policy['historical_context_tokens']
                                     or current != policy['runtime_context_tokens']):
            raise ValueError('context adaptation capacities differ from actual model fragments')
        return copy.deepcopy(policy)

    def runtime_binding(self,repo):
        binding=self.config['runtime_binding']
        if set(binding)!={'model_fragment','service_session','profile','runtime_receipt'}:
            raise ValueError('exact paired deployment binding required')
        for k in ('model_fragment','service_session','profile'):
            if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]*',binding[k]):raise ValueError('safe deployment identity required')
        model=read(confined(repo,'configs/model/'+binding['model_fragment']+'.json'))
        profile=read(confined(repo,'configs/deployment/'+binding['profile']+'.json'))
        receipt=read(confined(repo/'docs',binding['runtime_receipt']))
        if model['runtime_receipt']!=binding['runtime_receipt'] or not profile.get('instances'):
            raise ValueError('model/profile runtime differs')
        if [x['name'] for x in profile['instances']] != [x['instance_id'] for x in receipt['deployment']['instances']]:
            raise ValueError('profile and resident membership differ')
        cards=[c for x in profile['instances'] for c in x['cards']]
        if len(set(cards))!=len(cards):raise ValueError('overlapping paired deployment cards')
        if self.config['execution']['admitted_level']>profile['admitted_level']:
            raise ValueError('paired width exceeds configured lane admission')
        if profile['model_fragment']!=binding['model_fragment']:
            deployed=read(confined(repo,'configs/model/'+profile['model_fragment']+'.json'))
            operational={'geometry_basis','decoding_params','reasoning_budget_tokens','answer_reserve_tokens','enable_thinking'}
            if {k:v for k,v in model.items() if k not in operational}!={k:v for k,v in deployed.items() if k not in operational}:
                raise ValueError('deployed model differs from registered alias')
        return copy.deepcopy(binding)

    def material_checked(self, repo, data, material_pin):
        """Replay material construction from separately pinned source artifacts."""
        import gepa_pair_material
        if not isinstance(material_pin, dict) or set(material_pin) != {'relpath', 'sha256'}:
            raise ValueError('externally pinned repository seed material required')
        material = pinned(repo, material_pin)
        checked = gepa_pair_material.verify(repo, data, material)
        if 'intervention' in material and self.config.get('evaluation_partition', 'feedback') != 'feedback':
            raise ValueError('research intervention is approved for feedback only')
        if material.get('schema') != 'saea-gepa-pair-material-v1':
            raise ValueError('exact seed mechanism material required')
        return material, checked


    def population(self, repo, partition_pin, partition=None):
        from speech_aware_evidence_acquisition.rails.gepa_partition import digest
        from speech_aware_evidence_acquisition.data.splits import prefixed, split_identity_hash
        h1, view = observation.population(repo)
        partition = pinned(repo, {k:v for k,v in partition_pin.items() if k != "root"}) if partition is None else partition
        if (partition.get('schema') != 'saea-gepa-partition-v1'
                or partition.get('partition_sha256') != digest({k:v for k,v in partition.items()
                    if k not in ('partition_sha256', 'input_files_sha256')})):
            raise ValueError('fixed GEPA partition integrity differs')
        expected_inputs = (observation.H1, observation.H2,
            f'configs/dataset/{observation.DATASET}.json', 'configs/dataset/slurp-speech-native-test-full.json')
        pins = partition.get('input_files_sha256', {})
        if set(pins) != set(expected_inputs) or any(sha(confined(repo,p)) != pins[p] for p in expected_inputs):
            raise ValueError('partition original population pins changed')
        feedback, selection = partition['feedback_ids'], partition['selection_ids']
        h1_ids = [r['sample_id'] for r in h1['items']]
        if (len(feedback) != 233 or len(selection) != 80 or len(set(feedback+selection)) != 313
                or set(feedback+selection) != set(h1_ids)
                or feedback != sorted(feedback) or selection != sorted(selection)):
            raise ValueError('exact fixed233/80 H1 partition required')
        ids = self.config["sample_ids"]
        role = self.config.get('evaluation_partition', 'feedback')
        allowed = feedback if role == 'feedback' else selection
        if not set(ids) <= set(allowed):
            raise ValueError('paired evaluation IDs must belong to configured ' + role + ' partition; no H2/test')
        by_id = {r['sample_id']: r for r in h1['items']}
        items = [copy.deepcopy(by_id[s]) for s in ids]
        roster = {'schema':'saea-roster-receipt-v1', 'carrier_lock_key':'slurp',
            'roster_id':self.RUN, 'roster_role':'construction',
            'split_role':'discovery', 'count':self.ROWS, 'items':items,
            'split_identity_hash':split_identity_hash(prefixed('slurp', ids)),
            'strata_counts':dict(sorted(Counter(r['stratum'] for r in items).items())),
            'parent':pin(repo,observation.H1), 'partition':partition_pin,
            'selection_rule':'Exact configured IDs from the existing feedback partition; no redraw.',
            'claims_benefit':False}
        if role == 'selection':
            roster['selection_rule'] = 'Exact configured IDs from the existing selection partition; no redraw or feedback mixing.'
        return roster, view


    def derive(self, repo, data, material_pin):
        material, checked = self.material_checked(repo, data, material_pin)
        binding = self.runtime_binding(repo)
        roster, view = self.population(repo, checked['source_pins']['partition'], checked['partition'])
        base = copy.deepcopy(checked['compiled_baseline'])
        source = checked['source_baseline']
        live_model=read(repo/f"configs/model/{binding['model_fragment']}.json")
        archived_model=checked['model']
        from speech_aware_evidence_acquisition.core.reasoning_knobs import (
            fold_into_decoding, resolve_node_decoding, validate_fragment_knobs, validate_thinking_policy)
        policy=source.get('thinking_policy')
        locked={role:observation.THINKING_ARMS['baseline'] for role in ('perception','solver')}
        if policy!=locked:
            raise ValueError('paired requests require explicit original P/S node budgets')
        def request_decoding(model):
            validate_fragment_knobs(model);validate_thinking_policy(policy,model)
            values=fold_into_decoding(model) or model
            decoding=values['decoding_params']
            return {role:resolve_node_decoding(decoding,policy,role,cap=decoding['max_tokens'])
                for role in ('perception','solver')}
        if request_decoding(live_model)!=request_decoding(archived_model):
            raise ValueError('paired native request decoding differs from historical P/S')
        operational={'base_url','runtime_receipt','timeout_seconds','geometry_basis','answer_reserve_tokens'}
        def science(model):
            document={k:copy.deepcopy(v) for k,v in model.items() if k not in operational}
            # A resident completion ceiling is not the evaluated P/S budget.
            # Native node resolution above proves every wire field identical;
            # the registered fragment/session identity itself is never aliased.
            document['decoding_params'].pop('max_tokens',None)
            args=document.get('server_args',[])
            if args.count('--max-model-len')!=1:
                raise ValueError('paired model requires explicit native context geometry')
            index=args.index('--max-model-len')+1
            capacity=int(args[index])
            if capacity<1 or capacity>65536 or any('rope' in str(arg).lower() for arg in args):
                raise ValueError('paired context cannot enable RoPE extrapolation')
            args[index]='<native-context-capacity>'
            return document,capacity
        current,current_context=science(live_model);historical,historical_context=science(archived_model)
        if current!=historical:
            raise ValueError('paired deployment changes frozen model or decoding semantics')
        adaptation = self.context_adaptation(historical_context, current_context)
        if adaptation is not None and not isinstance(base.get('native_contact_fit'), dict):
            raise ValueError('context adaptation requires actual native contact FIT')
        # The compiled historical baseline carries its old window. Bind the
        # adapter to the already-validated live capacity while keeping node
        # request budgets and scientific inputs unchanged.
        base['serving_slot_tokens']=current_context
        # Material verifier owns exact recompilation; these are independent scientific invariants.
        for key in ('knowledge_stores', 'phase_material', 'thinking_policy', 'phase_tools'):
            if base.get(key) != source.get(key):
                raise ValueError('probe changed frozen ' + key)
        if source.get('thinking_policy') != {role:observation.THINKING_ARMS['baseline'] for role in ('perception','solver')}:
            raise ValueError('seed probe requires original3072 P/S reference')
        base.pop('prepared_phase_supply', None)
        base['native_contact_fit'].update(run_id=self.RUN+'-FIT', attempt_id='fit-v1', contact_cap=self.CONTACTS*self.ROWS)
        base['what_this_is'] = 'Configured fixed feedback paired GEPA evaluation: one P, current parent S, child S and original lock; no adoption claim.'
        if self.config.get('evaluation_partition') == 'selection':
            base['what_this_is'] = 'Configured fixed selection paired GEPA evaluation: one P, parent/child/lock S; no feedback mixing or adoption claim.'
        exp = read(repo/observation.SOURCE_EXP)
        cell, = exp['cells']
        cell.update(cell_id=self.NAME, baseline_fragment=self.NAME, model_fragment=binding['model_fragment'],
            run_id=self.RUN, attempt_id='generation-v1', allowed_execution_stages=['check','prepare-supply','worker','campaign'],
            readiness={'state':'held','requirements':[], 'reason':'Prepare, register, admit and preflight exact GEPA probe before dispatch.'})
        exp.update(experiment_id=self.RUN, purpose=base['what_this_is'], service_session=binding['service_session'])
        exp['population'].update(count=self.ROWS, roster_receipt=self.ROSTER, split_identity_hash=roster['split_identity_hash'])
        exp['populations'] = [{'dataset_fragment':observation.DATASET, 'ids':[r['sample_id'] for r in roster['items']]}]
        exp['execution_plan'].update(planned_model_calls=2*self.CONTACTS*self.ROWS,
            planned_speech_audio_seconds=math.ceil(8*math.fsum(r['audio_seconds'] for r in roster['items'])))
        exp['stage3_algorithm'] = {'id':'shared-rollout', 'operators_used':['rollout','scoring'],
            'note':'Mutation-derived current parent/child same-P comparison; no automatic acceptance.'}
        exp['stage3_measurement'].update(rows=self.ROWS, read_roster=roster['roster_id'], claims_benefit=False,
            expected_effect_pp=0, note=base['what_this_is'])
        return base, exp, roster, view, binding


    def fresh_output(self, data, out):
        if out.exists() or not out.resolve().is_relative_to((data/'work/codex').resolve()):
            raise ValueError('fresh owned CPU output required')


    def stage(self, repo, data, material, out):
        from speech_aware_evidence_acquisition.contracts import assert_execution_environment
        from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
        assert_execution_environment(); head = require_clean_study_repo(repo)
        observation.require_unspent(repo, data, self.RUN); self.fresh_output(data, out)
        paths = [repo/f'configs/{kind}/{self.NAME}.json' for kind in ('baseline','experiment')] + [repo/self.ROSTER]
        if any(p.exists() for p in paths):
            raise ValueError('fresh probe configuration and roster required')
        base, exp, roster, _, binding = self.derive(repo, data, material)
        from speech_aware_evidence_acquisition.core.stage3_algorithms import validate_stage3_algorithm_label
        validate_stage3_algorithm_label(exp.get('stage3_algorithm'), cell=self.NAME)
        for path, doc in zip(paths, (base, exp, roster)):
            publish_new(path, raw(doc))
        out.mkdir(parents=True)
        record = {'schema':'saea-gepa-pair-stage-v1', 'cell':self.NAME, 'run_id':self.RUN, 'rows':self.ROWS,
            'material':material, 'runtime_binding':binding, 'source_commit':head,
            **({'context_adaptation':self.context_adaptation()} if self.context_adaptation() is not None else {}),
            'dataset':pin(repo,f'configs/dataset/{observation.DATASET}.json'),
            'source_experiment':pin(repo,observation.SOURCE_EXP),
            'output_hashes':{str(p.relative_to(repo)):sha(p) for p in paths},
            'normal_generation_calls':self.CONTACTS*self.ROWS, 'fallback_inclusive_cap':2*self.CONTACTS*self.ROWS, 'fit_calls':self.CONTACTS*self.ROWS,
            'new_model_calls':0, 'generation_authorized':False, 'claims_benefit':False}
        publish_new(out/'complete.json', raw(record)); return record


    def verify_stage(self, repo, data, path):
        staged = read(path)
        if (staged.get('schema') != 'saea-gepa-pair-stage-v1' or staged['cell'] != self.NAME
                or staged['run_id'] != self.RUN or staged['rows'] != self.ROWS):
            raise ValueError('exact fixed feedback probe stage required')
        for rel, digest in staged['output_hashes'].items():
            if sha(confined(repo, rel)) != digest:
                raise ValueError('staged probe configuration changed')
        pinned(repo, staged['source_experiment']); pinned(repo, staged['dataset'])
        base, exp, roster, _, binding = self.derive(repo, data, staged['material'])
        expected = {f'configs/baseline/{self.NAME}.json':base, f'configs/experiment/{self.NAME}.json':exp, self.ROSTER:roster}
        if set(staged['output_hashes']) != set(expected) or any(read(repo/p) != doc for p,doc in expected.items()):
            raise ValueError('stage differs from verified material derivation')
        if staged.get('context_adaptation') != self.context_adaptation():
            raise ValueError('staged context adaptation changed')
        if staged['runtime_binding'] != binding:
            raise ValueError('stage runtime differs')
        return staged


    def prepare(self, repo, data, stage_record, out):
        from speech_aware_evidence_acquisition.contracts import assert_execution_environment
        from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
        from speech_aware_evidence_acquisition.core.flight.factory import build_flight_runtime
        from tfrl_odd40_cells import verify_output
        assert_execution_environment(); require_clean_study_repo(repo)
        observation.require_unspent(repo, data, self.RUN); self.fresh_output(data, out)
        staged = self.verify_stage(repo, data, stage_record)
        runtime = build_flight_runtime(repo, data, model=staged['runtime_binding']['model_fragment'],
            dataset=observation.DATASET, baseline=self.NAME, experiment=self.NAME, run_id=self.RUN, attempt_id='generation-v1',
            runtime_root=data/'work/scratch/runtime-root', lock_path=repo.parents[1]/'docs/datasets.lock.json')
        target = data/'prepared'/self.RUN/'phase-supply-v1.json'
        runtime.prepare_phase_supply(target, max_workers=16)
        verify_output(read(target)['output'], [r['sample_id'] for r in read(repo/self.ROSTER)['items']])
        out.mkdir(parents=True)
        result = {'schema':'saea-gepa-pair-prepared-v1', 'stage_sha256':sha(stage_record), 'rows':self.ROWS,
            'run_id':self.RUN, 'supply':pin(data,str(target.relative_to(data))), 'new_model_calls':0}
        publish_new(out/'complete.json', raw(result)); return result


    def validate(self, repo, data, name, spec, exp, view):
        from tfrl_odd40_cells import verify_output
        from speech_aware_evidence_acquisition.core.flight.native_recovery import native_contact_custody_enabled
        base, expected, roster, expected_view, binding = self.derive(repo, data, spec['material'])
        cell, = exp['cells']
        if spec.get('source_experiment') != pin(repo, observation.SOURCE_EXP):
            raise ValueError('source experiment changed after staging')
        fixed = {'cell':self.NAME,'experiment':self.NAME,'baseline':self.NAME,'run_id':self.RUN,'attempt_id':'generation-v1',
            'fit_run_id':self.RUN+'-FIT','fit_attempt_id':'fit-v1','rows':self.ROWS,'workload_kind':self.KIND,
            'model_fragment':binding['model_fragment'],'runtime_binding':binding,'profile':binding['profile'],
            'service_session':binding['service_session'],'runtime_receipt':binding['runtime_receipt'],
            'split_identity_hash':roster['split_identity_hash'],'execution':self.config['execution']}
        if name != self.NAME or any(spec.get(k) != v for k,v in fixed.items()):
            raise ValueError('exact GEPA probe runtime identity required')
        if (view != expected_view or pinned(repo,spec['dataset']) != view or pinned(repo,spec['roster']) != roster
                or spec['roster']['relpath'] != self.ROSTER):
            raise ValueError('fixed probe population changed')
        for key in ('population','populations','execution_plan','stage3_algorithm','stage3_measurement','service_session'):
            if exp.get(key) != expected.get(key):
                raise ValueError('probe population, budget or declared algorithm changed')
        for key in ('cell_id','baseline_fragment','model_fragment','dataset_fragment','run_id','attempt_id'):
            if cell.get(key) != expected['cells'][0].get(key):
                raise ValueError('registered probe cell changed')
        actual = read(repo/f'configs/baseline/{self.NAME}.json')
        if (scientific_hash(actual) != scientific_hash(base)
                or scientific_hash(actual) != spec['scientific_baseline_sha256']
                or not native_contact_custody_enabled({**actual,'transport':'vllm','carrier_lock_key':'slurp'})):
            raise ValueError('registered probe program, material or recovery changed')
        if spec['gain_lock'] != {'relpath':observation.LOCK,'sha256':observation.LOCK_SHA} or pinned(repo,spec['gain_lock'])['status'] != 'locked_full_train_fold_b':
            raise ValueError('locked first-round baseline required')
        verify_output(pinned(data,actual['prepared_phase_supply'])['output'], [r['sample_id'] for r in roster['items']])
        return self.ROWS


    def register(self, repo, data, stage_record, prepared):
        from speech_aware_evidence_acquisition.contracts import assert_execution_environment, ExecutionPlan
        from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
        from speech_aware_evidence_acquisition.core.config import compose, protocol_hash
        assert_execution_environment(); head = require_clean_study_repo(repo)
        observation.require_unspent(repo,data,self.RUN); staged = self.verify_stage(repo,data,stage_record); prep = read(prepared)
        if (prep.get('schema') != 'saea-gepa-pair-prepared-v1' or prep['stage_sha256'] != sha(stage_record)
                or prep['rows'] != self.ROWS or prep['run_id'] != self.RUN or prep['new_model_calls'] != 0):
            raise ValueError('exact prepared probe required')
        bp, ep = [repo/f'configs/{kind}/{self.NAME}.json' for kind in ('baseline','experiment')]
        base, exp = read(bp), read(ep); pkg = repo/'docs/readiness'/self.NAME
        if pkg.exists(): raise ValueError('registration exists')
        base['prepared_phase_supply'] = prep['supply']
        exp['cells'][0]['readiness']['requirements'] = [{'kind':'preparation',**prep['supply']}]
        bp.write_bytes(raw(base)); ep.write_bytes(raw(exp))
        binding = staged['runtime_binding']; model = binding['model_fragment']
        cfg = compose(repo,model,observation.DATASET,self.NAME,self.NAME); split = exp['population']['split_identity_hash']
        ph = protocol_hash(cfg.config_hash,cfg.values['arm'],split)
        identity = {'cell':self.NAME,'experiment':self.NAME,'run_id':self.RUN,'attempt_id':'generation-v1',
            'fit_run_id':self.RUN+'-FIT','fit_attempt_id':'fit-v1','rows':self.ROWS,'composed_config_hash':cfg.config_hash,
            'protocol_hash':ph,'split_identity_hash':split}
        spec = {**identity, 'workload_kind':self.KIND,'model_fragment':model,'baseline':self.NAME,'runtime_binding':binding,
            'profile':binding['profile'],'service_session':binding['service_session'],'runtime_receipt':binding['runtime_receipt'],
            'authority':self.AUTH,'decision_id':self.NAME+'-owner-admission','decision_relpath':f'coordination/{self.NAME}/root-admission-v1.json',
            'roster':pin(repo,self.ROSTER),'dataset':staged['dataset'],'gain_lock':pin(repo,observation.LOCK),
            'job_config':self.config,'material':staged['material'],'source_experiment':staged['source_experiment'],'execution':self.config['execution'],
            'scientific_baseline_sha256':scientific_hash(base),'stage_sha256':sha(stage_record),
            'generation_program':self.PROGRAM}
        self.validate(repo,data,self.NAME,spec,exp,read(repo/spec['dataset']['relpath']))
        protocol = {'schema':'saea-gepa-pair-generation-v1','cells':{'arm':identity},'profile':binding['profile'],
            'roster':spec['roster'],'material':spec['material'],'normal_generation_calls':self.CONTACTS*self.ROWS,
            'fallback_inclusive_cap':2*self.CONTACTS*self.ROWS,'fit_calls':self.CONTACTS*self.ROWS,'claims_benefit':False,
            **({'context_adaptation':self.context_adaptation()} if self.context_adaptation() is not None else {})}
        protocol_pin = {'relpath':str((pkg/'protocol-registered-v1.json').relative_to(repo)),
            'sha256':hashlib.sha256(raw(protocol)).hexdigest()}
        docs = {'candidate.json':spec,'protocol-registered-v1.json':protocol}
        rows = []
        for kind,rid,aid,cap in [('generation',self.RUN,'generation-v1',2*self.CONTACTS*self.ROWS),('fit',self.RUN+'-FIT','fit-v1',self.CONTACTS*self.ROWS)]:
            budget = {**exp['execution_plan'],'planned_model_calls':cap}
            ExecutionPlan(run_id=rid,attempt_id=aid,carrier_lock_key='slurp',split_identity_hash=split,protocol_hash=ph,**budget).validate(source_root=repo)
            doc = {'schema':f'saea-registered-{kind}-plan-v1','plan_kind':kind,'cell':self.NAME,'experiment':self.NAME,
                'composed_config_hash':cfg.config_hash,'expected_session_id':binding['service_session'],
                'execution_plan':{**budget,'schema':'saea-execution-plan-v1','study':'speech-aware-evidence-acquisition',
                    'run_id':rid,'attempt_id':aid,'carrier_lock_key':'slurp','split_identity_hash':split,'protocol_hash':ph},
                'prepared_phase_supply':prep['supply'],'protocol_document':protocol_pin,'status':'held'}
            rel=f'{kind}-plan-registered-v1.json'; docs[rel]=doc
            rows.append(['2026-11-03',rid,'bounded-discovery-probe',self.LABEL+' '+kind,
                model,'slurp TRAIN fixed '+self.config.get('evaluation_partition','feedback'),'discovery',split,'no',str(cap),str(budget['planned_speech_audio_seconds']),
                str(budget['planned_gpu_hours']),'All historical costs retained; not a search result',str((pkg/rel).relative_to(repo)),
                'Plan SHA256 '+hashlib.sha256(raw(doc)).hexdigest()+'; held'])
        docs['analysis-recipe-v1.json'] = {'schema':self.RESULT_SCHEMA,'rows':self.ROWS,'generation_run_id':self.RUN,
            'attempt_id':'generation-v1','roster':spec['roster'],'arms':self.ARMS,'delivered_arm':'lock','pair_material':staged['material'],
            'new_model_calls':0,'claims_benefit':False,'candidate_accepted':False}
        pkg.mkdir(parents=True)
        for rel,doc in docs.items(): publish_new(pkg/rel,raw(doc))
        ledger=repo/'docs/exposure-ledger.md'
        ledger.write_text(ledger.read_text()+'\n## '+self.LABEL+'\n\n| date | run/experiment id | execution profile | scope | model/tool | speech carrier + split | split role | split identity hash | consumed | calls | speech audio seconds | gpu hours | inherited exposure | artifacts | notes |\n| '+' | '.join(['---']*15)+' |\n'+'\n'.join('| '+' | '.join(row)+' |' for row in rows)+'\n')
        publish_new(repo/'docs/readiness'/f'{self.NAME}-preregistration.md', ('# '+self.LABEL+'\n\n'+self.PROGRAM+' Exact configured population and verified source; '+str(self.CONTACTS)+'N normal calls, '+str(2*self.CONTACTS)+'N fallback cap; all costs retained. Commit, admit and preflight before detached dispatch; no quality claim.\n').encode())
        return {'cell':self.NAME,'rows':self.ROWS,'source_commit':head,'new_model_calls':0,'readiness':'held'}



def validate(repo,data,name,spec,exp,view):
    job=PairWorkload(spec['job_config'])
    if spec['material']!=job.config['material']:raise ValueError('paired material differs from configured source')
    return job.validate(repo,data,name,spec,exp,view)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['stage','prepare','register'])
    for k in ('repo','data','config'):p.add_argument('--'+k,type=Path,required=True)
    for k in ('out','stage-record','prepared'):p.add_argument('--'+k,type=Path)
    a=p.parse_args();job=PairWorkload(read(a.config))
    if a.mode=='stage':result=job.stage(a.repo,a.data,job.config['material'],a.out)
    elif a.mode=='prepare':result=job.prepare(a.repo,a.data,a.stage_record,a.out)
    else:result=job.register(a.repo,a.data,a.stage_record,a.prepared)
    print(json.dumps(result,sort_keys=True))

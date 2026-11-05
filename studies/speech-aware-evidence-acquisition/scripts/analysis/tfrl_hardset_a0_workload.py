"""Prepare one independent A0 P/S observation on a fixed eligible error roster.

No sampling, generation, tokenizer, admission or readiness promotion. Reuse the
original625 dataset view/media and derive only a population subset and held cell.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/analysis')]
from speech_aware_evidence_acquisition.core.atomic_file import publish_new
from speech_aware_evidence_acquisition.data.splits import prefixed,split_identity_hash
from speech_aware_evidence_acquisition.rails import consumption_module as cm
from speech_aware_evidence_acquisition.rails.adapters.slurp_module_hooks import carrier_module_hooks
from speech_aware_evidence_acquisition.rails.controllers import default_phase_program
from speech_aware_evidence_acquisition.rails.native_episode_config import load_native_episode_config
NAME='slurp-tfrl-hardset-a0-observation-r1-v1'
RUN='SAEA-SLURP-TFRL-HARDSET-A0-OBSERVATION-R1-V1'
REASON='Fixed error candidates; independent P/S observation held for recovery activation, registration and admission.'

def input_path(p):
    from speech_aware_evidence_acquisition.core.experiment_assets import resolve_asset
    p=Path(p)
    # Only this imported source root is logical; data and already-mapped paths
    # can themselves contain directories named configs/docs.
    if p.is_relative_to(ROOT):
        return resolve_asset(ROOT,p.relative_to(ROOT).as_posix())
    return p
def read(p):return json.loads(input_path(p).read_bytes())
def sha(p):return hashlib.sha256(input_path(p).read_bytes()).hexdigest()
def raw(d):return (json.dumps(d,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def confined(root,rel):
    from speech_aware_evidence_acquisition.core.experiment_assets import resolve_asset
    original=(root/rel).resolve()
    if not original.is_relative_to(root.resolve()):raise ValueError('source path escapes root')
    logical_root=root.parent if root.name in ('docs','configs') else root
    logical_rel=(root.name+'/'+rel) if root!=logical_root else rel
    p=resolve_asset(logical_root,logical_rel).resolve()
    if p!=original:return p
    if not p.is_relative_to(root.resolve()):raise ValueError('source path escapes root')
    return p


def derive_a0(repo,original):
    block=copy.deepcopy(original['native_episode']['consumption_module'])
    cm.verify_declared_module(block,original,repo_root=repo)
    if len(block['columns'])!=4 or any(c['kind']!='decoding' for c in block['columns']):
        raise ValueError('reviewed K4 source required')
    if 'tfrl_v2_bank' in block:raise ValueError('unbanked A0 required')
    block['columns']=[];block.pop('decoding_frame')
    compiled=cm.compile_module(block,carrier_module_hooks(repo_root=repo),repo_root=repo)
    base=copy.deepcopy(original)
    base['policy']=compiled.program_id
    del base['native_episode']['roles']['solver']['decoding_variants']
    base['native_episode']['consumption_module']=block
    base.pop('prepared_phase_supply',None)
    parsed=load_native_episode_config({**base,'carrier_lock_key':'slurp'},default_phase_program(base['policy']))
    if compiled.program_ceiling!=2 or parsed.max_contacts!=2:raise ValueError('exact P/S two-contact graph required')
    return base


def population(parent,queue):
    items={r['sample_id']:r for r in parent['items']}
    ids=[r['sample_id'] for r in queue]
    if not ids or len(ids)!=len(set(ids)) or not set(ids)<=items.keys():raise ValueError('unique existing subset required')
    if any(r['job_kind']!='independent_a0_difficulty_observation' or r['model_calls_authorized'] is not False for r in queue):
        raise ValueError('held independent-observation queue required')
    chosen=[copy.deepcopy(items[s]) for s in ids]
    if any(type(r['audio_seconds']) not in (int,float) or not math.isfinite(r['audio_seconds']) or r['audio_seconds']<=0 for r in chosen):
        raise ValueError('recorded positive audio durations required')
    roster={k:copy.deepcopy(v) for k,v in parent.items() if k not in ('items','sibling','half','strata_counts','per_stratum')}
    roster.update(roster_id='SLURP-TFRL-HARDSET-A0-OBSERVATION-R1-V1',items=chosen,
        split_identity_hash=split_identity_hash(prefixed('slurp',ids)),
        roster_role='fixed-source-isolated-observed-error-candidates',
        selection_rule='Exact fixed missing-observation queue; no new draw or resampling.',
        draw_method='No draw. Preserve fixed queue membership and order.',
        authority='Owner requested fixed hard-example learning and independent A0 observations; this roster is not statistical hardness confirmation.')
    return roster


def stage(repo,source,data,manifest,out):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    assert_execution_environment()
    if out.exists() or not out.resolve().is_relative_to(data.resolve()/'work/codex'):raise ValueError('fresh owned output required')
    for kind in ('runs','attempts','prepared'):
        for run in (RUN,RUN+'-FIT'):
            if (data/kind/run).exists():raise ValueError('workload identity already has state')
    complete=read(manifest/'complete.json')
    qpath=manifest/'difficulty-observation-queue.jsonl'
    if sha(qpath)!=complete['artifacts']['difficulty-observation-queue.jsonl']:raise ValueError('queue changed')
    queue=[json.loads(s) for s in qpath.read_text().splitlines()]
    if len(queue)!=439 or complete['confirmed_hard_examples']!=0:raise ValueError('exact observed-error439 required')
    old='slurp-tfrl-current-seed-i1-h2-v1';pins={};outputs={}
    def source_doc(rel):
        path=confined(source,rel);pins[rel]=sha(path);return read(path)
    def dependency(rel):
        src=confined(source,rel);dst=confined(repo,rel);pins[rel]=sha(src)
        if dst.exists():
            if dst.read_bytes()!=src.read_bytes():raise ValueError('existing dependency differs: '+rel)
        else:outputs[rel]=src.read_bytes()
    original=source_doc(f'configs/baseline/{old}.json')
    exp=source_doc(f'configs/experiment/{old}.json')
    parent=source_doc('docs/receipts/rosters/slurp-stage3-construction-r01c.json')
    roster=population(parent,queue);ids=[r['sample_id'] for r in roster['items']]
    base=derive_a0(repo,original)
    base['native_contact_fit'].update(run_id=RUN+'-FIT',attempt_id='fit-v1',contact_cap=2*len(ids))
    base['what_this_is']=REASON
    cell,=exp['cells'];dataset_rel=f"configs/dataset/{cell['dataset_fragment']}.json"
    dataset=source_doc(dataset_rel);media={r['sample_id']:r for r in dataset['items']}
    if len(media)!=625 or len(dataset['items'])!=625:raise ValueError('original full625 dataset must remain intact')
    for item in roster['items']:
        row=media[item['sample_id']]
        if row['audio_seconds']!=item['audio_seconds'] or not row['audio_sha256'] or '/slurp_real/' not in row['media_relpath']:
            raise ValueError('original recorded media binding differs')
        if not confined(data,row['media_relpath']).is_file():raise ValueError('media missing')
    roster_rel=f'docs/receipts/rosters/{NAME}.json'
    exp.pop('service_session',None)
    # The unchanged dataset owns the runtime split namespace.
    exp.update(experiment_id=RUN,purpose=REASON)
    exp['population'].update(count=len(ids),roster_receipt=roster_rel,split_identity_hash=roster['split_identity_hash'])
    exp['populations'][0]['ids']=ids
    cell.update(cell_id=NAME,baseline_fragment=NAME,run_id=RUN,attempt_id='generation-v1',
        allowed_execution_stages=['check'],readiness={'state':'held','requirements':[],'reason':REASON})
    audio=math.fsum(r['audio_seconds'] for r in roster['items'])
    exp['execution_plan'].update(planned_model_calls=4*len(ids),planned_speech_audio_seconds=math.ceil(4*audio))
    exp['stage3_algorithm']={'id':'shared-rollout','operators_used':['rollout','scoring'],
        'note':'One complete independent A0 P/S trajectory per fixed error candidate; no shared historical P.'}
    exp['stage3_measurement'].update(rows=len(ids),read_roster=roster['roster_id'],claims_benefit=False,
        expected_effect_pp=0,measurement_design='material-source',tuning_roster='none',
        note='Difficulty observation only. No precision-improvement claim and no long-run error-rate claim from one repeat.')
    data_pins={}
    for pin in base['preparation_source_receipts']:
        if pin['root']=='repository':dependency(pin['relpath'])
        elif pin['root']=='data':
            if sha(confined(data,pin['relpath']))!=pin['sha256']:raise ValueError('data preparation source changed')
            data_pins[pin['relpath']]=pin['sha256']
        else:raise ValueError('unknown preparation root')
        if pin['root']=='repository' and pins[pin['relpath']]!=pin['sha256']:raise ValueError('repo preparation source changed')
    model_rel=f"configs/model/{cell['model_fragment']}.json";model=source_doc(model_rel)
    for rel in (dataset_rel,model_rel,'docs/'+model['runtime_receipt']):dependency(rel)
    for rel,doc in [(roster_rel,roster),(f'configs/baseline/{NAME}.json',base),(f'configs/experiment/{NAME}.json',exp)]:
        if (repo/rel).exists():raise ValueError('fresh workload config required')
        outputs[rel]=raw(doc)
    shards=[{'instance_index':i,'sample_ids':[],'audio_seconds':0.0} for i in range(2)]
    for item in sorted(roster['items'],key=lambda r:(-r['audio_seconds'],r['sample_id'])):
        shard=min(shards,key=lambda r:(r['audio_seconds'],r['instance_index']))
        shard['sample_ids'].append(item['sample_id']);shard['audio_seconds']+=item['audio_seconds']
    record={'schema':'saea-hardset-a0-held-workload-v1','cell':NAME,'run_id':RUN,'rows':len(ids),
        'normal_generation_calls':2*len(ids),'fallback_inclusive_call_cap':4*len(ids),
        'fit_contact_cap':2*len(ids),'unique_audio_seconds':audio,'generation_audio_cap':math.ceil(4*audio),
        'model_fragment':cell['model_fragment'],'dataset_fragment':cell['dataset_fragment'],
        'dataset_reused_unchanged':True,'dataset_sha256':pins[dataset_rel],
        'manifest_sha256':sha(manifest/'complete.json'),'queue_sha256':sha(qpath),
        'source_hashes':pins,'data_hashes':data_pins,
        'output_hashes':{p:hashlib.sha256(b).hexdigest() for p,b in outputs.items()},
        'runtime_config_loaded':True,'execution_contact_ceiling':2,
        'registered':False,'generation_authorized':False,'readiness':'held',
        'new_model_calls':0,'new_tokenizer_calls':0,
        'operational_shards':shards,'shards_applied_to_dispatch':False,
        'deployment':{'instances':2,'tensor_parallel':4,'max_num_seqs_per_instance':128},
        'remaining':['A0 contact recovery activation','current service verification','registration and admission',
            'real FIT during dispatch','detached resumable runner and CPU scoring handoff'],
        'note':'Shards are deterministic duration-balanced operational hints; existing runtime dispatch remains authoritative. One extra complete trajectory does not confirm statistical hardness.'}
    out.mkdir(parents=True);publish_new(out/'intent.json',raw(record))
    for rel,blob in outputs.items():
        path=repo/rel;path.parent.mkdir(parents=True,exist_ok=True);publish_new(path,blob);path.chmod(0o644)
    publish_new(out/'complete.json',raw(record));return record


def prepare(repo,data,derivation,lock,out):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.flight.factory import build_flight_runtime
    assert_execution_environment();d=read(derivation)
    if d['schema']!='saea-hardset-a0-held-workload-v1' or d['rows']!=439 or d['generation_authorized'] is not False:
        raise ValueError('exact held439 workload required')
    if out.exists() or not out.resolve().is_relative_to(data.resolve()/'work/codex'):raise ValueError('fresh owned output required')
    for rel,pin in d['output_hashes'].items():
        if sha(confined(repo,rel))!=pin:raise ValueError('derived input changed')
    exp=read(repo/f'configs/experiment/{NAME}.json');cell,=exp['cells']
    runtime=build_flight_runtime(repo,data,model=cell['model_fragment'],dataset=cell['dataset_fragment'],baseline=NAME,
        experiment=NAME,run_id=RUN,attempt_id=cell['attempt_id'],runtime_root=data/'work/scratch/runtime-root',lock_path=lock)
    target=data/'prepared'/RUN/'phase-supply-v1.json'
    result=runtime.prepare_phase_supply(target,max_workers=16)
    output=read(target)['output'];ids=exp['populations'][0]['ids']
    from tfrl_odd40_cells import verify_output
    verify_output(output,ids)
    record={'schema':'saea-hardset-a0-first-p-supply-v1','rows':len(ids),'run_id':RUN,
        'derivation_sha256':sha(derivation),'supply':{'relpath':str(target.relative_to(data)),'sha256':sha(target)},
        'new_model_calls':0,'new_tokenizer_calls':0,'bound_into_baseline':False,'generation_authorized':False}
    out.mkdir(parents=True);publish_new(out/'complete.json',raw(record));return record

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['stage','prepare'])
    for k in ('repo','data','out'):p.add_argument('--'+k,type=Path,required=True)
    for k in ('source','manifest','derivation','lock'):p.add_argument('--'+k,type=Path)
    a=vars(p.parse_args());mode=a.pop('mode')
    for k in (('derivation','lock') if mode=='stage' else ('source','manifest')):a.pop(k)
    r=(stage if mode=='stage' else prepare)(**a);print(json.dumps({k:v for k,v in r.items() if k not in ('source_hashes','data_hashes','output_hashes','operational_shards')},sort_keys=True))

"""Registered real K4 bank production, with exact journaled tokenizer counts."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/analysis'),str(ROOT/'scripts/checks')]
from tfrl_current_sources import read,sha,pinned,load_half
from tfrl_current_distill import build,common_resolver,write_bank,distiller,text,common
from tfrl_token_journal import JournalTokenizer
from register_real_engine_candidate import write
RUN='SAEA-SLURP-TFRL-CURRENT-SEED-I1-BANK-V1'
REL='docs/readiness/tfrl-current-seed-i1-bank/analysis-recipe-v1.json'
AUTH='docs/decisions/2026-10-31-owner-lock-baseline-and-cost-repair.md'
CAP=20000


def register(repo,source):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    from tfrl_current_half_result import validate_recipe
    assert_execution_environment();require_clean_study_repo(repo);require_clean_study_repo(source)
    ledger=repo/'docs/exposure-ledger.md';old=ledger.read_text()
    if (repo/REL).exists() or '| '+RUN+' |' in old:raise ValueError('bank production already registered')
    recipes={};stores=[];models=[];sessions=[]
    for h in ('h1','h2'):
        name=f'slurp-tfrl-current-seed-i1-{h}-v1';r=f'docs/readiness/{name}/analysis-recipe-v1.json'
        validate_recipe(source,source/r);recipes[h]={'relpath':r,'sha256':sha(source/r)}
        exp=read(source/f'configs/experiment/{name}.json');cell,=exp['cells']
        stores.append(read(source/f"configs/baseline/{cell['baseline_fragment']}.json")['knowledge_stores'])
        models.append(f"configs/model/{cell['model_fragment']}.json");sessions.append(exp['service_session'])
    if stores[0]!=stores[1] or models[0]!=models[1] or sessions[0]!=sessions[1]:raise ValueError('common source binding required')
    model=read(source/models[0]);receipt='docs/'+model['runtime_receipt']
    recipe={'schema':'saea-tfrl-current-bank-recipe-v1','read_run_id':RUN,'source_recipes':recipes,
        'source_model':{'relpath':models[0],'sha256':sha(source/models[0])},
        'runtime_receipt':{'relpath':receipt,'sha256':sha(source/receipt)},'service_session':sessions[0],
        'tokenizer_contact_cap':CAP,'generation_calls':0,'official_scoring_jobs':0,'rows':625,
        'tokenizer_route':'POST /tokenize','token_measure':'vllm-tokenize','authority':AUTH,
        'entrypoint':'scripts/analysis/tfrl_current_bank_run.py',
        'method':'Existing deterministic contrast distiller; exact source evidence and official per-row credit; common verified knowledge binding.',
        'exemplar_token_cap':2000,'rule_token_cap':400,'rule_support_rows':3,
        'next_iteration_authorized':False,'claims_benefit':False}
    write(repo/REL,recipe)
    fields=['2026-10-31',RUN,'bounded-discovery-probe','Real625 experience bank token measurement','tokenizer only; no generation',
        'slurp TRAIN construction625','discovery','two registered source half identities','no',str(CAP),'0','0',
        'All generation and official score costs retained',REL,
        f'Recipe SHA256 {sha(repo/REL)}; root countersigned under {AUTH}, coordinator Codex; cap counts token requests including failed or unresolved; no generation; no test tuning']
    ledger.write_text(old+'\n## Current real K4 bank production\n\n| date | run/experiment id | execution profile | scope | model/tool | speech carrier + split | split role | split identity hash | consumed | calls | speech audio seconds | gpu hours | inherited exposure | artifacts | notes |\n| '+' | '.join(['---']*15)+' |\n| '+' | '.join(fields)+' |\n')
    return recipe


def validate_recipe(repo):
    import flight_registration_check as reg
    recipe=read(repo/REL);entries=reg.parse_ledger(repo,Path('docs/exposure-ledger.md')).get(RUN,[])
    if (len(entries)!=1 or not reg.is_countersigned(entries[0]) or entries[0]['artifacts']!=REL
            or sha(repo/REL) not in entries[0].get('notes','')):raise ValueError('exact countersigned bank recipe required')
    expected={'schema':'saea-tfrl-current-bank-recipe-v1','read_run_id':RUN,'rows':625,'tokenizer_contact_cap':CAP,
        'generation_calls':0,'official_scoring_jobs':0,'tokenizer_route':'POST /tokenize','token_measure':'vllm-tokenize',
        'exemplar_token_cap':2000,'rule_token_cap':400,'rule_support_rows':3,'claims_benefit':False,'next_iteration_authorized':False}
    if any(recipe.get(k)!=v for k,v in expected.items()) or set(recipe['source_recipes'])!={'h1','h2'}:
        raise ValueError('registered bank method or budget differs')
    return recipe


def run(repo,source,data):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.session.serving import find_resident_session
    from speech_aware_evidence_acquisition.core.session.receipt import SessionReceipt
    from speech_aware_evidence_acquisition.core.session.vllm_identity import verify_vllm_session
    from speech_aware_evidence_acquisition.core.driver.speech_native import _LoopbackPromptAuditClient
    assert_execution_environment();recipe=validate_recipe(repo)
    halves={h:load_half(source,data,h,recipe['source_recipes'][h]) for h in ('h1','h2')}
    return produce(repo,source,data,recipe,halves,out=data/'analysis'/RUN,recipe_path=repo/REL)


def produce(repo,source,data,recipe,halves,*,out,recipe_path):
    """Produce banks from already verified, explicitly sourced halves."""
    from speech_aware_evidence_acquisition.core.session.serving import find_resident_session
    from speech_aware_evidence_acquisition.core.session.receipt import SessionReceipt
    from speech_aware_evidence_acquisition.core.session.vllm_identity import verify_vllm_session
    from speech_aware_evidence_acquisition.core.driver.speech_native import _LoopbackPromptAuditClient
    if not out.resolve().is_relative_to(data.resolve()/'analysis'):raise ValueError('owned bank output required')
    resolver=common_resolver(halves,data)
    model=read(pinned(source,recipe['source_model']));runtime_path=pinned(source,recipe['runtime_receipt']);runtime=read(runtime_path)
    resident=find_resident_session(data)
    if resident is None or resident[1]['session_id']!=recipe['service_session']:raise ValueError('expected resident absent')
    session_path=resident[0]/'session.json';session_sha=sha(session_path)
    instances=runtime['deployment']['instances']
    for instance in instances:
        s=SessionReceipt.from_json(read(data/'serving'/instance['instance_id']/'session-receipt.json'))
        verify_vllm_session(s,runtime)
    # Selected profile is loopback two-TP4; use its first tokenizer, no generation route.
    deployment=source/'configs/deployment/qwen3-omni-real500-2xtp4-s128-b2048-v1.json'
    profile=read(deployment);port=profile['instances'][0]['port']
    if profile['bind_host']!='127.0.0.1' or port!=8200:raise ValueError('expected tokenizer endpoint required')
    def guard():
        if sha(session_path)!=session_sha or sha(runtime_path)!=recipe['runtime_receipt']['sha256']:
            raise ValueError('resident identity changed during token measurement')
    identity={'recipe_sha256':sha(recipe_path),'runtime_receipt_sha256':sha(runtime_path),'session_sha256':session_sha,
        'sources':{h:r['source'] for h,r in halves.items()},'knowledge_resolver':resolver.identity()}
    journal=JournalTokenizer(_LoopbackPromptAuditClient(host='127.0.0.1',port=port,timeout_seconds=600.),
        out/'tokenizer',identity=identity,cap=CAP,guard=guard)
    measure=text.VllmTokenizeMeasure(journal,model=model['served_model_name'],
        chat_template_kwargs={'enable_thinking':model['enable_thinking']})
    loaded={};receipts={}
    for half in ('h1','h2'):
        result=build(halves[half],measure,resolver=resolver)
        receipts[half]=write_bank(result,out/half)
        loaded[half]=distiller.load_bank(out/half/'bank-receipt.json')
        other='h2' if half=='h1' else 'h1'
        common.check_bank_measured(receipts[half])
        common.check_cross_fit(receipts[half],halves[other]['plan']['half'],
            consuming_row_ids=[r['row_id'] for r in halves[other]['plan']['row_plan']],
            consuming_split_identity=halves[other]['roster']['split_identity_hash'])
    union=distiller.union_banks(loaded['h1'],loaded['h2'],measure=measure)
    distiller.write_union_outputs(union,out/'union')
    summary={'schema':'saea-tfrl-current-banks-complete-v1','rows':625,'recipe_sha256':sha(recipe_path),
        'sources':identity['sources'],'knowledge_resolver':resolver.identity(),'accounting':journal.accounting(),
        'banks':{h:{'receipt_sha256':sha(out/h/'bank-receipt.json'),
            'exemplars':read(out/h/'bank-receipt.json')['exemplar_bank']['entries'],
            'rules':read(out/h/'bank-receipt.json')['rule_bank']['rules']} for h in ('h1','h2','union')},
        'next_iteration_authorized':False,'claims_benefit':False}
    if (out/'complete.json').exists():
        if read(out/'complete.json')!=summary:raise ValueError('existing bank completion differs')
    else:write(out/'complete.json',summary)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['register','run'])
    for k in ('repo','source'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--data',type=Path);a=vars(p.parse_args());mode=a.pop('mode')
    if mode=='register':a.pop('data');r=register(**a)
    else:r=run(**a)
    print(json.dumps(r))

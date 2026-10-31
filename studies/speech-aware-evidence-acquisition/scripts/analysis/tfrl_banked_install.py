"""Re-derive and install held real banked configs in a clean execution checkout.

Always derive with this checkout's implementation and verified source results;
never import an older waiting queue's unreviewed configuration bytes. No model,
FIT, tokenizer, scoring, readiness promotion or automatic retry.
"""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/analysis')]
from tfrl_current_sources import read,sha,pinned
from speech_aware_evidence_acquisition.core.atomic_file import publish_new


def install(repo,data,out,iteration=2):
    from speech_aware_evidence_acquisition.contracts import assert_execution_environment
    from speech_aware_evidence_acquisition.core.bundle.git_commit import require_clean_study_repo
    from tfrl_current_next_plan import run,decide
    assert_execution_environment();head=require_clean_study_repo(repo)
    if type(iteration) is not int or iteration not in (2,3):raise ValueError('banked iteration2/3 required')
    if out.exists() or not out.resolve().is_relative_to(data.resolve()/'work/codex'):
        raise ValueError('fresh owned installation record required; inspect existing outputs')
    names={h:f'slurp-tfrl-current-bank-i{iteration}-{h}-v1' for h in ('h1','h2')}
    destinations={}
    for h,name in names.items():
        for kind in ('baseline','experiment'):
            rel=f'configs/{kind}/{name}.json';destinations[rel]=rel
        destinations[f'{h}-sampling-plan.json']=f'docs/readiness/{name}/sampling-plan.json'
        run_id=f'SAEA-SLURP-TFRL-CURRENT-BANK-I{iteration}-{h.upper()}-V1'
        for rid in (run_id,run_id+'-FIT'):
            if any((data/k/rid).exists() for k in ('runs','attempts','prepared')):
                raise ValueError('existing banked artifacts require explicit continuation')
    for rel in destinations.values():
        target=(repo/rel).resolve()
        if not target.is_relative_to(repo.resolve()) or target.exists():
            raise ValueError('target exists or escapes execution checkout')
    def write(path,document):publish_new(path,(json.dumps(document,indent=2)+'\n').encode())
    out.mkdir(parents=True)
    write(out/'intent.json',{'source_commit':head,'iteration':iteration,'model_calls':0,
        'automatic_retry':False,'readiness':'held','targets':list(destinations.values())})
    try:
        decision=decide(repo,data,out/'first-decision.json')
        if decision['decision']!='prepare-held-successor':
            write(out/'stopped.json',{'decision':'stop-family','construction_gain_pp':decision['construction_gain_pp'],
                'installed_files':0,'model_calls':0})
            return {'state':'stopped-family','installed_files':0,'model_calls':0}
        staging=out/'verified-staging'
        completion=run(repo,data,staging,iteration=iteration)
        if (completion['schema']!='saea-tfrl-current-next-held-plans-v1'
                or completion['source_commit']!=head or completion['iteration']!=iteration
                or completion['rows']!=625 or completion['readiness']!='held'
                or completion['new_model_calls']!=0 or completion['generation_authorized_by_this_file'] is not False):
            raise ValueError('unexpected verified staging completion')
        if completion['construction_gate']['construction_gain_pp']!=decision['construction_gain_pp']:
            raise ValueError('staging and first decision differ')
        hashes=dict(completion['held_cell_files'])
        hashes.update({f'{h}-sampling-plan.json':v for h,v in completion['plans'].items()})
        if set(hashes)!=set(destinations):raise ValueError('exact six banked inputs required')
        payloads={}
        for source_rel,target_rel in destinations.items():
            source=pinned(staging,{'relpath':source_rel,'sha256':hashes[source_rel]})
            document=read(source)
            if source_rel.startswith('configs/experiment/'):
                cell,=document['cells']
                if (document['stage3_algorithm']['id']!='tfrl-v2'
                        or document['stage3_algorithm']['operators_used']!=['rollout','scoring','context_application']
                        or cell['readiness']['state']!='held' or cell['allowed_execution_stages']!=['check']):
                    raise ValueError('held banked algorithm label required')
            payloads[target_rel]=source.read_bytes()
        # Re-read the shared real-data/bank guard against a private overlay,
        # keeping the execution checkout unchanged until all checks pass.
        import tempfile,shutil
        from tfrl_banked_supply import validate_preparation_inputs
        with tempfile.TemporaryDirectory(prefix='banked-install-',dir=out) as scratch:
            overlay=Path(scratch)/'study';overlay.mkdir()
            # Symlink unchanged top-level inputs; copy only directories receiving
            # newly derived documents so no write can follow an inherited link.
            for entry in repo.iterdir():
                if entry.name in ('configs','docs'):
                    shutil.copytree(entry,overlay/entry.name,symlinks=False)
                elif entry.name!='.git':
                    (overlay/entry.name).symlink_to(entry.resolve(),target_is_directory=entry.is_dir())
            for rel,payload in payloads.items():
                path=overlay/rel;path.parent.mkdir(parents=True,exist_ok=True);publish_new(path,payload)
            for name in names.values():
                validate_preparation_inputs(overlay,data,name,iteration,out/'first-decision.json')
        if require_clean_study_repo(repo)!=head:raise ValueError('execution checkout changed during verification')
        for rel,payload in payloads.items():
            path=repo/rel;path.parent.mkdir(parents=True,exist_ok=True);publish_new(path,payload)
        result={'schema':'saea-tfrl-banked-install-v1','source_commit':head,'iteration':iteration,
            'rows':625,'output_hashes':{rel:sha(repo/rel) for rel in payloads},
            'first_decision':{'relpath':str((out/'first-decision.json').relative_to(data)),
                              'sha256':sha(out/'first-decision.json')},
            'verified_staging_sha256':sha(staging/'complete.json'),
            'new_model_calls':0,'readiness':'held','registered':False,
            'remaining':['review and commit imported configurations','prepare and bind own supplies',
                         'register reads before resident staging','generation/FIT registration and repin',
                         'admission and full preflight']}
        write(out/'complete.json',result);return result
    except BaseException as error:
        write(out/'failed.json',{'error':str(error),'automatic_retry':False,
            'preserve':'source decisions, staging files and any partially installed configs'})
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('repo','data','out'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--iteration',type=int,default=2)
    print(json.dumps(install(**vars(p.parse_args()))))

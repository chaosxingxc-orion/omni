"""Start a qualified P experiment from one data-only configuration; the existing controller owns training."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from build import pin, save
from paired_experiment import digest, graph, rank_profile, validate
from source.config_profile import resolve, bind_environment

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_bytes())


def checked(ref):
    if pin(ref['path']) != ref:
        raise ValueError('registered asset changed: ' + ref['path'])
    return Path(ref['path'])


def execute(argv, **kwargs):
    return subprocess.run(list(map(str, argv)), check=True, **kwargs)


def spec(path, site):
    request = read(path)
    if set(request) != {'schema','run_id','baseline','experiment'} or request['schema'] != 'formal-p-start-v1':
        raise ValueError('expected run_id, qualified baseline and experiment')
    run_id = request['run_id']
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,47}', run_id):
        raise ValueError('invalid run_id')
    baseline = site['baselines'][request['baseline']]
    # Operators override experiment values; source rosters and macro bindings
    # come from the registered, immutable template.
    template = request['experiment']
    if 'template' in baseline:
        def merge(defaults, overrides):
            if not isinstance(defaults,dict):return overrides
            if not isinstance(overrides,dict) or overrides.keys()-defaults.keys():
                raise ValueError('unknown experiment override')
            return {k:merge(v,overrides[k]) if k in overrides else v for k,v in defaults.items()}
        template=merge(read(checked(baseline['template'])),template)
        request=dict(request,experiment=template)
    profile = read(site['profile'])
    if profile['schema']=='omni-local-profile-v2':
        profile,_ = bind_environment(profile,os.environ)
    effective = resolve(template,profile)
    policy_name=effective.pop('policy_base')
    if Path(policy_name).name!=policy_name: raise ValueError('invalid policy name')
    effective['policy_base_sha256']=pin(HERE/policy_name)['sha256']
    readers=effective.pop('reader_sources')
    for name,member in readers.items():
        if Path(member).name!=name or not (HERE.parent/member).resolve().is_relative_to(HERE.parent):
            raise ValueError('invalid reader path: '+name)
    effective['readers']={n:pin(HERE.parent/p)['sha256'] for n,p in readers.items()}
    rows=validate(effective)
    accepted=read(checked(baseline['experiment']))
    if (rank_profile(effective)!=rank_profile(accepted) or graph(effective)!=graph(accepted) or
        any(effective[k]!=accepted[k] for k in ('model','model_path','lanes','training_devices'))):
        raise ValueError('no matching qualified initialization/graph; select a registered baseline')
    if effective['episodes_per_recording']!=accepted['episodes_per_recording']:
        raise ValueError('the registered learning Clock uses a different group size')
    return request,baseline,effective,rows


def start(path,site_path,plan=False,rehearse=False):
    site=read(site_path)
    if site.get('schema')!='formal-p-site-v1':raise ValueError('unknown site profile')
    req,b,exp,rows=spec(path,site)
    parent=Path(site['run_root']).resolve(strict=True)
    root=(parent/'rehearsals' if rehearse else parent)/req['run_id']
    run_id=req['run_id']
    source=Path(site['image_source'])
    old=read(checked(b['runner']))
    image=checked(old['image_recipe'])
    release=Path(old['release'])
    impl=release/'tree/parents/jev-system-rl/implementation'
    training=read(checked(b['training']))
    branch=checked(b['branch'])
    diagnostic=checked(b['diagnostic_ready']).parent
    binding=dict(config=pin(path),site=pin(site_path),experiment_sha256=digest(exp))
    info=dict(run_id=run_id,output=str(root),recordings_per_step=rows,training_memory_gib=exp['training_memory_gib'],
              graph=graph(exp),experiment_sha256=digest(exp),controller='run_dedicated_controller.py')
    frozen=b.get('dispatch_inputs')
    if frozen is not None:
        if (set(frozen) not in ({'static','lengths','threads'},{'static','lengths','threads','producer'}) or type(frozen['threads']) is not int
                or not 1<=frozen['threads']<=32):
            raise ValueError('invalid registered dispatch inputs')
        checked(frozen['static']);checked(frozen['lengths'])
        if 'producer' in frozen:
            producer=frozen['producer']
            if (set(producer)!={'recordings_per_macro','training_input_layout'} or type(producer['recordings_per_macro'])!=int
                    or producer['recordings_per_macro']<1 or producer['training_input_layout']!='group_refs_v1'):
                raise ValueError('invalid recording producer settings')
    if plan: return info
    if root.resolve()!=root:raise ValueError('run root escaped')
    root.mkdir(mode=0o700,parents=True,exist_ok=True)
    with Path(site['service_registry']).with_suffix('.guard').open('a') as site_guard, (root/'dispatch.guard').open('a') as guard:
        fcntl.flock(site_guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'SUBMIT-INTENT.json').exists() or (root/'run').exists():
            raise ValueError('run already submitted or outcome unknown; inspect original CURSOR; no resubmission')
        save(root/'LAUNCH-BINDING.json',binding)
        def stage(name,fn):
            t=time.time()
            print(name,flush=True)
            try: result=fn()
            except Exception as exc:
                event=dict(stage=name,started=t,elapsed_seconds=time.time()-t,error=str(exc))
                save(root/'startup-events'/(str(time.time_ns())+'.json'),event)
                print(json.dumps(event),flush=True)
                raise
            event=dict(stage=name,started=t,elapsed_seconds=time.time()-t)
            save(root/'startup-events'/(str(time.time_ns())+'.json'),event)
            print(json.dumps(event),flush=True)
            return result
        def inside(module,*args,**kwargs):
            return execute([image,source/module,*args],**kwargs)
        package=root/'package'; template=Path(save(root/'experiment.template.json',req['experiment'])['path'])
        stage('compile',lambda:inside('paired_experiment.py','--template',template,'--profile',site['profile'],
                                      '--output',package,'--release',release))
        if read(package/'PACKAGE.json')['experiment_sha256']!=digest(exp):raise ValueError('resolved experiment changed')
        stage('initial_actor',lambda:execute([image,'-c',
            'import sys;sys.path.insert(0,sys.argv[1]);from actor_start import compile_start;from run_package import write;write(sys.argv[3],compile_start(sys.argv[2],sys.argv[4]))',
            impl,branch,root/'ACTOR-START.json',run_id]))
        # The site owns the predecessor identity. Only this guard-checked pair may be replaced.
        from stage_experiment_lanes import provision
        if rehearse:
            residents=checked(b['client']).parent
            if read(residents/'READY.json')['experiment_sha256']!=digest(exp):
                raise ValueError('offline rehearsal requires matching archived service receipts')
        else:
            dep=stage('inference',lambda:provision(site,b,root,exp,old['rollout_target'],req['experiment']['reader_sources']))
            residents=root/'residents'
            stage('observe',lambda:execute([image,'-c',
                'import sys;sys.path[:0]=sys.argv[1:4];from pathlib import Path;from observe_replaced_pair import observe;observe(*(Path(v) for v in sys.argv[4:]))',
                source,impl,site['observation_helpers'],checked(b['client']),dep,residents,source]))
        cfg=dict(diagnostic=str(diagnostic),residents=str(residents),actor_start=str(root/'ACTOR-START.json'),
                 previous=b['previous_precision'],measured_experiment=str(checked(b.get('measured_experiment',b['experiment']))),
                 formal_experiment=str(package/'EXPERIMENT.json'))
        precision=root/'PRECISION.json'
        stage('precision',lambda:inside('lock_branch_precision.py',save(root/'precision-config.json',cfg)['path'],precision))
        policy=package/'TRAINING-POLICY.json'
        def result(module,*args):
            return json.loads(inside(module,*args,stdout=subprocess.PIPE).stdout)
        candidate=stage('lock',lambda:result('lock_training_candidate.py','--root',source.parent,'--policy',policy,
                      '--precision',precision,'--output',root/'candidates'))
        delta=read(release.parent/'DELTA-SPEC.json')
        build=stage('build',lambda:result('build_continuous_release.py','--root',source.parent,'--policy',policy,
                         '--locked',candidate['path'],'--precision',precision,'--base',release,'--logical',delta['logical_root'],
                         '--composer',delta['composer_ref']['path'],'--budget',delta['storage_budget_bytes'],'--output',root/'builds'))
        training.update(build=build['path'],providers=str(residents/'providers/READY.json'),package=str(package/'PACKAGE.json'),
            dataset_start_mode='new_epoch',actor_capture_root=str(Path(training['actor_capture_root']).parent/(run_id+'-capture')))
        stage('configure',lambda:inside('configure_branch_training.py','--config',save(root/'training.local.json',training)['path'],'--output',root))
        runner=root/'RUNNER.json'
        if frozen is not None:save(root/'DISPATCH-INPUTS.json',frozen)
        if rehearse:
            stage('actor_preview',lambda:execute([sys.executable,'-B',HERE/'run_dedicated_controller.py',
                   runner,read(runner)['compiler'],'--manifest',root/'MANIFEST.json','--output',root/'preview']))
            preview=dict(info,submitted=False,rehearsal=True,actor=pin(root/'preview/ACTOR.json'))
            if frozen is not None:
                from prepare_transition_config import build as build_transition
                folder=root/'dispatch-transition';folder.mkdir(exist_ok=True);candidate=folder/'TRANSITION.json'
                build_transition(runner,root/'preview/PREVIEW-NOT-FOR-LAUNCH.json',HERE,frozen['lengths']['path'],
                                 frozen['static']['path'],candidate,frozen['threads'],producer=frozen.get('producer'))
                stage('dispatch_preview',lambda:execute([sys.executable,'-B',HERE/'run_dedicated_controller.py',
                      runner,folder/'transition_controller.py',candidate,'--check']))
                preview['dispatch']=pin(candidate)
            return preview
        shutil.copyfile(HERE/'report_training_updates.py',root/'report_training_updates.py')
        save(root/'SUBMIT-INTENT.json',dict(run_id=run_id,runner=pin(runner),binding=pin(root/'LAUNCH-BINDING.json')))
        # No re-dispatch on a lost acknowledgement. The fixed controller takes the runtime guard.
        with (root/'submit.log').open('ab') as log:
            child=subprocess.Popen([sys.executable,'-B',str(HERE/'run_dedicated_controller.py'),'--submit',str(runner)],
                                   stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
        save(root/'SUBMITTED.json',dict(pid=child.pid,boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                                       intent=pin(root/'SUBMIT-INTENT.json')))
        return dict(info,submitted=True,pid=child.pid)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('config',type=Path)
    default_site=Path(os.environ.get('XDG_CONFIG_HOME',Path.home()/'.config'))/'omni-system-rl/SITE.local.json'
    p.add_argument('--site',type=Path,default=os.environ.get('OMNI_EXPERIMENT_SITE',default_site))
    p.add_argument('--plan',action='store_true')
    p.add_argument('--rehearse',action='store_true',help='build and compile against recorded matching services; no remote launch')
    a=p.parse_args()
    if not a.site.is_file():p.error('install the site profile or set OMNI_EXPERIMENT_SITE')
    print(json.dumps(start(a.config,a.site,a.plan,a.rehearse),ensure_ascii=False))

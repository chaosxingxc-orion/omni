"""Reuse the closed FSDP scoring release and bind it to every real response in a panel."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from io_utils import digest, doc, pin, require, save, storage
from decode_action_replay import choice


def stage(c):
    old = Path(c['score_template']).resolve(strict=True)
    target = Path(c['target'])
    require(target.resolve()==target and target.parent==old.parent and not target.exists(), 'new sibling required')
    cfg=doc(c['template_config'])
    require(Path(c['template_config']['path'])==old/'TRAIN.local.json' and
            doc(pin(old/'hot/EXIT.json'))['complete_ranks']==8 and cfg['devices']==list(range(8))
            and cfg['parity_restore_step']==c['restore_step'], 'unexpected training reference')
    panel=doc(pin(c['panel_source']))
    natural=doc(panel['config'])
    lane=c['lane']
    s, = [s for s in natural['services'] if s['name']==lane]
    start=doc(pin(natural['start_receipts'][lane]))
    inspected=json.loads(subprocess.check_output(['docker','inspect',s['container']]))[0]
    require(inspected['Id']==start['container_id'], 'original natural service changed')
    # docker cp works on the closed original container, including its read-only mount.
    import io, tarfile
    body=subprocess.check_output(['docker','cp',inspected['Id']+':/runtime/state/policy/ACTIVATION.json','-'])
    with tarfile.open(fileobj=io.BytesIO(body)) as bundle:
        member,=bundle.getmembers()
        activation=json.load(bundle.extractfile(member))
    ck=cfg['restore_checkpoint']
    proof=pin(Path(ck['directory'])/'COMMITTED.json')
    adapter=activation['weights_ref']['sha256']
    require(activation['alias']==activation['policy_version']=='checkpoint-sha256:'+adapter
            and activation['service_identity']['container_name']==s['container']
            and sorted(x['rank'] for x in activation['workers'])==list(range(4))
            and proof['sha256']==ck['sha256'] and
            doc(proof)['ranks'][0]['files']['lora_adapter/adapter_model.safetensors']==adapter,
            'service does not carry the restored training adapter')
    rows=[]
    for r in panel['rows']:
        w=dict(doc(r['natural']),cache_salt=panel['salt'])
        require(digest(w)==doc(r['intent'])['wire_sha256'],'different natural input')
        rr=doc(r['response'])
        b=json.loads(rr['raw'])
        ids,logs=choice(r['response'])
        require(b['model']==w['model'] and b['prompt_token_ids'] and
                len(b['prompt_token_ids'])==b['usage']['prompt_tokens'], 'missing expanded input')
        rows.append(dict(packet=dict(wire=w,wire_sha256=digest(w),request_id=b['id']),
                         response=rr['response_sha256'],prompt=b['prompt_token_ids'],
                         tokens=ids,logprobs=[x['logprob'] for x in logs]))
    storage(target,cfg['storage_budget_bytes'])
    target.mkdir(mode=0o700)
    # Link immutable release members and create new inodes for every modified member.
    names=set(cfg['files']) | {'tree','dispatch_retained_hf_boundary.py','retained_hf_fsdp_control.py'}
    for name in sorted(names-{'updated_http_parity.py','updated-http.json','retained_group_parity.py'}):
        require(Path(name).name==name, 'unexpected release member')
        src=old/name
        if name in cfg['files']:
            require(pin(src)['sha256']==cfg['files'][name], 'source drift: '+name)
        if src.is_dir():
            shutil.copytree(src,target/name,copy_function=os.link,symlinks=True)
        else:
            os.link(src,target/name)
    prior_scorer=old/'retained_group_parity.py'
    require(pin(prior_scorer)['sha256']==cfg['files'][prior_scorer.name], 'changed reference scorer')
    code=prior_scorer.read_text()
    for name in ('flips','inadmissible'):
        marker='        '+name+'=[j+1'
        require(code.count(marker)==1,'unrecognized scorer')
        code=code.replace(marker,'        '+name+"=None if r['raw_top1'] is None else [j+1")
    compile(code,prior_scorer.name,'exec')
    (target/prior_scorer.name).write_text(code)
    shutil.copyfile(Path(__file__).with_name('fsdp_prefill_panel.py'),target/'updated_http_parity.py')
    action_ref=save(target/'updated-http.json',dict(schema='natural-prefill-score-panel-v1',rows=rows,
                width=panel['width'],adapter_sha=adapter,checkpoint_sha=proof['sha256'],training_use=False))
    cfg.update(name=target.name,hot=str(target/'hot'))
    for name in ('updated_http_parity.py','updated-http.json','retained_group_parity.py'):
        cfg['files'][name]=pin(target/name)['sha256']
    save(target/'TRAIN.local.json',cfg)
    save(target/'PANEL-SOURCE.json',dict(source=pin(c['panel_source']),natural=panel['config'],panel=action_ref,
         activation_sha256=digest(activation),checkpoint=proof,reference=c['template_config']))


if __name__=='__main__':
    stage(doc(pin(sys.argv[1])))

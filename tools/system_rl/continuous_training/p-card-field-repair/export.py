"""Consumable streaming export and unmodified original P renderer adapter.

Use iter_supplies() for a full export; get_supply() is a convenience for bounded
CPU work, not an O(N) per-request production lookup.
"""
import copy
import json
from pathlib import Path
import sys
import types

sys.dont_write_bytecode=True
from repair import HERE, pm, canonical, sha, permanent_labels


def iter_supplies(root):
    root=Path(root).resolve()
    marker=json.loads((root/'COMPLETE.json').read_bytes())
    refs={Path(p['path']).name:p for p in marker['artifacts']}
    reader=pm.Reader([root])
    # The file can be relocated only by a separately verified publication map.
    ref=dict(refs['supplies-v1.private.jsonl'],path=str(root/'supplies-v1.private.jsonl'))
    for _,_,row in reader.lines(ref):
        assert row['status']=='complete_offline_text_supply' and not row['gaps']
        assert not any(row[k] for k in ('current_qualification','dispatch_allowed','P_consumed'))
        yield row


def get_supply(root, reference_id):
    found=None
    for row in iter_supplies(root):
        if row['reference_id']==reference_id:
            assert found is None
            found=row
    if found is None: raise KeyError(reference_id)
    return found


class Renderer:
    def __init__(self,config=HERE/'CONFIG.json'):
        self.cfg=json.loads(Path(config).read_bytes())
        self.reader=pm.Reader(self.cfg['readonly_roots'])
        self.seed=self.reader.doc(self.cfg['renderer']['p_seed'])
        self.prompt=self.reader.doc(self.cfg['renderer']['p_prompt_config'])
        old=sys.modules.get('common')
        self.modules={}
        try:
            for key in ('p_common','p_renderer','p_prompter'):
                ref=self.cfg['renderer'][key]
                obj=types.ModuleType('repair_'+key)
                exec(compile(self.reader.raw(ref),ref['path'],'exec'),obj.__dict__)
                self.modules[key]=obj
                if key=='p_common':sys.modules['common']=obj
        finally:
            if old is None:sys.modules.pop('common',None)
            else:sys.modules['common']=old

        self.permanent=permanent_labels(self.cfg)
        frozen=self.modules['p_common'].LABELS
        if set(self.permanent)!=frozen or self.permanent!=[x['label'] for x in self.seed['job']['ranking'] if x['label'] in frozen]:
            raise ValueError('permanent knowledge roster differs from frozen renderer/seed')

    def render(self,row,audio):
        assert row['status']=='complete_offline_text_supply' and not row['gaps']
        supply=row['supply']; labels=supply['dynamic_labels']; permanent=self.permanent
        assert set(supply['cards'])==set(labels+permanent) and set(supply['fewshots'])==set(labels)
        seed=copy.deepcopy(self.seed['wire']);job=copy.deepcopy(self.seed['job'])
        # The frozen renderer asserts that permanent cards equal its input seed.
        # Derive a versioned seed by rebinding exactly those two text/hash fields;
        # do not bypass assertions or modify original renderer source.
        changed=[]
        for item in job['ranking']:
            if item['label'] in permanent:
                before=item['content_sha256'];item['content_sha256']=sha(supply['cards'][item['label']]['value']['content'].encode())
                changed.append(dict(label=item['label'],before=before,after=item['content_sha256']))
        seed['messages'][1]['content'][1]['text']='Permanent supplied knowledge (not retrieval evidence):\n'+'\n\n'.join(
            '['+str(x['index'])+'] '+supply['cards'][x['label']]['value']['content'] for x in job['ranking'] if x['label'] in permanent)
        seed['messages'][1]['content'][0]=copy.deepcopy(audio)
        wire,ranking=self.modules['p_renderer'].render(seed,job,labels,supply['cards'],supply['fewshots'])
        wire,_,_=self.modules['p_prompter'].render(wire,ranking,dict(empty=not labels,supported_labels=labels),self.prompt)
        assert wire.pop('chat_template_kwargs')=={'enable_thinking':False}
        wire['model']='qwen3-omni-30b-a3b-instruct'
        schema=wire['response_format']['json_schema']['schema']
        schema['minItems']=schema['maxItems']=min(3,len(ranking))
        assert wire['messages'][1]['content'][0]==audio
        assert all(wire[k]==self.seed['wire'][k] for k in ('max_tokens','temperature','seed'))
        assert [r['label'] for r in ranking]==labels+permanent
        return dict(schema='p-card-field-repair-CPU-wire-v1',reference_id=row['reference_id'],wire=wire,
            wire_sha256=sha(canonical(wire)),ranking=ranking,
            supply_knowledge_ids={l:c['provenance']['knowledge_id'] for l,c in supply['cards'].items()},
            seed_derivation=dict(parent_seed=self.cfg['renderer']['p_seed'],permanent_text_hash_rebinding=changed,
                audio_object_sha256=sha(canonical(audio)),original_renderer_source_unchanged=True),
            thinking_policy={'enable_thinking':False},current_qualification=False,dispatch_allowed=False,P_consumed=False,
            note='Real audio CPU wire only. New request, no paid response, no tokenizer/FIT qualification.')

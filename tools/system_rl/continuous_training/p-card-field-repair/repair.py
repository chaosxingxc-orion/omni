"""CPU-only field repair and query-specific provenance views, no vector loading."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from collections import Counter

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parent.parent
def permanent_labels(cfg):
    labels = cfg['policy']['permanent_fixed_only']
    if not isinstance(labels,list) or not all(isinstance(x,str) and x for x in labels) or len(labels)!=len(set(labels)):
        raise ValueError('invalid permanent knowledge roster')
    return labels


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


pm = module(BASE/'p-supply-materializer/materialize.py', 'field_repair_io')
canonical, sha = pm.canonical, pm.sha


def pin(p):
    p = Path(p).resolve()
    h = hashlib.sha256()
    with p.open('rb') as f:
        for raw in iter(lambda: f.read(1024**2), b''): h.update(raw)
    return dict(path=str(p), sha256=h.hexdigest(), bytes=p.stat().st_size)


class Repair:
    def __init__(self, config):
        self.cfg = json.loads(Path(config).read_bytes())
        self.permanent = permanent_labels(self.cfg)
        self.reader = pm.Reader(self.cfg['readonly_roots'])
        self.reader.raw(pin(config))
        i = self.cfg['inputs']
        self.spec = self.reader.doc(i['field_spec'])
        self.catalog = self.reader.doc(i['card_catalog'])['labels']
        self.registry = {r['source_id']:r for _,_,r in self.reader.lines(i['source_registry'])}
        self.groups = self.reader.doc(i['groups'])['groups']
        self.members = self.reader.doc(i['members'])
        # Complete original TRAIN metadata, not just the locked example bank.
        self.evidence = {}
        self.train_count = 0
        for n,h,r in self.reader.lines(i['support_text']):
            self.train_count += 1
            u = str(r['slurp_id'])
            reg = self.registry['train_real:'+u]
            self.evidence[u] = dict(source_id='train_real:'+u, utterance_id=u,
                text=r['sentence'], text_sha256=sha(r['sentence'].encode()),
                annotation=r['sentence_annotation'], entities=r['entities'], intent=r['intent'],
                text_origin='official TRAIN metadata reference; not ASR; no audio review',
                source_ref=dict(i['support_text'],line=n,line_sha256=h),
                registry_record=reg, source_group=self.groups.get(u),
                registry_ref=i['source_registry'])
        self.validate_spec()

    def validate_spec(self):
        for label, fields in self.spec['fields'].items():
            for field, variants in fields.items():
                for v in variants:
                    assert v.get('policy') or 'groups' in v
                    for group in v.get('groups',[]):
                        assert group
                        for uid in group:
                            e = self.evidence[str(uid)]
                            assert e['registry_record']['decision'] == 'allowed', (label,field,uid)
                            assert e['source_group'] is not None

    def legal(self, uid, row):
        e = self.evidence[str(uid)]
        if e['registry_record']['decision'] != 'allowed': return False
        if row is None: return True
        return (e['source_group'] not in row['excluded_source_groups'] and
                e['registry_record']['component_id'] != row['identity']['component_id'])

    def view(self, label, row=None):
        original = self.catalog[label]['original_card']
        if label not in self.spec['fields']:
            deps = self.catalog[label]['dependency_utterance_ids']
            assert all(self.legal(u,row) for u in deps), (label,'unrepaired dependency')
            return dict(value=dict(content=original['value']['content']), provenance=dict(
                kind='preserved_original',parent_catalog=self.cfg['inputs']['card_catalog'],
                pointer='/labels/'+label,source_ids=['train_real:'+u for u in deps],
                knowledge_id=original['knowledge_id'],version='original_sealed'))
        fields = self.spec['fields'][label]
        selected = {}
        receipts = []
        omitted = []
        missing = []
        for field, variants in fields.items():
            selected[field] = []
            for idx, v in enumerate(variants):
                chosen = []
                failures = []
                if v['text'] and not v.get('policy'):
                    for gi, group in enumerate(v['groups']):
                        allowed = [str(u) for u in group if self.legal(u,row)]
                        if not allowed: failures.append(gi)
                        else: chosen.append(allowed[0])
                usable = bool(v['text']) and not failures
                # Activation variants are alternatives, not cumulative rules.
                if field == 'activation' and selected[field]: continue
                if usable:
                    selected[field].append(v['text'])
                    receipts.append(dict(field=field,variant=idx,text=v['text'],
                        evidence_source_ids=['train_real:'+u for u in sorted(set(chosen),key=int)],
                        policy=v.get('policy',False),why=v.get('why','Preserve existing authoring policy.')))
                else:
                    omitted.append(dict(field=field,variant=idx,text=v['text'],missing_groups=failures,
                        why=v.get('why'),reason='removed_in_spec' if not v['text'] else 'query_self_evidence_unavailable'))
            if field == 'activation' and not selected[field]: missing.append('activation')
        assert selected['uncertainty'] and selected['limitations'], 'policy guard lost'
        text = ('Intent '+label+'.\nActivation boundary:\n'+'\n'.join(selected['activation'])+
                '\nConfusable exclusions:\n'+'\n'.join(selected['confusable'])+
                '\nUncertainty behavior: '+'\n'.join(selected['uncertainty'])+
                '\nLimitations:\n'+'\n'.join(selected['limitations']))
        used = sorted({s for r in receipts for s in r['evidence_source_ids']})
        version_hash = sha(canonical(dict(label=label,text=text,receipts=receipts)))
        return dict(value=dict(content=text),provenance=dict(
            kind='field_recompiled_query_safe',knowledge_id='p-card-field-repair-v1::'+label+'::'+version_hash[:20],
            version='p-card-field-repair-v1',content_sha256=sha(text.encode()),
            parent_knowledge_id=original['knowledge_id'],parent_catalog=self.cfg['inputs']['card_catalog'],
            parent_pointer='/labels/'+label,fields=receipts,source_ids=used,omitted=omitted,
            gaps=missing,classification_guard_status='complete' if not missing else 'pending',
            embedding_binding=None))

    def assemble(self, row, candidate):
        assert row['reference_id'] == candidate['reference_id']
        old = row['supply'] or row['partial_supply']
        labels = candidate['result']['actual_returned_ranked_intent_ids']
        assert labels == old['dynamic_labels'] and not set(labels)&set(self.permanent)
        supply = dict(dynamic_labels=labels,cards={},fewshots=old['fewshots'],
            retrieval_event_ref=dict(self.cfg['inputs']['q_candidates'],reference_id=row['reference_id']))
        gaps = []
        for label in labels+self.permanent:
            card = self.view(label,row)
            supply['cards'][label] = card
            if card['provenance'].get('gaps'): gaps.append(dict(label=label,fields=card['provenance']['gaps']))
        for label, ex in supply['fewshots'].items():
            if ex is None: continue
            selection = row['fewshot_audit'][label]['selected']
            uid = str(selection['utterance_id'])
            # Metadata contains legacy bare action names (e.g. addcontact),
            # whereas the frozen bank carries the canonical scenario label.
            member = self.members[selection['member_index']]
            assert self.legal(uid,row) and member['intent']==label and str(member['utterance_id'])==uid
            assert sha(ex['text'].encode()) == self.evidence[uid]['text_sha256']
        return dict(schema='p-card-field-repair-supply-v1',reference_id=row['reference_id'],
            identity=row['identity'],historical_sample_id=row['historical_sample_id'],
            audio_identity=row['audio_identity'],excluded_source_groups=row['excluded_source_groups'],
            source_mask_sha256=row['source_mask_sha256'],supply=supply,
            status='complete_offline_text_supply' if not gaps else 'pending_field_support',gaps=gaps,
            fewshot_audit=row['fewshot_audit'],current_qualification=False,dispatch_allowed=False,P_consumed=False,
            scientific_baseline=self.cfg['scientific_baseline'],
            retrieval_binding='frozen Q-only candidates and old D description vectors; supply version is independent')


def run(config, hotroot):
    wall,cpu = time.monotonic(),time.process_time()
    repair = Repair(config)
    io = pm.LocalIO(hotroot,repair.cfg)
    spec = repair.spec
    io.save('STORAGE.json',dict(preflight=io.preflight,HOME_retained=os.environ['HOME']))
    # Inventory/pins make explicit that new text is not bound to any old vector.
    cards = {l:repair.view(l) for l in repair.catalog}
    io.save('knowledge-cards.private.json',dict(schema='p-card-field-repair-knowledge-v1',labels=cards))
    io.save('TRAIN-EVIDENCE.private.json',dict(train_rows_read=repair.train_count,
        sources={u:e for u,e in repair.evidence.items() if any(int(u) in g for f in spec['fields'].values() for vs in f.values() for v in vs for g in v.get('groups',[]))}))
    diffs = []
    tasks = []
    for label, fields in spec['fields'].items():
        original = repair.catalog[label]['original_card']
        new = cards[label]
        oldtext = original['value']['content']
        old_fields = dict(activation=oldtext.split('Activation boundary:\n')[1].split('\nConfusable exclusions:')[0],
            confusable=oldtext.split('Confusable exclusions:\n')[1].split('\nUncertainty behavior:')[0],
            uncertainty=oldtext.split('Uncertainty behavior: ')[1].split('\nLimitations:')[0],
            limitations=oldtext.split('Limitations:\n')[1])
        for field in fields:
            parts = [r for r in new['provenance']['fields'] if r['field']==field]
            after = '\n'.join(r['text'] for r in parts)
            diffs.append(dict(label=label,field=field,before=old_fields[field],after=after,
                text_exact=old_fields[field]==after,why=[v.get('why','existing guard policy preserved') for v in fields[field]],
                evidence=parts,removed_or_unavailable=[r for r in new['provenance']['omitted'] if r['field']==field],
                old_knowledge_id=original['knowledge_id'],new_knowledge_id=new['provenance']['knowledge_id']))
        activation = next(r['text'] for r in new['provenance']['fields'] if r['field']=='activation').removeprefix('- ')
        target_description = label+' '+activation
        old_description = original['facet']['representation']
        tasks.append(dict(label=label,old_knowledge_id=original['knowledge_id'],
            old_description=old_description,old_description_sha256=sha(old_description.encode()),
            proposed_description=target_description,proposed_description_sha256=sha(target_description.encode()),
            supply_knowledge_id=new['provenance']['knowledge_id'],description_changed=old_description!=target_description,
            old_vectors_bind_only='old_knowledge_id and exact old_description',
            new_vector_sha256=None,new_embedding_executed=False,
            status='pending_new_embedding_for_changed_description' if old_description!=target_description else 'unchanged_description_old_retrieval_identity_only',
            query_specific_description_changes_require_separate_task=True))
    io.save('FIELD-DIFF.private.json',dict(fields=diffs))
    io.save('EMBEDDING-TASKS.json',dict(schema='independent-retrieval-successor-tasks-v1',tasks=tasks,
        changed_description_tasks=sum(t['description_changed'] for t in tasks),
        dispatch_allowed=False,mixing_risk='Do not label frozen old-D selection as retrieval against repaired P supply. Query views may have different text; never attach one global new vector to all variants.'))
    count = Counter(); perlabel = Counter(); views = {}; shrink = []; self_rows = []
    candidates = repair.reader.lines(repair.cfg['inputs']['q_candidates'])
    with io.writer('supplies-v1.private.jsonl') as out, io.writer('assembly-v1.private.jsonl') as audit:
        for _,_,row in repair.reader.lines(repair.cfg['inputs']['q_supplies']):
            _,_,candidate = next(candidates)
            new = repair.assemble(row,candidate)
            old = row['supply'] or row['partial_supply']
            changed = []; preserved = []; ids = {}
            for label,card in new['supply']['cards'].items():
                p = card['provenance']; perlabel[label] += 1
                oldtext = repair.catalog[label]['original_card']['value']['content']
                (preserved if card['value']['content']==oldtext else changed).append(label)
                if label not in spec['fields']:
                    assert card['value']==old['cards'][label]['value']; count['unaffected_card_values_exact'] += 1
                if p['kind']=='field_recompiled_query_safe':
                    assert all(repair.legal(s.split(':')[1],row) for s in p['source_ids'])
                    kid=p['knowledge_id'];ids[label]=kid
                    views[kid]=card
                    if kid!=cards[label]['provenance']['knowledge_id']:
                        shrink.append(dict(reference_id=row['reference_id'],label=label,knowledge_id=kid,
                            base_knowledge_id=cards[label]['provenance']['knowledge_id'],
                            text_changed=card['value']!=cards[label]['value'],omitted=p['omitted']))
                else:
                    assert all(repair.legal(s.split(':')[1],row) for s in p['source_ids'])
            original_self = {l:a['source_conflicts'] for l,a in row['card_audit'].items()
                if l in repair.permanent and any(any('query_' in why for why in reasons) for reasons in a['source_conflicts'].values())}
            if original_self: self_rows.append(dict(reference_id=row['reference_id'],old_conflicts=original_self,new_views={l:ids[l] for l in original_self}))
            assert new['supply']['fewshots']==old['fewshots'] and new['fewshot_audit']==row['fewshot_audit']
            count['fewshots_exact_including_None']+=len(old['fewshots'])
            count['None_exact']+=sum(v is None for v in old['fewshots'].values())
            count['complete' if not new['gaps'] else 'pending']+=1
            count['previous_pending_resolved']+=row['status']=='pending_supply_evidence' and not new['gaps']
            count['rows']+=1
            out.write(canonical(new)+b'\n')
            audit.write(canonical(dict(reference_id=row['reference_id'],dynamic_labels_exact=True,
                result_sha256=sha(canonical(candidate['result'])),candidate_line_sha256=sha(canonical(candidate)),
                labels=new['supply']['dynamic_labels'],new_knowledge_ids=ids,changed_card_text=changed,
                preserved_card_text=preserved,fewshots_exact=True,gaps=new['gaps'],
                permanent_only=repair.permanent,query_self_exclusion=True,all_new_dependencies_allowed=True))+b'\n')
            if count['rows']%256==0:io.check()
    assert next(candidates,None) is None
    assert count['rows']==9705
    io.save('query-views.private.json',dict(schema='query-specific-card-views-v1',views=views))
    io.save('query-view-diff.private.json',dict(rows=shrink))
    view_tasks=[]
    for kid,card in views.items():
        label=kid.split('::')[1]
        activation=next(r['text'] for r in card['provenance']['fields'] if r['field']=='activation').removeprefix('- ')
        old_description=repair.catalog[label]['original_card']['facet']['representation']
        description=label+' '+activation
        if description!=old_description:
            view_tasks.append(dict(supply_knowledge_id=kid,label=label,proposed_description=description,
                proposed_description_sha256=sha(description.encode()),old_description=old_description,
                old_description_sha256=sha(old_description.encode()),new_vector_sha256=None,
                status='pending_new_embedding',dispatch_allowed=False))
    io.save('QUERY-VIEW-EMBEDDING-TASKS.json',dict(tasks=view_tasks,
        scope='Exact changed query-view description identities only; no embedding performed.'))
    io.save('permanent-self-repair.private.json',dict(rows=self_rows,queries=len(self_rows),
        old_card_occurrences=sum(len(r['old_conflicts']) for r in self_rows),
        old_source_dependency_occurrences=sum(len(v) for r in self_rows for v in r['old_conflicts'].values()),
        note='Count source dependencies separately from card occurrences and reason flags.'))
    # Precise, finite removed clauses; absence is not a blocker for remaining text.
    io.save('FIELD-GAPS.json',dict(removed_detail=[dict(label='alarm_remove',field='activation',detail='recurring',reason='No allowed wording in selected full-TRAIN evidence establishes recurrence.'),
        dict(label='cooking_query',field='activation/confusable',detail='quickest method / method tendency',reason='Only two full-TRAIN allowed sources; quickest source remains pending.'),
        dict(label='cooking_query',field='limitations',detail='three selected rows and recipe member',reason='Historical source recipe is pending; old sample-count claim removed.'),
        dict(label='iot_coffee',field='confusable',detail='home qualifier',reason='Machine evidence does not establish home location.'),
        dict(label='takeaway_order',field='limitations',detail='business-name annotation not occurring in words',reason='The exact observation depended on excluded 3048; do not reproduce it in new supplied content.')],
        unresolved_registry_sources=['train_real:'+u for u in ['13479','14736','2229','3828','4713','4637']],
        registry_unchanged=True,new_supply_blockers=count['pending'],
        query_specific_omissions=len([x for x in shrink if x['text_changed']]),
        unsupported_detail_is_removed_not_new_business_rule=True))
    cost=dict(wall_seconds=time.monotonic()-wall,cpu_seconds=time.process_time()-cpu,
        seconds_per_row=(time.monotonic()-wall)/9705,cards=0,instances=1,model_calls=0,embedding_calls=0,
        distance_computations=0,training_steps=0,remote_calls=0,CLI_agent_spawned=0,currency_CPU_cost=None,
        completed=9705,total=9705,remaining=0,ETA_seconds=0,formal_fixed_image_test=False)
    io.save('SUMMARY.json',dict(counts=dict(count),per_label_card_occurrences=dict(perlabel),
        all_labels_preserved=True,frozen_candidate_rows=9705,query_views=len(views),
        query_specific_variants=len(shrink),new_supply_complete=count['complete'],new_supply_pending=count['pending'],
        cost=cost,current_qualification=False,dispatch_allowed=False,P_consumed=False))
    io.save('READ-PINS.json',dict(files=list(repair.reader.pins.values())))
    io.save('COMPLETE.json',dict(status='SEALED_CPU_FIELD_REPAIR_SUPPLY',
        artifacts=[io.pin(p.name) for p in sorted(io.root.glob('*.json*'))],watermark=9705,
        source_boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        current_qualification=False,dispatch_allowed=False))
    print(json.dumps(dict(counts=dict(count),query_views=len(views),cost=cost)))


def deny_network(event,args):
    if event in ('socket.connect','socket.getaddrinfo','subprocess.Popen'):
        raise RuntimeError('CPU field repair forbids '+event)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default=str(HERE/'CONFIG.json'));p.add_argument('--hotroot',required=True)
    a=p.parse_args();sys.addaudithook(deny_network);run(a.config,a.hotroot)

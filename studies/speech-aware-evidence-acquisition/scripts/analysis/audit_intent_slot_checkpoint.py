"""Audit the first complete native P/Slot/Decision row without model calls."""
import argparse,hashlib,json,sys,time
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from speech_aware_evidence_acquisition.core.row_checkpoint import read_checkpoint
from speech_aware_evidence_acquisition.rails.intent_slot_pipeline import validate_receipt


def classify(row):
    defect=row.keep_defect()
    if defect is not None:return 'unrendered',None,None
    if len(row.output_rows)!=1 or not isinstance(row.output_rows[0],dict):raise ValueError('expected exactly one output row')
    out=row.output_rows[0];receipt=out.get('native_episode_receipt')
    if not isinstance(receipt,dict) or not isinstance(receipt.get('role_visits'),list):raise ValueError('malformed native episode receipt')
    visits=receipt['role_visits']; stages=[]
    for visit in visits:
        if not isinstance(visit,dict) or not isinstance(visit.get('stage'),str):raise ValueError('malformed role visit')
        stages.append(visit['stage'])
    disposition=receipt.get('terminal_disposition')
    if disposition is not None:
        if not isinstance(disposition,dict):raise ValueError('malformed terminal disposition')
        reason=disposition.get('reason')
        return ('finite_completion' if reason=='finite_completion' else 'refusal' if reason=='seal_refusal' else 'terminal_other'),out,receipt
    if stages!=['perception','slot','solver']:
        raise ValueError('successful row lacks ordered P/Slot/Decision visits')
    if not isinstance(receipt.get('intent_slot_pipeline'),dict):raise ValueError('complete row lacks pipeline receipt')
    return 'complete',out,receipt


def audit(path):
    raw=Path(path).read_bytes()
    if not raw:
        return {'status':'waiting','reason':'empty checkpoint','checkpoint_sha256':hashlib.sha256(raw).hexdigest(),'terminal_rows':0,'complete_rows':0,'classifications':{},'torn_tail_bytes':0,'new_model_calls':0}
    doc=read_checkpoint(path);counts=Counter();selected=None
    for row in doc.rows:
        kind,out,receipt=classify(row);counts[kind]+=1
        if kind=='complete' and selected is None:selected=(row,out,receipt)
    result={'status':'waiting' if selected is None else 'complete','checkpoint_sha256':doc.sha256,
            'checkpoint_rows':len(doc.rows),'terminal_rows':len(doc.rows)-counts['unrendered'],'classifications':dict(counts),'complete_rows':counts['complete'],
            'torn_tail_bytes':doc.torn_tail_bytes,'new_model_calls':0}
    if selected is None:return result
    row,out,receipt=selected; records=[json.loads(x) for x in row.trace_lines]
    if any(not isinstance(x,dict) or 'record_id' not in x or 'payload' not in x for x in records):raise ValueError('malformed trace record')
    trace={v['record_id']:v['payload'] for v in records}
    if len(trace)!=len(records):raise ValueError('duplicate trace record ID')
    validate_receipt(receipt,trace_records=trace);visits=[]
    for visit in receipt['role_visits']:
        source=visit['execution_receipt']['final_contact'];rid=source['request_id'];req=trace[rid+'/request'];resp=trace[rid+'/response'];payload=req['payload']
        if hashlib.sha256(json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()!=req['payload_sha256'] or req['payload_sha256']!=source['request_sha256'] or resp['response_text_sha256']!=source['response_text_sha256'] or hashlib.sha256(resp['text'].encode()).hexdigest()!=source['response_text_sha256'] or payload.get('sample_id')!=row.sample_id:raise ValueError('visit source or sample identity differs')
        messages=payload.get('messages',payload.get('rendered',{}).get('messages',[]));parts=[p for msg in messages for p in msg.get('content',[]) if isinstance(p,dict)]
        visits.append({'stage':visit['stage'],'source':source,'request_sample_id_matches':True,'native_projection':trace.get(rid+'/native-evidence-projection'),'knowledge_views':[{'evidence_id':v['evidence_id'],'source_uri':v.get('source_uri'),'view_sha256':hashlib.sha256(v['text'].encode()).hexdigest()} for v in parts if 'evidence_id' in v and 'text' in v]})
    last=receipt['role_visits'][-1]['execution_receipt']['final_contact'];raw=trace[last['request_id']+'/response']['text']
    if not isinstance(out.get('text'),str):raise ValueError('missing delivered text')
    result['first_complete']={'sample_id':row.sample_id,'stages':[v['stage'] for v in visits],'visits':visits,'pipeline':receipt.get('intent_slot_pipeline'),'raw_solver_sha256':hashlib.sha256(raw.encode()).hexdigest(),'delivered_sha256':hashlib.sha256(out['text'].encode()).hexdigest(),'raw_equals_delivered_bytes':raw==out['text'],'cost':row.cost}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('checkpoint',type=Path);p.add_argument('--output',type=Path);p.add_argument('--watch',action='store_true');p.add_argument('--event-module',type=Path);a=p.parse_args()
    if a.watch:
        if not a.output or not a.event_module: p.error('--watch requires --output and --event-module')
        sys.path.insert(0,str(a.event_module));from generation_release_handoff import DirectoryEvents
        while True:
            parent=a.checkpoint.parent
            while not parent.exists():parent=parent.parent
            events=DirectoryEvents(parent)
            try:
                result=audit(a.checkpoint) if a.checkpoint.exists() else {'status':'waiting','reason':'checkpoint absent','new_model_calls':0}
                if result['status']=='complete':break
                events.wait(600)
            finally:events.close()
    else:result=audit(a.checkpoint)
    text=json.dumps(result,sort_keys=True,indent=2)+'\n'
    if a.output:
        from speech_aware_evidence_acquisition.core.atomic_file import publish_new
        a.output.parent.mkdir(parents=True,exist_ok=True);publish_new(a.output,text.encode())
    print(text,end='')
if __name__=='__main__':main()

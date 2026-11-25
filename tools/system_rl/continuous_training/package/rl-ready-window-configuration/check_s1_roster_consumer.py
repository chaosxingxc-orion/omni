"""Read real historical S1 receipts through the roster join and fresh-capture packet reader."""
import argparse,base64,json,sys
from pathlib import Path
from source.panel_receipts import collection_calls
from build import save,pin
from prepare_s1_recordings import checked,needed,read
from prepare_s1_training_inputs import pcm
from seal_s1_training_roster import member
from s1_panel_selection import classify,packet_parser


def check(history,release,output):
    from collections import Counter
    import tempfile
    h=read(history);seal=read(checked(h['seal']));v=read(checked(seal['version']))
    ds=read(checked(h['dataset']));meta={d['domain']:(d['metadata']['path'],[json.loads(line) for line in checked(d['metadata']).open()]) for d in ds['domains']}
    sys.path.insert(0,str(release/'tree/parents/fresh-cohort-successor'))
    from source.fresh_material import node_screen_packet
    from source.config_profile import digest
    parse=packet_parser();counts=Counter();reject=0
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        tmp=Path(tmp)
        for row in collection_calls(v):
            if row['stage']!='S1' or row['status']!='known':continue
            p=row['packet'];b=p['source_binding'];rid=b['reference_id'];dom,line,i=rid.split(':')
            path,rows=meta[dom];raw=rows[int(line)-1];sha=digest(raw)
            allowed=p['allowed_types'];gold={e['type'] for e in raw['entities']}
            sup=dict(annotated_types=allowed,present_types=[t for t in allowed if t in gold])
            kind,_,_=classify(p,sup,row['receipt'],parse);key=tmp/p['request_id']
            packet=save(key/'packet.json',p);parts=[part for m in p['wire']['messages'] if isinstance(m['content'],list) for part in m['content'] if part.get('type')=='input_audio']
            needed(len(parts)==1 and b['source_id']==dom+':'+str(raw['slurp_id']) and p['sample_id'].endswith(':'+Path(raw['recordings'][int(i)]['file']).stem),'wrong recording')
            origin=dict(recording_ref_id=rid,source_id=b['source_id'],domain=dom,split='TRAIN',audio_ref=pcm(tmp/'media',base64.b64decode(parts[0]['input_audio']['data'],validate=True)))
            record=dict(recording_ref_id=rid,source_id=b['source_id'],sample_id=p['sample_id'],domain=dom,split='TRAIN',online=dict(node_input_ref=packet))
            static=dict(reference_id=rid,fold=p['branch'],source_record_sha256=sha,record_ref=save(key/'static.json',record),recording_origin_ref=save(key/'origin.json',origin))
            s=dict(reference_id=rid,component=b['source_component'],supervision=sup,packet=packet,fold=p['branch'],
                   panel_conditional_inclusion=None,fold_probability=None,outcome_conditional_inclusion=None)
            obs=dict(receipt=row['receipt']['native_envelope'],outcome=kind,source_wire=p['wire_sha256'],seal=h['seal'])
            entry=member(s,obs,static,key);draw=dict(record=read(checked(entry['record_ref'])))
            annotation=dict(raw,recording_ref_id=rid,source_record_sha256=sha)
            snap=dict(rollout_scope='node',target_nodes=['S1'])
            needed(node_screen_packet(draw,snap,annotation,origin)==p,'consumer did not return the original packet')
            try:member(dict(s,fold='incorrect-fold'),obs,static,key)
            except ValueError:reject+=1
            else:raise AssertionError('wrong fold accepted')
            counts[kind]+=1
    needed(dict(counts)==h['historical_replay'] and sum(counts.values())==reject>0,'incomplete historical replay')
    return save(output,dict(historical_receipt=pin(history),consumer_counts=dict(counts),wrong_fold_rejected=reject,
                            current_panel_replies=0,model_calls=0,formal_training_set=False))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('history','release','output'):p.add_argument(k,type=Path)
    a=p.parse_args();print(json.dumps(check(a.history,a.release,a.output)))

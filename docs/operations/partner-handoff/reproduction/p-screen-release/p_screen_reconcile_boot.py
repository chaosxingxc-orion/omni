"""Archive a closed wave rejected by the pre-generation boot check, then reopen its recordings."""
import fcntl
import json
import sqlite3
import sys
from pathlib import Path

from dataset_encoder import doc


def reconcile(spec):
    root = Path(spec['output'])
    first = spec['last_good_rowid']
    with (root/'SCREEN.guard').open('rb') as guard, sqlite3.connect(root/'p-screen.sqlite') as db:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        db.execute('begin immediate')
        rows = db.execute('select c.rowid,c.rid,c.status,c.finished,c.raw,b.engine_id '
                          'from calls c left join screen_bindings b using(rid) where c.rowid>?', (first,)).fetchall()
        if len(rows) != spec['requests'] or not rows:
            raise ValueError('different rejected wave')
        for _, _, status, finished, raw, engine in rows:
            if (status != 'http_failure' or finished is None or engine != spec['runtime'] or
                    json.loads(raw) != {'error':{'message':'ROLLOUT_BOOT_CHANGED','type':'BadRequestError',
                                                 'param':None,'code':400}}):
                raise ValueError('cannot reconcile an unknown or generated response')
        db.execute('create table if not exists screen_pre_generation_refusals as '
                   'select c.*,b.source_wire,b.model,b.engine_id,c.rowid as old_rowid from calls c '
                   'join screen_bindings b using(rid) where 0')
        db.execute('insert into screen_pre_generation_refusals select c.*,b.source_wire,b.model,b.engine_id,c.rowid '
                   'from calls c join screen_bindings b using(rid) where c.rowid>?',(first,))
        db.execute('delete from screen_bindings where rid in (select rid from calls where rowid>?)',(first,))
        db.execute('delete from calls where rowid>?',(first,))
        db.commit()


if __name__ == '__main__':
    reconcile(doc(sys.argv[1]))

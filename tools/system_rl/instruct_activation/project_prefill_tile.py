"""Project a single prefill tile change onto an isolated locked TP4 lane."""
import sys
from pathlib import Path
from decode_lineage import digest, snapshot, changes
from project_screen_lanes import project, read, save


def build(spec):
    size=spec['prefill_tile_rows']
    if type(size) is not int or not 16<=size<=512 or size%16:
        raise ValueError('unsupported prefill tile')
    proj=spec['projection']; lock=Path(spec['lock'])
    base=read(lock)['runtime']
    if size==base['rows']['decode_layer_graph']['prefill_tile_rows']:
        raise ValueError('candidate equals baseline')
    project(proj,lock)
    root=Path(proj['target']); rows_path=root/'bundle/ROWS.json'
    rows=read(rows_path)
    rows['decode_layer_graph']['prefill_tile_rows']=size
    save(rows_path,rows)
    for lane in proj['lanes']:
        path=root/'bundle'/('SERVER-'+lane+'.json')
        server=read(path)
        server['local_activation']['rows_ref'].update(bytes=rows_path.stat().st_size,sha256=digest(rows_path))
        save(path,server)
    delta=changes(base,snapshot(root))
    assert not delta['engine'] and not delta['files'] and list(delta['rows'])==['decode_layer_graph']
    assert delta['rows']['decode_layer_graph']['new']==dict(base['rows']['decode_layer_graph'],prefill_tile_rows=size)
    save(root/'CANDIDATE.json',dict(parent_sha256=digest(lock),changes=delta))

if __name__=='__main__':build(read(sys.argv[1]))

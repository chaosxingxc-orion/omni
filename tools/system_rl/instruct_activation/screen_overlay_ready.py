"""Check the measured serving packet and the actual screen process before and after its first batch."""
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

from cutover import load
from decode_lineage import snapshot
from io_utils import digest, doc, pin, require


def check(packet, lane, attempt, alias, runtime, switch, phase):
    packet = Path(packet)
    launcher = load(packet / ('lane-' + lane), 'screen_launcher')
    cfg = launcher.C
    name = launcher.name(attempt)
    started = doc(pin(Path(switch) / ('STARTED-' + str(cfg['port']) + '.json')))
    state = launcher.inspect(name)
    require(state['Id'] == started['container_id'] and state['State']['Running']
            and state['Image'] == cfg['image'] and started['physical_devices'] == cfg['cards'],
            'screen service identity changed')
    require(digest(snapshot(packet)) == runtime, 'different screen runtime')
    server = doc(pin(cfg['policy_overlay']['server']))
    activation = server['local_activation']
    require((activation['source_boot'],activation['guard'],activation['model']) ==
            (cfg['host_boot_id'],cfg['guard'],cfg['model']), 'activation owner differs from host')
    require(server['release_kind'] == 'formal' and server['service_identity']['container_name'] == name
            and 'diagnostic_logits' not in server['local_activation'], 'expected formal serving packet')
    with urllib.request.urlopen(f"http://{cfg['host']}:{cfg['port']}/v1/models", timeout=8) as res:
        models = json.load(res)['data']
    base = cfg['served_model_name']
    require(phase in ('ready','active') and any(row['id'] == base for row in models)
            and all(row['id'] in (base,alias) for row in models), 'unexpected loaded model')
    if phase == 'active' or any(row['id'] == alias for row in models):
        receipt = json.loads(subprocess.check_output(['docker','exec',state['Id'],'cat',
                                                      '/runtime/state/policy/ACTIVATION.json']))
        require(any(row['id'] == alias and row['parent'] == base for row in models)
                and receipt['alias'] == receipt['policy_version'] == alias
                and alias == 'checkpoint-sha256:' + receipt['weights_ref']['sha256']
                and all(receipt['service_identity'].get(k) == v for k,v in server['service_identity'].items())
                and receipt['service_identity'].get('source_boot') == cfg['host_boot_id'], 'adapter activation differs')
    print(json.dumps(dict(phase=phase,lane=lane,runtime=runtime,container=state['Id'],policy=alias)))


if __name__ == '__main__':
    check(*sys.argv[1:])

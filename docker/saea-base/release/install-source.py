"""Retarget only this project's editable hooks; leave engine and adapters intact."""
import json
import pathlib
import sysconfig
root='/opt/saea-release/studies/speech-aware-evidence-acquisition/src'
site=pathlib.Path(sysconfig.get_paths()['purelib'])
changed=[]
for p in site.glob('__editable__*speech_aware_evidence_acquisition*'):
    if p.suffix in ('.py','.pth'):
        text=p.read_text()
        old='/opt/umbrella/studies/speech-aware-evidence-acquisition/src'
        if old in text:
            p.write_text(text.replace(old,root));changed.append(str(p))
# A path hook is required even if the parent's editable format differs.
p=site/'00-saea-release.pth';p.write_text(root+'\n')
print(json.dumps({'python':__import__('sys').executable,'site':str(site),'retargeted':changed}))

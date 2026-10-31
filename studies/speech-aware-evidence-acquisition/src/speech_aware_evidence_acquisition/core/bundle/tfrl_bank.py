"""Bind banked request verification to a locally sealed complete configuration.

Old non-banked manifests acquire no new field. A banked manifest pins the full
composed values, whose canonical hash is independently checked against the
manifest. The declared bank and every bank artifact are reverified before any
row may use the narrow bank-context projection.
"""
import hashlib,json
from pathlib import Path
from . import BundleError
from .artifacts import _verify_artifact
from ..config import _canonical_hash
from ...rails import tfrl_v2_consumption as rail


def _bank_paths(decl,data):
    """Constrain the receipt and its files before the common bank loader."""
    path=(data/decl.receipt_relpath).resolve()
    if not path.is_relative_to(data.resolve()):raise BundleError('bank receipt escapes data root')
    receipt=json.loads(path.read_bytes())
    for kind in ('exemplar_bank','rule_bank'):
        name=receipt[kind]['file']
        if not isinstance(name,str) or Path(name).name!=name or name in ('.','..'):
            raise BundleError('bank artifact must be a receipt-local filename')
        if not (path.parent/name).resolve().is_relative_to(path.parent):
            raise BundleError('bank artifact escapes receipt directory')


def seal(composed,data,split,run_dir):
    declaration=rail.declaration_of_values(composed.values)
    if declaration is None:return None
    data=Path(data).resolve();run_dir=Path(run_dir).resolve()
    if _canonical_hash(composed.values)!=composed.config_hash:raise BundleError('bank composed hash differs')
    from ...rails.tfrl_supply_allocation import from_block
    from_block(composed.values['native_episode']['consumption_module'])
    _bank_paths(declaration,data)
    rail.verify_for_flight(composed.values,data,split_identity_hash=split.identity_hash,consuming_row_ids=split.ids)
    doc={'schema':'saea-tfrl-bundle-config-v1','values':dict(composed.values),'data_root':str(data)}
    raw=(json.dumps(doc,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n').encode()
    path=run_dir/'tfrl-composed-config.json'
    from ..atomic_file import publish_new
    if path.exists():
        if path.read_bytes()!=raw:raise BundleError('sealed bank config changed')
    else:publish_new(path,raw)
    return {'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()}


def load(manifest,manifest_path,split_ids,split_identity):
    pin=manifest.get('tfrl_bank_config')
    if pin is None:return None
    try:
        raw=_verify_artifact(pin,'TFRL composed config',confine_to=Path(manifest_path).resolve().parent)
        doc=json.loads(raw)
        if set(doc)!={'schema','values','data_root'} or doc['schema']!='saea-tfrl-bundle-config-v1':
            raise BundleError('invalid TFRL composed configuration snapshot')
        values=doc['values']
        if _canonical_hash(values)!=manifest['config']['config_hash']:
            raise BundleError('TFRL snapshot does not match manifest config hash')
        data=Path(doc['data_root'])
        if not data.is_absolute() or data.resolve()!=data:raise BundleError('canonical data root required')
        attempt=manifest['attempt']
        expected=data/'attempts'/manifest['run_id']/(attempt['attempt_id']+'.json')
        if Path(attempt['state_path']).resolve()!=expected.resolve():
            raise BundleError('bank root differs from durable attempt root')
        decl=rail.declaration_of_values(values)
        if decl is None:raise BundleError('bank snapshot lacks a bank declaration')
        _bank_paths(decl,data)
        bank=rail.verify_for_flight(values,data,split_identity_hash=split_identity,consuming_row_ids=tuple(split_ids))
        role=values['native_episode']['roles']['perception']
        from ...rails.tfrl_supply_allocation import from_block
        allocation=from_block(values['native_episode']['consumption_module'])
        return {'bank':bank,'perception_phase':role.get('phase_id','perception'),
                **({'allocation':allocation} if allocation is not None else {})}
    except (KeyError,TypeError,ValueError,OSError) as error:
        raise BundleError(f'banked bundle source binding failed: {error}') from error

"""Select verified offline typing_extensions before Torch/Pydantic import, including a stale cache.

Only this leaf compatibility module may be replaced in sys.modules. Never reload
Torch, CUDA, IsaacSim or an already-imported incompatible Pydantic dependency tree.
"""
from __future__ import annotations
import hashlib
import importlib.util
from pathlib import Path
import sys
import zipfile

WHEEL='typing_extensions-4.16.0-py3-none-any.whl'
WHEEL_SHA256='481caa481374e813c1b176ada14e97f1f67a4539ce9cfeb3f350d78d6370c2e8'

def activate(root):
    root=Path(root).resolve();deps=root/'.flux-deps';target=deps/'typing_extensions.py'
    wheel=root/'flux/offline-deps/wheelhouse'/WHEEL
    if deps.is_symlink() or target.is_symlink() or not target.is_file():
        raise RuntimeError('Verified offline typing_extensions installation required')
    if wheel.is_symlink() or not wheel.is_file():raise RuntimeError('Verified typing_extensions wheel required')
    data=wheel.read_bytes()
    if hashlib.sha256(data).hexdigest()!=WHEEL_SHA256:raise ValueError('typing_extensions wheel digest mismatch')
    with zipfile.ZipFile(wheel) as archive:expected=archive.read('typing_extensions.py')
    source=target.read_bytes();source_hash=hashlib.sha256(source).hexdigest()
    if source!=expected:raise ValueError('Installed typing_extensions differs from verified wheel')
    # A partially-imported incompatible Pydantic graph cannot be repaired by a leaf switch.
    # Reject it explicitly rather than purging arbitrary dependencies or touching Torch/CUDA.
    for name,module in tuple(sys.modules.items()):
        if name.split('.')[0] in ('pydantic','pydantic_core','typing_inspection') and module is not None:
            origin=getattr(module,'__file__',None)
            if not origin or not Path(origin).resolve().is_relative_to(deps.resolve()):
                raise RuntimeError('Runtime boundary invoked after incompatible dependency import: '+name)
    old=sys.modules.get('typing_extensions');old_path=getattr(old,'__file__',None)
    already_selected=old is not None and old_path and Path(old_path).resolve()==target.resolve() and hasattr(old,'Sentinel')
    original_path=list(sys.path)
    sys.path[:]=[str(deps)]+[p for p in sys.path if p!=str(deps)]
    try:
        if not already_selected:
            spec=importlib.util.spec_from_file_location('typing_extensions',target)
            if spec is None or spec.loader is None:raise RuntimeError('Cannot load verified typing_extensions')
            module=importlib.util.module_from_spec(spec)
            sys.modules['typing_extensions']=module
            spec.loader.exec_module(module)
        selected=sys.modules['typing_extensions']
        if not callable(getattr(selected,'Sentinel',None)) or Path(selected.__file__).resolve()!=target.resolve():
            raise RuntimeError('Sentinel or module provenance check failed')
    except BaseException:
        sys.path[:]=original_path
        if old is None:sys.modules.pop('typing_extensions',None)
        else:sys.modules['typing_extensions']=old
        raise
    return dict(typing_extensions_source=str(target),typing_extensions_sha256=source_hash,
        typing_extensions_version='4.16.0',sentinel_available=True,
        replaced_preloaded_typing_extensions=bool(old is not None and not already_selected),
        torch_cuda_reloaded=False,network_download=False)
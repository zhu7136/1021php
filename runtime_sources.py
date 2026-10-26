"""Verify actual executing product-module bytes against reviewed official tags."""
import hashlib
import inspect
from pathlib import Path

EXPECTED={
    'lab_sim':'340450726276d321c48b57de35f846c5a231c30a358b4922b5b7dbb8d42ec80e',
    'asset_base':'d83721120739d2e8c0e66dc8af0b3dae9a4d5606dd59fb3a319d2659cce3c3a1',
    'core_sim':'ebafc6bcb30a454925fe21b96dcdbd4637c922a3fa9d5a6947308c9796ba5028',
    'sim_manager':'9f608b061c7aa4285a4a722806d3858e7e44ab82d907309635c6295a272dd659',
}

def verify_paths(paths,event):
    if set(paths)!=set(EXPECTED):raise ValueError('Incomplete product source inventory')
    rows=[]
    for name in EXPECTED:
        path=Path(paths[name])
        if not path.is_file() or path.stat().st_size>2_000_000:raise ValueError('Product source not available')
        raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        rows.append(dict(name=name,source_path=str(path),source_sha256=digest,
            expected_official_sha256=EXPECTED[name],passed=digest==EXPECTED[name]))
    event('runtime_sources_inventory',files=rows,all_passed=all(r['passed'] for r in rows))
    if not all(r['passed'] for r in rows):raise ValueError('Unreviewed runtime product source bytes')
    event('runtime_sources_verified',passed=True,files=4)
    return rows

def verify_runtime_sources(event):
    from isaaclab.sim import SimulationContext as LabContext
    from isaaclab.assets.asset_base import AssetBase
    from isaacsim.core.api.simulation_context import SimulationContext as CoreContext
    from isaacsim.core.simulation_manager import SimulationManager
    if not issubclass(LabContext,CoreContext):raise ValueError('Unexpected actual simulation inheritance')
    objects=dict(lab_sim=LabContext,asset_base=AssetBase,core_sim=CoreContext,sim_manager=SimulationManager)
    paths={name:inspect.getsourcefile(obj) for name,obj in objects.items()}
    if any(p is None for p in paths.values()):raise ValueError('Actual product source path unavailable')
    return verify_paths(paths,event)
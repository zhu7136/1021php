"""Version-gated programmatic-stop protection; acceptance requires native evidence.

Use the same guard IsaacLab2.3.1 uses while resetting, but scoped only around
final sim.stop(). Never patch callbacks/render, skip stop, or claim close success.
"""
import ast
import hashlib
import inspect
from pathlib import Path

EXPECTED_SOURCE_SHA256='340450726276d321c48b57de35f846c5a231c30a358b4922b5b7dbb8d42ec80e'

def inspect_stop_source(source):
    raw=source.encode() if isinstance(source,str) else source
    source_digest=hashlib.sha256(raw).hexdigest()
    if source_digest!=EXPECTED_SOURCE_SHA256:
        raise ValueError('Unreviewed SimulationContext source bytes')
    tree=ast.parse(raw.decode())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SimulationContext')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_app_control_on_stop_handle_fn')
    body=fn.body
    if body and isinstance(body[0],ast.Expr) and isinstance(body[0].value,ast.Constant) and isinstance(body[0].value.value,str):
        body=body[1:]
    # Never use ast.dump strings as cross-Python-version security pins.
    # Exact executing-file bytes are stricter than body-shape equivalence.
    reset=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='reset')
    assignments=[]
    for n in reset.body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='_disable_app_control_on_stop_handle' for t in n.targets):
            assignments.append(n.value.value if isinstance(n.value,ast.Constant) else None)
    if assignments!=[True,False]:raise ValueError('Expected upstream reset guard pattern not present')
    return dict(source_sha256=source_digest,source_pin_verified=True,
                reviewed_upstream='IsaacLab-v2.3.1',exact_upstream_file_match=True)

def guarded_stop(sim,event):
    # Derive source from the executing class, NOT an argv-selected substitute.
    source=inspect.getsourcefile(type(sim))
    if not source:raise ValueError('Executing SimulationContext source unavailable')
    raw=Path(source).read_bytes()
    event('stop_guard_source_inspected',source_path=str(source),source_sha256=hashlib.sha256(raw).hexdigest())
    evidence=inspect_stop_source(raw)
    if getattr(sim,'_disable_app_control_on_stop_handle',None) is not False:
        raise ValueError('Unexpected current STOP guard state')
    event('stop_guard_source_verified',**evidence)
    sim._disable_app_control_on_stop_handle=True
    try:
        if sim._disable_app_control_on_stop_handle is not True:raise ValueError('STOP guard readback failed')
        event('stop_interactive_wait_disabled',value=True)
        sim.stop()
        event('guarded_stop_returned')
    finally:
        sim._disable_app_control_on_stop_handle=False
        event('stop_interactive_wait_guard_restored',value=sim._disable_app_control_on_stop_handle)
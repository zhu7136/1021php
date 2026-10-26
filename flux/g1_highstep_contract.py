"""Separate high-speed teacher; preserves the proven low-step budget/gates."""
from pathlib import Path
from flux import g1_lowstep_contract as base
from flux.formal_contract import execute

def adapt(text):
    for old,new in [('g1_lowstep','g1_highstep'),('G1Lowstep','G1Highstep'),
                    ('motions-low-step','motions-high-step'),('low-speed-step','high-speed-step'),
                    ('Low-speed','High-speed'),('Low-step','High-step'),('low-step','high-step'),
                    ('lowstep','highstep'),('files=113,motions=56','files=121,motions=60')]:
        text=text.replace(old,new)
    return text

def validate_scope(scope):
    if scope.get('dataset')!='motions-high-step':raise ValueError('High-step dataset required')
    base.validate_scope({**scope,'dataset':'motions-low-step'})
    return scope

budget_times=base.budget_times
def teacher_source(root):return adapt(base.teacher_source(root))
def bootstrap_source(root):return adapt(base.bootstrap_source(root))
def worker_source(root):return adapt(base.worker_source(root))
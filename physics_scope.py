"""End physical-owner scope and collect cycles before native app shutdown."""
import faulthandler
import gc
import weakref

def finish_physics_scope(physics, event):
    refs=physics()
    if not isinstance(refs,dict) or set(refs)!={'robot','sim'} or not all(
            isinstance(v,weakref.ReferenceType) for v in refs.values()):
        raise ValueError('Physics must return exactly two weak ownership measurements')
    event('physics_scope_returned')
    event('physics_gc_started')
    faulthandler.enable()
    faulthandler.dump_traceback_later(5,repeat=True)
    try:
        count=gc.collect()
    finally:
        faulthandler.cancel_dump_traceback_later()
    released={name:ref() is None for name,ref in refs.items()}
    event('physics_gc_returned',unreachable_collected=count,owners_released=released)
    if not all(released.values()):
        raise RuntimeError('Physics owner still alive before native app close')
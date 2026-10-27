"""Bounded close-path call journal. Observes calls; does not replace signal handlers or app.close.

Only a CPU-tested diagnostic candidate; no runtime repair or simulator claim.
"""
from contextlib import contextmanager
import json,signal,sys,time
from pathlib import Path

def code_of(callable_obj):
    return getattr(getattr(callable_obj,'__func__',callable_obj),'__code__',None)

@contextmanager
def profile_close(output,app,*,get_signal=signal.getsignal):
    """Current Python thread only. Preserve existing profiler; reject ambiguous instrumentation."""
    if sys.getprofile() is not None:raise RuntimeError('Existing profiler; refusing to replace it')
    handler=get_signal(signal.SIGSEGV)
    close_code=code_of(app.close);handler_code=code_of(handler)
    if close_code is None:raise ValueError('app.close has no inspectable Python code; no unsupported probe')
    targets={close_code:'app.close'}
    if handler_code is not None:targets[handler_code]='SIGSEGV_handler'
    path=Path(output)/'close-calls.jsonl'
    stream=path.open('x',buffering=1)
    counters=dict(close_calls=0,signal_handler_calls=0,max_close_depth=0,events_written=0,truncated=False)
    depth=0;start=time.monotonic()
    def emit(data):
        if counters['events_written']>=64:counters['truncated']=True;return
        stream.write(json.dumps(dict(data,elapsed_seconds=time.monotonic()-start))+'\n');stream.flush()
        counters['events_written']+=1
    def profile(frame,event,arg):
        nonlocal depth
        name=targets.get(frame.f_code)
        if name is None or event not in ('call','return'):return
        if event=='call' and name=='app.close':
            depth+=1;counters['close_calls']+=1;counters['max_close_depth']=max(counters['max_close_depth'],depth)
        elif event=='call':counters['signal_handler_calls']+=1
        emit(dict(event=event,callable=name,close_depth=depth))
        if event=='return' and name=='app.close':depth-=1
    try:
        emit(dict(event='probe_enter',signal_handler_inspectable=handler_code is not None,
            signal_handler_module=getattr(handler,'__module__',None),signal_handler_name=getattr(handler,'__qualname__',None),
            current_thread_only=True,signal_handlers_modified=False))
        sys.setprofile(profile)
        yield counters
    finally:
        sys.setprofile(None)
        emit(dict(event='probe_exit',**counters));stream.close()
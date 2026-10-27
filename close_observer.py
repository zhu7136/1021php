"""Observe one original native close call; no extra app, alternate close or forced exit."""
from contextlib import contextmanager
import faulthandler,json,time
from pathlib import Path
from close_profile import profile_close

def record(output,name,**data):
    with (Path(output)/name).open('x') as f:
        json.dump(data,f,indent=2);f.flush()

def settings_provider():
    import carb.settings
    return carb.settings.get_settings()

@contextmanager
def observe_native_close(module,output,*,get_settings=settings_provider,trace=faulthandler):
    """Instrument exact train module reference, delegate once, restore even on error.

    The only shutdown-setting change is fastShutdown=False immediately before
    the original helper. Parent supervisor enforces 30s from close-entered.
    Python traces alone cannot identify native C/CUDA frames.
    """
    output=Path(output);original=module.close_simulation_app
    def observed(app):
        started=time.monotonic()
        record(output,'close-entered.json',native_close_entered=True,monotonic_started=started,
            parent_close_budget_seconds=30,original_helper_delegated=False)
        try:
            settings=get_settings();before=settings.get('/app/fastShutdown')
            if type(before) is not bool:raise ValueError('Unknown fastShutdown setting; no close success claim')
            settings.set_bool('/app/fastShutdown',False)
            after=settings.get('/app/fastShutdown')
            if after is not False:raise ValueError('fastShutdown readback failed')
            record(output,'close-settings.json',original=before,requested=False,readback=after,
                modified_only_at_native_close_entry=True)
            with (output/'close-stack.log').open('x') as log:
                trace.dump_traceback_later(5,repeat=True,file=log)
                try:
                    with profile_close(output,app):
                        result=original(app)
                finally:trace.cancel_dump_traceback_later()
            record(output,'close-returned.json',native_close_returned=True,
                original_helper_delegated=True,elapsed_seconds=time.monotonic()-started,
                fast_shutdown_readback=False)
            return result
        except BaseException as error:
            record(output,'close-error.json',native_close_returned=False,error_type=type(error).__name__,
                elapsed_seconds=time.monotonic()-started)
            raise
    module.close_simulation_app=observed
    try:yield
    finally:module.close_simulation_app=original
"""Observe real cleanup; do not bypass stage cleanup or force process exit."""
import inspect,json,time
def write(out,report):
 p=out/'lifecycle.pending.json';p.write_text(json.dumps(report,indent=2));p.replace(out/'lifecycle.json')
def configure(app,out,report):
 import carb.settings
 app.set_setting('/app/fastShutdown',False)
 report['fast_shutdown_disabled_verified']=carb.settings.get_settings().get('/app/fastShutdown') is False
 if not report['fast_shutdown_disabled_verified']:raise RuntimeError('Fast shutdown remains enabled')
 (out/'installed-close-source.json').write_text(json.dumps({'source':inspect.getsource(type(app).close)}))
 write(out,report)
def close(app,out,report):
 from isaaclab.sim import SimulationContext
 context=SimulationContext.instance()
 if context is not None and hasattr(context,'_disable_app_control_on_stop_handle'):context._disable_app_control_on_stop_handle=True
 report['close_started']=True;write(out,report);started=time.monotonic()
 app.close(wait_for_replicator=False,skip_cleanup=False)
 report.update(close_returned=True,close_elapsed_seconds=time.monotonic()-started);write(out,report)
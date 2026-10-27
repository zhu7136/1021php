"""One bounded empty-app close comparison. No robot, PPO, model, install or secret input."""
import argparse,faulthandler,hashlib,inspect,json,os,signal,subprocess,sys,time,types
from pathlib import Path
import supervisor_base
from close_observer import observe_native_close
PREFIX='HUD04_R27_EVENT '
supervisor_base.PREFIX=PREFIX

def event(name,**kw):print(PREFIX+json.dumps(dict(event=name,**kw),sort_keys=True),flush=True)

def runtime_callable(fn):
    fn=getattr(fn,'__func__',fn)
    path=inspect.getsourcefile(fn) if inspect.isfunction(fn) else None
    row=dict(module=getattr(fn,'__module__',None),name=getattr(fn,'__qualname__',None),python_source=path)
    if path:
        p=Path(path)
        if not p.is_file() or p.stat().st_size>2*1024*1024:raise ValueError('Unbounded runtime source')
        row['source_file_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
        text=inspect.getsource(fn)
        row['callable_source_sha256']=hashlib.sha256(text.encode()).hexdigest()
        if len(text)>8192:raise ValueError('Unexpected callback source length')
        row['callable_source']=text
    return row

def check_device(properties,count,available):
    observed=dict(name=properties.name,memory_bytes=properties.total_memory,major=properties.major,minor=properties.minor,count=count,available=available)
    if available is not True or count!=1 or properties.name not in ('NVIDIA GeForce RTX 4090','NVIDIA GeForce RTX 4090 D','NVIDIA GeForce RTX 4090D'):
        raise ValueError('Previously reconciled single RTX4090 family required')
    if not 23*2**30<=properties.total_memory<=25*2**30 or (properties.major,properties.minor)!=(8,9):
        raise ValueError('GPU memory or capability mismatch')
    return observed

def child(output):
    from runtime_versions import check_runtime_versions
    event('versions_verified',**check_runtime_versions(),python=sys.version.split()[0])
    from isaaclab.app import AppLauncher
    parser=argparse.ArgumentParser();AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args([]);args.headless=True;args.device='cuda:0';args.enable_cameras=False
    app=None
    try:
        event('app_launch_started');launcher=AppLauncher(args);app=launcher.app;event('app_launch_returned')
        handler=signal.getsignal(signal.SIGSEGV)
        evidence=dict(handler=runtime_callable(handler),close=runtime_callable(app.close),signal_handlers_modified=False,
            current_thread_only=True,robot_loaded=False,ppo_updates=0)
        (output/'runtime-callables.json').write_text(json.dumps(evidence,indent=2)+'\n')
        event('runtime_callables',**evidence)
        import carb.settings
        settings=carb.settings.get_settings();old=settings.get('/app/fastShutdown')
        if type(old) is not bool:raise ValueError('Unknown fastShutdown setting')
        settings.set_bool('/app/fastShutdown',False)
        if settings.get('/app/fastShutdown') is not False:raise ValueError('fastShutdown readback failed')
        event('fast_shutdown_verified',original=old,value=False)
        import torch
        observed=check_device(torch.cuda.get_device_properties(0),torch.cuda.device_count(),torch.cuda.is_available())
        event('gpu_observed',**observed)
        if float(torch.ones(4,device='cuda:0').sum())!=4.:raise ValueError('CUDA arithmetic failed')
        event('cuda_verified',torch=torch.__version__,cuda=torch.version.cuda,**observed)
        for _ in range(8):app.update()
        event('empty_app_updates_completed',updates=8,robot_loaded=False,ppo_updates=0)
    except BaseException as error:
        event('child_failure',error_type=type(error).__name__);raise
    finally:
        if app is not None:
            event('close_started',strategy='original_direct_app_close_with_observer',parent_close_budget_seconds=30)
            faulthandler.enable()
            module=types.SimpleNamespace(close_simulation_app=lambda instance:instance.close(wait_for_replicator=False))
            with observe_native_close(module,output):module.close_simulation_app(app)
            event('close_returned')

def read_owned_journal(output):
    path=Path(output)/'close-calls.jsonl'
    if not path.exists():return dict(present=False,rows=[],missing_does_not_rule_out_native_failure=True)
    if path.is_symlink() or not path.is_file() or path.stat().st_size>65536:raise ValueError('Invalid owned journal')
    rows=[json.loads(s) for s in path.read_text().splitlines()]
    if len(rows)>64:raise ValueError('Journal event limit')
    return dict(present=True,rows=rows,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size,
        missing_does_not_rule_out_native_failure=True)

def main():
    start=time.monotonic();p=argparse.ArgumentParser();p.add_argument('--child',action='store_true');p.add_argument('--output')
    p.add_argument('--execute',action='store_true')
    for name in ('commit','task-id','thread-id','turn-id'):p.add_argument('--'+name)
    args=p.parse_args();root=Path(__file__).resolve().parent
    if args.child:
        out=Path(args.output)
        if not out.is_dir() or out.is_symlink():raise ValueError('Owned output required')
        return child(out)
    if not args.execute:return event('read_only',execution_requested=False,gpu_started=False)
    if not all([args.commit,args.task_id,args.thread_id,args.turn_id]) or args.task_id=='PENDING':raise ValueError('Task/commit/provenance binding required')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True,timeout=5).strip()!=args.commit:raise ValueError('Wrong source commit')
    manifest=json.loads((root/'manifest.json').read_text())
    for row in manifest['files']:
        path=root/row['path']
        if path.is_symlink() or not path.is_file() or path.stat().st_size!=row['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Wrong source manifest')
    python=next((x for x in [Path('/isaac-sim/python.sh'),Path('/workspace/isaaclab/_isaac_sim/python.sh')]
        if x.is_file() and os.access(x,os.X_OK) and (x.parent/'kit/python/bin/python3').is_file()),None)
    if python is None:raise ValueError('Known vendor Python unavailable')
    keep={'PATH','LD_LIBRARY_PATH','PYTHONPATH','PYTHONHOME','ISAAC_PATH','ISAACSIM_PATH','ISAACLAB_PATH','CARB_APP_PATH','EXP_PATH','HOME','USER','TMPDIR','DISPLAY','LANG','ACCEPT_EULA','PRIVACY_CONSENT'}
    env={k:v for k,v in os.environ.items() if k in keep}
    env.update(PYTHONUNBUFFERED='1',PYTHONHASHSEED='42',WANDB_MODE='disabled',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    event('parent_started',task_id=args.task_id,commit=args.commit,worker_seconds=360,automatic_retries=0,
        provenance=dict(thread_id=args.thread_id,turn_id=args.turn_id),robot_loaded=False,ppo_updates=0)
    out=root/('diagnostic-'+args.task_id);remaining=360-(time.monotonic()-start)
    if remaining<10:raise TimeoutError('Preparation exceeded budget')
    result=supervisor_base.supervise([str(python),str(Path(__file__).resolve()),'--child','--output',str(out)],
        cwd=root,env=env,out=out,seconds=remaining-3,close_seconds=30)
    journal=read_owned_journal(out);event('close_journal',**journal)
    probe=journal.get('rows',[])
    probe_complete=any(x.get('event')=='probe_exit' for x in probe)
    verified=result['passed'] and probe_complete and any(x.get('callable')=='app.close' and x.get('event')=='call' for x in probe)
    event('diagnostic_result',passed=verified,lifecycle_parent_passed=result['passed'],probe_complete=probe_complete,
        task_id=args.task_id,commit=args.commit,returncode=result['returncode'],reason=result['reason'],
        elapsed_seconds=result['elapsed_seconds'],robot_loaded=False,ppo_updates=0)
    if not verified:raise SystemExit(1)

if __name__=='__main__':main()
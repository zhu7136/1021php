"""One read-only image inventory + empty-app diagnostic. No installs/robot/PPO."""
import argparse
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

PREFIX='HUD04_R6_EVENT '
def event(name,**fields):
    print(PREFIX+json.dumps({'event':name,**fields},sort_keys=True),flush=True)

def child():
    import importlib.metadata as metadata
    requirements=json.loads((Path(__file__).resolve().parent/'target-packages.json').read_text())
    inventory={}
    for name in requirements:
        try:inventory[name]=metadata.version(name)
        except metadata.PackageNotFoundError:inventory[name]=None
    event('image_dependency_inventory',packages=inventory,expected_candidate=requirements,
          installed_packages_changed=False,target_overlay_installed=False)
    from runtime_versions import check_runtime_versions
    event('version_check_started')
    versions=check_runtime_versions()
    event('versions_verified',packages=versions['packages'],python=sys.version.split()[0])
    from isaaclab.app import AppLauncher
    parser=argparse.ArgumentParser();AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args([]);args.headless=True;args.device='cuda:0';args.enable_cameras=False
    app=None
    try:
        event('app_launch_started')
        launcher=AppLauncher(args);app=launcher.app
        event('app_launch_returned')
        import carb.settings
        settings=carb.settings.get_settings();old=settings.get('/app/fastShutdown')
        if type(old) is not bool:raise ValueError('unknown fastShutdown capability')
        settings.set_bool('/app/fastShutdown',False)
        if settings.get('/app/fastShutdown') is not False:raise ValueError('fastShutdown disable failed')
        event('fast_shutdown_verified',original=old,value=False)
        import torch
        available=torch.cuda.is_available();count=torch.cuda.device_count()
        event('cuda_device_count',available=available,count=count)
        if not available or count!=1:raise ValueError('one GPU required')
        properties=torch.cuda.get_device_properties(0)
        device={'name':properties.name,'total_memory_bytes':properties.total_memory,
                'major':properties.major,'minor':properties.minor,'multi_processor_count':properties.multi_processor_count,
                'torch':torch.__version__,'torch_cuda':torch.version.cuda,'count':count}
        event('gpu_properties_recorded',**device)
        capacity_ok=23*2**30<=properties.total_memory<=25*2**30
        model_ok='4090D' in properties.name
        strict_contract=capacity_ok and model_ok
        event('resource_contract',passed=strict_contract,capacity_ok=capacity_ok,model_ok=model_ok,
              expected_name_contains='4090D',expected_memory_gib=[23,25],
              diagnostic_continuation_only=capacity_ok and '4090' in properties.name)
        # A label mismatch remains failure. Continue only the already bounded empty-app
        # measurement on a 4090-family, capacity-qualified device; never training.
        if not capacity_ok or '4090' not in properties.name:raise ValueError('GPU diagnostic capacity/family mismatch')
        if float(torch.ones(4,device='cuda:0').sum())!=4.:raise ValueError('CUDA arithmetic failed')
        x=torch.ones((16,16),device='cuda:0');y=x@x;torch.cuda.synchronize()
        if float(y.sum())!=4096.:raise ValueError('CUDA matrix multiplication failed')
        event('cuda_verified',device=properties.name,bytes=properties.total_memory,torch=torch.__version__,cuda=torch.version.cuda)
        for _ in range(8):app.update()
        event('empty_app_updates_completed',updates=8,robot_loaded=False,ppo_updates=0)
        if not strict_contract:raise ValueError('Catalog GPU model mismatch; bounded measurements completed, contract remains failed')
    except BaseException as error:
        event('child_failure',error_type=type(error).__name__)
        raise
    finally:
        if app is not None:
            event('close_started',strategy='direct_app_close_no_holosoma_patch')
            faulthandler.enable();faulthandler.dump_traceback_later(5,repeat=True)
            try:
                import carb.settings
                settings=carb.settings.get_settings();settings.set_bool('/app/fastShutdown',False)
                if settings.get('/app/fastShutdown') is not False:raise ValueError('close fastShutdown readback failed')
                app.close(wait_for_replicator=False)
                event('close_returned')
            finally:faulthandler.cancel_dump_traceback_later()

def supervise(command,*,cwd,env,out,seconds=350,close_seconds=30):
    """Only owns/signals the process group started here. No sandbox inspection."""
    out=Path(out);out.mkdir(exist_ok=False)
    started=time.monotonic();end=started+seconds
    events=[];offset=0;pending=b'';close_end=None;reason='not_started';code=None;proc=None
    def interrupt(*_):raise InterruptedError('parent interrupted')
    previous=signal.signal(signal.SIGTERM,interrupt)
    def scan():
        nonlocal offset,pending,close_end
        with (out/'child.log').open('rb') as stream:
            stream.seek(offset);chunk=stream.read(1024*1024);offset+=len(chunk)
        pending+=chunk
        if len(pending)>2*1024*1024:raise ValueError('oversized log line')
        lines=pending.split(b'\n');pending=lines.pop()
        for raw in lines:
            if raw.startswith(PREFIX.encode()):
                row=json.loads(raw[len(PREFIX):]);events.append(row)
                if len(events)>100:raise ValueError('event limit')
                print(PREFIX+json.dumps(row,sort_keys=True),flush=True)
                if row.get('event')=='close_started' and close_end is None:
                    close_end=min(end-5,time.monotonic()+close_seconds)
    try:
        with (out/'child.log').open('xb') as log:
            proc=subprocess.Popen(command,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while True:
                scan()
                if (out/'child.log').stat().st_size>32*1024*1024:reason='log_limit';break
                code=proc.poll()
                if code is not None:reason='child_exited';scan();break
                if time.monotonic()>=min(end-5,close_end or end-5):
                    reason='close_timeout' if close_end is not None else 'total_timeout';break
                time.sleep(.1)
    except BaseException as error:
        reason=type(error).__name__
    finally:
        if proc is not None:
            for sig,wait in [(signal.SIGTERM,2),(signal.SIGKILL,2)]:
                try:os.killpg(proc.pid,sig)
                except ProcessLookupError:pass
                try:proc.wait(timeout=wait)
                except subprocess.TimeoutExpired:pass
            code=proc.poll()
        signal.signal(signal.SIGTERM,previous)
    names=[r.get('event') for r in events]
    rows={r.get('event'):r for r in events}
    passed=(reason=='child_exited' and code==0 and 'child_failure' not in names
        and all(n in names for n in ['image_dependency_inventory','gpu_properties_recorded','resource_contract',
                                    'versions_verified','app_launch_returned','fast_shutdown_verified',
                                    'cuda_verified','empty_app_updates_completed','close_started','close_returned'])
        and rows.get('resource_contract',{}).get('passed') is True
        and rows.get('fast_shutdown_verified',{}).get('value') is False
        and rows.get('empty_app_updates_completed',{}).get('updates')==8
        and names.index('close_started')<names.index('close_returned'))
    if not passed and (out/'child.log').is_file():
        # Only this newly-created diagnostic log, never other processes/files.
        with (out/'child.log').open('rb') as stream:
            stream.seek(max(0,(out/'child.log').stat().st_size-14000))
            print('HUD04_R6_OWNED_CHILD_LOG_TAIL_BEGIN\n'+stream.read(14000).decode(errors='replace')+'\nHUD04_R6_OWNED_CHILD_LOG_TAIL_END',flush=True)
    result={'passed':passed,'reason':reason,'returncode':code,'elapsed_seconds':round(time.monotonic()-started,3),
            'events':events,'ppo_updates':0,'robot_loaded':False,'native_hud04_checkpoint_verified':False}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    event('parent_result',**{k:v for k,v in result.items() if k!='events'})
    return result

def main():
    started=time.monotonic()
    p=argparse.ArgumentParser();p.add_argument('--child',action='store_true')
    p.add_argument('--execute',action='store_true');p.add_argument('--commit');p.add_argument('--task-id')
    p.add_argument('--thread-id');p.add_argument('--turn-id');args=p.parse_args()
    if args.child:return child()
    if not args.execute:
        event('read_only',execution_requested=False,gpu_started=False);return
    if not all([args.commit,args.task_id,args.thread_id,args.turn_id]) or args.task_id=='PENDING':
        raise ValueError('task binding/provenance required')
    root=Path(__file__).resolve().parent
    actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True,timeout=5).strip()
    if actual!=args.commit:raise ValueError('commit mismatch')
    manifest=json.loads((root/'manifest.json').read_text())
    for row in manifest['files']:
        f=root/row['path']
        if f.is_symlink() or hashlib.sha256(f.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('source mismatch')
    python=None
    for f in [Path('/isaac-sim/python.sh'),Path('/workspace/isaaclab/_isaac_sim/python.sh')]:
        if f.is_file() and os.access(f,os.X_OK) and (f.parent/'kit/python/bin/python3').is_file():python=f;break
    if python is None:raise ValueError('known vendor Python launcher unavailable')
    keep={'PATH','LD_LIBRARY_PATH','PYTHONPATH','PYTHONHOME','ISAAC_PATH','ISAACSIM_PATH','ISAACLAB_PATH',
          'CARB_APP_PATH','EXP_PATH','HOME','USER','TMPDIR','DISPLAY','LANG','ACCEPT_EULA','PRIVACY_CONSENT'}
    env={k:v for k,v in os.environ.items() if k in keep}
    env.update(PYTHONUNBUFFERED='1',PYTHONHASHSEED='42',WANDB_MODE='disabled',CUDA_VISIBLE_DEVICES='0',
               OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    event('parent_started',task_id=args.task_id,commit=args.commit,worker_seconds=360,automatic_retries=0,
          provenance={'thread_id':args.thread_id,'turn_id':args.turn_id},purpose='image_inventory_empty_app_not_training')
    output=root/('diagnostic-'+args.task_id)
    remaining=360-(time.monotonic()-started)
    if remaining<8:raise TimeoutError('no remaining diagnostic budget')
    result=supervise([str(python),str(Path(__file__).resolve()),'--child'],cwd=root,env=env,out=output,
                     seconds=remaining)
    if not result['passed']:raise SystemExit(1)

if __name__=='__main__':main()

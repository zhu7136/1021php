"""CPU-prepared Hu-D04 low-step bootstrap. Never submit or start a task here.

Fresh training only; 1800 seconds from the original worker start; zero retries.
"""
import base64,json,os,shutil,subprocess,sys,threading,time
from pathlib import Path
from bootstrap_base import safe_extract,clean_env
from scoped_transport import transfer,digest,validate_capability
from range_input import download as private_download
from worker_utils import stage,pack,watchdog,emit
from budget_policy import deadlines,require_training_window
BASE_BYTES=113889241
BASE_SHA='6ed95a54d8846611b8e2519dcc014a9aec9c5dd885914d15137cc18e8f1473e0'
ALLOWED={'bundle/reference/full-reference.npz','bundle/terrain/full-terrain.obj','teacher-code/config.json','teacher-code/cases.json','teacher-code/teacher_child.py','teacher-code/teacher_algo.py','teacher-code/close_audit.py'}

def parse(argv=None):
    from bootstrap_base import SafeParser
    import re
    p=SafeParser();p.add_argument('--execute',action='store_true')
    for n in ['task-id','commit','base-cap','input-cap','output-cap','input-sha256','thread-id','turn-id','skill']:p.add_argument('--'+n)
    p.add_argument('--input-bytes',type=int);a=p.parse_args(argv)
    if a.execute and (not all(vars(a).values()) or a.skill!='low-step' or not re.fullmatch(r'TASK_\d{8}_\d+',a.task_id) or not re.fullmatch('[0-9a-f]{40}',a.commit) or not re.fullmatch('[0-9a-f]{64}',a.input_sha256) or not 0<a.input_bytes<512*1024**2):raise ValueError('Missing or invalid execution binding')
    return a

def validate_overlay(overlay,manifest):
    scope=manifest['scope']
    if {r['path'] for r in manifest['files']}!=ALLOWED or scope['skill']!='low-step' or scope['clip_count']!=56 or scope['frame_count']!=25676 or scope['num_envs']!=4096 or scope['worker_seconds_max']!=1800 or scope['automatic_retries']!=0 or not scope['fresh_training']:raise ValueError('Full input scope mismatch')
    cases=json.loads((overlay/'teacher-code/cases.json').read_text())
    cfg=json.loads((overlay/'teacher-code/config.json').read_text())
    if len(cases)!=56 or sum(c['frames'] for c in cases)!=25676:raise ValueError('Full dataset coverage mismatch')
    if cfg['robot']['asset']['robot_type']!='hud04' or not cfg['robot']['asset']['urdf_file'].endswith('HU_D04_01.urdf') or len(cfg['robot']['dof_names'])!=31 or cfg['robot']['actions_dim']!=29:raise ValueError('Wrong robot embodiment')
    if cfg['training']['num_envs']!=4096 or any(cfg['training'][k] is not None for k in ['checkpoint','teacher_checkpoint','registry_name','preprocess_hook']):raise ValueError('Fresh training contract failed')
    return scope

def validate_runtime(manifest):
    scope=manifest['scope']
    if scope.get('runtime_only') is not True or scope.get('checkpoint_files')!=0:
        raise ValueError('Runtime-only contract required')
    if any(Path(r['path']).suffix.lower() in {'.pt','.pth','.ckpt'} for r in manifest['files']):
        raise ValueError('Inherited checkpoint prohibited')

def main(argv=None):
    start=time.monotonic();a=parse(argv)
    if not a.execute:
        emit('execution_not_requested',worker_seconds_max=1800,automatic_retries=0,gpu_started=False);return 0
    ends=deadlines(start,time.monotonic())
    caps={n:json.loads(base64.urlsafe_b64decode(getattr(a,n+'_cap'))) for n in ['base','input','output']}
    for n,method in [('base','GET'),('input','GET'),('output','PUT')]:validate_capability(caps[n],method)
    root=Path(__file__).resolve().parent;env=clean_env()
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,env=env,text=True,timeout=10).strip()!=a.commit:raise ValueError('Source commit mismatch')
    threading.Thread(target=watchdog,args=(start,),daemon=True).start()
    work=root/('private-'+a.task_id);work.mkdir(exist_ok=False)
    state={'task_id':a.task_id,'bootstrap_commit':a.commit,'worker_seconds_max':1800,'automatic_retries':0,'requested_num_envs':4096,'skill':'low-step','robot':'Hu-D04','fresh_training':True,'teacher_converged':False,'worker_passed':False,'stage':'private_full_input','input_verified':False,'native_training_started':False,'transport_revision':'four-stream-shared-600s-v1','account_credentials_forwarded':False}
    def report(label):
        def callback(p):
            (work/(label+'-transfer-progress.json')).write_text(json.dumps(p,indent=2))
            emit('input_transfer_progress',input_kind=label,**p)
        return callback
    try:
        emit('stage',stage=state['stage'],task_id=a.task_id)
        state['private_input_transfer']={}
        private_download(caps['input'],work/'full-input.zip',size=a.input_bytes,sha256=a.input_sha256,absolute_deadline=ends['input'],progress=state['private_input_transfer'],on_progress=report('motion'))
        overlay=work/'overlay';manifest=safe_extract(work/'full-input.zip',overlay);validate_overlay(overlay,manifest)
        state.update(clip_count=56,frame_count=25676,private_input_verified=True,stage='base_runtime')
        emit('stage',stage='base_runtime',private_input_verified=True,clip_count=56,frame_count=25676)
        state['base_transfer']={}
        private_download(caps['base'],work/'input.zip',size=BASE_BYTES,sha256=BASE_SHA,absolute_deadline=ends['input'],progress=state['base_transfer'],on_progress=report('base'))
        payload=work/'payload';base=safe_extract(work/'input.zip',payload);validate_runtime(base)
        if time.monotonic()>ends['input']:raise TimeoutError('Input preparation deadline reached')
        for row in manifest['files']:
            target=payload/row['path'];target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():raise ValueError('Refuse to overwrite inherited input')
            shutil.copyfile(overlay/row['path'],target)
        state.update(input_verified=True,stage='install');emit('stage',stage='install')
        vendor='/isaac-sim/python.sh';site=work/'dependency-site'
        if not Path(vendor).is_file():raise ValueError('Vendor interpreter missing')
        rc=stage([vendor,'-m','pip','install','--isolated','--disable-pip-version-check','--no-index','--no-cache-dir','--no-deps','--only-binary=:all:','--require-hashes','--find-links',str(payload/'deps/wheels'),'--target',str(site),'-r',str(payload/'deps/overlay.lock')],work/'install.log',env,ends['install'])
        if rc:raise ValueError('Offline install failed')
        env['PYTHONPATH']=str(site)+os.pathsep+env.get('PYTHONPATH','');env['PHP_FLUX_TASK_ID']=a.task_id
        state['stage']='gpu_preflight';emit('stage',stage='gpu_preflight')
        probe="import torch,numpy;p=torch.cuda.get_device_properties(0);print(p.name,p.total_memory,flush=True);assert torch.__version__=='2.7.0+cu128' and numpy.__version__=='1.26.4' and torch.cuda.device_count()==1 and '4090' in p.name and [p.major,p.minor]==[8,9] and 23<=p.total_memory/2**30<=25"
        if stage([vendor,'-c',probe],work/'gpu-preflight.log',env,ends['gpu']):raise ValueError('GPU contract failed')
        require_training_window(start,time.monotonic())
        out=work/'native-output';out.mkdir();state.update(stage='native_training',native_training_started=True)
        emit('stage',stage='native_training',requested_num_envs=4096,clip_count=56)
        cmd=[vendor,str(payload/'teacher-code/teacher_child.py'),'--payload',str(payload),'--output',str(out),'--site',str(site),'--thread-id',a.thread_id,'--turn-id',a.turn_id]
        rc=stage(cmd,out/'native.log',env,ends['train_hard'],stop_file=out/'STOP_AFTER_UPDATE',soft=ends['train_soft']);state['native_exit_code']=rc;state['stage']='result_validation'
        learn=json.loads((out/'learning_state.json').read_text());life=json.loads((out/'lifecycle.json').read_text())
        required=['actor_changed','critic_changed','actor_optimizer_steps_increased','critic_optimizer_steps_increased','parameters_finite']
        state.update(completed_updates=learn['completed_updates'],checkpoint_sha256=learn['checkpoint_sha256'],close_returned=life.get('close_returned',False),train_returned=life.get('train_returned',False))
        if rc or not 1<=learn['completed_updates']<=100000 or not all(learn[k] for k in required) or life.get('task_id')!=a.task_id or not all(life.get(k) for k in ['close_returned','train_returned','child_returned','fast_shutdown_disabled_verified']):raise ValueError('Learning or normal-return evidence incomplete')
        state.update(worker_passed=True,stage='complete')
    except Exception as e:
        state['error_type']=type(e).__name__;emit('stage_failure',stage=state['stage'],error_type=type(e).__name__)
    state['elapsed_before_upload_seconds']=round(time.monotonic()-start,3)
    result=pack(work,state)
    receipt=transfer(caps['output'],result,method='PUT',size=result.stat().st_size,sha256=digest(result),absolute_deadline=ends['result'])
    emit('result_uploaded',task_id=a.task_id,bytes=receipt['bytes'],sha256=receipt['sha256'],worker_passed=state['worker_passed'],elapsed_seconds=round(time.monotonic()-start,3))
    return 0 if state['worker_passed'] else 1

if __name__=='__main__':
    try:sys.exit(main())
    except Exception as e:emit('bootstrap_failure',error_type=type(e).__name__);sys.exit(1)
"""One 292 high-speed continuation; 60m learning incl. 20-update gate."""
from pathlib import Path
from flux.formal_contract import replace_once,execute
from flux import g1_highstep_contract as base
from flux.g1_resume292_input import SOURCE_TASK,SOURCE_RUN,CHECKPOINT_SHA256

def budget_times(validation_deadline,gate):
    if not gate or gate.get('passed') is not True:return validation_deadline-30,validation_deadline
    started=gate['training_started_monotonic']
    return started+3480,started+3600

def validate_scope(scope):
    required={'fresh_training':False,'resume_source_task':SOURCE_TASK,'resume_source_sha256':CHECKPOINT_SHA256,'formal_seconds':3600}
    for k,v in required.items():
        if type(scope.get(k)) is not type(v) or scope[k]!=v:raise ValueError('Invalid292 scope '+k)
    if type(scope.get('worker_seconds')) is not int or not 180<=scope['worker_seconds']<=4980:raise ValueError('Invalid worker budget')
    checked={**scope,'fresh_training':True,'formal_seconds':1800,'worker_seconds':min(scope['worker_seconds'],3180)}
    base.validate_scope(checked)
    return scope

def teacher_source(root):
    text=base.teacher_source(root)
    text=replace_once(text,"resume_checkpoint = None\n    if os.environ.get('PHP_FLUX_RESUME_CHECKPOINT'):\n        raise ValueError('High-step must train fresh')",
        "resume_checkpoint = os.environ.get('PHP_FLUX_RESUME_CHECKPOINT')\n    if not resume_checkpoint:raise ValueError('292 continuation requires the verified native checkpoint')")
    text=replace_once(text,"target = 'flux.g1_highstep_algo.G1HighstepPPO'","target = 'flux.g1_resume292_algo.G1Resume292PPO'")
    text=replace_once(text,"    if resume_checkpoint and args.updates != 2:\n        raise ValueError('Resume validation requires exactly two updates')",
        "    if args.updates!=20020:raise ValueError('292 continuation update ceiling changed')")
    text=replace_once(text,"fresh_training=True, flux_task_id=os.environ.get('PHP_G1_TASK_ID')",
        "fresh_training=False, resume_source_task='TASK_20261026_292', resume_source_run='00vxykeq', source_iteration=833, source_completed_updates=834, flux_task_id=os.environ.get('PHP_G1_TASK_ID')")
    return text.replace('g1-high-speed-step-teacher-4096-30m','g1-high-speed-step-resume292-4096-60m')

def bootstrap_source(root):
    text=base.bootstrap_source(root)
    text=replace_once(text,'from flux.g1_highstep_contract import validate_scope','from flux.g1_resume292_contract import validate_scope')
    text=replace_once(text,'from flux.g1_highstep_contract import budget_times','from flux.g1_resume292_contract import budget_times')
    text=replace_once(text,"ROOT / 'flux/g1_highstep_teacher.py'","ROOT / 'flux/g1_resume292_teacher.py'")
    text=replace_once(text,"and formal.get('formal_updates',0)>0)","and formal.get('formal_updates',0)>0 and json.loads((output/'resume_result.json').read_text()).get('passed') is True)")
    return text

def worker_source(root):
    text=base.worker_source(root)
    text=replace_once(text,"    parser.add_argument('--highstep-input-b64',required=True)","    parser.add_argument('--highstep-input-b64',required=True)\n    parser.add_argument('--g1-resume-input-b64',required=True)")
    text=replace_once(text,'    updates = 20020',"    updates = 20020\n    from flux.g1_resume292_input import validate_control,fetch_checkpoint,CHECKPOINT_SHA256,SOURCE_TASK\n    source_control=decode_control(args.g1_resume_input_b64)\n    validate_control(source_control)")
    text=replace_once(text,"'g1-highstep-' + task_id","'g1-resume292-' + task_id")
    text=replace_once(text,"        env['PHP_G1_TASK_ID']=args.task_id","        source=fetch_checkpoint(source_control,OUT/'resume-input')\n        env['PHP_FLUX_RESUME_CHECKPOINT']=str(source)\n        emit('g1_resume292_checkpoint_hash_verified',source_task=SOURCE_TASK,source_sha256=CHECKPOINT_SHA256)\n        env['PHP_G1_TASK_ID']=args.task_id")
    text=replace_once(text,"'formal_seconds':1800,'dataset':'motions-high-step','fresh_training':True,","'formal_seconds':3600,'dataset':'motions-high-step','fresh_training':False,'resume_source_task':SOURCE_TASK,'resume_source_sha256':CHECKPOINT_SHA256,")
    text=replace_once(text,"ROOT/'flux/g1_highstep_bootstrap.py'","ROOT/'flux/g1_resume292_bootstrap.py'")
    for old,new in [('signal.alarm(3180)','signal.alarm(4980)'),('worker_seconds=3300','worker_seconds=5100'),
                    ('int(3060-','int(4860-'),('int(3298-','int(5098-'),('formal<=1800','learning<=3600 incl20-update gate'),
                    ('30minute','60minute'),('One fresh G1 High-speed step4096env trial:','One G1 High-speed step4096env continuation from292:')]:
        text=text.replace(old,new)
    return text
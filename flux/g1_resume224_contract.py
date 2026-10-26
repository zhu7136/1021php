"""Isolated224 continuation; original fresh/old source guards remain unchanged."""
from flux.formal_contract import replace_once,execute
from flux import g1_lowstep_contract as base
from flux.g1_resume224_input import SOURCE_TASK,SOURCE_RUN,CHECKPOINT_SHA256

def validate_scope(scope):
    for key,value in {'fresh_training':False,'resume_source_task':SOURCE_TASK,'resume_source_sha256':CHECKPOINT_SHA256}.items():
        if type(scope.get(key)) is not type(value) or scope[key]!=value:raise ValueError('Invalid224 continuation scope '+key)
    checked=dict(scope);checked['fresh_training']=True
    base.validate_scope(checked)
    return scope

def teacher_source(root):
    text=base.teacher_source(root)
    text=replace_once(text,"resume_checkpoint = None\n    if os.environ.get('PHP_FLUX_RESUME_CHECKPOINT'):\n        raise ValueError('Low-step must train fresh')",
                      "resume_checkpoint = os.environ.get('PHP_FLUX_RESUME_CHECKPOINT')\n    if not resume_checkpoint:raise ValueError('224 continuation requires the verified native checkpoint')")
    text=replace_once(text,"target = 'flux.g1_lowstep_algo.G1LowstepPPO'","target = 'flux.g1_resume224_algo.G1Resume224PPO'")
    text=replace_once(text,"    if resume_checkpoint and args.updates != 2:\n        raise ValueError('Resume validation requires exactly two updates')",
                      "    if args.updates!=20020:raise ValueError('224 continuation update ceiling changed')")
    text=replace_once(text,"fresh_training=True, flux_task_id=os.environ.get('PHP_G1_TASK_ID')",
                      "fresh_training=False, resume_source_task='TASK_20261026_224', resume_source_run='r1f8gw5h', source_iteration=841, source_completed_updates=842, flux_task_id=os.environ.get('PHP_G1_TASK_ID')")
    return text.replace('g1-low-speed-step-teacher-4096-30m','g1-low-speed-step-resume224-4096-30m')

def bootstrap_source(root):
    text=base.bootstrap_source(root)
    text=replace_once(text,'from flux.g1_lowstep_contract import validate_scope','from flux.g1_resume224_contract import validate_scope')
    text=replace_once(text,"ROOT / 'flux/g1_lowstep_teacher.py'","ROOT / 'flux/g1_resume224_teacher.py'")
    text=replace_once(text,"and formal.get('formal_updates',0)>0)",
        "and formal.get('formal_updates',0)>0 and json.loads((output/'resume_result.json').read_text()).get('passed') is True)")
    return text

def worker_source(root):
    text=base.worker_source(root)
    text=replace_once(text,"    parser.add_argument('--lowstep-input-b64',required=True)",
        "    parser.add_argument('--lowstep-input-b64',required=True)\n    parser.add_argument('--g1-resume-input-b64',required=True)")
    text=replace_once(text,'    updates = 20020',
        "    updates = 20020\n    from flux.g1_resume224_input import validate_control,fetch_checkpoint,CHECKPOINT_SHA256,SOURCE_TASK\n    source_control=decode_control(args.g1_resume_input_b64)\n    validate_control(source_control)")
    text=replace_once(text,"'g1-lowstep-' + task_id","'g1-resume224-' + task_id")
    text=replace_once(text,"        env['PHP_G1_TASK_ID']=args.task_id",
        "        source=fetch_checkpoint(source_control,OUT/'resume-input')\n        env['PHP_FLUX_RESUME_CHECKPOINT']=str(source)\n        emit('g1_resume224_checkpoint_hash_verified',source_task=SOURCE_TASK,source_sha256=CHECKPOINT_SHA256)\n        env['PHP_G1_TASK_ID']=args.task_id")
    text=replace_once(text,"'formal_seconds':1800,'dataset':'motions-low-step','fresh_training':True,",
        "'formal_seconds':1800,'dataset':'motions-low-step','fresh_training':False,'resume_source_task':SOURCE_TASK,'resume_source_sha256':CHECKPOINT_SHA256,")
    text=replace_once(text,"ROOT/'flux/g1_lowstep_bootstrap.py'","ROOT/'flux/g1_resume224_bootstrap.py'")
    text=text.replace('One fresh G1 Low-speed step4096env trial:','One G1 Low-speed step4096env continuation from224:')
    return text
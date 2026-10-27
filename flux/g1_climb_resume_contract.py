"""One native continuation per approved climb teacher, total learning<=1800s."""
from pathlib import Path
from flux import g1_climb_contract as base
from flux.g1_climb_resume_input import SOURCE_TASK,SOURCE_RUN,SOURCE_ITERATION,SOURCE_UPDATES,CHECKPOINT_SHA256
from flux.formal_contract import replace_once,execute
budget_times=base.budget_times
def validate_scope(scope):
    for key,value in {'fresh_training':False,'resume_source_task':SOURCE_TASK,'resume_source_sha256':CHECKPOINT_SHA256}.items():
        if type(scope.get(key)) is not type(value) or scope[key]!=value:raise ValueError('Invalid climb continuation '+key)
    base.validate_scope({**scope,'fresh_training':True})
    return scope
def teacher_source(root):
    text=base.teacher_source(root)
    old="resume_checkpoint = None\n    if os.environ.get('PHP_FLUX_RESUME_CHECKPOINT'):\n        raise ValueError('"+base.kind().title()+"-climb-76 must train fresh')"
    text=replace_once(text,old,"resume_checkpoint = os.environ.get('PHP_FLUX_RESUME_CHECKPOINT')\n    if not resume_checkpoint:raise ValueError('Verified climb checkpoint required')")
    text=replace_once(text,"target = 'flux.g1_climb_algo.G1ClimbPPO'","target = 'flux.g1_climb_resume_algo.G1ClimbResumePPO'")
    text=replace_once(text,"    if resume_checkpoint and args.updates != 2:\n        raise ValueError('Resume validation requires exactly two updates')","    if args.updates!=20020:raise ValueError('Climb continuation update ceiling changed')")
    text=replace_once(text,"fresh_training=True, flux_task_id=os.environ.get('PHP_G1_TASK_ID')",f"fresh_training=False, resume_source_task={SOURCE_TASK!r}, resume_source_run={SOURCE_RUN!r}, source_iteration={SOURCE_ITERATION}, source_completed_updates={SOURCE_UPDATES}, flux_task_id=os.environ.get('PHP_G1_TASK_ID')")
    return text.replace('teacher-4096-30m','continuation-4096-30m')
def bootstrap_source(root):
    text=base.bootstrap_source(root)
    for a,b in [('from flux.g1_climb_contract import validate_scope','from flux.g1_climb_resume_contract import validate_scope'),('from flux.g1_climb_contract import budget_times','from flux.g1_climb_resume_contract import budget_times'),("ROOT / 'flux/g1_climb_teacher.py'","ROOT / 'flux/g1_climb_resume_teacher.py'"),("and formal.get('formal_updates',0)>0)","and formal.get('formal_updates',0)>0 and json.loads((output/'resume_result.json').read_text()).get('passed') is True)")]:
        text=replace_once(text,a,b)
    return text
def worker_source(root):
    text=base.worker_source(root)
    for a,b in [("    parser.add_argument('--climb-input-b64',required=True)","    parser.add_argument('--climb-input-b64',required=True)\n    parser.add_argument('--g1-resume-input-b64',required=True)"),
      ('    updates = 20020',"    updates = 20020\n    from flux.g1_climb_resume_input import validate_control,fetch_checkpoint,CHECKPOINT_SHA256,SOURCE_TASK\n    source_control=decode_control(args.g1_resume_input_b64)\n    validate_control(source_control)"),
      ("        env['PHP_G1_TASK_ID']=args.task_id","        source=fetch_checkpoint(source_control,OUT/'resume-input')\n        env['PHP_FLUX_RESUME_CHECKPOINT']=str(source)\n        emit('climb_source_hash_verified',source_task=SOURCE_TASK,source_sha256=CHECKPOINT_SHA256)\n        env['PHP_G1_TASK_ID']=args.task_id"),
      ("'fresh_training':True,","'fresh_training':False,'resume_source_task':SOURCE_TASK,'resume_source_sha256':CHECKPOINT_SHA256,"),
      ("ROOT/'flux/g1_climb_bootstrap.py'","ROOT/'flux/g1_climb_resume_bootstrap.py'")]:
        text=replace_once(text,a,b)
    return text
def algo_source(root):
    text=Path(root).joinpath('g1_resume224_algo.py').read_text()
    for a,b in [('g1_resume224','g1_climb_resume'),('G1Resume224','G1ClimbResume'),('g1_lowstep_algo','g1_climb_algo'),('G1LowstepPPO','G1ClimbPPO'),('224','climb')]:text=text.replace(a,b)
    return text
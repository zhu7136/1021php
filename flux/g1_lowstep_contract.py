"""Exact-source adaptation of tested infrastructure; old task protections unchanged."""
from pathlib import Path
from flux.formal_contract import replace_once,execute

def budget_times(validation_deadline,gate):
    if not gate or gate.get('passed') is not True:return validation_deadline-30,validation_deadline
    return gate['formal_started_monotonic']+1680,gate['formal_started_monotonic']+1800

def validate_scope(scope):
    required={'gpu_start_authorized':True,'trial_count':1,'automatic_retries':0,'gpu':'1x4090D-24G',
              'num_envs':4096,'updates':20020,'formal_seconds':1800,'dataset':'motions-low-step',
              'fresh_training':True,'output_upload_probe_passed':True}
    for k,v in required.items():
        if type(scope.get(k)) is not type(v) or scope[k]!=v:raise ValueError('Invalid scope '+k)
    if not scope.get('code_commit') or not scope.get('billing_acceptance'):raise ValueError('Missing authorization')
    if not 180<=scope['worker_seconds']<=3180:raise ValueError('Invalid worker budget')
    return scope

def teacher_source(root):
    text=(Path(root)/'teacher_worker.py').read_text()
    changes=[
      ('from flux.preflight import runtime_checks','from flux.g1_lowstep_input import runtime_checks'),
      ("resume_checkpoint = os.environ.get('PHP_FLUX_RESUME_CHECKPOINT')","resume_checkpoint = None\n    if os.environ.get('PHP_FLUX_RESUME_CHECKPOINT'):\n        raise ValueError('Low-step must train fresh')"),
      ("target = 'flux.resume_algo.ReplayResumePPO' if resume_checkpoint else 'flux.teacher_algo.BoundedPPO'","target = 'flux.g1_lowstep_algo.G1LowstepPPO'"),
      ('save_interval=5','save_interval=500'),
      ("        kw['config'] = config","        config.update(record_role='native_training', dataset='motions-low-step', fresh_training=True, flux_task_id=os.environ.get('PHP_G1_TASK_ID'))\n        kw['config'] = config")]
    for a,b in changes:text=replace_once(text,a,b)
    return text.replace('php-locomotion-teacher-smoke','g1-low-speed-step-teacher-4096-30m')

def bootstrap_source(root):
    text=(Path(root)/'bootstrap.py').read_text()
    start=text.index('def load_approval(');end=text.index('\ndef sha256(',start)
    text=text[:start]+'''def load_approval(path):
    from flux.g1_lowstep_contract import validate_scope
    return validate_scope(json.loads(path.read_text()))

'''+text[end:]
    changes=[
      ('from flux.preflight import static_checks','from flux.g1_lowstep_input import static_checks'),
      ("ROOT / 'flux/teacher_worker.py'","ROOT / 'flux/g1_lowstep_teacher.py'"),
      ("    soft_deadline = started + scope['worker_seconds'] - 120\n    hard_deadline = started + scope['worker_seconds'] - 30",
       "    validation_deadline = float(env['PHP_G1_VALIDATION_DEADLINE'])\n    soft_deadline = validation_deadline-30\n    hard_deadline = validation_deadline"),
      ('            now = time.monotonic()',
       "            now = time.monotonic()\n            gate_path=output/'capacity_gate.json'\n            try:\n                gate=json.loads(gate_path.read_text()) if gate_path.is_file() else {}\n            except json.JSONDecodeError:\n                gate={}\n            from flux.g1_lowstep_contract import budget_times\n            soft_deadline,hard_deadline=budget_times(validation_deadline,gate)"),
      ("passed = child.returncode == 0 and not timed_out and completed == scope['updates'] and bool(checkpoints)",
       "native=json.loads((output/'native_reload.json').read_text()) if (output/'native_reload.json').is_file() else {}\n    formal=json.loads((output/'formal_result.json').read_text()) if (output/'formal_result.json').is_file() else {}\n    passed = (child.returncode == 0 and not timed_out and 20 < completed <= scope['updates'] and bool(checkpoints) and native.get('native_load_passed') is True and formal.get('validation_passed') is True and formal.get('formal_updates',0)>0)"),
      ("'checkpoint_reload_verified': False","'checkpoint_reload_verified': native.get('native_load_passed') is True"),
      ("'acceptance_passed': False","'acceptance_passed': passed, 'teacher_converged': False, 'dataset': 'motions-low-step'")]
    for a,b in changes:text=replace_once(text,a,b)
    return text

def worker_source(root):
    text=(Path(root)/'run_smoke.py').read_text()
    text=replace_once(text,'Exactly one authorized 128env/20update Flux smoke; no account credentials.',
                      'One fresh G1 Low-speed step4096env trial: validate20updates,then formal<=1800seconds; no account credentials.')
    changes=[
      ("    parser.add_argument('--resume-input-b64')","    parser.add_argument('--resume-input-b64')\n    parser.add_argument('--lowstep-input-b64',required=True)"),
      ('updates = 2 if resume_input is not None else 20',"if resume_input is not None:raise ValueError('Fresh G1 low-step never resumes old checkpoints')\n    updates = 20020"),
      ("'php-smoke-' + task_id","'g1-lowstep-' + task_id"),
      ('signal.alarm(1140)','signal.alarm(3180)'),
      ('num_envs=128, updates=updates, worker_seconds=1200','num_envs=4096, updates=updates, worker_seconds=3300'),
      ('from flux.preflight import static_checks','from flux.g1_lowstep_input import static_checks'),
      ("        emit('data_and_dependencies_installed')", "        from flux.g1_lowstep_input import fetch_data\n        lowstep=fetch_data(decode_control(args.lowstep_input_b64),OUT/'lowstep-data')\n        env['PHP_G1_TASK_ID']=args.task_id\n        env['PHP_G1_VALIDATION_DEADLINE']=str(started+1080)\n        emit('lowstep_data_and_dependencies_verified',files=113,motions=56)"),
      ('remaining = min(1050, int(1110-(time.monotonic()-started)))','remaining = int(3060-(time.monotonic()-started))'),
      ("'gpu': '1x4090D-24G', 'worker_seconds': remaining, 'num_envs':128,'updates':updates,","'gpu': '1x4090D-24G', 'worker_seconds': remaining, 'num_envs':4096,'updates':updates,\n                 'formal_seconds':1800,'dataset':'motions-low-step','fresh_training':True,"),
      ('User authorized exactly one bounded smoke and waived currency/unit as a gate','User approved one 4096env30minute formal segment after required validation; actual platform billing accepted; no automatic retries or continuation'),
      ("ROOT/'flux/bootstrap.py'","ROOT/'flux/g1_lowstep_bootstrap.py'"),
      ("str(inputs/'motions-locomotion')","str(lowstep)"),
      ("completion['completed_updates'] != updates","not 20 < completion['completed_updates'] <= updates"),
      ("emit('smoke_worker_passed', completed_updates=updates, native_reload=True,","emit('g1_lowstep_formal_passed', completed_updates=completion['completed_updates'], native_reload=True,"),
      ('int(1198-(time.monotonic()-started))','int(3298-(time.monotonic()-started))'),
      ('        manifest = []',"        candidates=sorted(p for p in OUT.rglob('model_[0-9]*.pt') if '/.wandb/' not in p.as_posix())\n        keep={p.name for p in candidates[-2:]}|{'model_smoke_final.pt','model_validation20.pt'}\n        manifest = []"),
      ("if path.is_file() and path.suffix in ('.json','.pt','.yaml','.yml','.log') and not {'inputs','resume-input'}.intersection(path.relative_to(OUT).parts):", "if path.is_file() and path.suffix in ('.json','.jsonl','.pt','.yaml','.yml','.log','.wandb') and not {'inputs','resume-input','lowstep-data','upstream-wheels'}.intersection(path.relative_to(OUT).parts) and (path.suffix!='.pt' or ('/.wandb/' not in path.as_posix() and path.name in keep)):"),
    ]
    for a,b in changes:text=replace_once(text,a,b)
    text=replace_once(text,"    # Parse control tokens BEFORE any setup/download subprocess or simulator.","    # Parse controls before any simulator.")
    # Live console and redacted persistent log; a heartbeat alone is not PPO evidence.
    start=text.index('def run(command, env, timeout, log_path):');end=text.index('\ndef main():',start)
    text=text[:start]+'''def run(command, env, timeout, log_path):
    from flux.live_monitor import run_live
    return run_live(command,env,timeout,log_path,ROOT)

'''+text[end:]
    return text
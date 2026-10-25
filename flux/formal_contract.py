"""Fail-closed adaptations of the pinned, tested Flux worker infrastructure.

Only formal entry modules use these transformations. Original smoke protections
remain unchanged. Every replacement must match the expected source exactly once.
"""
from pathlib import Path


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('Pinned worker contract changed: '+old[:65])
    return source.replace(old,new,1)


def teacher_source(root):
    source=(Path(root)/'teacher_worker.py').read_text()
    source=replace_once(source,"target = 'flux.resume_algo.ReplayResumePPO' if resume_checkpoint else 'flux.teacher_algo.BoundedPPO'",
                        "target = 'flux.formal_algo.FormalContinuationPPO'")
    source=replace_once(source,'if resume_checkpoint and args.updates != 2:',
                        'if not resume_checkpoint or args.updates != 10000:')
    source=replace_once(source,'Resume validation requires exactly two updates','Formal continuation requires pinned source and 10000-update ceiling')
    source=replace_once(source,'save_interval=5','save_interval=50')
    return source.replace('php-locomotion-teacher-smoke','php-locomotion-teacher-formal-30m')


def validate_scope(scope):
    required={'gpu_start_authorized':True,'trial_count':1,'automatic_retries':0,
              'gpu':'1x4090D-24G','num_envs':128,'updates':10000,
              'outer_worker_seconds':1800,'training_phase':'formal_30m_continuation'}
    for key,value in required.items():
        if type(scope.get(key)) is not type(value) or scope[key] != value:
            raise ValueError('Unexpected formal scope: '+key)
    seconds=scope.get('worker_seconds')
    if type(seconds) is not int or not 180 <= seconds <= 1650:
        raise ValueError('Formal child time outside bounded budget')
    if not scope.get('code_commit') or not scope.get('billing_acceptance'):
        raise ValueError('Missing formal approval provenance')
    if scope.get('output_upload_probe_passed') is not True:
        raise ValueError('Output upload contract must pass first')
    return scope


def bootstrap_source(root):
    source=(Path(root)/'bootstrap.py').read_text()
    start=source.index('def load_approval(')
    end=source.index('\ndef sha256(',start)
    source=source[:start]+'''def load_approval(path):
    from flux.formal_contract import validate_scope
    return validate_scope(json.loads(path.read_text()))

'''+source[end:]
    source=replace_once(source,"ROOT / 'flux/teacher_worker.py'","ROOT / 'flux/formal_teacher.py'")
    source=replace_once(source,"passed = child.returncode == 0 and not timed_out and completed == scope['updates'] and bool(checkpoints)",
        """native_path = output/'native_reload.json'
    resume_path = output/'resume_validation.json'
    native = json.loads(native_path.read_text()) if native_path.is_file() else {}
    resumed = json.loads(resume_path.read_text()) if resume_path.is_file() else {}
    passed = (child.returncode == 0 and not timed_out and 1 <= completed <= scope['updates']
              and bool(checkpoints) and native.get('native_load_passed') is True
              and native.get('completed_updates') == completed and resumed.get('passed') is True
              and resumed.get('completed_updates') == completed)""")
    source=replace_once(source,"'checkpoint_reload_verified': False","'checkpoint_reload_verified': bool(native.get('native_load_passed'))")
    source=replace_once(source,"'acceptance_passed': False","'acceptance_passed': passed, 'formal_stage': True, 'teacher_converged': False")
    return source


def worker_source(root):
    source=(Path(root)/'run_smoke.py').read_text()
    changes=[
      ("updates = 2 if resume_input is not None else 20", "if resume_input is None:\n        raise ValueError('Formal run requires immutable verified source')\n    updates = 10000"),
      ('signal.alarm(1140)','signal.alarm(1680)'),
      ('int(1198-(time.monotonic()-started))','int(1798-(time.monotonic()-started))'),
      ("'php-smoke-' + task_id","'php-formal-' + task_id"),
      ('worker_seconds=1200','worker_seconds=1800'),
      ("source_task='TASK_20261025_027'","source_task='TASK_20261025_033'"),
      ('remaining = min(1050, int(1110-(time.monotonic()-started)))',
       'remaining = min(1650, int(1650-(time.monotonic()-started)))'),
      ("'gpu': '1x4090D-24G', 'worker_seconds': remaining, 'num_envs':128,'updates':updates,",
       "'gpu': '1x4090D-24G', 'worker_seconds': remaining, 'num_envs':128,'updates':updates,\n                 'outer_worker_seconds':1800,'training_phase':'formal_30m_continuation',"),
      ('User authorized exactly one bounded smoke and waived currency/unit as a gate',
       'User selected one 30-minute worker on same GPU; platform startup/clone/release may be separately billed; no automatic retries'),
      ("ROOT/'flux/bootstrap.py'","ROOT/'flux/formal_bootstrap.py'"),
      ("completion['completed_updates'] != updates","not 1 <= completion['completed_updates'] <= updates"),
      ("            replay = json.loads((OUT/'training/replay_validation.json').read_text())\n            resumed = json.loads((OUT/'training/resume_validation.json').read_text())\n            if replay.get('passed') is not True or replay.get('steps') != 32 or resumed.get('passed') is not True:",
       "            resumed = json.loads((OUT/'training/resume_validation.json').read_text())\n            if resumed.get('passed') is not True or resumed.get('completed_updates') != completion['completed_updates']:"),
      ("emit('smoke_worker_passed', completed_updates=updates, native_reload=True,",
       "emit('formal_worker_passed', completed_updates=completion['completed_updates'], native_reload=True,"),
      ("        manifest = []", """        # Preserve final and two latest periodic checkpoints in the returned
        # archive. Keep all periodic files on the worker; no deletion occurs.
        candidates=sorted(p.name for p in OUT.rglob('model_[0-9]*.pt') if '/.wandb/' not in p.as_posix())
        keep_checkpoints=set(candidates[-2:]) | {'model_smoke_final.pt'}
        manifest = []"""),
      ("if path.is_file() and path.suffix in ('.json','.pt','.yaml','.yml','.log') and not {'inputs','resume-input'}.intersection(path.relative_to(OUT).parts):",
       "if path.is_file() and path.suffix in ('.json','.jsonl','.pt','.yaml','.yml','.log','.wandb') and not {'inputs','resume-input'}.intersection(path.relative_to(OUT).parts) and (path.suffix != '.pt' or ('/.wandb/' not in path.as_posix() and path.name in keep_checkpoints)):")]
    for old,new in changes:
        source=replace_once(source,old,new)
    source=source.replace('from flux.resume_input import','from flux.formal_input import')
    source=source.replace('smoke_begin','formal_begin').replace('smoke_entrypoint_complete','formal_entrypoint_complete')
    source=source.replace('smoke_failed','formal_failed')
    return source


def execute(builder, namespace):
    root=Path(namespace['__file__']).resolve().parent
    source=builder(root)
    exec(compile(source,namespace['__file__'],'exec'),namespace)
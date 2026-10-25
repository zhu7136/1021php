"""Local preparation only; secure Flux credential injection remains unverified."""
from flux import longtrain_contract as old
replace_once = old.replace_once
execute = old.execute

def validate_scope(scope):
    required = {'gpu_start_authorized':True,'trial_count':1,'automatic_retries':0,
                'gpu':'1x4090D-24G','num_envs':4096,'updates':1000,
                'outer_worker_seconds':2700,'training_phase':'task048_continue1000_online',
                'output_upload_probe_passed':True,'online_monitoring_required':True,
                'checkpoint_every_new_updates':500}
    for key,value in required.items():
        if type(scope.get(key)) is not type(value) or scope[key] != value:
            raise ValueError('Unexpected online continuation scope: '+key)
    seconds = scope.get('worker_seconds')
    if type(seconds) is not int or not 180 <= seconds <= 2520:
        raise ValueError('Child runtime outside proposed45minute budget')
    if not scope.get('code_commit') or not scope.get('billing_acceptance'):
        raise ValueError('Missing execution approval/provenance')
    return scope

def teacher_source(root):
    text = old.teacher_source(root)
    for a,b in [
        ('flux.longtrain_algo.LongtrainPPO','flux.continue1000_algo.Continue1000PPO'),
        ('args.updates != 20000','args.updates != 1000'),
        ('20000-update ceiling','1000-update target'),
        ("mode='offline'","mode='online', id=None, resume='never'"),
        ("os.environ['WANDB_MODE'] = 'offline'","os.environ.update(online_environment(os.environ))"),
        ('    from flux.preflight import runtime_checks',
         '    from flux.live_monitor import online_environment\n    os.environ.update(online_environment(os.environ))\n    from flux.preflight import runtime_checks'),
        ("        kw['config'] = config", """        config.update(continuation_source_task='TASK_20261025_048',
                      continuation_source_iteration=3847, requested_new_updates=1000,
                      checkpoint_every_new_updates=500)
        kw['config'] = config
        kw['mode'] = 'online'
        kw['resume'] = 'never'
        kw.pop('id', None)"""),
        ('        return original_init(*a, **kw)', """        run = original_init(*a, **kw)
        if run is None or run.settings.mode != 'online':
            raise ValueError('Online native run failed; no offline fallback')
        print(json.dumps({'event':'native_online_run_started','run_id':run.id}),flush=True)
        return run"""),
        ('    wandb.init = init_with_provenance', """    wandb.init = init_with_provenance
    original_log = wandb.log
    def log_with_progress(data, *a, **kw):
        step = kw.get('step')
        if isinstance(step, int) and 3847 <= step < 4847:
            data = dict(data, **{'Continuation/completed_updates':step-3847+1,
                               'Continuation/target_updates':1000,
                               'Continuation/native_iteration':step})
        return original_log(data, *a, **kw)
    wandb.log = log_with_progress"""),
    ]:
        text = replace_once(text,a,b)
    return text.replace('php-teacher-fullscale4096-1h','php-teacher-task048-continue1000-online')

def _online_env(text, anchor):
    text = replace_once(text,"'WANDB_API_KEY',",'')
    text = replace_once(text,"WANDB_MODE='offline'","WANDB_MODE='online'")
    text = replace_once(text,anchor,'    from flux.live_monitor import online_environment\n    env = online_environment(env)\n'+anchor)
    return text

def bootstrap_source(root):
    text = old.bootstrap_source(root)
    for a,b in [
        ('from flux.longtrain_contract import validate_scope','from flux.continue1000_contract import validate_scope'),
        ('flux/longtrain_teacher.py','flux/continue1000_teacher.py'),
        ("1 <= completed <= scope['updates']","completed == scope['updates']"),
        ("'offline; sync after verified retrieval'","'online; worker connectivity and uploads require verification'"),
    ]:
        text = replace_once(text,a,b)
    return _online_env(text,"    soft_deadline = started + scope['worker_seconds'] - 120")

def worker_source(root):
    text = old.worker_source(root)
    for a,b in [
        ('updates = 20000','updates = 1000'),
        ('signal.alarm(3480)','signal.alarm(2580)'),
        ('int(3598-(time.monotonic()-started))','int(2698-(time.monotonic()-started))'),
        ('worker_seconds=3600','worker_seconds=2700'),
        ('remaining = min(3420, int(3420-(time.monotonic()-started)))','remaining = min(2520, int(2520-(time.monotonic()-started)))'),
        ("'outer_worker_seconds':3600","'outer_worker_seconds':2700, 'online_monitoring_required':True, 'checkpoint_every_new_updates':500"),
        ('fullscale_4096_1h_segment','task048_continue1000_online'),
        ("source_task='TASK_20261025_045'","source_task='TASK_20261025_048'"),
        ('flux/longtrain_bootstrap.py','flux/continue1000_bootstrap.py'),
        ("not 1 <= completion['completed_updates'] <= updates","completion['completed_updates'] != updates"),
        ('one3600second4096envsegment','one2700second4096env1000updatesegment'),
    ]:
        text = replace_once(text,a,b)
    if text.count('flux.longtrain_input') != 2:
        raise ValueError('Pinned source validation/delivery imports changed')
    text = text.replace('flux.longtrain_input','flux.continue1000_input')
    start = text.index('def run(command, env, timeout, log_path):')
    end = text.index('\ndef main():',start)
    text = text[:start]+'''def run(command, env, timeout, log_path):
    from flux.live_monitor import run_live
    return run_live(command, env, timeout, log_path, cwd=ROOT)

'''+text[end:]
    # Worker indentation is nested in the protected preparation try block.
    text = replace_once(text,"'WANDB_API_KEY',",'')
    text = replace_once(text,"WANDB_MODE='offline'","WANDB_MODE='online'")
    text = replace_once(text,"        env.pop('PHP_FLUX_RESUME_CHECKPOINT', None)",
        "        from flux.live_monitor import online_environment\n        env = online_environment(env)\n        env.pop('PHP_FLUX_RESUME_CHECKPOINT', None)")
    return text.replace('longtrain_', 'continue1000_')
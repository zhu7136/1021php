"""One-hour4096 native teacher segment; old smoke/capacity contracts untouched."""
from flux import formal_contract as old
replace_once=old.replace_once
execute=old.execute

def validate_scope(scope):
    required={'gpu_start_authorized':True,'trial_count':1,'automatic_retries':0,
              'gpu':'1x4090D-24G','num_envs':4096,'updates':20000,
              'outer_worker_seconds':3600,'training_phase':'fullscale_4096_1h_segment'}
    for key,value in required.items():
        if type(scope.get(key)) is not type(value) or scope[key]!=value:
            raise ValueError('Unexpected longtrain scope: '+key)
    seconds=scope.get('worker_seconds')
    if type(seconds) is not int or not 180<=seconds<=3420:
        raise ValueError('Longtrain child outside one-hour budget')
    if not scope.get('code_commit') or not scope.get('billing_acceptance'):
        raise ValueError('Missing longtrain provenance')
    if scope.get('output_upload_probe_passed') is not True:
        raise ValueError('Output probe missing')
    return scope

def teacher_source(root):
    text=old.teacher_source(root)
    for a,b in [('flux.formal_algo.FormalContinuationPPO','flux.longtrain_algo.LongtrainPPO'),
                ('args.updates != 10000','args.updates != 20000'),
                ('10000-update ceiling','20000-update ceiling'),
                ('save_interval=50','save_interval=500')]:
        text=replace_once(text,a,b)
    return text.replace('php-locomotion-teacher-formal-30m','php-teacher-fullscale4096-1h')

def bootstrap_source(root):
    text=old.bootstrap_source(root)
    for a,b in [('from flux.formal_contract import validate_scope','from flux.longtrain_contract import validate_scope'),
                ('flux/formal_teacher.py','flux/longtrain_teacher.py'),
                ("'formal_stage': True","'formal_stage': True, 'fullscale_segment': True")]:
        text=replace_once(text,a,b)
    return text

def worker_source(root):
    text=old.worker_source(root)
    for a,b in [('updates = 10000','updates = 20000'),
                ('signal.alarm(1680)','signal.alarm(3480)'),
                ('int(1798-(time.monotonic()-started))','int(3598-(time.monotonic()-started))'),
                ('worker_seconds=1800','worker_seconds=3600'),
                ('remaining = min(1650, int(1650-(time.monotonic()-started)))',
                 'remaining = min(3420, int(3420-(time.monotonic()-started)))'),
                ("'num_envs':128","'num_envs':4096"),
                ('num_envs=128','num_envs=4096'),
                ("'outer_worker_seconds':1800","'outer_worker_seconds':3600"),
                ('formal_30m_continuation','fullscale_4096_1h_segment'),
                ("source_task='TASK_20261025_033'","source_task='TASK_20261025_045'"),
                ('flux/formal_bootstrap.py','flux/longtrain_bootstrap.py'),
                ('User selected one 30-minute worker on same GPU; platform startup/clone/release may be separately billed; no automatic retries',
                 'User directly requested GPU training without another authorization gate; one3600second4096envsegment,1x4090D24GB,retries0; platform startup/clone/release may be separately billed')]:
        text=replace_once(text,a,b)
    text=text.replace('from flux.formal_input import','from flux.longtrain_input import')
    for a,b in [('formal_begin','longtrain_begin'),('formal_failed','longtrain_failed'),
                ('formal_worker_passed','longtrain_worker_passed'),
                ('formal_entrypoint_complete','longtrain_entrypoint_complete')]:
        text=text.replace(a,b)
    return text
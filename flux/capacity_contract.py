"""4096env/20update capacity verification; original128env guards remain intact."""
from flux import formal_contract as old
replace_once=old.replace_once
execute=old.execute

def validate_scope(scope):
    expected={'gpu_start_authorized':True,'trial_count':1,'automatic_retries':0,
              'gpu':'1x4090D-24G','num_envs':4096,'updates':20,
              'outer_worker_seconds':1800,'training_phase':'capacity_4096_validation'}
    for key,value in expected.items():
        if type(scope.get(key)) is not type(value) or scope[key]!=value:raise ValueError('Unexpected capacity scope: '+key)
    seconds=scope.get('worker_seconds')
    if type(seconds) is not int or not 180<=seconds<=1650:raise ValueError('Capacity time outside budget')
    if not scope.get('code_commit') or not scope.get('billing_acceptance'):raise ValueError('Missing capacity provenance')
    if scope.get('output_upload_probe_passed') is not True:raise ValueError('Output probe missing')
    return scope

def teacher_source(root):
    source=old.teacher_source(root)
    for a,b in [
        ('flux.formal_algo.FormalContinuationPPO','flux.capacity_algo.CapacityPPO'),
        ('args.updates != 10000','args.updates != 20'),
        ('10000-update ceiling','20-update capacity ceiling'),
        ('save_interval=50','save_interval=5'),
    ]:source=replace_once(source,a,b)
    return source.replace('php-locomotion-teacher-formal-30m','php-teacher-capacity-4096-20updates')

def bootstrap_source(root):
    source=old.bootstrap_source(root)
    for a,b in [
        ('from flux.formal_contract import validate_scope','from flux.capacity_contract import validate_scope'),
        ('flux/formal_teacher.py','flux/capacity_teacher.py'),
        ('and resumed.get(\'completed_updates\') == completed)',
         "and resumed.get('completed_updates') == completed\n              and completed == scope['updates'] and capacity.get('passed') is True)"),
        ("    passed = (child.returncode", "    capacity_path=output/'capacity.json'\n    capacity=json.loads(capacity_path.read_text()) if capacity_path.is_file() else {}\n    passed = (child.returncode"),
        ("'formal_stage': True", "'formal_stage': False, 'capacity_validation': True"),
    ]:source=replace_once(source,a,b)
    return source

def worker_source(root):
    source=old.worker_source(root)
    for a,b in [
        ('updates = 10000','updates = 20'),
        ("'num_envs':128","'num_envs':4096"),
        ('num_envs=128','num_envs=4096'),
        ("source_task='TASK_20261025_033'","source_task='TASK_20261025_044'"),
        ('formal_30m_continuation','capacity_4096_validation'),
        ('flux/formal_bootstrap.py','flux/capacity_bootstrap.py'),
        ("not 1 <= completion['completed_updates'] <= updates","completion['completed_updates'] != updates"),
        ("            if resumed.get('passed') is not True or resumed.get('completed_updates') != completion['completed_updates']:",
         "            capacity=json.loads((OUT/'training/capacity.json').read_text())\n            if capacity.get('passed') is not True or resumed.get('passed') is not True or resumed.get('completed_updates') != completion['completed_updates']:"),
    ]:source=replace_once(source,a,b)
    source=source.replace('from flux.formal_input import','from flux.capacity_input import')
    source=source.replace('formal_begin','capacity_begin').replace('formal_failed','capacity_failed')
    source=source.replace('formal_worker_passed','capacity_worker_passed').replace('formal_entrypoint_complete','capacity_entrypoint_complete')
    return source
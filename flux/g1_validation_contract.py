"""Exactly one fresh G1 4096-env / 20-update GPU validation; no formal stage."""
from pathlib import Path
from flux.formal_contract import execute, replace_once
from flux.g1_lowstep_contract import teacher_source as full_teacher_source


def validate_scope(scope):
    required = {'gpu_start_authorized': True, 'trial_count': 1, 'automatic_retries': 0,
                'gpu': '1x4090D-24G', 'num_envs': 4096, 'updates': 20,
                'formal_seconds': 0, 'dataset': 'motions-low-step', 'fresh_training': True,
                'output_upload_probe_passed': True}
    for key, value in required.items():
        if type(scope.get(key)) is not type(value) or scope[key] != value:
            raise ValueError('Invalid validation scope ' + key)
    if not scope.get('code_commit') or not scope.get('billing_acceptance'):
        raise ValueError('Missing authorization')
    if type(scope.get('worker_seconds')) is not int or not 180 <= scope['worker_seconds'] <= 1080:
        raise ValueError('Invalid validation worker budget')
    return scope


def teacher_source(root):
    text = full_teacher_source(root)
    text = replace_once(text, 'flux.g1_lowstep_algo.G1LowstepPPO', 'flux.g1_validation_algo.G1ValidationPPO')
    text = replace_once(text, 'save_interval=500', 'save_interval=5')
    return text.replace('g1-low-speed-step-teacher-4096-30m', 'g1-low-speed-step-4096-20update-validation')


def bootstrap_source(root):
    text = (Path(root)/'bootstrap.py').read_text()
    start = text.index('def load_approval(')
    end = text.index('\ndef sha256(', start)
    text = text[:start] + '''def load_approval(path):
    from flux.g1_validation_contract import validate_scope
    return validate_scope(json.loads(path.read_text()))

''' + text[end:]
    text = replace_once(text, 'from flux.preflight import static_checks', 'from flux.g1_lowstep_input import static_checks')
    text = replace_once(text, "ROOT / 'flux/teacher_worker.py'", "ROOT / 'flux/g1_validation_teacher.py'")
    text = replace_once(text, "    passed = child.returncode == 0 and not timed_out and completed == scope['updates'] and bool(checkpoints)",
        "    native = json.loads((output/'native_reload.json').read_text()) if (output/'native_reload.json').is_file() else {}\n"
        "    validation = json.loads((output/'validation_result.json').read_text()) if (output/'validation_result.json').is_file() else {}\n"
        "    passed = (child.returncode == 0 and not timed_out and completed == 20 and bool(checkpoints) and native.get('native_load_passed') is True and validation.get('passed') is True)")
    text = replace_once(text, "'checkpoint_reload_verified': False", "'checkpoint_reload_verified': native.get('native_load_passed') is True")
    text = replace_once(text, "'acceptance_passed': False", "'acceptance_passed': passed, 'teacher_converged': False, 'formal_updates': 0, 'dataset': 'motions-low-step'")
    return text


def worker_source(root):
    text = (Path(root)/'run_smoke.py').read_text()
    changes = [
        ('Exactly one authorized 128env/20update Flux smoke; no account credentials.',
         'Exactly one fresh G1 low-step4096env20update validation; zero formal updates; no account credentials.'),
        ("    parser.add_argument('--resume-input-b64')", "    parser.add_argument('--resume-input-b64')\n    parser.add_argument('--lowstep-input-b64', required=True)"),
        ('updates = 2 if resume_input is not None else 20', "if resume_input is not None: raise ValueError('Validation must train fresh')\n    updates = 20\n    lowstep_control = decode_control(args.lowstep_input_b64)"),
        ("'php-smoke-' + task_id", "'g1-validation-' + task_id"),
        ('num_envs=128, updates=updates, worker_seconds=1200', 'num_envs=4096, updates=updates, worker_seconds=1200'),
        ('from flux.preflight import static_checks', 'from flux.g1_lowstep_input import static_checks'),
        ("        emit('data_and_dependencies_installed')", "        from flux.g1_lowstep_input import fetch_data\n        lowstep = fetch_data(lowstep_control, OUT/'lowstep-data')\n        env['PHP_G1_TASK_ID'] = args.task_id\n        emit('lowstep_data_and_dependencies_verified', files=113, motions=56)"),
        ("'num_envs':128,'updates':updates,", "'num_envs':4096,'updates':updates,\n                 'formal_seconds':0,'dataset':'motions-low-step','fresh_training':True,"),
        ('User authorized exactly one bounded smoke and waived currency/unit as a gate', 'User approved2026-10-26T09:03:35Z one repaired GPU test; bounded4096env20updates only; zero formal training, automatic retries or continuation; actual platform billing as previously accepted'),
        ("ROOT/'flux/bootstrap.py'", "ROOT/'flux/g1_validation_bootstrap.py'"),
        ("str(inputs/'motions-locomotion')", 'str(lowstep)'),
        ("emit('smoke_worker_passed', completed_updates=updates, native_reload=True,", "emit('g1_validation_passed', completed_updates=updates, native_reload=True,"),
        ("if path.is_file() and path.suffix in ('.json','.pt','.yaml','.yml','.log') and not {'inputs','resume-input'}.intersection(path.relative_to(OUT).parts):",
         "if path.is_file() and path.suffix in ('.json','.jsonl','.pt','.yaml','.yml','.log','.wandb') and not {'inputs','resume-input','lowstep-data','upstream-wheels'}.intersection(path.relative_to(OUT).parts):"),
    ]
    for old, new in changes:
        text = replace_once(text, old, new)
    start = text.index('def run(command, env, timeout, log_path):')
    end = text.index('\ndef main():', start)
    text = text[:start] + '''def run(command, env, timeout, log_path):
    from flux.live_monitor import run_live
    return run_live(command, env, timeout, log_path, ROOT)

''' + text[end:]
    return text
"""Explicitly authorized Flux live console + offline native W&B history."""
from pathlib import Path
from flux import continue1000_contract as old
replace_once=old.replace_once
execute=old.execute

def validate_scope(scope):
    if scope.get('online_monitoring_required') is not False or scope.get('training_phase') != 'task048_continue1000_offline':
        raise ValueError('Explicit offline monitoring selection required')
    old.validate_scope(scope | {'online_monitoring_required':True,'training_phase':'task048_continue1000_online'})
    return scope

def teacher_source(root):
    text=old.teacher_source(root)
    for a,b in [
        ('flux.continue1000_algo.Continue1000PPO','flux.continue1000_offline_algo.OfflineContinue1000PPO'),
        ("mode='online'","mode='offline'"),
        ("kw['mode'] = 'online'","kw['mode'] = 'offline'"),
        ("run.settings.mode != 'online'","run.settings.mode != 'offline'"),
        ('Online native run failed; no offline fallback','Explicit offline native run failed'),
        ('native_online_run_started','native_offline_run_started'),
    ]:
        text=replace_once(text,a,b)
    return text.replace('online_environment','offline_environment').replace('continue1000-online','continue1000-offline')

def bootstrap_source(root):
    text=old.bootstrap_source(root)
    for a,b in [
        ('from flux.continue1000_contract import validate_scope','from flux.continue1000_offline_contract import validate_scope'),
        ('flux/continue1000_teacher.py','flux/continue1000_offline_teacher.py'),
        ("WANDB_MODE='online'","WANDB_MODE='offline'"),
        ('online; worker connectivity and uploads require verification','offline; user selected Flux live console and verified post-run sync'),
    ]:
        text=replace_once(text,a,b)
    return text.replace('online_environment','offline_environment')

def worker_source(root):
    text=old.worker_source(root)
    for a,b in [
        ("'online_monitoring_required':True","'online_monitoring_required':False"),
        ('task048_continue1000_online','task048_continue1000_offline'),
        ('flux/continue1000_bootstrap.py','flux/continue1000_offline_bootstrap.py'),
        ("WANDB_MODE='online'","WANDB_MODE='offline'"),
    ]:
        text=replace_once(text,a,b)
    return text.replace('online_environment','offline_environment')

def algo_source(root):
    text=(Path(root)/'continue1000_algo.py').read_text()
    for a,b in [
        ('class Continue1000PPO','class OfflineContinue1000PPO'),
        ('An online native training run must exist before PPO starts','Explicit offline native training run must exist before PPO starts'),
        ('Checkpoint saved locally but online run is unavailable','Checkpoint saved but offline native run is unavailable'),
        ("        wandb.save(path,base_path=self.log_dir,policy='now')",'        # Offline native helper retains checkpoint for final ZIP retrieval.'),
        ('checkpoint_upload_requested','checkpoint_saved_locally'),
    ]:
        text=replace_once(text,a,b)
    if text.count("wandb.run.settings.mode != 'online'")!=2:raise ValueError('Native logging guards changed')
    return text.replace("wandb.run.settings.mode != 'online'","wandb.run.settings.mode != 'offline'")
"""One 600-second continuation of task216; no retries or silent fresh start."""
from flux.five_distill_contract import SKILLS,TEACHERS,DATA,COMPONENTS,digest,unpack_data
SOURCE_SHA='b08a38e274cd612ec79784399634341e4d290288ead42fd0f10b49e343a6ed87'
SOURCE_BYTES=11248579
SOURCE_ITER=976
MAX_UPDATES=400
WORKER_SECONDS=600
SOFT_SECONDS=480
TRAIN_HARD_SECONDS=540
def validate_scope(s):
 required=dict(gpu_start_authorized=True,trial_count=1,automatic_retries=0,automatic_continuation=False,
  worker_seconds_max=600,gpu='1x4090D24GB',num_envs=80,teachers_frozen=True,student_count=1,
  mode='masked_pure_dagger',teacher_warmup_updates=5,num_steps_per_env=24,num_learning_epochs=2,
  num_mini_batches=8,training_updates_max=400,source_task='TASK_20261027_216',source_iteration=976,
  source_checkpoint_sha256=SOURCE_SHA)
 for k,v in required.items():
  if type(s.get(k)) is not type(v) or s[k]!=v:raise ValueError('Invalid continuation scope '+k)
 if s.get('skills')!=list(SKILLS) or not s.get('code_commit') or not s.get('user_approval_utc'):raise ValueError('Missing source/order/approval')
 return s
def stop_required(completed,now,deadline):
 if type(completed) is not int or completed<0:raise ValueError('Invalid update count')
 return completed>=MAX_UPDATES or now>=deadline
"""One approved, frozen, five-skill TASK216 replay; no training path."""
from flux.distill_resume_contract import SOURCE_SHA,SOURCE_BYTES,DATA,SKILLS,COMPONENTS,digest,unpack_data
APPROVAL='2026-10-28T06:02:59Z'
COUNTS=(100,56,60,40,40)
SELECTED=(0,100,156,216,256)
def input_jobs(datasets,student):
 from flux.distill_diagnostics import validate_input_name
 if len(datasets)!=4:raise ValueError('Expected four additional skill datasets')
 jobs=[('dataset-'+str(i)+'.tar.gz',c) for i,c in enumerate(datasets)]+[('source-student.pt',student)]
 for name,_ in jobs:validate_input_name(name)
 return jobs
def validate_scope(s):
 required=dict(gpu_start_authorized=True,user_approval_utc=APPROVAL,trial_count=1,
  automatic_retries=0,automatic_continuation=False,worker_seconds_max=600,gpu='1x4090D24GB',
  training_updates=0,source_task='TASK_20261027_216',source_checkpoint_sha256=SOURCE_SHA,
  student_only=True,num_envs=1,seconds_per_skill=10,terrain_variants=3,terrain_seed=42)
 for k,v in required.items():
  if type(s.get(k)) is not type(v) or s[k]!=v:raise ValueError('Invalid replay scope '+k)
 if s.get('skills')!=list(SKILLS) or not s.get('code_commit'):raise ValueError('Missing skill order or pinned commit')
 return s
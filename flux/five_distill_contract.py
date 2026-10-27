"""Single authorized five-expert -> one depth student, masked pure DAgger."""
import hashlib,json,math,tarfile
from pathlib import Path

SKILLS=('locomotion','low_step','high_step','low_climb','high_climb')
COMPONENTS=('motions-locomotion','motions-low-step','motions-high-step','motions-low-climb-76','motions-high-climb-76')
TEACHERS=(
 ('rgvmfjy8','task052-model-final-iter4846.pt',4846,10346299,'cb98b0762fbf3b478211bcb9131d53d0330c3459fa7dba04c7874205c1875399'),
 ('9fe07g0l','model_smoke_final.pt',2594,10346235,'416f68e33c18bb39a354f8de66277cb77967b3ac9bbb6022ee631262442174f2'),
 ('puid24lc','model_smoke_final.pt',2629,10346235,'3e9b71151ecdd423d93f6df294408bae6a48ad1ca5e27be82b4d2d253446cb20'),
 ('iodb0n0v','model_smoke_final.pt',1345,10346235,'e82625606bcf622e4c0950a95526dbce7b6a7bca35463fdabf1f2a196caed028'),
 ('od4r78u5','model_smoke_final.pt',1323,10346235,'c2ce9b74df347ce8818e936dd94794c78910d33f34924e2d1ecc9713f745d7be'))
DATA=(
 ('d6a3289e','g1-low-step-input.tar.gz',43034938,'85b3545a8a08e52db471a1de74d484fc48cb33ad09bb9a43f04f6f12782ac5aa'),
 ('33121561','g1-high-step-input.tar.gz',46059115,'d03baeb814b91862eec041a562017456dd850345e34eca0fedd2737edc5e3d59'),
 ('87092c9e','g1-low-climb-76-input.tar.gz',58901739,'d734ad14926e16e4a49649f236c931cf1fe63ad3371107d46c9fdb0433efda82'),
 ('857c1b21','g1-high-climb-76-input.tar.gz',45607220,'835ba791bc5b139149730f4bbafd06c929d1ae704ed3e981ea4dd1e277d8b3ba'),
)
NUM_ENVS=80
WORKER_SECONDS=1800
SOFT_SECONDS=1440
TRAIN_HARD_SECONDS=1580

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def validate_scope(s):
 required={'gpu_start_authorized':True,'trial_count':1,'automatic_retries':0,
  'automatic_continuation':False,'worker_seconds_max':1800,'gpu':'1x4090D24GB',
  'num_envs':80,'teachers_frozen':True,'student_count':1,'mode':'masked_pure_dagger',
  'teacher_warmup_updates':5,'num_steps_per_env':24,'num_learning_epochs':2,'num_mini_batches':8}
 for k,v in required.items():
  if type(s.get(k)) is not type(v) or s[k]!=v:raise ValueError('Invalid authorized scope '+k)
 if s.get('skills')!=list(SKILLS):raise ValueError('Teacher order changed')
 if not s.get('code_commit') or not s.get('user_approval_utc'):raise ValueError('Missing source/approval')
 return s

def masked_loss(student,target):
 """Only valid labels normalize the MSE; invalid zero labels never train toward zero."""
 import torch
 valid=~torch.all(target==0,dim=-1)
 raw=(student-target).square().mean(dim=-1)
 return (raw*valid).sum()/valid.sum().clamp_min(1)

def unpack_data(archive,destination,component):
 """Exact known regular members only, no links/extra bytes, all hashes checked."""
 dest=Path(destination);dest.mkdir(parents=True,exist_ok=False)
 expected={x['path']:x for x in component['files']};seen=set()
 if len(expected)>1000 or sum(x['size'] for x in expected.values())>1024**3:raise ValueError('Extraction bounds')
 with tarfile.open(archive,'r:gz') as tar:
  for m in tar:
   if not m.isfile() or m.name not in expected or m.name in seen:raise ValueError('Unexpected archive member')
   r=expected[m.name];p=dest/m.name
   if not p.resolve().is_relative_to(dest.resolve()) or m.size!=r['size']:raise ValueError('Unsafe archive')
   p.parent.mkdir(parents=True,exist_ok=True)
   with tar.extractfile(m) as incoming,p.open('xb') as out:
    remaining=m.size
    while remaining:
     block=incoming.read(min(1048576,remaining))
     if not block:raise ValueError('Truncated member')
     out.write(block);remaining-=len(block)
   if digest(p)!=r['sha256']:raise ValueError('Data member hash mismatch')
   seen.add(m.name)
 if seen!=set(expected):raise ValueError('Missing archive members')
 return dest
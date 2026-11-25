"""CPU input-delivery check. Never starts a simulator, Flux task or optimizer."""
import argparse,hashlib,io,json,pathlib,sys,xml.etree.ElementTree as ET,zipfile
import numpy as np
import torch
ROOT=pathlib.Path(__file__).resolve().parent
def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--archives-map',help='Optional local JSON mapping manifest archive names to verified local archives');p.add_argument('--report',required=True);a=p.parse_args()
 m=json.loads((ROOT/'input-delivery-manifest.json').read_text());lock=json.loads((ROOT/'teachers.lock.json').read_text());paths=json.loads(pathlib.Path(a.archives_map).read_text()) if a.archives_map else {};checks=[];blockers=[];archives={}
 for row in m['files']:
  path=pathlib.Path(paths.get(row['name'],ROOT/row['name']));assert path.is_file() and path.stat().st_size==row['bytes'] and sha(path)==row['sha256'];archives[row['name']]=path;checks.append('archive:'+row['name'])
 for s in lock['teachers']:
  path=ROOT/s['checkpoint_path'];assert sha(path)==s['sha256'];ck=torch.load(path,map_location='cpu',weights_only=True);st=ck['actor_model_state_dict'];assert st['actor_module.module.0.weight'].shape==(512,449) and st['actor_module.module.6.weight'].shape==(29,128);assert all(torch.isfinite(v).all() for v in st.values());checks.append('teacher:'+s['skill'])
 counts={}
 for skill in ['locomotion','low-step','high-step']:
  name='inputs/'+skill+'/'+skill+'-input.zip'
  with zipfile.ZipFile(archives[name]) as z:
   with np.load(io.BytesIO(z.read('bundle/reference/full-reference.npz')),allow_pickle=False) as n:
    counts[skill]=dict(clips=int(n['motion_ends'].sum()),frames=len(n['joint_pos']),has_vel_cmd='vel_cmd' in n.files);assert counts[skill]['frames']>0
    if 'vel_cmd' not in n.files:blockers.append(skill+': missing vel_cmd required by existing student preset')
 with zipfile.ZipFile(archives['inputs/runtime/runtime.zip']) as z:links={n.attrib['name'] for n in ET.fromstring(z.read('bundle/asset/urdf/HU_D04_01.urdf')).findall('link')}
 for body in ['torso_link','pelvis']:
  if body not in links:blockers.append('G1 camera/raycast body absent in HuD04: '+body)
 report=dict(status='BLOCKED_BEFORE_GPU' if blockers else 'CPU_INPUT_BYTES_ONLY_NOT_GPU_READY',byte_checks_passed=len(checks),checks=checks,references=counts,blockers=blockers,new_gpu_tasks=0,training_updates=0,student_optimizer_updates=0,native_replay_tested=False)
 pathlib.Path(a.report).write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
 return 2 if blockers else 0
if __name__=='__main__':sys.exit(main())
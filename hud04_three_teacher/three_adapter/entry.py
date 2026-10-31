"""CPU-only three-teacher config entry: no simulator imports, execute or submit flag."""
import argparse,json
from pathlib import Path
import numpy as np
from .contract import SKILLS,checkpoint_contract,load_verified
from .prepare_data import digest
from .smoke_config import build_smoke_dict

def verify_data(root):
 root=Path(root);manifest=json.loads((root/'split-manifest.json').read_text())
 if manifest.get('schema')!='hud04-three-skill-data-v1' or manifest.get('assembled') is not True or manifest.get('skills')!=list(SKILLS) or manifest.get('teacher_contract')!=checkpoint_contract():raise ValueError('Not the pinned three-skill dataset')
 required={'train-motion.npz','eval-motion.npz','terrain.obj','terrain_mesh.npz'}|{s+'-teacher-config.json' for s in SKILLS}|{'teachers/'+str(i)+'.pt' for i in range(3)}
 if set(manifest['files'])!=required:raise ValueError('Unexpected or missing dataset file manifest')
 for name,r in manifest['files'].items():
  p=root/name
  if p.stat().st_size!=r['bytes'] or digest(p)!=r['sha256']:raise ValueError('Input hash mismatch: '+name)
 rows=manifest['cases']
 if any(r['skill'] not in SKILLS or r['teacher_index']!=SKILLS.index(r['skill']) or r['split'] not in ('train','eval') for r in rows):raise ValueError('Invalid data routing/split')
 hashes=[{r['motion_sha256'] for r in rows if r['split']==s} for s in ('train','eval')]
 if hashes[0]&hashes[1]:raise ValueError('Train/eval motion leakage')
 for split in ('train','eval'):
  selected=[r for r in rows if r['split']==split];end=0
  if any(not any(r['skill']==s for r in selected) for s in SKILLS):raise ValueError('Missing split skill')
  with np.load(root/(split+'-motion.npz'),allow_pickle=False) as a:
   routes=a['motion_idxs'];ends=a['motion_ends']
   for r in selected:
    start,stop=r['output_frame_range']
    if start!=end or stop-start!=r['frames'] or stop-start<2 or stop>len(routes):raise ValueError('Noncontiguous data case range')
    if not np.all(routes[start:stop]==r['teacher_index']) or not ends[stop-1] or ends[start:stop-1].any():raise ValueError('Array route/clip boundary mismatch')
    end=stop
   if end!=len(routes) or int(ends.sum())!=len(selected):raise ValueError('Data coverage mismatch')
 checked=load_verified([root/'teachers'/(str(i)+'.pt') for i in range(3)]);del checked
 return manifest

def emit(data_root,bundle,output):
 root=Path(data_root).resolve();output=Path(output)
 if output.exists():raise FileExistsError('Refuse to overwrite a preparation result')
 manifest=verify_data(root)
 with np.load(root/'terrain_mesh.npz',allow_pickle=False) as mesh:
  config=build_smoke_dict(json.loads((root/'locomotion-teacher-config.json').read_text()),bundle=bundle,smoke=root,
   teachers=[str(root/'teachers'/(str(i)+'.pt')) for i in range(3)],output=output/'future-native-output',mesh_vertices=mesh['vertices'].tolist(),mesh_faces=mesh['faces'].tolist())
 config['command']['setup_terms']['motion_command']['params']['motion_config']['motion_file']=str(root/'train-motion.npz')
 if config['algo']['_target_']!='three_adapter.native_algo.HuD04SmokeDistillation' or config['training']['checkpoint'] is not None:raise ValueError('Native target/resume contract changed')
 serialized=json.dumps(config,ensure_ascii=False)
 if 'student_adapter.' in serialized or 'low-climb' in serialized or 'high-climb' in serialized:raise ValueError('Old adapter/skills leaked into native configuration')
 output.mkdir(parents=True)
 (output/'native-config.unapproved.json').write_text(json.dumps(config,indent=2,ensure_ascii=False))
 report={'operation':'cpu_config_preparation_only','gpu_started':False,'gpu_execution_approved':False,'existing_tasks_modified':False,
  'contract':checkpoint_contract(),'dataset_manifest_sha256':digest(root/'split-manifest.json'),
  'dataset_counts':{skill:{split:{'clips':sum(r['skill']==skill and r['split']==split for r in manifest['cases']),'frames':sum(r['frames'] for r in manifest['cases'] if r['skill']==skill and r['split']==split)} for split in ('train','eval')} for skill in SKILLS},
  'proposed_native_validation_only':{'num_envs':6,'max_updates':5,'max_training_seconds':600,'automatic_retries':0},
  'unresolved':['native simulator observation/depth dimensions and semantics','native forward/backward, checkpoint save/reload and clean shutdown','per-skill closed-loop performance','separate GPU worker approval, runtime, budget and parent watchdog'],
  'not_supported':['PPO updates','five-teacher continuation','trajectory-exact continuation','production-length training']}
 (output/'cpu-preparation.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
 return report

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ('data-root','bundle','output'):p.add_argument('--'+key,type=Path,required=True)
 a=p.parse_args();print(json.dumps(emit(a.data_root,a.bundle,a.output),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
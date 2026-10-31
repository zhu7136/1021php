"""CPU-only three-skill subset; preserves original clip split and terrain topology."""
import argparse,copy,hashlib,io,json,shutil,zipfile
from pathlib import Path
import numpy as np
from .contract import SKILLS,HASHES,load_verified,checkpoint_contract

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()

def parse_obj(raw):
 vertices=[];faces=[]
 for line in raw.decode().splitlines():
  if line.startswith('v '):vertices.append([float(x) for x in line[2:].split()])
  elif line.startswith('f '):
   face=[int(x.split('/')[0]) for x in line.split()[1:]]
   face=[x-1 if x>0 else len(vertices)+x for x in face]
   if min(face)<0 or max(face)>=len(vertices):raise ValueError('Invalid original OBJ indices')
   for j in range(1,len(face)-1):faces.append([face[0],face[j],face[j+1]])
 return np.asarray(vertices,dtype=np.float64),np.asarray(faces,dtype=np.int64)

def terrain_subset(vertices,faces,source_fourth_vertices,source_fourth_faces):
 shifted=source_fourth_vertices+np.array([0.,4500.,0.])
 matches=[]
 for i in np.flatnonzero(np.all(vertices==shifted[0].astype(np.float32),axis=1)):
  if i+len(shifted)<=len(vertices) and np.array_equal(vertices[i:i+len(shifted)],shifted.astype(np.float32)):matches.append(int(i))
 if len(matches)!=1:raise ValueError('Cannot uniquely prove original fourth-skill vertex block')
 cut=matches[0];end=cut+len(shifted)
 if cut<=0 or ((faces.min(1)<cut)&(faces.max(1)>=cut)).any():raise ValueError('Terrain face crosses proven skill boundary')
 fourth=(faces.min(1)>=cut)&(faces.max(1)<end)
 if not np.array_equal(faces[fourth]-cut,source_fourth_faces):raise ValueError('Fourth-skill face provenance mismatch')
 keep=faces.max(1)<cut
 if not keep.any():raise ValueError('Empty three-skill terrain')
 return vertices[:cut].copy(),faces[keep].copy(),{'original_vertices':len(vertices),'selected_vertices':cut,'selected_faces':int(keep.sum()),'cross_boundary_faces':0,'method':'full_original_fourth_skill_vertex_and_face_block_match_not_coordinate_crop'}

def select_rows(cases,split):
 rows=[];total=0
 for source in cases:
  if source['split']!=split or source['skill'] not in SKILLS:continue
  if source['teacher_index']!=SKILLS.index(source['skill']):raise ValueError('Teacher/data routing mismatch')
  row=copy.deepcopy(source);start,end=row['output_frame_range']
  if end-start!=row['frames'] or end-start<2:raise ValueError('Invalid clip range')
  row['original_output_frame_range']=[start,end];row['output_frame_range']=[total,total+row['frames']];total+=row['frames'];rows.append(row)
 if any(not any(r['skill']==s for r in rows) for s in SKILLS):raise ValueError('Missing selected skill')
 return rows,total

def prepare(core,motion_root,fourth_archive,low_step,output):
 core=Path(core);motion_root=Path(motion_root);output=Path(output)
 if output.exists():raise FileExistsError('Refuse to overwrite existing prepared data')
 if digest(fourth_archive)!='42dc240a35adea7cdaf49cfb38eaa83df89d39b86668dfa325580f2a79243128':raise ValueError('Fourth-skill original archive changed')
 output.mkdir(parents=True);teacher_dir=output/'teachers';teacher_dir.mkdir()
 with zipfile.ZipFile(core) as z:
  entries={r['path']:r for r in json.loads(z.read('payload-manifest.json'))['files']}
  def member(name):
   b=z.read(name);r=entries[name]
   if len(b)!=r['bytes'] or hashlib.sha256(b).hexdigest()!=r['sha256']:raise ValueError('Core member changed: '+name)
   return b
  source=json.loads(member('formal_data/split-manifest.json'));all_rows=[];files={}
  for i,skill in enumerate(SKILLS):
   b=Path(low_step).read_bytes() if i==1 else member('teachers/'+str(i)+'.pt')
   if hashlib.sha256(b).hexdigest()!=HASHES[i]:raise ValueError('Selected teacher changed')
   (teacher_dir/(str(i)+'.pt')).write_bytes(b)
   (output/(skill+'-teacher-config.json')).write_bytes(member('formal_data/'+skill+'-teacher-config.json'))
  checked=load_verified([teacher_dir/(str(i)+'.pt') for i in range(3)]);del checked
  for split in ('train','eval'):
   path=motion_root/(split+'-motion.npz');expected=source['files'][path.name]
   if path.stat().st_size!=expected['bytes'] or digest(path)!=expected['sha256']:raise ValueError('Motion source hash mismatch')
   rows,total=select_rows(source['cases'],split);all_rows.extend(rows)
   with np.load(path,allow_pickle=False) as data:
    routes=data['motion_idxs'];ends=data['motion_ends'];values={}
    for row in rows:
     start,end=row['original_output_frame_range']
     if start<0 or end>len(routes) or not np.all(routes[start:end]==row['teacher_index']):raise ValueError('Source route disagreement')
     if not ends[end-1] or ends[start:end-1].any() or (start and not ends[start-1]):raise ValueError('Broken original clip boundary')
    for key in data.files:
     a=data[key]
     if key in ('fps','joint_names','body_names'):values[key]=a.copy();continue
     if a.shape[0]!=len(routes):raise ValueError('Unexpected non-frame motion field')
     values[key]=np.concatenate([a[s:e] for s,e in (r['original_output_frame_range'] for r in rows)])
     if np.issubdtype(values[key].dtype,np.number) and not np.isfinite(values[key]).all():raise ValueError('Nonfinite selected data')
    if len(values['motion_idxs'])!=total or set(np.unique(values['motion_idxs']))!={0,1,2} or int(values['motion_ends'].sum())!=len(rows):raise ValueError('Subset coverage mismatch')
    np.savez(output/path.name,**values)
   print(json.dumps({'event':'three_skill_motion_prepared','split':split,'frames':total,'clips':len(rows)}),flush=True)
  train={r['motion_sha256'] for r in all_rows if r['split']=='train'};test={r['motion_sha256'] for r in all_rows if r['split']=='eval'}
  if train&test:raise ValueError('Motion hash train/eval leakage')
  with np.load(io.BytesIO(member('formal_data/terrain_mesh.npz')),allow_pickle=False) as mesh:vertices=mesh['vertices'];faces=mesh['faces']
  with zipfile.ZipFile(fourth_archive) as f:ov,of=parse_obj(f.read('bundle/terrain/full-terrain.obj'))
  v,f,terrain_proof=terrain_subset(vertices,faces,ov,of)
  original_obj=member('formal_data/terrain.obj');objv,objf=parse_obj(original_obj)
  if not np.array_equal(objv.astype(np.float32),vertices) or not np.array_equal(objf,faces):raise ValueError('OBJ/render mesh differ')
  # Preserve source double precision for simulator OBJ; camera uses source float32.
  np.savez_compressed(output/'terrain_mesh.npz',vertices=v,faces=f)
  with (output/'terrain.obj').open('w') as stream:
   for point in objv[:len(v)]:stream.write('v '+' '.join(format(float(x),'.12g') for x in point)+'\n')
   for face in f+1:stream.write('f '+' '.join(map(str,face))+'\n')
  for p in sorted(output.rglob('*')):
   if p.is_file():files[p.relative_to(output).as_posix()]={'bytes':p.stat().st_size,'sha256':digest(p)}
  result={'schema':'hud04-three-skill-data-v1','assembled':True,'skills':list(SKILLS),'teacher_contract':checkpoint_contract(),'cases':all_rows,'files':files,'terrain_proof':terrain_proof,'total_clips':len(all_rows),'total_frames':sum(r['frames'] for r in all_rows),'source_core_sha256':digest(core),'source_split_manifest_sha256':hashlib.sha256(member('formal_data/split-manifest.json')).hexdigest(),'motion_hash_train_eval_overlap':0,'scope':'held-out student imitation; teachers saw source motions; shared terrain NOT OOD','gpu_execution_approved':False}
  (output/'split-manifest.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
 return result

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('core','motion-root','fourth-archive','low-step','output'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();r=prepare(a.core,a.motion_root,a.fourth_archive,a.low_step,a.output)
 print(json.dumps({k:r[k] for k in ('schema','total_clips','total_frames','terrain_proof','gpu_execution_approved')}),flush=True)
if __name__=='__main__':main()
"""Isolate two held-out climbing clips while retaining their aligned terrain."""
import hashlib,json
from pathlib import Path
import numpy as np

SCOPE={'max_tasks':1,'gpu_num':1,'worker_seconds_max':300,'automatic_retries':0,
       'training_updates':0,'preview_seconds':2,'max_policy_steps':100,'num_envs':1,
       'skills':['low-climb'],'max_clips':1}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def isolate(data_root,output):
    data_root=Path(data_root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((data_root/'split-manifest.json').read_text())
    if sha(data_root/'eval-motion.npz')!=manifest['files']['eval-motion.npz']['sha256']:raise ValueError('Eval source hash')
    rows=[]
    with np.load(data_root/'eval-motion.npz',allow_pickle=False) as z:
        arrays={k:z[k] for k in z.files}
    with np.load(data_root/'terrain_mesh.npz',allow_pickle=False) as z:
        vertices=z['vertices']
    for route,skill in [(3,'low-climb'),(4,'high-climb')]:
        if skill not in SCOPE['skills']:continue
        pool=[r for r in manifest['cases'] if r['split']=='eval' and r['teacher_index']==route]
        # Preselect the first forward approach, not a successful rollout selected after the fact.
        row=next(r for r in pool if r['clip'].startswith('forward_walk' if route==3 else 'run_forward_'))
        start,end=row['output_frame_range'];n=end-start
        clip={k:v.copy() if k in ('fps','joint_names','body_names') else v[start:end].copy() for k,v in arrays.items()}
        if n!=row['frames'] or not clip['motion_ends'][-1] or clip['motion_ends'][:-1].any() or not (clip['motion_idxs']==route).all():raise ValueError('Clip boundary/route mismatch')
        # Keep source world coordinates; terrain and depth mesh remain the same aligned pair.
        target=output/(skill+'.npz');np.savez(target,**clip)
        roots=clip['joint_pos'][:,:3];lo=roots.min(axis=0);hi=roots.max(axis=0)
        nearby=vertices[((vertices[:,:2]>=lo[:2]-2).all(axis=1))&((vertices[:,:2]<=hi[:2]+2).all(axis=1))]
        if len(nearby)<4 or float(np.ptp(nearby[:,2]))<.2:raise ValueError('No raised obstacle near reference trajectory')
        center=(lo+hi)/2;extent=float(max(hi[0]-lo[0],hi[1]-lo[1]))
        distance=max(4.,extent*1.1)
        camera={'type':'fixed','position':(center+np.array([0.,-distance,distance*.65])).tolist(),
                'target':(center+np.array([0.,0.,-.25])).tolist()}
        rows.append({'skill':skill,'route':route,'case':row,'motion_file':str(target),'motion_sha256':sha(target),
                     'fps':float(np.asarray(clip['fps']).reshape(-1)[0]),'camera':camera,
                     'nearby_terrain_height_range_m':[float(nearby[:,2].min()),float(nearby[:,2].max())],
                     'reference_root_bounds_m':[lo.tolist(),hi.tolist()]})
    (output/'obstacle-inputs.json').write_text(json.dumps(rows,indent=2))
    return rows

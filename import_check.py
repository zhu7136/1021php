"""Version/origin and representative binary imports; no simulator or W&B run."""
import argparse,importlib,importlib.metadata as metadata,json,platform,sys
from pathlib import Path

def normalize(name):return name.lower().replace('_','-').replace('.','-')
def inside(path,root):
 try:Path(path).resolve().relative_to(root.resolve());return True
 except (ValueError,TypeError):return False
def inspect_install(target,manifest):
 target=Path(target).resolve()
 if any(name in sys.modules for name in ['numpy','torch','wandb','onnx']):raise ValueError('critical package preimported before overlay')
 sys.path.insert(0,str(target))
 expected={normalize(r['name']):r['version'] for r in manifest['files']};found={}
 for d in metadata.distributions(path=[str(target)]):
  name=normalize(d.metadata['Name'])
  if name in found:raise ValueError('duplicate isolated distribution')
  found[name]=d.version
 if found!=expected:raise ValueError('isolated distribution set/version mismatch')
 for name,version in expected.items():
  d=metadata.distribution(name)
  if d.version!=version or not inside(d.locate_file(''),target):raise ValueError('effective distribution shadowing')
 result=[]
 for name in ['numpy','torch','wandb','onnx','yourdfpy','trimesh','shapely','manifold3d','mapbox_earcut','vhacdx','xatlas','xxhash','embreex.rtcore_scene','stl','zmq']:
  module=importlib.import_module(name);origin=getattr(module,'__file__',None)
  if not origin:raise ValueError('module source absent')
  if name not in ['torch','trimesh'] and not inside(origin,target):raise ValueError('critical module outside overlay')
  if name in ['torch','trimesh'] and (inside(origin,target) or not str(Path(origin).resolve()).startswith(('/workspace/isaaclab/','/isaac-sim/'))):raise ValueError('image module origin changed')
  row=dict(module=name,origin=origin,version=str(getattr(module,'__version__','not_exposed')));result.append(row)
  print(json.dumps(dict(event='hud04_isolation_import',**row)),flush=True)
 np=sys.modules['numpy'];torch=sys.modules['torch']
 if np.__version__!='1.26.4' or torch.__version__!='2.7.0+cu128':raise ValueError('critical runtime version changed')
 array=np.array([1.,2.,3.],dtype=np.float32);tensor=torch.from_numpy(array);tensor.add_(2)
 if not np.array_equal(array,np.array([3.,4.,5.],dtype=np.float32)) or not np.array_equal(tensor.numpy(),array):raise ValueError('numpy torch interop failed')
 return dict(passed=True,python_version=platform.python_version(),isolated_distributions=found,imports=result,numpy_torch_interop_passed=True,torch_version=torch.__version__,image_torch_preserved=True,simulator_started=False,ppo_updates=0,wandb_run_created=False,full_runtime_closure_proven=False)
def main():
 p=argparse.ArgumentParser();p.add_argument('--target',required=True);p.add_argument('--manifest',required=True);a=p.parse_args()
 result=inspect_install(Path(a.target),json.loads(Path(a.manifest).read_text()));print(json.dumps(dict(event='hud04_isolation_import_final',**result)),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as error:
  import traceback
  print(json.dumps(dict(event='hud04_isolation_import_failure',error_type=type(error).__name__,error_message=str(error)[:400],frames=[dict(file=Path(f.filename).name,line=f.lineno,function=f.name) for f in traceback.extract_tb(error.__traceback__)[-8:]],passed=False)),flush=True);sys.exit(1)
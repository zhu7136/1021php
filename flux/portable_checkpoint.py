"""Publish a unique native checkpoint only after exact reload;no credentials or simulator imports."""
from pathlib import Path
import hashlib,os,re

SOURCE_SHA='3dd4286a4b5e825f8e224f476cc5cc510a375c77a9ffabd168f518d946c680f1'

def equal(a,b):
    import torch
    if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a.detach().cpu(),b.detach().cpu())
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return isinstance(b,type(a)) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

def filename(run_path):
    if not isinstance(run_path,str) or not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/[a-z0-9]{8}',run_path):
        raise ValueError('Exact run path required')
    if any(part in ('.','..') for part in run_path.split('/')):raise ValueError('Invalid run path segment')
    # Match the platform's already-observed model_00790.pt checkpoint convention.
    # Uniqueness is task + iteration;the embedded exact run path prevents cross-task mixups.
    return 'model_00201.pt'

def publish(algo,native_path,summary,run_path,directory):
    import torch
    name=filename(run_path)
    if (summary['source_iteration']!=200 or summary['source_sha256']!=SOURCE_SHA
        or summary['completed_updates']!=1 or summary['last_iteration']!=201 or summary['transitions']!=768
        or summary['restore']['actor_critic_optimizer_exact'] is not True
        or summary['final_checkpoint']['iteration']!=201
        or summary['final_checkpoint']['native_save_reload_verified'] is not True
        or summary['final_checkpoint']['actor_critic_optimizer_iteration_exact'] is not True):
        raise ValueError('Completed bounded native update required before export')
    native=Path(native_path)
    if native.is_symlink() or not native.is_file() or not 1024<=native.stat().st_size<=32*1024*1024:
        raise ValueError('Invalid native checkpoint')
    native_sha=hashlib.sha256(native.read_bytes()).hexdigest()
    if native_sha!=summary['final_checkpoint']['sha256']:raise ValueError('Native checkpoint changed')
    data=torch.load(native,map_location='cpu',weights_only=True)
    if data.get('iter')!=201 or not isinstance(data.get('experiment_config'),dict):raise ValueError('Native iteration/config missing')
    for key in ('actor_model_state_dict','critic_model_state_dict','actor_optimizer_state_dict','critic_optimizer_state_dict'):
        if not isinstance(data.get(key),dict) or not data[key]:raise ValueError('Native state missing')
    if '_aria_delivery' in data:raise ValueError('Never rewrap an existing export')
    provenance=dict(schema_version=1,run_path=run_path,source_iteration=200,source_sha256=SOURCE_SHA,
        final_iteration=201,completed_updates=1,transitions=768,num_envs=32,model_evaluations=0,
        regenerated_from_iteration200=True,old_task032_checkpoint_recovered=False,
        native_checkpoint_sha256=native_sha,native_save_reload_verified=True,
        actor_critic_optimizer_iteration_exact=True,policy_quality_evaluated=False)
    data['_aria_delivery']=provenance
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    destination=directory/name;partial=directory/(name+'.partial')
    if destination.exists() or partial.exists():raise FileExistsError('Never overwrite a checkpoint export')
    with partial.open('xb') as f:
        torch.save(data,f);f.flush();os.fsync(f.fileno())
    loaded=torch.load(partial,map_location='cpu',weights_only=True)
    if not equal(loaded,data):raise ValueError('Export serialization changed native state')
    # Native loader ignores the extra provenance key;prove model AND optimizer still restore exactly.
    algo.load(str(partial))
    for module,key in ((algo.actor,'actor_model_state_dict'),(algo.critic,'critic_model_state_dict'),
                       (algo.actor_optimizer,'actor_optimizer_state_dict'),(algo.critic_optimizer,'critic_optimizer_state_dict')):
        if not equal(module.state_dict(),data[key]):raise ValueError('Native export reload mismatch')
    if algo.current_learning_iteration!=201:raise ValueError('Native export reload iteration mismatch')
    if partial.stat().st_size>32*1024*1024:raise ValueError('Export exceeds byte bound')
    # A new hardlink is atomic,refuses an existing destination,and exposes only fully verified bytes.
    # Keep the .partial link as evidence;no deletion or replacement of any file.
    os.link(partial,destination)
    digest=hashlib.sha256(destination.read_bytes()).hexdigest()
    return dict(filename=name,path=str(destination),bytes=destination.stat().st_size,sha256=digest,
        iteration=201,native_save_reload_verified=True,actor_critic_optimizer_iteration_exact=True,
        provenance=provenance)

def verify(path,run_path,expected_sha=None):
    """Controller gate;metadata and native optimizer are checked before accepting any export."""
    import torch
    path=Path(path);filename(run_path)
    if path.is_symlink() or not path.is_file() or not 1024<=path.stat().st_size<=32*1024*1024:
        raise ValueError('Invalid downloaded checkpoint')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_sha is not None and digest!=expected_sha:raise ValueError('Checkpoint download hash mismatch')
    data=torch.load(path,map_location='cpu',weights_only=True)
    if not isinstance(data,dict) or data.get('iter')!=201 or not isinstance(data.get('experiment_config'),dict):
        raise ValueError('New native iteration201 required')
    p=data.get('_aria_delivery',{})
    required=dict(schema_version=1,run_path=run_path,source_iteration=200,source_sha256=SOURCE_SHA,
        final_iteration=201,completed_updates=1,transitions=768,num_envs=32,model_evaluations=0,
        regenerated_from_iteration200=True,old_task032_checkpoint_recovered=False,
        native_save_reload_verified=True,actor_critic_optimizer_iteration_exact=True,policy_quality_evaluated=False)
    if any(p.get(k)!=v for k,v in required.items()):raise ValueError('Export provenance differs from authorized run')
    if not isinstance(p.get('native_checkpoint_sha256'),str) or not re.fullmatch(r'[a-f0-9]{64}',p['native_checkpoint_sha256']):
        raise ValueError('Native checkpoint fingerprint missing')
    for key in ('actor_model_state_dict','critic_model_state_dict'):
        state=data.get(key)
        if not isinstance(state,dict) or not state:raise ValueError('Model state missing')
        if any(not torch.isfinite(v).all() for v in state.values() if torch.is_tensor(v) and (v.is_floating_point() or v.is_complex())):
            raise ValueError('Nonfinite model tensor')
    for key in ('actor_optimizer_state_dict','critic_optimizer_state_dict'):
        opt=data.get(key)
        if not isinstance(opt,dict) or not opt.get('state') or not opt.get('param_groups'):raise ValueError('Full optimizer missing')
        for state in opt['state'].values():
            if not {'step','exp_avg','exp_avg_sq'}<=set(state):raise ValueError('Optimizer moments missing')
            if any(not torch.isfinite(state[k]).all() for k in ('exp_avg','exp_avg_sq')):raise ValueError('Nonfinite optimizer moment')
            if float(state['step'])<=0:raise ValueError('Optimizer has no completed steps')
    return dict(sha256=digest,bytes=path.stat().st_size,iteration=201,optimizer_state_present=True,
        native_config_present=True,provenance=p,weights_and_moments_finite=True)
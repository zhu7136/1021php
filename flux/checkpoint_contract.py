"""Fail-fast native checkpoint save/load checks; no external checkpoint ingestion."""
from __future__ import annotations
import copy,hashlib
from pathlib import Path

def attach_metadata(algo,config,run_path=None):
    attach=getattr(algo,'attach_checkpoint_metadata',None)
    if not callable(attach):raise TypeError('Native attach_checkpoint_metadata required')
    attach(config,run_path)

def roundtrip(algo,path):
    import torch
    path=Path(path)
    if path.exists():raise FileExistsError('Never overwrite checkpoint validation output')
    before=copy.deepcopy(dict(actor=algo.actor.state_dict(),critic=algo.critic.state_dict(),
        actor_optimizer=algo.actor_optimizer.state_dict(),critic_optimizer=algo.critic_optimizer.state_dict(),
        iteration=algo.current_learning_iteration))
    algo.save(str(path))
    if not path.is_file() or path.stat().st_size<1024:raise RuntimeError('Native checkpoint missing or empty')
    data=torch.load(path,map_location='cpu',weights_only=True)
    if not isinstance(data.get('experiment_config'),dict):raise ValueError('Embedded experiment config missing')
    if data['iter']!=before['iteration']:raise ValueError('Saved iteration mismatch')
    algo.load(str(path))
    after=dict(actor=algo.actor.state_dict(),critic=algo.critic.state_dict(),
        actor_optimizer=algo.actor_optimizer.state_dict(),critic_optimizer=algo.critic_optimizer.state_dict(),
        iteration=algo.current_learning_iteration)
    def equal(a,b):
        if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a.detach().cpu(),b.detach().cpu())
        if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
        if isinstance(a,(list,tuple)):return isinstance(b,type(a)) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
        return a==b
    if not equal(before,after):raise ValueError('Native save/load changed actor, critic, optimizer, or iteration')
    return dict(path=path.name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        native_save_reload_verified=True,actor_critic_optimizer_iteration_exact=True,iteration=int(before['iteration']))
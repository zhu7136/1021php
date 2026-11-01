"""Fixed, explicitly authorized resume bounds; CPU-only helpers."""
import copy
import time

SOURCE_SHA='7828ba3698eb1607585559ff22f569cc94a3c6c44137473ef57df19e4fd52492'
SOURCE_BYTES=47769831
SOURCE_ITER=4
MAX_UPDATES=1000
TRAIN_SECONDS=6600
WORKER_SECONDS=7200

def require_equal(a,b):
    import torch
    if isinstance(a,torch.Tensor):
        if not isinstance(b,torch.Tensor) or not torch.equal(a.detach().cpu(),b.detach().cpu()):
            raise ValueError('Resume tensor mismatch')
    elif isinstance(a,dict):
        if not isinstance(b,dict) or set(a)!=set(b):raise ValueError('Resume keys mismatch')
        for k,v in a.items():require_equal(v,b[k])
    elif isinstance(a,(list,tuple)):
        if type(a)!=type(b) or len(a)!=len(b):raise ValueError('Resume sequence mismatch')
        for x,y in zip(a,b):require_equal(x,y)
    elif a!=b:raise ValueError('Resume scalar mismatch')

def verify_loaded(algo,saved):
    require_equal(saved['model_state_dict'],algo.policy.state_dict())
    require_equal(saved['optimizer_state_dict'],algo.optimizer.state_dict())
    require_equal(saved['iter'],algo.current_learning_iteration)
    require_equal(float(saved['learning_rate']),algo.learning_rate)
    require_equal(float(saved['ppo_coef']),algo.ppo_coef)
    require_equal(saved['teacher_loaded'],list(algo.policy.loaded_teachers))

class ResumeBudgetExpired(RuntimeError):pass

class ResumeBudget:
    def __init__(self,max_updates=MAX_UPDATES,seconds=TRAIN_SECONDS,clock=time.monotonic):
        if type(max_updates) is not int or not 1<=max_updates<=MAX_UPDATES or not 0<seconds<=TRAIN_SECONDS:
            raise ValueError('Resume exceeds authorized bounds')
        self.max_updates=max_updates;self.clock=clock;self.deadline=clock()+seconds
        self.completed=0;self.active=False
    def begin(self):
        if self.active or self.completed>=self.max_updates or self.clock()>=self.deadline:
            raise ResumeBudgetExpired('No additional iteration authorized')
        self.active=True
    def complete(self):
        if not self.active:raise RuntimeError('No active resume iteration')
        self.completed+=1;self.active=False

def next_iteration(saved):
    if type(saved.get('iter')) is not int or saved['iter']!=SOURCE_ITER:
        raise ValueError('Unexpected resume source iteration')
    return saved['iter']+1
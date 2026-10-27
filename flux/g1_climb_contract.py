"""Independent fresh climb teachers; 1800s TOTAL learning includes 20-update gate."""
import os
from pathlib import Path
from flux import g1_lowstep_contract as base
from flux.formal_contract import execute,replace_once

def kind():
    value=os.environ.get('PHP_G1_CLIMB_KIND')
    if value not in ('low','high'):raise ValueError('Explicit climb kind required')
    return value

def adapt(text):
    speed=kind();label=speed+'-climb-76'
    for a,b in [('g1_lowstep','g1_climb'),('G1Lowstep','G1Climb'),('motions-low-step','motions-'+label),
                ('low-speed-step',speed+'-speed-climb-76'),('Low-speed',speed.title()+'-speed'),
                ('Low-step',speed.title()+'-climb-76'),('low-step',label),('lowstep','climb'),
                ('files=113,motions=56','files=81,motions=40')]:
        text=text.replace(a,b)
    return text

def validate_scope(scope):
    if scope.get('dataset')!='motions-'+kind()+'-climb-76':raise ValueError('Wrong climb dataset')
    base.validate_scope({**scope,'dataset':'motions-low-step'})
    return scope

def budget_times(validation_deadline,gate):
    if not gate or gate.get('passed') is not True:return validation_deadline-30,validation_deadline
    started=gate['training_started_monotonic']
    return started+1680,started+1800

def teacher_source(root):return adapt(base.teacher_source(root))
def bootstrap_source(root):return adapt(base.bootstrap_source(root))
def worker_source(root):
    text=adapt(base.worker_source(root))
    return text.replace('validate20updates,then formal<=1800seconds','validate20updates, TOTAL learning<=1800seconds including gate')

def input_source(root):
    # Reuse verified downloader: bounded transfer-only retries, atomic publication.
    text=adapt(Path(root).joinpath('g1_lowstep_input.py').read_text())
    start=text.index("    h=hashlib.sha256();remaining=control['bytes']")
    end=text.index("    component=json.loads",start)
    text=text[:start]+"    from flux.verified_download import download\n    download(control['url'],archive,control['bytes'],control['sha256'],destination.parent/'data-download.json')\n"+text[end:]
    # Low climb repacking can exceed 60MiB: upper bound is still strict and metadata checked.
    text=replace_once(text,'<=60*1024**2','<=100*1024**2')
    return text

def algo_source(root):
    text=adapt(Path(root).joinpath('g1_lowstep_algo.py').read_text())
    start=text.index('def budget_times(');end=text.index('\nclass ',start)
    text=text[:start]+'from flux.g1_climb_contract import budget_times\n'+text[end:]
    text=replace_once(text,'        result=super().learn()',"        self._g1_training_started=time.monotonic()\n        result=super().learn()")
    text=replace_once(text,"'formal_started_monotonic':time.monotonic(),","'training_started_monotonic':self._g1_training_started,\n                    'formal_started_monotonic':time.monotonic(),")
    return text
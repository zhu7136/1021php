"""Exactly one explicitly selected fresh climb teacher; no checkpoint resume."""
import os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
flag='--climb-kind'
if sys.argv.count(flag)!=1:raise ValueError('Explicit climb kind required once')
pos=sys.argv.index(flag);value=sys.argv[pos+1]
if value not in ('low','high'):raise ValueError('Invalid climb kind')
os.environ['PHP_G1_CLIMB_KIND']=value
sys.argv=sys.argv[:pos]+sys.argv[pos+2:]
from flux.g1_climb_contract import execute,worker_source
execute(worker_source,globals())
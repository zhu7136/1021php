import os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
if sys.argv.count('--climb-kind')!=1:raise ValueError('Exactly one explicit climb kind required')
i=sys.argv.index('--climb-kind');kind=sys.argv[i+1]
if kind not in ('low','high'):raise ValueError('Invalid climb kind')
os.environ['PHP_G1_CLIMB_KIND']=kind
sys.argv=sys.argv[:i]+sys.argv[i+2:]
from flux.g1_climb_eval_contract import worker_source
exec(compile(worker_source(),__file__,'exec'),globals())
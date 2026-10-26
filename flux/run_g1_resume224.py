"""One authorized224 continuation, gate20+formal<=30min, no automatic retry."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flux.g1_resume224_contract import execute,worker_source
execute(worker_source,globals())
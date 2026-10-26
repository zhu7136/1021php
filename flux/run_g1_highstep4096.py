"""One authorized fresh high-speed teacher; no retries or old checkpoint load."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flux.g1_highstep_contract import execute,worker_source
execute(worker_source,globals())
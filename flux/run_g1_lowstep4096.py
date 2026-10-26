"""One authorized fresh G1 low-speed-step validation and 30-minute formal segment."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flux.g1_lowstep_contract import execute,worker_source
execute(worker_source,globals())
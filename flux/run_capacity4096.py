"""One authorized4096environment20PPOupdate capacity trial from task044."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flux.capacity_contract import execute,worker_source
execute(worker_source,globals())
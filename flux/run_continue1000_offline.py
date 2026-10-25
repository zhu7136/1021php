"""ONE bounded trial; Flux live logs, offline curves synced after retrieval."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flux.continue1000_offline_contract import execute,worker_source
execute(worker_source,globals())
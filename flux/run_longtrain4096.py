"""Single4096environment native teacher segment,worker<=3600seconds,retries0."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flux.longtrain_contract import execute,worker_source
execute(worker_source,globals())
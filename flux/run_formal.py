"""One authorized 1800-second Flux worker, verified task033 continuation, retries0."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flux.formal_contract import execute, worker_source
execute(worker_source,globals())
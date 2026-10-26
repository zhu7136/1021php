"""One authorized4096env20update fresh G1 GPU test; no formal stage or retries."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flux.g1_validation_contract import execute, worker_source
execute(worker_source, globals())
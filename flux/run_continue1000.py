"""Prepared45minute online continuation worker; no launch authority implied."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flux.continue1000_contract import execute,worker_source
execute(worker_source,globals())
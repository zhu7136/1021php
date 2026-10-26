"""One authorized276 continuation; no automatic retry or continuation."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flux.g1_resume276_contract import execute,worker_source
execute(worker_source,globals())
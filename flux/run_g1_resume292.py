"""Single native continuation; total learning cap60m, worker cap85m."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flux.g1_resume292_contract import execute,worker_source
execute(worker_source,globals())
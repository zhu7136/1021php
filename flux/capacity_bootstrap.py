"""Single30minute bounded4096capacity subprocess supervisor."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flux.capacity_contract import execute,bootstrap_source
execute(bootstrap_source,globals())
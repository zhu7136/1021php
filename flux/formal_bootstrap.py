"""Validated single-card formal subprocess supervisor; not a smoke entry."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flux.formal_contract import execute, bootstrap_source
execute(bootstrap_source,globals())
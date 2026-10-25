"""Bounded supervisor;1000updates is strict success criterion."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flux.continue1000_contract import execute,bootstrap_source
execute(bootstrap_source,globals())
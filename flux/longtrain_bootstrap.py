"""Bounded one-hour teacher supervisor; reserve final save/reload and result upload."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flux.longtrain_contract import execute,bootstrap_source
execute(bootstrap_source,globals())
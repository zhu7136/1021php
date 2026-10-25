"""Strict1000update supervisor, bounded45minutes including outer packaging."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flux.continue1000_offline_contract import execute,bootstrap_source
execute(bootstrap_source,globals())
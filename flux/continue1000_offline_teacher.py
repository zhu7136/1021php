"""Worker-selected native offline teacher with live Flux console."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flux.continue1000_offline_contract import execute,teacher_source
execute(teacher_source,globals())
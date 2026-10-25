"""4096environment,20update GPU capacity teacher; not full-scale long training."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flux.capacity_contract import execute,teacher_source
execute(teacher_source,globals())
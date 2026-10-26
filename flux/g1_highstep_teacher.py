from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flux.g1_highstep_contract import execute,teacher_source
execute(teacher_source,globals())
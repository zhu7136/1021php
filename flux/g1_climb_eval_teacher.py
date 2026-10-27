import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flux.g1_climb_eval_contract import teacher_source
exec(compile(teacher_source(),__file__,'exec'),globals())
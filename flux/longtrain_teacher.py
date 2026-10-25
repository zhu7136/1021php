"""4096native teacher,20000-update ceiling and500save interval under supervisor."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flux.longtrain_contract import execute,teacher_source
execute(teacher_source,globals())
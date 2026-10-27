from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flux.g1_climb_resume_contract import execute,bootstrap_source
execute(bootstrap_source,globals())
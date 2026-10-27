from pathlib import Path
from flux.g1_climb_resume_contract import algo_source
exec(compile(algo_source(Path(__file__).parent),__file__,'exec'),globals())
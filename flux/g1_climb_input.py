from pathlib import Path
from flux.g1_climb_contract import input_source
exec(compile(input_source(Path(__file__).parent),__file__,'exec'),globals())
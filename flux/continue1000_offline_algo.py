"""Native continuation; explicit offline metrics and relative500 local saves."""
from pathlib import Path
from flux.continue1000_offline_contract import algo_source
exec(compile(algo_source(Path(__file__).resolve().parent),__file__,'exec'),globals())
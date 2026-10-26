"""High-speed equivalent of the tested 20-update exact native reload gate."""
from pathlib import Path
from flux.g1_highstep_contract import adapt
exec(compile(adapt(Path(__file__).with_name('g1_lowstep_algo.py').read_text()),__file__,'exec'),globals())
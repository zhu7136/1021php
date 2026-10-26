"""Same strict extraction/hash validation, pinned official high-speed manifest."""
from pathlib import Path
from flux.g1_highstep_contract import adapt
exec(compile(adapt(Path(__file__).with_name('g1_lowstep_input.py').read_text()),__file__,'exec'),globals())
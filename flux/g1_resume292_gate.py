"""Strict 20-update reload gate, inside a 60-minute TOTAL learning budget."""
from pathlib import Path
from flux.g1_highstep_contract import adapt
text=adapt(Path(__file__).with_name('g1_lowstep_algo.py').read_text())
text=text.replace('G1HighstepPPO','G1Highstep60PPO').replace('1800','3600').replace('1680','3480')
text=text.replace('self._g1_peak=0;self._g1_gate=False','self._g1_peak=0;self._g1_gate=False\n        self._g1_training_started=time.monotonic()')
text=text.replace("'formal_started_monotonic':time.monotonic()","'formal_started_monotonic':time.monotonic(), 'training_started_monotonic':self._g1_training_started")
exec(compile(text,__file__,'exec'),globals())
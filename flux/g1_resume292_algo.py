"""Exact native high-step292 continuation, without optimizer/LR resets."""
from pathlib import Path
text=Path(__file__).with_name('g1_resume224_algo.py').read_text()
for old,new in [('g1_resume224','g1_resume292'),('G1Resume224','G1Resume292'),('224','292'),
                ('from flux.g1_lowstep_algo import G1LowstepPPO','from flux.g1_resume292_gate import G1Highstep60PPO'),
                ('(G1LowstepPPO)','(G1Highstep60PPO)')]:
    text=text.replace(old,new)
exec(compile(text,__file__,'exec'),globals())
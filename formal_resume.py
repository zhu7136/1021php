"""Exact native model/optimizer resume from the verified ot7i239c result."""
import hashlib,json,os
from pathlib import Path
import torch
from formal_algo import FormalPPO
from flux.teacher_algo import _require_same_state
SOURCE_SHA='aa543b193b8502142d03d4cbdaba0cb10a0909f2140c8c6a9e8d58b682e7cf1a'
START=128
MAX_UPDATES=100000

class ResumePPO(FormalPPO):
    def load(self,path):
        result=super().load(path)
        if Path(path).name=='source-checkpoint.pt':
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==SOURCE_SHA
            expected=torch.load(path,map_location='cpu',weights_only=True)
            for name,key in [('actor','actor_model_state_dict'),('critic','critic_model_state_dict'),('actor_optimizer','actor_optimizer_state_dict'),('critic_optimizer','critic_optimizer_state_dict')]:
                _require_same_state(expected[key],getattr(self,name).state_dict(),name)
            assert expected['iter']==self.current_learning_iteration==START
            assert self.actor_learning_rate==expected['actor_optimizer_state_dict']['param_groups'][0]['lr']
            assert self.critic_learning_rate==expected['critic_optimizer_state_dict']['param_groups'][0]['lr']
            self._resume_verified=True
            row={'native_new_process_load_verified':True,'source_checkpoint_sha256':SOURCE_SHA,
                 'source_run':'ot7i239c','source_iteration':START,'start_iteration':START,
                 'maximum_new_updates':MAX_UPDATES,'actor_critic_and_optimizers_exact':True,
                 'learning_rates_exact':True,'scope':'model and optimizer resume, not exact simulator/RNG replay'}
            with (Path(os.environ['PHP_FLUX_OUTPUT'])/'new_process_resume.json').open('x') as f:json.dump(row,f,indent=2)
        return result

    def learn(self):
        assert getattr(self,'_resume_verified',False) is True
        assert self.current_learning_iteration==START and self.config.num_learning_iterations==MAX_UPDATES
        return super().learn()
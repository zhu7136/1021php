"""Resume224 native state, then the unchanged G1 20-update gate and formal budget."""
import json,os
from pathlib import Path
import torch
from flux.g1_lowstep_algo import G1LowstepPPO
from flux.resume_algo import snapshot,require_loaded_state,require_updated,finite
from flux.teacher_algo import _require_same_state
from flux.g1_resume224_input import CHECKPOINT_SHA256,SOURCE_TASK,SOURCE_RUN,SOURCE_ITERATION,SOURCE_UPDATES,digest

class G1Resume224PPO(G1LowstepPPO):
    def learn(self):
        source=Path(os.environ['PHP_FLUX_RESUME_CHECKPOINT']);output=Path(os.environ['PHP_FLUX_OUTPUT'])
        if digest(source)!=CHECKPOINT_SHA256 or not self.config.load_optimizer or self.current_learning_iteration!=SOURCE_ITERATION:
            raise ValueError('224 native checkpoint/optimizer/iteration not restored')
        saved=torch.load(source,map_location='cpu',weights_only=True)
        before=snapshot(self);require_loaded_state(self,saved,before);finite(before)
        _require_same_state(saved['iter'],before['current_learning_iteration'],'source.iteration')
        for name,key in [('actor_learning_rate','actor_optimizer_state_dict'),('critic_learning_rate','critic_optimizer_state_dict')]:
            _require_same_state(saved[key]['param_groups'][0]['lr'],before[name],name)
        (output/'resume_loaded.json').write_text(json.dumps({
            'passed':True,'source_task':SOURCE_TASK,'source_run':SOURCE_RUN,'source_sha256':CHECKPOINT_SHA256,
            'source_iteration':SOURCE_ITERATION,'source_completed_updates':SOURCE_UPDATES,
            'native_loaded_state_matches_source':True,'checked_state':list(before),
            'load_optimizer':True,'fresh_training':False,'exact_environment_rng_resume_claimed':False}))
        print(json.dumps({'event':'g1_resume224_source_loaded_exactly','source_iteration':SOURCE_ITERATION,
                          'source_sha256':CHECKPOINT_SHA256,'load_optimizer':True}),flush=True)
        result=super().learn()
        after=snapshot(self);changes,steps=require_updated(before,after)
        updates=self._flux_completed_updates
        if self.current_learning_iteration!=SOURCE_ITERATION+updates-1:
            raise ValueError('Unexpected native continuation indexing')
        for row in steps.values():
            if row['min_step_increment']!=updates*20 or row['max_step_increment']!=updates*20:
                raise ValueError('Continuation optimizer steps inconsistent with native PPO updates')
        if digest(source)!=CHECKPOINT_SHA256:raise ValueError('224 source checkpoint changed')
        native=json.loads((output/'native_reload.json').read_text())
        if native.get('native_load_passed') is not True or native['completed_updates']!=updates:
            raise ValueError('Resumed final native reload missing')
        (output/'resume_result.json').write_text(json.dumps({
            'passed':True,'source_task':SOURCE_TASK,'source_run':SOURCE_RUN,'source_sha256':CHECKPOINT_SHA256,
            'source_iteration':SOURCE_ITERATION,'source_completed_updates':SOURCE_UPDATES,
            'new_completed_updates':updates,'cumulative_completed_updates':SOURCE_UPDATES+updates,
            'native_final_iteration':self.current_learning_iteration,'native_iteration_semantics_unchanged':True,
            'parameters_changed':changes,'optimizer_step_advancement':steps,'source_checkpoint_unchanged':True,
            'native_final_reload_passed':True,'fresh_training':False,'exact_environment_rng_resume_claimed':False}))
        print(json.dumps({'event':'g1_resume224_continuation_passed','new_updates':updates,
                          'cumulative_updates':SOURCE_UPDATES+updates}),flush=True)
        return result
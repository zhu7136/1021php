"""Hu-D04 three-teacher routing with a trainable privileged critic."""
import torch
from holosoma.agents.modules.student_teacher_modules import DepthStudentTeacherCritic
from .contract import routed_targets
from .masked_training import routing_validity

class StrictRoutedTeacherCritic(DepthStudentTeacherCritic):
    def __init__(self, *args, num_teacher_obs, routing_column, **kwargs):
        if num_teacher_obs != 450 or not 0 <= routing_column < 450 or kwargs.get('num_teachers') != 3:
            raise ValueError('Expected three Hu-D04 teachers with explicit routing')
        super().__init__(*args, num_teacher_obs=num_teacher_obs-1, **kwargs)
        self.routing_column = routing_column

    @torch.no_grad()
    def teacher_act(self, obs, teacher_idx=None):
        routing_validity(obs, self.routing_column)
        col = self.routing_column
        privileged = torch.cat((obs[:,:col], obs[:,col+1:]), dim=-1)
        labels, _ = routed_targets(self, privileged, obs[:,col].long())
        return labels
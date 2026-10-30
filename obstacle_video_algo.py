"""Student-only obstacle recording, using the guarded zero-update preview."""
import json,os
from pathlib import Path
from offline_video_algo import HuD04OfflineVideo

class HuD04ObstacleVideo(HuD04OfflineVideo):
    def learn(self,num_learning_iterations=None):
        request=json.loads(Path(os.environ['HUD04_VIDEO_REQUEST']).read_text())
        row=request['obstacle_case']
        if row['skill'] not in ['low-climb','high-climb'] or self.env.num_envs!=1:raise ValueError('Obstacle scope')
        super().learn(num_learning_iterations)
        out=Path(request['output']);path=out/'video-evidence.json'
        evidence=json.loads(path.read_text())
        if evidence['env0_route_counts']!={str(row['route']):evidence['policy_steps']}:raise ValueError('Wrong recorded obstacle route')
        evidence.update(skill=row['skill'],fixed_clip=row,aligned_terrain_preserved=True,
            camera_mode='fixed-wide-side-view',random_motion_switching=False)
        path.write_text(json.dumps(evidence,indent=2))

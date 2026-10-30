"""Scheduling-only evaluation variant of the existing fixed-clip command."""
from obstacle_video_command import HuD04FixedObstacleCommand
from hud04_tracking_cache_sync import install_cache_sync,native_refresh

class HuD04CacheSynchronizedCommand(HuD04FixedObstacleCommand):
    def setup(self):
        super().setup()
        install_cache_sync(self._env,native_refresh())
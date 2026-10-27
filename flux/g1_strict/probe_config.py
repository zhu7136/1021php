"""Configure a zero-PPO G1 lifecycle probe without constructing a logger or simulator."""
from dataclasses import replace
from holosoma.config_types.logger import DisabledLoggerConfig

def build_probe_config(cfg, registry_uri):
    return replace(
        cfg,
        training=replace(cfg.training, headless=True, num_envs=2,
                         checkpoint=None, teacher_checkpoint=None, wandb_path=None,
                         registry_name=registry_uri, export_onnx=False),
        logger=DisabledLoggerConfig(video=replace(cfg.logger.video, enabled=False)),
    )
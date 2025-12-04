# mmaction
from mmcv import Config
from mmcv.runner import load_checkpoint
from .mmaction.models import build_model
from .mmaction.utils import (build_ddp, build_dp, default_device,
                            register_module_hooks, setup_multi_processes)

def mmaction_model(config_path, checkpoint_path):
    cfg = Config.fromfile(config_path)
    # set multi-process settings
    setup_multi_processes(cfg)
    
    # build the model and load checkpoint
    model = build_model(
        cfg.model, train_cfg=None, test_cfg=None)
    load_checkpoint(model, checkpoint_path, map_location='cpu')
    return model
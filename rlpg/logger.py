import torch.utils.tensorboard as tensorboard
import wandb
import wandb.wandb_run as wwr

from typing import Dict, Any
import os
import json
import pathlib
import datetime
import torch
import pandas as pd
import dataclasses
from typing import List, Tuple
import pickle

class LoggingFS:
    def __init__(self, root_dir: str, model_prefix: str = None) -> None:
        self._initialized = False
        self._root_dir = root_dir
        self._meta_path = os.path.join(root_dir, "log_meta.json")
        self._model_prefix = model_prefix if model_prefix else "version"
        self._log_dir = None
        self._checkpoint_dir = None
    
    def _initialize (self):
        if self._initialized:
            return self._log_dir, self._checkpoint_dir

        pathlib.Path(self._root_dir).mkdir(parents=True, exist_ok=True)
        meta = dict(model_version = 0)
        if os.path.exists(self._meta_path):
            with open(self._meta_path, 'r') as f:
                meta = json.load(f)
                meta["model_version"] += 1
        with open(self._meta_path, 'w') as f:
            json.dump(meta, f)
        
        version_num = meta["model_version"]
        run_name = f"{self._model_prefix}_{version_num:>04}"

        self._log_dir = os.path.join(self._root_dir, run_name)

        self._checkpoint_dir = os.path.join(self._log_dir, "checkpoints")
        pathlib.Path(self._checkpoint_dir).mkdir(parents=True, exist_ok=True)

        self._initialized = True
        return self._log_dir, self._checkpoint_dir
    
    def log_dir (self):
        ld, _ = self._initialize()
        return ld
    
    def checkpoint_dir (self):
        _, cd = self._initialize()
        return cd

# Wrapper around tensorboard and wandb
class TrainLogger:
    def __init__(self) -> None:
        pass

    def log (self, data: Dict, step: int):
        pass

    def log_aggregate (self, name, data, step: int, include: List[str] = None):
        if not data:
            return

        prefix = name + "_"
        df = pd.DataFrame(data).describe()[0].to_dict()

        data_keys = [k for k in df.keys() if k != "count"]
        if include:
            data_keys = [k for k in data_keys if k in include]

        res = {prefix + k : df[k] for k in data_keys}
        return self.log(res, step)

    def config (self, config: Dict):
        pass


class WandbLogger (TrainLogger):
    def __init__(self, logger: wwr.Run, name: str = None) -> None:
        super().__init__()
        self._logger = logger

    def log (self, data: Dict, step: int):
        self._logger.log(data)

    def config (self, config: Dict):
        self._logger.config.update(config)
        return self._logger.config
    
class TensorboardLogger (TrainLogger):
    def __init__(self, logfs: LoggingFS) -> None:
        super().__init__()
        self._logfs = logfs
        self._logger = None
    
    def get_logger (self):
        if not self._logger:
            self._logger = tensorboard.SummaryWriter(self._logfs.log_dir())
        return self._logger
    
    def log (self, data: Dict, step: int):
        logger = self.get_logger()
        for k,v in data.items():
            logger.add_scalar(k, v, step)

    def config (self, config: Dict):
        logger = self.get_logger()
        logger.add_hparams({**config}, metric_dict={})
        return config

class ModelWriter:
    def __init__(self, logfs: LoggingFS) -> None:
        self._logging_fs = logfs
        self._format = "epoch={epoch}"
    
    def log (self, model: torch.nn.Module, params: Dict, epoch: int):
        # TODO: Support exporting the model to ONNX
        log_dir = self._logging_fs.checkpoint_dir()
        fname = self._format.format(epoch=epoch)
        chkpt_path = os.path.join(log_dir, fname + ".chkpt")
        pickle_path = os.path.join(log_dir, fname + ".pkl")

        torch.save({
            **params,
            "model_state_dict": model.state_dict()
        }, chkpt_path)


def make_tensorboard_logger (root_dir: str, prefix: str = None) -> Tuple[TrainLogger, ModelWriter]:
    lfs = LoggingFS(root_dir, prefix)
    return TensorboardLogger(lfs), ModelWriter(lfs)


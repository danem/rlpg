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

@dataclasses.dataclass
class LoggingFS:
    root_dir: str
    log_dir: str
    checkpoint_dir: str
    run_name: str

# TODO: This API is broken. Should be idempotent
# TODO: Should only create the directories etc when something is actually logged
def _make_log_fs (root_dir: str, model_prefix: str) -> LoggingFS:
    root_path = pathlib.Path(root_dir)
    root_path.mkdir(parents=True, exist_ok=True)

    model_prefix = model_prefix if model_prefix else "version"
    meta = dict(model_version = 0)
    meta_path = os.path.join(root_dir, "log_meta.json")

    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            meta["model_version"] += 1

    with open(meta_path, 'w') as f:
        json.dump(meta, f)
    
    version_num = meta["model_version"]
    run_name = f"{model_prefix}_{version_num:>04}"
    log_dir_path = root_path / run_name
    log_dir_path.mkdir(parents=True, exist_ok = True)

    checkpoint_dir = log_dir_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    return LoggingFS(
        root_dir=str(root_path),
        log_dir=str(log_dir_path),
        checkpoint_dir=str(checkpoint_dir),
        run_name=run_name
    )

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
    def __init__(self, logger: tensorboard.SummaryWriter) -> None:
        super().__init__()
        self._logger = logger
    
    def log (self, data: Dict, step: int):
        for k,v in data.items():
            self._logger.add_scalar(k, v, step)

    def config (self, config: Dict):
        self._logger.add_hparams({**config}, metric_dict={})
        return config

class ModelWriter:
    def __init__(self, log_dir) -> None:
        self._log_dir = log_dir
        self._format = "epoch={epoch}"
    
    def log (self, model: torch.nn.Module, params: Dict, epoch: int):
        # TODO: Support exporting the model to ONNX
        fname = self._format.format(epoch=epoch)
        chkpt_path = os.path.join(self._log_dir, fname + ".chkpt")
        pickle_path = os.path.join(self._log_dir, fname + ".pkl")

        torch.save({
            **params,
            "model_state_dict": model.state_dict()
        }, chkpt_path)

def make_logger (logger) -> TrainLogger:
    if isinstance(logger, wwr.Run):
        return WandbLogger(logger)
    else:
        return TensorboardLogger(logger)

def make_tensorboard_logger (root_dir: str, prefix: str = None) -> Tuple[TrainLogger, ModelWriter]:
    lfs = _make_log_fs(root_dir, prefix)
    twriter = tensorboard.SummaryWriter(lfs.log_dir)
    return make_logger(twriter), ModelWriter(lfs.checkpoint_dir)


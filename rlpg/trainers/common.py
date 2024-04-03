import rlpg.logger as rl_logger
import torch
import dataclasses

@dataclasses.dataclass(kw_only=True)
class TrainState:
    log_freq: int 
    logger: rl_logger.TrainLogger
    write_freq: int
    writer: rl_logger.ModelWriter
    device: str


    
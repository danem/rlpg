from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import dataclasses
import torch
import torchvision.transforms.functional as ttf
import gymnasium as gym
import PIL
from PIL import Image

import rlpg.rl_memory as rl_memory

@dataclasses.dataclass
class StepResult:
    observation: Any
    reward: float
    terminated: bool
    truncated: bool
    info: Dict = None

@dataclasses.dataclass
class RenderOptions:
    size: Tuple[int, int]
    color: bool = True
    crop_pos: Tuple[float,float] = (0,0)
    crop_size: Tuple[float, float] = (1,1)

    @staticmethod
    def crop_centered (crop_center: Tuple[float, float], crop_size: Tuple[float,float], out_size: Tuple[int, int], color: bool = True):
        hw, hh = crop_size[0]/2, crop_size[1]/2
        cp = (crop_center[0] - hw, crop_center[1] - hh)
        return RenderOptions(
            size=out_size,
            color=color,
            crop_pos=cp,
            crop_size=crop_size
        )

    @staticmethod
    def crop (crop_pos: Tuple[float, float], crop_size: Tuple[float,float], out_size: Tuple[int, int], color: bool = True):
        return RenderOptions(
            size=out_size,
            color=color,
            crop_pos=crop_pos,
            crop_size=crop_size
        )


def apply_renderopts (img: torch.Tensor, ropts: RenderOptions):
    if not isinstance(img, Image.Image):
        img = ttf.to_pil_image(img)

    if not ropts.color:
        img = ttf.to_grayscale(img)

    cw, ch = int(ropts.crop_size[0] * img.width), int(ropts.crop_size[1] * img.height)
    cx, cy = int(ropts.crop_pos[0] * img.width), int(ropts.crop_pos[1] * img.height)
    cropped = ttf.resized_crop(img, cy, cx, ch, cw, ropts.size)
    return cropped


class RewardEvaluator:
    """
    Score the current state of the environment
    """
    def __init__(self, env: gym.Env, **kwargs) -> None:
        self._env = env
        self._total_rewards = 0
        self._reward_multiplier = kwargs.get("reward_multiplier", 1)
    
    def reset (self):
        self._total_rewards = 0
    
    def total_rewards (self):
        return self._total_rewards

    def score (self, alive: bool, reward: int, episode: float, step: int, mem: torch.Tensor) -> float:
        return reward * self._reward_multiplier

    def __call__(self, alive: bool, reward: int, episode: float, step: int, mem: torch.Tensor) -> float:
        score = self.score(alive, reward, episode, step, mem)
        self._total_rewards += score
        return score









import rlpg.envs.types as rl_envs
import rlpg.utils as rl_utils

import numpy as np
import gymnasium as gym
import torch
import torchvision.transforms.functional as ttf
import dataclasses
from typing import Tuple


class RewardEvaluator (rl_envs.RewardEvaluator):
    pass

class SkipWrapper (gym.Wrapper):
    def reset (self, *args, **kwargs):
        self.env.reset(*args, **kwargs)
        self.unwrapped.t = 1
        obs, _, _, _, info = self.env.step(np.zeros(self.env.action_space.shape))
        return obs, info

class VAEWrapper (gym.ObservationWrapper):
    def __init__ (self, env, renderopts: rl_envs.RenderOptions, encoder: torch.nn.Module):
        super().__init__(env)
        rl_utils.freeze_module(encoder)
        self._ropts = renderopts
        self._encoder = encoder
        self._device, = list(set(p.device for p in encoder.parameters()))
    
    def observation (self, observation):
        img = rl_envs.apply_renderopts(observation, self._ropts)
        x = ttf.pil_to_tensor(img).to(self._device).float() / 255
        x = x.unsqueeze(0)
        x, kl = self._encoder(x)
        return x

class AgressiveDriving (gym.ActionWrapper):
    def __init__(self, env: gym.Env, multiplier: float):
        super().__init__(env)
        self._multiplier = multiplier

    def action(self, action):
        action[2] = 0
        action[1] *= self._multiplier
        return action

def _default_render_opts () -> rl_envs.RenderOptions:
    return rl_envs.RenderOptions((50,50), crop_size=(1,0.87))


@dataclasses.dataclass
class CarHParams:
    skip_start: bool = True
    randomize_colors: bool = False
    color: bool = True
    include_bar: bool = False
    vae_generation_drive_aggression: float = 5
    render_opts: rl_envs.RenderOptions = dataclasses.field(default_factory= _default_render_opts)
    device: str = "cpu"

def car_basic (
    hparams: CarHParams,
    items: int = 1
):
    env = gym.make("CarRacing-v2", render_mode="rgb_array", domain_randomize=hparams.randomize_colors)

    if hparams.skip_start:
        env = SkipWrapper(env)
    evaluator = RewardEvaluator(env)

    return env, evaluator

def car_vae_observer (
    encoder: torch.nn.Module,
    hparams: CarHParams,
    items: int = 1,
):
    encoder = encoder.to(hparams.device)
    rl_utils.freeze_module(encoder)
    env, evaluator = car_basic(hparams, items=items)
    env = VAEWrapper(env, renderopts=hparams.render_opts, encoder=encoder)
    return env, evaluator

def car_vae_generator (
    hparams: CarHParams,
    items: int = 1,
):
    env, evaluator = car_basic(hparams, items=items)
    env = AgressiveDriving(env, multiplier=hparams.vae_generation_drive_aggression)
    return env, evaluator



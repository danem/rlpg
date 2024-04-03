import rlpg.envs.types as rl_envs
import rlpg.utils as rl_utils

import numpy as np
import gymnasium as gym
import torch
import torchvision.transforms.functional as ttf


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
        self._ropts = renderopts
        self._encoder = encoder
        self._device, = list(set(p.device for p in encoder.parameters()))
    
    def observation (self, observation):
        img = rl_envs.apply_renderopts(observation, self._ropts)
        x = ttf.from_pil_image(img).to(self._device)
        return self._encoder(x)

def get_hparams (
    **kwargs
):
    params = {
        "skip_start": True
    }
    params = rl_utils.merge_dicts(params, kwargs)
    return rl_utils.DotDict(params)

def car_basic (
    randomize_colors: bool = False,
    skip_start: bool = True,
    **kwargs
):
    hparams = get_hparams(
        skip_start = skip_start,
        randomize_colors = randomize_colors
    )
    env = gym.make("CarRacing-v2", render_mode="rgb_array", domain_randomize=randomize_colors)
    if skip_start:
        env = SkipWrapper(env)
    evaluator = RewardEvaluator(env, **kwargs)
    return env, evaluator, hparams

def car_vae (
    encoder: torch.nn.Module,
    render_options: rl_envs.RenderOptions,
    *args,
    **kwargs
):
    rl_utils.freeze_module(encoder)
    env, evaluator, hparams = car_basic(*args, **kwargs)
    env = VAEWrapper(env, renderopts=render_options, encoder=encoder)
    return env, evaluator, hparams



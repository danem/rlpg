import rlpg.rl_memory as rl_memory
import rlpg.envs.types as rl_envs
import rlpg.utils as rl_utils
import rlpg.logger as rl_logger
import rlpg.trainers.common as rl_trainers
import rlpg.models.actor_critic as rl_ac

import gymnasium as gym
import dataclasses
import random
import torch
import torchvision.transforms.functional as ttf
import wandb
import tqdm
import itertools
import pandas as pd
import itertools
import numpy as np
from typing import Tuple, Callable
import collections
import functools

class PPOMemory (rl_memory.CircularBuffer):
    def __init__(self, size: int, observation_shape, action_shape) -> None:
        super().__init__(size)
        # Benchmarking tells me writing to numpy array is much faster than a tensor,
        # and lists are even faster still.
        observation_shape = observation_shape if isinstance(observation_shape, list) else [observation_shape]
        action_shape = action_shape if isinstance(action_shape, list) else [action_shape]
        self.states = np.zeros([size] + observation_shape, dtype=np.float32)
        self.actions = np.zeros([size] + action_shape, dtype=np.float32)
        self.log_probs = np.zeros([size] + action_shape, dtype=np.float32)
        self.q_vals = np.zeros((size,1), dtype=np.float32)
        self.rewards = np.zeros((size,1), dtype=np.float32)
    
    def push (self, state,  reward, action, log_probs, q_val):
        idx = self.get_push_idx()
        self.states[idx] = state
        self.actions[idx] = action
        self.log_probs[idx] = log_probs
        self.q_vals[idx] = q_val
        self.rewards[idx] = reward
    
    def to (self, device):
        return self

@dataclasses.dataclass
class OptimInfo:
    approx_kl: float
    entropy: float
    policy_loss: float
    critic_loss: float

@dataclasses.dataclass
class PPOHParams:
    max_episode_steps: int = 300
    discount_factor: float = 1
    objective_constraint: float = 1
    kl_target: float = 0.01
    gae_gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_clip_value: float = 0.2
    actor_optim_iters: int = 80
    critic_optim_iters: int = 80
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    memory_size: int = 10_000
    continuous: bool = True
    device: str = "cpu"

@dataclasses.dataclass
class PPOTrainState(rl_trainers.TrainState):
    env: gym.Env
    evaluator: rl_envs.RewardEvaluator

    actor: rl_ac.ActorModel
    critic: rl_ac.CriticModel
    actor_optim: torch.optim.Optimizer
    critic_optim: torch.optim.Optimizer

    memory: PPOMemory
    hparams: PPOHParams

def _make_train_state (
    env: gym.Env,
    evaluator: rl_envs.RewardEvaluator,
    actor_model: rl_ac.ActorModel,
    critic_model: rl_ac.CriticModel,
    observation_size,
    action_space,
    hparams: PPOHParams,
    log_dir: str = None,
    log_freq = 10,
    write_freq = 100
) -> PPOTrainState:
    actor_model.to(hparams.device)
    critic_model.to(hparams.device)
    aoptim = torch.optim.Adam(actor_model.parameters(), lr=hparams.actor_lr)
    coptim = torch.optim.Adam(critic_model.parameters(), lr=hparams.critic_lr)

    if hparams.continuous:
        memory = PPOMemory(hparams.memory_size, observation_size, action_space).to(hparams.device)
    else:
        memory = PPOMemory(hparams.memory_size, observation_size, 1).to(hparams.device)
        memory.actions = memory.actions.astype(np.int64)


    logger = None
    writer = None
    if log_dir:
        logger, writer = rl_logger.make_tensorboard_logger(log_dir)

    return PPOTrainState(
        env = env,
        evaluator=evaluator,
        actor=actor_model,
        critic=critic_model,
        actor_optim=aoptim,
        critic_optim=coptim,
        log_freq=log_freq,
        logger=logger,
        writer=writer,
        write_freq=write_freq,
        memory=memory,
        hparams=hparams,
        device=hparams.device
    )

@functools.wraps(_make_train_state)
def make_train_state_continuous (*args, hparams: PPOHParams = None, **kwargs) -> PPOTrainState:
    hparams.continuous = True
    return _make_train_state(*args, hparams=hparams, **kwargs)

@functools.wraps(_make_train_state)
def make_train_state_discrete (*args, hparams: PPOHParams = None, **kwargs) -> PPOTrainState:
    hparams.continuous = False
    return _make_train_state(*args, hparams=hparams, **kwargs)


def sample_action (tstate: PPOTrainState, state: torch.Tensor):
    # TODO: This might be a bug.... Not 100% sure...
    with torch.no_grad():
        params = tstate.actor(state)
        action, log_prob = tstate.actor.sample(params)
        qval = tstate.critic(state)
        action = action.squeeze(0)
        return action, log_prob, qval


def calculate_gaes (rewards: np.array, values: np.array, terminated: bool,  gae_lambda: float, gae_gamma: float):
    if terminated:
        rewards = np.append(rewards, 0)
        values = np.append(values, 0)

    deltas = rewards[:-1] + gae_gamma * values[1:] - values[:-1]
    res = rl_utils.discount_cumsum(rewards, gae_gamma)[:-1]
    advs = rl_utils.discount_cumsum(deltas, gae_gamma * gae_lambda)
    adv_mean = np.mean(advs)
    adv_std = np.std(advs)
    advs = (advs - adv_mean) / adv_std
    return res, advs

def optimize_models (tstate: PPOTrainState, steps: int) -> OptimInfo:
    traj_idxs = tstate.memory.tail(steps)

    states = torch.as_tensor(tstate.memory.states[traj_idxs], device=tstate.device)
    actions = torch.as_tensor(tstate.memory.actions[traj_idxs], device=tstate.device)
    old_log_probs = torch.as_tensor(tstate.memory.log_probs[traj_idxs], device=tstate.device)
    values = tstate.memory.q_vals[traj_idxs]
    rewards = tstate.memory.rewards[traj_idxs]

    old_qvals, advs = calculate_gaes(rewards, values, True, gae_lambda=tstate.hparams.gae_lambda, gae_gamma=tstate.hparams.gae_gamma)
    old_qvals = torch.as_tensor(old_qvals.copy(), device=tstate.device)
    advs = torch.as_tensor(advs, device=tstate.device)

    policy_losses = []
    critic_losses = []
    kl_means = []
    entropies = []

    for _ in range(tstate.hparams.actor_optim_iters):
        tstate.actor_optim.zero_grad(set_to_none=True)

        new_logits = tstate.actor(states)
        new_dist = tstate.actor.distribution(new_logits)
        new_log_probs = new_dist.log_prob(actions)

        policy_ratio = torch.exp(new_log_probs - old_log_probs)
        clipped_ratio = policy_ratio.clamp(
            1 - tstate.hparams.ppo_clip_value, 1 + tstate.hparams.ppo_clip_value
        )

        clipped_loss = clipped_ratio * advs
        full_loss = policy_ratio * advs
        policy_loss = -torch.min(full_loss, clipped_loss).mean()

        policy_loss.backward()
        tstate.actor_optim.step()

        policy_losses.append(policy_loss.item())

        kl_div = (old_log_probs - new_log_probs).mean().item()
        kl_means.append(kl_div)
        entropies.append(new_dist.entropy().mean().item())
        if kl_div >= tstate.hparams.kl_target:
            break
    
    for _ in range(tstate.hparams.critic_optim_iters):
        tstate.critic_optim.zero_grad(set_to_none=True)
        qvals = tstate.critic(states)
        loss = (qvals - old_qvals) ** 2
        loss = loss.mean()

        loss.backward()
        tstate.critic_optim.step()
        critic_losses.append(loss.item())

    return OptimInfo(
        approx_kl= np.mean(kl_means),
        entropy=np.mean(entropies),
        policy_loss=np.mean(policy_losses),
        critic_loss=np.mean(critic_losses)
    )


def generate_episode (
    tstate: PPOTrainState,
    episodes: int = 1,
    max_steps: int = 1000
):
    total_steps = 0
    total_rewards = []
    for ep in range(episodes):
        state, _ = tstate.env.reset()
        tstate.evaluator.reset()

        for step in itertools.count():
            state = torch.tensor(state).to(tstate.device)
            action, logprobs, qvals = sample_action(tstate, state)
            action_cpu = action.detach().cpu().numpy()
            new_state, reward, terminated, truncated, _ = tstate.env.step(action_cpu)
            alive = not (terminated or truncated)

            reward = tstate.evaluator(alive, reward, 0, step, tstate.memory)

            tstate.memory.push(state, reward, action, logprobs, qvals)
            if not alive or (step > max_steps):
                break
            state = new_state# torch.tensor(new_state).to(tstate.device)
            total_steps += 1
        total_rewards.append(tstate.evaluator.total_rewards())

    return np.mean(total_rewards), total_steps

def train (
    tstate: PPOTrainState,
    episodes: int
):
    stats = collections.defaultdict(list)
    for ep in tqdm.tqdm(range(episodes)):
        reward, steps = generate_episode(tstate, max_steps=tstate.hparams.max_episode_steps)
        stats["episode_durations"].append(steps)
        stats["episode_scores"].append(reward)

        optim_info = optimize_models(tstate, steps)

        stats["actor_losses"].append(optim_info.policy_loss)
        stats["critic_losses"].append(optim_info.critic_loss)
        stats["kl_divergence"].append(optim_info.approx_kl)
        stats["entropy"].append(optim_info.entropy)
        
        tstate.memory.clear()
        
        if (ep+1) % tstate.log_freq == 0 and tstate.logger is not None:
            tstate.logger.log_aggregate("duration", stats["episode_durations"], ep)
            tstate.logger.log_aggregate("score", stats["episode_scores"], ep)
            tstate.logger.log_aggregate("actor_loss", stats["actor_losses"], ep, include=["mean"])
            tstate.logger.log_aggregate("critic_loss", stats["critic_losses"], ep, include=["mean", "std"])
            tstate.logger.log_aggregate("kl_divergence", stats["kl_divergence"], ep, include=["mean"])
            tstate.logger.log_aggregate("entropy", stats["entropy"], ep, include=["mean"])
            stats.clear()
        
        if (ep + 1) % tstate.write_freq == 0 and tstate.writer is not None:
            hparams = dataclasses.asdict(tstate.hparams)
            tstate.writer.log(tstate.actor, hparams, ep)
        

def evaluate (tstate: PPOTrainState, max_steps = 500):
    tstate.actor.eval()
    imgs = []
    dist_imgs = []

    state, _ = tstate.env.reset()
    state = torch.from_numpy(state).to(tstate.device)
    tstate.evaluator.reset()
    memory = torch.zeros((max_steps,1)).to(tstate.device)

    with torch.no_grad():
        for step in itertools.count():
            # action, _, _ = sample_action(tstate, state)
            params = tstate.actor(state)
            dist = tstate.actor.distribution(params)
            action = dist.sample().item()
            # action = action.argmax().view(1).long().item()
            # action = tstate.env.action_space.sample()
            new_state, reward, terminated, _, _ = tstate.env.step(action)

            reward = tstate.evaluator(not terminated, reward, 0, step, memory)

            state = torch.from_numpy(new_state).to(tstate.device)
            imgs.append(ttf.to_pil_image(tstate.env.render()))

            if terminated or step > max_steps:
                break

    return imgs


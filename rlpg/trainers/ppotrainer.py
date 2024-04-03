import rlpg.rl_memory as rl_memory
import rlpg.envs.types as rl_envs
import rlpg.utils as rl_utils
import rlpg.logger as rl_logger
import rlpg.trainers.common as rl_trainers

import gymnasium as gym
import dataclasses
import random
import torch
import wandb
import tqdm
import itertools
import pandas as pd
import itertools
import numpy as np
from typing import Tuple
import collections

class PPOMemory (rl_memory.CircularBuffer):
    def __init__(self, size: int, observation_shape, action_shape) -> None:
        super().__init__(size)
        self.states = torch.zeros([size] + observation_shape)
        self.actions = torch.zeros([size] + action_shape)
        self.log_probs = torch.zeros([size] + action_shape)
        self.q_vals = torch.zeros((size,1))
        self.rewards = torch.zeros((size,1))
    
    def push (self, state,  reward, action, log_probs, q_val):
        idx = self.get_push_idx()
        self.states[idx] = state
        self.actions[idx] = action
        self.log_probs[idx] = log_probs
        self.q_vals[idx] = q_val
        self.rewards[idx] = reward
    
    def to (self, device):
        self.states.to(device)
        self.rewards.to(device)
        self.actions.to(device)
        self.log_probs.to(device)
        self.q_vals.to(device)
        return self


@dataclasses.dataclass
class PPOTrainState(rl_trainers.TrainState):
    env: gym.Env
    evaluator: rl_envs.RewardEvaluator

    actor: torch.nn.Module
    critic: torch.nn.Module
    actor_optim: torch.optim.Optimizer
    critic_optim: torch.optim.Optimizer

    memory: PPOMemory
    hparams: rl_utils.DotDict

def get_hparams (**kwargs) -> rl_utils.DotDict:
    opts = rl_utils.merge_dicts(
        kwargs,
        {
            "max_episode_steps": 300,
            "discount_factor": 1,
            "objective_constraint": 1,
            "kl_penalty": 1,
            "kl_target": 0.01,
            "ppo_clip_value": 0.2,
            "actor_optim_iters": 80,
            "critic_optim_iters": 80,
            "actor_lr": 1e-4,
            "critic_lr": 1e-4,
            "memory_size": 10_000
        }
    )
    return rl_utils.DotDict(opts)

def make_train_state_continuous (
    env: gym.Env,
    evaluator: rl_envs.RewardEvaluator,
    actor_model: torch.nn.Module,
    critic_model: torch.nn.Module,
    actor_lr,
    critic_lr,
    observation_size,
    action_space,
    memory_size: int = 10_000,
    device: str = "cpu",
    log_freq = 10,
    log_dir: str = None

) -> PPOTrainState:
    hparams = get_hparams(
        device = device, 
        actor_lr = actor_lr,
        critic_lr = critic_lr,
        memory_size = memory_size
    ) 
    actor_model.to(device)
    critic_model.to(device)
    aoptim = torch.optim.Adam(actor_model.parameters(), lr=hparams.actor_lr)
    coptim = torch.optim.Adam(critic_model.parameters(), lr=hparams.critic_lr)
    memory = PPOMemory(memory_size, observation_size, action_space)
    logger = None
    if log_dir:
        logger = rl_logger.make_tensorboard_logger(log_dir)
        logger.config(hparams)

    return PPOTrainState(
        env = env,
        evaluator=evaluator,
        actor=actor_model,
        critic=critic_model,
        actor_optim=aoptim,
        critic_optim=coptim,
        log_freq=log_freq,
        logger=logger,
        memory=memory,
        hparams=hparams,
        device=device
    )


def calculate_gaes (rewards, values, gamma = 0.99, decay = 0.97):
    # next_values = values.roll(-1)
    # values = torch.clone(values)
    # values[-1] = 0

    # deltas = rewards + gamma * next_values - values
    next_values = torch.concatenate([values.squeeze()[1:], torch.tensor([0])])
    deltas = [rew.item() + gamma * nxt.item() - val.item() for rew, val, nxt in zip(rewards, values.squeeze(), next_values)]
    gaes = [deltas[-1]]
    for i in reversed(range(len(deltas)-1)):
        gaes.append(deltas[i] + decay * gamma * gaes[-1])
    return torch.tensor(gaes).flip(0)


def optimize_actor (tstate: PPOTrainState, steps: int):
    traj_idxs = tstate.memory.tail(steps)

    states = tstate.memory.states[traj_idxs]
    actions = tstate.memory.actions[traj_idxs]
    old_log_probs = tstate.memory.log_probs[traj_idxs]
    values = tstate.memory.q_vals[traj_idxs]
    rewards = tstate.memory.rewards[traj_idxs]
    
    gaes = calculate_gaes(rewards, values).to(tstate.device).unsqueeze(1)

    for _ in range(tstate.hparams.actor_optim_iters):
        tstate.actor_optim.zero_grad()

        new_logits = tstate.actor(states)
        new_dist = tstate.actor.distribution(new_logits)
        new_log_probs = new_dist.log_prob(actions)

        # policy_ratio = torch.exp(new_log_probs - old_log_probs)
        policy_ratio = new_log_probs / old_log_probs
        clipped_ratio = policy_ratio.clamp(
            1 - tstate.hparams.ppo_clip_value, 1 + tstate.hparams.ppo_clip_value
        )

        clipped_loss = clipped_ratio * gaes
        full_loss = policy_ratio * gaes
        policy_loss = -torch.min(full_loss, clipped_loss).mean()

        policy_loss.backward()
        tstate.actor_optim.step()

        kl_div = (old_log_probs - new_log_probs).mean()
        if kl_div >= tstate.hparams.kl_target:
            break
    return policy_loss

def optimize_critic (tstate: PPOTrainState, steps: int):
    traj_idxs = tstate.memory.tail(steps)
    states = tstate.memory.states[traj_idxs]
    actions = tstate.action_sampler.action(tstate.memory.actions[traj_idxs])
    old_cvals = tstate.action_sampler.cval(tstate.memory.actions[traj_idxs])

    for _ in range(tstate.hparams.critic_optim_iters):
        tstate.critic_optim.zero_grad()

        cvals = tstate.critic(states, actions)

        loss = (cvals - old_cvals) ** 2
        loss = loss.mean()

        loss.backward()
        tstate.critic_optim.step()
    return loss


def generate_episode (
    env_factory: rl_envs.EnvFactory, 
    observer: rl_envs.EnvObserver,
    tstate: PPOTrainState,
    episodes: int = 1,
    max_steps: int = 1000
):
    total_steps = 0
    total_rewards = []
    for ep in range(episodes):
        env, evaluator = env_factory()
        state = observer(env).to(tstate.device)
        next_state_placeholder = torch.ones_like(state)
        for step in itertools.count():
            state = observer(env).to(tstate.device)
            action = tstate.action_sampler.sample(env, state).to(tstate.device)
            act = action[:,:3].squeeze().tolist() # TODO: For some reason I need to make this a list...
            res = env.step(act)
            alive = not (res.terminated or res.truncated)

            reward = torch.tensor([evaluator(
                alive, res.reward,
                0, step, tstate.memory
            )]).item()

            tstate.memory.push(state, action, next_state_placeholder, reward)
            if not alive or (step > max_steps):
                break
            total_steps += 1
        total_rewards.append(evaluator.total_rewards())

    return np.mean(total_rewards), total_steps

def train (
    tstate: PPOTrainState,
    episodes: int
):
    stats = collections.defaultdict(list)
    for ep in tqdm.tqdm(range(episodes)):
        reward, steps = generate_episode(tstate.env, tstate, max_steps=tstate.hparams.max_episode_steps)
        stats["episode_durations"].append(steps)
        stats["episode_scores"].append(reward)

        actor_loss = optimize_actor(tstate, steps).item()
        critic_loss = optimize_critic(tstate, steps).item()

        if actor_loss is not None:
            stats["actor_losses"].append(actor_loss)
        if critic_loss is not None:
            stats["critic_losses"].append(critic_loss)
        
        tstate.memory.clear()
        
        if (ep+1) % tstate.log_freq == 0 and tstate.logger is not None:
            tstate.logger.log_aggregate("duration", stats["episode_durations"], ep)
            tstate.logger.log_aggregate("score", stats["episode_scores"], ep)
            tstate.logger.log_aggregate("actor_loss", stats["actor_losses"], ep, include=["mean"])
            tstate.logger.log_aggregate("critic_loss", stats["critic_losses"], ep, include=["mean", "std"])
            stats.clear()
        
        if (ep + 1) % tstate.write_freq == 0 and tstate.writer is not None:
            pass
            # tstate.writer.log()
        

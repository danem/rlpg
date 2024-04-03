import rlpg.rl_memory as rl_memory
import rlpg.trainers.common as rl_trainers
import rlpg.utils as rl_utils
import rlpg.logger as rl_loggers
import rlpg.envs.types as rl_envs

import dataclasses
import torch
import torchvision.transforms.functional as ttf
import itertools
import tqdm
import collections
import gymnasium as gym
import random
from typing import Dict, List, Sequence
import json
import numpy as np

class QMemory (rl_memory.CircularBuffer):
    def __init__ (self, size, observation_shape: Sequence[int], action_shape: Sequence[int] = None, action_dtype: type = torch.float32):
        super().__init__(size)

        action_shape = list(action_shape) if action_shape else [1]
        observation_shape = list(observation_shape)

        self.states = torch.zeros([size] + observation_shape)
        self.next_states = torch.zeros([size] + observation_shape)
        self.non_final_mask = torch.zeros(size).bool()
        self.actions = torch.zeros([size] + action_shape, dtype=action_dtype)
        self.rewards = torch.zeros([size, 1]).float()
        self._device = "cpu"
    
    def push (self, state, action, next_state, reward):
        idx = self.get_push_idx()

        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        if next_state is not None:
            self.next_states[idx] = next_state

        self.non_final_mask[idx] = next_state is not None
        
    def to (self, dev):
        self.states = self.states.to(dev)
        self.next_states = self.next_states.to(dev)
        self.non_final_mask = self.non_final_mask.to(dev)
        self.actions = self.actions.to(dev)
        self.rewards = self.rewards.to(dev)
        self._device = dev
        return self

@dataclasses.dataclass(kw_only=True)
class QTrainState(rl_trainers.TrainState):
    env: gym.Env
    evaluator: rl_envs.RewardEvaluator

    policy: torch.nn.Module
    target: torch.nn.Module
    optim: torch.optim.Optimizer

    memory: QMemory
    hparams: rl_utils.DotDict

    search_scheduler: rl_utils.EpsilonScheduler
    search_steps: int = 0

    device: str

def get_hparams (**kwargs) -> rl_utils.DotDict:
    params = {
        "batch_size": 128,
        "bellman_gamma": 0.99,
        "bellman_tau": 0.005,
        "target_update_freq": 3,
        "grad_clip_value": 100,
        "search_start": 0.9,
        "search_stop": 0.1,
        "search_decay": 1000,
        "memory_size": 10_000,
        "lr": 1e-4
    }
    params = rl_utils.merge_dicts(params, kwargs)
    return rl_utils.DotDict(params)


def make_train_state (
    env: gym.Env,
    evaluator: rl_envs.RewardEvaluator,
    policy: torch.nn.Module,
    target: torch.nn.Module,
    log_dir: str = None,
    log_frequency: int = 5,
    write_frequency: int = 1000, 
    device: str = None,
    hparams: Dict = None
):
    # TODO: Rework how hyperparameters work
    hparams["device"] = device if device else hparams.get("device", "cpu")
    hparams = get_hparams(**hparams)

    logger = None
    writer = None
    if log_dir:
        logger, writer = rl_loggers.make_tensorboard_logger(log_dir)
        logger.config(hparams)

    policy.to(device)
    target.to(device)
    memory = QMemory(hparams.memory_size, env.observation_space.shape, env.action_space.shape, action_dtype=torch.int64).to(device)
    optim = torch.optim.Adam(policy.parameters(), lr = hparams.lr)
    return QTrainState(
        policy=policy,
        target= target, 
        optim = optim,
        memory= memory,
        hparams=hparams,
        search_scheduler=rl_utils.EpsilonScheduler(hparams.search_start, hparams.search_stop, hparams.search_decay),
        env=env,
        evaluator=evaluator,
        device=device,
        log_freq=log_frequency,
        logger=logger,
        writer = writer,
        write_freq=write_frequency
    )


def sample_action (env: gym.Env, state: torch.Tensor, tstate: QTrainState):
    sample = random.random()
    thresh = tstate.search_scheduler(tstate.search_steps)
    tstate.search_steps += 1

    if sample > thresh:
        res = tstate.policy(state.unsqueeze(0))
        res = res.argmax().view(1).long()
        return res.item()
    else:
        sample = env.action_space.sample()
        return sample

def optimize (tstate: QTrainState):
    if len(tstate.memory) < tstate.hparams.batch_size:
        return

    idxs = tstate.memory.sample_idxs(tstate.hparams.batch_size)
    states  = tstate.memory.states[idxs]
    actions = tstate.memory.actions[idxs]
    rewards = tstate.memory.rewards[idxs]
    non_final_mask = tstate.memory.non_final_mask[idxs]
    non_final_states = tstate.memory.next_states[idxs][non_final_mask]

    # Calculate predictions from old states with our new model
    # and grab the prediction that corresponds to the action actually taken
    state_action_values = tstate.policy(states).gather(1, actions)
    
    # Get the next state values. Set terminating states to zero
    next_state_values = torch.zeros(tstate.hparams.batch_size, requires_grad=True).float().to(tstate.device)
    
    with torch.no_grad():
        next_state_values[non_final_mask] = tstate.target(non_final_states).max(1)[0]
    
    expected_state_action_values = (next_state_values * tstate.hparams.bellman_gamma) + rewards.view(-1)

    criterion = torch.nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    tstate.optim.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_value_(tstate.policy.parameters(), tstate.hparams.grad_clip_value)
    tstate.optim.step()

    return loss.item()


def train (
    tstate: QTrainState,
    episodes: int = 5000
):
    stats = collections.defaultdict(list)
    tstate.policy.train(True)
    tstate.target.train(True)

    print("Hyperparamters")
    print(json.dumps(tstate.hparams))

    for episode in tqdm.tqdm(range(episodes)):
        state, _ = tstate.env.reset()
        state = torch.from_numpy(state).to(tstate.device)

        tstate.evaluator.reset()

        for step in itertools.count():
            action = sample_action(tstate.env, state, tstate)
            obs, reward, terminated, truncated, _ = tstate.env.step(action)
            alive = not (terminated or truncated)

            ep_idxs = tstate.memory.tail(step)
            reward = torch.tensor([tstate.evaluator(not terminated, reward, episode, step, tstate.memory.rewards[ep_idxs])])

            next_state = None
            if alive:
                next_state = torch.from_numpy(obs).to(tstate.device)
            
            tstate.memory.push(state, action, next_state, reward)
            state = next_state

            loss = optimize(tstate)
            if loss is not None:
                stats["losses"].append(loss)
            
            if not alive:
                stats["episode_durations"].append(step)
                stats["scores"].append(tstate.evaluator.total_rewards())
                # print(stats)
                break
            
        if episode % tstate.hparams.target_update_freq == 0:
            target_state_dict = tstate.target.state_dict()
            policy_state_dict = tstate.policy.state_dict()
            for key in policy_state_dict:
                target_state_dict[key] = policy_state_dict[key] \
                    *tstate.hparams.bellman_tau + target_state_dict[key]*(1-tstate.hparams.bellman_tau)
            tstate.target.load_state_dict(target_state_dict)
        
        if (episode + 1) % tstate.log_freq == 0 and tstate.logger is not None:
            tstate.logger.log_aggregate("loss", stats["losses"], episode, include=["mean"])
            tstate.logger.log_aggregate("duration", stats["episode_durations"], episode)
            tstate.logger.log_aggregate("score", stats["scores"], episode)
            stats.clear()
        
        if (episode + 1) % tstate.write_freq == 0 and tstate.writer is not None:
            tstate.writer.log(tstate.policy, tstate.hparams, episode)


def evaluate (tstate: QTrainState, max_steps = 500):
    tstate.policy.eval()
    imgs = []
    logits = []
    rewards = []

    state, _ = tstate.env.reset()
    state = torch.from_numpy(state).to(tstate.device)
    tstate.evaluator.reset()
    memory = torch.zeros((max_steps,1)).to(tstate.device)

    with torch.no_grad():
        for step in itertools.count():
            action = tstate.policy(state.unsqueeze(0))
            logits.append(action)
            action = action.argmax().view(1).long().item()
            # action = tstate.env.action_space.sample()
            new_state, reward, terminated, _, _ = tstate.env.step(action)

            reward = tstate.evaluator(not terminated, reward, 0, step, memory)
            rewards.append(reward)

            state = torch.from_numpy(new_state).to(tstate.device)
            imgs.append(ttf.to_pil_image(state))

            if terminated or step > max_steps:
                break

    return imgs, logits, rewards







    
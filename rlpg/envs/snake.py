import numpy as np
import enum
import random
from PIL import Image, ImageDraw
from typing import Any, Optional, Tuple
import torch
import functools
import gymnasium as gym
import einops

import rlpg.utils as rl_utils
import rlpg.rl_memory as rl_memory
import rlpg.envs.types as rl_envs
import rlpg.envs.wrappers as rl_wrappers

try:
  import google.colab
  RESAMPLING_MODE = Image.BOX
except:
  RESAMPLING_MODE = Image.Resampling.BOX

class Direction(enum.Enum):
    N = 0
    E = 1
    S = 2
    W = 3

DIRS = [0,1,2,3]
DIR_STEPS = np.array([[0,-1], [1,0], [0,1], [-1,0]])

def to_row_major (h,x,y):
    return h*y+x

def from_row_major (w,h,idx):
    x = idx % w
    y = idx // h
    return x.astype(int), y.astype(int)

def get_board_mat (state):
    board = np.zeros((3, state.width, state.height))
    fxs, fys = from_row_major(state.width, state.height, np.array(state._food_idxs))
    board[1,fys, fxs] = 1
    board[2,state._snake[:,1], state._snake[:,0]] = 1
    board[0,state._snake[-1,1], state._snake[-1,0]] = 1
    return board

def get_board_mat_gray (state):
    board = np.zeros((1, state.width, state.height))
    fxs, fys = from_row_major(state.width, state.height, np.array(state._food_idxs))
    board[0, fys, fxs] = 0.2
    board[0,state._snake[:,1], state._snake[:,0]] = 0.6
    board[0,state._snake[-1,1], state._snake[-1,0]] = 1
    return board

def get_board_img (board, w, h):
    mat = get_board_mat(board)
    img = Image.fromarray((mat * 255).astype(np.uint8).T, "RGB")
    return img.resize((w,h), RESAMPLING_MODE)

def get_board_img_gray (board, w, h):
    mat = get_board_mat_gray(board).squeeze(0)
    img = Image.fromarray((mat * 255).astype(np.uint8))
    return img.resize((w,h), RESAMPLING_MODE)
 

class BoardState:
    def __init__ (self, w, h, pos = None):
        if not pos:
            pos = [np.random.randint(0, w-1), np.random.randint(0, h-1)]

        self.width = w
        self.height = h

        self._food_idxs = []
        self._snake = np.array([pos])

        self._direction = Direction(np.random.randint(0,4)).value
        self._free_arr = np.arange(0, self.width * self.height)

        self.score = 0
    
    def get_cell_states (self):
        sidxs = to_row_major(self.height, self._snake[:,0], self._snake[:,1])
        idxs = np.concatenate([sidxs, np.array(self._food_idxs, dtype=np.int32)])
        mask = np.ones(self.width * self.height, dtype=np.bool_)
        mask[idxs] = False
        free = self._free_arr[mask]
        return free, idxs

    def is_in_bounds (self, pos):
        if pos[0] < 0 or pos[0] >= self.width or pos[1] < 0 or pos[1] >= self.height:
            return False
        return True

    def is_free (self, pos):
        idx = to_row_major(self.height, pos[0], pos[1])
        free, occupied = self.get_cell_states()
        return idx in free and self.is_in_bounds(pos)
    
    def set_direction (self, direction: Direction):
        self._direction = direction.value
    
    #@numba.njit
    def get_food_distance (self):
        head = self._snake[-1]
        food = np.vstack(from_row_major(self.width, self.height, np.array(self._food_idxs))).T
        ds = food - head
        xs, ys = ds[:,0] / self.width, ds[:,1] / self.height
        ts = np.sqrt(xs**2 + ys**2)
        return ts.min()
    
    def add_food (self, count = 1):
        free, occupied = self.get_cell_states()
        # TODO: handle full board
        idxs = np.random.choice(free, count, replace=False)
        self._food_idxs += list(idxs)
        
    
    def step (self):
        head = self._snake[-1]
        new_pos = head + DIR_STEPS[self._direction]
        idx = to_row_major(self.height, *new_pos)

        if not self.is_in_bounds(new_pos) or not (self._snake - new_pos).any(1).all():
            return False
        
        grow = False

        self._snake = np.append(self._snake, [new_pos], 0)

        if idx in self._food_idxs:
            self.score += 1
            self._food_idxs.remove(idx)
            self.add_food()
            grow = True
        
        if not grow:
            self._snake = self._snake[1:]

        return True

class SimpleDirectionSampler:
    def __init__ (self):
        self.n_actions = 4
    
    def __call__(self, board):
        idx = np.random.randint(0,4)
        return Direction(idx)

class RelativeDirectionSampler:
    def __init__(self):
        self.n_actions = 3

    def __call__(self, board):
        idx = np.random.randint(-1,2)
        new_dir = (board._direction + idx) % 4
        return Direction(new_dir)

def _shuffle (vals):
    v = list(vals)
    random.shuffle(v)
    return v

def _initialize_snake_simple (board, spos, slen):
    sampler = RelativeDirectionSampler()
    snake = [spos]
    for _ in range(slen):
        direction = sampler(board)
        npos = spos - DIR_STEPS[direction.value]
        if not board.is_free(npos):
            break
        spos = npos
        snake = [npos] + snake
        board._snake = np.array(snake)
        board.score += 1

def _initialize_snake_greedy (board, spos, slen):
    height = board.height
    stack = [(spos, _shuffle(DIR_STEPS))]
    idxs = [to_row_major(height, spos[0], spos[1])]

    while len(stack) < slen:
        pos, dirs  = stack.pop()
        pidx = idxs.pop()
        while len(dirs):
            d = dirs.pop()
            npos = pos + d
            nidx = to_row_major(height, npos[0], npos[1])
            if board.is_in_bounds(npos) and nidx not in idxs:
                stack.append((pos, dirs))
                idxs.append(pidx)
                stack.append((npos, _shuffle(DIR_STEPS)))
                idxs.append(nidx)
                break


    board._snake = np.array([v[0] for v in stack])
    board.score = len(stack)


def initialize_snake_random (board, max_len, food_count, greedy=False):
    slen = random.randint(0, max_len)

    x = random.randint(0, board.width - 1)
    y = random.randint(0, board.height - 1)
    tail = np.array([x,y])
    snake = [tail]
    board._snake = np.array(snake)

    if greedy:
        _initialize_snake_greedy(board, tail, slen)
    else:
        _initialize_snake_simple(board, tail, slen)

    fc = random.randint(1, food_count)
    board.add_food(fc)

class SnakeEnv (gym.Env):
    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 30
    }

    def __init__(
        self, 
        board_size: Tuple[int, int],
        init_max_snake_len: int = 5,
        init_max_food_count: int = 5,
        init_greedy: bool = True,
        render_opts: rl_envs.RenderOptions = None,
        **kwargs
    ):
        self._board: BoardState = None
        self._board_size = board_size
        self._board_cells = board_size[0] * board_size[1]
        self._init_max_snake_len = init_max_snake_len
        self._init_max_food_count = init_max_food_count
        self._init_greedy = init_greedy
        self._render_opts = render_opts
        self.action_space = gym.spaces.Discrete(4)

        if not self._render_opts:
            self.observation_space = gym.spaces.Box(0, 1, (2, self._board_cells), dtype=int)
        else:
            channels = 3 if render_opts.color else 1
            shape = (channels, render_opts.size[1], render_opts.size[0])
            self.observation_space = gym.spaces.Box(0, 1, shape, dtype=np.float32)
    
    def _get_observation (self):
        if not self._render_opts:
            res = np.zeros((2, self._board_cells), dtype=np.int64)
            res[0,self._board._snake] = 1
            res[1,self._board._food_idxs] = 1
            return res, {}
        else:
            return self._render(self._render_opts), {}
    
    def reset (
        self,
        seed: int = None,
        **kwargs
    ):
        super().reset(seed=seed)

        self._board = BoardState(self._board_size[0], self._board_size[1])
        initialize_snake_random(self._board, self._init_max_snake_len, self._init_max_food_count, self._init_greedy)
        self._board.score = 0
        return self._get_observation()

    def step (self, action):
        v = action
        self._board.set_direction(Direction(v))
        done = not self._board.step()
        score = self._board.score
        obs, info = self._get_observation()
        return obs, score, done, False, info
    
    def _render (self, ropts: rl_envs.RenderOptions) -> np.array:
        grayscale = not ropts.color
        w, h = ropts.size

        if grayscale:
            img = get_board_img_gray(self._board, w, h)
            arr = np.array(img, dtype = np.float32) / 255.0
            arr = arr[np.newaxis,:]
            return arr
        else:
            img = get_board_img(self._board, w, h)
            arr = np.array(img, dtype = np.float32).T / 255.0
            return arr
    
    def render (self):
        ropts = rl_envs.RenderOptions((100,100))
        img = self._render(ropts)
        return einops.rearrange(img, "C H W -> H W C")

class SnakeEvaluator (rl_envs.RewardEvaluator):
    def __init__(self, env: Any, **kwargs) -> None:
        super().__init__(env, **kwargs)
        self.max_history_discount_steps = kwargs.get("max_history_discount_steps")
        self.discount_multiplier = kwargs.get("discount_multiplier")
        self.food_schedule = rl_utils.EpsilonScheduler(stop = 0, decay = kwargs.get("food_schedule_decay"))
        self.back_reward = torch.linspace(0, 1, self.max_history_discount_steps).unsqueeze(1)
        self._last_reward_step = 0
        self._last_reward = 0
    
    def to (self, device):
        self.back_reward = self.back_reward.to(device)
        return self
    
    def reset(self):
        super().reset()
        self._last_reward = 0
        self._last_reward_step = 0

    def score(self, alive: bool, reward: int, episode: int, step: int, mem: torch.Tensor) -> float:
        if not alive:
            # Adjust rewards for previous steps that led to failure
            step_delta = min(step - self._last_reward_step, self.max_history_discount_steps)
            if step_delta > 0:
                mem[-step_delta:] -= self.back_reward[-step_delta:] * self.discount_multiplier
            return -10
        
        reward_delta = reward - self._last_reward
        self._last_reward = reward

        if reward_delta <= 0:
            return -1
        else:
            self._last_reward_step = step
            return reward_delta * 10
        

gym.register("Snake-Basic", SnakeEnv)

def get_hparams (**kwargs):
    hparams = {
        "board_width": 10,
        "board_height": 10,
        "init_max_snake_len": 5,
        "init_max_food_count": 10,
        "init_greedy": True,
        "max_history_discount_steps": 5,
        "reward_multiplier": 10,
        "discount_multiplier": 0.25,
        "food_schedule_decay": 0.5
    }
    hparams = rl_utils.merge_dicts(kwargs, hparams)
    return rl_utils.DotDict(hparams)


def basic_snake (
    board_size: Tuple[int, int],
    device: str,
    **kwargs
):
    hparams = get_hparams(
        board_width=board_size[0], 
        board_height = board_size[1], 
        device = device,
        **kwargs
    )
    env = gym.make("Snake-Basic", board_size=board_size, **hparams)
    evaluator = SnakeEvaluator(env, **hparams).to(device)
    return env, evaluator, hparams

def snake_image (
    board_size: Tuple[int, int],
    image_size: Tuple[int, int],
    device: str,
    color: bool = True,
    **kwargs
):
    hparams = get_hparams(
        board_width=board_size[0],
        board_height=board_size[1],
        image_width = image_size[0],
        image_height = image_size[1],
        color=color, 
        device = device,
        **kwargs
    )
    ropts = rl_envs.RenderOptions(image_size, color=color)
    env = gym.make("Snake-Basic", board_size = board_size, render_opts = ropts, **hparams)
    evaluator = SnakeEvaluator(env, **hparams).to(device)
    return env, evaluator, hparams





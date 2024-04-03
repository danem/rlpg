import torch
import itertools
from typing import Sequence

class CircularBuffer:
    def __init__(self, size: int) -> None:
        self._size = size
        self._idx = 0
        self._len = 0
    
    def get_push_idx (self):
        # TODO: Consider creating a guard style API:
        # with cb.push() as idx:
        #    foo[idx] = 1
        #    bar[idx] = 2
        res = self._idx
        self._idx = (self._idx + 1) % self._size
        self._len = min(self._len+1, self._size)
        return res

    def sample_idxs (self, batch_size):
        return torch.randperm(self._len)[:batch_size]
    
    def tail (self, n):
        if n > self._len:
            raise ValueError("n gretter than length")
        return list(self.idxs(self._idx-n, self._idx))
    
    def idxs(self, start, end):
        front = (self._idx - self._len) % self._size
        back = self._idx
        
        if start < 0: start = (start + back) % self._size
        else: start = (start + front) % self._size
            
        if end < 0: end = (end + back) % self._size
        else: end = (end + front) % self._size

        if start <= end:
            return range(start, end)
        else:
            return itertools.chain(range(start, self._len), range(0, end))

    def clear (self):
        self._idx = 0
        self._len = 0

    def __len__ (self):
        return self._len

class ReplayMemory (CircularBuffer):
    def __init__ (self, size, observation_shape: Sequence[int], action_shape: Sequence[int] = None, action_dtype: type = torch.float32):
        super().__init__(size)

        action_shape = list(action_shape) if action_shape else [1]
        observation_shape = list(observation_shape)

        self.states = torch.zeros([size] + observation_shape)
        self.next_states = torch.zeros([size] + observation_shape)
        self.non_final_mask = torch.zeros(size).bool()
        self.actions = torch.zeros([size] + action_shape, dtype=action_dtype)
        self.rewards = torch.zeros([size, 1]).float()
    
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
        


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

        


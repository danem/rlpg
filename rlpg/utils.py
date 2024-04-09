import numpy as np
import pathlib
import scipy
import torch
import torch.utils.data as tud
from datetime import datetime
import rlpg.envs.types as rl_envs
import torchvision.transforms.functional as ttf

class DotDict(dict):
    def __getattr__(self, attr):
        if attr in self:
            return self[attr]
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")

    def __setattr__(self, attr, value):
        self[attr] = value

def merge_dicts (a, b):
    c = {**a}
    c.update(b)
    return DotDict(c)

class EpsilonScheduler:
    def __init__ (self, start = 0.9, stop = 0.1, decay = 1000, **kwargs):
        self.start = kwargs.get("start",start)
        self.stop = kwargs.get("stop",stop)
        self.decay = kwargs.get("decay",decay)
    
    def __call__(self, x):
        return self.stop + (self.start - self.stop) * np.exp(-1 * x / self.decay)

class ModelWriter:
    ITER = "ModelWriter.ITER"

    def __init__ (self, directory, fmt):
        self._format = fmt
        self._directory = pathlib.Path(directory)
        self._directory.mkdir(exist_ok=True)
        self._iter = 0
    
    def save (self, model, params):
        params[ModelWriter.ITER] = self._iter
        self._iter += 1

        now = datetime.now()
        fname = now.strftime(self._format).format(**params)

        print(f"Saving checkpoint to {self._directory / fname}")

        torch.save({
            **params,
            "model_state_dict": model.state_dict()
        }, self._directory/fname)



def shape_logger(module, input, output):
    ishape = None
    oshape = None
    name   = module.__class__.__name__

    if isinstance(input[0], torch.Tensor):
        ishape = input[0].shape
        oshape = output.shape

    elif isinstance(input[0], torch.nn.utils.rnn.PackedSequence):
        ishape = input[0][0].shape
        oshape = output[0][0].shape
        
    print(f"{name}: {ishape}, {oshape}")

def log_model_shape (model):
    for m in model.modules():
        m.register_forward_hook(shape_logger)

def freeze_module (module: torch.nn.Module):
    for param in module.parameters():
        param.requires_grad = False

def conv_size (d, k, s, padding = 0, dilation=1):
    pad = 2 * padding
    return int(((d + pad - dilation * (k-1) - 1)/s) + 1)

def conv_size2d (w, h, k, s, padding = 0, dilation=1):
    wsz = conv_size(w, k, s, padding, dilation)
    hsz = conv_size(h, k, s, padding, dilation)
    return wsz, hsz



# Taken from OpenAI spinning up
def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
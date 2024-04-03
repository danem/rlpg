import pickle
import os
import matplotlib.pyplot as plt
import ipywidgets
from typing import List
from PIL import Image
import torch
import torchvision.utils as tu
import torchvision.transforms.functional as ttf
import einops
import numpy as np

import rlpg.models.common as models

def pickle_cache (path):
    def wrapper (fn):
        def final (*args, **kwargs):
            if os.path.exists(path):
                print("Loading from pickle")
                with open(path,'rb') as f:
                    return pickle.load(f)
            else:
                res = fn(*args, **kwargs)
                with open(path, 'wb') as f:
                    pickle.dump(res, f)
                return res
        return final
    return wrapper

def show_imgs(imgs, figsize = (15,15)):
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize=figsize)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = ttf.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

def slider (fn):
    def wrapper (items, **kwargs):
        slider = ipywidgets.IntSlider(
            min=0,
            max=len(items)-1,
            step=1,
            value=0
        )
        def update (idx):
            fn(items[idx], **kwargs)
        ipywidgets.interact(update, idx=slider)
    return wrapper
            

def plot_conv_block (block: models.Conv2DBlock, input: torch.Tensor):
    inter = [tu.make_grid(input, nrow=1)]

    for l in block.convs:
        input = l(input)
        x = input.detach().cpu()

        inter.append(tu.make_grid(x, nrow=1))

    for img in inter:
        show_imgs(img)
import rlpg.utils as rl_utils
import rlpg.envs.types as rl_envs
import rlpg.models.vae as rl_vae

from typing import Dict, List, Tuple
import pathlib
import os
import torch
import dataclasses
import torchvision.transforms.functional as ttf
import tqdm
import random
import json
import gymnasium as gym
import uuid
import lightning

def generate_vae_images (
    env: gym.Env,
    root_dir: str, 
    episodes: int, 
    max_episode_steps: int, 
    render_options: rl_envs.RenderOptions,
    hparams: Dict,
    skip_start_odds: float = 0.8
):
    if dataclasses.is_dataclass(hparams):
        hparams = dataclasses.asdict(hparams)

    metadata = {
        "render_options": dataclasses.asdict(render_options),
        "max_episode_steps": max_episode_steps,
        "episodes": episodes,
        **hparams
    }

    dirs = []
    for fp in os.listdir(root_dir):
        path = os.path.join(root_dir, fp)
        if os.path.isdir(path):
            dirs.append(path)

    for d in dirs:
        meta_file = os.path.join(d, "_meta.json")
        if not os.path.exists(meta_file):
            continue
        with open(meta_file, "r") as f:
            tmp_metadata = json.load(f)
            # TODO: JSON turns everything into lists instead of tuples so this fails
            if json.dumps(tmp_metadata) == json.dumps(metadata):
                return d

    
    out_dir = os.path.join(root_dir, f"vae_images_{uuid.uuid4().hex}")
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_idx = 0

    with open(os.path.join(out_dir,"_meta.json"), 'w') as f:
        json.dump(metadata, f)
    
    for i in tqdm.tqdm(range(episodes)):
        env.reset()
        ep_step = 0
        skip_start = random.random() < skip_start_odds
        while True:
            action = env.action_space.sample()
            obs, _, terminated, _, _ = env.step(action)
            if terminated or ep_step > max_episode_steps:
                break
            ep_step += 1

            if skip_start and ep_step < 40:
                continue
            
            img = rl_envs.apply_renderopts(obs, render_options)
            img.save(out_dir / f"img{img_idx}.png")
            img_idx += 1

    return str(out_dir)

def get_vae_render_options (img_dir):
    metapath = os.path.join(img_dir, "_meta.json")
    with open(metapath, 'r') as f:
        metadata = json.load(f)
        return rl_envs.RenderOptions(**metadata["render_options"])

class VAEModule2D (lightning.LightningModule):
    def __init__(self, img_w, img_h, in_channels, hidden_size, latent_dims, hparams = None) -> None:
        super().__init__()
        self.model = rl_vae.VariationalAutoencoder2D(
            img_w=img_w,
            img_h=img_h,
            in_channels=in_channels,
            hidden_size=hidden_size,
            latent_dims=latent_dims
        )
        self.latent_dims = latent_dims
        self.params = hparams
        self.save_hyperparameters()
    
    def forward (self, x):
        x = x.to(self.device)
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x = batch
        x = x.to(self.device)
        xhat, kl = self.model(x)
        loss = torch.nn.functional.mse_loss(xhat, x) #+ kl

        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
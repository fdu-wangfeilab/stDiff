import torch
import numpy as np
import os
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from einops import rearrange, repeat

from ray.air import session
import os

from .stDiff_scheduler import NoiseScheduler


def normal_train_stDiff(model,
                 dataloader,
                 lr: float = 1e-4,
                 num_epoch: int = 1400,
                 pred_type: str = 'noise',
                 diffusion_step: int = 1000,
                 device=torch.device('cuda:1'),
                 is_tqdm: bool = True,
                 is_tune: bool = False,
                 mask = None):
    """通用训练函数

    Args:
        lr (float):
        momentum (float): 动量
        max_iteration (int, optional): 训练的 iteration. Defaults to 30000.
        pred_type (str, optional): 预测的类型噪声或者 x_0. Defaults to 'noise'.
        batch_size (int, optional):  Defaults to 1024.
        diffusion_step (int, optional): 扩散步数. Defaults to 1000.
        device (_type_, optional): Defaults to torch.device('cuda:1').
        is_class_condi (bool, optional): 是否采用condition. Defaults to False.
        is_tqdm (bool, optional): 开启进度条. Defaults to True.
        is_tune (bool, optional): 是否用 ray tune. Defaults to False.
        condi_drop_rate (float, optional): 是否采用 classifier free guidance 设置 drop rate. Defaults to 0..

    Raises:
        NotImplementedError: _description_
    """

    noise_scheduler = NoiseScheduler(
        num_timesteps=diffusion_step,
        beta_schedule='cosine'
    )

    criterion = nn.MSELoss()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)

    if is_tqdm:
        t_epoch = tqdm(range(num_epoch), ncols=100)
    else:
        t_epoch = range(num_epoch)

    model.train()

    for epoch in t_epoch:
        epoch_loss = 0.
        for i, (x, x_cond) in enumerate(dataloader): # 去掉了, celltype
            x, x_cond = x.float().to(device), x_cond.float().to(device)
            # celltype = celltype.to(device)

            noise = torch.randn(x.shape).to(device)
            timesteps = torch.randint(1, diffusion_step, (x.shape[0],)).long()

            x_t = noise_scheduler.add_noise(x,
                                            noise,
                                            timesteps=timesteps)

            mask = torch.tensor(mask).to(device)
            x_noisy = x_t * (1 - mask) + x * mask

            noise_pred = model(x_noisy, t=timesteps.to(device), y=x_cond) # 去掉了, z=celltype
            # loss = criterion(noise_pred, noise)

            loss = criterion(noise*(1-mask), noise_pred*(1-mask))

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # type: ignore
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()

        epoch_loss = epoch_loss / (i + 1)  # type: ignore
        if is_tqdm:
            t_epoch.set_postfix_str(f'{pred_type} loss:{epoch_loss:.5f}')  # type: ignore
        if is_tune:
            session.report({'loss': epoch_loss})

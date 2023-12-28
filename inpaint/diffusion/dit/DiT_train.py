import torch
import numpy as np
import os
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from einops import rearrange, repeat

import ray
from ray import tune
from ray.air import session
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
import sys
pwd = '/home/lijiahao/projects/sc-diffusion/'
sys.path.append(pwd)
from Extenrnal.sc_DM.model.process_h5ad import get_data_ary,get_data_loader,get_data_msk_loader
from Extenrnal.sc_DM.model.diffusion_model import MLP,TransformerEncoder,CrossTransformer
from Extenrnal.sc_DM.synthesis.DiT_scheduler import NoiseScheduler
from Extenrnal.sc_DM.model.diffusion_model import SinusoidalEmbedding
from Extenrnal.sc_DM.synthesis.DiT_model import DiT

import os


def normal_train(model, 
                dataloader,
                lr: float = 1e-4,
                num_epoch:int=1400,
                pred_type:str='noise',
                diffusion_step:int=1000,
                device=torch.device('cuda:0'),
                is_tqdm:bool=True,
                is_tune:bool=False,):
    """通用训练函数

    Args:
        lr (float): 
        momentum (float): 动量
        max_iteration (int, optional): 训练的 iteration. Defaults to 30000.
        pred_type (str, optional): 预测的类型噪声或者 x_0. Defaults to 'noise'.
        batch_size (int, optional):  Defaults to 1024.
        diffusion_step (int, optional): 扩散步数. Defaults to 1000.
        device (_type_, optional): Defaults to torch.device('cuda:0').
        is_class_condi (bool, optional): 是否采用condition. Defaults to False.
        is_tqdm (bool, optional): 开启进度条. Defaults to True.
        is_tune (bool, optional): 是否用 ray tune. Defaults to False.
        condi_drop_rate (float, optional): 是否采用 classifier free guidance 设置 drop rate. Defaults to 0..

    Raises:
        NotImplementedError: _description_
    """
    pwd = '/home/lijiahao/projects/sc-diffusion/'
    
    # adata, data_ary, cell_type = get_data_ary(pwd + 'data/pbmc_ATAC.h5ad')
    
    # data_ary = np.load(pwd + f'data/npy_ary/{data_type}_data_ary.npy') * 2 - 1
    
    noise_scheduler = NoiseScheduler(
        num_timesteps=diffusion_step,
        beta_schedule='cosine'
    )
    
    criterion = nn.MSELoss()
    model.to(device)
    # num_epoch = max_iteration // len(dataloader)
    
    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     lr=lr,
    #     momentum=momentum
    # )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)
    
    if  is_tqdm:
        t_epoch = tqdm(range(num_epoch), ncols=100)
    else:
        t_epoch = range(num_epoch)
    
    model.train()
    
    for epoch in t_epoch:
        epoch_loss = 0.
        for i, (x,celltype) in enumerate(dataloader):
            
            x,celltype = x.float().to(device), celltype.long().to(device)
            
            noise = torch.randn(x.shape).to(device)
            timesteps = torch.randint(1, diffusion_step, (x.shape[0],)).long()
            
            x_t = noise_scheduler.add_noise(x,
                                            noise,
                                            timesteps=timesteps)
            
            noise_pred = model(x_t, t=timesteps.to(device), y=celltype)
            loss = criterion(noise_pred, noise)  
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0) # type: ignore
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            
        epoch_loss = epoch_loss/(i+1) # type: ignore
        if is_tqdm:
            t_epoch.set_postfix_str(f'{pred_type} loss:{epoch_loss:.5f}') # type: ignore
        if is_tune:
            session.report({'loss': epoch_loss})


def normal_train_palette(model,
                 dataloader,
                 lr: float = 1e-4,
                 num_epoch: int = 1400,
                 pred_type: str = 'noise',
                 diffusion_step: int = 1000,
                 device=torch.device('cuda:0'),
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
        device (_type_, optional): Defaults to torch.device('cuda:0').
        is_class_condi (bool, optional): 是否采用condition. Defaults to False.
        is_tqdm (bool, optional): 开启进度条. Defaults to True.
        is_tune (bool, optional): 是否用 ray tune. Defaults to False.
        condi_drop_rate (float, optional): 是否采用 classifier free guidance 设置 drop rate. Defaults to 0..

    Raises:
        NotImplementedError: _description_
    """
    pwd = '/home/lijiahao/projects/sc-diffusion/'

    # adata, data_ary, cell_type = get_data_ary(pwd + 'data/pbmc_ATAC.h5ad')

    # data_ary = np.load(pwd + f'data/npy_ary/{data_type}_data_ary.npy') * 2 - 1

    noise_scheduler = NoiseScheduler(
        num_timesteps=diffusion_step,
        beta_schedule='cosine'
    )

    criterion = nn.MSELoss()
    model.to(device)
    # num_epoch = max_iteration // len(dataloader)

    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     lr=lr,
    #     momentum=momentum
    # )

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


def train_scripts(device = torch.device('cuda:0'),
                  model_name:str = 'cosine_condi_ln_mlp.pth',
                  pred_type:str='noise',
                  num_epoch:int=30000):
    seed = 1202
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = DiT(
        input_size=10,
        hidden_size=1024,
        depth=6,
        num_heads=16,
        classes=13,
        mlp_ratio=4.0,
        dit_type='dit'
    )
    # model = TransformerEncoder(
    #     input_dim=10,
    #     emb_dim=40,
    #     num_classes=13,
    #     is_learned_timeebd=True,
    #     is_time_concat=True,
    #     is_condi_concat=True,
    #     is_msk_emb=False,
    #     num_heads=5,
    #     attn_types=['mlp','mlp']
    # )
    model.to(device)
    print(model)
    # parameters={'lr': 0.09930906002366541, 'momentum': 0.8865807468993898,'batch_size': 1024}
    # parameters={'lr': 0.05499951781790261, 'momentum': 0.4754472160656253,'batch_size': 1024}
    # parameters={'lr': 0.09972768793470284, 'momentum': 0.8865217158805158, 'batch_size': 2048}
    # rna 
    parameters={'lr': 0.09886909388175943, 'momentum': 0.7779814458926817, 'batch_size': 1024}

    data_ary = np.load('./data/npy_ary/atac_latent_ary.npy')
    print(f'data max:{data_ary.max()} data min:{data_ary.min()}')
    celltype_ary = np.load('./data/npy_ary/atac_celltype_ary.npy')
    dataset = TensorDataset(torch.from_numpy(data_ary.astype(np.float32)), torch.from_numpy(celltype_ary.astype(np.int32)))
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=parameters['batch_size'],
        num_workers=4,
        shuffle=True
    )
    
    normal_train(model,
                 dataloader=dataloader,
             num_epoch=num_epoch,
             diffusion_step=1000,
             device=device,
             pred_type='noise',)
    torch.save(model.state_dict(), f'./synthesis/ckpt/{model_name}')
    
    
if __name__ =='__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # os.environ["RAY_SESSION_DIR"] = "/home/lijiahao/ray_session"
    # ray_tune()
    train_scripts(model_name='scale2_latent_dit.pt',
                  device = torch.device('cuda:0'),
                  pred_type='noise',
                  num_epoch=600)
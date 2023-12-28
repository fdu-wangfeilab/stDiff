import torch
import numpy as np
import torch.nn as nn
import scanpy as sc
import anndata
from anndata import AnnData
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from model.process_h5ad import get_processed_data_ary, get_data_loader
from model.diffusion_model import MLP
from model.diffusion_scheduler import NoiseScheduler
from model.process_h5ad import plot_hvg_umap

def get_diffusion_res(forward_step: int,
                      scheduler: NoiseScheduler, 
                      dataloader: DataLoader):
    """ 指定 diffusion step 来生成 x_t

    Args:
        forward_step (int): 前向扩散步数
        scheduler (NoiseScheduler): 
        dataloader (_type_):  

    Returns:
        _type_: x_t
    """
    timesteps = list(range(scheduler.num_timesteps))
    timestep = timesteps[forward_step]
    noisy = []
    for i,data in enumerate(dataloader):
        noise = torch.randn(data[0].shape)
        t = torch.from_numpy(np.repeat(timestep, len(data[0]))).long()
        x_t = scheduler.add_noise(data[0], noise, t )
        noisy.append(x_t)
    noisy_t = torch.cat(noisy,dim=0)
    return noisy_t

def plot_diffusion_res(ori_adata:AnnData,
                       x_t: torch.Tensor
                       ):
    """将原始数copy一份, 之后进行加噪

    Args:
        ori_adata (AnnData): 原始未加噪数据
        x_t (torch.Tensor): 
    """
    adata_cpy = ori_adata.copy()
    adata_cpy.X = x_t.numpy()
    plot_hvg_umap(adata_cpy)
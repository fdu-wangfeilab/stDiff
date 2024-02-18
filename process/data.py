
import scipy
import anndata as ad
import scanpy as sc
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from scipy.sparse import issparse, csr
from anndata import AnnData
from sklearn.preprocessing import maxabs_scale, MaxAbsScaler

CHUNK_SIZE = 20000


def reindex(adata, genes, chunk_size=CHUNK_SIZE):
    """
    Reindex AnnData with gene list
    
    Parameters
    ----------
    adata
        AnnData
    genes
        gene list for indexing
    chunk_size
        chunk large data into small chunks
        
    Return
    ------
    AnnData
    """
    idx = [i for i, g in enumerate(genes) if g in adata.var_names]
    print('There are {} gene in selected genes'.format(len(idx)))
    if len(idx) == len(genes):
        adata = adata[:, genes]
    else:
        new_X = scipy.sparse.lil_matrix((adata.shape[0], len(genes)))
        for i in range(new_X.shape[0]//chunk_size+1):
            new_X[i*chunk_size:(i+1)*chunk_size, idx] = adata[i*chunk_size:(i+1)*chunk_size, genes[idx]].X
        adata = AnnData(new_X.tocsr(), obs=adata.obs, var={'var_names':genes}) 
    return adata

def plot_hvg_umap(hvg_adata,color=['celltype'],path = None, save_filename=None):
    sc.set_figure_params(dpi=80, figsize=(3,3)) # type: ignore
    hvg_adata = hvg_adata.copy()
    if save_filename:
        sc.settings.figdir = path
        # save = '.pdf'
        save = f'{save_filename}.pdf'
    else:
        save = None
    # ideal gas equation
    
    sc.pp.scale(hvg_adata, max_value=10)
    sc.tl.pca(hvg_adata)
    sc.pp.neighbors(hvg_adata, n_pcs=30, n_neighbors=30)
    sc.tl.umap(hvg_adata, min_dist=0.1)
    sc.pl.umap(hvg_adata, color=color,legend_fontsize=15, ncols=2 ,show=None,save=save)
    return hvg_adata


def get_data_loader(data_ary:np.ndarray, 
                    cell_type:np.ndarray, 
                    batch_size:int=512,
                    is_shuffle:bool=True,
                    ):
    
        data_tensor = torch.from_numpy(data_ary.astype(np.float32))
        cell_type_tensor = torch.from_numpy(cell_type.astype(np.float32))
        dataset = TensorDataset(data_tensor,cell_type_tensor)
        generator = torch.Generator(device='cuda')
        return DataLoader(
                dataset, batch_size=batch_size, shuffle=is_shuffle, drop_last=False , generator=generator) #, generator=torch.Generator(device = 'cuda') 
          


def scale(adata):
    scaler = MaxAbsScaler()
    # 对adata.X按行进行归一化
    normalized_data = scaler.fit_transform(adata.X.T).T

    # 更新归一化后的数据到adata.X
    adata.X = normalized_data
    return adata


def data_augment(adata: AnnData, fixed: bool, noise_std):
   
    # 定义增强参数，例如噪声的标准差
    noise_stddev = noise_std
    augmented_adata = adata.copy()
    gene_expression = adata.X
    
    if fixed: 
        augmented_adata.X = augmented_adata.X + np.full(gene_expression.shape, noise_stddev)
    else:
        # 对每个基因的表达值引入随机噪声
        augmented_adata.X = augmented_adata.X + np.abs(np.random.normal(0, noise_stddev, gene_expression.shape))   
    
    merge_adata = adata.concatenate(augmented_adata, join='outer')
    
    
    return merge_adata
    

    
    

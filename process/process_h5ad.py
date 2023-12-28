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

def batch_scale(adata, chunk_size=CHUNK_SIZE):
    """
    Batch-specific scale data
    
    Parameters
    ----------
    adata
        AnnData
    chunk_size
        chunk large data into small chunks
    
    Return
    ------
    AnnData
    """
    # 就是 b 就是 batchname，如 rna，atac 的数据，b就是('rna','atac')
    for b in adata.obs['batch'].unique():
        # 取某个batch 所有数据的 index
        idx = np.where(adata.obs['batch']==b)[0]
        
        # 对某个batch中的数据做归一化, 这样也是对列做归一化
        scaler = MaxAbsScaler(copy=False).fit(adata.X[idx])
        tr = tqdm(range(len(idx)//chunk_size+1),ncols=80)
        tr.set_description_str(desc=f'batch_scale')
        for i in tr:
            adata.X[idx[i*chunk_size:(i+1)*chunk_size]] = scaler.transform(adata.X[idx[i*chunk_size:(i+1)*chunk_size]])

    return adata


def preprocessing_rna(
    adata: AnnData,
    min_features: int = 600,
    min_cells: int = 3,
    target_sum: int = 10000,
    n_top_features = 2000,
    chunk_size: int = CHUNK_SIZE,
    log=None
):
    # 重新设置一遍预处理的参数
    if min_features is None: min_features = 600
    if n_top_features is None: n_top_features = 2000
    if target_sum is None: target_sum = 10000
    # 如果不是系数矩阵的格式则转换为稀疏矩阵
    if not isinstance(adata.X, csr.csr_matrix):
        tmp = scipy.sparse.csr_matrix(adata.X)
        adata.X = None
        adata.X = tmp
        
    # 筛选出 不是['ERCC', 'MT-', 'mt-'] 开头的 gene
    adata = adata[:, [gene for gene in adata.var_names if not str(gene).startswith(tuple(['ERCC', 'MT-', 'mt-']))]]
    
    # 直接调用 scanpy 来过滤 cell,gene 并且标准化数据
    # 要求细胞中至少有 min_features 个基因表达
    sc.pp.filter_cells(adata, min_genes=min_features)
    # 要求基因至少在 min cell 中表达
    sc.pp.filter_genes(adata, min_cells=min_cells)
    # 把每一行的数据放缩到 10000，但是保持和为 10000
    # 将每个细胞的总测量量标准化为相同的值，有助于消除细胞之间的测量偏差，
    # 因为在细胞总数相等的情况下，每个基因在不同细胞之间的相对表达量会更加可比。
    sc.pp.normalize_total(adata, target_sum=target_sum)
    
    # 使数据更加符合正态分布
    sc.pp.log1p(adata)
    
    # 保存一份数据在 adata.raw 中
    adata.raw = adata
    
    # 此处 n_top_features 是 2000
    if type(n_top_features) == int and n_top_features>0:
        # 此处是取 2000 个HVG 
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_features, batch_key='batch', inplace=False, subset=True)
    elif type(n_top_features) != int:
        adata = reindex(adata, n_top_features)
        
    # batch 采用 maxabs 归一化,因为数据全部>0， 最后范围为 [0，1], 
    # 分 chunck 归一化每次相当于 max 取chunk中的最大值
    adata = batch_scale(adata, chunk_size=chunk_size)
    return adata

def plot_hvg_umap(hvg_adata,color=['celltype'],save_filename=None):
    sc.set_figure_params(dpi=80, figsize=(3,3)) # type: ignore
    hvg_adata = hvg_adata.copy()
    if save_filename:
        sc.settings.figdir = save_filename
        save = '.pdf'
    else:
        save = None
    # ideal gas equation
    
    sc.pp.scale(hvg_adata, max_value=10)
    sc.tl.pca(hvg_adata)
    sc.pp.neighbors(hvg_adata, n_pcs=30, n_neighbors=30)
    sc.tl.umap(hvg_adata, min_dist=0.1)
    sc.pl.umap(hvg_adata, color=color,legend_fontsize=10, ncols=2, show=None,save=save, wspace = 1)
    return hvg_adata

def plot_hvg_latent_umap(cpy_x):
    sc.set_figure_params(dpi=80, figsize=(3,3))
    sc.pp.neighbors(cpy_x, n_neighbors=30, use_rep='latent')
    sc.tl.umap(cpy_x, min_dist=0.1)
    cpy_x.obsm['X_scalex_umap'] = cpy_x.obsm['X_umap']
    cols = ['batch', 'celltype']
    color = [c for c in cols if c in cpy_x.obs]
    sc.pl.umap(cpy_x, color=color, save=None, wspace=0.4, ncols=4, show=True)


def get_data_ary(h5ad_path, n_top_features=2000):
    adata = sc.read_h5ad(h5ad_path)
    # 将 batch 中的数据转化为 category 类型
    adata.obs['batch'] = adata.obs['batch'].astype('category')
    processed_adata = preprocessing_rna(
        adata, 
        min_features=600,
        min_cells=3,
        target_sum=None,
        n_top_features= n_top_features,
        chunk_size=CHUNK_SIZE,
        log=None
    )
    return processed_adata, processed_adata.X.toarray(), processed_adata.obs['celltype'].cat.codes.values


def get_processed_data_ary(adata_path, data_ary_path, cell_type_ary_path):
    adata = sc.read(adata_path)
    data_ary = np.load(data_ary_path)
    cell_type_ary = np.load(cell_type_ary_path)
    return adata, data_ary, cell_type_ary

def get_data_loader(data_ary:np.ndarray, 
                    cell_type:np.ndarray, 
                    batch_size:int=512,
                    is_shuffle:bool=True):
    
        data_tensor = torch.from_numpy(data_ary.astype(np.float32))
        cell_type_tensor = torch.from_numpy(cell_type.astype(np.float32))
        dataset = TensorDataset(data_tensor,cell_type_tensor)
        return DataLoader(
                dataset, batch_size=batch_size, shuffle=is_shuffle, drop_last=False  , generator=torch.Generator(device = 'cuda') ) #, generator=torch.Generator(device = 'cuda') 
                

def get_data_msk_loader(data_ary:np.ndarray, 
                    cell_type:np.ndarray,
                    non_zero_msk,
                    batch_size:int=512,
                    is_shuffle:bool=True):
    
        data_tensor = torch.from_numpy(data_ary.astype(np.float32))
        cell_type_tensor = torch.from_numpy(cell_type.astype(np.int32))
        msk_tensor = torch.from_numpy(non_zero_msk.astype(np.int32))
        dataset = TensorDataset(data_tensor,cell_type_tensor,msk_tensor)
        return DataLoader(
                dataset, batch_size=batch_size, shuffle=is_shuffle, drop_last=False)

if __name__ == '__main__':
    adata, data_ary, cell_types = get_data_ary('./data/pbmc_ATAC.h5ad')
    plot_hvg_umap(adata, save_filename='./model/out/test')
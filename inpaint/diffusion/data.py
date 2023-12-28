from scipy.sparse import issparse, csr
from anndata import AnnData

from Extenrnal.sc_DM.model.process_h5ad import *
CHUNK_SIZE = 20000

# def reindex(adata, genes, chunk_size=CHUNK_SIZE):

def batch_scale_gimVI(adata, chunk_size=CHUNK_SIZE):
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

    # 取某个batch 所有数据的 index
    idx = np.array(range(adata.n_obs)) # 这里就是所有细胞的id（因为只有一个batch）

    # 对某个batch中的数据做归一化 除每一列（基因）的最大值 这个操作会和原数据的相关性不为1
    scaler = MaxAbsScaler(copy=False).fit(adata.X[idx])
    tr = tqdm(range(len(idx) // chunk_size + 1), ncols=80)
    tr.set_description_str(desc=f'batch_scale')
    for i in tr:
        adata.X[idx[i * chunk_size:(i + 1) * chunk_size]] = scaler.transform(
            adata.X[idx[i * chunk_size:(i + 1) * chunk_size]])

    return adata

def scale(adata):
    scaler = MaxAbsScaler()
    # 对adata.X按行进行归一化
    normalized_data = scaler.fit_transform(adata.X.T).T

    # 更新归一化后的数据到adata.X
    adata.X = normalized_data
    return adata


# def batch_scale(adata, chunk_size=CHUNK_SIZE):

# def preprocessing_rna(
#         adata: AnnData,
#         min_features: int = 600,
#         min_cells: int = 3,
#         target_sum: int = 10000,
#         n_top_features=2000,
#         chunk_size: int = CHUNK_SIZE,
#         log=None
# ):


def preprocessing_rna_gimVI(
        adata: AnnData,
        min_features: int = 600,
        min_cells: int = 3,
        target_sum: int = 10000,
        n_top_features=2000,
        chunk_size: int = CHUNK_SIZE,
        log=None
):
    # 重新设置一遍预处理的参数
    if min_features is None: min_features = 600
    if n_top_features is None: n_top_features = 2000
    if target_sum is None: target_sum = 10000
    # 如果不是系数矩阵的格式则转换为稀疏矩阵
    if type(adata.X) != csr.csr_matrix:
        tmp = scipy.sparse.csr_matrix(adata.X) # 因为直接赋值会出错
        adata.X = None
        adata.X = tmp

    # 筛选出 不是['ERCC', 'MT-', 'mt-'] 开头的 gene
    # gimVI实验注掉了
    # adata = adata[:, [gene for gene in adata.var_names if not str(gene).startswith(tuple(['ERCC', 'MT-', 'mt-']))]]

    # 直接调用 scanpy 来过滤 cell,gene 并且标准化数据
    # 要求细胞中至少有 min_features 个基因表达
    sc.pp.filter_cells(adata, min_genes=min_features)
    # 要求基因至少在 min cell 中表达
    sc.pp.filter_genes(adata, min_cells=min_cells)
    # 把每一行的数据放缩到 10000，但是保持和为 10000？ 没太理解到
    sc.pp.normalize_total(adata, target_sum=target_sum)

    sc.pp.log1p(adata)

    # 保存一份数据在 adata.raw 中
    adata.raw = adata

    # 此处 n_top_features 是 2000 gimVI注掉了
    # if type(n_top_features) == int and n_top_features > 0:
    #     # 此处是取 2000 个HVG
    #     sc.pp.highly_variable_genes(adata, n_top_genes=n_top_features, batch_key='batch', inplace=False, subset=True)
    # elif type(n_top_features) != int:
    #     adata = reindex(adata, n_top_features)

    # 最后在 batch 内还需要归一化， 采用 maxabs 归一化
    # adata = batch_scale(adata, chunk_size=chunk_size)
    adata = batch_scale_gimVI(adata, chunk_size=chunk_size)
    # adata = scale(adata)
    return adata

def get_data_loader_no_guidance(data_ary:np.ndarray,
                    batch_size:int=512,
                    is_shuffle:bool=True):
    """no cell type"""
    data_tensor = torch.from_numpy(data_ary.astype(np.float32))
    dataset = TensorDataset(data_tensor)
    return DataLoader(
            dataset, batch_size=batch_size, shuffle=is_shuffle, drop_last=False)


if __name__ == '__main__':
    adata, data_ary, cell_types = get_data_ary('../datasets/pbmc_atac/pbmc_ATAC.h5ad')
    plot_hvg_umap(adata, save_filename='result')
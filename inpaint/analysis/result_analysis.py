from scipy.stats import spearmanr, pearsonr
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc

from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, homogeneity_score, \
    normalized_mutual_info_score

# spearman
def imputation_metrics(original, imputed):
    absolute_error = np.abs(original - imputed)
    relative_error = absolute_error / np.maximum( # 这样分母上一直是1 所以相对误差这里要改一下
        np.abs(original), np.ones_like(original)
    )
    m, n = original.shape
    total_error = np.sum(absolute_error)  # 求得总误差
    mae = total_error / (m * n)  # 计算MAE

    spearman_gene = []
    for g in range(imputed.shape[1]):
        if np.all(imputed[:, g] == 0):
            correlation = 0
        else:
            correlation = spearmanr(original[:, g], imputed[:, g])[0]
        spearman_gene.append(correlation)

    spearman_cell = []
    for c in range(imputed.shape[0]):
        if np.all(imputed[c, :] == 0):
            correlation = 0
        else:
            correlation = spearmanr(original[c, :], imputed[c, :])[0]
        spearman_cell.append(correlation)

    pearson_corr = []
    for i in range(imputed.shape[1]):
        x = original[:, i]
        y = imputed[:, i]
        if np.all(x == 0) or np.all(y == 0):
            pearson_corr.append(0)
            continue
        pearson_corr.append(pearsonr(x, y)[0])



    pearson_cell = []
    for i in range(imputed.shape[0]):
        x = original[i, :]
        y = imputed[i, :]
        if np.all(x == 0) or np.all(y == 0):
            pearson_cell.append(0)
            continue
        pearson_cell.append(pearsonr(x, y)[0])

    return {
        "median_absolute_error_per_gene": np.median(absolute_error, axis=0),
        "mean_absolute_error_per_gene": np.mean(absolute_error, axis=0),
        "mean_relative_error": np.mean(relative_error, axis=1),
        "median_relative_error": np.median(relative_error, axis=1),
        "spearman_per_gene": np.array(spearman_gene),
        "pearson_per_gene":np.array(pearson_corr),

        # Metric we report in the GimVI paper:
        "median_spearman_per_gene": np.median(spearman_gene),
        "median_spearman_per_cell": np.median(spearman_cell),
        "median_pearson_per_gene":np.median(pearson_corr),
        "median_pearson_per_cell": np.median(pearson_cell),
        "MAE":mae
    }


def benchmark_imputation(gt, imputed, test_indices):
    imputed = imputed # 只有预测的部分
    reality = gt[:, test_indices] # 用于和预测值比较
    imputation = imputation_metrics(reality, imputed)
    return imputation

def plot_density(gt, imputed, test_indices, field, label):
    tmp = benchmark_imputation(gt, imputed, test_indices)
    sns.distplot(tmp[field], label=label, hist=False)
    plt.title(field)
    plt.legend()
    # plt.show()


# 空间背景
def plot_gene_spatial(adata_spatial_gt, adata_spatial_impu, gene):
    # adata数据在0-1之间
    data_fish = adata_spatial_gt

    fig, (ax_gt, ax) = plt.subplots(1, 2)

    if type(gene) == str:
        gene_id = list(data_fish.var_names).index(gene)
    else:
        gene_id = gene

    x_coord = data_fish.obs.x_coord.ravel()
    y_coord = data_fish.obs.y_coord.ravel()

    def order_by_strenght(x, y, z):
        ind = np.argsort(z) # 从小到大排列返回 原下标
        return x[ind], y[ind], z[ind]

    s = 20

    def transform(data):
        return np.log(1 + 50 * data)

    # Plot groundtruth
    x, y, z = order_by_strenght(
        x_coord, y_coord, data_fish.X.toarray()[:, gene_id] / (data_fish.X.toarray().sum(axis=1) + 1) # +1是防止除以小于1的值
        # x_coord, y_coord, data_fish.X.toarray()[:, gene_id] # 要思考一下需不需要/分母
    )
    ax_gt.scatter(x, y, c=transform(z), s=s, edgecolors="none", marker="s", cmap="Reds")
    ax_gt.set_title("Groundtruth")
    ax_gt.axis("off")

    imputed = adata_spatial_impu
    x, y, z = order_by_strenght(x_coord, y_coord, imputed.X[:, gene_id] / (imputed.X.sum(axis=1) + 1))
    # x, y, z = order_by_strenght(x_coord, y_coord, imputed.X[:, gene_id])
    ax.scatter(x, y, c=transform(z), s=s, edgecolors="none", marker="s", cmap="Reds")
    ax.set_title("Imputed")
    ax.axis("off")
    plt.tight_layout()
    plt.show()

def clustering_metrics(adata, target, pred, mode="AMI"):
    """
    Evaluate clustering performance.
    
    Parameters
    ----------
    adata
        AnnData object of shape `n_obs` × `n_vars`. Rows correspond to cells and columns to genes.
    target
        Key in `adata.obs` where ground-truth spatial domain labels are stored.
    pred
        Key in `adata.obs` where clustering assignments are stored.
        
    Returns
    -------
    ami
        Adjusted mutual information score.
    ari
        Adjusted Rand index score.
    homo
        Homogeneity score.
    nmi
        Normalized mutual information score.

    """
    if(mode=="AMI"):
        ami = adjusted_mutual_info_score(adata.obs[target], adata.obs[pred])
        print("AMI ",ami)
    elif(mode=="ARI"):
        ari = adjusted_rand_score(adata.obs[target], adata.obs[pred])
        print("ARI ",ari)
    elif(mode=="Homo"):
        homo = homogeneity_score(adata.obs[target], adata.obs[pred])
        print("Homo ",homo)
    elif(mode=="NMI"):
        nmi = normalized_mutual_info_score(adata.obs[target], adata.obs[pred])
        print("NMI ", nmi)

def get_N_clusters(adata, n_cluster, cluster_method='louvain', range_min=0, range_max=3, max_steps=30, tolerance=0):
    """
    Tune the resolution parameter in clustering to make the number of clusters and the specified number as close as possible.
   
    Parameters
    ----------
    adata
        AnnData object of shape `n_obs` × `n_vars`. Rows correspond to cells and columns to genes.
    n_cluster
        Specified number of clusters.
    cluster_method
        Method (`louvain` or `leiden`) used for clustering. By default, cluster_method='louvain'.
    range_min
        Minimum clustering resolution for the binary search.
    range_max
        Maximum clustering resolution for the binary search.
    max_steps
        Maximum number of steps for the binary search.
    tolerance
        Tolerance of the difference between the number of clusters and the specified number.

    Returns
    -------
    adata
        AnnData object with clustering assignments in `adata.obs`:

        - `adata.obs['louvain']` - Louvain clustering assignments if `cluster_method='louvain'`.
        - `adata.obs['leiden']` - Leiden clustering assignments if `cluster_method='leiden'`.

    """

    this_step = 0
    this_min = float(range_min)
    this_max = float(range_max)
    # 处理数据
    sc.tl.pca(adata)
    sc.pp.neighbors(adata, n_pcs=30, n_neighbors=30)

    while this_step < max_steps:
        this_resolution = this_min + ((this_max-this_min)/2)
        if cluster_method=='leiden':
            sc.tl.leiden(adata, resolution=this_resolution)
        if cluster_method=='louvain':
            sc.tl.louvain(adata, resolution=this_resolution)
        this_clusters = adata.obs[cluster_method].nunique()

        if this_clusters > n_cluster+tolerance:
            this_max = this_resolution
        elif this_clusters < n_cluster-tolerance:
            this_min = this_resolution
        else:
            print("Succeed to find %d clusters at resolution %.3f."%(n_cluster, this_resolution))
            return adata
        this_step += 1

    print('Cannot find the number of clusters.')
    return adata


# batch merge


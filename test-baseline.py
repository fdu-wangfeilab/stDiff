import numpy as np
import pandas as pd
import sys
import os
import scipy.stats as st
import copy
from sklearn.model_selection import KFold
import pandas as pd
import scanpy as sc
import warnings
from os.path import join
import torch
warnings.filterwarnings('ignore')
torch.set_default_tensor_type('torch.cuda.FloatTensor')

from process.result_analysis import *
from baseline.scvi.model import GIMVI
from baseline.stPlus import *
from process.data import *
# import uniport as up

import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--sc_data", type=str, default="dataset5_seq_915.h5ad")
parser.add_argument("--sp_data", type=str, default='dataset5_spatial_915.h5ad')
parser.add_argument("--document", type=str, default='dataset5')
parser.add_argument("--rand", type=int, default=0)
args = parser.parse_args()
# ******** preprocess ********


# 过滤过的原始数据
n_splits = 5 # 交叉验证组数
adata_spatial = sc.read_h5ad('datasets/sp/' + args.sp_data)
adata_seq = sc.read_h5ad('datasets/sc/' + args.sc_data)

# 标准预处理
adata_seq2 = adata_seq.copy()
# tangram用
adata_seq3 =  adata_seq2.copy()
sc.pp.normalize_total(adata_seq2, target_sum=1e4)
sc.pp.log1p(adata_seq2)
data_seq_array = adata_seq2.X

adata_spatial2 = adata_spatial.copy()
sc.pp.normalize_total(adata_spatial2, target_sum=1e4)
sc.pp.log1p(adata_spatial2)
data_spatial_array = adata_spatial2.X

sp_genes = np.array(adata_spatial.var_names)
sp_data = pd.DataFrame(data=data_spatial_array, columns=sp_genes)
sc_data = pd.DataFrame(data=data_seq_array, columns=sp_genes)

# ****对比方法****

def SpaGE_impute():
    '''
    需要预处理
    Returns
    -------

    '''
    print ('We run SpaGE for this data\n')
    sys.path.append("baseline/SpaGE-master/")
    from SpaGE.main import SpaGE
    global sc_data, sp_data, adata_seq, adata_spatial


    raw_shared_gene = np.array(adata_spatial.var_names)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state= args.rand)
    kf.get_n_splits(raw_shared_gene)
    torch.manual_seed(10)
    idx = 1
    all_pred_res = np.zeros_like(adata_spatial.X)

    for train_ind, test_ind in kf.split(raw_shared_gene):
        print("\n===== Fold %d =====\nNumber of train genes: %d, Number of test genes: %d" % (
            idx, len(train_ind), len(test_ind)))
        train_gene = raw_shared_gene[train_ind]
        test_gene = raw_shared_gene[test_ind]
        pv = len(train_gene) / 2
        sp_data_partial = sp_data[train_gene]

        Imp_Genes = SpaGE(sp_data_partial, sc_data, n_pv=int(pv),
                          genes_to_predict=test_gene)

        all_pred_res[:, test_ind] = Imp_Genes
        idx += 1

    return all_pred_res


def Tangram_impute(annotate=None, modes='clusters', density='rna_count_based'):
    '''
    Returns
    -------

    '''
    import torch
    from torch.nn.functional import softmax, cosine_similarity, sigmoid
    import baseline.tangram as tg
    print('We run Tangram for this data\n')
    global adata_seq3, adata_spatial, locations
    from sklearn.model_selection import KFold


    
    raw_shared_gene = adata_spatial.var_names
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=args.rand)
    kf.get_n_splits(raw_shared_gene)
    torch.manual_seed(10)
    idx = 1
    all_pred_res = np.zeros_like(adata_spatial.X)

    for train_ind, test_ind in kf.split(raw_shared_gene):
        print("\n===== Fold %d =====\nNumber of train genes: %d, Number of test genes: %d" % (
            idx, len(train_ind), len(test_ind)))
        train_gene = list(raw_shared_gene[train_ind])
        test_gene = list(raw_shared_gene[test_ind])

        adata_seq_tmp = adata_seq2.copy()
        adata_spatial_tmp = adata_spatial2.copy()

        adata_spatial_partial = adata_spatial_tmp[:, train_gene]
        train_gene = np.array(train_gene)
        if annotate == None:
            RNA_data_adata_label = adata_seq3.copy()
            sc.pp.normalize_total(RNA_data_adata_label)
            sc.pp.log1p(RNA_data_adata_label)
            sc.pp.highly_variable_genes(RNA_data_adata_label)
            RNA_data_adata_label = RNA_data_adata_label[:, RNA_data_adata_label.var.highly_variable]
            sc.pp.scale(RNA_data_adata_label, max_value=10)
            sc.tl.pca(RNA_data_adata_label)
            sc.pp.neighbors(RNA_data_adata_label)
            sc.tl.leiden(RNA_data_adata_label, resolution=0.5)
            adata_seq_tmp.obs['leiden'] = RNA_data_adata_label.obs.leiden
        else:
            global CellTypeAnnotate
            adata_seq_tmp.obs['leiden'] = CellTypeAnnotate
        tg.pp_adatas(adata_seq_tmp, adata_spatial_partial, genes=train_gene) 

        device = torch.device('cuda:1')
        if modes == 'clusters':
            ad_map = tg.map_cells_to_space(adata_seq_tmp, adata_spatial_partial, device=device, mode=modes,
                                           cluster_label='leiden', density_prior=density)
            ad_ge = tg.project_genes(ad_map, adata_seq_tmp, cluster_label='leiden')
        else:
            ad_map = tg.map_cells_to_space(adata_seq_tmp, adata_spatial_partial, device=device)
            ad_ge = tg.project_genes(ad_map, adata_seq_tmp)
        test_list = list(set(ad_ge.var_names) & set(test_gene))
        test_list = np.array(test_list)
        all_pred_res[:, test_ind] = ad_ge.X[:, test_ind]

        idx += 1


    return all_pred_res


def gimVI_impute():
    '''
    本来处理的就是原始数据，所以传入预处理或者原始数据都可

    如果是标准与处理 需要传入adata_seq2 spatial2
    Returns
    -------

    '''
    print ('We run gimVI for this data\n')
    import baseline.scvi as scvi
    import scanpy as sc
    # from scvi.model import GIMVI
    import torch
    from torch.nn.functional import softmax, cosine_similarity, sigmoid
    global adata_seq2, adata_spatial2

    from sklearn.model_selection import KFold
    raw_shared_gene = adata_spatial2.var_names
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=args.rand)  # shuffle = false 不设置state，就是按顺序划分
    kf.get_n_splits(raw_shared_gene)
    torch.manual_seed(10)
    idx = 1
    all_pred_res = np.zeros_like(data_spatial_array)

    for train_ind, test_ind in kf.split(raw_shared_gene):
        print("\n===== Fold %d =====\nNumber of train genes: %d, Number of test genes: %d" % (
            idx, len(train_ind), len(test_ind)))
        train_gene = np.array(raw_shared_gene[train_ind])
        test_gene = np.array(raw_shared_gene[test_ind])

        Genes = list(adata_spatial2.var_names)
        rand_gene_idx = test_ind
        n_genes = len(Genes)
        rand_train_gene_idx = train_ind
        rand_train_genes = np.array(Genes)[rand_train_gene_idx] # 不就是train_genes吗
        rand_genes = np.array(Genes)[rand_gene_idx] # test_gene
        adata_spatial_partial = adata_spatial2[:, rand_train_genes]
        sc.pp.filter_cells(adata_spatial_partial, min_counts=0)
        seq_data = copy.deepcopy(adata_seq2)
        seq_data = seq_data[:, Genes]
        sc.pp.filter_cells(seq_data, min_counts=0)
        scvi.data.setup_anndata(adata_spatial_partial)
        scvi.data.setup_anndata(seq_data)
        model = GIMVI(seq_data, adata_spatial_partial)
        model.train(200)
        _, imputation = model.get_imputed_values(normalized=False)
        all_pred_res[:, test_ind] = imputation[:, rand_gene_idx]
        idx += 1

    return all_pred_res


def stPlus_impute():
    '''
    输入预处理后的数据，需要标准预处理
    Returns
    -------

    '''
    global sc_data, sp_data, outdir, adata_spatial

    raw_shared_gene = np.array(adata_spatial.var_names)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=args.rand)
    kf.get_n_splits(raw_shared_gene)
    torch.manual_seed(10)
    idx = 1
    all_pred_res = np.zeros_like(adata_spatial.X)
    for train_ind, test_ind in kf.split(raw_shared_gene):
        print("\n===== Fold %d =====\nNumber of train genes: %d, Number of test genes: %d" % (
            idx, len(train_ind), len(test_ind)))
        train_gene = np.array(raw_shared_gene[train_ind])
        test_gene = np.array(raw_shared_gene[test_ind])

        save_path_prefix = join(outdir, 'process_file/stPlus-demo')
        if not os.path.exists(join(outdir, "process_file")):
            os.mkdir(join(outdir, "process_file"))
        stPlus_res = stPlus(sp_data[train_gene], sc_data, test_gene, save_path_prefix)
        all_pred_res[:, test_ind] = stPlus_res
        idx += 1

    return all_pred_res


def uniport_impute():
    global sc_data, sp_data, adata_seq, adata_spatial


    raw_shared_gene = np.array(adata_spatial.var_names)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=args.rand) # 0
    kf.get_n_splits(raw_shared_gene)
    torch.manual_seed(10)
    idx = 1
    all_pred_res = np.zeros_like(adata_spatial.X)

    for train_ind, test_ind in kf.split(raw_shared_gene):
        print("\n===== Fold %d =====\nNumber of train genes: %d, Number of test genes: %d" % (
            idx, len(train_ind), len(test_ind)))
        train_gene = raw_shared_gene[train_ind]
        test_gene = raw_shared_gene[test_ind]
        
        adata_seq_tmp = adata_seq2.copy()
        adata_spatial_tmp = adata_spatial2.copy()
        
        adata_spatial_tmp.obs['domain_id'] = 0
        adata_spatial_tmp.obs['domain_id'] = adata_spatial_tmp.obs['domain_id'].astype('category')
        adata_spatial_tmp.obs['source'] = 'ST'

        adata_seq_tmp.obs['domain_id'] = 1
        adata_seq_tmp.obs['domain_id'] = adata_seq_tmp.obs['domain_id'].astype('category')
        adata_seq_tmp.obs['source'] = 'RNA'
        
        adata_cm = adata_spatial_tmp.concatenate(adata_seq_tmp, join='inner', batch_key='domain_id')
        print(adata_cm.obs)
        spatial_data = adata_cm[adata_cm.obs['source']=='ST'].copy()
        seq_data = adata_cm[adata_cm.obs['source']=='RNA'].copy()
        
        spatial_data_partial = spatial_data[:,train_gene].copy()
        adata_cm = spatial_data_partial.concatenate(seq_data, join='inner', batch_key='domain_id')
        print(adata_cm.X.shape)
        # return
        up.batch_scale(adata_cm)
        up.batch_scale(spatial_data_partial)
        up.batch_scale(seq_data)
        
        seq_data.X = scipy.sparse.coo_matrix(seq_data.X)
        spatial_data_partial.X = scipy.sparse.coo_matrix(spatial_data_partial.X)

        adatas = [spatial_data_partial, seq_data]

        adata = up.Run(adatas=adatas, adata_cm=adata_cm, lambda_kl=5.0, model_info=False)

        spatial_data_partial.X = spatial_data_partial.X.A

        adata_predict = up.Run(adata_cm=spatial_data_partial, out='predict', pred_id=1)
        model_res = pd.DataFrame(adata_predict.obsm['predict'], columns=raw_shared_gene)

        all_pred_res[:, test_ind] = model_res[test_gene]
        idx += 1

    return all_pred_res


Data = args.document
outdir = 'Result/' + Data + '/'
if not os.path.exists(outdir):
    os.mkdir(outdir)


SpaGE_result = SpaGE_impute() 
SpaGE_result_pd = pd.DataFrame(SpaGE_result, columns=sp_genes)
SpaGE_result_pd.to_csv(outdir +  '/SpaGE_impute.csv',header = 1, index = 1)

Tangram_result = Tangram_impute() 
Tangram_result_pd = pd.DataFrame(Tangram_result, columns=sp_genes)
Tangram_result_pd.to_csv(outdir +  '/Tangram_impute.csv',header = 1, index = 1)

gimVI_result = gimVI_impute() 
gimVI_result_pd = pd.DataFrame(gimVI_result, columns=sp_genes)
gimVI_result_pd.to_csv(outdir +  '/gimVI_impute.csv',header = 1, index = 1)


stPlus_result = stPlus_impute() 
stPlus_result_pd = pd.DataFrame(stPlus_result, columns=sp_genes)
stPlus_result_pd.to_csv(outdir +  '/stPlus_impute.csv',header = 1, index = 1)

uniport_result = uniport_impute()
uniport_result_pd = pd.DataFrame(uniport_result, columns=sp_genes)
uniport_result_pd.to_csv(outdir +  '/uniport_impute.csv',header = 1, index = 1)

#******** metrics ********

def cal_ssim(im1, im2, M):
    assert len(im1.shape) == 2 and len(im2.shape) == 2
    assert im1.shape == im2.shape
    mu1 = im1.mean()
    mu2 = im2.mean()
    sigma1 = np.sqrt(((im1 - mu1) ** 2).mean())
    sigma2 = np.sqrt(((im2 - mu2) ** 2).mean())
    sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()
    k1, k2, L = 0.01, 0.03, M
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    C3 = C2 / 2
    l12 = (2 * mu1 * mu2 + C1) / (mu1 ** 2 + mu2 ** 2 + C1)
    c12 = (2 * sigma1 * sigma2 + C2) / (sigma1 ** 2 + sigma2 ** 2 + C2)
    s12 = (sigma12 + C3) / (sigma1 * sigma2 + C3)
    ssim = l12 * c12 * s12

    return ssim


def scale_max(df):
    result = pd.DataFrame()
    for label, content in df.items():
        content = content / content.max()
        result = pd.concat([result, content], axis=1)
    return result


def scale_z_score(df):
    result = pd.DataFrame()
    for label, content in df.items():
        content = st.zscore(content)
        content = pd.DataFrame(content, columns=[label])
        result = pd.concat([result, content], axis=1)
    return result


def scale_plus(df):
    result = pd.DataFrame()
    for label, content in df.items():
        content = content / content.sum()
        result = pd.concat([result, content], axis=1)
    return result


def logNorm(df):
    df = np.log1p(df)
    df = st.zscore(df)
    return df


class CalculateMeteics:
    def __init__(self, raw_data, genes_name,impute_count_file, prefix, metric):
        self.impute_count_file = impute_count_file
        self.raw_count = pd.DataFrame(raw_data, columns=genes_name)
        self.raw_count.columns = [x.upper() for x in self.raw_count.columns]
        self.raw_count = self.raw_count.T
        self.raw_count = self.raw_count.loc[~self.raw_count.index.duplicated(keep='first')].T
        self.raw_count = self.raw_count.fillna(1e-20)

        self.impute_count = pd.read_csv(impute_count_file, header=0, index_col=0)
        self.impute_count.columns = [x.upper() for x in self.impute_count.columns]
        self.impute_count = self.impute_count.T
        self.impute_count = self.impute_count.loc[~self.impute_count.index.duplicated(keep='first')].T
        self.impute_count = self.impute_count.fillna(1e-20)
        self.prefix = prefix
        self.metric = metric

    def SSIM(self, raw, impute, scale='scale_max'):
        if scale == 'scale_max':
            raw = scale_max(raw)
            impute = scale_max(impute)
        else:
            print('Please note you do not scale data by scale max')
        if raw.shape[0] == impute.shape[0]:
            result = pd.DataFrame()
            for label in raw.columns:
                if label not in impute.columns:
                    ssim = 0
                else:
                    raw_col = raw.loc[:, label]
                    impute_col = impute.loc[:, label]
                    impute_col = impute_col.fillna(1e-20)
                    raw_col = raw_col.fillna(1e-20)
                    M = [raw_col.max(), impute_col.max()][raw_col.max() > impute_col.max()]
                    raw_col_2 = np.array(raw_col)
                    raw_col_2 = raw_col_2.reshape(raw_col_2.shape[0], 1)
                    impute_col_2 = np.array(impute_col)
                    impute_col_2 = impute_col_2.reshape(impute_col_2.shape[0], 1)
                    ssim = cal_ssim(raw_col_2, impute_col_2, M)

                ssim_df = pd.DataFrame(ssim, index=["SSIM"], columns=[label])
                result = pd.concat([result, ssim_df], axis=1)
        else:
            print("columns error")
        return result

    def SPCC(self, raw, impute, scale=None):
        if raw.shape[0] == impute.shape[0]:
            result = pd.DataFrame()
            for label in raw.columns:
                if label not in impute.columns:
                    spearmanr = 0
                else:
                    raw_col = raw.loc[:, label]
                    impute_col = impute.loc[:, label]
                    impute_col = impute_col.fillna(1e-20)
                    raw_col = raw_col.fillna(1e-20)
                    spearmanr, _ = st.spearmanr(raw_col, impute_col)
                spearman_df = pd.DataFrame(spearmanr, index=["SPCC"], columns=[label])
                result = pd.concat([result, spearman_df], axis=1)
        else:
            print("columns error")
        return result

    def JS(self, raw, impute, scale='scale_plus'):
        if scale == 'scale_plus':
            raw = scale_plus(raw)
            impute = scale_plus(impute)
        else:
            print('Please note you do not scale data by plus')
        if raw.shape[0] == impute.shape[0]:
            result = pd.DataFrame()
            for label in raw.columns:
                if label not in impute.columns:
                    JS = 1
                else:
                    raw_col = raw.loc[:, label]
                    impute_col = impute.loc[:, label]
                    raw_col = raw_col.fillna(1e-20)
                    impute_col = impute_col.fillna(1e-20)
                    M = (raw_col + impute_col) / 2
                    JS = 0.5 * st.entropy(raw_col, M) + 0.5 * st.entropy(impute_col, M)
                JS_df = pd.DataFrame(JS, index=["JS"], columns=[label])
                result = pd.concat([result, JS_df], axis=1)
        else:
            print("columns error")
        return result

    def RMSE(self, raw, impute, scale='zscore'):
        if scale == 'zscore':
            raw = scale_z_score(raw)
            impute = scale_z_score(impute)
        else:
            print('Please note you do not scale data by zscore')
        if raw.shape[0] == impute.shape[0]:
            result = pd.DataFrame()
            for label in raw.columns:
                if label not in impute.columns:
                    RMSE = 1.5
                else:
                    raw_col = raw.loc[:, label]
                    impute_col = impute.loc[:, label]
                    impute_col = impute_col.fillna(1e-20)
                    raw_col = raw_col.fillna(1e-20)
                    RMSE = np.sqrt(((raw_col - impute_col) ** 2).mean())

                RMSE_df = pd.DataFrame(RMSE, index=["RMSE"], columns=[label])
                result = pd.concat([result, RMSE_df], axis=1)
        else:
            print("columns error")
        return result


    def cluster(self, raw, impu,scale=None):

        ad_sp = adata_spatial2.copy()
        ad_sp.X = raw

        cpy_x = adata_spatial2.copy()
        cpy_x.X = impu

        sc.tl.pca(ad_sp)
        sc.pp.neighbors(ad_sp, n_pcs=30, n_neighbors=30)
        sc.tl.leiden(ad_sp)
        tmp_adata1 = ad_sp

        sc.tl.pca(cpy_x)
        sc.pp.neighbors(cpy_x, n_pcs=30, n_neighbors=30)
        sc.tl.leiden(cpy_x)
        tmp_adata2 = cpy_x

        tmp_adata2.obs['class'] = tmp_adata1.obs['leiden']
        ari = clustering_metrics(tmp_adata2, 'class', 'leiden', "ARI")
        ami = clustering_metrics(tmp_adata2, 'class', 'leiden', "AMI")
        homo = clustering_metrics(tmp_adata2, 'class', 'leiden', "Homo")
        nmi = clustering_metrics(tmp_adata2, 'class', 'leiden', "NMI")
        
        
        # tmp_adata2 = get_N_clusters(cpy_x, 23, 'leiden')
        
        # tmp_adata2.obs['class'] = ad_sp.obs['subclass_label']
        # ari = clustering_metrics(tmp_adata2, 'class', 'leiden', "ARI")
        # ami = clustering_metrics(tmp_adata2, 'class', 'leiden', "AMI")
        # homo = clustering_metrics(tmp_adata2, 'class', 'leiden', "Homo")
        # nmi = clustering_metrics(tmp_adata2, 'class', 'leiden', "NMI")
        
        result = pd.DataFrame([[ari, ami, homo, nmi]], columns=["ARI", "AMI", "Homo", "NMI"])
        return result

    def compute_all(self):
        raw = self.raw_count
        impute = self.impute_count
        prefix = self.prefix
        SSIM_gene = self.SSIM(raw, impute)
        Spearman_gene = self.SPCC(raw, impute)
        JS_gene = self.JS(raw, impute)
        RMSE_gene = self.RMSE(raw, impute)

        cluster_result = self.cluster(raw, impute)

        result_gene = pd.concat([Spearman_gene, SSIM_gene, RMSE_gene, JS_gene], axis=0)
        result_gene.T.to_csv(prefix + "_gene_Metrics.txt", sep='\t', header=1, index=1)

        cluster_result.to_csv(prefix + "_cluster_Metrics.txt", sep='\t', header=1, index=1)

        return result_gene


import seaborn as sns
import os
PATH = 'Result/'
DirFiles = os.listdir(PATH)


def CalDataMetric(Data):
    print ('We are calculating the : ' + Data + '\n')
    metrics = ['SPCC(gene)','SSIM(gene)','RMSE(gene)','JS(gene)']
    metric = ['SPCC','SSIM','RMSE','JS']
    impute_count_dir = PATH + Data
    impute_count = os.listdir(impute_count_dir)
    impute_count = [x for x in impute_count if x [-3:] == 'csv']
    methods = []
    if len(impute_count)!=0:
        medians = pd.DataFrame()
        for impute_count_file in impute_count:
            print(impute_count_file)
            if 'result_Tangram.csv' in impute_count_file:
                os.system('mv ' + impute_count_dir + '/result_Tangram.csv ' + impute_count_dir + '/Tangram_impute.csv')
            prefix = impute_count_file.split('_')[0]
            methods.append(prefix)
            prefix = impute_count_dir + '/' + prefix
            impute_count_file = impute_count_dir + '/' + impute_count_file
            # if not os.path.isfile(prefix + '_Metrics.txt'):
            print (impute_count_file)
            CM = CalculateMeteics(data_spatial_array, sp_genes, impute_count_file = impute_count_file, prefix = prefix, metric = metric)
            CM.compute_all()

            # 计算中位数
            median = []
            for j in ['_gene']:
            # j = '_gene'
            #     median = []
                tmp = pd.read_csv(prefix + j + '_Metrics.txt', sep='\t', index_col=0)
                for m in metric:
                    median.append(np.median(tmp[m]))
            median = pd.DataFrame([median], columns=metrics)
            # 聚类指标
            clu = pd.read_csv(prefix + '_cluster' + '_Metrics.txt', sep='\t', index_col=0)
            median = pd.concat([median, clu], axis=1)
            medians = pd.concat([medians, median], axis=0)

        metrics += ["ARI", "AMI", "Homo", "NMI"]
        medians.columns = metrics
        medians.index = methods
        medians.to_csv(outdir +  '/final_result.csv',header = 1, index = 1)

CalDataMetric(Data)

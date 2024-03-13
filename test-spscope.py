import numpy as np
import pandas as pd
import os
import scipy.stats as st
from sklearn.model_selection import KFold
import pandas as pd
import scanpy as sc
import warnings
import torch
from baseline.SpatialScope.utils import ConcatCells


from process.result_analysis import clustering_metrics


warnings.filterwarnings('ignore')
torch.set_default_tensor_type('torch.cuda.FloatTensor')

import torch
from scipy.spatial.distance import cdist

import subprocess



import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--sc_data", type=str, default="dataset11_seq_347.h5ad")
parser.add_argument("--sp_data", type=str, default='dataset11_spatial_347.h5ad')
parser.add_argument("--document", type=str, default='dataset11_std')
parser.add_argument("--rand", type=int, default=0)

args = parser.parse_args()
# ******** preprocess ********

n_splits = 5 
adata_spatial = sc.read_h5ad('datasets/sp/' + args.sp_data)
adata_seq = sc.read_h5ad('datasets/sc/' + args.sc_data)

adata_seq2 = adata_seq.copy()

# tangram
adata_seq3 =  adata_seq2.copy()

adata_spatial2 = adata_spatial.copy()

sc.pp.normalize_total(adata_seq2, target_sum=1e4)
sc.pp.log1p(adata_seq2)
data_seq_array = adata_seq2.X


sc.pp.normalize_total(adata_spatial2, target_sum=1e4)
sc.pp.log1p(adata_spatial2)
data_spatial_array = adata_spatial2.X


sp_genes = np.array(adata_spatial.var_names)
sp_data = pd.DataFrame(data=data_spatial_array, columns=sp_genes)
sc_data = pd.DataFrame(data=data_seq_array, columns=sp_genes)

# spscope
adata_spatial_partial = adata_spatial2.copy()
adata_spatial_partial.obs = adata_spatial_partial.obs.rename(columns = {'x_coord':'x', 'y_coord':'y'})
cell_locations = adata_spatial_partial.obs.copy()
cell_locations['spot_index'] = np.array(cell_locations.index)
cell_locations['cell_index'] = cell_locations['spot_index'].map(lambda x:x+'_0')
cell_locations['cell_nums'] = np.ones(cell_locations.shape[0]).astype(int)
adata_spatial_partial.uns['cell_locations'] = cell_locations

dataset = args.sc_data.split('_')[0]
dataset_sp = args.sc_data.split('.')[0]
adata_spatial_partial.write('./ckpt/sp/'+ dataset +'_spatial_part_spscope.h5ad')


celltype = ['python', './baseline/SpatialScope/Cell_Type_Identification.py', '--cell_class_column', 'subclass_label', 
        '--out_dir', './spscope_output/' + dataset, '--ST_Data', './ckpt/sp/'+ dataset +'_spatial_part_spscope.h5ad',
        '--SC_Data', './ckpt/sc/' + dataset_sp + '_spscope.h5ad', '--tissue', dataset , '--hs_ST']
subprocess.run(celltype)

csv_path = 'spscope_output/' + dataset+ '/' + dataset + '/CellTypeLabel_nu10.csv'
a = pd.read_csv(csv_path)
# data_spatial_array_spscope = data_spatial_array[np.array(a['spot_index']), :]
data_spatial_array_spscope = data_spatial_array


def spscope_impute():

    
    raw_shared_gene = np.array(adata_spatial.var_names)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=args.rand)
    kf.get_n_splits(raw_shared_gene)
    torch.manual_seed(10)
    idx = 1
    

    all_pred_res = np.zeros((len(a), adata_spatial.X.shape[1]))
    
    total_size = adata_spatial.X.shape[0]
    chunk_size = 1000
    arr = [(start, min(start + chunk_size, total_size)) for start in range(0, total_size, chunk_size)]

    
    for train_ind, test_ind in kf.split(raw_shared_gene):
        print("\n===== Fold %d =====\nNumber of train genes: %d, Number of test genes: %d" % (
            idx, len(train_ind), len(test_ind)))
        train_gene = np.array(raw_shared_gene[train_ind])
        test_gene = np.array(raw_shared_gene[test_ind])
        test_gene_str = ','.join(test_gene)

        
        for i in arr:
            print(f"{i[0]},{i[1]}")
            decomposition = ['python', './baseline/SpatialScope/Decomposition.py', '--tissue', dataset, '--out_dir', './spscope_output/' + dataset,
                        '--SC_Data', './ckpt/sc/' + dataset_sp + '_spscope.h5ad', '--cell_class_column', 'subclass_label',  '--ckpt_path', 'ckpt/' + dataset + '/model_05000.pt',
                        '--spot_range', f"{i[0]},{i[1]}", '--replicates', '5', '--gpu', '1', '--leave_out_test', '--test_genes', f'{test_gene_str}']
            subprocess.run(decomposition)
        
        spot_range = np.concatenate((np.arange(0,total_size,chunk_size), np.array([total_size])))
        ad_res = ConcatCells(spot_range,file_path='./spscope_output/' + dataset + '/' + dataset + '/',prefix='generated_cells_spot',suffix='.h5ad')

        all_pred_res[:, test_ind] = ad_res.X[:, test_ind]
        idx += 1
        

    return all_pred_res
    

Data = args.document
outdir = 'Result/spscope/' + Data + '/'
if not os.path.exists(outdir):
    os.mkdir(outdir)


spscope_result = spscope_impute()
spscope_result_pd = pd.DataFrame(spscope_result, columns=sp_genes)
spscope_result_pd.to_csv(outdir +  '/spscope_impute.csv',header = 1, index = 1)
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
        # ad_sp = adata_spatial2[np.array(a['spot_index']), :].copy()
        ad_sp.X = raw

        cpy_x = adata_spatial2.copy()
        # cpy_x = adata_spatial2[np.array(a['spot_index']), :].copy()
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
        # self.accuracy = result_all
        return result_gene


import seaborn as sns
import os
PATH = 'Result/spscope/'
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
            print (impute_count_file)
            CM = CalculateMeteics(data_spatial_array_spscope, sp_genes, impute_count_file = impute_count_file, prefix = prefix, metric = metric)
            CM.compute_all()

            # 计算中位数
            median = []
            for j in ['_gene']:

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

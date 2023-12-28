#%%
import sys

import warnings
import torch
from inpaint.diffusion.diffusion_model import TransformerEncoder, TransformerEncoderNoGuidance
from inpaint.diffusion.diffusion_scheduler import RepaintNoiseScheduler
from inpaint.diffusion.diffusion_train import normal_train, normal_train_no_guidance
warnings.filterwarnings('ignore')

save_path = "data"

#%%

import numpy as np
import copy
import matplotlib.pyplot as plt
import scanpy as sc
import anndata as ad
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
from inpaint.diffusion.data import *
from inpaint.analysis.result_analysis import *
#%%

# 处理数据
# adata_spatial1 = sc.read_h5ad('inpaint/datasets/data/scalex/scalex_spatial1.h5ad')
# adata_seq1 = sc.read_h5ad('inpaint/datasets/data/scalex/scalex_rna1.h5ad')
# adata_spatial2 = sc.read_h5ad('inpaint/datasets/data/smfish.h5ad')
# adata_seq2 = sc.read_h5ad('inpaint/datasets/data/rna/Allen_SSP.h5ad')
adata_spatial1 = sc.read_h5ad('E:/Project/Datasets/spatialscope/MERFISH_mop.h5ad')
adata_seq1 = sc.read_h5ad('E:/Project/Datasets/spatialscope/Ref_snRNA_mop_qc3_2Kgenes.h5ad')

adata_spatial2 = preprocessing_rna_gimVI(
    adata_spatial2,
    min_features=1,
    min_cells=1,
    target_sum=None,
    n_top_features=1,
    chunk_size=CHUNK_SIZE,
    log=None)
adata_spatial2 = ad.AnnData(adata_spatial2.X.toarray())

adata_seq2 = preprocessing_rna_gimVI(
    adata_seq2,
    min_features=1,
    min_cells=1,
    target_sum=None,
    n_top_features=1,
    chunk_size=CHUNK_SIZE,
    log=None)
adata_seq2 = ad.AnnData(adata_seq2.X.toarray())


adata_seq1.obs['batch'] = 'spatial-recon'
adata_seq1.obs['batch'] = adata_seq1.obs['batch'].astype('category')
adata_seq2.obs['batch'] = 'spatial-origin'
adata_seq2.obs['batch'] = adata_seq2.obs['batch'].astype('category')
merge_adata2 = ad.concat([adata_seq1, adata_seq2])
plot_hvg_umap(merge_adata2,color=['batch'], save_filename='inpaint/out')

adata_spatial1.obs['batch'] = 'spatial-recon'
adata_spatial1.obs['batch'] = adata_spatial1.obs['batch'].astype('category')
adata_spatial2.obs['batch'] = 'spatial-origin'
adata_spatial2.obs['batch'] = adata_spatial2.obs['batch'].astype('category')
merge_adata1 = ad.concat([adata_spatial1, adata_spatial2])
plot_hvg_umap(merge_adata1,color=['batch'], save_filename='inpaint/out')


tmp = benchmark_imputation(adata_seq1.X, adata_seq2.X, list(range(33)))
tmp['median_spearman_per_gene']

# test scalex

import scanpy as sc
from scalex import SCALEX

adata_spatial = sc.read_h5ad('inpaint/datasets/data/smfish.h5ad')
adata_seq = sc.read_h5ad('inpaint/datasets/data/rna/Allen_SSP.h5ad')

adata_spatial.obs['batch'] = 'spatial'
adata_seq.obs['batch'] = 'rna_seq'
data_list = [adata_seq, adata_spatial]
batch_categories = ['rna_seq', 'spatial']
adata = SCALEX(data_list,
               batch_categories,
               min_cells=1,
               min_features=1,
               n_top_features=33,
               impute='spatial'
               )

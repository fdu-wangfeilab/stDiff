import numpy as np
import anndata as ad
import pandas as pd
import scanpy as sc
from Extenrnal.sc_DM.inpaint.diffusion.data import reindex


path = 'E:\Project\Datasets\data'
# data_seq = pd.read_csv(path + '\DataUpload\Dataset6\scRNA_count.txt', sep='\t', index_col=0)
# data_spatial = pd.read_csv(path +'\DataUpload\Dataset6\Spatial_count.txt', sep='\t')
# data_seq = data_seq.transpose()


adata_seq = sc.read('DataUpload/Dataset3/scRNA_count.txt', sep = '\t', first_column_names = True).T
adata_spatial = sc.read('DataUpload/Dataset3/Spatial_count.txt', sep = '\t')
locations = sc.read('DataUpload/Dataset3/Locations.txt', sep = '\t')
# RNA_data_adata = sc.read(RNA_file, sep = '\t', first_column_names = True).T
# Spatial_data_adata = sc.read(path +'\DataUpload\Dataset6\Spatial_count.txt', sep = '\t')

adata_seq = reindex(adata_seq, adata_spatial.var_names)
sc.pp.filter_genes(adata_seq, min_cells=1)
sc.pp.filter_cells(adata_seq, min_genes=1)
adata_spatial = reindex(adata_spatial, adata_seq.var_names)
sc.pp.filter_genes(adata_spatial, min_cells=1)
sc.pp.filter_cells(adata_spatial, min_genes=1)
# 要求基因至少在 min cell 中表达
# sc.pp.filter_genes(adata_spatial, min_cells=1)

adata_seq.write('datasets/sc/dataset3_seq_170.h5ad')
adata_spatial.write('datasets/sp/dataset3_spatial_170.h5ad')




# adata_spatial.X = None
# adata_spatial.X = data_spatial_array
# adata_seq.X = None
# adata_seq.X = data_seq_array

# adata_spatial.write_h5ad('prepro_data3_spatial_169.h5ad')
# adata_seq.write_h5ad('prepro_data3_seq_169.h5ad')

# for train_ind, test_ind in kf.split(raw_shared_gene):
#      sb = 'star+visp_test%d.npy' % idx
#      b = test_ind
#      np.save(sb, b)
#      idx += 1


#
# labelnames = ['Astro', 'Endo', 'L2/3 IT', 'L5 IT', 'L5 ET', 'L5/6 NP', 'L6 IT', 'L6 CT', 'L6b', 'Lamp5', 'Micro', 'Oligo', 'OPC', 'Peri', 'Pvalb',
# 'PVM', 'Sncg', 'Sst', 'Vip', 'VLMC']
# adata_seq.obs['subclass_id'] = [-1] * 13516
# adata_spatial.obs['subclass_id'] = [-1] * 5551
# for  i in labelnames:
#     id = np.where(ad_sp.obs['subclass'] == i)
#     ad_sp.obs['subclass_id'][id[0]] = labelnames.index(i)
#
#     id2 = np.where(adata_seq.obs['subclass_label'] == i)
#     adata_seq.obs['subclass_id'][id2[0]] = labelnames.index(i)
#
#
#
# a = []
# mask = adata_spatial.obs['subclass'].isin(labelnames)
# ad_sp = adata_spatial[mask].copy()


adata_spatial = sc.read_h5ad('E:/Project/Datasets/spatialscope/spscope-sp-247.h5ad')
adata_seq = sc.read_h5ad('E:/Project/Datasets/spatialscope/spscope-sc-247.h5ad')

genes = adata_seq.var_names
cells = adata_seq.obs_names

expression_matrix = adata_seq.X

df = pd.DataFrame(data=expression_matrix, index=cells, columns=genes)

df.to_csv('spscope-sc-247.csv')
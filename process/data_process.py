import numpy as np
import anndata as ad
import pandas as pd
import scanpy as sc
from process.data import reindex


adata_seq = sc.read('DataUpload/Dataset3/scRNA_count.txt', sep = '\t', first_column_names = True).T
adata_spatial = sc.read('DataUpload/Dataset3/Spatial_count.txt', sep = '\t')
locations = sc.read('DataUpload/Dataset3/Locations.txt', sep = '\t')
# RNA_data_adata = sc.read(RNA_file, sep = '\t', first_column_names = True).T
# Spatial_data_adata = sc.read(path +'\DataUpload\Dataset6\Spatial_count.txt', sep = '\t')

sc.pp.filter_genes(adata_spatial, min_cells=1)
sc.pp.filter_cells(adata_spatial, min_genes=1)
adata_seq = reindex(adata_seq, adata_spatial.var_names)
sc.pp.filter_genes(adata_seq, min_cells=1)
sc.pp.filter_cells(adata_seq, min_genes=1)
# sc.pp.filter_genes(adata_spatial, min_cells=1)

adata_seq.write('datasets/sc/dataset3_seq_170.h5ad')
adata_spatial.write('datasets/sp/dataset3_spatial_170.h5ad')

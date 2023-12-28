#!/usr/bin/env python
"""
# Authors: Shengquan Chen, Boheng Zhang, Xiaoyang Chen
# Created Time : Sat 28 Nov 2020 08:31:29 PM CST
# File Name: model.py
# Description: 

"""
import numpy as np
import pandas as pd
import pickle
import os
from .loss import *
from .dataset import *
from .utils import *
import torch
from torch import nn
import torch.nn.functional as F

reconstruction_function = nn.MSELoss(reduction='sum')

torch.set_default_tensor_type('torch.cuda.FloatTensor')

class VAE(nn.Module):
    def __init__(self, n_features):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(n_features, 1000)
        self.fc2 = nn.Linear(1000, n_features)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return h1

    def decode(self, z):
        h2 = F.relu(self.fc2(z))
        return h2

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z),z



def stPlus(spatial_df, scrna_df, genes_to_predict, save_path_prefix='./stPlus',
          top_k=2000, t_min=5, data_quality=None, random_seed=None, verbose=True, n_neighbors=50,
           converge_ratio=0.004, max_epoch_num=10000, batch_size=512, learning_rate=None, weight_decay=0.0002):
    """
    spatial_df:       [pandas dataframe] normalized and logarithmized original spatial data (cell by gene)
    scrna_df:         [pandas dataframe] normalized and logarithmized reference scRNA-seq data (cell by gene)
    genes_to_predict: [1D numpy array] spatial genes to be predicted
    save_path_prefix: [str] prefix of path of trained t models with minimal loss
    top_k:            [int] number of highly variable genes to use
    t_min:            [int] number of epochs with minimal loss using to ensemble learning
    data_quality:     [float] user-specified or 1 minus the sparsity of scRNA-seq data (default)
    random_seed:      [int] random seed in torch
    verbose:          [bool] display the running progress or not
    n_neighbors:      [int] number of neighbors used to predict
    converge_ratio:   [float] loss converge ratio
    max_epoch_num:    [int] maximum number of epochs
    batch_size:       [int] batch size for model training
    learning_rate:    [float] learning rate for model training
    weight_decay:     [float] weight decay for model training
    stPlus_res:       [pandas dataframe] predicted spatial data (cell by gene)
    """
    genes_to_predict = np.array(genes_to_predict)
    if np.sum(~np.isin(genes_to_predict, scrna_df.columns.values)) > 0:
        print('[ERROR] The following genes donot exist in the reference scRNA-seq data:')
        print(genes_to_predict[~np.isin(genes_to_predict, scrna_df.columns.values)])
        return 0
    
    if random_seed is not None:
        torch.manual_seed(random_seed)
    if verbose: 
        print('Models will be saved in: %s-%dmin*.pt\n'%(save_path_prefix, t_min))
        print('Spatial transcriptomics data: %d cells * %d genes'%(spatial_df.shape))
        print('Reference scRNA-seq data:     %d cells * %d genes'%(scrna_df.shape))
        print('%d genes to be predicted\n'%(genes_to_predict.shape[0]))
        print('Start initialization')
    shared_gene = np.intersect1d(spatial_df.columns, scrna_df.columns)
    reserved_gene = np.hstack((shared_gene,genes_to_predict))
    
    spatial_df = spatial_df[shared_gene]
    raw_scrna_uniq_gene = np.unique(scrna_df.columns.values[~np.isin(scrna_df.columns.values, reserved_gene)])
    scrna_df = scrna_df[np.hstack((reserved_gene, raw_scrna_uniq_gene))]
    
    
    spatial_df_appended = np.hstack((spatial_df.values, 
           np.zeros((spatial_df.shape[0], scrna_df.shape[1]-spatial_df.shape[1]))))
    spatial_df_appended = pd.DataFrame(data=spatial_df_appended,
                               index = spatial_df.index, columns=scrna_df.columns)
    
    t_min_loss = np.array(list(range(1,t_min+1)))*1e20
    
    # select gene
    dedup_ind = ~scrna_df.columns.duplicated()
    spatial_df_appended = spatial_df_appended.loc[:,dedup_ind]
    scrna_df = scrna_df.loc[:,dedup_ind]
    
    other_genes = np.setdiff1d(scrna_df.columns.values, reserved_gene)
    other_genes_mtx = scrna_df[other_genes].values
    selected_ind = select_top_variable_genes(other_genes_mtx, top_k)
    selected_gene = other_genes[selected_ind]
    new_genes = np.hstack((shared_gene, genes_to_predict, selected_gene))
    spatial_df_appended = spatial_df_appended[new_genes]
    scrna_df = scrna_df[new_genes]

    zero_pred_res = pd.DataFrame(np.zeros((spatial_df_appended.shape[0],genes_to_predict.shape[0])), columns=genes_to_predict)

    sorted_spatial_data_label = np.ones(spatial_df_appended.shape[0])
    sorted_scRNA_data_label = np.zeros(scrna_df.shape[0])

    train_dat = torch.from_numpy(np.vstack((spatial_df_appended, scrna_df))).float()
    train_lab = torch.from_numpy(np.hstack((sorted_spatial_data_label, sorted_scRNA_data_label))).float()

    net = VAE(train_dat.shape[1]).cuda()
    if learning_rate is None:
        learning_rate = 4e-3 if scrna_df.shape[0]<1e4 else 8e-5
        
    optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate, weight_decay=weight_decay)

    train_set = TensorsDataset(train_dat,train_lab)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=4, generator=torch.Generator(device = 'cuda'))
    val_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=False, num_workers=4,generator=torch.Generator(device = 'cuda'))
    if data_quality is None:
        data_quality = 1 - np.sum(np.sum(scrna_df==0)) / (scrna_df.shape[0]*scrna_df.shape[1])
        
    loss_last = 0
    if verbose: print('Start embedding')
    for e in range(max_epoch_num):
        train_loss = 0
        train_loss_recon = 0
        train_loss_ref = 0
        for batch_idx, (train_x, train_y) in enumerate(train_loader):
            train_x = train_x.cuda()
            is_spatial = train_y==1
            pre, mu = net(train_x)

            common_size = min(torch.sum(is_spatial), torch.sum(~is_spatial))
            loss_recon_target = loss_recon_func(pre[is_spatial,:shared_gene.shape[0]], train_x[is_spatial,:shared_gene.shape[0]])

            train_x_new = train_x[~is_spatial]
            train_x_new2 = train_x[~is_spatial]
            train_x_new2[:,shared_gene.shape[0]:] = 0
            decode_output,_ = net(train_x_new2)
            pred = decode_output[:,shared_gene.shape[0]:]
            gt = train_x_new[:,shared_gene.shape[0]:]
            loss_cor_source = loss_recon_sparsity_func(pred,gt,data_quality) * shared_gene.shape[0] / (train_x.shape[1]-shared_gene.shape[0]) * spatial_df_appended.shape[0] / scrna_df.shape[0] 

            loss = loss_recon_target + loss_cor_source

            train_loss += loss.item()
            train_loss_recon += loss_recon_target.item()
            train_loss_ref += loss_cor_source.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss = train_loss / (batch_idx + 1)
        train_loss_recon = train_loss_recon / (batch_idx + 1)
        train_loss_ref = train_loss_ref / (batch_idx + 1)
        if verbose: print('\t[{}] recon_loss: {:.3f}, pred_loss: {:.3f}, total_loss: {:.3f}'.format(e+1, train_loss_recon, train_loss_ref, train_loss))

        if e == 0:
            loss_last = train_loss
        else:
            ratio = np.abs(loss_last - train_loss) / loss_last
            loss_last = train_loss
        if e > 0 and ratio < converge_ratio: break

        if train_loss < max(t_min_loss):
            replace_ind = np.where(t_min_loss==max(t_min_loss))[0][0]
            t_min_loss[replace_ind] = train_loss
            torch.save({'epoch': e,'model_state_dict': net.state_dict(),'loss': train_loss,
                        'optimizer_state_dict': optimizer.state_dict()}, '%s-%dmin%d.pt'%(save_path_prefix,t_min,replace_ind))
    
    if verbose: print('Start prediction')
    t_min_loss_pred_mean = zero_pred_res.copy()
    t_min_cnt = 0
    for i_t_min in range(t_min):
        if os.path.exists('%s-%dmin%d.pt'%(save_path_prefix,t_min,i_t_min)):
            if verbose: print('\tUsing model %d to predict'%(i_t_min+1))
            checkpoint = torch.load('%s-%dmin%d.pt'%(save_path_prefix,t_min,i_t_min))
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            t_min_loss_pred_mean += pred_genes(net, val_loader, train_lab, scrna_df, genes_to_predict, n_neighbors)
            t_min_cnt += 1
        else:
            if verbose: print('\tSkipped model %d'%(i_t_min+1))

    stPlus_res = t_min_loss_pred_mean / t_min_cnt
    
    return stPlus_res
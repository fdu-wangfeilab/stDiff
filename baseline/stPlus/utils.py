#!/usr/bin/env python
"""
# Authors: Shengquan Chen, Boheng Zhang, Xiaoyang Chen
# Created Time : Sat 28 Nov 2020 08:31:29 PM CST
# File Name: utils.py
# Description: 

"""
import pickle
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import scipy.stats as st

def pred_genes(net, val_loader, train_lab, scRNA_data, genes_to_predict, n_neighbors=50):
    """
    net: trained model
    val_loader: DataLoader of train_set without shuffle
    train_lab: labels of train_set, 1 for spatial data while 0 for scRNA-seq data
    scRNA_data: measured scRNA-seq data
    genes_to_predict: genes to predict (1D numpy array)
    """
    net.eval()
    fm_mu = None
    for batch_idx, (x, _) in enumerate(val_loader):
        x = x.cuda()
        decode_output, mu = net(x)
        if fm_mu is None:
            fm_mu = mu.cpu().detach().numpy()
        else:
            fm_mu = np.concatenate((fm_mu,mu.cpu().detach().numpy()),axis=0)
    
    scRNA_transformed = fm_mu[train_lab!=1,:]
    spatial_transformed = fm_mu[train_lab==1,:]
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', metric = 'cosine').fit(scRNA_transformed)

    pred_res = pd.DataFrame(np.zeros((spatial_transformed.shape[0],genes_to_predict.shape[0])), columns=genes_to_predict)

    distances, indices = nbrs.kneighbors(spatial_transformed) # 50个与sp cell最近的sc cell
    for j in range(0,spatial_transformed.shape[0]): # 计算权重然后用 近邻估计结果作为预测
        weights = 1-(distances[j,:][distances[j,:]<1])/(np.sum(distances[j,:][distances[j,:]<1]))
        weights = weights/(len(weights)-1)
        pred_res.iloc[j,:] = np.dot(weights,scRNA_data[genes_to_predict].iloc[indices[j,:][distances[j,:] < 1]])
    
    net.train()
    return pred_res

def find_neighbors(net, val_loader, train_lab, scRNA_data, genes_to_predict, n_neighbors=50):
    """
    net: trained model
    val_loader: DataLoader of train_set without shuffle
    train_lab: labels of train_set, 1 for spatial data while 0 for scRNA-seq data
    scRNA_data: measured scRNA-seq data
    genes_to_predict: genes to predict (1D numpy array)
    """
    net.eval()
    fm_mu = None
    for batch_idx, (x, _) in enumerate(val_loader):
        x = x.cuda()
        decode_output, mu = net(x)
        if fm_mu is None:
            fm_mu = mu.cpu().detach().numpy()
        else:
            fm_mu = np.concatenate((fm_mu,mu.cpu().detach().numpy()),axis=0)
    
    scRNA_transformed = fm_mu[train_lab!=1,:]
    spatial_transformed = fm_mu[train_lab==1,:]
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', metric = 'cosine').fit(scRNA_transformed)

    pred_res = pd.DataFrame(np.zeros((spatial_transformed.shape[0],genes_to_predict.shape[0])), columns=genes_to_predict)

    distances, indices = nbrs.kneighbors(spatial_transformed) # 50个与sp cell最近的sc cell
    for j in range(0,spatial_transformed.shape[0]): # 计算权重然后用 近邻估计结果作为预测
        weights = 1-(distances[j,:][distances[j,:]<1])/(np.sum(distances[j,:][distances[j,:]<1]))
        weights = weights/(len(weights)-1)
        pred_res.iloc[j,:] = np.dot(weights,scRNA_data[genes_to_predict].iloc[indices[j,:][distances[j,:] < 1]])
    
    net.train()
    return pred_res, indices


def select_top_variable_genes(data_mtx, top_k):
    """
    data_mtx: data matrix (cell by gene)
    top_k: number of highly variable genes to choose
    """
    var = np.var(data_mtx, axis=0)
    ind = np.argpartition(var,-top_k)[-top_k:]
    return ind



def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)



def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        obj = pickle.load(f)
    return obj



def calc_corr(spatial_df, pred_res, test_gene):
    """
    spatial_df: original spatial data (cell by gene dataframe)
    pred_res: predicted results (cell by gene dataframe)
    test_gene: genes to calculate Spearman correlation
    """
    correlation = []
    for gene in test_gene:
        correlation.append(st.spearmanr(spatial_df[gene], pred_res[gene])[0])
    return correlation
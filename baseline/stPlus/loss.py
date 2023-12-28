#!/usr/bin/env python
"""
# Authors: Shengquan Chen, Boheng Zhang, Xiaoyang Chen
# Created Time : Sat 28 Nov 2020 08:31:29 PM CST
# File Name: loss.py
# Description: 

"""
from torch import nn
reconstruction_function = nn.MSELoss(reduction='sum')

def loss_recon_func(recon_x, origi_x):
    """
    recon_x: reconstructed data
    origi_x: original data
    """
    MSE = reconstruction_function(recon_x, origi_x)
    return MSE



def loss_recon_sparsity_func(recon_x, origi_x, data_quality):
    """
    recon_x: reconstructed data
    origi_x: original data
    data_quality: user-specified or 1 minus the sparsity of scRNA-seq data (default)
    """
    zero_ind = origi_x==0
    non_zero_ind = ~zero_ind
    recon_x_0 = recon_x[zero_ind]
    recon_x_1 = recon_x[non_zero_ind]
    origi_x_0 = origi_x[zero_ind]
    origi_x_1 = origi_x[non_zero_ind]
    MSE_0 = reconstruction_function(recon_x_0, origi_x_0)
    MSE_1 = reconstruction_function(recon_x_1, origi_x_1)
    MSE = data_quality * MSE_0 + MSE_1
    return MSE
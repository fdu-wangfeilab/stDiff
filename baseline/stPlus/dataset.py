#!/usr/bin/env python
"""
# Authors: Shengquan Chen, Boheng Zhang, Xiaoyang Chen
# Created Time : Sat 28 Nov 2020 08:31:29 PM CST
# File Name: dataset.py
# Description: 

"""
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader, TensorDataset

class TensorsDataset(TensorDataset):    
    def __init__(self, data, target=None, transforms=None, target_transforms=None):
        if target is not None:
            assert data.size(0) == target.size(0) 
        self.data = data
        self.target = target
        if transforms is None:                    
            transforms = []
        if target_transforms is None:         
            target_transforms = []
        if not isinstance(transforms, list):              
            transforms = [transforms]
        if not isinstance(target_transforms, list):
            target_transforms = [target_transforms]
        self.transforms = transforms
        self.target_transforms = target_transforms

    def __getitem__(self, index):                    
        data = self.data[index]             
        for transform in self.transforms:               
            data = transform(data)
        if self.target is None:                    
            return data
        target = self.target[index]                   
        for transform in self.target_transforms:           
            target = transforms.ToTensor(transform(transforms.ToPILImage(target)))
        return (data, target)

    def __len__(self):                           
        return self.data.size(0)
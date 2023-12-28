import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from collections import defaultdict
from torch.distributions import Normal, kl_divergence

import ray
from ray import tune
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

import sys
sys.path.append('./')
from model.VAE.VAE_model import VAE
from model.process_h5ad import get_processed_data_ary,get_data_loader

def kl_div(mu, var):
    return kl_divergence(Normal(mu, var.sqrt()),
                         Normal(torch.zeros_like(mu),torch.ones_like(var))).sum(dim=1).mean()
    
    
def vae_train(vae,data_ary, cell_type_ary,device = torch.device('cuda:0')):
    
    dataloader = get_data_loader(data_ary,
                             cell_type=cell_type_ary,
                             batch_size=64,
                             is_shuffle=True)
    
    # optimizer = torch.optim.SGD(
    #         vae.parameters(),
    #         lr=parameters['lr'],
    #         momentum=parameters['momentum']
    #     )
    optimizer = torch.optim.Adam(vae.parameters(), lr=0.0002, weight_decay=5e-4)
    vae.to(device=device)
    
    max_iter = 30000
    num_epoch = max_iter // len(dataloader)
    tq = tqdm(range(num_epoch), ncols=80)
    for epoch in tq:
        epoch_loss = defaultdict(float)
        for i, (x,y) in enumerate(dataloader):
            x,y = x.float().to(device),y.long().to(device)
            
            z, mu, var = vae.encoder(x)
            recon_x = vae.decoer(z,y)
            
            recon_loss = F.binary_cross_entropy(recon_x,x) * x.size(-1)
            
            kl_loss = kl_div(mu, var)
            loss = {'recon_loss':recon_loss, 'kl_loss':0.5*kl_loss} 
            optimizer.zero_grad()
            sum(loss.values()).backward()
            optimizer.step()
            
            for k,v in loss.items():
                epoch_loss[k] += loss[k].item()
                
        epoch_loss = {k:v/(i+1) for k, v in epoch_loss.items()}
        epoch_info = ','.join(['{}={:.3f}'.format(k, v) for k,v in epoch_loss.items()])
        tq.set_postfix_str(epoch_info)
    
def vae_tune(config):
    vae = VAE(input_dim=2000, 
          hidden_dim=1024,
          mu_var_dim=10)
    
    adata, data_ary, cell_type_ary = get_processed_data_ary(
        adata_path = '/home/lijiahao/Pythoncode/sc-diffusion/data/processed_pbmc_ATAC.h5ad',
        data_ary_path = '/home/lijiahao/Pythoncode/sc-diffusion/data/npy_ary/data_ary.npy',
        cell_type_ary_path = '/home/lijiahao/Pythoncode/sc-diffusion/data/npy_ary/cell_type_ary.npy'
    )
    dataloader = get_data_loader(data_ary,
                             cell_type=cell_type_ary,
                             batch_size=64,
                             is_shuffle=True)
    
    optimizer = torch.optim.SGD(
            vae.parameters(),
            lr=config['lr'],
            momentum=config['momentum']
        )
    device = torch.device('cuda:0')
    vae.to(device=device)
    
    max_iter = 6000
    num_epoch = max_iter // len(dataloader)
    # tq = tqdm(range(num_epoch), ncols=80)
    for epoch in range(num_epoch):
        epoch_loss = defaultdict(float)
        for i, (x,y) in enumerate(dataloader):
            x,y = x.float().to(device),y.long().to(device)
            
            z, mu, var = vae.encoder(x)
            recon_x = vae.decoer(z,y)
            
            recon_loss = F.binary_cross_entropy(recon_x,x) * x.size(-1)
            
            kl_loss = kl_div(mu, var)
            loss = {'recon_loss':recon_loss, 'kl_loss':0.5*kl_loss} 
            optimizer.zero_grad()
            sum(loss.values()).backward()
            optimizer.step()
            
            for k,v in loss.items():
                epoch_loss[k] += loss[k].item()
                
        epoch_loss = {k:v/(i+1) for k, v in epoch_loss.items()}
        epoch_info = ','.join(['{}={:.3f}'.format(k, v) for k,v in epoch_loss.items()])
        session.report({'loss': epoch_loss['recon_loss']})
        # tq.set_postfix_str(epoch_info)


def ray_tune():
    ray.init(num_cpus=24, num_gpus=3)
    seed = 1202
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    search_space = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "momentum": tune.uniform(0.1, 0.9)
    }
    
    # early stop
    max_num_epochs = 50
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    
    optuna_search = OptunaSearch()
    
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(vae_tune),
            resources={"gpu": 0.5}
        ),
        tune_config=tune.TuneConfig(
            search_alg=optuna_search,
            scheduler=scheduler,
            num_samples=20,
            metric='loss',
            mode='min'
        ),
        param_space=search_space
    )
    
    results = tuner.fit()
        
if __name__ == '__main__':
    ray_tune()
        
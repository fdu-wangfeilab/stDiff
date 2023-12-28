import torch
import numpy as np
import os
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

import ray
from ray import tune
from ray.air import session
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
import sys
sys.path.append('./')
from Extenrnal.sc_DM.model.process_h5ad import get_data_ary,get_data_loader,get_data_msk_loader
from Extenrnal.sc_DM.model.diffusion_model import MLP,TransformerEncoder,CrossTransformer
from Extenrnal.sc_DM.model.diffusion_scheduler import NoiseScheduler

import os


def normal_train(model, 
                 lr:float, 
                 momentum:float,
                 max_iteration:int=30000,
                 pred_type:str='noise',
                 data_type:str='atac',
                 batch_size:int=1024,
                 diffusion_step:int=1000,
                 device=torch.device('cuda:0'),
                 is_nonzero_msk:bool=False,
                 is_class_condi:bool=False,
                 is_tqdm:bool=True,
                 is_tune:bool=False,
                 condi_drop_rate:float=0.):
    """通用训练函数

    Args:
        lr (float): 
        momentum (float): 动量
        max_iteration (int, optional): 训练的 iteration. Defaults to 30000.
        pred_type (str, optional): 预测的类型噪声或者 x_0. Defaults to 'noise'.
        batch_size (int, optional):  Defaults to 1024.
        diffusion_step (int, optional): 扩散步数. Defaults to 1000.
        device (_type_, optional): Defaults to torch.device('cuda:0').
        is_class_condi (bool, optional): 是否采用condition. Defaults to False.
        is_tqdm (bool, optional): 开启进度条. Defaults to True.
        is_tune (bool, optional): 是否用 ray tune. Defaults to False.
        condi_drop_rate (float, optional): 是否采用 classifier free guidance 设置 drop rate. Defaults to 0..

    Raises:
        NotImplementedError: _description_
    """
    pwd = '/home/lijiahao/projects/sc-diffusion/'
    
    # adata, data_ary, cell_type = get_data_ary(pwd + 'data/pbmc_ATAC.h5ad')
    
    # data_ary = np.load(pwd + f'data/npy_ary/{data_type}_data_ary.npy') * 2 - 1
    data_ary = np.load(pwd + f'data/npy_ary/{data_type}_data_ary.npy') * 2 - 1
    print(f'type:{data_type} condi:{is_class_condi} data_min:{data_ary.min()} data_max:{data_ary.max()}')
    cell_type = np.load(pwd + f'data/npy_ary/{data_type}_celltype_ary.npy')
    
    if is_nonzero_msk:
        nonzero_msk = np.load(pwd +  f'data/npy_ary/{data_type}_latent_ary.npy')
        dataloader = get_data_msk_loader(
            data_ary=data_ary,
            cell_type=cell_type,
            non_zero_msk=nonzero_msk,
            batch_size=batch_size
        )
    else:
        dataloader = get_data_loader(
            data_ary,
            cell_type=cell_type,
            batch_size= batch_size )
    
    noise_scheduler = NoiseScheduler(
        num_timesteps=diffusion_step,
        beta_schedule='cosine'
    )
    
    criterion = nn.MSELoss()
    model.to(device)
    num_epoch = max_iteration // len(dataloader)
    
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum
    )
    
    
    if  is_tqdm:
        t_epoch = tqdm(range(num_epoch), ncols=100)
    else:
        t_epoch = range(num_epoch)
    
    model.train()
    
    for epoch in t_epoch:
        epoch_loss = 0.
        for i, data in enumerate(dataloader):
            if is_nonzero_msk:
                x,cell_type,msk = data
                x,cell_type,msk  = x.to(device), cell_type.to(device).long(), msk.to(device).float()
            else:
                x,cell_type = data
                x,cell_type,msk  = x.to(device), cell_type.to(device).long(), None
                
            noise = torch.randn(x.shape).to(device)
            timesteps = torch.randint(1,diffusion_step, (cell_type.shape[0],)).long().to(device)
            x_t = noise_scheduler.add_noise(x, noise, timesteps=timesteps)
            
            if not is_class_condi:
                cell_type = None
                
            if pred_type == 'noise':
                condi_preserve_flag = torch.rand(1) >= condi_drop_rate
                noise_pred = model(x_t, timesteps, cell_type, condi_flag=condi_preserve_flag, msk=msk)
                loss = criterion(noise_pred, noise)
            elif pred_type == 'x_start':
                x_0_pred = model(x_t, timesteps, cell_type)
                loss = criterion(x_0_pred, x)
            else:
                raise NotImplementedError
                
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0) # type: ignore
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            
        epoch_loss = epoch_loss/(i+1) # type: ignore
        if is_tqdm:
            t_epoch.set_postfix_str(f'{pred_type} loss:{epoch_loss:.5f}') # type: ignore
        if is_tune:
            session.report({'loss': epoch_loss})

def train_scripts(device = torch.device('cuda:0'),
                  model_name:str = 'cosine_condi_ln_mlp.pth',
                  pred_type:str='noise',
                  max_iteration:int=30000):
    seed = 1202
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = TransformerEncoder(
        input_dim=2000,
        emb_dim=4000,
        num_classes=13,
        is_learned_timeebd=True,
        is_time_concat=True,
        is_condi_concat=True,
        is_msk_emb=False,
        num_heads=20,
        attn_types=['mlp','mlp']
    )
    torch.save(model.state_dict(), f'./model/ckpt/{model_name}')
    model.to(device)
    # parameters={'lr': 0.09930906002366541, 'momentum': 0.8865807468993898,'batch_size': 1024}
    # parameters={'lr': 0.05499951781790261, 'momentum': 0.4754472160656253,'batch_size': 1024}
    # parameters={'lr': 0.09972768793470284, 'momentum': 0.8865217158805158, 'batch_size': 2048}
    # rna 
    parameters={'lr': 0.09886909388175943, 'momentum': 0.7779814458926817, 'batch_size': 1024}
    normal_train(model,
             max_iteration=max_iteration,
             diffusion_step=1000,
             batch_size=parameters['batch_size'],
             lr=parameters['lr'],
             momentum=parameters['momentum'],
             device=device,
             pred_type='noise',
             data_type='atac',
             is_class_condi=True,
             is_nonzero_msk=False,
             condi_drop_rate=0)
    torch.save(model.state_dict(), f'./model/ckpt/{model_name}')


def diffusion_train(config):
    
    # model = TransformerEncoder(
    #     input_dim=13928,
    #     num_classes=14,
    #     emb_dim=512,
    #     is_learned_timeebd=True,
    #     is_time_concat=True,
    #     is_condi_concat=True,
    #     num_heads=20,
    #     attn_types=['mlp','mlp']
    # )
    model = CrossTransformer(
        input_dim=2000,
        emb_dim=4000,
        num_classes=13,
        is_learned_timeebd=True,
        is_time_concat=True,
        is_condi_concat=True,
        is_msk_emb=True,
        num_heads=20,
        attn_types=['self',]
    )
    pred_type = 'noise'
    device = torch.device('cuda:0')
    
    normal_train(model, 
                 lr=config['lr'],
                 momentum=config['momentum'],
                 max_iteration=6000,
                 pred_type=pred_type,
                 data_type='atac',
                 batch_size=config['batch_size'],
                 device=device, 
                 is_class_condi=True,
                 is_tqdm=False,
                 is_nonzero_msk=True,
                 is_tune=True)
        


def ray_tune():
    ray.init(num_cpus=24, num_gpus=1)
    seed = 1202
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    search_space = {
        "lr": tune.loguniform(1e-3, 1e-1),
        "momentum": tune.uniform(0.1, 0.9),
        "batch_size": tune.choice([1024])
    }

    max_num_epochs = 100
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)

    optuna_search = OptunaSearch()


    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(diffusion_train),
            resources={"gpu": 0.2}
        ),
        tune_config=tune.TuneConfig(
            search_alg=optuna_search,
            scheduler=scheduler,
            num_samples=20,
            metric='loss',
            mode='min'
        ),
        param_space=search_space,
    )
    
    results = tuner.fit()



if __name__ =='__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # os.environ["RAY_SESSION_DIR"] = "/home/lijiahao/ray_session"
    # ray_tune()
    train_scripts(model_name='scale2_self2_1024_latent.pt',
                  device = torch.device('cuda:0'),
                  pred_type='noise',
                  max_iteration=5000)
    # model = TransformerEncoder(
    #     input_dim=2000,
    #     num_classes=13,
    #     is_learned_timeebd=True,
    #     is_time_concat=True,
    #     is_condi_concat=True,
    #     num_heads=20,
    #     attn_types=['mlp','mlp']
    # )
    # device = torch.device('cuda:0')
    # model.to(device)
    # model.load_state_dict(torch.load('./model/ckpt/scale2_mlp2_1024.pt'))
    # print(model)
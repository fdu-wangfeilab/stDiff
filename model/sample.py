import torch
from tqdm import tqdm
import numpy as np

def model_sample(model, 
                 device, 
                 dataloader, 
                 total_sample, 
                 time=250, 
                 is_condi=True,
                 condi_flag=True,
                 is_nonzero_msk=False):
    # 这里是从 x_t 恢复到 x_t-1 其实输出的就是 x_t-1
    noise = []
    i = 0
    for condition in dataloader:
        if is_nonzero_msk:
            _, cell_type, msk = condition
            cell_type,msk= cell_type.long().to(device),msk.float().to(device)
        else:
            _, cell_type = condition
            cell_type = cell_type.long().to(device)
            msk = None
            
        t = torch.from_numpy(np.repeat(time, cell_type.shape[0])).long().to(device)
        if not is_condi:
            n = model(total_sample[i:i+len(cell_type)], t, None, msk=msk)
        else:
            n = model(total_sample[i:i+len(cell_type)], t, cell_type, msk=msk, condi_flag=condi_flag)
        noise.append(n) 
        i = i+len(cell_type)
    noise = torch.cat(noise, dim=0)
    return noise

def sample_loop(model, 
                dataloader, 
                noise_scheduler, 
                device=torch.device('cuda:0'),
                num_step=1000, 
                sample_shape=(7060, 2000), 
                is_condi=False,
                sample_intermediate=200,
                model_pred_type:str='noise',
                is_classifier_guidance=False,
                is_nonzero_msk=False,
                omega=0.1):
    model.eval()
    # 初始化噪声
    sample = torch.randn(sample_shape[0],sample_shape[1]).to(device)
    timesteps = list(range(num_step))[::-1]
    
    # 是否只采样前 sample_intermediate 步
    if sample_intermediate:
        timesteps = timesteps[:sample_intermediate]
        
    ts = tqdm(timesteps)
    for t_idx,time in enumerate(ts):
        ts.set_description_str(desc=f'time: {time}')
        with torch.no_grad():
            model_output = model_sample(model,
                                        device=device, 
                                        dataloader=dataloader, 
                                        total_sample=sample, 
                                        time=time, 
                                        is_condi=is_condi,
                                        condi_flag=True,
                                        is_nonzero_msk=is_nonzero_msk)
            if is_classifier_guidance:
                # 如果采用 classifier free guidance 的话多预测一个没有 condition 的版本
                model_output_uncondi = model_sample(model,
                                        device=device, 
                                        dataloader=dataloader, 
                                        total_sample=sample, 
                                        time=time, 
                                        is_condi=is_condi,
                                        condi_flag=False,
                                        is_nonzero_msk=is_nonzero_msk)
                model_output = (1+omega)* model_output - omega * model_output_uncondi
           
        sample = noise_scheduler.step(model_output, 
                                      torch.from_numpy(np.array(time)).long().to(device), 
                                      sample,
                                      model_pred_type=model_pred_type)
        
        if time==0 and model_pred_type=='x_start':
            # 如果直接预测 x_0 的话，最后一步直接输出
            sample = model_output
            
    recon_x = sample.detach().cpu().numpy()
    return recon_x
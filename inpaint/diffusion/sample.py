import torch
from tqdm import tqdm
import numpy as np
from collections import defaultdict

from .diffusion_scheduler import get_schedule_jump

def model_sample_palette(model, device, dataloader, total_sample, time, is_condi, condi_flag):
    noise = []
    i = 0
    for _, x_cond in dataloader: # 计算整个shape得噪声 一次循环算batch大小  加上了celltype 去掉了, celltype
        x_cond = x_cond.float().to(device) # x.float().to(device)
        t = torch.from_numpy(np.repeat(time, x_cond.shape[0])).long().to(device)
        # celltype = celltype.to(device)
        if not is_condi:
            n = model(total_sample[i:i+len(x_cond)], t, None) # 一次计算batch大小得噪声
        else:
            n = model(total_sample[i:i+len(x_cond)], t, x_cond, condi_flag=condi_flag) # 加上了celltype 去掉了, celltype
        noise.append(n)
        i = i+len(x_cond)
    noise = torch.cat(noise, dim=0)
    return noise

def model_sample(model,
                 device, 
                 dataloader, 
                 total_sample,  # 大小是shape决定得
                 time = 250, 
                 is_condi=True,
                 condi_flag=True):
    noise = []
    i = 0
    for _, cell_type in dataloader: # 计算整个shape得噪声 一次循环算batch大小
        cell_type = cell_type.long().to(device)
        t = torch.from_numpy(np.repeat(time, cell_type.shape[0])).long().to(device)
        if not is_condi:
            n = model(total_sample[i:i+len(cell_type)], t, None) # 一次计算batch大小得噪声
        else:
            n = model(total_sample[i:i+len(cell_type)], t, cell_type, condi_flag=condi_flag)
        noise.append(n)
        i = i+len(cell_type)
    noise = torch.cat(noise, dim=0)
    return noise

def model_sample_no_guidance(model,
                 device,
                 dataloader,
                 total_sample,  # 大小是shape决定得
                 time = 250,
                 is_condi=True,
                 condi_flag=True):
    noise = []
    i = 0
    for x in dataloader: # 计算整个shape得噪声 一次循环算batch大小
        x = torch.stack(x, dim=1).flatten(1)
        x = x.to(device)
        t = torch.from_numpy(np.repeat(time, x.shape[0])).long().to(device)
        if not is_condi:
            n = model(total_sample[i:i+len(x)], t, None) # 一次计算batch大小得噪声
        else:
            n = model(total_sample[i:i+len(x)], t, condi_flag=condi_flag)
        noise.append(n)
        i = i+len(x)
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
                omega=0.1):
    model.eval()
    sample = torch.randn(sample_shape[0],sample_shape[1]).to(device)
    timesteps = list(range(num_step))[::-1] # 倒序
    
    # 是否只采样前 sample_intermediate 步
    if sample_intermediate:
        timesteps = timesteps[:sample_intermediate]
        
    ts = tqdm(timesteps)
    for t_idx,time in enumerate(ts):
        ts.set_description_str(desc=f'time: {time}')
        with torch.no_grad():
            # 输出噪声
            model_output = model_sample(model,
                                        device=device, 
                                        dataloader=dataloader, 
                                        total_sample=sample,  # x_t
                                        time=time, # t
                                        is_condi=is_condi,
                                        condi_flag=True)
            if is_classifier_guidance:
                model_output_uncondi = model_sample(model,
                                        device=device, 
                                        dataloader=dataloader, 
                                        total_sample=sample, 
                                        time=time, 
                                        is_condi=is_condi,
                                        condi_flag=False)
                model_output = (1+omega)* model_output - omega * model_output_uncondi

        # 计算x_{t-1}
        sample, _ = noise_scheduler.step(model_output, # 一般是噪声
                                      torch.from_numpy(np.array(time)).long().to(device), 
                                      sample,
                                      model_pred_type=model_pred_type)
        
        if time==0 and model_pred_type=='x_start':
            # 如果直接预测 x_0 的话，最后一步直接输出
            sample = model_output
            
    recon_x = sample.detach().cpu().numpy()
    return recon_x


def sample_loop_resample(
    model,
    noise=None,
    mask=None,
    gt=None,
    cond_fn=None,
    device=torch.device('cuda:0'),
    progress=False,
    conf=None,
    dataloader = None,
    noise_scheduler = None,
    num_step=1000,
    sample_shape=(2955, 32), # 这里是batch gene吗
    is_condi=False,
    sample_intermediate=200,
    model_pred_type:str='noise',
    is_classifier_guidance=False,
    omega=0.1
):
    """
    Generate samples from the model and yield intermediate samples from
    each timestep of diffusion.

    Arguments are the same as p_sample_loop().
    Returns a generator over dicts, where each dict is the return value of
    p_sample() 预测x_{t-1}.
    """
    if device is None:
        device = next(model.parameters()).device
    assert isinstance(sample_shape, (tuple, list))
    if noise is not None:
        x_after_step = noise
    else:
        x_after_step = torch.randn(sample_shape[0],sample_shape[1]).to(device)

    # debug_steps = conf.pget('debug.num_timesteps')

    gt_noises = None  # reset for next image


    pred_xstart = None

    idx_wall = -1
    sample_idxs = defaultdict(lambda: 0)

    times = get_schedule_jump(t_T=250, n_sample=1, jump_length=10, jump_n_sample=10) # 获得时间步，已经计算了重采样 \
    # 比如 249，248，249，248，247

    time_pairs = list(zip(times[:-1], times[1:]))  # 按对应顺序打包成元组再解压为元组列表
    if progress:
        from tqdm.auto import tqdm # tqdm是进度条
        time_pairs = tqdm(time_pairs)

    for t_last, t_cur in time_pairs:
        idx_wall += 1
        t_last_t = torch.tensor([t_last] * sample_shape[0],  # pylint: disable=not-callable
                             device=device)

        if t_cur < t_last:  # reverse 逆向过程去噪 主要是 函数p_sample() 得x_{t-1}
            with torch.no_grad():
                x_before_step = x_after_step.clone()

                model_output = model_sample(model,
                                            device=device,
                                            dataloader=dataloader,
                                            total_sample=x_after_step,  # x_t（所有细胞）
                                            time=t_last_t,  # t
                                            is_condi=is_condi,
                                            condi_flag=True)
                if is_classifier_guidance:
                    model_output_uncondi = model_sample(model,
                                                        device=device,
                                                        dataloader=dataloader,
                                                        total_sample=x_after_step,
                                                        time=t_last_t,
                                                        is_condi=is_condi,
                                                        condi_flag=False)
                    model_output = (1 + omega) * model_output - omega * model_output_uncondi

                # out = self.p_sample(
                #     model,
                #     x_after_step, #  纯噪声
                #     t_last_t,
                #     clip_denoised=clip_denoised,
                #     denoised_fn=denoised_fn,
                #     cond_fn=cond_fn,
                #     model_kwargs=model_kwargs,
                #     conf=conf,
                #     pred_xstart=pred_xstart
                # )

            # 计算x_{t-1}
            sample, pred_xstart = noise_scheduler.step(model_output,  # 一般是噪声
                                          torch.from_numpy(np.array(t_last_t)).long().to(device),
                                          sample,
                                          mask,
                                          gt,
                                          model_pred_type=model_pred_type)

            if t_last_t == 0 and model_pred_type == 'x_start':
                # 如果直接预测 x_0 的话，最后一步直接输出
                sample = model_output

            x_after_step = sample # x_{t-1}
            # pred_xstart = out["pred_xstart"] # x0'

            sample_idxs[t_cur] += 1

        else: # 加噪
            t_shift = conf.get('inpa_inj_time_shift', 1)

            x_before_step = x_after_step.clone()
            x_after_step = noise_scheduler.undo( # 加噪
                x_before_step, x_after_step,
                est_x_0=pred_xstart, t=t_last_t+t_shift, debug=False)
            pred_xstart = pred_xstart

    recon_x = sample.detach().cpu().numpy()
    return recon_x





def sample_loop_palette(model,
                dataloader,
                noise_scheduler,
                mask = None,
                gt = None,
                device=torch.device('cuda:0'),
                num_step=1000,
                sample_shape=(7060, 2000),
                is_condi=False,
                sample_intermediate=200,
                model_pred_type: str = 'noise',
                is_classifier_guidance=False,
                omega=0.1):
    model.eval()
    x_t = torch.randn(sample_shape[0], sample_shape[1]).to(device)
    timesteps = list(range(num_step))[::-1]  # 倒序
    mask = torch.tensor(mask).to(device)
    gt = torch.tensor(gt).to(device)
    x_t =  x_t * (1 - mask) + gt * mask

    # 是否只采样前 sample_intermediate 步
    if sample_intermediate:
        timesteps = timesteps[:sample_intermediate]

    ts = tqdm(timesteps)
    for t_idx, time in enumerate(ts):
        ts.set_description_str(desc=f'time: {time}')
        with torch.no_grad():
            # 输出噪声
            model_output = model_sample_palette(model,
                                        device=device,
                                        dataloader=dataloader,
                                        total_sample=x_t,  # x_t
                                        time=time,  # t
                                        is_condi=is_condi,
                                        condi_flag=True)
            if is_classifier_guidance:
                model_output_uncondi = model_sample(model,
                                                    device=device,
                                                    dataloader=dataloader,
                                                    total_sample=sample,
                                                    time=time,
                                                    is_condi=is_condi,
                                                    condi_flag=False)
                model_output = (1 + omega) * model_output - omega * model_output_uncondi

        # 计算x_{t-1}
        x_t, _ = noise_scheduler.step(model_output,  # 一般是噪声
                                         torch.from_numpy(np.array(time)).long().to(device),
                                         x_t,
                                         model_pred_type=model_pred_type)

        if mask is not None:
            x_t = x_t * (1. - mask) + mask * gt  # 真实值和预测部分的拼接

        if time == 0 and model_pred_type == 'x_start':
            # 如果直接预测 x_0 的话，最后一步直接输出
            sample = model_output

    recon_x = x_t.detach().cpu().numpy()
    return recon_x


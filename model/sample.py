import torch
from tqdm import tqdm
import numpy as np
from collections import defaultdict


def model_sample_stDiff(model, device, dataloader, total_sample, time, is_condi, condi_flag):
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

def sample_stDiff(model,
                dataloader,
                noise_scheduler,
                mask = None,
                gt = None,
                device=torch.device('cuda:1'),
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
            model_output = model_sample_stDiff(model,
                                        device=device,
                                        dataloader=dataloader,
                                        total_sample=x_t,  # x_t
                                        time=time,  # t
                                        is_condi=is_condi,
                                        condi_flag=True)
            if is_classifier_guidance:
                model_output_uncondi = model_sample_stDiff(model,
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


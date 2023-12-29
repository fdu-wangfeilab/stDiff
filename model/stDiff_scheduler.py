import torch
import matplotlib.pyplot as plt
from torch.nn import functional as F
import numpy as np
import math

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """ beta schedule
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class NoiseScheduler():
    def __init__(self,
                 num_timesteps=1000,
                 beta_start=0.0001,
                 beta_end=0.02,
                 beta_schedule="linear"):
        # 总的前向 diffusion step
        self.num_timesteps = num_timesteps
        
        if beta_schedule == "linear":
            self.betas = torch.linspace(
                beta_start, beta_end, num_timesteps, dtype=torch.float32)
        elif beta_schedule == "quadratic":
            self.betas = torch.linspace(
                beta_start ** 0.5, beta_end ** 0.5, num_timesteps, dtype=torch.float32) ** 2
        elif beta_schedule == 'cosine':
            self.betas = torch.from_numpy(betas_for_alpha_bar(num_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,).astype(np.float32))
        
        # 计算出 alpha，此处是后续跟 noise schedule 相关的一系列计算
        self.alphas = 1.0 - self.betas
        # alphas_cumprod 的每一项都是前i项 alpha 的连乘，是后面一系列变量的基础
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0) # type: ignore
        # 此处的 pad 就相当于在最前面拼了个 1 上去
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.)

        # required for self.add_noise
        self.sqrt_alphas_cumprod = self.alphas_cumprod ** 0.5
        
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod) ** 0.5

        # required for reconstruct_x0
        self.sqrt_inv_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod)
        self.sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(
            1 / self.alphas_cumprod - 1)

        # required for q_posterior
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)

    def reconstruct_x0(self, x_t, t, noise):
        # 输入 x_t,t,noise 来得到 x0, iDDPM 公式9
        s1 = self.sqrt_inv_alphas_cumprod[t]
        s2 = self.sqrt_inv_alphas_cumprod_minus_one[t]
        s1 = s1.reshape(-1, 1).to(x_t.device)
        s2 = s2.reshape(-1, 1).to(x_t.device)
        # 通过两个系数来得到 x0
        x0 = s1 * x_t - s2 * noise
        return torch.clamp(x0,min=-1,max=1)

    def q_posterior(self, x_0, x_t, t):
        # 通过 x_0,x_t,t 来得到 x_t-1 的均值，iDDPM 公式11
        s1 = self.posterior_mean_coef1[t]
        s2 = self.posterior_mean_coef2[t]
        
        s1 = s1.reshape(-1, 1).to(x_t.device)
        s2 = s2.reshape(-1, 1).to(x_t.device)
        mu = s1 * x_0 + s2 * x_t
        return mu

    def get_variance(self, t):
        # 这里应该是默认为固定方差，iDDPM 公式10
        if t == 0:
            return 0

        variance = self.betas[t] * (1. - self.alphas_cumprod_prev[t]) / (1. - self.alphas_cumprod[t])
        # 此处的 tensor 的 clip 是将 var 最小限制在 1e-20
        variance = variance.clip(1e-20)
        return variance.to(t.device)
    
    # 逆扩散的一步
    def step(self, 
             model_output, 
             timestep, 
             sample,
             model_pred_type: str='noise'):
        
        t = timestep
        # 用模型预测出的数值作为 noise 
        
        # 再用 x_0_pred，输入的 x_t，t 来得到均值
        if model_pred_type=='noise':
            pred_original_sample = self.reconstruct_x0(sample, t, model_output)
        elif model_pred_type=='x_start':
            pred_original_sample = model_output
        else:
            raise NotImplementedError()
            
        pred_prev_sample = self.q_posterior(pred_original_sample, sample, t)  # 得到x_t-1的均值mu
        # 这里采用公式造一个方差出来，用到noise作为重采样
        variance = 0
        if t > 0:
            noise = torch.randn_like(model_output)
            # 此时已经把噪声给加上去了
            variance = (self.get_variance(t) ** 0.5) * noise

        # 把均值和方差加起来，然后作为返回值
        pred_prev_sample = pred_prev_sample + variance  # 重参数化技巧得到x_t-1
        
        return pred_prev_sample  ,pred_original_sample  # imputation时候需要后面的

    def add_noise(self, x_start, x_noise, timesteps):  # 正向加噪的过程
        # 输入 x_0,noise,t 来得到 x_t
        s1 = self.sqrt_alphas_cumprod[timesteps]
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps]

        s1 = s1.reshape(-1, 1).to(x_start.device)
        s2 = s2.reshape(-1, 1).to(x_start.device)
        return s1 * x_start + s2 * x_noise

    def undo(self, image_before_step, img_after_model, est_x_0, t, debug=False):
        # 对x_t加噪
        return self._undo(img_after_model, t)

    def _undo(self, img_out, t):
        beta = self.betas[t]

        img_in_est = torch.sqrt(1 - beta) * img_out + \
                     torch.sqrt(beta) * torch.randn_like(img_out)

        return img_in_est

    def __len__(self):
        return self.num_timesteps

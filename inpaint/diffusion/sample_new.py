from collections import defaultdict

from .diffusion_scheduler import get_schedule_jump
# from model.sample import * # 有condition的时候
from Extenrnal.sc_DM.inpaint.diffusion.sample import *


def model_sample_repaint(
        model,
        device,
        dataloader,
        sample,
        time = 250,
        is_condi = False,
        condi_flag = True,
        mask = None,
        gt = None,
        noise_scheduler = None,
        model_pred_type:str='noise',
        is_classifier_guidance=False,
        omega = 0.1,
        pred_xstart = None):

    noise = torch.randn_like(sample)

    if pred_xstart is not None:
        # 获取mask的形状 和 原输入x0
        gt_keep_mask = torch.tensor(mask).to(device)
        gt = gt
        alpha_cumprod = noise_scheduler.alphas_cumprod[time] # 会有问题吗

        # 这里就是x0加噪至x_t
        gt_weight = torch.sqrt(alpha_cumprod)
        gt_part = gt_weight * gt

        noise_weight = torch.sqrt((1 - alpha_cumprod))  # 计算输入图像已知应该加入噪声的数目 以能够和\
        # 扩散模型生成的mask部分的噪声大小一致 然后相加
        noise_part = noise_weight * torch.randn_like(sample)

        weighed_gt = gt_part.to(device) + noise_part.to(device)  # 将计算出噪声加入到已知部分 加噪后得图片

        sample = (  # 将已知部分和预测部分 两部分相加
                gt_keep_mask * (
            weighed_gt
        )
                +
                (1 - gt_keep_mask) * (
                    sample
                )
        )
    # 计算噪声 no guidance 用model_sample_no_guidance
    model_output = model_sample_no_guidance(model,
                                device=device,
                                dataloader=dataloader,
                                total_sample=sample,  # x_t
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

    # 计算x_{t-1} x0'
    sample, pred_xstart = noise_scheduler.step(model_output,  # 一般是噪声
                                  torch.from_numpy(np.array(time)).long().to(device),
                                  sample,
                                  model_pred_type=model_pred_type)

    # if time == 0 and model_pred_type == 'x_start':
    #     # 如果直接预测 x_0 的话，最后一步直接输出
    #     sample = model_output
    # recon_x = sample.detach().cpu().numpy()
    result = {'sample':sample,'pred_xstart':pred_xstart}
    return result

def sample_loop_progressive(
        model,
        noise=None,
        mask=None,
        gt=None,
        cond_fn=None,
        device=torch.device('cuda:0'),
        progress=False,
        conf=None,
        dataloader=None,
        noise_scheduler=None,
        num_step=1000,
        sample_shape=(2955, 32),  # 这里是batch gene吗 是 实际这里用了全部的cell 恢复全部
        is_condi=False,
        sample_intermediate=200,
        model_pred_type: str = 'noise',
        is_classifier_guidance=False,
        omega=0.1
):
    """
    产生结果的generator
    包含了resample的过程
    model_sample_repaint 预测x_{t-1}.
    """
    if device is None:
        device = next(model.parameters()).device
    assert isinstance(sample_shape, (tuple, list))
    if noise is not None:
        x_after_step = noise
    else:
        x_after_step = torch.randn(sample_shape[0], sample_shape[1]).to(device) # [2955, 32]的噪声


    gt_noises = None  # reset for next image

    pred_xstart = None

    idx_wall = -1
    sample_idxs = defaultdict(lambda: 0)
    # 1000 250   1   10   10
    times = get_schedule_jump(t_T=1000, n_sample=1, jump_length=1, jump_n_sample=20)  # 获得时间步，已经计算了重采样 \
    # 比如 249，248，249，248，247
    # jump_length 跟从什么时候加噪有关 1 就是 249 248 249  10的话就会先降很多再开始增加
    # jump_n_sample 是加噪的次数  10 就是重复加噪十次
    # 一次比较好的(1000, 1, 1 , 20)  初始值(250, 1, 10, 10)

    time_pairs = list(zip(times[:-1], times[1:]))  # 按对应顺序打包成元组再解压为元组列表
    if progress:
        from tqdm.auto import tqdm  # tqdm是进度条
        time_pairs = tqdm(time_pairs)

    for t_last, t_cur in time_pairs:
        idx_wall += 1
        # t_last_t = torch.tensor([t_last] * sample_shape[0], # 生成了2955大小的t
        #                         device=device)
        t_last_t = t_last
        if t_cur < t_last:  # reverse 逆向过程去噪 主要是 函数p_sample() 得x_{t-1}
            with torch.no_grad():
                x_before_step = x_after_step.clone()
                out = model_sample_repaint( # 输出x_{t-1}  x0'
                    model,
                    device=device,
                    dataloader=dataloader,
                    sample=x_after_step,  # x_t（所有细胞）
                    time=t_last_t,  # t
                    is_condi=is_condi,
                    condi_flag=True,
                    mask = mask,
                    gt = gt,
                    noise_scheduler = noise_scheduler,
                    pred_xstart=pred_xstart,
                    model_pred_type = model_pred_type,
                    is_classifier_guidance = is_classifier_guidance,
                    omega = omega
                )
                x_after_step = out["sample"]  # x_{t-1}
                pred_xstart = out["pred_xstart"]  # x0'

                sample_idxs[t_cur] += 1

                yield out

        else:  # 加噪
            # t_shift = conf.get('inpa_inj_time_shift', 1)  # 没有这个名字的参数， 这是让他为1的意思吗
            t_shift = 1
            x_before_step = x_after_step.clone()
            x_after_step = noise_scheduler.undo(
                x_before_step, x_after_step,
                est_x_0=out['pred_xstart'], t=t_last_t + t_shift, debug=False)
            pred_xstart = out["pred_xstart"]

def sample_loop_resample(
        model,
        noise=None,
        mask = None,
        gt = None,
        cond_fn=None, # 会用在classifier中
        device=None,
        progress=True,
        conf = None,
        dataloader=None,
        noise_scheduler=None,
        num_step = 1000,
        sample_shape = (2955, 32),
        is_condi=False,
        return_all=False,
        sample_intermediate=200,
        model_pred_type: str = 'noise',
        is_classifier_guidance=False,
        omega=0.1
    ):
        """
        遍历progressive产生的（yield）的生成器
        """
        final = None
        for sample in sample_loop_progressive(
            model,
            sample_shape = sample_shape,
            noise=noise,
            mask=mask,
            gt=gt,
            cond_fn=cond_fn,
            device=device,
            dataloader=dataloader,
            noise_scheduler=noise_scheduler,
            progress=progress,
            conf=conf,
            is_condi=is_condi,
            num_step=num_step,
            sample_intermediate=sample_intermediate,
            model_pred_type=model_pred_type,
            is_classifier_guidance = is_classifier_guidance,
            omega = omega
        ):
            final = sample

        if return_all:
            return final
        else:
            return final["sample"]

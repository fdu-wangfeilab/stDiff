from Extenrnal.sc_DM.model.diffusion_scheduler import *

# def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):

def _check_times(times, t_0, t_T):
    # Check end
    assert times[0] > times[1], (times[0], times[1])

    # Check beginning
    assert times[-1] == -1, times[-1]

    # Steplength = 1
    for t_last, t_cur in zip(times[:-1], times[1:]):
        assert abs(t_last - t_cur) == 1, (t_last, t_cur)

    # Value range
    for t in times:
        assert t >= t_0, (t, t_0)
    assert t <= t_T, (t, t_T)

def get_schedule_jump(t_T, n_sample, jump_length, jump_n_sample,
                      jump2_length=1, jump2_n_sample=1,
                      jump3_length=1, jump3_n_sample=1,
                      start_resampling=100000000):

    jumps = {}
    for j in range(0, t_T - jump_length, jump_length):
        jumps[j] = jump_n_sample - 1

    jumps2 = {}
    for j in range(0, t_T - jump2_length, jump2_length):
        jumps2[j] = jump2_n_sample - 1

    jumps3 = {}
    for j in range(0, t_T - jump3_length, jump3_length):
        jumps3[j] = jump3_n_sample - 1

    t = t_T
    ts = []

    while t >= 1:
        t = t - 1
        ts.append(t)

        if (
                t + 1 < t_T - 1 and
                t <= start_resampling
        ):
            for _ in range(n_sample - 1):
                t = t + 1
                ts.append(t)

                if t >= 0:
                    t = t - 1
                    ts.append(t)

        if (
                jumps3.get(t, 0) > 0 and
                t <= start_resampling - jump3_length
        ):
            jumps3[t] = jumps3[t] - 1
            for _ in range(jump3_length):
                t = t + 1
                ts.append(t)

        if (
                jumps2.get(t, 0) > 0 and
                t <= start_resampling - jump2_length
        ):
            jumps2[t] = jumps2[t] - 1
            for _ in range(jump2_length):
                t = t + 1
                ts.append(t)
            jumps3 = {}
            for j in range(0, t_T - jump3_length, jump3_length):
                jumps3[j] = jump3_n_sample - 1

        if (
                jumps.get(t, 0) > 0 and
                t <= start_resampling - jump_length
        ):
            jumps[t] = jumps[t] - 1
            for _ in range(jump_length):
                t = t + 1
                ts.append(t)
            jumps2 = {}
            for j in range(0, t_T - jump2_length, jump2_length):
                jumps2[j] = jump2_n_sample - 1

            jumps3 = {}
            for j in range(0, t_T - jump3_length, jump3_length):
                jumps3[j] = jump3_n_sample - 1

    ts.append(-1)

    _check_times(ts, -1, t_T)

    return ts

class RepaintNoiseScheduler(NoiseScheduler):

    def step(self,
             model_output,
             timestep,
             sample,
             model_pred_type: str = 'noise'):

        t = timestep
        # 用模型预测出的数值作为 noise

        # 预测x0' 再用 x_0_pred，输入的 x_t，t 来得到均值
        if model_pred_type == 'noise':
            pred_original_sample = self.reconstruct_x0(sample, t, model_output)
        elif model_pred_type == 'x_start':
            pred_original_sample = model_output
        else:
            raise NotImplementedError()

        pred_prev_sample = self.q_posterior(pred_original_sample, sample, t)
        # 这里采用公式造一个方差出来，用到noise作为重采样
        variance = 0
        if t > 0:
            noise = torch.randn_like(model_output)
            # 此时已经把噪声给加上去了
            variance = (self.get_variance(t) ** 0.5) * noise

        # 把均值和方差加起来，然后作为返回值
        pred_prev_sample = pred_prev_sample + variance
        # 强制使输出在 -1，1 之间, 保持数据范围
        return torch.clamp(pred_prev_sample, min=-1, max=1), pred_original_sample

    def undo(self, image_before_step, img_after_model, est_x_0, t, debug=False):
        # 对x_t加噪
        return self._undo(img_after_model, t)

    def _undo(self, img_out, t):
        beta = self.betas[t]

        img_in_est = torch.sqrt(1 - beta) * img_out + \
                     torch.sqrt(beta) * torch.randn_like(img_out)

        return img_in_est
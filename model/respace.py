import numpy as np
import torch as th
from diffusion_scheduler import NoiseScheduler


def space_timesteps(num_timesteps, section_counts):
    """
    根据DDIM论文，respacing就是从原始扩散步骤中截取一个子序列，用于推理的过程

    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.

    :param num_timesteps:  训练时所用的步数
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.Ex: [50,100,150] 将原始步骤分为3个逆采样部分，每个部分的步数为
                           列表中的数字，逆采样的步骤总共就是300步，但是逻辑上来说，刚开始的步骤最好设置大一些
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):  # 当section_counts的实参用ddimN传递时，进入该语句
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)  # 分块
    extra = num_timesteps % len(section_counts)  # 分块的时候不一定整除
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)  # 返回一个新的时间步骤


class SpacedDiffusion(NoiseScheduler):
    """
    A diffusion process which can skip steps in a base diffusion process.

    :param use_timesteps: a collection (sequence or set) of timesteps from the
                          original diffusion process to retain. 扩散过程中的时间步骤
    :param kwargs: the kwargs to create the base diffusion process.
    """

    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)  # 指可用的时间步，可能是步长为1，也可能大于1(respacing)
        self.timestep_map = []
        self.original_num_steps = len(kwargs["betas"])  # 原始步长

        base_diffusion = NoiseScheduler(**kwargs)  # pylint: disable=missing-kwoa
        last_alpha_cumprod = 1.0

        # 重新定义betas序列
        new_betas = []
        # 这里alpha_cumprod是一个序列，记录了扩散过程每一步的alpha_bar，
        # 解包之后，i为idx，alpha_cumprod就是原序列中对应的值
        for i, alpha_cumprod  in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:  # 若i刚好在新设立的时间步中,就把当前的beta包含进new_betas中
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)  # 将当前的时间索引i送入到新的时间表中
        kwargs["betas"] = np.array(new_betas)  # 父类中的betas就被更新了
        super().__init__(**kwargs)

    def p_mean_variance(  # 利用模型的输出得到x_t-1的均值、方差和预测的x0
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    # 如果当前的model已经是_WrappedModel类型，就什么也不做
    # 否则，重新实例化
    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )

    def _scale_timesteps(self, t):
        # Scaling is done by the wrapped model.
        return t


class _WrappedModel:
    # 其输入就是model、新的时间序列timestep_map...
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps  # 训练时间步

    def __call__(self, x, ts, **kwargs):
        # 将新的时间序列送入到训练设备中
        map_tensor = th.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        # 由于ts是连续的索引，而map_tensor中包含的是spacing之后的索引
        # 因此__call__的作用就是将ts映射到真正spacing后的时间步骤
        new_ts = map_tensor[ts]
        if self.rescale_timesteps:  # 将时间步长固定在1000以内
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model(x, new_ts, **kwargs)


if __name__=="__main__":
    num_times = 1000
    lst = [100,100,100]
    print(space_timesteps(num_times,lst))

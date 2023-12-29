import torch
import torch.nn as nn
import numpy as np
import math
import einops
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
import sys


class Attention2(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        C, G = x.shape
        qkv = self.qkv(x).reshape(C, 3, self.num_heads, G // self.num_heads)
        qkv = einops.rearrange(qkv, 'c n h fph -> n h c fph') # 最后形状 num_repeat, num_heads, counts, feature_per_head
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple) (h c fph)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(0, 1) # c h fph
        x = einops.rearrange(x, 'c h fph -> c (h fph)')
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, k, v):
        C, G = x.shape
        qkv = torch.concat((self.q_proj(x), self.k_proj(k), self.v_proj(v)), dim=-1).reshape(C, 3, self.num_heads, G // self.num_heads).permute(1, 2, 0, 3)   # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(C,G)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
def modulate(x, shift, scale):
    # 对数据做 shift 和 scale 的方法
    # scale.unsqueeze(1) 将 scale 转换成列向量
    res = x * (1 + scale) + shift
    return res

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    将 time emb 成 frequency_embedding_size 维，再投影到 hidden_size
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        # 将输入 emb 成 frequency_embedding_size 维 
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        # 最后一维加 None 相当于拓展一维
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        # 采用 pos emb 之后过 mlp
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
    

class CrossDiTblock(nn.Module):
    # adaBN -> attn -> mlp
    def __init__(self,
                 feature_dim=2000,
                 mlp_ratio=4.0,
                 num_heads=10,
                 **kwargs) -> None:
        super().__init__()
        
        self.norm1 = nn.LayerNorm(num_features=feature_dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention2(feature_dim, num_heads=num_heads, qkv_bias=True, **kwargs)
        
        self.norm2 = nn.LayerNorm(num_features=feature_dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = CrossAttention(feature_dim, num_heads=num_heads, qkv_bias=True, **kwargs)
        
        self.norm3 = nn.LayerNorm(num_features=feature_dim, elementwise_affine=False, eps=1e-6)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        
        mlp_hidden_dim = int(feature_dim * mlp_ratio)
        self.mlp = Mlp(in_features=feature_dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        
    
    def forward(self,x,c):
        # 将 condition 投影到 6 * hiddensize 之后沿列切成 6 份
        
        # attention blk
        x = x + self.attn(self.norm1(x))
        x = x + self.cross_attn(self.norm2(x),c,c)
        # mlp blk 采用 ViT 中实现的版本
        x = x + self.mlp(self.norm3(x))
        return x

class DiTblock(nn.Module):
    # adaBN -> attn -> mlp
    def __init__(self,
                 feature_dim=2000,
                 mlp_ratio=4.0,
                 num_heads=10,
                 **kwargs) -> None:
        super().__init__()
        
        self.norm1 = nn.LayerNorm(feature_dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention2(feature_dim, num_heads=num_heads, qkv_bias=True, **kwargs) 
        
        self.norm2 = nn.LayerNorm(feature_dim, elementwise_affine=False, eps=1e-6)
        approx_gelu = lambda: nn.GELU()
        
        mlp_hidden_dim = int(feature_dim * mlp_ratio)
        self.mlp = Mlp(in_features=feature_dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(feature_dim, 6 * feature_dim, bias=True)
        )
    
    def forward(self,x,c):
        # 将 condition 投影到 6 * hiddensize 之后沿列切成 6 份
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        # attention blk
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        # mlp blk 采用 ViT 中实现的版本
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT. adaLN -> linear
    """
    def __init__(self, hidden_size, out_size):
        super().__init__()
        # 为什么最后要关掉 elementwise_affine ?
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # 投影成输出的 patch 形状
        self.linear = nn.Linear(hidden_size, out_size, bias=True)
        # 最后也是需要 shift scale
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        # 先做 shift scale
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        # 最后投影到输出维度
        x = self.linear(x)
        return x      

BaseBlock = {'dit':DiTblock,
             'cross_dit':CrossDiTblock}

class DiT_stDiff(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 depth,
                 dit_type,
                 num_heads,
                 classes,
                 mlp_ratio=4.0,
                 **kwargs) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.classes = classes
        self.mlp_ratio = mlp_ratio
        self.dit_type = dit_type

        self.in_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size)
        )
        self.cond_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size)
        )

        # celltype emb
        self.condi_emb = nn.Embedding(classes, hidden_size)

        # time emb
        self.time_emb = TimestepEmbedder(hidden_size=self.hidden_size)

        # DiT block
        self.blks = nn.ModuleList([
            BaseBlock[dit_type](self.hidden_size, mlp_ratio=self.mlp_ratio, num_heads=self.num_heads) for _ in
            range(self.depth)
        ])

        # out layer
        self.out_layer = FinalLayer(self.hidden_size, self.input_size)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                # 线性层都采用 xavier 初始化
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    # bias 都初始化为 0
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize label embedding table:
        nn.init.normal_(self.condi_emb.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.time_emb.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_emb.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        if self.dit_type == 'dit':
            for block in self.blks:
                # adaLN_modulation 其实是个 linear, 即将所有的 adaLN 进行 0 初始化？
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        # 输出层的所有参数都做 0 初始化
        nn.init.constant_(self.out_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.out_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.out_layer.linear.weight, 0)
        nn.init.constant_(self.out_layer.linear.bias, 0)

    def forward(self, x, t, y,**kwargs): 
        x = x.float()
        t = self.time_emb(t)

        y = self.cond_layer(y)
        # z = self.condi_emb(z)
        c = t + y
        # c = t

        x = self.in_layer(x)
        for blk in self.blks:
            x = blk(x, c)
        return self.out_layer(x, c)
        
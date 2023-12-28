import torch.nn as nn
import torch 
import numpy as np
from typing import Union
from einops import rearrange, repeat

class Swish(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self,x):
        return x * torch.sigmoid(x)


# 设置不同的 act 方法
activation = {
    'relu':nn.ReLU(),
    'rrelu':nn.RReLU(),
    'sigmoid':nn.Sigmoid(),
    'leaky_relu':nn.LeakyReLU(),
    'tanh':nn.Tanh(),
    'gelu':nn.GELU(),
    'swish':Swish(),
    '':None
}

        


# 实现一个 positional embedding
class SinusoidalEmbedding(nn.Module):
    """可学习的 positional encoding

    Args:
        nn (_type_): _description_
    """
    def __init__(self, 
                size: int, 
                scale: float = 1.0, 
                is_learned: bool=False):
        # 此处的 size 和 scale 表示 encoding 之后输出的维度， scale 表示是否对被编码数据进行放缩
        super().__init__()
        self.size = size
        self.is_learned = is_learned
        self.scale = scale
        self.act = Swish()
        self.lin = nn.Linear(self.size,self.size)

    def forward(self, x: torch.Tensor):
        # 先进行 scale 的缩放
        x = x * self.scale
        half_size = self.size // 2
        emb = torch.log(torch.Tensor([10000.0])) / (half_size - 1)
        # 每一项都需要乘以 log10000 / halfsize
        emb = torch.exp(-emb * torch.arange(half_size)).to(x.device)
        # 应该是把两个变量拓展到相同的维度
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        # 在最后一维上将两种 emb 进行拼接
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        if self.is_learned:
            emb = self.lin(self.act(emb))
        return emb

    def __len__(self):
        return self.size
    
class DSBatchNorm(nn.Module):
    """
    Domain-specific Batch Normalization
    """
    def __init__(self, num_features, n_domain, eps=1e-5, momentum=0.1):
        """
        Parameters
        ----------
        num_features
            dimension of the features
        n_domain
            domain number
        """
        super().__init__()
        self.n_domain = n_domain
        self.num_features = num_features
        self.bns = nn.ModuleList([nn.BatchNorm1d(num_features, eps=eps, momentum=momentum) for i in range(n_domain)])
        
    def reset_running_stats(self):
        for bn in self.bns:
            bn.reset_running_stats()
            
    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()
            
    def _check_input_dim(self, input):
        raise NotImplementedError
            
    def forward(self, x, y):
        # 就是区分 batch 做 norm
        out = torch.zeros(x.size(0), self.num_features, device=x.device) #, requires_grad=False)
        for i in range(self.n_domain):
            indices = np.where(y.cpu().numpy()==i)[0]

            if len(indices) > 1:
                out[indices] = self.bns[i](x[indices])
            elif len(indices) == 1:
                out[indices] = x[indices]
#                 self.bns[i].training = False
#                 out[indices] = self.bns[i](x[indices])
#                 self.bns[i].training = True
        return out
     
class Block(nn.Module):
    """作为最基础的神经网络类

    Args:
        nn (_type_): _description_
    """
    def __init__(self, 
                 input_dim:int, 
                 output_dim:int, 
                 norm:str='', 
                 act:str='', 
                 dropout:Union[int,float]=0,
                 gene_dim:int=2000,):
        super().__init__()
        """基础的 fc -> norm -> act -> dropout, 默认在启用 dropout 时采用 skip connection
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc = nn.Linear(input_dim, output_dim)
        self.norm_type = 'normal'
        # 此处考虑输入网络的数据做 BatchNorm 的方法
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(output_dim)
        elif norm == 'ln':
            self.norm = nn.LayerNorm(output_dim)
        elif norm == 'dsbn':
            self.norm_type = 'dsbn'
            print('used dsbn')
            # 目前 n_domain 设置为 2 复现 scalex
            self.norm = DSBatchNorm(num_features = output_dim, n_domain=2)
        elif norm == 'genebn':
            self.norm = nn.BatchNorm1d(gene_dim)
        else:
            self.norm = None
        
        # 激活函数
        self.act = activation[act]
        
        if dropout >0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        
    def forward(self, x, y=None):
        
        h = self.fc(x)
        # 是否需要做 batchnorm
        if self.norm:
            if len(x) == 1:
                pass
            elif self.norm_type=='dsbn':
                h = self.norm(h,y)
            else:
               h = self.norm(h)
               
        # 是否需要激活函数
        if self.act:
            h = self.act(h)
            
        # 是否需要 dropout
        if self.dropout:
            h = x + self.dropout(h)
            
        return h
    
    
    
class AttentionBlock(nn.Module):
    # 最基础的 带 add & norm 的 multihead self-attention
    def __init__(self,
                 input_dim,
                 num_heads=20) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.attn = nn.MultiheadAttention(input_dim, num_heads=num_heads)
        self.norm = nn.BatchNorm1d(input_dim)
        self.dropout = nn.Dropout(p=0.2)
    
    def forward(self, q, k, v):
        attn_score = self.attn(q,k,v)[0]
        return self.norm(q + self.dropout(attn_score))

class FeedForward(nn.Module):
    def __init__(self,
                 input_dim:int,
                 hidden_dim:int,
                 dropout:float=0.2):
        super().__init__()
        # 将数据投影到 hidden dim 再投影回来
        self.ff = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.norm = nn.BatchNorm1d(input_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.norm(x + self.dropout(self.ff(x)))
    

class TransformerBlock(nn.Module):
    def __init__(self,
                input_dim:int,
                hidden_dim:int,
                num_heads:int,
                is_cross:bool=False,
                ):
        super().__init__()
        # 此处都是带 add&norm 的层
        self.attn_blk = AttentionBlock(input_dim=input_dim, num_heads=num_heads)
        self.ff_blk = FeedForward(input_dim=input_dim, hidden_dim=hidden_dim)
        # 是否是 cross attention 默认是self attention
        self.is_cross = is_cross
        
    def forward(self, x,k=None,v=None):
        # 如果是 cross attention 则需要输入额外的 key value
        if self.is_cross:
            out = self.ff_blk(self.attn_blk(x,k,v))
        else:
            out = self.ff_blk(self.attn_blk(x,x,x))
        return out
    

class TransformerEncoder(nn.Module):
    def __init__(self,
                 input_dim:int,
                 num_classes:int,
                 emb_dim:int=2000,
                 num_heads:int=20,
                 is_learned_timeebd:bool=True,
                 is_time_concat:bool = False,
                 is_condi_concat:bool = False,
                 is_msk_emb:bool =False,
                 attn_types=['self','self','self']) -> None:
        super().__init__()
        self.input_dim  = input_dim
        self.emb_dim = emb_dim
        self.is_msk_emb = is_msk_emb
        self.layers = nn.ModuleList()
        
        self.attn_types = attn_types
        self.needed_attn_types = ['self','cross']
        self.needed_mlp_types = ['mlp']
        
        self.is_time_concat = is_time_concat
        self.is_condi_concat = is_condi_concat
        concat_times = sum([int(is_time_concat), int(is_condi_concat)])
        self.concat_size = concat_times*self.emb_dim + self.input_dim
        # 对于输入的 raw 数据进行一个 bn
        self.bn = nn.BatchNorm1d(self.input_dim)
        
        # 保证 hidden_dim 稍微比 concat size 大一些
        self.hidden_dim = (concat_times+1) * self.emb_dim + self.input_dim
        # attention 的层数
        for attn_type in self.attn_types:
            # 判断是否是 cross attention
            if attn_type in self.needed_attn_types:
                self.layers.append(TransformerBlock(input_dim=self.concat_size,
                                                    hidden_dim=self.hidden_dim ,
                                                    num_heads=num_heads,
                                                    is_cross= attn_type == 'cross'))
            elif attn_type in self.needed_mlp_types:
                # 其实 mlp 就调用普通的 block
                self.layers.append(Block(input_dim=self.concat_size,
                                        output_dim=self.concat_size,
                                        norm='bn',
                                        act='relu',
                                        dropout=0.5))
        
        # 是否采用 condition emb
        if num_classes:
            self.condition_emb = nn.Sequential(
                    nn.Embedding(num_classes, self.emb_dim),
                    nn.BatchNorm1d(self.emb_dim)
                )
        
        if is_msk_emb:
            self.msk_emb = Block(
                input_dim=self.emb_dim,
                output_dim=self.emb_dim,
                norm='',
                act=''
            )
        # 可学习的 pos emb
        self.time_emb = nn.Sequential(
            SinusoidalEmbedding(self.emb_dim, is_learned=is_learned_timeebd),
            nn.BatchNorm1d(self.emb_dim)
        )
        
        self.out_layer = nn.Linear(self.concat_size, input_dim)
    
    
    def forward(self, x, t, y, msk=None, condi_flag=True):
        time_emb = self.time_emb(t)
        
        assert y is not None, "condition is none!!"
        condi_emb = self.condition_emb(y)
        
        if not condi_flag:
            # classifier free guidance 相关，是否遮挡掉 condi_emb
            condi_emb = torch.zeros_like(condi_emb)
        if msk is not None:
            if self.is_msk_emb:
                msk = self.msk_emb(msk)
            h = torch.concat((self.bn(x), msk, time_emb, condi_emb), dim=-1)
        else:
            h = torch.concat((self.bn(x), time_emb, condi_emb), dim=-1)
            
        for i, attn_type in enumerate(self.attn_types):
            if attn_type in self.needed_attn_types:
                if attn_type == 'self':
                    h = self.layers[i](h)
                elif attn_type == 'cross':
                    # x + time, condi 作为 cross attn
                    h = self.layers[i](h, condi_emb, condi_emb)
            elif attn_type in self.needed_mlp_types:
                if attn_type == 'mlp':
                    h = self.layers[i](h)
        
        return self.out_layer(h)


class CrossTransformer(TransformerEncoder):
    def __init__(self, 
                 input_dim: int,
                 num_classes: int, 
                 gene_dim:int = 2000,
                 emb_dim: int = 512, 
                 num_heads: int = 20, 
                 is_learned_timeebd: bool = True, 
                 is_time_concat: bool = False, 
                 is_condi_concat: bool = False, 
                 is_msk_emb: bool = False, 
                 attn_types=['self', 'self', 'self']) -> None:
        super().__init__(input_dim, num_classes, emb_dim, num_heads, is_learned_timeebd, is_time_concat, is_condi_concat, is_msk_emb, attn_types)
        self.input_dim = input_dim
        self.gene_dim = gene_dim
        self.emb_dim = emb_dim
        
        self.bn = nn.BatchNorm1d(self.gene_dim)
        
        self.input_layer = nn.Sequential(
            ExtractBlock(input_dim=self.input_dim*4,
                                        output_dim=input_dim,
                                        act='',
                                        norm=''))
        
        self.hidden_dim = self.input_dim  * 2
        # attention 的层数
        self.layers = nn.ModuleList()
        for attn_type in self.attn_types:
            # 判断是否是 cross attention
            if attn_type in self.needed_attn_types:
                self.layers.append(TransformerBlock(input_dim=self.input_dim*2,
                                                    hidden_dim=self.hidden_dim*2,
                                                    num_heads=num_heads,
                                                    is_cross= attn_type == 'cross'))
            elif attn_type in self.needed_mlp_types:
                # 其实 mlp 就调用普通的 block
                self.layers.append(Block(input_dim=self.input_dim+self.emb_dim,
                                        output_dim=self.input_dim+self.emb_dim,
                                        gene_dim=self.gene_dim,
                                        norm='genebn',
                                        act='gelu',
                                        dropout=0.2))
        # 是否采用 condition emb
        if num_classes:
            self.condition_emb = nn.Sequential(
                    nn.Embedding(num_classes, self.emb_dim),
                    # nn.BatchNorm1d(self.emb_dim//2)
                )
        
        if is_msk_emb:
            self.msk_emb = Block(
                input_dim=10,
                output_dim=self.input_dim,
                norm='',
                act=''
            )
        # 可学习的 pos emb
        self.time_emb = nn.Sequential(
            SinusoidalEmbedding(self.emb_dim, is_learned=is_learned_timeebd),
            # nn.BatchNorm1d(self.emb_dim//2)
        )
        
        self.out_layer = nn.Linear(self.input_dim+self.emb_dim,input_dim )
        
    def forward(self, x, t, y=None, msk=None, condi_flag=False):
        time_emb = self.time_emb(t)
        time_emb = repeat(time_emb,'c e -> c g e', g=self.gene_dim)
        # condi_emb = self.condition_emb(y)
        

        h = torch.concat((self.bn(x), time_emb), dim=-1)
        
        # h = self.input_layer(h)
        
        for i, attn_type in enumerate(self.attn_types):
            if attn_type in self.needed_attn_types:
                if attn_type == 'self':
                    h = self.layers[i](h)
                elif attn_type == 'cross':
                    # x + time, condi 作为 cross attn
                    h = self.layers[i](h)
            elif attn_type in self.needed_mlp_types:
                if attn_type == 'mlp':
                    h = self.layers[i](h)
        return self.out_layer(h)
        

class ExtractBlock(Block):
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int, 
                 norm: str = '', 
                 act: str = '', 
                 dropout: Union[int, float] = 0):
        super().__init__(input_dim, output_dim, norm, act, dropout)
    
    def forward(self, x, y=None):
        # 压缩到原来的维度
        h = self.fc(x)
        # h = self.dropout(h) + x[:,:self.output_dim]
        # return self.act(self.norm(h))
        return h

class MLP(nn.Module):
    """这个模块目前弃用

    Args:
        nn (_type_): _description_

    Returns:
        _type_: _description_
    """
    # 这个 MLP 主要采用的是 concat 策略
    def __init__(self,
                 input_dim,
                 num_classes,
                 is_concat=True,
                 is_learned_timeebd=False,
                 # 第一个表示 batchnorm， 第二个表示 激活函数， 第三个表示是否使用 attention
                 blks=[['ln','relu',None],['','relu',None],['','relu',None]]
                 ):
        super().__init__()
        
        self.is_concat = is_concat
        self.blks = blks
        self.time_emb = nn.Sequential(
            SinusoidalEmbedding(input_dim, is_learned=is_learned_timeebd),
            nn.BatchNorm1d(input_dim)
        )
        # 如果加上 class 的话就多一倍
        if num_classes:
            self.condition_emb = nn.Sequential(
                nn.Embedding(num_classes, input_dim),
                nn.BatchNorm1d(input_dim)
            )
        
        # self.cross_attn = nn.MultiheadAttention(input_dim,num_heads=20)
        self.bn = nn.BatchNorm1d(input_dim)
            
        if self.is_concat:
            self.concat_size = input_dim * 2 
            if num_classes:
                self.concat_size = input_dim * 3
        else:
            self.concat_size = input_dim 
            if num_classes:
                # 采用 cross attn
                self.concat_size = input_dim * 3
        
        self.layers = nn.ModuleList()
        for blk in self.blks:
            self.layers.append(
                Block(
                    input_dim=self.concat_size,
                    output_dim=self.concat_size,
                    norm=blk[0],
                    act=blk[1],
                    attn=blk[2]
                )
            )
        
        self.layers.append(nn.Linear(self.concat_size,input_dim))
        
    def forward(self, x, t, y=None):
        time_emb = self.time_emb(t)
        
        if y is not None:
            condi_emb =  self.condition_emb(y)
            # 判断是否采用 concat
            if self.is_concat:
                x = torch.cat((self.bn(x), time_emb, condi_emb),dim=-1)
            else:
                x = self.bn(x) + time_emb + condi_emb
        else:
            if self.is_concat:
                x = torch.cat((self.bn(x), time_emb), dim=-1)
            else:
                x = self.bn(x) + time_emb

        for layer in self.layers:
                x = layer(x)
        return x


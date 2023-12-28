from Extenrnal.sc_DM.model.diffusion_model import *


class TransformerEncoderNoGuidance(nn.Module):
    def __init__(self,
                 input_dim: int,
                 num_classes: int,
                 emb_dim: int = 2000,
                 num_heads: int = 20,
                 is_learned_timeebd: bool = True,
                 is_time_concat: bool = False,
                 is_condi_concat: bool = False,
                 is_msk_emb: bool = False,
                 attn_types=['self', 'self', 'self']) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.is_msk_emb = is_msk_emb
        self.layers = nn.ModuleList()

        self.attn_types = attn_types
        self.needed_attn_types = ['self', 'cross']
        self.needed_mlp_types = ['mlp']

        self.is_time_concat = is_time_concat
        self.is_condi_concat = is_condi_concat
        concat_times = sum([int(is_time_concat), int(is_condi_concat)])
        self.concat_size = concat_times * self.emb_dim + self.input_dim
        # 对于输入的 raw 数据进行一个 bn
        self.bn = nn.BatchNorm1d(self.input_dim)

        # 保证 hidden_dim 稍微比 concat size 大一些
        self.hidden_dim = (concat_times + 1) * self.emb_dim + self.input_dim
        # attention 的层数
        for attn_type in self.attn_types:
            # 判断是否是 cross attention
            if attn_type in self.needed_attn_types:
                self.layers.append(TransformerBlock(input_dim=self.concat_size,
                                                    hidden_dim=self.hidden_dim,
                                                    num_heads=num_heads,
                                                    is_cross=attn_type == 'cross'))
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

    def forward(self, x, t, msk=None, condi_flag=True):
        time_emb = self.time_emb(t)

        # assert y is not None, "condition is none!!"
        #         # condi_emb = self.condition_emb(y)

        if msk is not None:
            if self.is_msk_emb:
                msk = self.msk_emb(msk)
            h = torch.concat((self.bn(x), msk, time_emb), dim=-1)
        else:
            h = torch.concat((self.bn(x), time_emb), dim=-1)

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


class TransformerPalette(nn.Module):
    def __init__(self,
                 input_dim: int,
                 num_classes: int,
                 emb_dim: int = 2000,
                 num_heads: int = 20,
                 is_learned_timeebd: bool = True,
                 is_time_concat: bool = False,
                 is_condi_concat: bool = False,
                 is_msk_emb: bool = False,
                 attn_types=['self', 'self', 'self']) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.is_msk_emb = is_msk_emb
        self.layers = nn.ModuleList()

        self.attn_types = attn_types
        self.needed_attn_types = ['self', 'cross']
        self.needed_mlp_types = ['mlp']

        self.is_time_concat = is_time_concat
        self.is_condi_concat = is_condi_concat
        concat_times = sum([int(is_time_concat), int(is_condi_concat)])
        self.concat_size = self.emb_dim + self.input_dim * 2
        # 对于输入的 raw 数据进行一个 bn
        self.bn = nn.BatchNorm1d(self.input_dim)

        # 保证 hidden_dim 稍微比 concat size 大一些
        self.hidden_dim = (concat_times + 1) * self.emb_dim + self.input_dim
        # attention 的层数
        for attn_type in self.attn_types:
            # 判断是否是 cross attention
            if attn_type in self.needed_attn_types:
                self.layers.append(TransformerBlock(input_dim=self.concat_size,
                                                    hidden_dim=self.hidden_dim,
                                                    num_heads=num_heads,
                                                    is_cross=attn_type == 'cross'))
            elif attn_type in self.needed_mlp_types:
                # 其实 mlp 就调用普通的 block
                self.layers.append(Block(input_dim=self.concat_size,
                                         output_dim=self.concat_size,
                                         norm='bn',
                                         act='relu',
                                         dropout=0.5))

        # 是否采用 condition emb
        # if num_classes:
            # self.condition_emb = nn.Sequential(
            #     nn.Embedding(num_classes, self.emb_dim),
            #     nn.BatchNorm1d(self.emb_dim)
            # )

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
        condi_emb = y
        # if not condi_flag:
            # classifier free guidance 相关，是否遮挡掉 condi_emb
            # condi_emb = torch.zeros_like(condi_emb)
        if msk is not None:
            if self.is_msk_emb:
                msk = self.msk_emb(msk)
            h = torch.cat((self.bn(x), msk, time_emb, condi_emb), dim=-1)
        else:
            h = torch.cat((x, time_emb, condi_emb), dim=-1) # 去掉两个bn


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


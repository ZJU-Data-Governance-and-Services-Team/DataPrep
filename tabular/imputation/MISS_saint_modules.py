import torch
import torch.nn.functional as F
from torch import nn, einsum
import numpy as np
import pandas as pd
from einops import rearrange
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset


# helpers

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def ff_encodings(x, B):
    x_proj = (2. * np.pi * x.unsqueeze(-1)) @ B.t()
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

def _set_seed(seed):
    if seed is None:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
# classes

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# attention

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x, **kwargs):
        return self.net(x)


class Attention_withoutmask(nn.Module):
    def __init__(
            self,
            dim,
            heads=8,
            dim_head=16,
            dropout=0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            heads=8,
            dim_head=16,
            dropout=0.0
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None, ips=None):
        h = self.heads

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        # print("old", q.shape)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        # print("new", q.shape)
        # Compute attention scores
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        old_sim = sim

        # print("neww", sim.shape)
        # Apply key padding mask (if provided)
        if key_padding_mask is not None:
            array = key_padding_mask.cpu().numpy()
            # mask = ~mask

            # 检查每一行是否都为True
            all_true = np.all(array, axis=1)
            all_true_index = torch.tensor(all_true, device=sim.device)
            key_padding_mask = key_padding_mask.bool()
            # print("newww", key_padding_mask.shape)
            # print(key_padding_mask)
            key_padding_mask = ~key_padding_mask
            # print("sim", sim.shape, ips.shape)
            # print(key_padding_mask)
            ips = ips.unsqueeze(1).unsqueeze(2)
            # print(ips.shape, ips)

            sim = sim * ips
            sim = sim.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )
            # print("newww", key_padding_mask.shape)
            # print("s", sim.shape)
            # print("k", key_padding_mask.shape, key_padding_mask)
            sim = sim.float()
            sim[all_true_index] = old_sim[all_true_index]
        # Compute attention weights
        attn = sim.softmax(dim=-1)
        attn = attn.float()
        # Apply attention weights to values
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # Concatenate heads and apply output layer
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        out = self.to_out(out)
        return out


# 行列注意力transformer
class RowColTransformer(nn.Module):
    def __init__(self, num_tokens, dim, nfeats, depth, heads, dim_head, attn_dropout, ff_dropout, style='col'):
        super().__init__()
        self.embeds = nn.Embedding(num_tokens, dim)
        self.layers = nn.ModuleList([])
        self.mask_embed = nn.Embedding(nfeats, dim)
        self.style = style
        for _ in range(depth):
            if self.style == 'colrow':
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, Residual(Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout))),

                    # PreNorm(dim, Residual(Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout))),
                    PreNorm(dim, Residual(FeedForward(dim, dropout=ff_dropout))),
                    PreNorm(dim * nfeats,
                            Residual(Attention(dim * nfeats, heads=heads, dim_head=64, dropout=attn_dropout))),
                    PreNorm(dim * nfeats, Residual(FeedForward(dim * nfeats, dropout=ff_dropout))),
                ]))
            else:
                self.layers.append(nn.ModuleList([
                    PreNorm(dim * nfeats,
                            Residual(Attention(dim * nfeats, heads=heads, dim_head=64, dropout=attn_dropout))),
                    PreNorm(dim * nfeats, Residual(FeedForward(dim * nfeats, dropout=ff_dropout))),
                ]))

    def forward(self, x, x_cont=None, cont_mask=None, cat_mask=None, cat_ips=None, con_ips=None, row_ips=None):
        if cont_mask is not None:
            cont_mask = torch.cat((cat_mask, cont_mask), dim=1).bool()
            ips = torch.cat((cat_ips, con_ips), dim=1)
        if x_cont is not None:
            x = torch.cat((x, x_cont), dim=1)
        _, n, _ = x.shape
        # cont_mask = ~cont_mask
        # print(cont_mask)
        # print(x)
        if self.style == 'colrow':
            for attn1, ff1, attn2, ff2 in self.layers:
                if (cont_mask == None):
                    x = attn1(x)
                else:
                    x = attn1(x, key_padding_mask=cont_mask, ips=ips)
                x = ff1(x)
                x = rearrange(x, 'b n d -> 1 b (n d)')
                x = attn2(x, ips=row_ips)
                x = ff2(x)
                x = rearrange(x, '1 b (n d) -> b n d', n=n)
        else:
            for attn1, ff1 in self.layers:
                x = rearrange(x, 'b n d -> 1 b (n d)')
                x = attn1(x)
                x = ff1(x)
                x = rearrange(x, '1 b (n d) -> b n d', n=n)
        return x


# 列注意力transformer
class Transformer(nn.Module):
    def __init__(self, num_tokens, dim, depth, heads, dim_head, attn_dropout, ff_dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.attn = nn.MultiheadAttention(dim, heads, attn_dropout)

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Residual(Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout))),
                PreNorm(dim, Residual(FeedForward(dim, dropout=ff_dropout))),
            ]))

    def forward(self, x, x_cont=None, cont_mask=None, cat_mask=None, cat_ips=None, con_ips=None):
        if cont_mask is not None:
            cont_mask = torch.cat((cat_mask, cont_mask), dim=1).bool()
            ips = torch.cat((cat_ips, con_ips), dim=1)
            # cont_mask = ~cont_mask
        if x_cont is not None:
            x = torch.cat((x, x_cont), dim=1)
        _, n, _ = x.shape

        for attn1, ff1 in self.layers:
            # print("Aaaaaaaaaa", x, x.shape, mask, mask.shape)
            if (cont_mask is None):
                x = attn1(x)
            else:
                # x = x.transpose(0, 1)
                # print("123", co, cont_mask.shape,x.shape)
                # co += 1
                # x, _ = self.attn(x, x, x, key_padding_mask = cont_mask, need_weights = False) #torch.Size([59, 256, 32])
                x = attn1(x, key_padding_mask=cont_mask, ips=ips)
                # x = x.transpose(0, 1)
                # print(x.shape)
            x = ff1(x)
        # print("xshape", x.shape)
        return x


# mlp模块
class MLP(nn.Module):
    def __init__(self, dims, act=None):
        super().__init__()
        dims_pairs = list(zip(dims[:-1], dims[1:]))
        layers = []
        for ind, (dim_in, dim_out) in enumerate(dims_pairs):
            is_last = ind >= (len(dims) - 1)
            linear = nn.Linear(dim_in, dim_out)
            layers.append(linear)

            if is_last:
                continue
            if act is not None:
                layers.append(act)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class simple_MLP(nn.Module):
    def __init__(self, dims):
        super(simple_MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.ReLU(),
            nn.Linear(dims[1], dims[2])
        )

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x


class TabAttention(nn.Module):
    def __init__(
            self,
            *,
            categories,
            num_continuous,
            dim,
            depth,
            heads,
            dim_head=16,
            dim_out=1,
            mlp_hidden_mults=(4, 2),
            mlp_act=None,
            num_special_tokens=1,
            continuous_mean_std=None,
            attn_dropout=0.,
            ff_dropout=0.,
            lastmlp_dropout=0.,
            cont_embeddings='MLP',
            scalingfactor=10,
            attentiontype='col'
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'

        # categories related calculations
        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # create category embeddings table

        self.num_special_tokens = num_special_tokens
        self.total_tokens = self.num_unique_categories + num_special_tokens

        # for automatically offsetting unique category ids to the correct position in the categories embedding table
        categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value=num_special_tokens)
        categories_offset = categories_offset.cumsum(dim=-1)[:-1]

        self.register_buffer('categories_offset', categories_offset)

        self.norm = nn.LayerNorm(num_continuous)
        self.num_continuous = num_continuous
        self.dim = dim
        self.cont_embeddings = cont_embeddings
        self.attentiontype = attentiontype

        # 为每个连续数据构建映射mlp
        if self.cont_embeddings == 'MLP':
            self.simple_MLP = nn.ModuleList([simple_MLP([1, 100, self.dim]) for _ in range(self.num_continuous)])
            input_size = (dim * self.num_categories) + (dim * num_continuous)
            nfeats = self.num_categories + num_continuous
        else:
            print('Continous features are not passed through attention')
            input_size = (dim * self.num_categories) + num_continuous
            nfeats = self.num_categories

            # transformer
        if attentiontype == 'col':
            self.transformer = Transformer(
                num_tokens=self.total_tokens,
                dim=dim,
                depth=depth,
                heads=heads,
                dim_head=dim_head,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout
            )
        elif attentiontype in ['row', 'colrow']:
            self.transformer = RowColTransformer(
                num_tokens=self.total_tokens,
                dim=dim,
                nfeats=nfeats,
                depth=depth,
                heads=heads,
                dim_head=dim_head,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                style=attentiontype
            )

        l = input_size // 8
        hidden_dimensions = list(map(lambda t: l * t, mlp_hidden_mults))
        all_dimensions = [input_size, *hidden_dimensions, dim_out]

        self.mlp = MLP(all_dimensions, act=mlp_act)
        self.embeds = nn.Embedding(self.total_tokens, self.dim)  # .to(device)

        cat_mask_offset = F.pad(torch.Tensor(self.num_categories).fill_(2).type(torch.int8), (1, 0), value=0)
        cat_mask_offset = cat_mask_offset.cumsum(dim=-1)[:-1]

        con_mask_offset = F.pad(torch.Tensor(self.num_continuous).fill_(2).type(torch.int8), (1, 0), value=0)
        con_mask_offset = con_mask_offset.cumsum(dim=-1)[:-1]

        self.register_buffer('cat_mask_offset', cat_mask_offset)
        self.register_buffer('con_mask_offset', con_mask_offset)

        self.mask_embeds_cat = nn.Embedding(self.num_categories * 2, self.dim)
        self.mask_embeds_cont = nn.Embedding(self.num_continuous * 2, self.dim)

    def forward(self, x_categ, x_cont, x_categ_enc, x_cont_enc):
        device = x_categ.device
        if self.attentiontype == 'justmlp':
            if x_categ.shape[-1] > 0:
                flat_categ = x_categ.flatten(1).to(device)
                x = torch.cat((flat_categ, x_cont.flatten(1).to(device)), dim=-1)
            else:
                x = x_cont.clone()
        else:
            if self.cont_embeddings == 'MLP':
                x = self.transformer(x_categ_enc, x_cont_enc.to(device))
            else:
                if x_categ.shape[-1] <= 0:
                    x = x_cont.clone()
                else:
                    flat_categ = self.transformer(x_categ_enc).flatten(1)
                    x = torch.cat((flat_categ, x_cont), dim=-1)
        flat_x = x.flatten(1)
        return self.mlp(flat_x)


class sep_MLP(nn.Module):
    def __init__(self, dim, len_feats, categories):
        super(sep_MLP, self).__init__()
        self.len_feats = len_feats
        self.layers = nn.ModuleList([])
        for i in range(len_feats):
            self.layers.append(simple_MLP([dim, 5 * dim, categories[i]]))

    def forward(self, x):
        y_pred = list([])
        for i in range(self.len_feats):
            x_i = x[:, i, :]
            pred = self.layers[i](x_i)
            # print("pred", pred.shape)
            y_pred.append(pred)
        return y_pred


class SAINT(nn.Module):
    def __init__(
            self,
            *,
            categories,
            num_continuous,
            dim,
            depth,
            heads,
            dim_head=16,
            dim_out=1,
            mlp_hidden_mults=(4, 2),
            mlp_act=None,
            num_special_tokens=0,
            attn_dropout=0.,
            ff_dropout=0.,
            cont_embeddings='MLP',
            scalingfactor=10,
            attentiontype='col',
            final_mlp_style='common',
            y_dim=2
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'

        # categories related calculations

        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # create category embeddings table

        self.num_special_tokens = num_special_tokens
        self.total_tokens = self.num_unique_categories + num_special_tokens

        # for automatically offsetting unique category ids to the correct position in the categories embedding table

        categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value = num_special_tokens)
        categories_offset = categories_offset.cumsum(dim = -1)[:-1]

        self.register_buffer('categories_offset', categories_offset)


        self.norm = nn.LayerNorm(num_continuous)
        self.num_continuous = num_continuous
        self.dim = dim
        self.cont_embeddings = cont_embeddings
        self.attentiontype = attentiontype
        self.final_mlp_style = final_mlp_style

        if self.cont_embeddings == 'MLP':
            self.simple_MLP = nn.ModuleList([simple_MLP([1 ,100 ,self.dim]) for _ in range(self.num_continuous)])
            input_size = (dim * self.num_categories)  + (dim * num_continuous)
            nfeats = self.num_categories + num_continuous
        elif self.cont_embeddings == 'pos_singleMLP':
            self.simple_MLP = nn.ModuleList([simple_MLP([1 ,100 ,self.dim]) for _ in range(1)])
            input_size = (dim * self.num_categories)  + (dim * num_continuous)
            nfeats = self.num_categories + num_continuous
        else:
            print('Continous features are not passed through attention')
            input_size = (dim * self.num_categories) + num_continuous
            nfeats = self.num_categories

            # transformer
        if attentiontype == 'col':
            self.transformer = Transformer(
                num_tokens = self.total_tokens,
                dim = dim,
                depth = depth,
                heads = heads,
                dim_head = dim_head,
                attn_dropout = attn_dropout,
                ff_dropout = ff_dropout
            )
        elif attentiontype in ['row', 'colrow'] :
            self.transformer = RowColTransformer(
                num_tokens = self.total_tokens,
                dim = dim,
                nfeats= nfeats,
                depth = depth,
                heads = heads,
                dim_head = dim_head,
                attn_dropout = attn_dropout,
                ff_dropout = ff_dropout,
                style = attentiontype
            )

        l = input_size // 8
        hidden_dimensions = list(map(lambda t: l * t, mlp_hidden_mults))
        all_dimensions = [input_size, *hidden_dimensions, dim_out]

        self.mlp = MLP(all_dimensions, act = mlp_act)
        self.embeds = nn.Embedding(self.total_tokens, self.dim) # .to(device)

        cat_mask_offset = F.pad(torch.Tensor(self.num_categories).fill_(2).type(torch.int8), (1, 0), value = 0)
        cat_mask_offset = cat_mask_offset.cumsum(dim = -1)[:-1]

        con_mask_offset = F.pad(torch.Tensor(self.num_continuous).fill_(2).type(torch.int8), (1, 0), value = 0) # 左侧填1列为0
        con_mask_offset = con_mask_offset.cumsum(dim = -1)[:-1]

        self.register_buffer('cat_mask_offset', cat_mask_offset)
        self.register_buffer('con_mask_offset', con_mask_offset)

        self.mask_embeds_cat = nn.Embedding(self.num_categories *2, self.dim)
        self.mask_embeds_cont = nn.Embedding(self.num_continuous *2, self.dim)
        self.single_mask = nn.Embedding(2, self.dim)
        self.pos_encodings = nn.Embedding(self.num_categories+ self.num_continuous, self.dim)
        # 构建将embedding后的分类数据和连续数据恢复原有形式的mlp
        if self.final_mlp_style == 'common':
            self.mlp1 = simple_MLP([dim ,(self.total_tokens ) *2, self.total_tokens])
            self.mlp2 = simple_MLP([dim ,(self.num_continuous), 1])

        else:
            self.mlp1 = sep_MLP(dim, self.num_categories, categories)
            self.mlp2 = sep_MLP(dim, self.num_continuous, np.ones(self.num_continuous).astype(int))

        self.mlpfory = simple_MLP([dim, 1000, y_dim])
        self.pt_mlp = simple_MLP([dim * (self.num_continuous + self.num_categories),
                                  6 * dim * (self.num_continuous + self.num_categories) // 5,
                                  dim * (self.num_continuous + self.num_categories) // 2])
        self.pt_mlp2 = simple_MLP([dim * (self.num_continuous + self.num_categories),
                                   6 * dim * (self.num_continuous + self.num_categories) // 5,
                                   dim * (self.num_continuous + self.num_categories) // 2])

    def forward(self, x_categ, x_cont, cont_mask=None, cat_mask=None, cat_ips=None, con_ips=None, rowips=None):
        # 将数据通过transformer并通过mlp
        if (cont_mask == None):
            x = self.transformer(x_categ, x_cont)
        else:
            if rowips is None:
                x = self.transformer(x_categ, x_cont, cont_mask, cat_mask, cat_ips, con_ips)
            else:
                x = self.transformer(x_categ, x_cont, cont_mask, cat_mask, cat_ips, con_ips, rowips)
        cat_outs = self.mlp1(x[:, :self.num_categories, :])
        con_outs = self.mlp2(x[:, self.num_categories:, :])
        return cat_outs, con_outs


# ---------------------------------------------------------------------------
# Strict MISS-SAINT path: in-memory adaptation of code/saint/saint-main/train.py
# ---------------------------------------------------------------------------

criterion2 = nn.MSELoss(reduction='none')
criterion3 = nn.LogSoftmax(dim=1)


def sampling(datafull, datamiss, num, method='feature'):
    """Original MISS expert-sampling step.

    It fills a small number of missing entries with their full-data values and
    marks them as 0.5 in the indicator matrix. Indicator convention:
    1 = observed, 0 = missing, 0.5 = sampled/expert-filled.
    """
    datafull = np.array(datafull, dtype=object)
    datamiss = np.array(datamiss, dtype=object).copy()
    indicator = np.array(~pd.isna(datamiss), dtype=np.float32)
    p = datamiss.shape[1]

    if method == 'feature':
        for i in range(p):
            if pd.isna(datamiss[:, i]).any():
                index = np.where(pd.isna(datamiss[:, i]))[0]
                sample_pos = np.random.choice(len(index), num)
                target = index[sample_pos]
                datamiss[target, i] = datafull[target, i]
                indicator[target, i] = 0.5
    elif method == 'sample':
        index = np.where(pd.isna(datamiss).any(axis=1))[0]
        sample_pos = np.random.choice(len(index), num)
        target = index[sample_pos]
        for i in target:
            for j in range(p):
                if indicator[i, j] == 0:
                    indicator[i, j] = 0.5
    else:
        raise ValueError(f"Unsupported sampling method: {method}")

    return datamiss, indicator


def train_test_split_random(datamiss, indicator, column, sample_num=20, complete_num=20):
    complete_data_index = np.where(indicator[:, column] == 1)[0]
    human_sample_index = np.where(indicator[:, column] == 0.5)[0]
    complete_sample_index = np.random.choice(
        complete_data_index,
        size=min(complete_num, len(complete_data_index)),
        replace=False,
    )
    train_index = np.concatenate((complete_sample_index, human_sample_index), axis=0)
    predict_index = np.setdiff1d(complete_data_index, complete_sample_index)
    return train_index, predict_index


def train_test_split_no_random(datamiss, indicator, column, sample_num=20, complete_num=20):
    human_sample_index = np.where(indicator[:, column] == 0.5)[0]
    complete_data_index = np.where(indicator[:, column] == 1)[0]

    if len(human_sample_index) == 0:
        complete_sample_index = np.random.choice(
            complete_data_index,
            size=min(complete_num, len(complete_data_index)),
            replace=False,
        )
        return complete_sample_index, np.setdiff1d(complete_data_index, complete_sample_index)

    imputation_data = pd.DataFrame(datamiss[human_sample_index, column]).astype(float)
    imputation_mean = imputation_data.mean()
    complete_mean = 1 - imputation_mean
    complete_data = pd.DataFrame(
        datamiss[complete_data_index, column],
        index=complete_data_index,
    ).astype(float)
    df_abs_diff = complete_data.sub(complete_mean).abs().sort_values(by=0, ascending=False)
    closest_indices = np.array(df_abs_diff[:complete_num].index)

    train_index = np.concatenate((closest_indices, human_sample_index), axis=0)
    predict_index = np.setdiff1d(complete_data_index, closest_indices)
    return train_index, predict_index


def compute_ips(datamiss, indicator, num=20, method='xgb', observed_num=1, complete_sample="no-Random"):
    datamiss = np.asarray(datamiss, dtype=float)
    indicator = np.asarray(indicator, dtype=np.float32)
    n, p = datamiss.shape
    p_miss = np.zeros((n, p), dtype=np.float32)

    for i in range(p):
        if not (indicator[:, i] == 0.5).any():
            continue

        if complete_sample == "Random":
            train_index, predict_index = train_test_split_random(
                datamiss, indicator, i, sample_num=num, complete_num=num * observed_num
            )
        else:
            train_index, predict_index = train_test_split_no_random(
                datamiss, indicator, i, sample_num=num, complete_num=num * observed_num
            )

        if len(train_index) == 0:
            continue

        y = np.array((indicator[:, i] != 1), dtype=np.float32)
        y_train = y[train_index]
        data_X_train = pd.DataFrame(datamiss[train_index, :])
        data_X_predict = pd.DataFrame(datamiss[predict_index, :])

        if method == 'xgb':
            import xgboost as xgb
            if len(np.unique(y_train)) < 2:
                p_miss[train_index, i] = y_train
                continue
            params = {
                'objective': 'reg:logistic',
                'booster': 'gbtree',
                'max_depth': 3,
                'verbosity': 0,
                'scale_pos_weight': max(float(sum(y == 0)) / max(float(sum(y == 1)), 1.0), 1e-6),
            }
            xgb_model = xgb.train(params, xgb.DMatrix(data_X_train, y_train))
            if len(predict_index) > 0:
                p_miss[predict_index, i] = xgb_model.predict(xgb.DMatrix(data_X_predict))
            p_miss[train_index, i] = y_train
        elif method == 'lr':
            from sklearn.linear_model import LogisticRegression as LR
            if len(np.unique(y_train)) < 2:
                p_miss[train_index, i] = y_train
                continue
            data_X_train = data_X_train.apply(lambda col: col.fillna(col.mean()), axis=0).fillna(0)
            data_X_predict = data_X_predict.apply(lambda col: col.fillna(col.mean()), axis=0).fillna(0)
            lr_model = LR(class_weight='balanced', max_iter=1000)
            lr_model.fit(data_X_train, y_train)
            if len(predict_index) > 0:
                p_miss[predict_index, i] = lr_model.predict_proba(data_X_predict)[:, 1]
            p_miss[train_index, i] = y_train
        elif method == 'bayes':
            from sklearn.naive_bayes import GaussianNB
            data_X = pd.DataFrame(datamiss).apply(lambda col: col.fillna(col.mean()), axis=0).fillna(0)
            bayes_model = GaussianNB()
            bayes_model.fit(data_X, y)
            p_miss[:, i] = bayes_model.predict_proba(data_X)[:, 1]
        else:
            raise ValueError(f"Unsupported IPS method: {method}")

    p_miss[p_miss > 0.95] = 0.95
    ips = 1 / (1 - p_miss)
    ips[indicator == 0] = 0
    return ips.astype(np.float32)


def embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model, vision_dset=False):
    device = x_cont.device
    x_categ = x_categ + model.categories_offset.type_as(x_categ)
    x_categ_enc = model.embeds(x_categ)
    n1, n2 = x_cont.shape

    if model.cont_embeddings != 'MLP':
        raise Exception('This case should not work!')

    x_cont_enc = torch.empty(n1, n2, model.dim, device=device)
    for i in range(model.num_continuous):
        x_cont_enc[:, i, :] = model.simple_MLP[i](x_cont[:, i])

    cat_mask_temp = cat_mask + model.cat_mask_offset.type_as(cat_mask)
    con_mask_temp = con_mask + model.con_mask_offset.type_as(con_mask)
    cat_mask_temp = model.mask_embeds_cat(cat_mask_temp)
    con_mask_temp = model.mask_embeds_cont(con_mask_temp)
    x_categ_enc[cat_mask == 0] = cat_mask_temp[cat_mask == 0]
    x_cont_enc[con_mask == 0] = con_mask_temp[con_mask == 0]
    return x_categ, x_categ_enc, x_cont_enc


def _data_split(X, y, nan_mask, ips, rowips, indices):
    x_d = {
        'data': X.to_numpy()[indices],
        'mask': nan_mask.to_numpy()[indices],
        'ips': ips.to_numpy()[indices],
        'rowips': rowips.to_numpy()[indices],
    }
    y_d = {'data': y[indices].reshape(-1, 1)}
    return x_d, y_d


class DataSetCatCon(Dataset):
    def __init__(self, X, Y, cat_cols, task='clf', continuous_mean_std=None):
        cat_cols = list(cat_cols)
        X_mask = X['mask'].copy()
        X_ips = X['ips'].copy()
        rowips = X['rowips'].copy().squeeze()
        X_data = X['data'].copy()

        con_cols = list(set(np.arange(X_data.shape[1])) - set(cat_cols))
        self.cat_cols = cat_cols
        self.con_cols = con_cols
        self.X1 = X_data[:, cat_cols].copy().astype(np.int64)
        self.X2 = X_data[:, con_cols].copy().astype(np.float32)
        self.X1_mask = X_mask[:, cat_cols].copy().astype(np.int64)
        self.X2_mask = X_mask[:, con_cols].copy().astype(np.int64)
        self.X1_ips = X_ips[:, cat_cols].copy().astype(np.float32)
        self.X2_ips = X_ips[:, con_cols].copy().astype(np.float32)
        self.rowips = rowips.copy().astype(np.float32)
        self.y = Y['data'] if task == 'clf' else Y['data'].astype(np.float32)
        self.cls = np.zeros_like(self.y, dtype=int)
        self.cls_mask = np.ones_like(self.y, dtype=int)
        self.cls_ips = np.ones_like(self.y, dtype=int)

        if continuous_mean_std is not None and len(con_cols) > 0:
            mean, std = continuous_mean_std
            self.X2 = (self.X2 - mean) / std

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            np.concatenate((self.cls[idx], self.X1[idx])),
            self.X2[idx],
            self.y[idx],
            np.concatenate((self.cls_mask[idx], self.X1_mask[idx])),
            self.X2_mask[idx],
            np.concatenate((self.cls_ips[idx], self.X1_ips[idx])),
            self.X2_ips[idx],
            self.rowips[idx],
        )


def _as_dataframe(data, columns=None):
    if isinstance(data, pd.DataFrame):
        return data.copy()
    return pd.DataFrame(data, columns=columns)


def _infer_task(y):
    if pd.api.types.is_float_dtype(y) and y.nunique(dropna=True) > 20:
        return 'regression'
    if y.nunique(dropna=True) <= 2:
        return 'binary'
    return 'multiclass'


def _prepare_dataframes(data, full_data, label_column=None, labels=None):
    datamiss = _as_dataframe(data)
    datafull = _as_dataframe(full_data, columns=datamiss.columns)

    if labels is not None:
        label_name = label_column or 'target'
        datamiss = datamiss.copy()
        datafull = datafull.copy()
        datamiss[label_name] = labels
        datafull[label_name] = labels
        label_column = label_name
    elif label_column is None:
        label_column = datamiss.columns[-1]

    if label_column not in datamiss.columns or label_column not in datafull.columns:
        raise ValueError("MISS-SAINT strict mode requires a label column or labels argument.")
    if list(datamiss.columns) != list(datafull.columns):
        datafull = datafull[datamiss.columns]
    return datamiss, datafull, label_column


def _restore_sampled_dtypes(sampled_df, reference_df):
    restored = sampled_df.copy()
    for col in restored.columns:
        if col not in reference_df.columns:
            continue
        if pd.api.types.is_numeric_dtype(reference_df[col]):
            restored[col] = pd.to_numeric(restored[col], errors="coerce")
        else:
            restored[col] = restored[col].astype(reference_df[col].dtype, copy=False)
    return restored


def _build_strict_dataset(datamiss_df, indicator, ips, row_ips, task, seed, label_column,
                          datasplit, cat_idxs=None, cat_threshold=100, strict_fillna=True):
    np.random.seed(seed)
    data = datamiss_df.copy()
    X = data.drop(columns=[label_column]).copy()
    y_series = data[label_column].copy()

    ips_df = pd.DataFrame(ips, index=X.index, columns=X.columns)
    rowips_df = pd.DataFrame(row_ips, index=X.index)

    if cat_idxs is None:
        nunique = X.nunique(dropna=True)
        types = X.dtypes
        categorical_indicator = np.zeros(X.shape[1], dtype=bool)
        for col in X.columns:
            if types[col] == 'object' or nunique[col] < cat_threshold:
                categorical_indicator[X.columns.get_loc(col)] = True
        cat_idxs = list(np.where(categorical_indicator)[0])
    else:
        cat_idxs = list(cat_idxs)
        categorical_indicator = np.zeros(X.shape[1], dtype=bool)
        categorical_indicator[cat_idxs] = True

    categorical_columns = X.columns[cat_idxs].tolist()
    con_idxs = list(set(range(len(X.columns))) - set(cat_idxs))
    cont_columns = X.columns[con_idxs].tolist()

    set_labels = np.random.choice(["train", "valid", "test"], p=datasplit, size=(X.shape[0],))
    train_indices = np.where(set_labels == "train")[0]
    valid_indices = np.where(set_labels == "valid")[0]
    test_indices = np.where(set_labels == "test")[0]

    temp = X.fillna("MissingValue")
    nan_mask = temp.ne("MissingValue").astype(int)
    feature_indicator = pd.DataFrame(indicator[:, :X.shape[1]], columns=X.columns, index=X.index)
    nan_mask[feature_indicator == 0.5] = 1

    encoders = {}
    categorical_modes = {}
    cat_dims = []
    for col in categorical_columns:
        non_missing = X[col].dropna()
        if len(non_missing) > 0:
            mode = non_missing.mode(dropna=True)
            categorical_modes[col] = mode.iloc[0] if len(mode) > 0 else non_missing.iloc[0]
        else:
            categorical_modes[col] = None
        X[col] = X[col].astype("str").fillna("MissingValue")
        encoder = LabelEncoder()
        X[col] = encoder.fit_transform(X[col].values)
        encoders[col] = encoder
        cat_dims.append(len(encoder.classes_))

    fill_values = {}
    for col in cont_columns:
        fill_value = pd.to_numeric(X.iloc[train_indices][col], errors='coerce').mean()
        if not np.isfinite(fill_value):
            fill_value = 0.0
        fill_values[col] = fill_value
        if strict_fillna:
            X.fillna(fill_value, inplace=True)
        else:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(fill_value)

    y = y_series.to_numpy()
    y_encoder = None
    if task != 'regression':
        y_encoder = LabelEncoder()
        y = y_encoder.fit_transform(y)

    X_train, y_train = _data_split(X, y, nan_mask, ips_df, rowips_df, train_indices)
    X_valid, y_valid = _data_split(X, y, nan_mask, ips_df, rowips_df, valid_indices)
    X_test, y_test = _data_split(X, y, nan_mask, ips_df, rowips_df, test_indices)

    if len(con_idxs) > 0:
        train_cont = np.array(X_train['data'][:, con_idxs], dtype=np.float32)
        train_mean = train_cont.mean(0)
        train_std = train_cont.std(0)
        train_std = np.where(train_std < 1e-6, 1e-6, train_std)
    else:
        train_mean = np.array([], dtype=np.float32)
        train_std = np.array([], dtype=np.float32)

    return {
        "cat_dims": cat_dims,
        "cat_idxs": cat_idxs,
        "con_idxs": con_idxs,
        "X_train": X_train,
        "y_train": y_train,
        "X_valid": X_valid,
        "y_valid": y_valid,
        "X_test": X_test,
        "y_test": y_test,
        "train_mean": train_mean,
        "train_std": train_std,
        "train_indices": train_indices,
        "valid_indices": valid_indices,
        "test_indices": test_indices,
        "feature_columns": list(X.columns),
        "label_column": label_column,
        "categorical_columns": categorical_columns,
        "continuous_columns": cont_columns,
        "encoders": encoders,
        "categorical_modes": categorical_modes,
        "y_encoder": y_encoder,
        "fill_values": fill_values,
        "processed_X": X,
        "nan_mask": nan_mask,
    }


def _build_loader(X, y, cat_idxs, task, continuous_mean_std, batch_size, shuffle):
    ds = DataSetCatCon(X, y, cat_idxs, task, continuous_mean_std)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _classification_scores(model, dloader, device, task, vision_dset, attentiontype):
    model.eval()
    softmax = nn.Softmax(dim=1)
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    prob = torch.empty(0).to(device)
    with torch.no_grad():
        for data in dloader:
            x_categ, x_cont, y_gts, cat_mask, con_mask, cat_ips, con_ips, rowips = [
                item.to(device) for item in data
            ]
            _, x_categ_enc, x_cont_enc = embed_data_mask(
                x_categ, x_cont, cat_mask, con_mask, model, vision_dset
            )
            if attentiontype == 'col':
                reps = model.transformer(x_categ_enc, x_cont_enc, con_mask, cat_mask, cat_ips, con_ips)
            else:
                reps = model.transformer(x_categ_enc, x_cont_enc, con_mask, cat_mask, cat_ips, con_ips, rowips)
            y_outs = model.mlpfory(reps[:, 0, :])
            y_test = torch.cat([y_test, y_gts.squeeze().float()], dim=0)
            y_pred = torch.cat([y_pred, torch.argmax(y_outs, dim=1).float()], dim=0)
            if task == 'binary':
                prob = torch.cat([prob, softmax(y_outs)[:, -1].float()], dim=0)
            elif task == 'multiclass':
                prob = torch.cat([prob, softmax(y_outs).float()], dim=0)

    acc = (y_pred == y_test).sum().float() / y_test.shape[0] * 100
    if task == 'binary':
        auc = roc_auc_score(y_score=prob.cpu(), y_true=y_test.cpu())
    else:
        auc = roc_auc_score(y_score=prob.cpu(), y_true=y_test.cpu(), multi_class='ovo')
    return acc.cpu().numpy(), auc


def _mean_sq_error(model, dloader, device, vision_dset, attentiontype):
    model.eval()
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    with torch.no_grad():
        for data in dloader:
            x_categ, x_cont, y_gts, cat_mask, con_mask, cat_ips, con_ips, rowips = [
                item.to(device) for item in data
            ]
            _, x_categ_enc, x_cont_enc = embed_data_mask(
                x_categ, x_cont, cat_mask, con_mask, model, vision_dset
            )
            if attentiontype == 'col':
                reps = model.transformer(x_categ_enc, x_cont_enc, con_mask, cat_mask, cat_ips, con_ips)
            else:
                reps = model.transformer(x_categ_enc, x_cont_enc, con_mask, cat_mask, cat_ips, con_ips, rowips)
            y_outs = model.mlpfory(reps[:, 0, :])
            y_test = torch.cat([y_test, y_gts.squeeze().float()], dim=0)
            y_pred = torch.cat([y_pred, y_outs.squeeze()], dim=0)
    y_true_np = y_test.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    rmse = float(np.sqrt(np.mean((y_true_np - y_pred_np) ** 2)))
    r2 = r2_score(y_true_np, y_pred_np)
    return float(rmse), float(r2)


def _count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _get_scheduler(params, optimizer):
    epochs = int(params.get("epoch", params.get("epochs", 200)))
    scheduler_name = params.get("scheduler", "cosine")
    if scheduler_name == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[epochs // 2.667, epochs // 1.6, epochs // 1.142],
        gamma=0.1,
    )


def train_miss_saint_algorithm(data, missing_mask=None, params=None, device=None, full_data=None,
                               labels=None, label_column=None):
    """Strict in-memory wrapper for the original MISS-SAINT train.py flow.

    `data` is the missing table (`datamiss`). `full_data` is required because
    original MISS-SAINT uses it in `sampling()` to simulate expert-filled
    examples before computing IPS.
    """
    params = params or {}
    if full_data is None:
        raise ValueError("Strict MISS-SAINT requires full_data, matching the original train.py datafull input.")

    datamiss_df, datafull_df, label_column = _prepare_dataframes(
        data, full_data, label_column=label_column, labels=labels
    )
    task = params.get("task") or _infer_task(datamiss_df[label_column])
    dtask = 'reg' if task == 'regression' else 'clf'
    seed = int(params.get("random_state", params.get("seed", 0)))
    _set_seed(seed)

    ips_num = int(params.get("ips_num", 40))
    sampling_method = params.get("sampling_method", "feature")
    datamiss_sampled, indicator = sampling(datafull_df, datamiss_df, ips_num, method=sampling_method)
    sampled_df = pd.DataFrame(datamiss_sampled, columns=datamiss_df.columns, index=datamiss_df.index)
    sampled_df = _restore_sampled_dtypes(sampled_df, datamiss_df)

    ips = compute_ips(
        sampled_df.drop(columns=[label_column]).to_numpy(),
        indicator[:, :sampled_df.shape[1] - 1],
        num=ips_num,
        method=params.get("ips_method", "xgb"),
        observed_num=int(params.get("observed_num", 1)),
        complete_sample=params.get("complete_sample", "no-Random"),
    )
    ips_tensor = torch.tensor(ips, dtype=torch.float32)
    row_softmax = nn.Softmax(dim=0)
    col_softmax = nn.Softmax(dim=1)
    row_ips = row_softmax(torch.sum(ips_tensor, dim=1)).numpy()
    ips = col_softmax(ips_tensor).numpy()

    prepared = _build_strict_dataset(
        sampled_df,
        indicator,
        ips,
        row_ips,
        task=task,
        seed=int(params.get("dset_seed", seed)),
        label_column=label_column,
        datasplit=params.get("datasplit", [.8, .1, .1]),
        cat_idxs=params.get("cat_idxs"),
        cat_threshold=int(params.get("cat_threshold", 100)),
        strict_fillna=bool(params.get("strict_fillna", True)),
    )

    batch_size = int(params.get("batch_size", params.get("batchsize", 256)))
    nfeat = prepared["X_train"]["data"].shape[1]
    embedding_size = int(params.get("embedding_size", 32))
    attentiontype = params.get("attentiontype", "col")
    transformer_depth = int(params.get("transformer_depth", 6))
    attention_heads = int(params.get("attention_heads", 8))
    attention_dropout = float(params.get("attention_dropout", 0.1))
    ff_dropout = float(params.get("ff_dropout", 0.1))

    if nfeat > 100:
        embedding_size = min(8, embedding_size)
        batch_size = min(64, batch_size)
    if attentiontype != 'col':
        transformer_depth = 1
        attention_heads = min(4, attention_heads)
        attention_dropout = 0.8
        embedding_size = min(32, embedding_size)
        ff_dropout = 0.8

    continuous_mean_std = np.array([prepared["train_mean"], prepared["train_std"]], dtype=np.float32)
    trainloader = _build_loader(
        prepared["X_train"], prepared["y_train"], prepared["cat_idxs"], dtask,
        continuous_mean_std, batch_size, True
    )
    validloader = _build_loader(
        prepared["X_valid"], prepared["y_valid"], prepared["cat_idxs"], dtask,
        continuous_mean_std, batch_size, False
    )
    testloader = _build_loader(
        prepared["X_test"], prepared["y_test"], prepared["cat_idxs"], dtask,
        continuous_mean_std, batch_size, False
    )

    if task == 'regression':
        y_dim = 1
    else:
        y_dim = len(np.unique(prepared["y_train"]['data'][:, 0]))

    cat_dims = np.append(np.array([1]), np.array(prepared["cat_dims"])).astype(int)
    device = torch.device(device or params.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = SAINT(
        categories=tuple(cat_dims),
        num_continuous=len(prepared["con_idxs"]),
        dim=embedding_size,
        dim_out=1,
        depth=transformer_depth,
        heads=attention_heads,
        attn_dropout=attention_dropout,
        ff_dropout=ff_dropout,
        mlp_hidden_mults=(4, 2),
        cont_embeddings=params.get("cont_embeddings", "MLP"),
        attentiontype=attentiontype,
        final_mlp_style=params.get("final_mlp_style", "sep"),
        y_dim=y_dim,
    ).to(device)

    if y_dim == 2 and task == 'binary':
        criterion = nn.CrossEntropyLoss().to(device)
    elif y_dim > 2 and task == 'multiclass':
        criterion = nn.CrossEntropyLoss().to(device)
    elif task == 'regression':
        criterion = nn.MSELoss().to(device)
    else:
        raise ValueError("Unsupported task/y_dim combination.")

    optimizer_name = params.get("optimizer", "AdamW")
    lr = float(params.get("lr", 0.0001))
    if optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        scheduler = _get_scheduler(params, optimizer)
    elif optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = None
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scheduler = None

    epochs = int(params.get("epoch", params.get("epochs", 200)))
    vision_dset = bool(params.get("vision_dset", False))
    train_loss = []
    val_scores = []
    test_scores = []
    best_state = None
    best_valid_accuracy = 0
    best_test_accuracy = 0
    best_test_auroc = 0
    best_valid_rmse = float("inf")
    best_test_rmse = float("inf")
    best_test_r2 = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in trainloader:
            optimizer.zero_grad()
            x_categ, x_cont, y_gts, cat_mask, con_mask, cat_ips, con_ips, row_ips_batch = [
                item.to(device) for item in batch
            ]
            _, x_categ_enc, x_cont_enc = embed_data_mask(
                x_categ, x_cont, cat_mask, con_mask, model, vision_dset
            )
            if attentiontype == 'col':
                reps = model.transformer(x_categ_enc, x_cont_enc, con_mask, cat_mask, cat_ips, con_ips)
            else:
                reps = model.transformer(x_categ_enc, x_cont_enc, con_mask, cat_mask, cat_ips, con_ips, row_ips_batch)

            y_reps = reps[:, 0, :]
            cat_outs = model.mlp1(reps[:, :model.num_categories, :])
            con_outs = model.mlp2(reps[:, model.num_categories:, :])
            y_outs = model.mlpfory(y_reps)

            if task == 'regression':
                loss = criterion(y_outs, y_gts)
            else:
                loss = criterion(y_outs, y_gts.squeeze().long())

            if len(con_outs) > 0:
                con_recon = torch.cat(con_outs, dim=1)
                l2 = criterion2(con_recon, x_cont)
                l2[con_mask == 0] = 0
                l2 = l2.mean()
            else:
                l2 = 0

            l1 = 0
            n_cat = x_categ.shape[-1]
            for j in range(1, n_cat):
                log_x = criterion3(cat_outs[j])
                log_x = log_x[range(cat_outs[j].shape[0]), x_categ[:, j]]
                log_x[cat_mask[:, j] == 0] = 0
                l1 += abs(sum(log_x) / cat_outs[j].shape[0])

            if params.get("include_reconstruction_loss", False):
                loss = loss + float(params.get("lam2", 1)) * l1 + float(params.get("lam3", 10)) * l2

            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            running_loss += float(loss.item())

        train_loss.append(running_loss)
        model.eval()
        with torch.no_grad():
            if task in ['binary', 'multiclass']:
                valid_acc, valid_auc = _classification_scores(model, validloader, device, task, vision_dset, attentiontype)
                test_acc, test_auc = _classification_scores(model, testloader, device, task, vision_dset, attentiontype)
                val_scores.append((float(valid_acc), float(valid_auc)))
                test_scores.append((float(test_acc), float(test_auc)))
                if valid_acc > best_valid_accuracy:
                    best_valid_accuracy = valid_acc
                    best_test_accuracy = test_acc
                    best_test_auroc = test_auc
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                valid_rmse, valid_r2 = _mean_sq_error(model, validloader, device, vision_dset, attentiontype)
                test_rmse, test_r2 = _mean_sq_error(model, testloader, device, vision_dset, attentiontype)
                val_scores.append((valid_rmse, valid_r2))
                test_scores.append((test_rmse, test_r2))
                if valid_rmse < best_valid_rmse:
                    best_valid_rmse = valid_rmse
                    best_test_rmse = test_rmse
                    best_test_r2 = test_r2
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if params.get("verbose", False):
            print(f"[MISS-SAINT] epoch {epoch + 1}/{epochs}, train_loss={running_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics = {
        "train_loss": train_loss,
        "val_scores": val_scores,
        "test_scores": test_scores,
        "total_parameters": _count_parameters(model),
    }
    if task in ['binary', 'multiclass']:
        metrics.update({
            "best_valid_accuracy": float(best_valid_accuracy),
            "best_test_accuracy": float(best_test_accuracy),
            "best_test_auroc": float(best_test_auroc),
        })
    else:
        metrics.update({
            "best_valid_rmse": float(best_valid_rmse),
            "best_test_rmse": float(best_test_rmse),
            "best_test_r2": float(best_test_r2),
        })

    return {
        "model": model,
        "prepared": prepared,
        "params": dict(params),
        "task": task,
        "dtask": dtask,
        "device": str(device),
        "attentiontype": attentiontype,
        "vision_dset": vision_dset,
        "continuous_mean_std": continuous_mean_std,
        "metrics": metrics,
        "sampled_data": sampled_df,
        "indicator": indicator,
        "ips": ips,
        "row_ips": row_ips,
        "label_column": label_column,
        "input_columns": list(datamiss_df.columns),
        "index": datamiss_df.index,
    }


def _prepare_predict_frame(data, train_state):
    df = _as_dataframe(data, columns=train_state["input_columns"])
    label_column = train_state["label_column"]
    if label_column not in df.columns:
        df[label_column] = train_state["sampled_data"][label_column].values[:len(df)]
    return df[train_state["input_columns"]].copy()


def _missing_frame_from_mask(df, label_column, missing_mask):
    X_missing = df.drop(columns=[label_column]).isna()
    if missing_mask is None:
        return X_missing

    mask = np.asarray(missing_mask)
    if mask.ndim != 2:
        raise ValueError("missing_mask must be a 2D array.")

    feature_columns = df.drop(columns=[label_column]).columns
    if mask.shape == df.shape:
        label_pos = df.columns.get_loc(label_column)
        feature_positions = [i for i in range(df.shape[1]) if i != label_pos]
        feature_mask = mask[:, feature_positions]
    elif mask.shape == (len(df), len(feature_columns)):
        feature_mask = mask
    else:
        raise ValueError("missing_mask shape must match either the full table or feature table.")

    return X_missing | pd.DataFrame(feature_mask == 0, index=df.index, columns=feature_columns)


def _encoder_missing_token(encoder):
    classes = set(encoder.classes_)
    if "MissingValue" in classes:
        return "MissingValue"
    if "nan" in classes:
        return "nan"
    return encoder.classes_[0]


def _safe_numeric_cast(values, template):
    index = values.index if isinstance(values, pd.Series) else template.index
    series = pd.Series(values, index=index)
    if pd.api.types.is_numeric_dtype(template):
        converted = pd.to_numeric(series, errors="coerce")
        return converted.where(converted.notna(), series)
    return series


def _transform_for_prediction(df, train_state, missing_mask=None):
    prepared = train_state["prepared"]
    label_column = train_state["label_column"]
    X = df.drop(columns=[label_column]).copy()
    original_missing = _missing_frame_from_mask(df, label_column, missing_mask)
    X = X.mask(original_missing)
    nan_mask = (~original_missing).astype(int)

    for col, encoder in prepared["encoders"].items():
        missing_token = _encoder_missing_token(encoder)
        values = X[col].where(~X[col].isna(), missing_token).astype("str")
        known = set(encoder.classes_)
        values = values.where(values.isin(known), missing_token)
        X[col] = encoder.transform(values)

    for col, fill_value in prepared["fill_values"].items():
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(fill_value)

    ips = pd.DataFrame(np.ones_like(X.to_numpy(dtype=np.float32)), columns=X.columns, index=X.index)
    rowips = pd.DataFrame(np.ones((len(X),), dtype=np.float32), index=X.index)
    y = np.zeros((len(X),), dtype=np.float32)
    x_d, y_d = _data_split(X, y, nan_mask, ips, rowips, np.arange(len(X)))
    return X, x_d, original_missing


def predict_miss_saint_algorithm(model, data, missing_mask=None, train_state=None, device=None):
    if train_state is None:
        raise ValueError("train_state is required for MISS-SAINT prediction.")

    df = _prepare_predict_frame(data, train_state)
    X_processed, x_d, original_missing = _transform_for_prediction(df, train_state, missing_mask)
    prepared = train_state["prepared"]
    device = torch.device(device or train_state.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    model.eval()

    ds = DataSetCatCon(
        x_d,
        {'data': np.zeros((len(df), 1), dtype=np.float32)},
        prepared["cat_idxs"],
        train_state["dtask"],
        train_state["continuous_mean_std"],
    )
    loader = DataLoader(ds, batch_size=int(train_state["params"].get("batch_size", 256)), shuffle=False, num_workers=0)

    feature_result = df.drop(columns=[train_state["label_column"]]).copy()
    con_idxs = prepared["con_idxs"]
    cat_idxs = prepared["cat_idxs"]
    con_values = []
    cat_values = {idx: [] for idx in cat_idxs}

    with torch.no_grad():
        for batch in loader:
            x_categ, x_cont, y_gts, cat_mask, con_mask, cat_ips, con_ips, rowips = [
                item.to(device) for item in batch
            ]
            _, x_categ_enc, x_cont_enc = embed_data_mask(
                x_categ, x_cont, cat_mask, con_mask, model, train_state["vision_dset"]
            )
            if train_state["attentiontype"] == 'col':
                reps = model.transformer(x_categ_enc, x_cont_enc, con_mask, cat_mask, cat_ips, con_ips)
            else:
                reps = model.transformer(x_categ_enc, x_cont_enc, con_mask, cat_mask, cat_ips, con_ips, rowips)

            cat_outs = model.mlp1(reps[:, :model.num_categories, :])
            con_outs = model.mlp2(reps[:, model.num_categories:, :])
            if len(con_outs) > 0:
                con_batch = torch.cat(con_outs, dim=1).cpu().numpy()
                mean, std = train_state["continuous_mean_std"]
                con_values.append(con_batch * std + mean)

            for local_j, feature_idx in enumerate(cat_idxs, start=1):
                preds = torch.argmax(cat_outs[local_j], dim=1).cpu().numpy()
                cat_values[feature_idx].append(preds)

    if con_values:
        con_values = np.vstack(con_values)
        for local_idx, feature_idx in enumerate(con_idxs):
            col = feature_result.columns[feature_idx]
            miss_rows = original_missing[col].to_numpy()
            feature_result.loc[miss_rows, col] = con_values[miss_rows, local_idx]

    for feature_idx, chunks in cat_values.items():
        if not chunks:
            continue
        encoded = np.concatenate(chunks)
        col = feature_result.columns[feature_idx]
        miss_rows = original_missing[col].to_numpy()
        if not miss_rows.any():
            continue
        encoder = prepared["encoders"][col]
        decoded = encoder.inverse_transform(encoded.astype(int))
        fill_value = prepared.get("categorical_modes", {}).get(col)
        missing_tokens = {"MissingValue", "nan", "None", "<NA>"}
        decoded_series = pd.Series(decoded[miss_rows], index=feature_result.index[miss_rows])
        if fill_value is not None:
            decoded_series = decoded_series.where(~decoded_series.astype("str").isin(missing_tokens), fill_value)
        decoded_series = _safe_numeric_cast(decoded_series, feature_result[col])
        feature_result.loc[miss_rows, col] = decoded_series

    result = df.copy()
    for col in feature_result.columns:
        result[col] = feature_result[col]
    return result

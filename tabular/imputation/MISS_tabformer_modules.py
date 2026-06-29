import copy
import math

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from torch import einsum, nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from einops import rearrange
except ImportError as exc:
    raise ImportError("MISS-tabformer requires einops, matching the original TabTransformer code.") from exc

try:
    import dataprep.tabular.imputation.MISS_saint_modules as saint_modules
except ImportError:
    import tabular.imputation.MISS_saint_modules as saint_modules


def _set_seed(seed):
    saint_modules._set_seed(seed)


def _as_dataframe(data, columns=None):
    if isinstance(data, pd.DataFrame):
        return data.copy()
    return pd.DataFrame(data, columns=columns)


def _infer_task(y):
    return saint_modules._infer_task(y)


def _prepare_dataframes(data, full_data, label_column=None, labels=None):
    return saint_modules._prepare_dataframes(
        data,
        full_data,
        label_column=label_column,
        labels=labels,
    )


def _restore_sampled_dtypes(sampled_df, reference_df):
    return saint_modules._restore_sampled_dtypes(sampled_df, reference_df)


def _safe_numeric_ips_frame(X):
    result = pd.DataFrame(index=X.index)
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            result[col] = pd.to_numeric(X[col], errors="coerce")
        else:
            codes, _ = pd.factorize(X[col].astype("string"), sort=True)
            result[col] = codes.astype(float)
            result.loc[X[col].isna(), col] = np.nan
    return result


def _split_indices(n_rows, datasplit, seed):
    rng = np.random.default_rng(seed)
    labels = rng.choice(["train", "valid", "test"], p=datasplit, size=n_rows)
    train = np.where(labels != "test")[0]
    test = np.where(labels == "test")[0]
    if len(train) == 0:
        train = np.arange(n_rows)
    if len(test) == 0:
        test = train[:min(len(train), 1)]
    return train, test


def _choose_categorical_columns(X, cat_idxs=None, cat_threshold=100):
    if cat_idxs is None:
        nunique = X.nunique(dropna=True)
        types = X.dtypes
        categorical_indicator = np.zeros(X.shape[1], dtype=bool)
        for col in X.columns:
            if types[col] == "object" or str(types[col]) == "category" or nunique[col] < cat_threshold:
                categorical_indicator[X.columns.get_loc(col)] = True
        cat_idxs = list(np.where(categorical_indicator)[0])
    else:
        cat_idxs = list(cat_idxs)
    con_idxs = [idx for idx in range(X.shape[1]) if idx not in set(cat_idxs)]
    return cat_idxs, con_idxs


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


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
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x, **kwargs):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=16, dropout=0.0):
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
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.bool()
            missing_mask = ~key_padding_mask
            if ips is None:
                ips = torch.ones_like(key_padding_mask, dtype=sim.dtype)
            sim = sim * ips.to(sim.device).unsqueeze(1).unsqueeze(2)
            all_missing = missing_mask.all(dim=1)
            if all_missing.any():
                missing_mask = missing_mask.clone()
                missing_mask[all_missing] = False
            sim = sim.masked_fill(missing_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = sim.softmax(dim=-1).float()
        attn = self.dropout(attn)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, num_tokens, dim, depth, heads, dim_head, attn_dropout, ff_dropout):
        super().__init__()
        self.embeds = nn.Embedding(num_tokens, dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout))),
                Residual(PreNorm(dim, FeedForward(dim, dropout=ff_dropout))),
            ]))

    def forward(self, x, cat_mask=None, cat_ips=None):
        x = self.embeds(x)
        for attn1, ff1 in self.layers:
            if cat_mask is None:
                x = attn1(x)
            else:
                x = attn1(x, key_padding_mask=cat_mask, ips=cat_ips)
            x = ff1(x)
        return x


class MLP(nn.Module):
    def __init__(self, dims, act=None):
        super().__init__()
        dims_pairs = list(zip(dims[:-1], dims[1:]))
        layers = []
        for ind, (dim_in, dim_out) in enumerate(dims_pairs):
            is_last = ind >= (len(dims_pairs) - 1)
            layers.append(nn.Linear(dim_in, dim_out))
            if is_last:
                self.dim_out = dim_out
                continue
            layers.append(default(act, nn.ReLU()))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        x = self.mlp(x)
        if self.dim_out > 1:
            x = torch.softmax(x, dim=1)
        return x


class simple_MLP(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.ReLU(),
            nn.Linear(dims[1], dims[2]),
        )

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.view(x.size(0), -1)
        return self.layers(x)


class sep_MLP(nn.Module):
    def __init__(self, dim, len_feats, categories):
        super().__init__()
        self.layers = nn.ModuleList([
            simple_MLP([dim, 5 * dim, int(categories[i])])
            for i in range(len_feats)
        ])

    def forward(self, x):
        return [layer(x[:, i, :]) for i, layer in enumerate(self.layers)]


class Lora(nn.Module):
    def __init__(self, m, n, rank=10):
        super().__init__()
        self.m = m
        self.A = nn.Parameter(torch.randn(m, rank))
        self.B = nn.Parameter(torch.zeros(rank, n))

    def forward(self, x):
        x = x.view(-1, self.m)
        return torch.mm(torch.mm(x, self.A), self.B)


class TabTransformerModel(nn.Module):
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
            num_special_tokens=2,
            continuous_mean_std=None,
            attn_dropout=0.,
            ff_dropout=0.):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)
        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens

        categories_offset = F.pad(
            torch.tensor(list(categories), dtype=torch.long),
            (1, 0),
            value=num_special_tokens,
        ).cumsum(dim=-1)[:-1]
        self.register_buffer('categories_offset', categories_offset)

        if exists(continuous_mean_std):
            assert continuous_mean_std.shape == (num_continuous, 2)
        self.register_buffer('continuous_mean_std', continuous_mean_std)
        self.num_continuous = num_continuous
        if num_continuous > 0:
            self.norm = nn.LayerNorm(num_continuous)
        else:
            self.norm = None

        if self.num_categories > 0:
            self.lora = Lora(len(categories), len(categories) * dim, 10)
            self.transformer = Transformer(
                num_tokens=total_tokens,
                dim=dim,
                depth=depth,
                heads=heads,
                dim_head=dim_head,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
            )
            self.pt_mlp = simple_MLP([
                dim * self.num_categories,
                max(1, 6 * dim * self.num_categories // 5),
                max(1, dim * self.num_categories // 2),
            ])
            self.mlp1 = sep_MLP(dim, self.num_categories, categories)
        else:
            self.lora = None
            self.transformer = None
            self.pt_mlp = None
            self.mlp1 = None

        input_size = (dim * self.num_categories) + num_continuous
        hidden_base = max(1, input_size // 8)
        hidden_dimensions = list(map(lambda t: hidden_base * t, mlp_hidden_mults))
        all_dimensions = [input_size, *hidden_dimensions, dim_out]
        self.mlp = MLP(all_dimensions, act=mlp_act)

    def forward(self, x_categ, x_cont, x_categ_mask, x_categ_ips):
        if x_categ is not None and self.num_categories > 0:
            assert x_categ.shape[-1] == self.num_categories
            x_categ = x_categ + self.categories_offset
            x = self.transformer(x_categ, x_categ_mask, x_categ_ips)
            flat_categ = x.flatten(1)
            flat_categ = flat_categ + self.lora(x_categ.float())
        else:
            flat_categ = None

        if self.num_continuous > 0:
            assert x_cont.shape[1] == self.num_continuous
            if exists(self.continuous_mean_std):
                mean, std = self.continuous_mean_std.unbind(dim=-1)
                x_cont = (x_cont - mean) / std
            normed_cont = self.norm(x_cont)
        else:
            normed_cont = None

        if flat_categ is not None and normed_cont is not None:
            x = torch.cat((flat_categ, normed_cont), dim=-1)
        elif flat_categ is not None:
            x = flat_categ
        elif normed_cont is not None:
            x = normed_cont
        else:
            raise ValueError("MISS-tabformer requires at least one feature.")
        return self.mlp(x)


def _fit_preprocessing(sampled_df, datafull_df, indicator, ips, label_column, task, params):
    X = sampled_df.drop(columns=[label_column]).copy()
    X_full = datafull_df.drop(columns=[label_column]).copy()
    feature_columns = list(X.columns)
    cat_idxs, con_idxs = _choose_categorical_columns(
        X,
        params.get("cat_idxs"),
        int(params.get("cat_threshold", 100)),
    )
    cat_cols = [feature_columns[idx] for idx in cat_idxs]
    con_cols = [feature_columns[idx] for idx in con_idxs]

    original_missing = pd.DataFrame(
        pd.isna(sampled_df.drop(columns=[label_column])).to_numpy(),
        index=X.index,
        columns=feature_columns,
    )
    original_missing[pd.DataFrame(indicator[:, :len(feature_columns)], index=X.index, columns=feature_columns) == 0.5] = True
    observed_mask = pd.DataFrame(
        indicator[:, :len(feature_columns)] != 0,
        index=X.index,
        columns=feature_columns,
    )

    processed = pd.DataFrame(index=X.index)
    full_processed = pd.DataFrame(index=X.index)
    fill_values = {}
    encoders = {}
    categorical_modes = {}
    cat_cardinalities = []

    for col in cat_cols:
        values = pd.concat([X[col], X_full[col]], axis=0).astype("string").fillna("MissingValue")
        encoder = LabelEncoder()
        encoder.fit(values.to_numpy())
        encoders[col] = encoder
        categorical_modes[col] = X[col].dropna().mode(dropna=True).iloc[0] if len(X[col].dropna()) else None
        encoded = pd.Series(encoder.transform(X[col].astype("string").fillna("MissingValue")), index=X.index).astype(float)
        encoded_full = pd.Series(encoder.transform(X_full[col].astype("string").fillna("MissingValue")), index=X.index).astype(float)
        encoded = encoded.mask(X[col].isna())
        fill_value = encoded.mean()
        if not np.isfinite(fill_value):
            fill_value = 0.0
        fill_values[col] = float(fill_value)
        processed[col] = encoded.fillna(fill_value)
        full_processed[col] = encoded_full.fillna(fill_value)
        max_code = int(max(processed[col].max(), full_processed[col].max(), len(encoder.classes_) - 1))
        cat_cardinalities.append(max_code + 1)

    for col in con_cols:
        values = pd.to_numeric(X[col], errors="coerce")
        full_values = pd.to_numeric(X_full[col], errors="coerce")
        fill_value = values.mean()
        if not np.isfinite(fill_value):
            fill_value = full_values.mean()
        if not np.isfinite(fill_value):
            fill_value = 0.0
        fill_values[col] = float(fill_value)
        processed[col] = values.fillna(fill_value)
        full_processed[col] = full_values.fillna(fill_value)

    processed = processed[feature_columns]
    full_processed = full_processed[feature_columns]

    y = sampled_df[label_column].copy()
    y_encoder = None
    if task == "regression":
        y_values = pd.to_numeric(y, errors="coerce")
        y_array = y_values.fillna(y_values.mean()).fillna(0).to_numpy(dtype=np.float32)
    else:
        y_encoder = LabelEncoder()
        y_array = y_encoder.fit_transform(y.astype("string").fillna("MissingLabel")).astype(np.int64)

    ips_df = pd.DataFrame(ips, index=X.index, columns=feature_columns)
    ips_softmax = torch.softmax(torch.tensor(ips_df.to_numpy(dtype=np.float32)), dim=1).numpy()
    train_idx, test_idx = _split_indices(
        len(X),
        params.get("datasplit", [.8, .1, .1]),
        int(params.get("dset_seed", params.get("random_state", 0))),
    )

    return {
        "feature_columns": feature_columns,
        "label_column": label_column,
        "cat_idxs": cat_idxs,
        "con_idxs": con_idxs,
        "categorical_columns": cat_cols,
        "continuous_columns": con_cols,
        "cat_cardinalities": cat_cardinalities,
        "encoders": encoders,
        "categorical_modes": categorical_modes,
        "fill_values": fill_values,
        "processed_X": processed,
        "full_processed_X": full_processed,
        "observed_mask": observed_mask,
        "original_missing": original_missing,
        "ips": pd.DataFrame(ips_softmax, index=X.index, columns=feature_columns),
        "y": y_array,
        "y_encoder": y_encoder,
        "train_indices": train_idx,
        "test_indices": test_idx,
    }


def _build_tensors(prepared, device):
    X = prepared["processed_X"]
    cat_cols = prepared["categorical_columns"]
    con_cols = prepared["continuous_columns"]
    x_cat = X[cat_cols].to_numpy(dtype=np.float32).astype(np.int64) if cat_cols else np.empty((len(X), 0), dtype=np.int64)
    x_cont = X[con_cols].to_numpy(dtype=np.float32) if con_cols else np.empty((len(X), 0), dtype=np.float32)
    cat_mask = prepared["observed_mask"][cat_cols].to_numpy(dtype=np.int64) if cat_cols else np.empty((len(X), 0), dtype=np.int64)
    cat_ips = prepared["ips"][cat_cols].to_numpy(dtype=np.float32) if cat_cols else np.empty((len(X), 0), dtype=np.float32)
    return {
        "x_cat": torch.tensor(x_cat, dtype=torch.long, device=device),
        "x_cont": torch.tensor(x_cont, dtype=torch.float32, device=device),
        "cat_mask": torch.tensor(cat_mask, dtype=torch.long, device=device),
        "cat_ips": torch.tensor(cat_ips, dtype=torch.float32, device=device),
        "y": torch.tensor(prepared["y"], device=device),
    }


def _task_output_dim(task, y):
    if task == "regression" or task == "binary":
        return 1
    return int(np.max(y) + 1)


def _loss_fn(task):
    if task == "regression":
        return nn.MSELoss()
    if task == "multiclass":
        return nn.CrossEntropyLoss()
    return nn.BCEWithLogitsLoss()


def _supervised_loss(out, y, task, loss_func):
    if task == "regression":
        return loss_func(out.squeeze(), y.float())
    if task == "multiclass":
        return loss_func(out, y.long())
    return loss_func(out.squeeze(), y.float())


def _model_outputs(model, tensors, row_idx):
    x_cat = tensors["x_cat"][row_idx] if tensors["x_cat"].shape[1] else None
    x_cont = tensors["x_cont"][row_idx]
    cat_mask = tensors["cat_mask"][row_idx] if tensors["cat_mask"].shape[1] else None
    cat_ips = tensors["cat_ips"][row_idx] if tensors["cat_ips"].shape[1] else None
    return model(x_cat, x_cont, cat_mask, cat_ips)


def _pretrain_tabformer(model, tensors, train_idx, params, device):
    if model.num_categories == 0:
        return []

    model.train()
    batch_size = int(params.get("batch_size", 128))
    epochs = int(params.get("pretrain_epoch", 20))
    loader = DataLoader(
        TensorDataset(torch.tensor(train_idx, device=device)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(params.get("pretrain_lr", 0.0001)))
    criterion_ce = nn.CrossEntropyLoss()
    log_softmax = nn.LogSoftmax(dim=1)
    history = []

    for epoch in range(epochs):
        running_loss = 0.0
        for (row_idx,) in loader:
            x_categ = tensors["x_cat"][row_idx]
            x_categ_mask = tensors["cat_mask"][row_idx]
            x_categ_ips = tensors["cat_ips"][row_idx]

            optimizer.zero_grad()
            aug_1 = model.transformer(x_categ, x_categ_mask, x_categ_ips)
            aug_2 = model.transformer(x_categ, x_categ_mask, x_categ_ips)
            aug_1 = (aug_1 / aug_1.norm(dim=-1, keepdim=True).clamp_min(1e-8)).flatten(1, 2)
            aug_2 = (aug_2 / aug_2.norm(dim=-1, keepdim=True).clamp_min(1e-8)).flatten(1, 2)
            aug_1 = model.pt_mlp(aug_1)
            aug_2 = model.pt_mlp(aug_2)
            logits_1 = aug_1 @ aug_2.t() / 0.7
            logits_2 = aug_2 @ aug_1.t() / 0.7
            targets = torch.arange(logits_1.size(0), device=device)
            loss = (criterion_ce(logits_1, targets) + criterion_ce(logits_2, targets)) / 2

            cat_outs = model.mlp1(model.transformer(x_categ, x_categ_mask, x_categ_ips))
            l1 = torch.tensor(0.0, device=device)
            for j in range(1, x_categ.shape[-1]):
                log_x = log_softmax(cat_outs[j])
                selected = log_x[range(cat_outs[j].shape[0]), x_categ[:, j]]
                selected = selected.masked_fill(x_categ_mask[:, j] == 0, 0)
                l1 = l1 + torch.abs(selected.sum() / cat_outs[j].shape[0])
                l1 = l1 + criterion_ce(cat_outs[j], x_categ[:, j])
            loss = loss + l1
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu())

        history.append(running_loss)
        if params.get("verbose", False):
            print(f"[MISS-tabformer pretrain] epoch {epoch + 1}/{epochs}, loss={running_loss:.6f}")
    return history


@torch.no_grad()
def _evaluate(model, tensors, indices, task, batch_size):
    model.eval()
    y_true = []
    y_pred = []
    y_prob = []
    loader = DataLoader(
        TensorDataset(torch.tensor(indices, device=tensors["y"].device)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    for (row_idx,) in loader:
        out = _model_outputs(model, tensors, row_idx)
        y_true.append(tensors["y"][row_idx].detach().cpu())
        if task == "regression":
            y_pred.append(out.squeeze().detach().cpu())
        elif task == "multiclass":
            y_prob.append(out.detach().cpu())
            y_pred.append(torch.argmax(out, dim=1).detach().cpu())
        else:
            prob = torch.sigmoid(out.squeeze())
            y_prob.append(prob.detach().cpu())
            y_pred.append((prob > 0.5).long().detach().cpu())

    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    if task == "regression":
        rmse = float(mean_squared_error(y_true, y_pred, squared=False))
        return {"score": -rmse, "rmse": rmse, "r2": float(r2_score(y_true, y_pred))}

    acc = float(accuracy_score(y_true, y_pred))
    metrics = {"score": acc, "accuracy": acc}
    try:
        prob = torch.cat(y_prob).numpy()
        if task == "binary":
            metrics["roc_auc"] = float(roc_auc_score(y_true, prob))
        else:
            metrics["roc_auc"] = float(roc_auc_score(y_true, prob, multi_class="ovo"))
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics


def _decode_filled_frame(prepared):
    result = prepared["processed_X"].copy()
    for col in prepared["categorical_columns"]:
        encoded = result[col].to_numpy(dtype=np.float32).astype(int)
        encoded = np.clip(encoded, 0, len(prepared["encoders"][col].classes_) - 1)
        decoded = prepared["encoders"][col].inverse_transform(encoded)
        series = pd.Series(decoded, index=result.index)
        mode = prepared["categorical_modes"].get(col)
        if mode is not None:
            series = series.where(series != "MissingValue", mode)
        result[col] = saint_modules._safe_numeric_cast(series, result[col])
    return result[prepared["feature_columns"]]


def train_miss_tabformer_algorithm(data, missing_mask=None, params=None, device=None, full_data=None,
                                   labels=None, label_column=None):
    params = params or {}
    if full_data is None:
        raise ValueError("Strict MISS-tabformer requires full_data for expert sampling and IPS.")

    datamiss_df, datafull_df, label_column = _prepare_dataframes(
        data,
        full_data,
        label_column=label_column,
        labels=labels,
    )
    task = params.get("task") or _infer_task(datamiss_df[label_column])
    if task == "classification":
        task = "multiclass"
    seed = int(params.get("random_state", params.get("seed", 0)))
    _set_seed(seed)

    ips_num = int(params.get("ips_num", 20))
    sampled_array, indicator = saint_modules.sampling(
        datafull_df,
        datamiss_df,
        ips_num,
        method=params.get("sampling_method", "feature"),
    )
    sampled_df = pd.DataFrame(sampled_array, columns=datamiss_df.columns, index=datamiss_df.index)
    sampled_df = _restore_sampled_dtypes(sampled_df, datamiss_df)

    feature_for_ips = _safe_numeric_ips_frame(sampled_df.drop(columns=[label_column]))
    ips = saint_modules.compute_ips(
        feature_for_ips.to_numpy(),
        indicator[:, :feature_for_ips.shape[1]],
        num=ips_num,
        method=params.get("ips_method", "xgb"),
        observed_num=int(params.get("observed_num", 1)),
        complete_sample=params.get("complete_sample", "no-Random"),
    )

    prepared = _fit_preprocessing(sampled_df, datafull_df, indicator, ips, label_column, task, params)
    device = torch.device(device or params.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    tensors = _build_tensors(prepared, device)
    y_dim = _task_output_dim(task, prepared["y"])

    model_dim = int(params.get("dim", 128))
    if len(prepared["feature_columns"]) >= 50:
        model_dim = min(model_dim, 8)
        params = dict(params)
        params["batch_size"] = min(int(params.get("batch_size", 128)), 64)

    model = TabTransformerModel(
        categories=tuple(prepared["cat_cardinalities"]),
        num_continuous=len(prepared["continuous_columns"]),
        dim_out=y_dim,
        mlp_act=nn.ReLU(),
        dim=model_dim,
        depth=int(params.get("depth", 1)),
        heads=int(params.get("heads", 4)),
        attn_dropout=float(params.get("dropout", 0.4)),
        ff_dropout=float(params.get("dropout", 0.4)),
        mlp_hidden_mults=(4, 2),
    ).to(device)

    train_idx = prepared["train_indices"]
    test_idx = prepared["test_indices"]
    pretrain_loss = _pretrain_tabformer(model, tensors, train_idx, params, device)
    for name, p in model.named_parameters():
        if "transformer" in name:
            p.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(params.get("lr", 1e-5)),
        weight_decay=float(params.get("weight_decay", 1e-5)),
    )
    loss_func = _loss_fn(task).to(device)
    batch_size = int(params.get("batch_size", 128))
    val_batch_size = int(params.get("val_batch_size", 128))
    epochs = int(params.get("epoch", params.get("epochs", 1000)))
    early_stopping_rounds = int(params.get("early_stopping_rounds", 20))
    train_loader = DataLoader(
        TensorDataset(torch.tensor(train_idx, device=device)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    min_val_loss = float("inf")
    min_val_loss_idx = 0
    best_state = None
    loss_history = []
    val_loss_history = []
    val_scores = []

    for epoch in range(epochs):
        model.train()
        for (row_idx,) in train_loader:
            out = _model_outputs(model, tensors, row_idx)
            loss = _supervised_loss(out, tensors["y"][row_idx], task, loss_func)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_history.append(float(loss.detach().cpu()))

        model.eval()
        val_loss = 0.0
        val_batches = 0
        val_loader = DataLoader(
            TensorDataset(torch.tensor(test_idx, device=device)),
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=0,
        )
        with torch.no_grad():
            for (row_idx,) in val_loader:
                out = _model_outputs(model, tensors, row_idx)
                val_loss = val_loss + float(_supervised_loss(out, tensors["y"][row_idx], task, loss_func).detach().cpu())
                val_batches += 1
        val_loss = val_loss / max(val_batches, 1)
        val_loss_history.append(val_loss)
        val_metrics = _evaluate(model, tensors, test_idx, task, val_batch_size)
        val_scores.append(val_metrics)

        if params.get("verbose", False):
            print(f"[MISS-tabformer] epoch {epoch + 1}/{epochs}, val_loss={val_loss:.6f}")

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            min_val_loss_idx = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if min_val_loss_idx + early_stopping_rounds < epoch:
            if params.get("verbose", False):
                print(
                    f"[MISS-tabformer] early stopping at epoch {epoch + 1}, "
                    f"best epoch {min_val_loss_idx + 1}"
                )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    imputed_features = _decode_filled_frame(prepared)
    imputed_data = sampled_df.copy()
    for col in prepared["feature_columns"]:
        miss = prepared["original_missing"][col].to_numpy()
        imputed_data.loc[miss, col] = imputed_features.loc[miss, col]

    metrics = {
        "pretrain_loss": pretrain_loss,
        "train_loss": loss_history,
        "val_loss": val_loss_history,
        "val_scores": val_scores,
        "test_scores": list(val_scores),
        "best_val_loss": float(min_val_loss),
        "best_epoch": int(min_val_loss_idx + 1),
        "total_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }

    return {
        "model": model,
        "prepared": prepared,
        "params": dict(params),
        "task": task,
        "device": str(device),
        "metrics": metrics,
        "sampled_data": sampled_df,
        "imputed_data": imputed_data,
        "indicator": indicator,
        "ips": ips,
        "label_column": label_column,
        "input_columns": list(datamiss_df.columns),
        "index": datamiss_df.index,
    }


def _missing_frame_from_mask(df, label_column, missing_mask):
    feature_columns = [col for col in df.columns if col != label_column]
    X_missing = df[feature_columns].isna()
    if missing_mask is None:
        return X_missing
    mask = np.asarray(missing_mask)
    if mask.ndim != 2:
        raise ValueError("missing_mask must be a 2D array.")
    if mask.shape == df.shape:
        label_pos = df.columns.get_loc(label_column)
        feature_positions = [i for i in range(df.shape[1]) if i != label_pos]
        feature_mask = mask[:, feature_positions]
    elif mask.shape == (len(df), len(feature_columns)):
        feature_mask = mask
    else:
        raise ValueError("missing_mask shape must match either the full table or feature table.")
    return X_missing | pd.DataFrame(feature_mask == 0, index=df.index, columns=feature_columns)


def _prepare_predict_frame(data, train_state):
    df = _as_dataframe(data, columns=train_state["input_columns"])
    label_column = train_state["label_column"]
    if label_column not in df.columns:
        df[label_column] = train_state["sampled_data"][label_column].values[:len(df)]
    return df[train_state["input_columns"]].copy()


def predict_miss_tabformer_algorithm(model, data, missing_mask=None, train_state=None, device=None):
    if train_state is None:
        raise ValueError("train_state is required for MISS-tabformer prediction.")

    df = _prepare_predict_frame(data, train_state)
    prepared = train_state["prepared"]
    label_column = train_state["label_column"]
    original_missing = _missing_frame_from_mask(df, label_column, missing_mask)
    result_features = df.drop(columns=[label_column]).copy()

    for col in prepared["continuous_columns"]:
        miss = original_missing[col].to_numpy()
        result_features.loc[miss, col] = prepared["fill_values"][col]

    for col in prepared["categorical_columns"]:
        miss = original_missing[col].to_numpy()
        fill_value = prepared["fill_values"][col]
        encoded_fill = int(fill_value)
        encoded_fill = np.clip(encoded_fill, 0, len(prepared["encoders"][col].classes_) - 1)
        decoded = prepared["encoders"][col].inverse_transform([encoded_fill])[0]
        mode = prepared["categorical_modes"].get(col)
        if decoded == "MissingValue" and mode is not None:
            decoded = mode
        result_features.loc[miss, col] = decoded

    result = df.copy()
    for col in result_features.columns:
        result[col] = result_features[col]
    return result

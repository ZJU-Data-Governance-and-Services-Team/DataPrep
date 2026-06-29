import copy
import math

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

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


def _count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _choose_categorical_columns(X, cat_idxs=None, cat_threshold=100):
    if cat_idxs is None:
        nunique = X.nunique(dropna=True)
        types = X.dtypes
        categorical_indicator = np.zeros(X.shape[1], dtype=bool)
        for col in X.columns:
            if types[col] == 'object' or str(types[col]) == 'category' or nunique[col] < cat_threshold:
                categorical_indicator[X.columns.get_loc(col)] = True
        cat_idxs = list(np.where(categorical_indicator)[0])
    else:
        cat_idxs = list(cat_idxs)
    con_idxs = [idx for idx in range(X.shape[1]) if idx not in set(cat_idxs)]
    return cat_idxs, con_idxs


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
    train = np.where(labels == "train")[0]
    valid = np.where(labels == "valid")[0]
    test = np.where(labels == "test")[0]
    if len(train) == 0:
        train = np.arange(n_rows)
    if len(valid) == 0:
        valid = train[:min(len(train), 1)]
    if len(test) == 0:
        test = valid
    return train, valid, test


def _fit_feature_state(sampled_df, datafull_df, indicator, ips, label_column, task, params):
    X = sampled_df.drop(columns=[label_column]).copy()
    X_full = datafull_df.drop(columns=[label_column]).copy()
    feature_columns = list(X.columns)
    cat_idxs, con_idxs = _choose_categorical_columns(
        X,
        params.get("cat_idxs"),
        int(params.get("cat_threshold", 100)),
    )
    categorical_columns = [feature_columns[idx] for idx in cat_idxs]
    continuous_columns = [feature_columns[idx] for idx in con_idxs]

    train_idx, valid_idx, test_idx = _split_indices(
        len(X),
        params.get("datasplit", [.8, .1, .1]),
        int(params.get("dset_seed", params.get("random_state", 0))),
    )

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

    encoders = {}
    categorical_modes = {}
    cat_cardinalities = []
    X_processed = pd.DataFrame(index=X.index)
    X_full_processed = pd.DataFrame(index=X.index)
    for col in categorical_columns:
        non_missing = X[col].dropna()
        if len(non_missing) > 0:
            mode = non_missing.mode(dropna=True)
            categorical_modes[col] = mode.iloc[0] if len(mode) > 0 else non_missing.iloc[0]
        else:
            categorical_modes[col] = None

        values = pd.concat([X[col], X_full[col]], axis=0).astype("string").fillna("MissingValue")
        encoder = LabelEncoder()
        encoder.fit(values.to_numpy())
        encoders[col] = encoder
        cat_cardinalities.append(len(encoder.classes_))
        X_processed[col] = encoder.transform(X[col].astype("string").fillna("MissingValue"))
        X_full_processed[col] = encoder.transform(X_full[col].astype("string").fillna("MissingValue"))

    fill_values = {}
    means = []
    stds = []
    for col in continuous_columns:
        full_col = pd.to_numeric(X_full[col], errors="coerce")
        train_col = pd.to_numeric(X.iloc[train_idx][col], errors="coerce")
        fill_value = train_col.mean()
        if not np.isfinite(fill_value):
            fill_value = full_col.mean()
        if not np.isfinite(fill_value):
            fill_value = 0.0
        fill_values[col] = float(fill_value)
        X_processed[col] = pd.to_numeric(X[col], errors="coerce").fillna(fill_value)
        X_full_processed[col] = full_col.fillna(fill_value)

        train_values = X_processed.iloc[train_idx][col].to_numpy(dtype=np.float32)
        mean = float(np.mean(train_values)) if len(train_values) else 0.0
        std = float(np.std(train_values)) if len(train_values) else 1.0
        if not np.isfinite(std) or std < 1e-6:
            std = 1.0
        means.append(mean)
        stds.append(std)

    X_processed = X_processed[feature_columns]
    X_full_processed = X_full_processed[feature_columns]

    y = sampled_df[label_column].copy()
    y_encoder = None
    if task == "regression":
        y_values = pd.to_numeric(y, errors="coerce").fillna(pd.to_numeric(y, errors="coerce").mean()).fillna(0)
        y_array = y_values.to_numpy(dtype=np.float32)
    else:
        y_encoder = LabelEncoder()
        y_array = y_encoder.fit_transform(y.astype("string").fillna("MissingLabel")).astype(np.int64)

    ips_df = pd.DataFrame(ips, index=X.index, columns=feature_columns)
    ips_tensor = torch.tensor(ips_df.to_numpy(dtype=np.float32))
    ips_softmax = torch.softmax(ips_tensor, dim=1).numpy()

    return {
        "feature_columns": feature_columns,
        "label_column": label_column,
        "cat_idxs": cat_idxs,
        "con_idxs": con_idxs,
        "categorical_columns": categorical_columns,
        "continuous_columns": continuous_columns,
        "cat_cardinalities": cat_cardinalities,
        "encoders": encoders,
        "categorical_modes": categorical_modes,
        "fill_values": fill_values,
        "train_mean": np.array(means, dtype=np.float32),
        "train_std": np.array(stds, dtype=np.float32),
        "processed_X": X_processed,
        "full_processed_X": X_full_processed,
        "observed_mask": observed_mask,
        "original_missing": original_missing,
        "ips": pd.DataFrame(ips_softmax, index=X.index, columns=feature_columns),
        "y": y_array,
        "y_encoder": y_encoder,
        "train_indices": train_idx,
        "valid_indices": valid_idx,
        "test_indices": test_idx,
    }


class _FeatureTokenizer(nn.Module):
    def __init__(self, n_num_features, cat_cardinalities, d_token):
        super().__init__()
        self.n_num_features = n_num_features
        self.n_cat_features = len(cat_cardinalities)
        self.num_tokenizers = nn.ModuleList(
            [nn.Linear(1, d_token) for _ in range(n_num_features)]
        )
        self.cat_tokenizers = nn.ModuleList(
            [nn.Embedding(cardinality, d_token) for cardinality in cat_cardinalities]
        )
        n_features = n_num_features + self.n_cat_features
        self.feature_bias = nn.Parameter(torch.empty(n_features, d_token))
        self.missing_embedding = nn.Parameter(torch.empty(n_features, d_token))
        nn.init.uniform_(self.feature_bias, -1 / math.sqrt(d_token), 1 / math.sqrt(d_token))
        nn.init.uniform_(self.missing_embedding, -1 / math.sqrt(d_token), 1 / math.sqrt(d_token))

    def forward(self, x_num, x_cat, num_mask, cat_mask):
        tokens = []
        if self.n_num_features:
            num_tokens = [
                layer(x_num[:, idx:idx + 1])
                for idx, layer in enumerate(self.num_tokenizers)
            ]
            tokens.append(torch.stack(num_tokens, dim=1))
        if self.n_cat_features:
            cat_tokens = [
                layer(x_cat[:, idx])
                for idx, layer in enumerate(self.cat_tokenizers)
            ]
            tokens.append(torch.stack(cat_tokens, dim=1))
        x = torch.cat(tokens, dim=1)
        mask = []
        if self.n_num_features:
            mask.append(num_mask)
        if self.n_cat_features:
            mask.append(cat_mask)
        mask = torch.cat(mask, dim=1).float().unsqueeze(-1)
        x = x + self.feature_bias.unsqueeze(0)
        x = x + (1.0 - mask) * self.missing_embedding.unsqueeze(0)
        return x


class _TransformerBlock(nn.Module):
    def __init__(self, d_token, n_heads, attention_dropout, ffn_dropout, residual_dropout):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_token)
        self.attn = nn.MultiheadAttention(
            d_token,
            n_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(d_token)
        self.ffn = nn.Sequential(
            nn.Linear(d_token, d_token * 4),
            nn.ReLU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(d_token * 4, d_token),
        )
        self.residual_dropout = nn.Dropout(residual_dropout)

    def forward(self, x, ips=None, mask=None):
        x_norm = self.attn_norm(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + self.residual_dropout(attn_out)
        ffn_out = self.ffn(self.ffn_norm(x))
        return x + self.residual_dropout(ffn_out)


class MISSTabularNet(nn.Module):
    def __init__(
            self,
            n_num_features,
            cat_cardinalities,
            d_token,
            n_blocks,
            attention_heads,
            attention_dropout,
            ffn_dropout,
            residual_dropout,
            y_dim):
        super().__init__()
        if d_token % attention_heads != 0:
            raise ValueError("d_token must be divisible by attention_heads.")
        self.n_num_features = n_num_features
        self.n_cat_features = len(cat_cardinalities)
        self.tokenizer = _FeatureTokenizer(n_num_features, cat_cardinalities, d_token)
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_token))
        nn.init.uniform_(self.cls_token, -1 / math.sqrt(d_token), 1 / math.sqrt(d_token))
        self.blocks = nn.ModuleList([
            _TransformerBlock(d_token, attention_heads, attention_dropout, ffn_dropout, residual_dropout)
            for _ in range(n_blocks)
        ])
        self.head_norm = nn.LayerNorm(d_token)
        self.y_head = nn.Linear(d_token, y_dim)
        self.num_heads = nn.ModuleList([nn.Linear(d_token, 1) for _ in range(n_num_features)])
        self.cat_heads = nn.ModuleList([
            nn.Linear(d_token, cardinality) for cardinality in cat_cardinalities
        ])

    def forward(self, x_num, x_cat, num_mask, cat_mask, num_ips=None, cat_ips=None):
        x = self.tokenizer(x_num, x_cat, num_mask, cat_mask)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([x, cls], dim=1)
        if num_ips is not None and cat_ips is not None:
            ips = torch.cat([num_ips, cat_ips], dim=1)
            mask = torch.cat([num_mask, cat_mask], dim=1)
        else:
            ips = None
            mask = None
        for block in self.blocks:
            x = block(x, ips=ips, mask=mask)
        feature_tokens = x[:, :-1, :]
        y_out = self.y_head(self.head_norm(x[:, -1, :]))
        num_tokens = feature_tokens[:, :self.n_num_features, :]
        cat_tokens = feature_tokens[:, self.n_num_features:, :]
        if self.n_num_features:
            num_out = torch.cat(
                [head(num_tokens[:, idx, :]) for idx, head in enumerate(self.num_heads)],
                dim=1,
            )
        else:
            num_out = torch.empty((x.shape[0], 0), device=x.device)
        cat_outs = [
            head(cat_tokens[:, idx, :])
            for idx, head in enumerate(self.cat_heads)
        ]
        return y_out, num_out, cat_outs, feature_tokens


def _build_tensors(prepared, device):
    X = prepared["processed_X"]
    X_full = prepared["full_processed_X"]
    con_cols = prepared["continuous_columns"]
    cat_cols = prepared["categorical_columns"]

    if con_cols:
        mean = prepared["train_mean"]
        std = prepared["train_std"]
        x_num = (X[con_cols].to_numpy(dtype=np.float32) - mean) / std
        full_num = (X_full[con_cols].to_numpy(dtype=np.float32) - mean) / std
    else:
        x_num = np.empty((len(X), 0), dtype=np.float32)
        full_num = np.empty((len(X), 0), dtype=np.float32)

    if cat_cols:
        x_cat = X[cat_cols].to_numpy(dtype=np.int64)
        full_cat = X_full[cat_cols].to_numpy(dtype=np.int64)
    else:
        x_cat = np.empty((len(X), 0), dtype=np.int64)
        full_cat = np.empty((len(X), 0), dtype=np.int64)

    obs = prepared["observed_mask"]
    miss = prepared["original_missing"]
    ips = prepared["ips"]
    return {
        "x_num_current": torch.tensor(x_num, dtype=torch.float32, device=device),
        "x_num_target": torch.tensor(full_num, dtype=torch.float32, device=device),
        "x_cat_current": torch.tensor(x_cat, dtype=torch.long, device=device),
        "x_cat_target": torch.tensor(full_cat, dtype=torch.long, device=device),
        "num_obs_mask": torch.tensor(obs[con_cols].to_numpy(dtype=np.float32), device=device),
        "cat_obs_mask": torch.tensor(obs[cat_cols].to_numpy(dtype=np.float32), device=device),
        "num_missing": torch.tensor(miss[con_cols].to_numpy(dtype=bool), device=device),
        "cat_missing": torch.tensor(miss[cat_cols].to_numpy(dtype=bool), device=device),
        "num_ips": torch.tensor(ips[con_cols].to_numpy(dtype=np.float32), device=device),
        "cat_ips": torch.tensor(ips[cat_cols].to_numpy(dtype=np.float32), device=device),
        "y": torch.tensor(prepared["y"], device=device),
    }


def _loss_fn(task):
    if task == "regression":
        return nn.MSELoss()
    return nn.CrossEntropyLoss()


def _supervised_loss(y_out, y, task, criterion):
    if task == "regression":
        return criterion(y_out.squeeze(-1), y.float())
    return criterion(y_out, y.long())


def _make_reconstruction_masks(num_obs, cat_obs, mask_rate):
    if num_obs.numel():
        num_mask = (num_obs > 0) & (torch.rand_like(num_obs.float()) < mask_rate)
        if not num_mask.any() and (num_obs > 0).any():
            num_mask = num_obs > 0
    else:
        num_mask = torch.zeros_like(num_obs, dtype=torch.bool)

    if cat_obs.numel():
        cat_mask = (cat_obs > 0) & (torch.rand_like(cat_obs.float()) < mask_rate)
        if not cat_mask.any() and (cat_obs > 0).any():
            cat_mask = cat_obs > 0
    else:
        cat_mask = torch.zeros_like(cat_obs, dtype=torch.bool)
    return num_mask, cat_mask


def _reconstruction_losses(num_out, cat_outs, x_num_target, x_cat_target,
                           num_recon_mask, cat_recon_mask):
    num_loss = torch.tensor(0.0, device=x_num_target.device)
    if num_out.shape[1] and num_recon_mask.any():
        num_loss = F.mse_loss(num_out[num_recon_mask], x_num_target[num_recon_mask])
    cat_loss = torch.tensor(0.0, device=x_num_target.device)
    if cat_outs:
        count = 0
        for idx, logits in enumerate(cat_outs):
            mask = cat_recon_mask[:, idx]
            if mask.any():
                cat_loss = cat_loss + F.cross_entropy(logits[mask], x_cat_target[mask, idx])
                count += 1
        if count:
            cat_loss = cat_loss / count
    return num_loss, cat_loss


def _update_missing_values(tensors, row_idx, num_out, cat_outs):
    if tensors["x_num_current"].shape[1]:
        num_missing = tensors["num_missing"][row_idx]
        for col in range(num_missing.shape[1]):
            rows = row_idx[num_missing[:, col]]
            if len(rows):
                tensors["x_num_current"][rows, col] = num_out.detach()[num_missing[:, col], col]
    if tensors["x_cat_current"].shape[1]:
        cat_missing = tensors["cat_missing"][row_idx]
        for col, logits in enumerate(cat_outs):
            rows = row_idx[cat_missing[:, col]]
            if len(rows):
                tensors["x_cat_current"][rows, col] = torch.argmax(logits.detach()[cat_missing[:, col]], dim=1)


@torch.no_grad()
def _refresh_part(model, tensors, indices, batch_size):
    model.eval()
    loader = DataLoader(TensorDataset(torch.tensor(indices, device=tensors["y"].device)), batch_size=batch_size)
    for (row_idx,) in loader:
        y_out, num_out, cat_outs, _ = model(
            tensors["x_num_current"][row_idx],
            tensors["x_cat_current"][row_idx],
            tensors["num_obs_mask"][row_idx],
            tensors["cat_obs_mask"][row_idx],
            tensors["num_ips"][row_idx],
            tensors["cat_ips"][row_idx],
        )
        _update_missing_values(tensors, row_idx, num_out, cat_outs)


@torch.no_grad()
def _evaluate_supervised(model, tensors, indices, task, batch_size):
    model.eval()
    y_true = []
    y_pred = []
    y_prob = []
    loader = DataLoader(TensorDataset(torch.tensor(indices, device=tensors["y"].device)), batch_size=batch_size)
    for (row_idx,) in loader:
        out, _, _, _ = model(
            tensors["x_num_current"][row_idx],
            tensors["x_cat_current"][row_idx],
            tensors["num_obs_mask"][row_idx],
            tensors["cat_obs_mask"][row_idx],
            tensors["num_ips"][row_idx],
            tensors["cat_ips"][row_idx],
        )
        y_true.append(tensors["y"][row_idx].detach().cpu())
        if task == "regression":
            y_pred.append(out.squeeze(-1).detach().cpu())
        else:
            y_prob.append(torch.softmax(out, dim=1).detach().cpu())
            y_pred.append(torch.argmax(out, dim=1).detach().cpu())
    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    if task == "regression":
        rmse = float(mean_squared_error(y_true, y_pred, squared=False))
        return {"score": -rmse, "rmse": rmse, "r2": float(r2_score(y_true, y_pred))}
    acc = float(accuracy_score(y_true, y_pred))
    metrics = {"score": acc, "accuracy": acc}
    try:
        probs = torch.cat(y_prob).numpy()
        if probs.shape[1] == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_true, probs[:, 1]))
        else:
            metrics["roc_auc"] = float(roc_auc_score(y_true, probs, multi_class="ovo"))
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics


def _decode_imputed_frame(prepared, tensors):
    result = prepared["processed_X"].copy()
    con_cols = prepared["continuous_columns"]
    cat_cols = prepared["categorical_columns"]

    if con_cols:
        values = tensors["x_num_current"].detach().cpu().numpy()
        values = values * prepared["train_std"] + prepared["train_mean"]
        for idx, col in enumerate(con_cols):
            result[col] = values[:, idx]

    for idx, col in enumerate(cat_cols):
        encoded = tensors["x_cat_current"][:, idx].detach().cpu().numpy().astype(int)
        decoded = prepared["encoders"][col].inverse_transform(encoded)
        decoded_series = pd.Series(decoded, index=result.index)
        mode = prepared["categorical_modes"].get(col)
        if mode is not None:
            decoded_series = decoded_series.where(decoded_series != "MissingValue", mode)
        result[col] = saint_modules._safe_numeric_cast(decoded_series, result[col])

    return result[prepared["feature_columns"]]


def train_miss_tabular_algorithm(data, missing_mask=None, params=None, device=None, full_data=None,
                                 labels=None, label_column=None):
    params = params or {}
    if full_data is None:
        raise ValueError("Strict MISS-tabular requires full_data, matching the original train5.py datafull input.")

    datamiss_df, datafull_df, label_column = _prepare_dataframes(
        data,
        full_data,
        label_column=label_column,
        labels=labels,
    )
    task = params.get("task") or _infer_task(datamiss_df[label_column])
    seed = int(params.get("random_state", params.get("seed", 0)))
    _set_seed(seed)

    ips_num = int(params.get("ips_num", 40))
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

    prepared = _fit_feature_state(
        sampled_df,
        datafull_df,
        indicator,
        ips,
        label_column,
        task,
        params,
    )

    device = torch.device(device or params.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    tensors = _build_tensors(prepared, device)
    y_dim = 1 if task == "regression" else int(np.max(prepared["y"]) + 1)
    model = MISSTabularNet(
        n_num_features=len(prepared["continuous_columns"]),
        cat_cardinalities=prepared["cat_cardinalities"],
        d_token=int(params.get("d_token", 32)),
        n_blocks=int(params.get("n_blocks", 3)),
        attention_heads=int(params.get("attention_heads", 8)),
        attention_dropout=float(params.get("attention_dropout", 0.2)),
        ffn_dropout=float(params.get("ffn_dropout", 0.1)),
        residual_dropout=float(params.get("residual_dropout", 0.0)),
        y_dim=y_dim,
    ).to(device)

    if params.get("optimizer", "AdamW") == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=float(params.get("lr", 0.0001)))
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(params.get("lr", 0.0001)),
            weight_decay=float(params.get("weight_decay", 1e-5)),
        )

    batch_size = int(params.get("batch_size", 256))
    epochs = int(params.get("epoch", params.get("epochs", 200)))
    criterion = _loss_fn(task).to(device)
    train_idx = prepared["train_indices"]
    valid_idx = prepared["valid_indices"]
    test_idx = prepared["test_indices"]
    train_loader = DataLoader(
        TensorDataset(torch.tensor(train_idx, device=device)),
        batch_size=batch_size,
        shuffle=True,
    )

    mask_rate = float(params.get("mask_rate", 0.1))
    weights_cat = float(params.get("weights_cat", 1.0))
    weights_num = float(params.get("weights_num", 1.0))
    verbose = bool(params.get("verbose", False))
    patience = params.get("patience")
    patience = None if patience is None else int(patience)
    best_score = -float("inf")
    best_state = None
    bad_epochs = 0
    train_loss = []
    val_scores = []
    test_scores = []

    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        for (row_idx,) in train_loader:
            optimizer.zero_grad()
            y_out, num_out, cat_outs, _ = model(
                tensors["x_num_current"][row_idx],
                tensors["x_cat_current"][row_idx],
                tensors["num_obs_mask"][row_idx],
                tensors["cat_obs_mask"][row_idx],
                tensors["num_ips"][row_idx],
                tensors["cat_ips"][row_idx],
            )
            loss = _supervised_loss(y_out, tensors["y"][row_idx], task, criterion)
            num_mask, cat_mask = _make_reconstruction_masks(
                tensors["num_obs_mask"][row_idx],
                tensors["cat_obs_mask"][row_idx],
                mask_rate,
            )
            num_loss, cat_loss = _reconstruction_losses(
                num_out,
                cat_outs,
                tensors["x_num_target"][row_idx],
                tensors["x_cat_target"][row_idx],
                num_mask,
                cat_mask,
            )
            loss = loss + weights_num * num_loss + weights_cat * cat_loss
            loss.backward()
            optimizer.step()
            _update_missing_values(tensors, row_idx, num_out, cat_outs)
            epoch_losses.append(float(loss.detach().cpu()))

        _refresh_part(model, tensors, valid_idx, batch_size)
        _refresh_part(model, tensors, test_idx, batch_size)
        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        val_metrics = _evaluate_supervised(model, tensors, valid_idx, task, batch_size)
        test_metrics = _evaluate_supervised(model, tensors, test_idx, task, batch_size)
        train_loss.append(mean_loss)
        val_scores.append(val_metrics)
        test_scores.append(test_metrics)

        if verbose:
            print(
                f"[MISS-tabular] epoch {epoch + 1}/{epochs}, "
                f"train_loss={mean_loss:.6f}, val_score={val_metrics['score']:.6f}"
            )

        if val_metrics["score"] > best_score:
            best_score = val_metrics["score"]
            best_state = {
                "model": copy.deepcopy(model.state_dict()),
                "x_num_current": tensors["x_num_current"].detach().clone(),
                "x_cat_current": tensors["x_cat_current"].detach().clone(),
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
            if patience is not None and bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state["model"])
        tensors["x_num_current"] = best_state["x_num_current"]
        tensors["x_cat_current"] = best_state["x_cat_current"]

    imputed_features = _decode_imputed_frame(prepared, tensors)
    imputed_data = sampled_df.copy()
    for col in prepared["feature_columns"]:
        miss = prepared["original_missing"][col].to_numpy()
        imputed_data.loc[miss, col] = imputed_features.loc[miss, col]

    metrics = {
        "train_loss": train_loss,
        "val_scores": val_scores,
        "test_scores": test_scores,
        "best_score": float(best_score),
        "total_parameters": _count_parameters(model),
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


def _transform_predict_features(df, train_state, missing_mask=None):
    prepared = train_state["prepared"]
    label_column = train_state["label_column"]
    X = df.drop(columns=[label_column]).copy()
    original_missing = _missing_frame_from_mask(df, label_column, missing_mask)
    X = X.mask(original_missing)

    processed = pd.DataFrame(index=X.index)
    for col in prepared["categorical_columns"]:
        encoder = prepared["encoders"][col]
        values = X[col].astype("string").fillna("MissingValue")
        known = set(encoder.classes_)
        values = values.where(values.isin(known), "MissingValue")
        processed[col] = encoder.transform(values)
    for col in prepared["continuous_columns"]:
        fill_value = prepared["fill_values"][col]
        processed[col] = pd.to_numeric(X[col], errors="coerce").fillna(fill_value)
    processed = processed[prepared["feature_columns"]]
    return processed, original_missing


def predict_miss_tabular_algorithm(model, data, missing_mask=None, train_state=None, device=None):
    if train_state is None:
        raise ValueError("train_state is required for MISS-tabular prediction.")

    df = _prepare_predict_frame(data, train_state)
    processed, original_missing = _transform_predict_features(df, train_state, missing_mask)
    prepared = train_state["prepared"]
    device = torch.device(device or train_state.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    model.eval()

    con_cols = prepared["continuous_columns"]
    cat_cols = prepared["categorical_columns"]
    if con_cols:
        x_num = (processed[con_cols].to_numpy(dtype=np.float32) - prepared["train_mean"]) / prepared["train_std"]
    else:
        x_num = np.empty((len(processed), 0), dtype=np.float32)
    if cat_cols:
        x_cat = processed[cat_cols].to_numpy(dtype=np.int64)
    else:
        x_cat = np.empty((len(processed), 0), dtype=np.int64)

    num_mask = (~original_missing[con_cols]).to_numpy(dtype=np.float32) if con_cols else np.empty((len(processed), 0), dtype=np.float32)
    cat_mask = (~original_missing[cat_cols]).to_numpy(dtype=np.float32) if cat_cols else np.empty((len(processed), 0), dtype=np.float32)
    num_ips = np.ones_like(num_mask, dtype=np.float32)
    cat_ips = np.ones_like(cat_mask, dtype=np.float32)

    x_num_t = torch.tensor(x_num, dtype=torch.float32, device=device)
    x_cat_t = torch.tensor(x_cat, dtype=torch.long, device=device)
    num_mask_t = torch.tensor(num_mask, dtype=torch.float32, device=device)
    cat_mask_t = torch.tensor(cat_mask, dtype=torch.float32, device=device)
    num_ips_t = torch.tensor(num_ips, dtype=torch.float32, device=device)
    cat_ips_t = torch.tensor(cat_ips, dtype=torch.float32, device=device)

    result_features = df.drop(columns=[train_state["label_column"]]).copy()
    batch_size = int(train_state["params"].get("batch_size", 256))
    pred_num_chunks = []
    pred_cat_chunks = [[] for _ in cat_cols]
    with torch.no_grad():
        for start in range(0, len(df), batch_size):
            end = start + batch_size
            _, num_out, cat_outs, _ = model(
                x_num_t[start:end],
                x_cat_t[start:end],
                num_mask_t[start:end],
                cat_mask_t[start:end],
                num_ips_t[start:end],
                cat_ips_t[start:end],
            )
            if con_cols:
                pred_num_chunks.append(num_out.detach().cpu().numpy())
            for idx, logits in enumerate(cat_outs):
                pred_cat_chunks[idx].append(torch.argmax(logits, dim=1).detach().cpu().numpy())

    if con_cols and pred_num_chunks:
        pred_num = np.vstack(pred_num_chunks)
        pred_num = pred_num * prepared["train_std"] + prepared["train_mean"]
        for idx, col in enumerate(con_cols):
            miss = original_missing[col].to_numpy()
            result_features.loc[miss, col] = pred_num[miss, idx]

    for idx, col in enumerate(cat_cols):
        if not pred_cat_chunks[idx]:
            continue
        encoded = np.concatenate(pred_cat_chunks[idx]).astype(int)
        decoded = prepared["encoders"][col].inverse_transform(encoded)
        decoded_series = pd.Series(decoded, index=result_features.index)
        mode = prepared["categorical_modes"].get(col)
        if mode is not None:
            decoded_series = decoded_series.where(decoded_series != "MissingValue", mode)
        decoded_series = saint_modules._safe_numeric_cast(decoded_series, result_features[col])
        miss = original_missing[col].to_numpy()
        result_features.loc[miss, col] = decoded_series.loc[miss]

    result = df.copy()
    for col in result_features.columns:
        result[col] = result_features[col]
    return result

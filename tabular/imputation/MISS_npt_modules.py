import copy
import math
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, roc_auc_score
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset

try:
    import dataprep.tabular.imputation.MISS_saint_modules as saint_modules
    import dataprep.tabular.imputation.MISS_tabformer_modules as tabformer_modules
except ImportError:
    import tabular.imputation.MISS_saint_modules as saint_modules
    import tabular.imputation.MISS_tabformer_modules as tabformer_modules


def _set_seed(seed):
    saint_modules._set_seed(seed)


class Lamb(Optimizer):
    """LAMB optimizer used by the original NPT default configuration."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad
                if grad.is_sparse:
                    raise RuntimeError("LAMB does not support sparse gradients.")
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                state["step"] += 1
                beta1, beta2 = group["betas"]
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                update = exp_avg / exp_avg_sq.sqrt().add(group["eps"])
                if group["weight_decay"]:
                    update.add_(parameter, alpha=group["weight_decay"])
                weight_norm = parameter.norm().clamp(0, 10)
                update_norm = update.norm()
                trust_ratio = 1.0 if weight_norm == 0 or update_norm == 0 else weight_norm / update_norm
                parameter.add_(update, alpha=-group["lr"] * trust_ratio)
        return loss


class Lookahead(Optimizer):
    def __init__(self, base_optimizer, alpha=0.5, k=6):
        self.base_optimizer = base_optimizer
        self.param_groups = base_optimizer.param_groups
        self.defaults = base_optimizer.defaults
        self.state = defaultdict(dict)
        self.alpha = alpha
        self.k = k
        self.steps = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = self.base_optimizer.step(closure)
        self.steps += 1
        if self.steps % self.k == 0:
            for group in self.param_groups:
                for parameter in group["params"]:
                    if parameter.grad is None:
                        continue
                    state = self.state[parameter]
                    if "slow" not in state:
                        state["slow"] = parameter.detach().clone()
                    state["slow"].add_(parameter - state["slow"], alpha=self.alpha)
                    parameter.copy_(state["slow"])
        return loss

    def zero_grad(self, set_to_none=False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)


class WeightedAttention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("Attention dimension must be divisible by num_heads.")
        self.heads = heads
        self.head_dim = dim // heads
        # Original NPT divides attention scores by sqrt(dim_KV), not by the
        # per-head dimension used in PyTorch's standard attention.
        self.scale = dim ** -0.5
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_weights=None, observed_mask=None, kv=None):
        batch, tokens, dim = x.shape
        kv = x if kv is None else kv
        q = self.to_q(x)
        k = self.to_k(kv)
        v = self.to_v(kv)
        q = q.reshape(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if key_weights is not None:
            scores = scores * key_weights.to(scores.device).unsqueeze(1).unsqueeze(2)
        if observed_mask is not None:
            missing = ~observed_mask.bool()
            all_missing = missing.all(dim=1)
            if all_missing.any():
                missing = missing.clone()
                missing[all_missing] = False
            scores = scores.masked_fill(missing.unsqueeze(1).unsqueeze(2), float("-inf"))

        attention = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.matmul(attention, v)
        out = out.transpose(1, 2).reshape(batch, tokens, dim)
        return self.to_out(out)


class AttentionBlock(nn.Module):
    def __init__(self, dim, heads, hidden_dropout, attention_dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = WeightedAttention(dim, heads, attention_dropout)
        self.residual = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(hidden_dropout),
        )

    def forward(self, x, key_weights=None, observed_mask=None):
        x = self.residual(x) + self.attention(
            self.norm1(x),
            key_weights,
            observed_mask,
            kv=x,
        )
        return x + self.ffn(self.norm2(x))


class NPTNet(nn.Module):
    """In-memory NPT preserving alternating row/column attention."""

    def __init__(
            self,
            n_num_features,
            cat_cardinalities,
            dim_hidden,
            stacking_depth,
            num_heads,
            hidden_dropout,
            attention_dropout,
            y_dim):
        super().__init__()
        if stacking_depth < 2 or stacking_depth % 2:
            raise ValueError("stacking_depth must be an even integer >= 2.")
        self.n_num_features = n_num_features
        self.n_cat_features = len(cat_cardinalities)
        self.n_features = n_num_features + self.n_cat_features
        self.dim_hidden = dim_hidden

        self.num_embeddings = nn.ModuleList([
            nn.Linear(1, dim_hidden) for _ in range(n_num_features)
        ])
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(cardinality, dim_hidden) for cardinality in cat_cardinalities
        ])
        self.feature_index_embedding = nn.Embedding(self.n_features + 1, dim_hidden)
        self.feature_type_embedding = nn.Embedding(3, dim_hidden)
        self.missing_embedding = nn.Parameter(torch.empty(self.n_features, dim_hidden))
        self.label_mask_token = nn.Parameter(torch.empty(1, 1, dim_hidden))
        nn.init.normal_(self.missing_embedding, std=0.02)
        nn.init.normal_(self.label_mask_token, std=0.02)
        self.embedding_dropout = nn.Dropout(hidden_dropout)

        row_dim = (self.n_features + 1) * dim_hidden
        if row_dim % num_heads != 0:
            raise ValueError("(number of features + label token) * dim_hidden must divide num_heads.")
        self.row_blocks = nn.ModuleList()
        self.col_blocks = nn.ModuleList()
        for layer in range(stacking_depth):
            if layer % 2 == 0:
                self.row_blocks.append(AttentionBlock(row_dim, num_heads, hidden_dropout, attention_dropout))
            else:
                self.col_blocks.append(AttentionBlock(dim_hidden, num_heads, hidden_dropout, attention_dropout))

        self.num_heads = nn.ModuleList([nn.Linear(dim_hidden, 1) for _ in range(n_num_features)])
        self.cat_heads = nn.ModuleList([
            nn.Linear(dim_hidden, cardinality) for cardinality in cat_cardinalities
        ])
        self.label_head = nn.Linear(dim_hidden, y_dim)
        projection_dim = max(1, row_dim // 2)
        self.projection_head = nn.Sequential(
            nn.Linear(row_dim, max(projection_dim, dim_hidden)),
            nn.ReLU(),
            nn.Linear(max(projection_dim, dim_hidden), projection_dim),
        )
        self.stacking_depth = stacking_depth

    def tokenize(self, x_num, x_cat, num_mask, cat_mask):
        tokens = []
        if self.n_num_features:
            tokens.append(torch.stack([
                layer(x_num[:, idx:idx + 1])
                for idx, layer in enumerate(self.num_embeddings)
            ], dim=1))
        if self.n_cat_features:
            tokens.append(torch.stack([
                layer(x_cat[:, idx])
                for idx, layer in enumerate(self.cat_embeddings)
            ], dim=1))
        x = torch.cat(tokens, dim=1)

        feature_indices = torch.arange(self.n_features, device=x.device)
        x = x + self.feature_index_embedding(feature_indices).unsqueeze(0)
        type_ids = torch.tensor(
            [0] * self.n_num_features + [1] * self.n_cat_features,
            device=x.device,
        )
        x = x + self.feature_type_embedding(type_ids).unsqueeze(0)
        observed = torch.cat([num_mask, cat_mask], dim=1).float()
        x = x + (1.0 - observed.unsqueeze(-1)) * self.missing_embedding.unsqueeze(0)

        label = self.label_mask_token.expand(x.shape[0], -1, -1)
        label_index = torch.tensor([self.n_features], device=x.device)
        label = label + self.feature_index_embedding(label_index).unsqueeze(0)
        label = label + self.feature_type_embedding(torch.tensor([2], device=x.device)).unsqueeze(0)
        return self.embedding_dropout(torch.cat([x, label], dim=1))

    def encode(self, x_num, x_cat, num_mask, cat_mask, row_ips, num_ips, cat_ips):
        x = self.tokenize(x_num, x_cat, num_mask, cat_mask)
        col_weights = torch.cat([
            num_ips,
            cat_ips,
            torch.ones((x.shape[0], 1), device=x.device),
        ], dim=1)
        col_observed = torch.cat([
            num_mask,
            cat_mask,
            torch.ones((x.shape[0], 1), device=x.device),
        ], dim=1).bool()

        row_i = 0
        col_i = 0
        for layer in range(self.stacking_depth):
            if layer % 2 == 0:
                flat = x.reshape(1, x.shape[0], -1)
                # MISS-NPT applies IPS only in MAB_first, the first row block.
                first_row_block = row_i == 0
                flat = self.row_blocks[row_i](
                    flat,
                    key_weights=row_ips.reshape(1, -1) if first_row_block else None,
                    observed_mask=None,
                )
                x = flat.reshape(x.shape[0], self.n_features + 1, self.dim_hidden)
                row_i += 1
            else:
                x = self.col_blocks[col_i](x)
                col_i += 1
        return x

    def forward(self, x_num, x_cat, num_mask, cat_mask, row_ips, num_ips, cat_ips):
        reps = self.encode(x_num, x_cat, num_mask, cat_mask, row_ips, num_ips, cat_ips)
        feature_reps = reps[:, :self.n_features]
        num_reps = feature_reps[:, :self.n_num_features]
        cat_reps = feature_reps[:, self.n_num_features:]
        if self.n_num_features:
            num_out = torch.cat([
                head(num_reps[:, idx]) for idx, head in enumerate(self.num_heads)
            ], dim=1)
        else:
            num_out = torch.empty((reps.shape[0], 0), device=reps.device)
        cat_out = [
            head(cat_reps[:, idx]) for idx, head in enumerate(self.cat_heads)
        ]
        label_out = self.label_head(reps[:, -1])
        return label_out, num_out, cat_out, reps


def _fit_state(sampled_df, datafull_df, indicator, ips, label_column, task, params):
    prepared = tabformer_modules._fit_preprocessing(
        sampled_df, datafull_df, indicator, ips, label_column, task, params
    )
    # NPT original split is 0.8 / 0.1 / 0.1, unlike TabFormer train/test.
    rng = np.random.default_rng(int(params.get("dset_seed", params.get("random_state", 42))))
    split = rng.choice(["train", "valid", "test"], p=params.get("datasplit", [.8, .1, .1]), size=len(sampled_df))
    train_idx = np.where(split == "train")[0]
    valid_idx = np.where(split == "valid")[0]
    test_idx = np.where(split == "test")[0]
    if not len(train_idx):
        train_idx = np.arange(len(sampled_df))
    if not len(valid_idx):
        valid_idx = train_idx[:1]
    if not len(test_idx):
        test_idx = valid_idx
    prepared["train_indices"] = train_idx
    prepared["valid_indices"] = valid_idx
    prepared["test_indices"] = test_idx
    prepared["npt_filled_X"] = tabformer_modules._decode_filled_frame(prepared)

    # Original NPT mean-fills first and then standardizes every continuous
    # feature. HI's modified loader does not expose a missing-value mask to
    # the encoder after this step.
    means = {}
    scales = {}
    for col in prepared["continuous_columns"]:
        values = prepared["processed_X"][col].to_numpy(dtype=np.float64)
        mean = float(np.mean(values))
        scale = float(np.std(values))
        if not np.isfinite(scale) or scale == 0:
            scale = 1.0
        means[col] = mean
        scales[col] = scale
        prepared["processed_X"][col] = (prepared["processed_X"][col] - mean) / scale
        prepared["full_processed_X"][col] = (prepared["full_processed_X"][col] - mean) / scale
    prepared["standardization_means"] = means
    prepared["standardization_scales"] = scales
    prepared["observed_mask"].loc[:, :] = True

    raw_feature_ips = np.asarray(ips, dtype=np.float32)
    prepared["row_ips"] = torch.softmax(torch.tensor(raw_feature_ips.sum(axis=1)), dim=0).numpy()
    return prepared


def _build_tensors(prepared, device):
    X = prepared["processed_X"]
    X_full = prepared["full_processed_X"]
    cat_cols = prepared["categorical_columns"]
    con_cols = prepared["continuous_columns"]
    x_num = X[con_cols].to_numpy(dtype=np.float32) if con_cols else np.empty((len(X), 0), dtype=np.float32)
    target_num = X_full[con_cols].to_numpy(dtype=np.float32) if con_cols else np.empty((len(X), 0), dtype=np.float32)
    x_cat = X[cat_cols].to_numpy(dtype=np.float32).astype(np.int64) if cat_cols else np.empty((len(X), 0), dtype=np.int64)
    target_cat = X_full[cat_cols].to_numpy(dtype=np.float32).astype(np.int64) if cat_cols else np.empty((len(X), 0), dtype=np.int64)
    num_mask = prepared["observed_mask"][con_cols].to_numpy(dtype=np.float32) if con_cols else np.empty((len(X), 0), dtype=np.float32)
    cat_mask = prepared["observed_mask"][cat_cols].to_numpy(dtype=np.float32) if cat_cols else np.empty((len(X), 0), dtype=np.float32)
    num_ips = prepared["ips"][con_cols].to_numpy(dtype=np.float32) if con_cols else np.empty((len(X), 0), dtype=np.float32)
    cat_ips = prepared["ips"][cat_cols].to_numpy(dtype=np.float32) if cat_cols else np.empty((len(X), 0), dtype=np.float32)
    row_ips = prepared["row_ips"]
    return {
        "x_num": torch.tensor(x_num, dtype=torch.float32, device=device),
        "target_num": torch.tensor(target_num, dtype=torch.float32, device=device),
        "x_cat": torch.tensor(x_cat, dtype=torch.long, device=device),
        "target_cat": torch.tensor(target_cat, dtype=torch.long, device=device),
        "num_mask": torch.tensor(num_mask, dtype=torch.float32, device=device),
        "cat_mask": torch.tensor(cat_mask, dtype=torch.float32, device=device),
        "num_ips": torch.tensor(num_ips, dtype=torch.float32, device=device),
        "cat_ips": torch.tensor(cat_ips, dtype=torch.float32, device=device),
        "row_ips": torch.tensor(row_ips, dtype=torch.float32, device=device),
        "y": torch.tensor(prepared["y"], device=device),
    }


def _batch_forward(model, tensors, idx):
    return model(
        tensors["x_num"][idx],
        tensors["x_cat"][idx],
        tensors["num_mask"][idx],
        tensors["cat_mask"][idx],
        tensors["row_ips"][idx],
        tensors["num_ips"][idx],
        tensors["cat_ips"][idx],
    )


def _label_loss(out, y, task):
    if task == "regression":
        return F.mse_loss(out.squeeze(-1), y.float())
    if task == "binary":
        return F.cross_entropy(out, y.long())
    return F.cross_entropy(out, y.long())


def _reconstruction_loss(num_out, cat_out, tensors, idx):
    loss = torch.tensor(0.0, device=tensors["y"].device)
    if num_out.shape[1]:
        observed = tensors["num_mask"][idx].bool()
        if observed.any():
            loss = loss + F.mse_loss(num_out[observed], tensors["target_num"][idx][observed])
    if cat_out:
        cat_loss = torch.tensor(0.0, device=loss.device)
        count = 0
        for col, logits in enumerate(cat_out):
            observed = tensors["cat_mask"][idx, col].bool()
            if observed.any():
                cat_loss = cat_loss + F.cross_entropy(logits[observed], tensors["target_cat"][idx, col][observed])
                count += 1
        if count:
            loss = loss + cat_loss / count
    return loss


def _pretrain(model, tensors, train_idx, params, device):
    # The standalone pretraining loop exists in train.py, but the original
    # MISS-NPT train_and_eval entry point comments it out.
    if not params.get("enable_pretrain", False):
        return []
    epochs = int(params.get("pretrain_epoch", 20))
    batch_size = int(params.get("batch_size", 128))
    loader = DataLoader(TensorDataset(torch.tensor(train_idx, device=device)), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(params.get("pretrain_lr", params.get("lr", 0.001))))
    history = []
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for (idx,) in loader:
            optimizer.zero_grad()
            _, num_1, cat_1, reps_1 = _batch_forward(model, tensors, idx)
            _, _, _, reps_2 = _batch_forward(model, tensors, idx)
            proj_1 = F.normalize(model.projection_head(reps_1.flatten(1)), dim=-1)
            proj_2 = F.normalize(model.projection_head(reps_2.flatten(1)), dim=-1)
            logits = proj_1 @ proj_2.t() / 0.7
            targets = torch.arange(len(idx), device=device)
            contrastive = (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets)) / 2
            reconstruction = _reconstruction_loss(num_1, cat_1, tensors, idx)
            loss = (
                float(params.get("contrastive_weight", 1.0)) * contrastive
                + float(params.get("reconstruction_weight", 1.0)) * reconstruction
            )
            loss.backward()
            optimizer.step()
            running += float(loss.detach().cpu())
        history.append(running)
        if params.get("verbose", False):
            print(f"[MISS-NPT pretrain] epoch {epoch + 1}/{epochs}, loss={running:.6f}")
    return history


def _mixed_batches(prepared, batch_size, shuffle, seed):
    """Build NPT semi-supervised batches containing all dataset modes."""
    rng = np.random.default_rng(seed)
    splits = [
        np.asarray(prepared["train_indices"]),
        np.asarray(prepared["valid_indices"]),
        np.asarray(prepared["test_indices"]),
    ]
    if shuffle:
        splits = [rng.permutation(indices) for indices in splits]
    n_batches = max(1, int(np.ceil(sum(map(len, splits)) / batch_size)))
    chunks = [np.array_split(indices, n_batches) for indices in splits]
    for batch_number in range(n_batches):
        indices = np.concatenate([parts[batch_number] for parts in chunks])
        if shuffle:
            rng.shuffle(indices)
        yield indices


@torch.no_grad()
def _evaluate(model, tensors, prepared, mode, task, batch_size):
    model.eval()
    true, pred, prob, losses = [], [], [], []
    target_indices = set(prepared[f"{mode}_indices"].tolist())
    for indices in _mixed_batches(prepared, batch_size, shuffle=False, seed=0):
        local_target = np.array([index in target_indices for index in indices])
        if not local_target.any():
            continue
        idx = torch.tensor(indices, device=tensors["y"].device)
        target = torch.tensor(local_target, dtype=torch.bool, device=tensors["y"].device)
        out, _, _, _ = _batch_forward(model, tensors, idx)
        out = out[target]
        y = tensors["y"][idx][target]
        losses.append((_label_loss(out, y, task).item(), int(target.sum())))
        true.append(y.cpu())
        if task == "regression":
            pred.append(out.squeeze(-1).cpu())
        elif task == "binary":
            p = torch.softmax(out, dim=1)[:, 1]
            prob.append(p.cpu())
            pred.append((p > 0.5).long().cpu())
        else:
            p = torch.softmax(out, dim=1)
            prob.append(p.cpu())
            pred.append(torch.argmax(p, dim=1).cpu())
    y_true = torch.cat(true).numpy()
    y_pred = torch.cat(pred).numpy()
    label_loss = float(sum(loss * count for loss, count in losses) / sum(count for _, count in losses))
    if task == "regression":
        rmse = float(mean_squared_error(y_true, y_pred, squared=False))
        return {"score": -rmse, "label_loss": label_loss, "rmse": rmse, "r2": float(r2_score(y_true, y_pred))}
    acc = float(accuracy_score(y_true, y_pred))
    metrics = {"score": acc, "label_loss": label_loss, "accuracy": acc}
    try:
        p = torch.cat(prob).numpy()
        metrics["roc_auc"] = float(
            roc_auc_score(y_true, p) if task == "binary"
            else roc_auc_score(y_true, p, multi_class="ovo")
        )
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics


def train_miss_npt_algorithm(data, missing_mask=None, params=None, device=None, full_data=None,
                             labels=None, label_column=None):
    params = params or {}
    if full_data is None:
        raise ValueError("Strict MISS-NPT requires full_data for expert sampling and IPS.")
    datamiss_df, datafull_df, label_column = saint_modules._prepare_dataframes(
        data, full_data, label_column=label_column, labels=labels
    )
    task = params.get("task") or saint_modules._infer_task(datamiss_df[label_column])
    if task == "classification":
        task = "multiclass"
    seed = int(params.get("random_state", 42))
    _set_seed(seed)

    ips_num = int(params.get("ips_num", 40))
    sampled_array, indicator = saint_modules.sampling(
        datafull_df, datamiss_df, ips_num, method=params.get("sampling_method", "feature")
    )
    sampled_df = pd.DataFrame(sampled_array, columns=datamiss_df.columns, index=datamiss_df.index)
    sampled_df = saint_modules._restore_sampled_dtypes(sampled_df, datamiss_df)
    numeric_ips = tabformer_modules._safe_numeric_ips_frame(sampled_df.drop(columns=[label_column]))
    ips = saint_modules.compute_ips(
        numeric_ips.to_numpy(),
        indicator[:, :numeric_ips.shape[1]],
        num=ips_num,
        method=params.get("ips_method", "xgb"),
        observed_num=int(params.get("observed_num", 1)),
        complete_sample=params.get("complete_sample", "no-Random"),
    )

    prepared = _fit_state(sampled_df, datafull_df, indicator, ips, label_column, task, params)
    device = torch.device(device or params.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    tensors = _build_tensors(prepared, device)
    y_dim = 1 if task == "regression" else int(np.max(prepared["y"]) + 1)
    model = NPTNet(
        n_num_features=len(prepared["continuous_columns"]),
        cat_cardinalities=prepared["cat_cardinalities"],
        dim_hidden=int(params.get("dim_hidden", 64)),
        stacking_depth=int(params.get("stacking_depth", 4)),
        num_heads=int(params.get("num_heads", 8)),
        hidden_dropout=float(params.get("hidden_dropout", 0.1)),
        attention_dropout=float(params.get("attention_dropout", 0.1)),
        y_dim=y_dim,
    ).to(device)

    pretrain_loss = _pretrain(model, tensors, prepared["train_indices"], params, device)
    base_optimizer = Lamb(
        model.parameters(),
        lr=float(params.get("lr", 0.001)),
        weight_decay=float(params.get("weight_decay", 0.0)),
    )
    optimizer = Lookahead(
        base_optimizer,
        alpha=float(params.get("lookahead_alpha", 0.5)),
        k=int(params.get("lookahead_update_cadence", 6)),
    )
    batch_size = int(params.get("batch_size", 128))
    n_batches = max(1, int(np.ceil(len(sampled_df) / batch_size)))
    total_steps = int(params.get("total_steps", 100000))
    original_max_epochs = int(np.ceil(total_steps / n_batches))
    epochs = min(int(params.get("epoch", original_max_epochs)), original_max_epochs)
    flat_steps = int(total_steps * float(params.get("flat_lr_proportion", 0.7)))

    def lr_factor(step):
        if step <= flat_steps:
            return 1.0
        progress = min(1.0, (step - flat_steps) / max(1, total_steps - flat_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(base_optimizer, lr_lambda=lr_factor)
    patience = int(params.get("early_stopping_rounds", 5))
    eval_every = int(params.get("eval_every_n", 5))
    gradient_clip = float(params.get("gradient_clip", 1.0))
    best_val_loss = float("inf")
    best_score = -float("inf")
    best_epoch = 0
    best_state = None
    bad_evaluations = 0
    train_loss, val_scores, test_scores = [], [], []
    train_index_set = set(prepared["train_indices"].tolist())

    for epoch in range(epochs):
        model.train()
        losses = []
        for indices in _mixed_batches(prepared, batch_size, shuffle=True, seed=seed + epoch):
            local_train = np.array([index in train_index_set for index in indices])
            if not local_train.any():
                continue
            idx = torch.tensor(indices, device=device)
            target = torch.tensor(local_train, dtype=torch.bool, device=device)
            optimizer.zero_grad()
            out, num_out, cat_out, _ = _batch_forward(model, tensors, idx)
            loss = _label_loss(out[target], tensors["y"][idx][target], task)
            supervised_reconstruction_weight = float(params.get("supervised_reconstruction_weight", 0.0))
            if supervised_reconstruction_weight:
                loss = loss + supervised_reconstruction_weight * _reconstruction_loss(
                    num_out, cat_out, tensors, idx
                )
            loss.backward()
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_value_(model.parameters(), gradient_clip)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))
        train_loss.append(float(np.mean(losses)) if losses else 0.0)
        if (epoch + 1) % eval_every != 0 and epoch + 1 != epochs:
            if params.get("verbose", False):
                print(
                    f"[MISS-NPT] epoch {epoch + 1}/{epochs}, "
                    f"train_label_loss={train_loss[-1]:.6f}"
                )
            continue

        val = _evaluate(model, tensors, prepared, "valid", task, batch_size)
        test = _evaluate(model, tensors, prepared, "test", task, batch_size)
        val["epoch"] = epoch + 1
        test["epoch"] = epoch + 1
        val_scores.append(val)
        test_scores.append(test)
        if params.get("verbose", False):
            print(
                f"[MISS-NPT] epoch {epoch + 1}/{epochs}, "
                f"train_label_loss={train_loss[-1]:.6f}, "
                f"val_label_loss={val['label_loss']:.6f}, val_score={val['score']:.6f}"
            )
        if val["label_loss"] < best_val_loss:
            best_val_loss = val["label_loss"]
            best_score = val["score"]
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_evaluations = 0
        else:
            bad_evaluations += 1
            if bad_evaluations >= patience:
                if params.get("verbose", False):
                    print(f"[MISS-NPT] early stopping at epoch {epoch + 1}, best epoch {best_epoch}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    filled_features = prepared["npt_filled_X"]
    imputed_data = sampled_df.copy()
    for col in prepared["feature_columns"]:
        miss = prepared["original_missing"][col].to_numpy()
        imputed_data.loc[miss, col] = filled_features.loc[miss, col]

    return {
        "model": model,
        "prepared": prepared,
        "params": dict(params),
        "task": task,
        "device": str(device),
        "metrics": {
            "pretrain_loss": pretrain_loss,
            "train_loss": train_loss,
            "val_scores": val_scores,
            "test_scores": test_scores,
            "best_score": float(best_score),
            "best_val_loss": float(best_val_loss),
            "best_epoch": best_epoch,
            "total_parameters": sum(p.numel() for p in model.parameters()),
        },
        "sampled_data": sampled_df,
        "imputed_data": imputed_data,
        "indicator": indicator,
        "ips": ips,
        "label_column": label_column,
        "input_columns": list(datamiss_df.columns),
        "index": datamiss_df.index,
    }


def predict_miss_npt_algorithm(model, data, missing_mask=None, train_state=None, device=None):
    if train_state is None:
        raise ValueError("train_state is required for MISS-NPT prediction.")
    df = tabformer_modules._prepare_predict_frame(data, train_state)
    prepared = train_state["prepared"]
    label_column = train_state["label_column"]
    original_missing = tabformer_modules._missing_frame_from_mask(df, label_column, missing_mask)
    result_features = df.drop(columns=[label_column]).copy()

    for col in prepared["continuous_columns"]:
        result_features.loc[original_missing[col], col] = prepared["fill_values"][col]
    for col in prepared["categorical_columns"]:
        fill_code = int(prepared["fill_values"][col])
        fill_code = np.clip(fill_code, 0, len(prepared["encoders"][col].classes_) - 1)
        decoded = prepared["encoders"][col].inverse_transform([fill_code])[0]
        mode = prepared["categorical_modes"].get(col)
        if decoded == "MissingValue" and mode is not None:
            decoded = mode
        filled = result_features[col].astype(object).copy()
        filled.loc[original_missing[col]] = decoded
        result_features[col] = saint_modules._safe_numeric_cast(filled, df[col])

    result = df.copy()
    for col in result_features.columns:
        result[col] = result_features[col]
    return result

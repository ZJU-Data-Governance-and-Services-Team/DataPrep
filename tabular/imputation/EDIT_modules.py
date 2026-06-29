"""
EDIT 算法的底层组件 (网络结构、工具函数、训练循环、影响函数)

参考论文:
    Miao et al. "Efficient and effective data imputation with influence functions." VLDB 2021.

整体流程:
    1. 用一小部分初始训练集训练 GAIN
    2. 用影响函数 (Hessian^-1 · grad) 估计每个样本对验证集 loss 的贡献
    3. 按影响值从大到小累加，选出 Top-k 个最有用的样本
    4. 用选出的 Top-k 样本重训练 GAIN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm


# ==========================================
# 1. 神经网络组件 (Neural Networks)
# ==========================================

class EditGenerator(nn.Module):
    """生成器: 输入 (X + Mask)，输出补全后的数据"""
    def __init__(self, dim, h_dim):
        super(EditGenerator, self).__init__()
        self.fc1 = nn.Linear(dim * 2, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, m):
        inputs = torch.cat([x, m], dim=1)
        h1 = F.relu(self.fc1(inputs))
        h2 = F.relu(self.fc2(h1))
        return torch.sigmoid(self.fc3(h2))


class EditDiscriminator(nn.Module):
    """判别器: 输入 (Hat_X + Hint)，输出每个位置是观测还是补全的概率"""
    def __init__(self, dim, h_dim):
        super(EditDiscriminator, self).__init__()
        self.fc1 = nn.Linear(dim * 2, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, h):
        inputs = torch.cat([x, h], dim=1)
        h1 = F.relu(self.fc1(inputs))
        h2 = F.relu(self.fc2(h1))
        return torch.sigmoid(self.fc3(h2))


# ==========================================
# 2. 工具函数 (Utils)
# ==========================================

def normalization(data):
    """Min-Max 归一化，处理 NaN"""
    _min = np.nanmin(data, axis=0)
    _max = np.nanmax(data, axis=0)
    _den = _max - _min
    _den[_den == 0] = 1e-6
    norm_data = (data - _min) / _den
    norm_parameters = {'min': _min, 'max': _max, 'den': _den}
    return norm_data, norm_parameters


def normalization_with_parameter(data, norm_parameters):
    return (data - norm_parameters['min']) / norm_parameters['den']


def renormalization(norm_data, norm_parameters):
    return norm_data * norm_parameters['den'] + norm_parameters['min']

def rounding(imputed_data, data_x):
    """
    对类别型变量做四舍五入。
    若某列观测到的唯一值数量少于 20，则认为它可能是类别型变量。
    """
    rounded_data = imputed_data.copy()
    _, dim = data_x.shape

    for i in range(dim):
        observed = data_x[~np.isnan(data_x[:, i]), i]
        if len(np.unique(observed)) < 20:
            rounded_data[:, i] = np.round(rounded_data[:, i])

    return rounded_data


def sample_Z(batch_size, dim):
    """生成随机噪声 Z (用于初始化缺失位置)"""
    return np.random.uniform(0., 0.01, size=[batch_size, dim])


def sample_M(batch_size, dim, p):
    """生成 Hint 向量所需的随机掩码 """
    unif_random_matrix = np.random.uniform(0., 1., size=[batch_size, dim])
    return 1. * (unif_random_matrix < p)

def sample_M_fixed(batch_size, dim, p, seed=50):
    """influence 阶段使用的固定 hint mask"""
    rng = np.random.RandomState(seed)
    unif_random_matrix = rng.uniform(0., 1., size=[batch_size, dim])
    return 1. * (unif_random_matrix < p)

# ==========================================
# 3. Loss 计算 (Loss Helpers)
# ==========================================

def _compute_generator_loss(generator, discriminator, x_in, m, h, alpha, damping):
    """
    生成器损失 = G_loss1 + alpha * MSE + damping * L2(theta_G)
    对应原 TF 代码:
        G_loss = G_loss1 + alpha * MSE_train_loss
                       + damping * tf.reduce_mean([tf.nn.l2_loss(i) for i in theta_G])
    """
    g_sample = generator(x_in, m)
    hat_x = x_in * m + g_sample * (1 - m)
    d_prob = discriminator(hat_x, h)

    g_loss1 = -torch.mean((1 - m) * torch.log(d_prob + 1e-8))
    mse_loss = torch.mean((m * x_in - m * g_sample) ** 2) / (torch.mean(m) + 1e-8)

    g_params = list(generator.parameters())
    l2_reg = sum(0.5 * (p ** 2).sum() for p in g_params) / max(1, len(g_params))

    return g_loss1 + alpha * mse_loss + damping * l2_reg, mse_loss


# ==========================================
# 4. 训练步骤 (Training Loops)
# ==========================================

def _train_step(generator, discriminator, data_x, mask, batch_idx,
                opt_g, opt_d, params, device):
    """单个 mini-batch 的训练（D 一次 + G 一次）"""
    x_mb = data_x[batch_idx]
    m_mb = mask[batch_idx]
    bs, dim = x_mb.shape

    z_mb = sample_Z(bs, dim)
    h_mb_temp = sample_M(bs, dim, params['hint_rate'])
    h_mb = m_mb * h_mb_temp
    x_in = m_mb * x_mb + (1 - m_mb) * z_mb

    x_in_t = torch.tensor(x_in, dtype=torch.float32).to(device)
    m_t = torch.tensor(m_mb, dtype=torch.float32).to(device)
    h_t = torch.tensor(h_mb, dtype=torch.float32).to(device)

    # ----- Train Discriminator -----
    opt_d.zero_grad()
    g_sample = generator(x_in_t, m_t)
    hat_x = x_in_t * m_t + g_sample * (1 - m_t)
    d_prob = discriminator(hat_x.detach(), h_t)
    d_loss = -torch.mean(m_t * torch.log(d_prob + 1e-8) +
                          (1 - m_t) * torch.log(1 - d_prob + 1e-8))
    d_loss.backward()
    opt_d.step()

    # ----- Train Generator -----
    opt_g.zero_grad()
    g_loss, mse_loss = _compute_generator_loss(
        generator, discriminator, x_in_t, m_t, h_t,
        params['alpha'], params['damping']
    )
    g_loss.backward()
    opt_g.step()

    return g_loss.item(), d_loss.item(), mse_loss.item()


def _run_training_phase(generator, discriminator, data_x, mask,
                         opt_g, opt_d, params, device, phase_name="Training"):
    """跑一个完整的训练阶段（多个 epoch）"""
    no = data_x.shape[0]
    pbar = tqdm(range(params['epoch']), desc=phase_name)
    for _ in pbar:
        idx = np.random.permutation(no)
        cur_g, cur_d, cur_mse = 0.0, 0.0, 0.0
        for i in range(0, no, params['batch_size']):
            if i + params['batch_size'] > no:
                break
            batch_idx = idx[i:i + params['batch_size']]
            cur_g, cur_d, cur_mse = _train_step(
                generator, discriminator, data_x, mask,
                batch_idx, opt_g, opt_d, params, device
            )
        pbar.set_postfix({
            'G': f'{cur_g:.4f}',
            'D': f'{cur_d:.4f}',
            'MSE': f'{cur_mse:.4f}',
        })


# ==========================================
# 5. 影响函数 (Influence Function Core)
# ==========================================

def _make_inputs_torch(data_x, mask, params, device, fixed_hint=False):
    """把 np 数据组装成训练时的 (x_in, m, h)，全部 GPU tensor"""
    no, dim = data_x.shape
    z = sample_Z(no, dim)
    hint_sampler = sample_M_fixed if fixed_hint else sample_M
    h = mask * hint_sampler(no, dim, params['hint_rate'])
    x_in = mask * data_x + (1 - mask) * z

    return (
        torch.tensor(x_in, dtype=torch.float32).to(device),
        torch.tensor(mask, dtype=torch.float32).to(device),
        torch.tensor(h, dtype=torch.float32).to(device),
    )


def _compute_inverse_hessian_approx(generator, discriminator, init_data, init_mask,
                                       params, device):
    """
    用 grad · grad^T 近似 Hessian (Fisher information matrix)，并求逆。
    对应原 TF 代码里的:
        Fast_Hessians = [final[i] * final_zhuan[i] for ...]
        H_hessians   = Fast_Hessians + eye * 1e-3
        H_invert     = [tf.linalg.inv(item + 1e-3) for ...]
    返回: list[Tensor], 每层一个 H^-1 矩阵
    """
    x_in, m, h = _make_inputs_torch(init_data, init_mask, params, device, fixed_hint=True)
    g_loss, _ = _compute_generator_loss(
        generator, discriminator, x_in, m, h,
        params['alpha'], params['damping']
    )

    g_params = list(generator.parameters())
    grads = torch.autograd.grad(g_loss, g_params, retain_graph=False)

    h_inv_list = []
    for g in grads:
        g_flat = g.contiguous().view(-1, 1)              # [P, 1]
        hess = g_flat @ g_flat.t()                        # [P, P]
        eye = torch.eye(hess.size(0), device=device) * 1e-3
        h_inv = torch.linalg.inv(hess + eye + 1e-3)
        h_inv_list.append(h_inv)
    return h_inv_list


def _compute_layer_gradients(generator, discriminator, data_x, mask, params, device, fixed_hint=False):
    """计算 generator 各层的梯度（flatten 后），返回 list[Tensor] (1, P) 形状"""
    x_in, m, h = _make_inputs_torch(data_x, mask, params, device, fixed_hint=fixed_hint)
    g_loss, _ = _compute_generator_loss(
        generator, discriminator, x_in, m, h,
        params['alpha'], params['damping']
    )
    g_params = list(generator.parameters())
    grads = torch.autograd.grad(g_loss, g_params)
    return [g.contiguous().view(1, -1) for g in grads]   # [1, P_i]


def compute_influence_scores(generator, discriminator,
                              full_data, full_mask, val_data, val_mask,
                              init_data, init_mask,
                              params, device):
    """
    对 full_data 中每个样本 i 计算影响分数:
        score_i = sum_layer ( val_grad · H_init^-1 · train_grad_i )

    Args:
        full_data / full_mask: 待打分的完整数据集
        val_data / val_mask  : 用来定义 "对什么有帮助" 的验证集
        init_data / init_mask: 用来估计 Hessian 的初始训练集
    """
    # 1. Hessian 的逆（基于初始训练集）
    h_inv = _compute_inverse_hessian_approx(
        generator, discriminator, init_data, init_mask, params, device
    )

    # 2. 验证集的梯度（行向量）
    val_grad = _compute_layer_gradients(
        generator, discriminator, val_data, val_mask, params, device, fixed_hint=True
    )

    # 3. IHVP[i] = val_grad[i] @ H_inv[i]  形状 [1, P_i]
    ihvp = [val_grad[i] @ h_inv[i] for i in range(len(h_inv))]
    ihvp_concat = torch.cat(ihvp, dim=1)                # [1, sum P_i]

    # 4. 对每个训练样本单独求梯度并算内积
    n_full = full_data.shape[0]
    scores = np.zeros(n_full, dtype=np.float32)

    print(f"[EDIT] Computing influence for {n_full} samples...")
    for i in tqdm(range(n_full), desc="Influence"):
        x_row = full_data[i:i + 1]
        m_row = full_mask[i:i + 1]
        # 单样本梯度 [P_i, 1] 拼成 [sum P_i, 1]
        layer_grads = _compute_layer_gradients(
            generator, discriminator, x_row, m_row, params, device, fixed_hint=True
        )
        train_grad = torch.cat([g.t() for g in layer_grads], dim=0)
        infl = (ihvp_concat @ train_grad).item()
        scores[i] = 0.0 if np.isnan(infl) else infl

    return scores


def select_top_k_by_influence(scores):
    """
    按论文逻辑选 Top-k:
        从大到小累加 score，直到累计值首次超过 sum(scores) 为止。
    """
    influence_sum = float(scores.sum())
    order = np.argsort(-scores)                          # 降序

    top_k = []
    cumulative = 0.0
    for idx in order:
        if cumulative > influence_sum:
            break
        cumulative += float(scores[idx])
        top_k.append(int(idx))

    # 兜底: 至少保留一定数量样本，避免极端情况下 top_k 太短
    if len(top_k) < max(1, int(0.1 * len(scores))):
        top_k = order[:max(1, int(0.1 * len(scores)))].tolist()

    return top_k


# ==========================================
# 6. 主算法入口 (Main Algorithm)
# ==========================================

def train_edit_algorithm(generator, discriminator, data_x, mask, params, device):
    """
    EDIT 三阶段训练:
        Phase 1: 初始训练 (小训练集)
        Phase 2: 影响函数打分 + Top-k 样本筛选
        Phase 3: 重训练 (基于 Top-k 样本)
    """
    no, dim = data_x.shape
    params.setdefault('damping', 1e-2)

    raw_data = np.array(data_x, dtype=np.float64)
    raw_mask = np.array(mask, dtype=np.float32)
    raw_data[raw_mask == 0] = np.nan

    # 切分: init + val (剩下的还是参与影响函数打分)
    init_n = int(params['initial_size'])
    val_n = int(params['validation_size'])
    sample_idx = np.random.randint(len(data_x), size=init_n + val_n)
    init_idx = sample_idx[:init_n]
    val_idx = sample_idx[init_n:]

    # full data normalization
    norm_data, norm_parameters = normalization(raw_data)
    norm_data_x = np.nan_to_num(norm_data, 0).astype(np.float32)

    # Initial data normalization
    init_raw = raw_data[init_idx]
    init_mask = raw_mask[init_idx]
    init_norm_data, _ = normalization(init_raw)
    init_data = np.nan_to_num(init_norm_data, 0).astype(np.float32)

    # Validation data normalization
    val_raw = raw_data[val_idx]
    val_mask = raw_mask[val_idx]
    val_norm_data, _ = normalization(val_raw)
    val_data = np.nan_to_num(val_norm_data, 0).astype(np.float32)

    opt_g = optim.Adam(generator.parameters())
    opt_d = optim.Adam(discriminator.parameters())

    # ===== Phase 1: 初始训练 =====
    print(f"\n[EDIT] Phase 1: Initial training on {init_n} samples for {params['epoch']} epochs")
    _run_training_phase(
        generator, discriminator, init_data, init_mask,
        opt_g, opt_d, params, device, phase_name="Initial"
    )

    # ===== Phase 2: 影响函数打分 + Top-k =====
    print(f"\n[EDIT] Phase 2: Scoring all {no} samples via influence function")
    scores = compute_influence_scores(
        generator, discriminator,
        norm_data_x, raw_mask,    # 给全量数据打分
        val_data, val_mask,       # 看对哪些验证样本有帮助
        init_data, init_mask,     # Hessian 用初始训练集估
        params, device
    )
    top_k = select_top_k_by_influence(scores)
    print(f"[EDIT] Selected {len(top_k)} / {no} samples for retraining "
          f"({len(top_k) / no:.1%}).")

    # ===== Phase 3: 重训练 =====
    final_data = norm_data_x[top_k]
    final_mask = raw_mask[top_k]
    print(f"\n[EDIT] Phase 3: Retraining on {len(top_k)} selected samples")
    _run_training_phase(
        generator, discriminator, final_data, final_mask,
        opt_g, opt_d, params, device, phase_name="Retrain"
    )

    print("[EDIT] Training complete.")
    return norm_parameters

import torch
import torch.nn.functional as F
import math
from typing import Union
from models.diffae.core import TabularAE, TabularUNet
from models.imputation.standard_pipeline import combine_features
from models.diffae.core import compute_embedding_diffusion_loss, DiffusionUtils

def create_cond_mask(original_error_mask: torch.Tensor, device: str) -> torch.Tensor:
    """
    在原始正确的数据上生成条件掩码。
    
    Args:
        original_error_mask (torch.Tensor): 形状为 [B, F], 1表示原始错误, 0表示原始正确。
        t (torch.Tensor): 当前时间步, 形状 [B, F]。(此参数保留但不使用)
        total_timesteps (int): 总扩散步数 T。(此参数保留但不使用)
        device (str): 设备。
        
    Returns:
        torch.Tensor: 条件掩码 cond_mask, 形状 [B, F]。1表示该位置作为条件, 0表示作为预测目标。
    """
    batch_size, feature_number = original_error_mask.shape
    
    # 1. 生成随机的mask概率 (在0到1之间)
    mask_prob = torch.rand(batch_size, feature_number, device=device)
    
    # 2. 生成随机阈值
    random_threshold = torch.rand(batch_size, 1, device=device)
    
    # 3. 确定在"原始正确"的位置上哪些要被保留
    # is_correct_and_kept: 值为True的位置是原始正确且根据随机概率被保留的
    is_correct_and_kept = (original_error_mask == 0) & (random_threshold < mask_prob)
    
    # 4. 构建最终的 cond_mask
    # 只有 is_correct_and_kept 的位置为1, 其他所有位置(包括原始错误的位置)都为0
    cond_mask = torch.zeros_like(original_error_mask, dtype=torch.float32)
    cond_mask[is_correct_and_kept] = 1
    
    return cond_mask


def _expand_mask_to_onehot_dim(mask_orig_dim: torch.Tensor, actual_cat_sizes: list) -> torch.Tensor:
    """
    将原始维度的掩码扩展到one-hot维度
    
    Args:
        mask_orig_dim: 原始维度的掩码 [B, d_numerical + d_categorical]
        actual_cat_sizes: 实际分类特征大小列表
        
    Returns:
        扩展后的掩码 [B, d_numerical + sum(actual_cat_sizes)]
    """
    batch_size = mask_orig_dim.shape[0]
    device = mask_orig_dim.device
    
    # 数值特征在前，分类特征在后
    d_numerical = mask_orig_dim.shape[1] - len(actual_cat_sizes)
    
    # 数值特征部分不变
    num_mask = mask_orig_dim[:, :d_numerical]
    
    # 分类特征部分需要扩展 - 使用向量化操作提高效率
    if actual_cat_sizes:
        # 提取所有分类特征的掩码值 [B, d_categorical]
        cat_mask_orig = mask_orig_dim[:, d_numerical:]
        
        # 使用repeat_interleave一次性扩展所有分类特征
        cat_mask_expanded = torch.repeat_interleave(
            cat_mask_orig, 
            torch.tensor(actual_cat_sizes, device=device), 
            dim=1
        )
        cat_mask = cat_mask_expanded
    else:
        cat_mask = torch.empty(batch_size, 0, device=device)
    
    # 合并数值和分类特征掩码
    if actual_cat_sizes:
        expanded_mask = torch.cat([num_mask, cat_mask], dim=1)
    else:
        expanded_mask = num_mask
    
    return expanded_mask


def _convert_onehot_to_indices(x_cat_onehot: torch.Tensor, actual_cat_sizes: list, device: str) -> torch.Tensor:
    """
    将one-hot编码转换回分类索引
    
    Args:
        x_cat_onehot: one-hot编码的分类特征 [B, sum(actual_cat_sizes)]
        actual_cat_sizes: 实际分类特征大小列表
        device: 设备
        
    Returns:
        分类特征索引 [B, d_categorical]
    """
    if x_cat_onehot.shape[1] == 0:
        return torch.empty(x_cat_onehot.shape[0], 0, dtype=torch.long, device=device)
    
    batch_size = x_cat_onehot.shape[0]
    cat_indices = []
    start_idx = 0
    
    for cat_size in actual_cat_sizes:
        if cat_size > 0:
            end_idx = start_idx + cat_size
            cat_slice = x_cat_onehot[:, start_idx:end_idx]
            cat_idx = torch.argmax(cat_slice, dim=1)
            cat_indices.append(cat_idx)
            start_idx = end_idx
    
    return torch.stack(cat_indices, dim=1) if cat_indices else torch.empty(batch_size, 0, dtype=torch.long, device=device)


def _create_onehot_from_indices(x_cat_indices: torch.Tensor, actual_cat_sizes: list) -> torch.Tensor:
    """
    从分类索引创建one-hot编码
    
    Args:
        x_cat_indices: 分类特征索引 [B, d_categorical]
        actual_cat_sizes: 实际分类特征大小列表
        
    Returns:
        one-hot编码 [B, sum(actual_cat_sizes)]
    """
    device = x_cat_indices.device
    batch_size = x_cat_indices.shape[0]
    
    if x_cat_indices.shape[1] > 0:
        # Use torch.nn.functional.one_hot directly for each categorical feature
        onehot_parts = []
        for i, cat_size in enumerate(actual_cat_sizes):
            if cat_size > 0:
                # Clamp indices to valid range for this categorical feature
                indices_clamped = x_cat_indices[:, i].clamp(0, cat_size - 1)
                # Create one-hot encoding for this feature
                onehot = torch.nn.functional.one_hot(indices_clamped, num_classes=cat_size)
                onehot_parts.append(onehot)
    else:
        onehot_parts = []
    
    if onehot_parts:
        return torch.cat(onehot_parts, dim=1).float()  # 确保返回float类型
    else:
        return torch.empty(batch_size, 0, device=device, dtype=torch.float32)


def compute_diffusion_loss(
    repair_model: Union[TabularAE, TabularUNet], 
    feature_embedder, 
    diffusion_utils: DiffusionUtils, 
    processor,
    train_batch_x, 
    train_batch_y, 
    train_batch_mask,  # 传入训练批次张量
    actual_cat_sizes, 
    device, 
    args
) -> torch.Tensor:
    """
    计算外层优化中的Diffusion重建损失 L_diff。
    支持两种模式：
    1. 嵌入空间扩散 (diffusion_embed=True): 在嵌入空间中进行扩散过程
    2. 原始数据空间扩散 (diffusion_embed=False): 在原始特征空间中进行扩散
    
    Args:
        repair_model: 修复模型
        feature_embedder: 特征嵌入器 
        diffusion_utils: 扩散工具类
        processor: 数据处理器
        train_batch_x: 训练批次特征张量 (已经过transform处理)
        train_batch_y: 训练批次标签张量
        train_batch_mask: 训练批次缺失掩码张量
        actual_cat_sizes: 实际分类特征大小
        device: 设备
        args: 参数配置
        
    Returns:
        torch.Tensor: Diffusion损失值
    """
    # 检查是否使用嵌入空间扩散
    use_embedding_diffusion = getattr(args, 'diffusion_embed', True)
    
    if use_embedding_diffusion:
        # 嵌入空间扩散模式
        return compute_diffusion_loss_embedding(
            repair_model, feature_embedder, diffusion_utils, processor,
            train_batch_x, train_batch_y, train_batch_mask,
            actual_cat_sizes, device, args
        )
    else:
        # 原始数据空间扩散模式
        return compute_diffusion_loss_original(
            repair_model, feature_embedder, diffusion_utils, processor,
            train_batch_x, train_batch_y, train_batch_mask,
            actual_cat_sizes, device, args
        )


def compute_diffusion_loss_embedding(
    repair_model: Union[TabularAE, TabularUNet], 
    feature_embedder, 
    diffusion_utils: DiffusionUtils, 
    processor,
    train_batch_x, 
    train_batch_y, 
    train_batch_mask,
    actual_cat_sizes, 
    device, 
    args
) -> torch.Tensor:
    """
    计算嵌入空间中的Diffusion损失（新版本）
    
    Args:
        repair_model: 修复模型
        feature_embedder: 特征嵌入器 
        diffusion_utils: 扩散工具类
        processor: 数据处理器
        train_batch_x: 训练批次特征张量 (已经过transform处理)
        train_batch_y: 训练批次标签张量
        train_batch_mask: 训练批次缺失掩码张量
        actual_cat_sizes: 实际分类特征大小
        device: 设备
        args: 参数配置
        
    Returns:
        torch.Tensor: Diffusion损失值
    """
    # 将输入数据移到设备上
    train_batch_x = train_batch_x.to(device)
    train_batch_mask = train_batch_mask.to(device)
    
    # 1. 从合并的特征中分离数值和分类特征
    d_numerical = len(processor.num_features) if hasattr(processor, 'num_features') else 0
    d_categorical = len(processor.cat_features) if hasattr(processor, 'cat_features') else 0
    
    # 提前计算原始特征维度
    expected_orig_dim = d_numerical + d_categorical

    if d_numerical > 0 and d_categorical > 0:
        # 计算独热编码后的分类特征维度
        cat_onehot_dim = sum(actual_cat_sizes)
        x_num = train_batch_x[:, :d_numerical]
        x_cat_onehot = train_batch_x[:, d_numerical:d_numerical + cat_onehot_dim]
        
        # 将one-hot编码转换回分类索引，使用已有的转换函数
        x_cat = _convert_onehot_to_indices(x_cat_onehot, actual_cat_sizes, device)

    elif d_numerical > 0:
        x_num = train_batch_x
        x_cat = torch.empty(x_num.shape[0], 0, dtype=torch.long, device=device)
        x_cat_onehot = torch.empty(x_num.shape[0], 0, device=device)
    else:
        x_num = torch.empty(train_batch_x.shape[0], 0, device=device)
        x_cat_onehot = train_batch_x
        # 将one-hot编码转换回分类索引，使用已有的转换函数
        x_cat = _convert_onehot_to_indices(x_cat_onehot, actual_cat_sizes, device)

    # 2. 获取干净的初始嵌入 e_0
    batch_size = train_batch_x.shape[0]
    # 创建全1掩码表示所有特征都有效
    num_mask = torch.ones_like(x_num) if x_num.shape[1] > 0 else torch.empty(batch_size, 0, device=device)
    cat_mask = torch.ones_like(x_cat, dtype=torch.float) if x_cat.shape[1] > 0 else torch.empty(batch_size, 0, device=device)
    
    # 通过feature_embedder获取干净的嵌入
    e_0, _ = feature_embedder(x_num, x_cat, num_mask, cat_mask)
    
    # 3. 在嵌入空间中执行前向加噪
    # 随机采样时间步 t (batch-level，在嵌入空间中统一时间步)
    if getattr(args, 't_batch_level', False):
        t = torch.randint(0, diffusion_utils.num_timesteps, (batch_size,), device=device).long()
    else:
        # 随机采样时间步 t (cell-level，每个特征都有独立的时间步)
        num_features = d_numerical + d_categorical
        t = torch.randint(0, diffusion_utils.num_timesteps, (batch_size, num_features), device=device).long()
    
    
    # 在嵌入空间中加噪
    e_t, epsilon = diffusion_utils.q_sample_embedding(e_0, t)
    
    # 4. 生成条件掩码
    # train_batch_mask可能已经是合并特征的掩码，需要检查其维度
    expected_onehot_dim = d_numerical + sum(actual_cat_sizes)
    
    if train_batch_mask.shape[1] == expected_onehot_dim:
        # train_batch_mask已经是one-hot维度的掩码
        # 需要将其转换回原始维度
        original_error_mask = torch.zeros((batch_size, expected_orig_dim), device=device, dtype=train_batch_mask.dtype)
        if d_numerical > 0:
            original_error_mask[:, :d_numerical] = train_batch_mask[:, :d_numerical]
        if d_categorical > 0:
            # 对于分类特征，取每个分类特征的第一个one-hot位置作为代表
            cat_starts = torch.cumsum(torch.tensor([d_numerical] + actual_cat_sizes[:-1]), dim=0)
            original_error_mask[:, d_numerical:d_numerical + d_categorical] = train_batch_mask[:, cat_starts]
    elif train_batch_mask.shape[1] == expected_orig_dim:
        # train_batch_mask已经是原始维度的掩码
        original_error_mask = train_batch_mask
    else:
        # 维度不匹配，使用零掩码作为fallback
        print(f"警告：train_batch_mask维度({train_batch_mask.shape[1]})不匹配预期的原始维度({expected_orig_dim})或one-hot维度({expected_onehot_dim})")
        original_error_mask = torch.zeros((batch_size, expected_orig_dim), device=device, dtype=torch.float32)
    
    cond_mask_orig_dim = create_cond_mask(original_error_mask, device)
    
    # 5. 构建新的两通道模型输入：condition + noise
    # 将条件掩码扩展到嵌入维度
    emb_dim = e_0.shape[1]
    feature_emb_dim = emb_dim // expected_orig_dim  # 每个特征的嵌入维度
    e_mask = cond_mask_orig_dim.unsqueeze(2).expand(-1, -1, feature_emb_dim).reshape(batch_size, -1).float()
    ori_err_mask_dim = original_error_mask.unsqueeze(2).expand(-1, -1, feature_emb_dim).reshape(batch_size, -1).float()

    # 通道一：条件数据 - 只包含已知（条件）部分的干净数据，其他位置为0
    e_condition = e_0 * e_mask
    
    # 通道二：噪声数据 - 只包含需要预测位置的噪声，其他位置为0  
    e_noise = e_t * (1 - e_mask - ori_err_mask_dim)
    
    # 拼接两个通道：condition + noise
    model_input = torch.cat([e_condition, e_noise], dim=1)  # [B, emb_dim*2]
    
    # 6. 模型预测噪声
    epsilon_pred = repair_model(model_input, t, y_labels=None)
    
    # 7. 计算损失 - MSE between true noise and predicted noise
    # 只在未知区域计算损失
    loss_mask_emb = (1 - e_mask - ori_err_mask_dim)
    
    # 确保epsilon_pred的形状与epsilon匹配
    if isinstance(epsilon_pred, tuple):
        # 如果模型返回的是元组（例如(pred_num, pred_cat)），我们需要提取预测的噪声
        # 暂时假设模型已经调整为返回单个噪声预测
        epsilon_pred = epsilon_pred[0]  # 取第一个元素作为噪声预测
    
    # 计算MSE损失
    loss = F.mse_loss(epsilon * loss_mask_emb, epsilon_pred * loss_mask_emb)
    
    return loss


def compute_diffusion_loss_original(
    repair_model: Union[TabularAE, TabularUNet], 
    feature_embedder, 
    diffusion_utils: DiffusionUtils, 
    processor,
    train_batch_x, 
    train_batch_y, 
    train_batch_mask,
    actual_cat_sizes, 
    device, 
    args
) -> torch.Tensor:
    """
    计算原始数据空间中的Diffusion损失（旧版本）
    
    Args:
        repair_model: 修复模型
        feature_embedder: 特征嵌入器 
        diffusion_utils: 扩散工具类
        processor: 数据处理器
        train_batch_x: 训练批次特征张量 (已经过transform处理)
        train_batch_y: 训练批次标签张量
        train_batch_mask: 训练批次缺失掩码张量
        actual_cat_sizes: 实际分类特征大小
        device: 设备
        args: 参数配置
        
    Returns:
        torch.Tensor: Diffusion损失值
    """
    # 将输入数据移到设备上
    train_batch_x = train_batch_x.to(device)
    train_batch_mask = train_batch_mask.to(device)
    
    # 1. 从合并的特征中分离数值和分类特征
    d_numerical = len(processor.num_features) if hasattr(processor, 'num_features') else 0
    d_categorical = len(processor.cat_features) if hasattr(processor, 'cat_features') else 0
    
    if d_numerical > 0 and d_categorical > 0:
        # 计算独热编码后的分类特征维度
        cat_onehot_dim = sum(actual_cat_sizes)
        x_num = train_batch_x[:, :d_numerical]
        x_cat_onehot = train_batch_x[:, d_numerical:d_numerical + cat_onehot_dim]
        
        # 将one-hot编码转换回分类索引
        x_cat = _convert_onehot_to_indices(x_cat_onehot, actual_cat_sizes, device)
    elif d_numerical > 0:
        x_num = train_batch_x
        x_cat = torch.empty(x_num.shape[0], 0, dtype=torch.long, device=device)
        x_cat_onehot = torch.empty(x_num.shape[0], 0, device=device)
    else:
        x_num = torch.empty(train_batch_x.shape[0], 0, device=device)
        x_cat_onehot = train_batch_x
        x_cat = _convert_onehot_to_indices(x_cat_onehot, actual_cat_sizes, device)
    
    # 2. 随机采样时间步 t
    batch_size = train_batch_x.shape[0]
    if getattr(args, 't_batch_level', False):
        t = torch.randint(0, diffusion_utils.num_timesteps, (batch_size,), device=device).long()
    else:
        # 随机采样时间步 t (cell-level，每个特征都有独立的时间步)
        num_features = d_numerical + d_categorical
        t = torch.randint(0, diffusion_utils.num_timesteps, (batch_size, num_features), device=device).long()
    
    # 3. 在原始数据空间中加噪
    x_combined = train_batch_x
    x_t = diffusion_utils.q_sample(x_combined, t)
    
    # 4. 生成条件掩码
    expected_orig_dim = d_numerical + d_categorical
    expected_onehot_dim = d_numerical + sum(actual_cat_sizes)
    
    if train_batch_mask.shape[1] == expected_onehot_dim:
        # train_batch_mask已经是one-hot维度的掩码，直接使用
        cond_mask_onehot = create_cond_mask(train_batch_mask, device)
    else:
        # 需要将原始维度掩码扩展到one-hot维度
        original_error_mask = train_batch_mask
        cond_mask_orig_dim = create_cond_mask(original_error_mask, device)
        cond_mask_onehot = _expand_mask_to_onehot_dim(cond_mask_orig_dim, actual_cat_sizes)
    
    # 5. 构建三通道输入
    single_channel_dim = x_combined.shape[1]
    
    # 通道1: noisy_target (对缺失位置加噪，已知位置为0)
    noisy_target = x_t * (1 - cond_mask_onehot)
    
    # 通道2: condition_data (对已知位置使用原始数据，缺失位置为0)
    condition_data = x_combined * cond_mask_onehot
    
    # 通道3: condition_mask
    condition_mask = cond_mask_onehot
    
    # 通过feature_embedder编码三个通道
    if x_num.shape[1] > 0:
        # 分离数值和分类特征进行编码
        noisy_num = noisy_target[:, :d_numerical]
        noisy_cat_onehot = noisy_target[:, d_numerical:]
        noisy_cat = _convert_onehot_to_indices(noisy_cat_onehot, actual_cat_sizes, device)
        
        cond_num = condition_data[:, :d_numerical]
        cond_cat_onehot = condition_data[:, d_numerical:]
        cond_cat = _convert_onehot_to_indices(cond_cat_onehot, actual_cat_sizes, device)
    else:
        noisy_num = torch.empty(batch_size, 0, device=device)
        noisy_cat = _convert_onehot_to_indices(noisy_target, actual_cat_sizes, device)
        
        cond_num = torch.empty(batch_size, 0, device=device)
        cond_cat = _convert_onehot_to_indices(condition_data, actual_cat_sizes, device)
    
    # 创建全1掩码用于编码
    ones_mask_num = torch.ones_like(noisy_num) if noisy_num.shape[1] > 0 else torch.empty(batch_size, 0, device=device)
    ones_mask_cat = torch.ones_like(noisy_cat, dtype=torch.float) if noisy_cat.shape[1] > 0 else torch.empty(batch_size, 0, device=device)
    
    # 编码三个通道
    e_noisy, _ = feature_embedder(noisy_num, noisy_cat, ones_mask_num, ones_mask_cat)
    e_cond, _ = feature_embedder(cond_num, cond_cat, ones_mask_num, ones_mask_cat)
    
    # 编码掩码到嵌入维度
    emb_dim = e_noisy.shape[1]
    if train_batch_mask.shape[1] == expected_orig_dim:
        feature_emb_dim = emb_dim // expected_orig_dim
        e_mask = cond_mask_orig_dim.unsqueeze(2).expand(-1, -1, feature_emb_dim).reshape(batch_size, -1).float()
    elif train_batch_mask.shape[1] == expected_onehot_dim:
        # train_batch_mask已经是one-hot维度的掩码
        # 需要将其转换回原始维度
        original_error_mask = torch.zeros((batch_size, expected_orig_dim), device=device)
        if d_numerical > 0:
            original_error_mask[:, :d_numerical] = train_batch_mask[:, :d_numerical]
        if d_categorical > 0:
            # 对于分类特征，取每个分类特征的第一个one-hot位置作为代表
            cat_starts = torch.cumsum(torch.tensor([d_numerical] + actual_cat_sizes[:-1]), dim=0)
            original_error_mask[:, d_numerical:d_numerical + d_categorical] = train_batch_mask[:, cat_starts]
    else:
        # 使用one-hot维度的掩码
        feature_emb_dim = emb_dim // expected_onehot_dim
        e_mask = cond_mask_onehot.unsqueeze(2).expand(-1, -1, feature_emb_dim).reshape(batch_size, -1).float()
    
    # 构建两通道输入：condition + noise
    emb_dim = e_noisy.shape[1]
    feature_emb_dim = emb_dim // expected_orig_dim  # 每个特征的嵌入维度
    e_mask = original_error_mask.unsqueeze(2).expand(-1, -1, feature_emb_dim).reshape(batch_size, -1).float()
    
    # 通道一：条件数据 - 只包含已知部分的干净数据
    e_condition = e_cond
    
    # 通道二：噪声数据 - 只包含需要预测位置的噪声
    e_noise = e_noisy * (1 - e_mask)
    
    model_input = torch.cat([e_condition, e_noise], dim=1)
    
    
    # 6. 模型预测
    pred_x0_num, pred_x0_cat_logits = repair_model(model_input, t, y_labels=None)
    
    # 7. 计算损失
    # 数值特征使用MSE损失
    loss_num = torch.tensor(0.0, device=device)
    if d_numerical > 0:
        # 只在缺失位置计算损失
        num_loss_mask = (1 - cond_mask_onehot[:, :d_numerical])
        loss_num = F.mse_loss(pred_x0_num * num_loss_mask, x_num * num_loss_mask)
    
    # 分类特征使用交叉熵损失
    loss_cat = torch.tensor(0.0, device=device)
    if d_categorical > 0 and pred_x0_cat_logits:
        cat_feature_count = 0
        
        if isinstance(pred_x0_cat_logits, list):
            # 列表格式：每个元素对应一个分类特征
            for i, logits_i in enumerate(pred_x0_cat_logits):
                if i < x_cat.shape[1]:
                    labels_i = x_cat[:, i]
                    # 获取该特征的掩码
                    feature_mask = (1 - original_error_mask[:, d_numerical + i])
                    valid_indices = feature_mask > 0
                    
                    if valid_indices.sum() > 0:
                        loss_cat += F.cross_entropy(
                            logits_i[valid_indices],
                            labels_i[valid_indices],
                            reduction='mean'
                        )
                        cat_feature_count += 1
        else:
            # 张量格式：需要按cat_sizes切分
            start_idx = 0
            for i, n_cats in enumerate(actual_cat_sizes):
                if n_cats > 0 and i < x_cat.shape[1]:
                    logits_i = pred_x0_cat_logits[:, start_idx:start_idx + n_cats]
                    labels_i = x_cat[:, i]
                    
                    # 获取该特征的掩码
                    feature_mask = (1 - original_error_mask[:, d_numerical + i])
                    valid_indices = feature_mask > 0
                    
                    if valid_indices.sum() > 0:
                        loss_cat += F.cross_entropy(
                            logits_i[valid_indices],
                            labels_i[valid_indices],
                            reduction='mean'
                        )
                        cat_feature_count += 1
                    start_idx += n_cats
        
        if cat_feature_count > 0:
            loss_cat /= cat_feature_count
    
    # 总损失
    lambda_cat = getattr(args, 'lambda_cat', 1.0)
    loss = loss_num + lambda_cat * loss_cat
    
    return loss
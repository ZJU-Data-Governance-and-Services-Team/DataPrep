import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from typing import Tuple, List, Dict, Optional, Union
import torch.nn as nn

def train_embedding_diffae_guided(
    model_g, model_c, feature_embedder, diffusion, optimizer_g, train_loader, args,
    processor=None, criterion_c=None, scheduler_g=None, device='cuda',
    guidance_weight=1.0, epochs=10, batch_size=32, align_step_size=0.01,
    verbose=True, log_interval=5):
    """
    训练引导式嵌入扩散自编码器
    
    Args:
        model_g: 扩散填补模型
        model_c: 分类器模型（固定参数）
        feature_embedder: 特征嵌入器
        diffusion: 扩散过程
        optimizer_g: 模型G的优化器
        train_loader: 训练数据加载器
        args: 配置参数
        processor: 数据处理器
        criterion_c: 分类任务的损失函数
        scheduler_g: 学习率调整器
        device: 计算设备
        guidance_weight: 分类器引导权重
        epochs: 训练轮数
        batch_size: 批次大小
        align_step_size: 梯度对齐步长
        verbose: 是否打印详细信息
        log_interval: 日志打印间隔
    
    Returns:
        训练后的模型
    """
    if criterion_c is None:
        criterion_c = nn.CrossEntropyLoss()
    
    # 确保分类器处于评估模式，不更新参数
    model_c.eval()
    for param in model_c.parameters():
        param.requires_grad = False
    
    # 将扩散模型设为训练模式
    model_g.train()
    
    # 获取相关数据特征信息
    if processor is not None:
        if hasattr(processor, 'cat_sizes_without_mask'):
            actual_cat_sizes = processor.cat_sizes_without_mask
        else:
            actual_cat_sizes = getattr(processor, 'cat_dims', None)
            if actual_cat_sizes is None:
                print("警告: 无法从processor获取categorical_sizes，使用空列表")
                actual_cat_sizes = []
        
        d_numerical = processor.d_numerical if hasattr(processor, 'd_numerical') else 0
        d_categorical = processor.d_categorical if hasattr(processor, 'd_categorical') else 0
    
    for epoch in range(epochs):
        running_loss = 0.0
        processed_samples = 0
        diffusion_loss_epoch = 0.0
        task_loss_epoch = 0.0
        for batch_idx, batch in enumerate(train_loader):
            # 解包批次数据 - TabularDataset格式为 (x_num, x_cat, labels, num_mask, cat_mask)
            if len(batch) == 5:
                x_num, x_cat, labels, num_mask, cat_mask = batch
                x_num = x_num.to(device)
                x_cat = x_cat.to(device)
                labels = labels.to(device)
                num_mask = num_mask.to(device)
                cat_mask = cat_mask.to(device)
                
                feature_mask = torch.cat([num_mask, cat_mask], dim=1)
                t = torch.randint(0, diffusion.num_timesteps, (x_num.shape[0],), device=device)
                
                num_noise, cat_noise, _ = diffusion.q_sample(
                    x_0=None, 
                    t=t, 
                    noise=None, 
                    x_0_num=x_num, 
                    x_0_cat_int=x_cat, 
                    actual_cat_sizes=actual_cat_sizes
                )
                
                # 获取嵌入
                with torch.no_grad():
                    e_t, _ = feature_embedder(num_noise, cat_noise)
                    
                batch_size = e_t.shape[0]
            # 预先计算的嵌入格式 (e_t, t, labels)
            elif len(batch) == 3:
                e_t, t, labels = batch
                batch_size = e_t.shape[0]
                e_t = e_t.to(device)
                t = t.to(device)
                labels = labels.to(device)
                
                # 当我们只有预计算的嵌入时，我们没有原始特征和掩码
                x_num = None
                x_cat = None 
                feature_mask = None
            else:
                print(f"警告: 无法处理的batch格式，元素数量: {len(batch)}")
                continue
            
            # 每个步骤开始时清零梯度
            optimizer_g.zero_grad()
            
            # 1. 获取G的预测
            pred_e0_num, pred_e0_cat = model_g(e_t, t, y_labels=labels)
            
            # 2. 计算扩散重建损失
            diffusion_loss, _, _ = compute_embedding_diffusion_loss(
                pred_e0_num, pred_e0_cat, x_num, x_cat, feature_mask
            )
            
            # 3. 准备分类器的输入 - 使用Gumbel-Softmax生成近似one-hot向量
            decoded_cat_onehot = []
            temperature = 0.5  # 温度参数，越低越接近硬性one-hot
            for i, logits in enumerate(pred_e0_cat):
                if logits.size(1) > 0:  # 确保logits不是空的
                    # 应用Gumbel-Softmax
                    gumbel_softmax_sample = F.gumbel_softmax(logits, tau=temperature, hard=True, dim=1)
                    decoded_cat_onehot.append(gumbel_softmax_sample)
            
            # 将所有特征连接成分类器的输入格式
            decoded_cat_onehot_tensor = torch.cat(decoded_cat_onehot, dim=1) if decoded_cat_onehot else torch.empty((batch_size, 0), device=device)
            x_c_input = torch.cat([pred_e0_num, decoded_cat_onehot_tensor], dim=1)
            
            # 4. 计算分类任务损失
            outputs_c = model_c(x_c_input)
            task_loss = criterion_c(outputs_c, labels)
            
            # 5. 简单地结合两种损失，不需要复杂的梯度处理
            total_loss = diffusion_loss + guidance_weight * task_loss
            
            # 6. 反向传播和优化
            total_loss.backward()
            optimizer_g.step()
            
            running_loss += total_loss.item()
            diffusion_loss_epoch += diffusion_loss.item()
            task_loss_epoch += task_loss.item()
            processed_samples += batch_size
            
            # if verbose and (batch_idx % log_interval == 0 or batch_idx == len(train_loader) - 1):
            #     print(f"Batch [{batch_idx}/{len(train_loader)}], "
            #           f"Loss: {total_loss.item():.4f}, "
            #           f"Diff Loss: {diffusion_loss.item():.4f}, "
            #           f"Task Loss: {task_loss.item():.4f}")
        
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Diff Loss: {diffusion_loss_epoch/len(train_loader):.4f}, Task Loss: {task_loss_epoch/len(train_loader):.4f}")
        
        if scheduler_g is not None:
            scheduler_g.step()
    
    return model_g

# 从diffae.py复制的函数，确保与扩散模型的损失计算一致
def compute_embedding_diffusion_loss(
    pred_e0_num: torch.Tensor,
    pred_e0_cat: List[torch.Tensor],
    x_num: Optional[torch.Tensor] = None,
    x_cat: Optional[torch.Tensor] = None,
    emb_mask: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    contrastive_weight: float = 0.0  # 在引导训练中不使用对比损失
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算数值和分类特征的扩散损失。
    
    Args:
        pred_e0_num: 预测的数值特征
        pred_e0_cat: 预测的分类特征列表
        x_num: 真实的数值特征
        x_cat: 真实的分类特征
        emb_mask: 特征掩码
        labels: 分类标签（可选）
        contrastive_weight: 对比损失权重
        
    Returns:
        总损失, 数值损失, 分类损失
    """
    # 确保至少有一个设备可用于计算
    device = pred_e0_num.device if pred_e0_num is not None and pred_e0_num.nelement() > 0 else \
             (pred_e0_cat[0].device if pred_e0_cat and len(pred_e0_cat) > 0 else 'cuda')
    
    # 如果没有提供真实值或掩码，则假设完全重建任务
    if x_num is None and x_cat is None:
        # 如果在训练循环中没有提供真实值，我们不能计算正常的重建损失
        # 返回一个零损失，这样模型仍然可以被引导式分类损失更新
        return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
    
    # 如果没有提供掩码，假设全部都要重建
    if emb_mask is None:
        batch_size = x_num.shape[0] if x_num is not None else x_cat.shape[0]
        num_numerical_features = x_num.shape[1] if x_num is not None else 0
        num_categorical_features = x_cat.shape[1] if x_cat is not None else 0
        emb_mask = torch.ones(batch_size, num_numerical_features + num_categorical_features, device=device)
    else:
        batch_size = emb_mask.shape[0]
        num_numerical_features = x_num.shape[1] if x_num is not None else 0
        num_categorical_features = x_cat.shape[1] if x_cat is not None else 0

    # 分离数值和分类掩码
    if num_numerical_features > 0:
        mask_num = emb_mask[:, :num_numerical_features]
    else:
        mask_num = torch.empty(batch_size, 0, dtype=emb_mask.dtype, device=device)

    if num_categorical_features > 0:
        mask_cat = emb_mask[:, num_numerical_features:]
    else:
        mask_cat = torch.empty(batch_size, 0, dtype=emb_mask.dtype, device=device)

    # 计算数值损失（掩码MSE）
    num_loss = torch.tensor(0.0, device=device)
    if num_numerical_features > 0 and x_num is not None and pred_e0_num is not None and mask_num.sum() > 0:
        squared_errors = (pred_e0_num - x_num)**2
        masked_squared_errors_sum = (squared_errors * mask_num).sum()
        num_loss = masked_squared_errors_sum / mask_num.sum().clamp(min=1e-9)  # 避免除零

    # 计算分类损失（交叉熵之和）
    cat_loss = torch.tensor(0.0, device=device)
    if num_categorical_features > 0 and x_cat is not None and pred_e0_cat and mask_cat.sum() > 0:
        num_cat_features_with_loss = 0
        for i in range(min(num_categorical_features, len(pred_e0_cat))):
            if i >= len(pred_e0_cat):
                continue
                
            pred_logits_i = pred_e0_cat[i]
            true_labels_i = x_cat[:, i]
            feature_mask_i = mask_cat[:, i] 

            valid_indices = feature_mask_i > 0
            
            if valid_indices.sum() > 0:
                # 确保true_labels是长整型
                current_ce_loss = F.cross_entropy(
                    pred_logits_i[valid_indices],
                    true_labels_i[valid_indices].long(),
                    reduction='mean'
                )
                cat_loss += current_ce_loss
                num_cat_features_with_loss += 1
        
        if num_cat_features_with_loss > 0:
           cat_loss /= num_cat_features_with_loss

    total_loss = num_loss + cat_loss

    return total_loss, num_loss, cat_loss

def train_embedding_direct_guided(
    model_g, model_c, feature_embedder, optimizer_g, train_loader, args,
    processor=None, criterion_c=None, scheduler_g=None, device='cuda',
    guidance_weight=1.0, epochs=10, batch_size=32,
    verbose=True, log_interval=5):
    """
    训练引导式直接预测模型（不使用扩散过程）
    
    Args:
        model_g: 预测模型
        model_c: 分类器模型（固定参数）
        feature_embedder: 特征嵌入器
        optimizer_g: 模型G的优化器
        train_loader: 训练数据加载器
        args: 配置参数
        processor: 数据处理器
        criterion_c: 分类任务的损失函数
        scheduler_g: 学习率调整器
        device: 计算设备
        guidance_weight: 分类器引导权重
        epochs: 训练轮数
        batch_size: 批次大小
        verbose: 是否打印详细信息
        log_interval: 日志打印间隔
    
    Returns:
        训练后的模型
    """
    if criterion_c is None:
        criterion_c = nn.CrossEntropyLoss()
    
    # 确保分类器处于评估模式，不更新参数
    model_c.eval()
    for param in model_c.parameters():
        param.requires_grad = False
    
    # 将模型设为训练模式
    model_g.train()
    
    # 获取相关数据特征信息
    if processor is not None:
        if hasattr(processor, 'cat_sizes_without_mask'):
            actual_cat_sizes = processor.cat_sizes_without_mask
        else:
            actual_cat_sizes = getattr(processor, 'cat_dims', None)
            if actual_cat_sizes is None:
                print("警告: 无法从processor获取categorical_sizes，使用空列表")
                actual_cat_sizes = []
        
        d_numerical = processor.d_numerical if hasattr(processor, 'd_numerical') else 0
        d_categorical = processor.d_categorical if hasattr(processor, 'd_categorical') else 0
    
    for epoch in range(epochs):
        running_loss = 0.0
        processed_samples = 0
        recon_loss_epoch = 0.0
        task_loss_epoch = 0.0
        for batch_idx, batch in enumerate(train_loader):
            # 解包批次数据 - TabularDataset格式为 (x_num, x_cat, labels, num_mask, cat_mask)
            if len(batch) == 5:
                x_num, x_cat, labels, num_mask, cat_mask = batch
                x_num = x_num.to(device)
                x_cat = x_cat.to(device)
                labels = labels.to(device)
                num_mask = num_mask.to(device)
                cat_mask = cat_mask.to(device)
                
                feature_mask = torch.cat([num_mask, cat_mask], dim=1)
                
                # 获取嵌入
                with torch.no_grad():
                    e_input = torch.randn(x_num.shape[0], feature_embedder.d_embed * (d_numerical + d_categorical), device=device)
                    
                batch_size = x_num.shape[0]
            else:
                print(f"警告: 无法处理的batch格式，元素数量: {len(batch)}")
                continue
            
            # 每个步骤开始时清零梯度
            optimizer_g.zero_grad()
            
            # 1. 使用t=0直接获取G的预测（无扩散）
            t = torch.zeros((batch_size,), device=device, dtype=torch.long)
            pred_x0_num, pred_x0_cat = model_g(e_input, t, y_labels=labels)
            
            # 2. 计算重建损失（与真实值比较）
            recon_loss, _, _ = compute_embedding_diffusion_loss(
                pred_x0_num, pred_x0_cat, x_num, x_cat, feature_mask
            )
            
            # 3. 准备分类器的输入 - 使用Gumbel-Softmax生成近似one-hot向量
            decoded_cat_onehot = []
            temperature = 0.5  # 温度参数，越低越接近硬性one-hot
            for i, logits in enumerate(pred_x0_cat):
                if logits.size(1) > 0:  # 确保logits不是空的
                    # 应用Gumbel-Softmax
                    gumbel_softmax_sample = F.gumbel_softmax(logits, tau=temperature, hard=True, dim=1)
                    decoded_cat_onehot.append(gumbel_softmax_sample)
            
            # 将所有特征连接成分类器的输入格式
            decoded_cat_onehot_tensor = torch.cat(decoded_cat_onehot, dim=1) if decoded_cat_onehot else torch.empty((batch_size, 0), device=device)
            x_c_input = torch.cat([pred_x0_num, decoded_cat_onehot_tensor], dim=1)
            
            # 4. 计算分类任务损失
            outputs_c = model_c(x_c_input)
            task_loss = criterion_c(outputs_c, labels)
            
            # 5. 结合两种损失
            total_loss = recon_loss + guidance_weight * task_loss
            
            # 6. 反向传播和优化
            total_loss.backward()
            optimizer_g.step()
            
            running_loss += total_loss.item()
            recon_loss_epoch += recon_loss.item()
            task_loss_epoch += task_loss.item()
            processed_samples += batch_size
        
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Recon Loss: {recon_loss_epoch/len(train_loader):.4f}, Task Loss: {task_loss_epoch/len(train_loader):.4f}")
        
        if scheduler_g is not None:
            scheduler_g.step()
    
    return model_g 
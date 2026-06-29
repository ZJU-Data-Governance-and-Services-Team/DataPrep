"""
Specialized functions for training and using diffusion models with embeddings
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Union, List, Tuple, Optional
from tqdm import tqdm
from torch.utils.data import DataLoader

from models.diffusion_models import TabularDiffAE, TabularUNet
from models.diffusion_utils import DiffusionUtils
from utils.feature_embedder import FeatureEmbedder

def compute_embedding_diffusion_loss(
    pred_e0: torch.Tensor,
    true_e0: torch.Tensor,
    emb_mask: torch.Tensor = None,
    cat_feature_dims: list = None,  # 分类特征的维度列表
    cat_sizes: list = None,  # 每个分类特征的类别数
) -> torch.Tensor:
    """
    分别计算分类特征和数值特征的扩散损失
    
    Args:
        pred_e0: 模型预测的嵌入
        true_e0: 真实的特征嵌入
        emb_mask: 特征掩码 (1=观测值, 0=缺失值)
        cat_feature_dims: 分类特征对应的维度列表
        cat_sizes: 每个分类特征的类别数列表
    """
    device = pred_e0.device
    batch_size = pred_e0.shape[0]
    
    # 初始化损失
    num_loss = torch.tensor(0.0, device=device)
    cat_loss = torch.tensor(0.0, device=device)
    
    # 1. 处理数值特征的MSE损失
    num_mask = torch.ones_like(emb_mask)
    if cat_feature_dims:
        for dim in cat_feature_dims:
            num_mask[:, dim] = 0
    num_pred = pred_e0 * num_mask
    num_true = true_e0 * num_mask
    
    if num_mask.sum() > 0:
        num_loss = torch.nn.functional.mse_loss(num_pred, num_true, reduction='sum') / num_mask.sum()
    
    # 2. 处理分类特征的交叉熵损失
    if cat_feature_dims and cat_sizes:
        for dim, n_classes in zip(cat_feature_dims, cat_sizes):
            # 提取当前分类特征的预测和真实值
            cat_pred = pred_e0[:, dim].view(batch_size, -1)
            cat_true = true_e0[:, dim].view(batch_size, -1)
            
            if emb_mask is not None:
                feature_mask = emb_mask[:, dim]
                # 只对有观测值的样本计算损失
                valid_samples = feature_mask > 0
                if valid_samples.sum() > 0:
                    cat_loss += torch.nn.functional.cross_entropy(
                        cat_pred[valid_samples], 
                        cat_true[valid_samples].long(),
                        reduction='mean'
                    )
            else:
                cat_loss += torch.nn.functional.cross_entropy(
                    cat_pred, 
                    cat_true.long(),
                    reduction='mean'
                )
    
    # 3. 合并损失 (可以添加权重系数)
    total_loss = num_loss + cat_loss
    
    return total_loss

def train_embedding_diffae(
    model: Union[TabularDiffAE, TabularUNet],
    feature_embedder: FeatureEmbedder,
    diffusion: DiffusionUtils,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epochs: int,
    d_embed: int,
    actual_cat_sizes: List[int],
    log_interval: int = 10
) -> List[float]:
    """
    Train the diffusion model using feature embeddings.
    
    Args:
        model: Diffusion model (TabularDiffAE or TabularUNet)
        feature_embedder: Feature embedder model
        diffusion: Diffusion utilities
        dataloader: DataLoader containing the training data
        optimizer: Optimizer for training
        device: Device to train on
        epochs: Number of training epochs
        d_embed: Dimension of embeddings
        actual_cat_sizes: List of category sizes for each categorical feature
        log_interval: How often to log training progress
        
    Returns:
        List of losses during training
    """
    losses = []
    diffusion.compute_cat_marginals(dataloader, actual_cat_sizes)
    
    # Track feature dimensions for categorical features
    cat_feature_dims = []
    cat_sizes = []
    for i, size in enumerate(actual_cat_sizes):
        if size > 0:
            cat_feature_dims.append(i)
            cat_sizes.append(size)
    
    for epoch in range(epochs):
        epoch_losses = []
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            # Unpack batch (x_num, x_cat, y, num_mask, cat_mask)
            x_num, x_cat, _, num_mask, cat_mask = batch
            
            # Move data to device
            x_num = x_num.to(device) if x_num is not None else None
            x_cat = x_cat.to(device) if x_cat is not None else None
            num_mask = num_mask.to(device) if num_mask is not None else None
            cat_mask = cat_mask.to(device) if cat_mask is not None else None
            
            batch_size = x_num.shape[0] if x_num is not None else x_cat.shape[0]
            
            # Create feature mask (concatenate numerical and categorical masks)
            feature_mask = torch.cat([num_mask, cat_mask], dim=1) if num_mask is not None and cat_mask is not None else None
            
            # Sample random timesteps
            t = torch.randint(0, diffusion.num_timesteps, (batch_size,), device=device)
            
            # Add noise to original data
            num_noise, cat_noise, _ = diffusion.q_sample(
                x_0=None, 
                t=t, 
                noise=None, 
                x_0_num=x_num, 
                x_0_cat_int=x_cat, 
                actual_cat_sizes=actual_cat_sizes
            )
            
            # Generate clean embeddings for original data (target)
            with torch.no_grad():
                e0, _ = feature_embedder(x_num, x_cat)
            
            # Generate embeddings for noisy data (input to diffusion model)
            e_t, _ = feature_embedder(num_noise, cat_noise)
            
            # Predict embeddings at t=0
            optimizer.zero_grad()
            pred_e0 = model(e_t, t)
            
            # Compute masked loss between predicted and true embeddings
            loss = compute_embedding_diffusion_loss(pred_e0, e0, feature_mask, cat_feature_dims, cat_sizes)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            # Record loss
            epoch_losses.append(loss.item())
            progress_bar.set_postfix(loss=loss.item())
        
        # Calculate average epoch loss
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_epoch_loss)
        
        # Log progress
        if (epoch + 1) % log_interval == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.6f}")
    
    return losses

def denoise_embedding(
    model: Union[TabularDiffAE, TabularUNet],
    feature_embedder: FeatureEmbedder,
    diffusion_utils: DiffusionUtils,
    noisy_data_num: Optional[torch.Tensor],
    noisy_data_cat: Optional[torch.Tensor],
    noise_level: int, 
    device: str,
    guidance_grad=None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Directly denoise embedded data at a specific noise level and convert back to feature space
    
    Args:
        model: Diffusion model (TabularDiffAE or TabularUNet)
        feature_embedder: Feature embedder model
        diffusion_utils: DiffusionUtils instance
        noisy_data_num: Numerical features tensor
        noisy_data_cat: Categorical features tensor
        noise_level: Integer time step (0-999) indicating noise level
        device: Torch device
        guidance_grad: Optional gradient guidance for the denoising process
        
    Returns:
        Tuple of (denoised_numerical, denoised_categorical)
    """
    model.eval()
    batch_size = noisy_data_num.shape[0] if noisy_data_num is not None else noisy_data_cat.shape[0]
    
    # Create time tensor with the specified noise level
    t = torch.full((batch_size,), noise_level, device=device, dtype=torch.long)
    
    # Generate embeddings for noisy data
    with torch.no_grad():
        e_t, _ = feature_embedder(noisy_data_num, noisy_data_cat)
        
        # Forward pass through diffusion model
        pred_e0 = model(e_t, t, guidance_grad)
        
        # Decode predicted embeddings back to feature space
        denoised_num, denoised_cat = feature_embedder.decode_embeddings(pred_e0)
    
    return denoised_num, denoised_cat

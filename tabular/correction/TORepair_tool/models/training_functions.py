import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Union, List, Tuple, Dict
from tqdm import tqdm
from torch.utils.data import DataLoader

from models.diffusion_models import TabularDiffAE, TabularUNet
from models.diffusion_utils import DiffusionUtils
from utils.feature_embedder import FeatureEmbedder


def compute_diffae_loss(pred_x0_num, pred_x0_cat_flat_logits, 
                       true_x0_num, true_x0_cat_int,
                       actual_cat_sizes, device, lambda_cat=1.0):
    """
    Compute combined loss for the predictions vs. true values
    """
    batch_size = true_x0_num.shape[0] if true_x0_num is not None else true_x0_cat_int.shape[0]
    loss = torch.tensor(0.0, device=device)
    
    # Numerical features loss (MSE)
    if true_x0_num is not None and pred_x0_num is not None:
        num_loss = F.mse_loss(pred_x0_num, true_x0_num, reduction='sum') / batch_size
        loss = loss + num_loss
    
    # Categorical features loss (Cross-Entropy)
    if true_x0_cat_int is not None and pred_x0_cat_flat_logits is not None:
        cat_start = 0
        cat_loss = torch.tensor(0.0, device=device)
        cat_feature_count = 0
        
        for i, n_cats in enumerate(actual_cat_sizes):
            if n_cats > 0:  # Skip features with no categories
                # Get logits for this feature
                feat_logits = pred_x0_cat_flat_logits[:, cat_start:cat_start + n_cats]
                # Get true category indices for this feature
                feat_true = true_x0_cat_int[:, cat_feature_count].long()
                # Compute cross-entropy loss
                cat_loss = cat_loss + F.cross_entropy(feat_logits, feat_true, reduction='sum')
                cat_start += n_cats
                cat_feature_count += 1
        
        # Normalize and add to total loss with weight
        if cat_feature_count > 0:
            cat_loss = (cat_loss / (batch_size * cat_feature_count)) * lambda_cat
            loss = loss + cat_loss
    
    return loss


def train_diffae(model, diffusion, dataloader, optimizer, device, epochs, 
                actual_cat_sizes, lambda_cat=1.0, log_interval=10):
    """
    Train the DiffAE model
    """
    losses = []
    diffusion.compute_cat_marginals(dataloader, actual_cat_sizes)
    
    for epoch in range(epochs):
        epoch_losses = []
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            # Unpack batch (x_num, x_cat, y, num_mask, cat_mask)
            x_num, x_cat, _, num_mask, cat_mask = batch
            
            # Move data to device
            x_num = x_num.to(device) if x_num is not None else None
            x_cat = x_cat.to(device) if x_cat is not None else None
            
            batch_size = x_num.shape[0] if x_num is not None else x_cat.shape[0]
            
            # Sample random timesteps
            t = torch.randint(0, diffusion.num_timesteps, (batch_size,), device=device)
            
            # Add noise to the original data
            noised_num, noised_cat_int, combined_noise = diffusion.q_sample(
                x_0=None, 
                t=t, 
                noise=None, 
                x_0_num=x_num, 
                x_0_cat_int=x_cat, 
                actual_cat_sizes=actual_cat_sizes
            )
            
            # Convert categorical indices to one-hot
            cat_one_hot = []
            if noised_cat_int is not None:
                cat_idx = 0
                for i, n_cats in enumerate(actual_cat_sizes):
                    if n_cats > 0:  # Skip features with no categories
                        one_hot = F.one_hot(noised_cat_int[:, cat_idx].long(), n_cats).float()
                        cat_one_hot.append(one_hot)
                        cat_idx += 1
                    
            noised_cat_onehot = torch.cat(cat_one_hot, dim=1) if cat_one_hot else None
            
            # Combine features for model input
            model_input = torch.cat([noised_num, noised_cat_onehot], dim=1) if noised_cat_onehot is not None else noised_num
            
            # Forward pass
            optimizer.zero_grad()
            pred_x0_num, pred_x0_cat_flat_logits = model(model_input, t)
            
            # Compute loss
            loss = compute_diffae_loss(
                pred_x0_num, pred_x0_cat_flat_logits, 
                x_num, x_cat, 
                actual_cat_sizes, device, lambda_cat
            )
            
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


def compute_embedding_diffusion_loss(
    pred_e0: torch.Tensor,
    true_e0: torch.Tensor,
    emb_mask: torch.Tensor = None,
    cat_feature_dims: list = None,
    cat_sizes: list = None,
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
    
    for epoch in range(epochs):
        epoch_losses = []
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            # Unpack batch (x_num, x_cat, y, num_mask, cat_mask)
            x_num, x_cat, _, num_mask, cat_mask = batch
            
            # Move data to device
            x_num = x_num.to(device) if x_num is not None else None
            x_cat = x_cat.to(device) if x_cat is not None else None
            num_mask = num_mask.to(device)
            cat_mask = cat_mask.to(device)
            
            batch_size = x_num.shape[0] if x_num is not None else x_cat.shape[0]
            
            # Create feature mask (concatenate numerical and categorical masks)
            feature_mask = torch.cat([num_mask, cat_mask], dim=1)
            t = torch.randint(0, diffusion.num_timesteps, (batch_size,), device=device)
            
            # Add noise to original data
            num_noise, cat_noise, combine_noise = diffusion.q_sample(
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
            
            # Generate embeddings for noisy data
            e_t, _ = feature_embedder(num_noise, cat_noise)
            
            # Predict noise or x0 (depending on the model)
            optimizer.zero_grad()
            pred_e0 = model(e_t, t)
            
            # Get the dimensionality information for categorical features
            cat_feature_dims = []
            cat_sizes = []
            for i, size in enumerate(actual_cat_sizes):
                if size > 0:
                    # In embedding space, each feature has d_embed dimensions
                    # So we need to find the indices of the categorical features
                    cat_feature_dims.append(len(cat_feature_dims) + i)
                    cat_sizes.append(size)
            
            # Compute masked loss
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
    noisy_data_num,
    noisy_data_cat,
    noise_level: int, 
    device: str,
    guidance_grad=None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Directly denoise embedded data at a specific noise level
    
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


def test_diffae(data_config, model, diffusion_utils, test_df, processor, device, d_numerical, 
                actual_cat_sizes, batch_size=32, t_eval=10, verbose=False, show_modif_info=True, guidance_grad=None):
    """
    Test the DiffAE model on test data
    """
    model.eval()
    
    # Set a specific noise level for evaluation (t=0 for clean data, t=1000 for pure noise)
    noise_level = min(t_eval, diffusion_utils.num_timesteps - 1)
    
    # Process test data
    processed_data = processor.transform(test_df)
    
    # Extract numerical and categorical features
    x_num = processed_data['x_num']
    x_cat = processed_data['x_cat']
    
    # Initialize lists for processing in batches
    all_denoised_num = []
    all_denoised_cat = []
    
    # Process data in batches
    n_samples = len(test_df)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    if verbose:
        print(f"Testing on {n_samples} samples using noise level t={noise_level}")
        print(f"Processing in {n_batches} batches...")
    
    for batch_idx in range(n_batches):
        # Extract batch
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_samples)
        batch_size_actual = end_idx - start_idx
        
        # Create batch tensors
        batch_x_num = torch.tensor(x_num[start_idx:end_idx], dtype=torch.float32, device=device) if x_num is not None else None
        batch_x_cat = torch.tensor(x_cat[start_idx:end_idx], dtype=torch.long, device=device) if x_cat is not None else None
        
        # Add noise to data
        noised_num, noised_cat_int, combined_noise = diffusion_utils.q_sample(
            x_0=None, 
            t=torch.full((batch_size_actual,), noise_level, device=device), 
            noise=None, 
            x_0_num=batch_x_num, 
            x_0_cat_int=batch_x_cat, 
            actual_cat_sizes=actual_cat_sizes
        )
        
        # Convert categorical indices to one-hot
        cat_one_hot = []
        if noised_cat_int is not None:
            cat_idx = 0
            for i, n_cats in enumerate(actual_cat_sizes):
                if n_cats > 0:  # Skip features with no categories
                    one_hot = F.one_hot(noised_cat_int[:, cat_idx].long(), n_cats).float()
                    cat_one_hot.append(one_hot)
                    cat_idx += 1
        
        noised_cat_onehot = torch.cat(cat_one_hot, dim=1) if cat_one_hot else None
        
        # Combine features for model input
        model_input = torch.cat([noised_num, noised_cat_onehot], dim=1) if noised_cat_onehot is not None else noised_num
        
        # Denoise
        with torch.no_grad():
            # Model timestep tensor
            t = torch.full((batch_size_actual,), noise_level, device=device, dtype=torch.long)
            
            # Forward pass through model
            pred_x0_num, pred_x0_cat_flat_logits = model(model_input, t, guidance_grad)
            
            # Process categorical predictions
            cat_indices = []
            if pred_x0_cat_flat_logits is not None:
                cat_start = 0
                for n_cats in actual_cat_sizes:
                    if n_cats > 0:
                        cat_logits = pred_x0_cat_flat_logits[:, cat_start:cat_start + n_cats]
                        cat_probs = F.softmax(cat_logits, dim=1)
                        cat_idx = torch.argmax(cat_probs, dim=1)
                        cat_indices.append(cat_idx.cpu().numpy())
                        cat_start += n_cats
            
            # Convert predictions to numpy
            denoised_num = pred_x0_num.cpu().numpy() if pred_x0_num is not None else None
            denoised_cat = np.column_stack(cat_indices) if cat_indices else None
            
        # Append to results
        if denoised_num is not None:
            all_denoised_num.append(denoised_num)
        if denoised_cat is not None:
            all_denoised_cat.append(denoised_cat)
    
    # Combine batch results
    final_denoised_num = np.concatenate(all_denoised_num, axis=0) if all_denoised_num else None
    final_denoised_cat = np.concatenate(all_denoised_cat, axis=0) if all_denoised_cat else None
    
    # Convert back to DataFrame
    denoised_df = processor.inverse_transform(final_denoised_num, final_denoised_cat)
    
    # Analyze results
    original_df = test_df.copy().reset_index(drop=True)
    
    # Calculate the percentage of data that was modified after denoising for each column
    modification_stats = {}
    for col in original_df.columns:
        if col in data_config.get("categorical_features", []):
            if col in denoised_df.columns:
                # Count the number of modified values
                modified_count = (original_df[col] != denoised_df[col]).sum()
                # Calculate the percentage
                modification_percentage = (modified_count / len(original_df)) * 100
            modification_stats[col] = modification_percentage
        elif col in data_config.get("numerical_features", []):
            if col in denoised_df.columns:
                # Count the number of modified values
                modified_count = (abs(original_df[col] - denoised_df[col])<0.001).sum()
                # Calculate the percentage
                modification_percentage = (modified_count / len(original_df)) * 100
            modification_stats[col] = modification_percentage
    
    # Print the modification statistics
    if show_modif_info:
        print("\n--- Modification Statistics ---")
        print("Percentage of data modified after denoising for each column:")
        print(f"Length of denoised data: {len(denoised_df)}")
        for col, percentage in modification_stats.items():
            print(f"  {col}: {percentage:.2f}%")
    
    if verbose:
        print("Denoising complete.")
        
        # Print some examples
        n_examples = min(5, len(denoised_df))
        print(f"\nShowing {n_examples} examples:")
        
        for i in range(n_examples):
            print(f"\nExample {i+1}:")
            print("Original:")
            for col in original_df.columns:
                print(f"  {col}: {original_df.iloc[i][col]}")
            print("Denoised:")
            for col in denoised_df.columns:
                print(f"  {col}: {denoised_df.iloc[i][col]}")
    
    return denoised_df, original_df


def denoise_data(model, diffusion_utils, noisy_data, noise_level, d_numerical, actual_cat_sizes, device, guidance_grad=None):
    """
    Directly denoise data at a specific noise level without full sampling
    
    Args:
        model: TabularDiffAE model
        diffusion_utils: DiffusionUtils instance
        noisy_data: Combined tensor of numerical and one-hot categorical features
        noise_level: Integer time step (0-999) indicating noise level
        d_numerical: Number of numerical features
        actual_cat_sizes: List of actual category sizes
        device: Torch device
        guidance_grad: Optional gradient guidance for the denoising process
        
    Returns:
        Tuple of (denoised_numerical, denoised_categorical_int)
    """
    model.eval()
    batch_size = noisy_data.shape[0]
    
    # Create time tensor with the specified noise level
    t = torch.full((batch_size,), noise_level, device=device, dtype=torch.long)
    
    # Forward pass through model
    with torch.no_grad():
        pred_x0_num, pred_x0_cat_flat_logits = model(noisy_data, t, guidance_grad)
    
    # Process categorical predictions
    cat_indices = []
    cat_start = 0
    for n_cats in actual_cat_sizes:
        if n_cats > 0:
            cat_logits = pred_x0_cat_flat_logits[:, cat_start:cat_start + n_cats]
            # Apply softmax and get most likely class
            cat_probs = F.softmax(cat_logits, dim=1)
            cat_idx = torch.argmax(cat_probs, dim=1)
            cat_indices.append(cat_idx)
            cat_start += n_cats
    
    denoised_cat_int = torch.stack(cat_indices, dim=1) if cat_indices else None
    
    return pred_x0_num, denoised_cat_int

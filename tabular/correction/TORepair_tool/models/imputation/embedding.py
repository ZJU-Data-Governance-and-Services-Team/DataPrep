import torch
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Union
from torch.utils.data import DataLoader, TensorDataset
from models.diffae.core import DiffusionUtils, TabularAE, TabularUNet
from datasets.data_processor import DataProcessor
from utils.feature_embedder import FeatureEmbedder
from tqdm import tqdm


class EmbeddingImputation:
    """
    Embedding-based imputation method using a diffusion model.
    
    This class implements a conditional imputation approach using a trained diffusion model
    operating on feature embeddings to fill in missing values in tabular data.
    """
    
    def __init__(
        self,
        model: Union[TabularAE, TabularUNet],
        feature_embedder: FeatureEmbedder,
        diffusion_utils: DiffusionUtils,
        processor: DataProcessor,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        masked_label_value: int = 0  # 掩码标签的值
    ):
        """
        Initialize the EmbeddingImputation class.
        
        Args:
            model: Trained diffusion model (TabularDiffAE or TabularUNet)
            feature_embedder: Trained FeatureEmbedder for embedding numerical and categorical features
            diffusion_utils: DiffusionUtils instance for handling diffusion process
            processor: DataProcessor instance for data transformation
            device: Device to run the model on ("cuda" or "cpu")
            masked_label_value: Value to use for masked labels
        """
        self.model = model
        self.feature_embedder = feature_embedder
        self.diffusion_utils = diffusion_utils
        self.processor = processor
        self.device = device
        self.masked_label_value = masked_label_value
        
        # Set models to evaluation mode
        self.model.eval()
        self.feature_embedder.eval()
        
        # Get dimensions and category sizes
        self.d_numerical = processor.d_numerical
        self.d_categorical = processor.d_categorical
        self.actual_cat_sizes = processor.categories
        self.d_embed = feature_embedder.d_embed
        
    def _prepare_initial_data(
        self, 
        df: pd.DataFrame,
        dataset_name: str = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, pd.DataFrame]:
        """
        Prepare initial data for imputation by transforming data with special handling for missing values.
        
        Args:
            df: Input dataframe with missing values
            dataset_name: Dataset name for caching optimization
            
        Returns:
            x_num: Tensor of numerical features with temporary fill values
            x_cat: Tensor of categorical features with MASK tokens for missing values
            num_mask: Mask for numerical features (1=observed, 0=missing)
            cat_mask: Mask for categorical features (1=observed, 0=missing)
            df_temp: Temporary dataframe with filled values (for reference)
        """
        # Create a copy of the dataframe
        df_temp = df.copy()
        
        # Transform the data including handling of missing values
        # The processor now returns masks as well
        x_num, x_cat, labels, num_mask, cat_mask = self.processor.transform(df_temp, dataset_name=dataset_name)
        
        # Move data to device
        if x_num is not None:
            x_num = x_num.to(self.device)
        if x_cat is not None:
            x_cat = x_cat.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        num_mask = num_mask.to(self.device)
        cat_mask = cat_mask.to(self.device)
        
        return x_num, x_cat, labels, num_mask, cat_mask, df_temp
    
    def _conditional_reverse_diffusion(
        self,
        x_num: torch.Tensor,
        x_cat: torch.Tensor,
        num_mask: torch.Tensor,
        cat_mask: torch.Tensor,
        y_labels: torch.Tensor = None,  # 标签条件
        num_steps: int = 100,
        verbose: bool = False
    ) -> torch.Tensor:
        """
        Perform conditional reverse diffusion to generate imputed embeddings.
        
        Args:
            x_num: Numerical features tensor with temporary fill values
            x_cat: Categorical features tensor with MASK tokens
            num_mask: Mask for numerical features (1=observed, 0=missing)
            cat_mask: Mask for categorical features (1=observed, 0=missing)
            y_labels: Label tensor for conditioning (optional)
            num_steps: Number of diffusion steps
            verbose: Whether to show progress bar
            
        Returns:
            Final predicted embeddings
        """
        # Get initial embeddings for known parts
        with torch.no_grad():
            e0_filled, _ = self.feature_embedder(x_num, x_cat, num_mask, cat_mask)
        
        # Get feature-level mask (1=observed, 0=missing) for original features
        # This mask will be used to combine known and predicted parts at the end of each step
        feature_mask_original = torch.cat([num_mask, cat_mask], dim=1)
        
        # Expand the original feature mask to embedding dimensions for final combination
        # XXX: need expanding or not?
        emb_mask = self.feature_embedder.expand_emb_mask(feature_mask_original)
        
        timesteps = torch.linspace(
            0, self.diffusion_utils.num_timesteps - 1, 
            num_steps, dtype=torch.int64, device=self.device
        )
        
        if x_num is not None:
            batch_size = x_num.shape[0]
        elif x_cat is not None:
            batch_size = x_cat.shape[0]
        else:
            raise ValueError("No data to impute")

        # Calculate total embedding dimension if needed (e.g. for initial noise shape)
        # This assumes d_embed is uniform across features for simplicity here.
        # If feature_embedder produces varied embedding sizes per feature that are then concatenated,
        # total_embedding_dim should be feature_embedder.total_embedding_dim or similar.
        # For now, using the same calculation as before.
        total_embedding_dim = (self.d_numerical + self.d_categorical) * self.d_embed
        if e0_filled.nelement() > 0 : # If e0_filled is not empty
            total_embedding_dim = e0_filled.shape[1]

        e_t = torch.randn((batch_size, total_embedding_dim), device=self.device)
        
        time_range = range(len(timesteps) - 1, -1, -1)
        # if verbose:
        #     time_range = tqdm(time_range, desc="Conditional Reverse Diffusion", ncols=100)
            
        for i in time_range:
            t = torch.full((batch_size,), timesteps[i], device=self.device, dtype=torch.long)
            t_idx = t.long()

            # 1. Model predicts clean ORIGINAL features (x0_num, x0_cat_logits/list) from noisy embedding e_t
            with torch.no_grad():
                # 将标签条件传递给模型
                pred_x0_num_from_model, pred_x0_cat_output = self.model(e_t, t, y_labels=y_labels)
            # 2. Convert predicted categorical logits to indices
            cat_indices_list = []
            if self.d_categorical > 0:
                if isinstance(pred_x0_cat_output, list): # e.g., TabularDiffAE with cat_reconstructor
                    for logits_i in pred_x0_cat_output:
                        if logits_i.nelement() > 0:
                             cat_indices_list.append(torch.argmax(logits_i, dim=1))
                elif pred_x0_cat_output is not None and pred_x0_cat_output.nelement() > 0: # e.g., TabularUNet (flat logits)
                    current_pos = 0
                    for cat_size in self.actual_cat_sizes: # self.actual_cat_sizes from processor
                        if cat_size > 0:
                            logits_i = pred_x0_cat_output[:, current_pos : current_pos + cat_size]
                            cat_indices_list.append(torch.argmax(logits_i, dim=1))
                            current_pos += cat_size
            
            pred_x0_cat_indices_from_model = torch.stack(cat_indices_list, dim=1) if cat_indices_list else \
                                             torch.empty(batch_size, 0, dtype=torch.long, device=self.device)

            # Handle numerical predictions (ensure it's a tensor, even if empty)
            if pred_x0_num_from_model is None or pred_x0_num_from_model.nelement() == 0:
                pred_x0_num_from_model = torch.empty(batch_size, 0, device=self.device)

            # 3. Re-embed the predicted clean original features to get e0_pred_from_model
            # Create all-ones masks for re-embedding, as these are model's best guess of clean data
            ones_mask_num = torch.ones_like(pred_x0_num_from_model, device=self.device)
            ones_mask_cat = torch.ones_like(pred_x0_cat_indices_from_model, dtype=torch.float32, device=self.device) # Mask should be float

            with torch.no_grad():
                e0_pred_from_model, _ = self.feature_embedder(
                    pred_x0_num_from_model,
                    pred_x0_cat_indices_from_model,
                    ones_mask_num,
                    ones_mask_cat
                )
            
            # 4. Perform diffusion step in embedding space
            # Get diffusion coefficients for current timestep t
            posterior_mean_coef1 = self.diffusion_utils.posterior_mean_coef1.to(self.device)[t_idx].view(-1, 1)
            posterior_mean_coef2 = self.diffusion_utils.posterior_mean_coef2.to(self.device)[t_idx].view(-1, 1)
            posterior_log_variance = self.diffusion_utils.posterior_log_variance_clipped.to(self.device)[t_idx].view(-1, 1)
            
            # Calculate posterior mean for predicted part (using e0_pred_from_model)
            posterior_mean_pred = posterior_mean_coef1 * e0_pred_from_model + posterior_mean_coef2 * e_t
            
            # Calculate posterior mean for known part (using e0_filled)
            # Ensure e0_filled is correctly broadcastable with posterior_mean_coef1
            posterior_mean_known = posterior_mean_coef1 * e0_filled + posterior_mean_coef2 * e_t
            
            # Sample noise
            noise = torch.randn_like(e_t) if t[0] > 0 else torch.zeros_like(e_t) # noise for t=0 should be zero
            
            # Sample e_{t-1} for predicted parts
            e_t_minus_1_pred = posterior_mean_pred + torch.exp(0.5 * posterior_log_variance) * noise
            
            # Sample e_{t-1} for known parts (using the same noise for consistency)
            e_t_minus_1_known = posterior_mean_known + torch.exp(0.5 * posterior_log_variance) * noise
            
            # 5. Combine known and predicted parts using the embedding mask
            e_t = e_t_minus_1_known * emb_mask + e_t_minus_1_pred * (1 - emb_mask)
            # Note: emb_mask must correctly correspond to features in the embedding space.
            # If e0_filled was created with original num_mask, cat_mask, and then expanded by feature_embedder, this should align.

        # Return the final imputed embeddings (which is e_t at t=0)
        return e_t
    
    def _decode_embeddings(
        self,
        embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode embeddings back to numerical and categorical features.
        
        Args:
            embeddings: Final predicted embeddings
            
        Returns:
            Numerical and categorical features tensors
        """
        with torch.no_grad():
            return self.feature_embedder.decode_embeddings(embeddings)
    
    def _impute_batch(
        self,
        df_batch: pd.DataFrame,
        y_labels: torch.Tensor = None,  # 标签条件
        num_steps: int = 100,
        verbose: bool = False,
        dataset_name: str = None
    ) -> pd.DataFrame:
        """
        Impute missing values for a batch of data.
        
        Args:
            df_batch: Dataframe batch with missing values
            y_labels: Label tensor for conditioning (optional)
            num_steps: Number of diffusion steps
            verbose: Whether to show progress bar
            dataset_name: Dataset name for caching optimization
            
        Returns:
            Dataframe with imputed values
        """
        x_num, x_cat, labels, num_mask, cat_mask, _ = self._prepare_initial_data(df_batch, dataset_name=dataset_name)
        
        # 如果没有提供外部标签，使用处理后的内部标签
        if y_labels is None and labels is not None:
            # 这里可以选择使用实际标签或者掩码标签
            # 对于训练数据，我们可能想用实际标签；对于测试数据，使用掩码标签
            y_cond = labels
        else:
            y_cond = y_labels
            
        # 如果掩码所有标签，且掩码值存在
        if y_cond is not None and hasattr(self, 'masked_label_value') and self.masked_label_value is not None:
            # 全部替换为掩码值
            y_cond = torch.full_like(y_cond, self.masked_label_value)
        
        embeddings = self._conditional_reverse_diffusion(
            x_num, x_cat, num_mask, cat_mask, y_cond, num_steps, verbose
        )
        
        x_num_imputed, x_cat_imputed = self.model._reconstruct_output(embeddings)
        
        imputed_df = self.processor.inverse_transform(x_num_imputed, x_cat_imputed)
        
        # Copy original values for observed features
        result_df = df_batch.copy()
        for i, col in enumerate(self.processor.num_features):
            # Only replace missing values
            mask = result_df[col].isna()
            result_df.loc[mask, col] = imputed_df.loc[mask, col]
            
        for i, col in enumerate(self.processor.cat_features):
            # Only replace missing values
            mask = result_df[col].isna()
            result_df.loc[mask, col] = imputed_df.loc[mask, col]
            
        return result_df
    
    def impute(
        self,
        df: pd.DataFrame,
        batch_size: int = 32,
        num_steps: int = 100,
        verbose: bool = False,
        use_masked_labels: bool = True,  # 是否使用掩码标签
        external_labels: torch.Tensor = None,  # 外部提供的标签
        dataset_name: str = None
    ) -> pd.DataFrame:
        """
        Impute missing values for the entire dataframe.
        
        Args:
            df: Dataframe with missing values
            batch_size: Batch size for processing
            num_steps: Number of diffusion steps
            verbose: Whether to show progress
            use_masked_labels: If True, use masked label value for all samples
            external_labels: Optional external labels to use instead of those in the dataframe
            dataset_name: Dataset name for caching optimization
            
        Returns:
            Dataframe with imputed values
        """
        if len(df) == 0:
            raise ValueError("Empty dataframe")
        
        result_dfs = []
        
        for i in tqdm(range(0, len(df), batch_size), desc="Processing batch", total=(len(df) + batch_size - 1) // batch_size, ncols=80):
                
            df_batch = df.iloc[i:i+batch_size].reset_index(drop=True)
            
            # 准备外部标签（如果有）
            if external_labels is not None:
                batch_labels = external_labels[i:i+batch_size].to(self.device)
            else:
                batch_labels = None
                
            # 如果使用掩码标签，将所有标签设为掩码值
            if use_masked_labels and hasattr(self.model, 'masked_label_value') and self.model.masked_label_value is not None:
                # 这将覆盖batch_labels
                batch_size = len(df_batch)
                batch_labels = torch.full((batch_size,), self.model.masked_label_value, device=self.device)
            
            imputed_batch = self._impute_batch(df_batch, batch_labels, num_steps, verbose, dataset_name=dataset_name)
            result_dfs.append(imputed_batch)
            
        result_df = pd.concat(result_dfs, ignore_index=True)
        
        return result_df


def rep_embeds_diffusion(
    df: pd.DataFrame,
    model: Union[TabularAE, TabularUNet],
    feature_embedder: FeatureEmbedder,
    diffusion_utils: DiffusionUtils,
    processor: DataProcessor,
    num_steps: int = 100,
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = False,
    use_masked_labels: bool = True,  # 是否使用掩码标签
    external_labels: torch.Tensor = None,  # 外部提供的标签
    masked_label_value: int = 0,  # 掩码标签的值
    dataset_name: str = None
) -> pd.DataFrame:
    """
    Convenience function to impute missing values in a dataframe using the embedding approach.
    
    Args:
        df: Input dataframe with missing values
        model: Trained diffusion model (TabularDiffAE or TabularUNet)
        feature_embedder: Trained FeatureEmbedder
        diffusion_utils: DiffusionUtils instance for handling diffusion process
        processor: DataProcessor instance for data transformation
        num_steps: Number of diffusion steps to use for reverse process
        batch_size: Batch size for processing
        device: Device to run the model on ("cuda" or "cpu")
        verbose: Whether to print progress
        use_masked_labels: If True, use masked label value for all samples
        external_labels: Optional external labels to use instead of those in the dataframe
        masked_label_value: Value to use for masked labels
        dataset_name: Dataset name for caching optimization
        
    Returns:
        Dataframe with imputed values
    """
    imputer = EmbeddingImputation(
        model=model,
        feature_embedder=feature_embedder,
        diffusion_utils=diffusion_utils,
        processor=processor,
        device=device,
        masked_label_value=masked_label_value
    )
    
    return imputer.impute(
        df=df,
        batch_size=batch_size,
        num_steps=num_steps,
        verbose=verbose,
        use_masked_labels=use_masked_labels,
        external_labels=external_labels,
        dataset_name=dataset_name
    )

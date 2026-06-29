import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Dict, Union

class FeatureEmbedder(nn.Module):
    """
    Feature Embedder for tabular data.
    
    This module embeds both numerical and categorical features into a shared embedding space
    of dimension d_embed. Numerical features are projected using a linear layer, while 
    categorical features use embedding layers. Missing values are handled with special 
    embeddings or representations.
    """
    
    def __init__(
        self,
        d_numerical: int,
        d_categorical: int,
        actual_cat_sizes: List[int],
        d_embed: int = 64,
        dropout: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the FeatureEmbedder.
        
        Args:
            d_numerical: Number of numerical features
            d_categorical: Number of categorical features
            actual_cat_sizes: List of category counts for each categorical feature
            d_embed: Dimension of the feature embeddings
            dropout: Dropout rate
            device: Device to use for tensors
        """
        super().__init__()
        self.d_numerical = d_numerical
        self.d_categorical = d_categorical
        self.actual_cat_sizes = actual_cat_sizes
        self.d_embed = d_embed
        self.device = device
        
        # Create numerical feature embedders (one per feature)
        self.num_embedders = nn.ModuleList()
        if d_numerical > 0:
            for _ in range(d_numerical):
                # Linear layer to project each numerical feature to d_embed dimensions
                self.num_embedders.append(nn.Sequential(
                    nn.Linear(1, d_embed),
                    nn.LayerNorm(d_embed),
                    nn.SiLU(),
                    nn.Dropout(dropout)
                ))
        
        # Create categorical feature embedders (one per feature)
        self.cat_embedders = nn.ModuleList()
        if d_categorical > 0:
            for cat_size in actual_cat_sizes:
                # Each category (including MASK) gets its own embedding
                self.cat_embedders.append(nn.Embedding(cat_size, d_embed))
                
        # Linear layers for decoding
        self.num_decoders = nn.ModuleList()
        if d_numerical > 0:
            for _ in range(d_numerical):
                self.num_decoders.append(nn.Linear(d_embed, 1))
                
        # Print model structure summary
        print(f"Initialized FeatureEmbedder: d_numerical={d_numerical}, "
              f"d_categorical={d_categorical}, d_embed={d_embed}")
    
    def forward(
        self,
        x_num: Optional[torch.Tensor] = None,
        x_cat: Optional[torch.Tensor] = None,
        num_mask: Optional[torch.Tensor] = None,
        cat_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the embedding layer.
        
        Args:
            x_num: Numerical features tensor [batch_size, d_numerical]
            x_cat: Categorical features tensor [batch_size, d_categorical]
            num_mask: Mask for numerical features (1=observed, 0=missing) [batch_size, d_numerical]
            cat_mask: Mask for categorical features (1=observed, 0=missing) [batch_size, d_categorical]
            
        Returns:
            e0: Combined embeddings tensor [batch_size, (d_numerical + d_categorical) * d_embed]
            emb_mask: Mask for embeddings (1=observed, 0=missing) [batch_size, (d_numerical + d_categorical)]
        """
        batch_size = x_num.shape[0] if x_num is not None else x_cat.shape[0]
        
        # Initialize output tensors
        all_embeddings = []
        emb_masks = []
        
        # Process numerical features
        if self.d_numerical > 0 and x_num is not None:
            for i in range(self.d_numerical):
                # Reshape to [batch_size, 1] for the linear layer
                feat = x_num[:, i:i+1]
                # Embed the feature to d_embed dimensions
                feat_emb = self.num_embedders[i](feat)  # [batch_size, d_embed]
                all_embeddings.append(feat_emb)
                
                # Add the mask for this feature (expand to match batch size)
                if num_mask is not None:
                    emb_masks.append(num_mask[:, i:i+1])
                else:
                    emb_masks.append(torch.ones(batch_size, 1, device=self.device))
        
        # Process categorical features
        if self.d_categorical > 0 and x_cat is not None:
            for i in range(self.d_categorical):
                # Get embeddings for this categorical feature
                feat = x_cat[:, i]
                feat_emb = self.cat_embedders[i](feat)  # [batch_size, d_embed]
                all_embeddings.append(feat_emb)
                
                # Add the mask for this feature (expand to match batch size)
                if cat_mask is not None:
                    emb_masks.append(cat_mask[:, i:i+1])
                else:
                    emb_masks.append(torch.ones(batch_size, 1, device=self.device))
        
        # Stack all embeddings along feature dimension
        if len(all_embeddings) > 0:
            e0 = torch.cat(all_embeddings, dim=1)  # [batch_size, (d_numerical + d_categorical) * d_embed]
        else:
            e0 = torch.empty((batch_size, 0), device=self.device)
            
        # Stack all masks along feature dimension
        if len(emb_masks) > 0:
            emb_mask = torch.cat(emb_masks, dim=1)  # [batch_size, d_numerical + d_categorical]
        else:
            emb_mask = torch.empty((batch_size, 0), device=self.device)
            
        return e0, emb_mask
    
    def decode_embeddings(
        self,
        embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode embeddings back to numerical and categorical features.
        
        Args:
            embeddings: Feature embeddings tensor [batch_size, (d_numerical + d_categorical) * d_embed]
            
        Returns:
            x_num: Numerical features tensor [batch_size, d_numerical]
            x_cat: Categorical features tensor [batch_size, d_categorical]
        """
        batch_size = embeddings.shape[0]
        total_features = self.d_numerical + self.d_categorical
        
        # Reshape to [batch_size, total_features, d_embed]
        embeddings = embeddings.reshape(batch_size, total_features, self.d_embed)
        
        # Initialize output tensors
        if self.d_numerical > 0:
            x_num = torch.zeros(batch_size, self.d_numerical, device=self.device)
        else:
            x_num = torch.empty((batch_size, 0), device=self.device)
            
        if self.d_categorical > 0:
            x_cat = torch.zeros(batch_size, self.d_categorical, dtype=torch.long, device=self.device)
        else:
            x_cat = torch.empty((batch_size, 0), dtype=torch.long, device=self.device)
        
        # Decode numerical features
        num_idx = 0
        if self.d_numerical > 0:
            for i in range(self.d_numerical):
                # Get the embedding for this numerical feature
                feat_emb = embeddings[:, num_idx, :]  # [batch_size, d_embed]
                # Decode the embedding to a scalar
                x_num[:, i] = self.num_decoders[i](feat_emb).squeeze(-1)
                num_idx += 1
        
        # Decode categorical features
        cat_idx = 0
        if self.d_categorical > 0:
            for i in range(self.d_categorical):
                # Get the embedding for this categorical feature
                feat_emb = embeddings[:, self.d_numerical + cat_idx, :]  # [batch_size, d_embed]
                
                # Compute similarity with all possible category embeddings
                embedding_matrix = self.cat_embedders[i].weight  # [cat_size, d_embed]
                
                # Calculate cosine similarity
                feat_emb_norm = feat_emb / feat_emb.norm(dim=1, keepdim=True)
                embedding_matrix_norm = embedding_matrix / embedding_matrix.norm(dim=1, keepdim=True)
                cos_sim = torch.matmul(feat_emb_norm, embedding_matrix_norm.t())  # [batch_size, cat_size]
                
                # Get the most similar category
                _, predicted_cat = torch.max(cos_sim, dim=1)
                x_cat[:, i] = predicted_cat
                cat_idx += 1
                
        return x_num, x_cat
        
    def expand_emb_mask(
        self,
        emb_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Expand feature-level mask to embedding-level mask.
        
        Args:
            emb_mask: Feature-level mask [batch_size, d_numerical + d_categorical]
            
        Returns:
            expanded_mask: Embedding-level mask [batch_size, (d_numerical + d_categorical) * d_embed]
        """
        batch_size = emb_mask.shape[0]
        total_features = self.d_numerical + self.d_categorical
        
        # Reshape to [batch_size, total_features, 1] and repeat along embedding dimension
        expanded_mask = emb_mask.view(batch_size, total_features, 1).repeat(1, 1, self.d_embed)
        
        # Reshape to [batch_size, total_features * d_embed]
        expanded_mask = expanded_mask.view(batch_size, total_features * self.d_embed)
        
        return expanded_mask

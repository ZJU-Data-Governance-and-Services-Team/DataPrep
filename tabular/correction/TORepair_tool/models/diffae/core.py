import time
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import math
from typing import Union, List
from torch.utils.data import DataLoader
from utils.feature_embedder import FeatureEmbedder

# --- Time Embedding (Remains the same) ---
class SinusoidalPosEmb(nn.Module):
    """Sinusoidal Positional Embedding for time steps."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        # Avoid potential division by zero or log(<=0) if half_dim is 0 or 1
        denominator = max(half_dim - 1, 1e-6)
        emb = torch.log(torch.tensor(10000.0, device=device)) / denominator
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :] # Shape: [batch_size, half_dim]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1) # Shape: [batch_size, half_dim * 2]
        # Pad if dim is odd
        if self.dim % 2 == 1:
           emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1) # Pad with zero
        # Ensure final dimension matches self.dim, handling potential rounding issues
        emb = emb[:, :self.dim]
        return emb

class TimeEmbedMLP(nn.Module):
    """MLP to process sinusoidal time embeddings."""
    def __init__(self, time_emb_dim, output_dim):
        super().__init__()
        self.time_emb = SinusoidalPosEmb(time_emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(time_emb_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, time):
        # Handle cell-level timestep sampling: time can be [batch_size] or [batch_size, num_features]
        if time.dim() == 2:  # Cell-level: [batch_size, num_features]
            # For cell-level timestep sampling, we need to process each feature's timestep
            batch_size, num_features = time.shape
            # Flatten to process all timesteps at once
            time_flat = time.view(-1)  # [batch_size * num_features]
            emb_flat = self.time_emb(time_flat.float())  # [batch_size * num_features, time_emb_dim]
            emb_flat = self.mlp(emb_flat)  # [batch_size * num_features, output_dim]
            # Reshape back and take mean across features
            emb = emb_flat.view(batch_size, num_features, -1).mean(dim=1)  # [batch_size, output_dim]
        else:  # Batch-level: [batch_size]
            emb = self.time_emb(time.float())
            emb = self.mlp(emb)
        return emb


# --- Transformer 块 ---
class TransformerBlock(nn.Module):
    """
    A single Transformer block with Multi-Head Self-Attention and Feed-Forward layers.
    Includes Layer Normalization, Residual Connections, and conditioning on time/guidance.
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        """
        Args:
            embed_dim (int): The embedding dimension of the input/output.
            num_heads (int): Number of attention heads.
            ff_dim (int): Dimension of the feed-forward layer.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim

        # Multi-Head Self-Attention
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.SiLU(), # Changed from ReLU to SiLU (Swish) which often works well
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )

        # Layer Normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor. Shape [batch_size, seq_len, embed_dim].
                               For tabular data, seq_len is typically 1.
        Returns:
            torch.Tensor: Output tensor. Shape [batch_size, seq_len, embed_dim].
        """
        # --- Conditioning and First Residual Connection ---
        residual = x
        x_norm1 = self.norm1(x)

        # --- Multi-Head Self-Attention ---
        # For tabular data (seq_len=1), self-attention acts like a gated projection.
        # If features were treated as a sequence (seq_len=num_features), it would capture inter-feature relationships.
        # We'll assume input is [batch, 1, embed_dim] for now.
        attn_output, _ = self.attn(x_norm1, x_norm1, x_norm1)
        x = residual + self.dropout(attn_output) # Add residual connection

        # --- Feed-Forward Network and Second Residual Connection ---
        residual = x
        x_norm2 = self.norm2(x)
        ffn_output = self.ffn(x_norm2)
        x = residual + self.dropout(ffn_output) # Add residual connection

        return x

# --- 主 Transformer 模型 ---
class TabularTransformer(nn.Module):
    """
    Transformer-based architecture for Tabular Data Diffusion.
    Uses additive guidance.
    """
    def __init__(self, input_dim, num_layers=4, embed_dim=256, num_heads=8, ff_dim=1024,
                 time_emb_dim=256, d_numerical=0, d_categorical=0, actual_cat_sizes=None, dropout=0.1):
        """
        Initializes the TabularTransformer model.

        Args:
            input_dim (int): Total dimension of the input (numerical + one-hot categorical).
            num_layers (int): Number of Transformer blocks.
            embed_dim (int): Internal embedding dimension of the Transformer.
            num_heads (int): Number of attention heads in each Transformer block.
            ff_dim (int): Dimension of the feed-forward layer within Transformer blocks.
            time_emb_dim (int): Dimension for the raw sinusoidal time embedding.
            d_numerical (int): Number of numerical features.
            d_categorical (int): Total dimension of one-hot encoded categorical features.
            actual_cat_sizes (list): List of category counts for each categorical feature.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.input_dim = input_dim
        self.d_numerical = d_numerical
        self.d_categorical = d_categorical
        self.actual_cat_sizes = actual_cat_sizes
        self.embed_dim = embed_dim

        # --- Embeddings ---
        # 1. Input Projection: Project flattened input features to embed_dim
        self.input_proj = nn.Linear(input_dim, embed_dim)

        # 2. Time Embedding MLP: Process sinusoidal time embedding
        # Output dimension should match the dimension expected by TransformerBlock's time_proj
        self.time_mlp = TimeEmbedMLP(time_emb_dim, embed_dim) # Output dim = embed_dim

        # 3. Guidance Projection Layer: Project raw guidance_grad to embed_dim
        # Output dimension should match the dimension expected by TransformerBlock's guidance_proj
        self.guidance_proj_model = nn.Linear(self.input_dim, embed_dim) # Output dim = embed_dim
        self.time_proj = nn.Linear(embed_dim, embed_dim)
        self.guidance_proj = nn.Linear(embed_dim, embed_dim)

        # --- Transformer Body ---
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        # --- Output ---
        # Final Layer Normalization
        self.final_norm = nn.LayerNorm(embed_dim)
        # Final projection back to the original input dimension
        self.final_proj = nn.Linear(embed_dim, input_dim)

        print(f"Initialized TabularTransformer: input_dim={input_dim}, num_layers={num_layers}, "
              f"embed_dim={embed_dim}, num_heads={num_heads}, ff_dim={ff_dim}")

    def forward(self, x_t, t, guidance_grad=None):
        """
        Forward pass of the TabularTransformer model.

        Args:
            x_t (torch.Tensor): Noisy input data at timestep t. Shape [batch_size, input_dim].
            t (torch.Tensor): Timesteps for each sample in the batch. Shape [batch_size].
            guidance_grad (torch.Tensor, optional): Precomputed guidance gradient.
                                                    Shape [batch_size, input_dim]. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Predicted numerical features (pred_x0_num) and
                                               predicted categorical logits (pred_x0_cat_flat_logits).
        """
        # 1. Get Time Embedding
        t_emb = self.time_mlp(t) # Shape: [batch_size, embed_dim]

        # 2. Process Guidance Gradient (if provided)
        guidance_emb = None
        if guidance_grad is not None:
            guidance_emb = self.guidance_proj_model(guidance_grad) # Shape: [batch_size, embed_dim]

        # 3. Project Input
        x_embed = self.input_proj(x_t) # Shape: [batch_size, embed_dim]
        
        x_embed = x_embed + self.time_proj(t_emb)
        if guidance_emb is not None:
            x_embed = x_embed + self.guidance_proj(guidance_emb)

        # 4. Reshape for Transformer: Add a sequence dimension (length 1 for tabular)
        # Transformer blocks expect [batch_size, seq_len, embed_dim]
        x = x_embed.unsqueeze(1) # Shape: [batch_size, 1, embed_dim]

        # 5. Pass through Transformer Layers
        for layer in self.transformer_layers:
            x = layer(x) # Pass processed embeddings

        # 6. Final Normalization
        x = self.final_norm(x)

        # 7. Remove sequence dimension
        x = x.squeeze(1) # Shape: [batch_size, embed_dim]

        # 8. Final Projection
        output_raw = self.final_proj(x) # Shape: [batch_size, input_dim]

        # 9. Split Output
        pred_x0_num = output_raw[:, :self.d_numerical]
        pred_x0_cat_flat_logits = output_raw[:, self.d_numerical:]

        return pred_x0_num, pred_x0_cat_flat_logits


# --- U-Net Building Block (Modified for Additive Guidance) ---
class UNetLayer(nn.Module):
    """
    A single layer block for the U-Net, processing features and time embedding.
    """
    def __init__(self, input_dim, output_dim, dropout=0.2):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

        # Residual connection only if input and output dimensions match
        self.has_residual = (input_dim == output_dim)

    def forward(self, x):
        """Forward pass for the layer."""
        # Store input for residual connection if dimensions match
        residual = x if self.has_residual else None

        # Apply linear transformation
        h = self.linear(x)

        # Apply residual connection *before* normalization and activation (common practice)
        if residual is not None:
            h = h + residual

        # Apply normalization, activation, and dropout
        h = self.norm(h)
        h = self.act(h)
        h = self.dropout(h)

        return h


# --- Main Tabular U-Net Model (Additive Guidance) ---
class TabularUNet(nn.Module):
    """
    U-Net architecture adapted for Tabular Data Diffusion.
    Uses addition for guidance and includes skip connections.
    """
    def __init__(self, input_dim, time_emb_dim=256, hidden_dim=256, latent_dim=64,
                 d_numerical=0, d_categorical=0, actual_cat_sizes=None, dropout=0.2,
                 predict_noise=False, embedding_dim=None):
        """
        Initializes the TabularUNet model.

        Args:
            input_dim (int): Total dimension of the input (numerical + one-hot categorical).
            time_emb_dim (int): Dimension for the raw sinusoidal time embedding.
            hidden_dim (int): Base hidden dimension for layers and embeddings.
            latent_dim (int): Dimension of the bottleneck layer.
            d_numerical (int): Number of numerical features.
            d_categorical (int): Total dimension of one-hot encoded categorical features.
            actual_cat_sizes (list): List of category counts for each categorical feature.
            dropout (float): Dropout rate.
            predict_noise (bool): Whether to predict noise (for embedding space) or features.
            embedding_dim (int): Dimension of the embedding space when predict_noise=True.
        """
        super().__init__()
        self.input_dim = input_dim
        self.d_numerical = d_numerical
        self.d_categorical = d_categorical
        self.actual_cat_sizes = actual_cat_sizes
        self.hidden_dim = hidden_dim
        self.predict_noise = predict_noise
        self.embedding_dim = embedding_dim
        self.latent_dim = self.hidden_dim // 8

        # Time embedding MLP
        self.time_mlp = TimeEmbedMLP(time_emb_dim, hidden_dim)
        self.time_proj_unet = nn.Linear(hidden_dim, hidden_dim)

        # Determine input dimensions based on predict_noise mode
        if predict_noise and embedding_dim is not None:
            # New architecture: input is [e_conditional, e_mask] -> 2 * embedding_dim
            single_channel_dim = embedding_dim
            expected_input_dim = 2 * embedding_dim
        else:
            # Legacy 2-channel architecture
            single_channel_dim = self.input_dim // 2 if self.input_dim % 2 == 0 else self.input_dim
            expected_input_dim = self.input_dim

        # --- Encoder (Down-sampling Path) ---
        # Initial projection from expected_input_dim to hidden_dim
        self.input_proj = nn.Linear(expected_input_dim, hidden_dim)

        # Layer 1: Takes projected input, adds time only (no guidance)
        self.enc_layer1 = UNetLayer(hidden_dim, hidden_dim, dropout)  # guidance_dim=0

        # Layer 2: Reduces dimension, adds time only
        self.enc_layer2 = UNetLayer(hidden_dim, hidden_dim // 2, dropout)  # guidance_dim=0

        # Bottleneck Layer: Adds time only
        self.bottleneck = UNetLayer(hidden_dim // 2, self.latent_dim, dropout)  # guidance_dim=0

        # --- Decoder (Up-sampling Path) ---
        # Layer 1: Takes bottleneck output, adds time only
        self.dec_layer1 = UNetLayer(self.latent_dim, hidden_dim // 2, dropout)  # guidance_dim=0

        # Layer 2: Takes concatenated input (output of dec_layer1 + skip connection from enc_layer2)
        concat_dim_dec2_in = (hidden_dim // 2) + (hidden_dim // 2)  # dec1_out + enc2_skip
        self.dec_layer2 = UNetLayer(concat_dim_dec2_in, hidden_dim, dropout)  # guidance_dim=0

        # Final Layer: Takes concatenated input (output of dec_layer2 + skip connection from enc_layer1)
        concat_dim_dec_final_in = hidden_dim + hidden_dim  # dec2_out + enc1_skip
        self.dec_final_layer = UNetLayer(concat_dim_dec_final_in, hidden_dim, dropout)  # guidance_dim=0

        # Final projection to either embedding_dim or single_channel_dim based on predict_noise
        if predict_noise and embedding_dim is not None:
            self.final_proj = nn.Linear(hidden_dim, embedding_dim)
        else:
            self.final_proj = nn.Linear(hidden_dim, single_channel_dim)

        # Reconstructors
        if actual_cat_sizes and sum(actual_cat_sizes) > 0:
            self.cat_reconstructor = nn.ModuleList(
                [nn.Linear(single_channel_dim, size) for size in actual_cat_sizes if size > 0]
            )
        else:
            self.cat_reconstructor = None

        if self.d_numerical > 0:
            self.num_reconstructor = nn.ModuleList(
                [nn.Linear(single_channel_dim, 1) for _ in range(self.d_numerical)]
            )
        else:
            self.num_reconstructor = None


        # Weight initialization
        self.apply(self._init_weights)
        
        # Apply specific initialization for reconstructors
        if self.cat_reconstructor is not None:
            for recon in self.cat_reconstructor:
                nn.init.xavier_uniform_(recon.weight, gain=1 / math.sqrt(2))
        
        if self.num_reconstructor is not None:
            for recon in self.num_reconstructor:
                nn.init.xavier_uniform_(recon.weight, gain=1 / math.sqrt(2))

        print(f"Initialized TabularUNet: input_dim={input_dim}, expected_input_dim={expected_input_dim}, "
              f"hidden_dim={hidden_dim}, latent_dim={self.latent_dim}, predict_noise={predict_noise}")
        if predict_noise and embedding_dim is not None:
            print(f"Output dimension: {embedding_dim} (embedding space)")
        else:
            print(f"Output dimension: {single_channel_dim} (feature reconstruction)")
        print(f"Decoder Layer 2 input dim: {concat_dim_dec2_in}")
        print(f"Decoder Final Layer input dim: {concat_dim_dec_final_in}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _reconstruct_output(self, output_raw):
        """
        Reconstructs the model output into numerical and categorical predictions.
        """
        # Numerical part
        if self.num_reconstructor is not None:
            num_outputs = []
            for recon in self.num_reconstructor:
                num_output = recon(output_raw)
                num_outputs.append(num_output)
            pred_x0_num = torch.cat(num_outputs, dim=1)
        else:
            pred_x0_num = output_raw[:, :self.d_numerical]

        # Categorical part
        if self.cat_reconstructor is not None and self.actual_cat_sizes and len(self.actual_cat_sizes) > 0:
            cat_outputs = []
            recon_idx = 0
            for i in range(len(self.actual_cat_sizes)):
                if self.actual_cat_sizes[i] > 0:
                    cat_output = self.cat_reconstructor[recon_idx](output_raw)
                    cat_outputs.append(cat_output)
                    recon_idx += 1
            
            if cat_outputs:
                pred_x0_cat_logits = cat_outputs
            else:
                pred_x0_cat_logits = []
            
            return pred_x0_num, pred_x0_cat_logits
        else:
            return pred_x0_num, []


    def forward(self, x_t, t, guidance_grad=None, y_labels=None):
        """
        Forward pass of the TabularUNet model.

        Args:
            x_t (torch.Tensor): Noisy input data at timestep t. Shape [batch_size, input_dim].
            t (torch.Tensor): Timesteps for each sample in the batch. Shape [batch_size] or [batch_size, num_features].
            guidance_grad (torch.Tensor, optional): Not used, kept for compatibility.
            y_labels (torch.Tensor, optional): Not used, kept for compatibility.

        Returns:
            For predict_noise=True: torch.Tensor of predicted noise
            For predict_noise=False: tuple[torch.Tensor, List[torch.Tensor]] of (pred_x0_num, pred_x0_cat_logits)
        """
        batch_size = x_t.shape[0]
        
        # Check if we're in noise prediction mode
        if self.predict_noise:
            # New 2-channel input: [e_conditional, e_mask]
            expected_dim = 2 * self.embedding_dim
            if x_t.shape[1] != expected_dim:
                raise ValueError(f"Expected input dimension {expected_dim}, got {x_t.shape[1]}")
        else:
            # Legacy mode - check expected dimensions
            if x_t.shape[1] != self.input_dim:
                # Handle potential 2-channel format compatibility
                expected_single_channel_dim = self.input_dim // 2
                if x_t.shape[1] == self.input_dim:
                    # Already in correct format
                    pass
                else:
                    raise ValueError(f"Expected input dimension {self.input_dim}, got {x_t.shape[1]}")
        
        # Time embedding - TimeEmbedMLP already handles both batch-level and cell-level timesteps
        t_emb = self.time_mlp(t)
        
        # Initial Projection
        h = self.input_proj(x_t)  # Shape: [batch_size, hidden_dim]
        h = h + self.time_proj_unet(t_emb)

        # --- Encoder Path ---
        # Store outputs for skip connections
        skip_connections = []

        # All layers now pass None for guidance_emb
        h_enc1 = self.enc_layer1(h)  # Shape: [batch_size, hidden_dim]
        skip_connections.append(h_enc1)

        h_enc2 = self.enc_layer2(h_enc1)  # Shape: [batch_size, hidden_dim // 2]
        skip_connections.append(h_enc2)

        # --- Bottleneck ---
        h_bottle = self.bottleneck(h_enc2)  # Shape: [batch_size, self.latent_dim]

        # --- Decoder Path ---
        h = h_bottle
        h_dec1 = self.dec_layer1(h)  # Shape: [batch_size, hidden_dim // 2]

        # Concatenate with skip connection from enc_layer2
        skip2 = skip_connections.pop()  # h_enc2, Shape: [batch_size, hidden_dim // 2]
        h_cat2 = torch.cat((h_dec1, skip2), dim=-1)  # Shape: [batch_size, hidden_dim]
        h_dec2 = self.dec_layer2(h_cat2)  # Shape: [batch_size, hidden_dim]

        # Concatenate with skip connection from enc_layer1
        skip1 = skip_connections.pop()  # h_enc1, Shape: [batch_size, hidden_dim]
        h_cat_final = torch.cat((h_dec2, skip1), dim=-1)  # Shape: [batch_size, hidden_dim * 2]
        h_dec_final = self.dec_final_layer(h_cat_final)  # Shape: [batch_size, hidden_dim]

        # --- Final Projection ---
        output_raw = self.final_proj(h_dec_final)
        
        if self.predict_noise:
            # In noise prediction mode, directly return the predicted noise
            return output_raw
        else:
            # Legacy mode: reconstruct features
            return self._reconstruct_output(output_raw)


# --- DiffAE Layer ---
class AELayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.2):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        
        # Optional residual connection if dimensions match
        self.has_residual = (input_dim == output_dim)
    
    def forward(self, x):
        h = self.linear(x)
        h = self.norm(h)
        h = self.act(h)
        h = self.dropout(h)
        # Add residual if possible
        if self.has_residual:
            h = h + x
        return h

# --- Main DiffAE Model ---
class TabularAE(nn.Module):
    def __init__(self, input_dim, time_emb_dim=256, hidden_dim=256, latent_dim=64, 
                 d_numerical=0, d_categorical=0, actual_cat_sizes=None, dropout=0.2,
                 num_classes=0, d_label_embed=32, masked_label_value=None,
                 predict_noise=False, embedding_dim=None):
        super().__init__()
        self.input_dim = input_dim
        self.d_numerical = d_numerical
        self.d_categorical = d_categorical
        self.actual_cat_sizes = actual_cat_sizes
        self.num_classes = num_classes
        self.masked_label_value = masked_label_value
        self.hidden_dim = hidden_dim
        self.predict_noise = predict_noise
        self.embedding_dim = embedding_dim
        
        # Time embedding
        self.time_mlp = TimeEmbedMLP(time_emb_dim, self.hidden_dim)
        self.time_proj_ae = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        # Guidance projection layer - expects single-channel dimension
        # For 2-channel input (embedding + mask), adjust accordingly
        if predict_noise and embedding_dim is not None:
            # New architecture: input is [e_conditional, e_mask] -> 2 * embedding_dim
            single_channel_dim = embedding_dim
            expected_input_dim = 2 * embedding_dim
        else:
            # Legacy 2-channel architecture
            single_channel_dim = self.input_dim // 2 if self.input_dim % 2 == 0 else self.input_dim
            expected_input_dim = self.input_dim
            
        self.guidance_proj = nn.Linear(single_channel_dim, self.hidden_dim)
        self.guidance_proj_ae = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        # Label embedding layers
        if num_classes > 0:
            self.label_embedding = nn.Embedding(num_classes, d_label_embed)
            self.label_mlp = nn.Sequential(
                nn.Linear(d_label_embed, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim)
            )
            label_cond_dim = self.hidden_dim
        else:
            self.label_embedding = None
            self.label_mlp = None
            label_cond_dim = 0
        
        # Encoder
        self.encoder = nn.ModuleDict({
            'input_proj': nn.Linear(expected_input_dim, self.hidden_dim),
            'layer1': AELayer(self.hidden_dim, self.hidden_dim, dropout),
            'layer2': AELayer(self.hidden_dim, self.hidden_dim // 2, dropout),
            'bottleneck': AELayer(self.hidden_dim // 2, latent_dim, dropout),
        })
        
        # Decoder
        self.decoder = nn.ModuleDict({
            'layer1': AELayer(latent_dim, self.hidden_dim // 2, dropout),
            'layer2': AELayer(self.hidden_dim // 2, self.hidden_dim, dropout),
            'final_layer': AELayer(self.hidden_dim, self.hidden_dim, dropout),
            'final_proj': nn.Linear(self.hidden_dim, embedding_dim if predict_noise else single_channel_dim)
        })
        
        # Reconstructors - only needed for legacy mode (not when predicting noise)
        if actual_cat_sizes and sum(actual_cat_sizes) > 0:
            self.cat_reconstructor = nn.ModuleList(
                [nn.Linear(single_channel_dim, size) for size in actual_cat_sizes if size > 0]
            )
        else:
            self.cat_reconstructor = None

        if self.d_numerical > 0:
            self.num_reconstructor = nn.ModuleList(
                [nn.Linear(single_channel_dim, 1) for _ in range(self.d_numerical)]
            )
        else:
            self.num_reconstructor = None
            
        # --- Weight Initialization ---
        self.apply(self._init_weights)
        
        # Apply specific initialization for reconstructors to override the general one
        if self.cat_reconstructor is not None:
            for recon in self.cat_reconstructor:
                nn.init.xavier_uniform_(recon.weight, gain=1 / math.sqrt(2))
        
        if self.num_reconstructor is not None:
            for recon in self.num_reconstructor:
                nn.init.xavier_uniform_(recon.weight, gain=1 / math.sqrt(2))
        
        print(f"Initialized TabularDiffAE: input_dim={input_dim}, "
              f"hidden_dim={hidden_dim}, latent_dim={latent_dim}, "
              f"num_classes={num_classes}, d_label_embed={d_label_embed}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def _reconstruct_output(self, output_raw):
        """
        Reconstructs the model output into numerical and categorical predictions.
        """
        # Numerical part
        if self.num_reconstructor is not None:
            num_outputs = []
            for recon in self.num_reconstructor:
                num_output = recon(output_raw)
                num_outputs.append(num_output)
            pred_x0_num = torch.cat(num_outputs, dim=1)
        else:
            pred_x0_num = output_raw[:, :self.d_numerical]

        # Categorical part
        if self.cat_reconstructor is not None and self.actual_cat_sizes and len(self.actual_cat_sizes) > 0:
            cat_outputs = []
            recon_idx = 0
            for i in range(len(self.actual_cat_sizes)):
                if self.actual_cat_sizes[i] > 0:
                    cat_output = self.cat_reconstructor[recon_idx](output_raw)
                    cat_outputs.append(cat_output)
                    recon_idx += 1
            
            if cat_outputs:
                pred_x0_cat_logits = cat_outputs
            else:
                pred_x0_cat_logits = []
            
            return pred_x0_num, pred_x0_cat_logits
        else:
            return pred_x0_num, []
    
    def forward(self, x_t, t, guidance_grad=None, y_labels=None):
        batch_size = x_t.shape[0]
        
        # Check if we're in noise prediction mode
        if self.predict_noise:
            # New 2-channel input: [e_conditional, e_mask]
            expected_dim = 2 * self.embedding_dim
            if x_t.shape[1] != expected_dim:
                raise ValueError(f"Expected input dimension {expected_dim}, got {x_t.shape[1]}")
            
            # Split into conditional embedding and mask
            e_conditional = x_t[:, :self.embedding_dim]
            e_mask = x_t[:, self.embedding_dim:]
            
            # Use the full input for processing
            combined_input = x_t
        else:
            # Legacy 2-channel mode
            # Check if input is 2-channel format (noisy_target + condition_data + condition_mask)
            expected_single_channel_dim = self.input_dim // 2
            if x_t.shape[1] == self.input_dim:
                # 2-channel input: split into two channels
                noisy_target = x_t[:, :expected_single_channel_dim]
                condition_data = x_t[:, expected_single_channel_dim:2*expected_single_channel_dim]
                condition_mask = x_t[:, 2*expected_single_channel_dim:]
                
                # Combine channels with channel-aware processing
                # The model will learn to process the combined information
                combined_input = x_t  # Use all 2 channels directly
            else:
                # Legacy single-channel input - replicate to 2 channels for compatibility
                combined_input = torch.cat([x_t, x_t, torch.ones_like(x_t)], dim=1)
        
        # Time embedding - TimeEmbedMLP already handles both batch-level and cell-level timesteps
        t_emb = self.time_mlp(t)
        
        # Process guidance if provided
        guidance_emb = None
        if guidance_grad is not None:
            guidance_emb = self.guidance_proj(guidance_grad)
        
        # 处理标签条件
        label_emb = None
        if y_labels is not None and self.label_embedding is not None:
            # 将标签转换为长整型，并确保在有效范围内
            y_labels = y_labels.long()
            if self.masked_label_value is not None:
                # 对于被掩码的标签，用指定的值替代
                y_labels = torch.where(
                    y_labels < 0, 
                    torch.tensor(self.masked_label_value, device=y_labels.device),
                    y_labels
                )
            # 确保标签在有效范围内
            y_labels = torch.clamp(y_labels, 0, self.num_classes - 1)
            # 获取标签嵌入并通过MLP处理
            label_emb = self.label_embedding(y_labels)
            label_emb = self.label_mlp(label_emb)
        
        # --- Encoder ---
        h = self.encoder['input_proj'](combined_input)
        
        h = h + self.time_proj_ae(t_emb)
        if guidance_emb is not None:
            h = h + self.guidance_proj_ae(guidance_emb)
        if label_emb is not None:
            h = h + label_emb
            
        h = self.encoder['layer1'](h)
        h = self.encoder['layer2'](h)
        h = self.encoder['bottleneck'](h)
        
        # --- Decoder ---
        h = self.decoder['layer1'](h)
        h = self.decoder['layer2'](h)
        h = self.decoder['final_layer'](h)
        output_raw = self.decoder['final_proj'](h)
        
        if self.predict_noise:
            # In noise prediction mode, directly return the predicted noise
            return output_raw
        else:
            # Legacy mode: reconstruct features
            return self._reconstruct_output(output_raw)
        

# --- Residual MLP Model ---
class ResidualMLP(nn.Module):
    """
    A simple Residual MLP architecture, compatible with the TabularUNet/TabularAE interface.
    """
    def __init__(self, input_dim, num_layers=4, time_emb_dim=256, hidden_dim=256, 
                 d_numerical=0, d_categorical=0, actual_cat_sizes=None, dropout=0.2,
                 predict_noise=False, embedding_dim=None,
                 # Compatibility args from other models
                 latent_dim=None, num_classes=0, d_label_embed=32, masked_label_value=None):
        super().__init__()
        self.input_dim = input_dim
        self.d_numerical = d_numerical
        self.d_categorical = d_categorical
        self.actual_cat_sizes = actual_cat_sizes
        self.hidden_dim = hidden_dim
        self.predict_noise = predict_noise
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.masked_label_value = masked_label_value

        # Time embedding MLP
        self.time_mlp = TimeEmbedMLP(time_emb_dim, hidden_dim)

        # Label embedding layers
        if num_classes > 0:
            self.label_embedding = nn.Embedding(num_classes, d_label_embed)
            self.label_mlp = nn.Sequential(
                nn.Linear(d_label_embed, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim)
            )
            label_cond_dim = self.hidden_dim
        else:
            self.label_embedding = None
            self.label_mlp = None
            label_cond_dim = 0

        # Determine input dimensions based on predict_noise mode
        if predict_noise and embedding_dim is not None:
            expected_input_dim = 2 * embedding_dim
            single_channel_dim = embedding_dim
        else:
            single_channel_dim = self.input_dim // 2 if self.input_dim % 2 == 0 else self.input_dim
            expected_input_dim = self.input_dim

        # Input projection
        self.input_proj = nn.Linear(expected_input_dim, hidden_dim)

        # Residual blocks
        self.layers = nn.ModuleList([
            self._build_block(hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        # Final projection to either embedding_dim or single_channel_dim
        if predict_noise and embedding_dim is not None:
            self.final_proj = nn.Linear(hidden_dim, embedding_dim)
        else:
            self.final_proj = nn.Linear(hidden_dim, single_channel_dim)

        # Reconstructors for non-noise prediction mode
        if actual_cat_sizes and sum(actual_cat_sizes) > 0:
            self.cat_reconstructor = nn.ModuleList(
                [nn.Linear(single_channel_dim, size) for size in actual_cat_sizes if size > 0]
            )
        else:
            self.cat_reconstructor = None

        if self.d_numerical > 0:
            self.num_reconstructor = nn.ModuleList(
                [nn.Linear(single_channel_dim, 1) for _ in range(self.d_numerical)]
            )
        else:
            self.num_reconstructor = None
        
        self.apply(self._init_weights)
        print(f"Initialized ResidualMLP: input_dim={input_dim}, expected_input_dim={expected_input_dim}, "
              f"hidden_dim={hidden_dim}, num_layers={num_layers}, predict_noise={predict_noise}")

    def _build_block(self, hidden_dim, dropout):
        class Block(nn.Module):
            def __init__(self, hidden_dim, dropout):
                super().__init__()
                self.linear1 = nn.Linear(hidden_dim, hidden_dim)
                self.linear2 = nn.Linear(hidden_dim, hidden_dim)
                self.norm1 = nn.LayerNorm(hidden_dim)
                self.norm2 = nn.LayerNorm(hidden_dim)
                self.act = nn.SiLU()
                self.dropout = nn.Dropout(dropout)

            def forward(self, x):
                residual = x
                h = self.norm1(x)
                h = self.act(h)
                h = self.linear1(h)

                h = self.norm2(h)
                h = self.act(h)
                h = self.dropout(h)
                h = self.linear2(h)

                return h + residual
        return Block(hidden_dim, dropout)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _reconstruct_output(self, output_raw):
        if hasattr(self, 'num_reconstructor') and self.num_reconstructor is not None:
            num_outputs = [recon(output_raw) for recon in self.num_reconstructor]
            pred_x0_num = torch.cat(num_outputs, dim=1)
        else:
            pred_x0_num = output_raw[:, :self.d_numerical]

        if self.cat_reconstructor is not None and self.actual_cat_sizes:
            cat_outputs = []
            recon_idx = 0
            for size in self.actual_cat_sizes:
                if size > 0:
                    cat_outputs.append(self.cat_reconstructor[recon_idx](output_raw))
                    recon_idx += 1
            return pred_x0_num, cat_outputs
        
        return pred_x0_num, []

    def forward(self, x_t, t, guidance_grad=None, y_labels=None):
        if self.predict_noise:
            expected_dim = 2 * self.embedding_dim
            if x_t.shape[1] != expected_dim:
                raise ValueError(f"Expected input dimension {expected_dim}, got {x_t.shape[1]}")
        
        t_emb = self.time_mlp(t)
        
        label_emb = None
        if y_labels is not None and self.label_embedding is not None:
            y_labels = y_labels.long()
            if self.masked_label_value is not None:
                y_labels = torch.where(
                    y_labels < 0, 
                    torch.tensor(self.masked_label_value, device=y_labels.device),
                    y_labels
                )
            y_labels = torch.clamp(y_labels, 0, self.num_classes - 1)
            label_emb = self.label_embedding(y_labels)
            label_emb = self.label_mlp(label_emb)
            
        h = self.input_proj(x_t)
        
        h = h + t_emb
        if label_emb is not None:
            h = h + label_emb

        for layer in self.layers:
            h = layer(h)
        
        output_raw = self.final_proj(h)
        
        if self.predict_noise:
            return output_raw
        else:
            return self._reconstruct_output(output_raw)



# --- Diffusion Scheduler ---
def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    """
    Linear beta schedule
    """
    return torch.linspace(beta_start, beta_end, timesteps)

# --- Diffusion Utilities ---
class DiffusionUtils:
    def __init__(self, num_timesteps=1000, schedule='cosine', device='cpu'):
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Setup noise schedule
        if schedule == 'cosine':
            betas = cosine_beta_schedule(num_timesteps).to(device)
        else:  # linear
            betas = linear_beta_schedule(num_timesteps).to(device)
        
        # Precompute diffusion parameters
        self.betas = betas
        alphas = 1. - betas
        self.alphas = alphas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - self.alphas_cumprod)
        
        # Initialize storage for categorical marginal distributions
        self.cat_marginals = None
    
    def compute_cat_marginals(self, dataloader, actual_cat_sizes):
        """
        Compute marginal distributions for all categorical features from the training dataset.
        
        Args:
            dataloader: DataLoader containing the training data
            actual_cat_sizes: List of category sizes for each categorical feature
        
        Returns:
            List of tensors, each containing the marginal distribution for a categorical feature
        """
        if self.cat_marginals is not None:
            return self.cat_marginals
        
        # Initialize counters for each category in each feature
        cat_counts = []
        for size in actual_cat_sizes:
            if size > 0:
                cat_counts.append(torch.zeros(size, device=self.device))
            else:
                cat_counts.append(None)
        
        # Count occurrences of each category
        total_samples = 0
        for batch in dataloader:
            if len(batch) == 5:  # If using neighbor information
                _, x_cat_int, _, _, _ = batch
            else:
                _, x_cat_int, _ = batch
            
            if x_cat_int is None:
                continue
                
            x_cat_int = x_cat_int.to(self.device)
            batch_size = x_cat_int.shape[0]
            total_samples += batch_size
            
            # Update counts for each categorical feature
            for j, counts in enumerate(cat_counts):
                if counts is not None:
                    # Get category indices for feature j
                    cat_indices = x_cat_int[:, j].long()
                    # Ensure indices are valid (clamp to valid range)
                    valid_indices = cat_indices.clamp(0, counts.size(0) - 1)
                    # Count occurrences
                    for idx in valid_indices:
                        counts[idx] += 1
        
        # Convert counts to probabilities
        cat_marginals = []
        for counts in cat_counts:
            if counts is not None:
                # Add small epsilon to avoid zeros
                smoothed_counts = counts + 1e-5
                # Normalize to get probabilities
                probs = smoothed_counts / smoothed_counts.sum()
                cat_marginals.append(probs)
            else:
                cat_marginals.append(None)
        
        self.cat_marginals = cat_marginals
        return cat_marginals
    
    def q_sample_embedding(self, e_0, t, noise=None):
        """
        Forward diffusion in embedding space: sample q(e_t | e_0)
        
        Args:
            e_0: Clean embeddings tensor of shape [batch_size, embedding_dim]
            t: Timesteps tensor of shape [batch_size] or [batch_size, num_features]
            noise: Optional pre-generated noise for embeddings
            
        Returns:
            e_t: Noisy embeddings at timestep t
            epsilon: The noise that was added
        """
        if noise is None:
            noise = torch.randn_like(e_0)
        
        # Handle both batch-level and cell-level timesteps
        if t.dim() > 1:
            # Cell-level timesteps: apply noise to each feature's embedding slice independently
            batch_size, num_features = t.shape
            total_embedding_dim = e_0.shape[1]
            
            if total_embedding_dim % num_features != 0:
                raise ValueError(
                    "Total embedding dimension must be divisible by the number of features "
                    "for cell-level noising."
                )
            feature_emb_dim = total_embedding_dim // num_features
            
            e_t_parts = []
            for i in range(num_features):
                # Get the slice for the current feature
                start_dim = i * feature_emb_dim
                end_dim = start_dim + feature_emb_dim
                e_0_slice = e_0[:, start_dim:end_dim]
                noise_slice = noise[:, start_dim:end_dim]
                
                # Get the timestep for the current feature
                t_feature = t[:, i]  # Shape: [batch_size]
                
                # Get noise schedule parameters for the feature's timestep
                sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t_feature].view(-1, 1)
                sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t_feature].view(-1, 1)
                
                # Apply noise to the slice
                e_t_slice = sqrt_alphas_cumprod_t * e_0_slice + sqrt_one_minus_alphas_cumprod_t * noise_slice
                e_t_parts.append(e_t_slice)
            
            # Concatenate the noised parts
            e_t = torch.cat(e_t_parts, dim=1)
        else:
            # Batch-level timesteps
            sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1)
            sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
            # Apply the formula: e_t = sqrt(alpha_cumprod_t) * e_0 + sqrt(1 - alpha_cumprod_t) * noise
            e_t = sqrt_alphas_cumprod_t * e_0 + sqrt_one_minus_alphas_cumprod_t * noise
        
        return e_t, noise

    def q_sample(self, x_0, t, noise=None, x_0_num=None, x_0_cat_int=None, actual_cat_sizes=None):
        """
        Forward diffusion: sample q(x_t | x_0)
        
        This method can be used in two ways:
        1. With combined x_0 tensor (original behavior)
        2. With separate x_0_num and x_0_cat_int tensors (new behavior for separate noise processes)
        
        Args:
            x_0: Combined tensor of numerical and one-hot categorical features (original input)
            t: Timesteps tensor of shape [batch_size]
            noise: Optional pre-generated noise for numerical features
            x_0_num: Numerical features tensor (optional)
            x_0_cat_int: Categorical features as integer indices (optional)
            actual_cat_sizes: List of category sizes for each categorical feature (required if x_0_cat_int is provided)
            
        Returns:
            x_t: Noisy data at timestep t
        """
        # Check if we're using the separate noise processes
        using_separate_features = (x_0_num is not None or x_0_cat_int is not None)
        
        if using_separate_features:
            assert actual_cat_sizes is not None, "actual_cat_sizes must be provided when using separate features"
            
            batch_size = t.shape[0]
            device = t.device
            
            # Process numerical features with Gaussian noise
            if x_0_num is not None and x_0_num.shape[1] > 0:
                if noise is None:
                    noise_num = torch.randn_like(x_0_num)
                else:
                    noise_num = noise
                
                # Handle both batch-level and cell-level timesteps
                if t.dim() > 1:
                    # Cell-level timesteps: t is [batch_size, features]
                    # For numerical features, we need to extract appropriate alpha values
                    batch_size, num_features = t.shape
                    sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]  # [batch_size, features]
                    sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]  # [batch_size, features]
                    
                    # Only use the portion corresponding to numerical features
                    if x_0_num.shape[1] < num_features:
                        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t[:, :x_0_num.shape[1]]
                        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t[:, :x_0_num.shape[1]]
                else:
                    # Batch-level timesteps: t is [batch_size]
                    sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1)
                    sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
                
                # Apply the formula: x_t_num = sqrt(alpha_cumprod_t) * x_0_num + sqrt(1 - alpha_cumprod_t) * noise_num
                x_t_num = sqrt_alphas_cumprod_t * x_0_num + sqrt_one_minus_alphas_cumprod_t * noise_num
            else:
                x_t_num = torch.empty((batch_size, 0), device=device)

            # Process categorical features with MASK for noisy values
            if x_0_cat_int is not None and x_0_cat_int.shape[1] > 0:
                # Initialize output tensor for noisy categorical indices
                x_t_cat_int = torch.zeros_like(x_0_cat_int)
                
                # Handle both batch-level and cell-level timesteps for categorical features
                if t.dim() > 1:
                    # Cell-level timesteps: t is [batch_size, features]
                    # For categorical features, we need to extract appropriate alpha values
                    batch_size, num_features = t.shape
                    
                    # Find the range of categorical features in the timestep tensor
                    num_num_features = x_0_num.shape[1] if x_0_num is not None else 0
                    cat_start_idx = num_num_features
                    cat_end_idx = cat_start_idx + x_0_cat_int.shape[1]
                    
                    # Extract categorical timesteps
                    if cat_end_idx <= num_features:
                        t_cat = t[:, cat_start_idx:cat_end_idx]  # [batch_size, n_cat_features]
                        alphas_cumprod_t = self.alphas_cumprod[t_cat]  # [batch_size, n_cat_features]
                    else:
                        # Fallback: use the first timestep for all categorical features
                        t_first = t[:, 0]  # [batch_size]
                        alphas_cumprod_t = self.alphas_cumprod[t_first].view(-1, 1)  # [batch_size, 1]
                else:
                    # Batch-level timesteps: t is [batch_size]
                    alphas_cumprod_t = self.alphas_cumprod[t]  # Shape: [batch_size]
                
                # Generate uniform random numbers for all categorical features at once
                # Shape: [batch_size, n_cat_features]
                u = torch.rand(batch_size, x_0_cat_int.shape[1], device=device)
                
                # Create a mask for keeping original values vs. replacing with MASK
                # Handle different shapes of alphas_cumprod_t
                if alphas_cumprod_t.dim() > 1:
                    # Cell-level: alphas_cumprod_t is [batch_size, n_cat_features]
                    keep_mask = u < alphas_cumprod_t
                else:
                    # Batch-level: alphas_cumprod_t is [batch_size]
                    keep_mask = u < alphas_cumprod_t.view(-1, 1)
                
                # Apply noise to each categorical feature by replacing with MASK index
                for j, size in enumerate(actual_cat_sizes):
                    if size <= 0:
                        continue
                    
                    # Get original category indices for feature j
                    orig_cat_indices = x_0_cat_int[:, j].long()
                    
                    # MASK index is the last index (size-1) for each categorical feature
                    mask_index = size - 1
                    
                    # Use keep_mask to decide whether to keep original or use MASK
                    # Shape: [batch_size]
                    x_t_cat_int[:, j] = torch.where(
                        keep_mask[:, j],
                        orig_cat_indices,
                        torch.full_like(orig_cat_indices, mask_index)
                    )
            else:
                x_t_cat_int = torch.empty((batch_size, 0), dtype=torch.long, device=device)
            
            return x_t_num, x_t_cat_int, x_t_cat_int  # Return x_t_cat_int twice as placeholder for compatibility
        
        else:
            # Original behavior: apply Gaussian noise to the combined tensor
            if noise is None:
                noise = torch.randn_like(x_0)
            
            # Extract the appropriate alpha_cumprod for each t in the batch
            sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1)
            sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
            
            # Apply the formula: x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
            return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_sample(self, model, x_t, t, d_numerical, actual_cat_sizes, guidance_grad=None):
        """
        Reverse step: sample p(x_{t-1} | x_t) using the model
        For t > 0, add noise according to the posterior variance
        For t = 0, just return the prediction (no added noise)
        """
        with torch.no_grad():
            # Get model prediction (estimate of x_0)
            pred_x0_num, pred_x0_cat_flat_logits = model(x_t, t, guidance_grad)
            
            # Process categorical outputs to one-hot
            cat_start = 0
            pred_x0_cat_parts = []
            
            # 检查pred_x0_cat_flat_logits是列表还是tensor
            if isinstance(pred_x0_cat_flat_logits, list):
                # 如果是列表（TabularDiffAE with cat_reconstructor），直接使用每个元素
                for cat_logits in pred_x0_cat_flat_logits:
                    # Apply softmax to get probabilities
                    cat_probs = F.softmax(cat_logits, dim=1)
                    pred_x0_cat_parts.append(cat_probs)
            else:
                # 如果是tensor（原有逻辑），按原方式处理
                for n_cats in actual_cat_sizes:
                    if n_cats > 0:
                        cat_logits = pred_x0_cat_flat_logits[:, cat_start:cat_start + n_cats]
                        # Apply softmax to get probabilities
                        cat_probs = F.softmax(cat_logits, dim=1)
                        pred_x0_cat_parts.append(cat_probs)
                        cat_start += n_cats
            
            # Combine numerical and categorical predictions
            pred_x0_cat = torch.cat(pred_x0_cat_parts, dim=1) if pred_x0_cat_parts else torch.empty(
                (x_t.shape[0], 0), device=x_t.device
            )
            
            x0_pred = torch.cat((pred_x0_num, pred_x0_cat), dim=1)
            
            # Calculate posterior mean (µ_θ(x_t, t))
            # For DDPM, this is: coef1 * x0_pred + coef2 * x_t
            posterior_mean = (
                self.posterior_mean_coef1[t].view(-1, 1) * x0_pred + 
                self.posterior_mean_coef2[t].view(-1, 1) * x_t
            )
            
            posterior_variance = self.posterior_variance[t].view(-1, 1)
            posterior_log_variance = self.posterior_log_variance_clipped[t].view(-1, 1)
            
            # Sample from posterior
            noise = torch.randn_like(x_t) if t[0] > 0 else 0.
            x_t_minus_1 = posterior_mean + torch.exp(0.5 * posterior_log_variance) * noise
            
            return x_t_minus_1
    
    def p_sample_loop(self, model, shape, d_numerical, actual_cat_sizes, guidance_grad=None, verbose=True):
        """
        Full sampling loop: start from x_T ~ N(0, I) and sample backwards to x_0
        """
        device = self.device
        batch_size = shape[0]
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        # Iterate backward through timesteps
        for t_idx in reversed(range(self.num_timesteps)):
            if verbose and t_idx % 100 == 0:
                print(f"Sampling step {t_idx}/{self.num_timesteps}")
            
            t = torch.full((batch_size,), t_idx, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t, d_numerical, actual_cat_sizes, guidance_grad)
        
        # Post-process: split numerical and categorical, apply argmax to categorical
        x_num = x[:, :d_numerical]
        x_cat_probs = x[:, d_numerical:]
        
        # Convert categorical probabilities to integer indices
        cat_indices = []
        cat_start = 0
        for n_cats in actual_cat_sizes:
            if n_cats > 0:
                cat_end = cat_start + n_cats
                # Get the class with highest probability
                cat_idx = torch.argmax(x_cat_probs[:, cat_start:cat_end], dim=1)
                cat_indices.append(cat_idx)
                cat_start = cat_end
        
        x_cat_int = torch.stack(cat_indices, dim=1) if cat_indices else None
        
        return x_num, x_cat_int

# --- Training Function ---
def compute_diffae_loss(pred_x0_num, pred_x0_cat_flat_logits, 
                       true_x0_num, true_x0_cat_int,
                       actual_cat_sizes, device, lambda_cat=1.0):
    """
    Compute combined loss for the predictions vs. true values
    """
    # MSE Loss for numerical features
    loss_num = torch.tensor(0.0, device=device)
    if pred_x0_num.shape[1] > 0 and true_x0_num is not None:
        loss_num = F.mse_loss(pred_x0_num, true_x0_num)
    
    # Cross-Entropy Loss for categorical features
    loss_cat = torch.tensor(0.0, device=device)
    
    # 检查pred_x0_cat_flat_logits是列表还是tensor
    if isinstance(pred_x0_cat_flat_logits, list):
        # 如果是列表（TabularDiffAE with cat_reconstructor），每个元素对应一个分类特征
        if len(pred_x0_cat_flat_logits) > 0 and true_x0_cat_int is not None:
            cat_feature_count = 0
            for i, logits_i in enumerate(pred_x0_cat_flat_logits):
                if i < true_x0_cat_int.shape[1]:  # 确保索引有效
                    labels_i = true_x0_cat_int[:, i]
                    if logits_i.shape[0] > 0 and labels_i.shape[0] == logits_i.shape[0]:
                        loss_cat += F.cross_entropy(logits_i, labels_i)
                        cat_feature_count += 1
            if cat_feature_count > 0:
                loss_cat /= cat_feature_count  # Average CE loss
    else:
        # 如果是tensor（原有逻辑），按原方式处理
        if pred_x0_cat_flat_logits.shape[1] > 0 and true_x0_cat_int is not None:
            start_idx = 0
            cat_feature_count = 0
            for i, n_cats in enumerate(actual_cat_sizes):
                if n_cats > 0:
                    logits_i = pred_x0_cat_flat_logits[:, start_idx : start_idx + n_cats]
                    labels_i = true_x0_cat_int[:, i]
                    if logits_i.shape[0] > 0 and labels_i.shape[0] == logits_i.shape[0]:
                        loss_cat += F.cross_entropy(logits_i, labels_i)
                        cat_feature_count += 1
                start_idx += n_cats
            if cat_feature_count > 0:
                loss_cat /= cat_feature_count  # Average CE loss
    
    total_loss = loss_num + lambda_cat * loss_cat
    return total_loss, loss_num, loss_cat


def test_diffae(data_config, model, diffusion_utils, test_df, processor, device, d_numerical, 
                actual_cat_sizes, batch_size=32, t_eval=10, verbose=False, show_modif_info=True, guidance_grad=None):
    """
    Test the DiffAE model on test data
    """
    model.eval()
    
    # Process test data
    test_x_num, test_x_cat, test_y = processor.transform(test_df)
    
    # Convert to tensors
    if test_x_num is not None:
        test_x_num = test_x_num.clone().detach().float().to(device)
    else:
        test_x_num = torch.empty((len(test_y), 0), device=device)
    
    if test_x_cat is not None:
        test_x_cat = test_x_cat.clone().detach().long().to(device)
    else:
        test_x_cat = torch.empty((len(test_y), 0), dtype=torch.long, device=device)
    
    # Create one-hot encodings for categorical features
    test_x_cat_onehot_list = []
    if test_x_cat.shape[1] > 0:
        for i, n_cats in enumerate(actual_cat_sizes):
            if n_cats > 0:
                valid_indices = test_x_cat[:, i].clamp(0, n_cats - 1)
                onehot = F.one_hot(valid_indices, num_classes=n_cats).float()
                test_x_cat_onehot_list.append(onehot)
        test_x_cat_onehot = torch.cat(test_x_cat_onehot_list, dim=1)
    else:
        test_x_cat_onehot = torch.empty((test_x_num.shape[0], 0), device=device)
    
    # Combine numerical and one-hot categorical features
    if test_x_num.shape[1] == 0:
        test_x_combined = test_x_cat_onehot
    elif test_x_cat_onehot.shape[1] == 0:
        test_x_combined = test_x_num
    else:
        test_x_combined = torch.cat((test_x_num, test_x_cat_onehot), dim=1)
    
    # Create time tensor for evaluation
    t_tensor = torch.full((test_x_combined.shape[0],), t_eval, device=device, dtype=torch.long)
    
    # Add noise to the data using separate noise processes
    if verbose:
        print(f"Adding noise at level t={t_eval}...")
    
    # Make sure we have marginal distributions for categorical features
    if diffusion_utils.cat_marginals is None and test_x_cat is not None and test_x_cat.shape[1] > 0:
        print("Warning: Categorical marginals not computed. Using default noise process.")
        noisy_x = diffusion_utils.q_sample(test_x_combined, t_tensor)
    else:
        # Use separate noise processes
        noise_num = torch.randn_like(test_x_num) if test_x_num is not None and test_x_num.shape[1] > 0 else None
        noisy_x = diffusion_utils.q_sample(
            x_0=None,
            t=t_tensor,
            noise=noise_num,
            x_0_num=test_x_num,
            x_0_cat_int=test_x_cat,
            actual_cat_sizes=actual_cat_sizes
        )
    
    if verbose:
        print("Denoising data...")
    
    # Process in batches to avoid OOM
    num_samples = noisy_x.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    denoised_num_list = []
    denoised_cat_int_list = []
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)
        
        batch_noisy = noisy_x[start_idx:end_idx]
        batch_t = t_tensor[start_idx:end_idx]
        
        # Denoise the batch
        with torch.no_grad():
            pred_x0_num, pred_x0_cat_flat_logits = model(batch_noisy, batch_t, guidance_grad)
        
        # Process categorical predictions
        cat_indices = []
        
        # 检查pred_x0_cat_flat_logits是列表还是tensor
        if isinstance(pred_x0_cat_flat_logits, list):
            # 如果是列表（TabularDiffAE with cat_reconstructor），直接使用每个元素
            for cat_logits in pred_x0_cat_flat_logits:
                # Apply softmax and get most likely class
                cat_probs = F.softmax(cat_logits, dim=1)
                cat_idx = torch.argmax(cat_probs, dim=1)
                cat_indices.append(cat_idx)
        else:
            # 如果是tensor（原有逻辑），按原方式处理
            cat_start = 0
            for n_cats in actual_cat_sizes:
                if n_cats > 0:
                    cat_logits = pred_x0_cat_flat_logits[:, cat_start:cat_start + n_cats]
                    # Apply softmax and get most likely class
                    cat_probs = F.softmax(cat_logits, dim=1)
                    cat_idx = torch.argmax(cat_probs, dim=1)
                    cat_indices.append(cat_idx)
                    cat_start += n_cats
        
        batch_denoised_cat_int = torch.stack(cat_indices, dim=1) if cat_indices else None
        
        # Add batch results to lists
        denoised_num_list.append(pred_x0_num.cpu().numpy())
        if batch_denoised_cat_int is not None:
            denoised_cat_int_list.append(batch_denoised_cat_int.cpu().numpy())
    
    # Combine batch results
    denoised_num = np.vstack(denoised_num_list) if denoised_num_list else None
    denoised_cat_int = np.vstack(denoised_cat_int_list) if denoised_cat_int_list else None
    
    # Inverse transform to get back to original data format
    denoised_df = processor.inverse_transform(denoised_num, denoised_cat_int)
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
    
    # 检查pred_x0_cat_flat_logits是列表还是tensor
    if isinstance(pred_x0_cat_flat_logits, list):
        # 如果是列表（TabularDiffAE with cat_reconstructor），直接使用每个元素
        for cat_logits in pred_x0_cat_flat_logits:
            # Apply softmax and get most likely class
            cat_probs = F.softmax(cat_logits, dim=1)
            cat_idx = torch.argmax(cat_probs, dim=1)
            cat_indices.append(cat_idx)
    else:
        # 如果是tensor（原有逻辑），按原方式处理
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

def contrastive_loss(features, labels, temperature=0.5, margin=1.0):
    """
    Compute contrastive loss, making similar samples closer and dissimilar samples farther apart.
    
    Args:
        features (torch.Tensor): Feature vectors, shape [batch_size, feature_dim]
        labels (torch.Tensor): Class labels, shape [batch_size]
        temperature (float): Temperature parameter, controls the smoothness of similarity distribution
        margin (float): Margin parameter, minimum distance between dissimilar samples
        
    Returns:
        torch.Tensor: Contrastive loss value
    """
    batch_size = features.shape[0]
    if batch_size <= 1:
        return torch.tensor(0.0, device=features.device)
    
    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T) / temperature
    
    labels = labels.view(-1, 1)
    mask_pos = (labels == labels.T).float()
    mask_neg = (labels != labels.T).float()
    
    mask_diag = torch.eye(batch_size, device=features.device)
    mask_pos = mask_pos * (1 - mask_diag)
    
    pos_loss = -torch.log(torch.exp(similarity_matrix) + 1e-8) * mask_pos
    pos_loss = pos_loss.sum() / mask_pos.sum().clamp(min=1e-8)
    
    neg_loss = torch.clamp(similarity_matrix - margin, min=0) * mask_neg
    neg_loss = neg_loss.sum() / mask_neg.sum().clamp(min=1e-8)
    
    return pos_loss + neg_loss

def compute_embedding_diffusion_loss(
    pred_e0_num: torch.Tensor,
    pred_e0_cat: List[torch.Tensor],
    x_num: torch.Tensor,
    x_cat: torch.Tensor,
    emb_mask: torch.Tensor,
    labels: torch.Tensor = None,
    contrastive_weight: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the diffusion loss for numerical and categorical features.

    Args:
        pred_e0_num (torch.Tensor): Predicted numerical embeddings/features.
                                    Shape: (batch_size, num_numerical_features).
        pred_e0_cat (List[torch.Tensor]): List of predicted categorical feature logits.
                                          Each element pred_e0_cat[i] has shape
                                          (batch_size, num_classes_for_ith_cat_feature).
        x_num (torch.Tensor): True numerical features.
                              Shape: (batch_size, num_numerical_features).
        x_cat (torch.Tensor): True categorical features (label encoded).
                              Shape: (batch_size, num_categorical_features).
        emb_mask (torch.Tensor): Mask indicating observed (1) vs. missing (0) features.
                                 Applied to the concatenated original features (numerical first, then categorical).
                                 Shape: (batch_size, num_numerical_features + num_categorical_features).
        labels (torch.Tensor, optional): Sample labels.
        contrastive_weight (float, optional): Contrastive loss weight.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - total_loss: The combined loss.
            - num_loss: The loss from numerical features.
            - cat_loss: The loss from categorical features.
    """
    device = pred_e0_num.device if pred_e0_num.nelement() > 0 else \
             (pred_e0_cat[0].device if pred_e0_cat else x_cat.device) # Handle empty tensors gracefully

    batch_size = emb_mask.shape[0]
    num_numerical_features = x_num.shape[1]
    num_categorical_features = x_cat.shape[1]

    if num_numerical_features > 0:
        mask_num = emb_mask[:, :num_numerical_features]
    else:
        mask_num = torch.empty(batch_size, 0, dtype=emb_mask.dtype, device=device)


    if num_categorical_features > 0:
        mask_cat = emb_mask[:, num_numerical_features:]
    else:
        mask_cat = torch.empty(batch_size, 0, dtype=emb_mask.dtype, device=device)


    # Calculate Numerical Loss (Masked MSE)
    num_loss = torch.tensor(0.0, device=device)
    if num_numerical_features > 0 and mask_num.sum() > 0:
        squared_errors = (pred_e0_num - x_num)**2
        masked_squared_errors_sum = (squared_errors * mask_num).sum()
        num_loss = masked_squared_errors_sum / mask_num.sum().clamp(min=1e-9) # clamp to avoid div by zero

    # Calculate Categorical Loss (Sum of Cross-Entropies)
    cat_loss = torch.tensor(0.0, device=device)
    if num_categorical_features > 0 and pred_e0_cat:
        num_cat_features_with_loss = 0
        for i in range(num_categorical_features):
            pred_logits_i = pred_e0_cat[i]
            true_labels_i = x_cat[:, i]
            feature_mask_i = mask_cat[:, i] 

            valid_indices = feature_mask_i > 0
            
            if valid_indices.sum() > 0:
                # Ensure true_labels are long type for cross_entropy
                # pred_logits should be float
                current_ce_loss = F.cross_entropy(
                    pred_logits_i[valid_indices],
                    true_labels_i[valid_indices].long(),
                    reduction='mean'  # Averages over the valid samples for this specific feature
                )
                cat_loss += current_ce_loss
                num_cat_features_with_loss +=1
        
        # Optional: Average the sum of CE losses over the number of categorical features that had valid samples
        # This is a design choice. Summing mean CE losses is also common.
        if num_cat_features_with_loss > 0:
           cat_loss /= num_cat_features_with_loss

    # contrast_loss = torch.tensor(0.0, device=device)
    # if labels is not None and batch_size > 1:
    #     # 使用预测的数值特征和类别特征的组合作为样本表示
    #     # 对于类别特征，我们取每个类别预测的最大概率值对应的特征
    #     features = pred_e0_num
        
    #     if num_categorical_features > 0 and pred_e0_cat:
    #         cat_features = []
    #         for i in range(num_categorical_features):
    #             pred_logits_i = pred_e0_cat[i]
    #             cat_probs = F.softmax(pred_logits_i, dim=1)
    #             cat_features.append(cat_probs)
            
    #         cat_features = torch.cat(cat_features, dim=1)
    #         features = torch.cat([features, cat_features], dim=1)
        
        # contrast_loss = contrastive_loss(features, labels)

    total_loss = (num_loss + cat_loss) * 0.5

    return total_loss, num_loss, cat_loss


def train_embedding_diffae(
    model: Union[TabularAE, TabularUNet],
    feature_embedder: FeatureEmbedder,
    diffusion: DiffusionUtils,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epochs: int,
    actual_cat_sizes: List[int],
    log_interval: int = 10,
    label_mask_rate: float = 0.2,
    num_classes: int = 0,
    masked_label_value: int = 0,
    contrastive_weight: float = 0.1  # 添加对比损失权重参数
) -> List[float]:
    """
    Train the diffusion model using feature embeddings with label conditioning.
    
    Args:
        model: Diffusion model (TabularDiffAE or TabularUNet)
        feature_embedder: Feature embedder model
        diffusion: Diffusion utilities
        dataloader: DataLoader containing the training data
        optimizer: Optimizer for training
        device: Device to train on
        epochs: Number of training epochs
        actual_cat_sizes: List of category sizes
        log_interval: How often to log training progress
        label_mask_rate: Probability of masking a label during training
        num_classes: Number of label classes
        masked_label_value: Value to use for masked labels
        contrastive_weight: 对比损失的权重系数
        
    Returns:
        List of losses during training
    """
    losses = []
    diffusion.compute_cat_marginals(dataloader, actual_cat_sizes)
    
    epoch_times = []
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        epoch_losses = []
        
        for batch in dataloader:
            x_num, x_cat, labels, num_mask, cat_mask = batch
            
            x_num = x_num.to(device) if x_num is not None else None
            x_cat = x_cat.to(device) if x_cat is not None else None
            num_mask = num_mask.to(device)
            cat_mask = cat_mask.to(device)
            labels = labels.to(device)
            
            batch_size = x_num.shape[0] if x_num is not None else x_cat.shape[0]
            
            # 准备标签条件
            if num_classes > 0:
                # 以label_mask_rate的概率掩码标签
                mask = torch.rand(batch_size, device=device) < label_mask_rate
                
                # 创建条件标签：对于被掩码的位置，使用masked_label_value
                y_cond = labels.clone()
                if mask.sum() > 0:
                    y_cond[mask] = masked_label_value
            else:
                y_cond = None
            
            feature_mask = torch.cat([num_mask, cat_mask], dim=1)
            t = torch.randint(0, diffusion.num_timesteps, (batch_size,), device=device)
            
            num_noise, cat_noise, combine_noise = diffusion.q_sample(
                x_0=None, 
                t=t, 
                noise=None, 
                x_0_num=x_num, 
                x_0_cat_int=x_cat, 
                actual_cat_sizes=actual_cat_sizes
            )
            
            e_t, _ = feature_embedder(num_noise, cat_noise)
            
            emb_mask = feature_mask
            
            optimizer.zero_grad()
            # 将标签条件传递给模型
            pred_e0_num, pred_e0_cat = model(e_t, t, y_labels=y_cond)
            total_loss, num_loss, cat_loss = compute_embedding_diffusion_loss(
                pred_e0_num, pred_e0_cat, x_num, x_cat, emb_mask, 
                labels=labels, contrastive_weight=contrastive_weight  # 传递标签和对比损失权重
            )
            
            total_loss.backward()
            optimizer.step()
            
            epoch_losses.append(total_loss.item())
        
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_epoch_loss)
        
        # 计算当前epoch所用时间
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_time)
        
        if (epoch + 1) % log_interval == 0 or epoch == epochs - 1:
            avg_epoch_time = sum(epoch_times) / len(epoch_times)
            remaining_epochs = epochs - (epoch + 1)
            est_remaining_time = remaining_epochs * avg_epoch_time
            
            avg_time_str = time.strftime("%H:%M:%S", time.gmtime(avg_epoch_time))
            remaining_time_str = time.strftime("%H:%M:%S", time.gmtime(est_remaining_time))
            
            print(f"Diffusion Training Epoch {epoch+1}/{epochs}, Avg Loss: {avg_epoch_loss:.4f}, "
                  f"Avg Time/Epoch: {avg_time_str}, Est. Remaining: {remaining_time_str}")
    
    return losses


def train_embedding_direct(
    model: Union[TabularAE, TabularUNet],
    feature_embedder: FeatureEmbedder,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epochs: int,
    log_interval: int = 10,
    label_mask_rate: float = 0.2,
    num_classes: int = 0,
    masked_label_value: int = 0
) -> List[float]:
    """
    训练模型使用直接重建（不使用扩散过程）
    
    Args:
        model: 预测模型 (TabularDiffAE或TabularUNet)
        feature_embedder: 特征嵌入器模型
        dataloader: 包含训练数据的DataLoader
        optimizer: 训练用的优化器
        device: 训练设备
        epochs: 训练轮数
        log_interval: 日志输出间隔
        label_mask_rate: 训练时掩码标签的概率
        num_classes: 标签类别数
        masked_label_value: 掩码标签使用的值
        
    Returns:
        训练过程中的损失列表
    """
    losses = []
    
    epoch_times = []
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        epoch_losses = []
        
        for batch in dataloader:
            x_num, x_cat, labels, num_mask, cat_mask = batch
            
            x_num = x_num.to(device) if x_num is not None else None
            x_cat = x_cat.to(device) if x_cat is not None else None
            num_mask = num_mask.to(device)
            cat_mask = cat_mask.to(device)
            labels = labels.to(device)
            
            batch_size = x_num.shape[0] if x_num is not None and x_num.shape[0] > 0 else x_cat.shape[0]
            
            # 准备标签条件
            if num_classes > 0:
                # 以label_mask_rate的概率掩码标签
                mask = torch.rand(batch_size, device=device) < label_mask_rate
                
                # 创建条件标签：对于被掩码的位置，使用masked_label_value
                y_cond = labels.clone()
                if mask.sum() > 0:
                    y_cond[mask] = masked_label_value
            else:
                y_cond = None
            
            feature_mask = torch.cat([num_mask, cat_mask], dim=1)
            e_input, _ = feature_embedder(x_num, x_cat)
            
            emb_mask = feature_mask
            emb_dim = e_input.shape[1]
            feature_emb_dim = emb_dim // feature_mask.shape[1] if feature_mask.shape[1] > 0 else emb_dim
            e_mask = feature_mask.unsqueeze(2).expand(-1, -1, feature_emb_dim).reshape(batch_size, -1).float()

            
            optimizer.zero_grad()
            
            # 使用t=0进行直接预测（无扩散）
            t = torch.zeros((batch_size, feature_mask.shape[1]), device=device, dtype=torch.long)
            
            # 将标签条件传递给模型
            pred_e0_num, pred_e0_cat = model(torch.cat([e_input, e_mask], dim=1), t, y_labels=y_cond)
            
            # 计算重建损失（在嵌入空间和原始特征空间之间）
            total_loss, num_loss, cat_loss = compute_embedding_diffusion_loss(
                pred_e0_num, pred_e0_cat, x_num, x_cat, emb_mask,
                labels=labels if num_classes > 0 else None, contrastive_weight=0.0
            )
            
            total_loss.backward()
            optimizer.step()
            
            epoch_losses.append(total_loss.item())
        
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_epoch_loss)
        
        # 计算当前epoch所用时间
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_time)
        
        if (epoch + 1) % log_interval == 0 or epoch == epochs - 1:
            avg_epoch_time = sum(epoch_times) / len(epoch_times)
            remaining_epochs = epochs - (epoch + 1)
            est_remaining_time = remaining_epochs * avg_epoch_time
            
            avg_time_str = time.strftime("%H:%M:%S", time.gmtime(avg_epoch_time))
            remaining_time_str = time.strftime("%H:%M:%S", time.gmtime(est_remaining_time))
            
            print(f"Direct Training Epoch {epoch+1}/{epochs}, Avg Loss: {avg_epoch_loss:.4f}, "
                  f"Avg Time/Epoch: {avg_time_str}, Est. Remaining: {remaining_time_str}")
    
    return losses

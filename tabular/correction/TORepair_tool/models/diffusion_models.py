import torch.nn.functional as F
import torch.nn as nn
import torch

# --- Time Embedding ---
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
        emb = self.time_emb(time.float())
        return self.mlp(emb)


# --- Transformer Block ---
class TransformerBlock(nn.Module):
    """
    A single Transformer block with Multi-Head Self-Attention and Feed-Forward layers.
    Includes Layer Normalization, Residual Connections, and conditioning on time/guidance.
    """
    def __init__(self, embed_dim, num_heads, ff_dim, time_emb_dim, guidance_dim, dropout=0.1):
        """
        Args:
            embed_dim (int): The embedding dimension of the input/output.
            num_heads (int): Number of attention heads.
            ff_dim (int): Dimension of the feed-forward layer.
            time_emb_dim (int): Dimension of the projected time embedding.
            guidance_dim (int): Dimension of the projected guidance embedding.
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

        # Projection layers for time and guidance (applied before attention/FFN)
        # These project the *already processed* time/guidance embeddings to embed_dim
        self.time_proj = nn.Linear(time_emb_dim, embed_dim)
        self.guidance_proj = nn.Linear(guidance_dim, embed_dim) if guidance_dim > 0 else None

        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, t_emb, guidance_emb=None):
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor. Shape [batch_size, seq_len, embed_dim].
                               For tabular data, seq_len is typically 1.
            t_emb (torch.Tensor): Processed time embedding. Shape [batch_size, time_emb_dim].
            guidance_emb (torch.Tensor, optional): Processed guidance embedding.
                                                  Shape [batch_size, guidance_dim]. Defaults to None.

        Returns:
            torch.Tensor: Output tensor. Shape [batch_size, seq_len, embed_dim].
        """
        # --- Conditioning and First Residual Connection ---
        residual = x
        x_cond = x + self.time_proj(t_emb).unsqueeze(1) # Add time conditioning (broadcast along seq_len=1)

        if guidance_emb is not None and self.guidance_proj is not None:
             x_cond = x_cond + self.guidance_proj(guidance_emb).unsqueeze(1) # Add guidance conditioning

        x_norm1 = self.norm1(x_cond)

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


# --- Main Transformer Model ---
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

        # --- Transformer Body ---
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                time_emb_dim=embed_dim, # Expects output from time_mlp
                guidance_dim=embed_dim, # Expects output from guidance_proj_model
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
            torch.Tensor: Predicted denoised data at timestep 0. Shape [batch_size, input_dim].
        """
        batch_size = x_t.shape[0]
        
        # Process time embedding
        t_emb = self.time_mlp(t)
        
        # Process guidance if provided
        guidance_emb = None
        if guidance_grad is not None:
            guidance_emb = self.guidance_proj_model(guidance_grad)
        
        # Project input to embedding space and reshape to [batch, seq_len=1, embed_dim]
        x = self.input_proj(x_t).view(batch_size, 1, self.embed_dim)
        
        # Pass through transformer blocks
        for layer in self.transformer_layers:
            x = layer(x, t_emb, guidance_emb)
        
        # Apply final normalization and project back to input space
        x = self.final_norm(x)
        x = x.view(batch_size, self.embed_dim)
        pred_x0 = self.final_proj(x)
        
        # For tabular data, we need to split into numerical and categorical parts
        if self.d_numerical > 0 and self.d_categorical > 0:
            pred_x0_num = pred_x0[:, :self.d_numerical]
            pred_x0_cat_flat_logits = pred_x0[:, self.d_numerical:]
            return pred_x0_num, pred_x0_cat_flat_logits
        else:
            # If we only have numerical or only categorical features
            return pred_x0, None


# --- U-Net Building Block ---
class UNetLayer(nn.Module):
    """
    A single layer block for the U-Net, processing features, time embedding,
    and optional guidance embedding via addition.
    """
    def __init__(self, input_dim, output_dim, time_emb_dim, guidance_dim, dropout=0.2):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

        # Projection layers for time and guidance embeddings
        self.time_proj = nn.Linear(time_emb_dim, output_dim)
        # Guidance projection is only needed if guidance_dim > 0
        self.guidance_proj = nn.Linear(guidance_dim, output_dim) if guidance_dim > 0 else None

        # Residual connection only if input and output dimensions match
        self.has_residual = (input_dim == output_dim)

    def forward(self, x, t_emb, guidance_emb=None):
        """
        Forward pass for the layer.

        Args:
            x (torch.Tensor): Input tensor. Shape [batch_size, input_dim].
            t_emb (torch.Tensor): Processed time embedding. Shape [batch_size, time_emb_dim].
            guidance_emb (torch.Tensor, optional): Processed guidance embedding.
                                                  Shape [batch_size, guidance_dim]. Defaults to None.

        Returns:
            torch.Tensor: Output tensor. Shape [batch_size, output_dim].
        """
        # Apply linear transformation, layer norm, and activation
        out = self.linear(x)
        out = self.norm(out)
        
        # Add time embedding
        out = out + self.time_proj(t_emb)
        
        # Add guidance embedding if provided and projection exists
        if guidance_emb is not None and self.guidance_proj is not None:
            out = out + self.guidance_proj(guidance_emb)
        
        # Apply activation, dropout, and add residual if dimensions match
        out = self.act(out)
        out = self.dropout(out)
        
        if self.has_residual:
            out = out + x
        
        return out


# --- Main Tabular U-Net Model ---
class TabularUNet(nn.Module):
    """
    U-Net architecture adapted for Tabular Data Diffusion.
    Uses addition for guidance and includes skip connections.
    """
    def __init__(self, input_dim, time_emb_dim=256, hidden_dim=256, latent_dim=64,
                 d_numerical=0, d_categorical=0, actual_cat_sizes=None, dropout=0.2):
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
        """
        super().__init__()
        self.input_dim = input_dim
        self.d_numerical = d_numerical
        self.d_categorical = d_categorical
        self.actual_cat_sizes = actual_cat_sizes
        self.hidden_dim = hidden_dim

        # Time embedding MLP
        self.time_mlp = TimeEmbedMLP(time_emb_dim, hidden_dim)

        # Guidance projection layer (projects raw guidance_grad to hidden_dim)
        # This projected embedding will be added within UNetLayers
        self.guidance_proj = nn.Linear(self.input_dim, hidden_dim)

        # --- Encoder (Down-sampling Path) ---
        # Initial projection from input_dim to hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Layer 1: Takes projected input, adds time and guidance
        # Guidance is added *inside* this layer
        self.enc_layer1 = UNetLayer(hidden_dim, hidden_dim, hidden_dim, hidden_dim, dropout)

        # Layer 2: Reduces dimension, adds time (no guidance added here in this example)
        self.enc_layer2 = UNetLayer(hidden_dim, hidden_dim // 2, hidden_dim, 0, dropout) # guidance_dim=0

        # Bottleneck Layer: Adds time (no guidance added here)
        self.bottleneck = UNetLayer(hidden_dim // 2, latent_dim, hidden_dim, 0, dropout) # guidance_dim=0

        # --- Decoder (Up-sampling Path) ---
        # Layer 1: Takes bottleneck output, adds time and guidance
        # Guidance is added *inside* this layer
        self.dec_layer1 = UNetLayer(latent_dim, hidden_dim // 2, hidden_dim, hidden_dim, dropout)

        # Layer 2: Takes concatenated input (output of dec_layer1 + skip connection from enc_layer2)
        # Adds time (no guidance added here)
        concat_dim_dec2_in = (hidden_dim // 2) + (hidden_dim // 2) # dec1_out + enc2_skip
        self.dec_layer2 = UNetLayer(concat_dim_dec2_in, hidden_dim, hidden_dim, 0, dropout) # guidance_dim=0

        # Final Layer: Takes concatenated input (output of dec_layer2 + skip connection from enc_layer1)
        # Adds time (no guidance added here)
        concat_dim_dec_final_in = hidden_dim + hidden_dim # dec2_out + enc1_skip
        self.dec_final_layer = UNetLayer(concat_dim_dec_final_in, hidden_dim, hidden_dim, 0, dropout) # guidance_dim=0

        # Final projection back to the original input dimension
        self.final_proj = nn.Linear(hidden_dim, input_dim)

        print(f"Initialized TabularUNet (Additive Guidance Mode): input_dim={input_dim}, "
              f"hidden_dim={hidden_dim}, latent_dim={latent_dim}")
        print(f"Decoder Layer 2 input dim: {concat_dim_dec2_in}")
        print(f"Decoder Final Layer input dim: {concat_dim_dec_final_in}")

    def forward(self, x_t, t, guidance_grad=None):
        """
        Forward pass of the TabularUNet model with additive guidance.

        Args:
            x_t (torch.Tensor): Noisy input data at timestep t. Shape [batch_size, input_dim].
            t (torch.Tensor): Timesteps for each sample in the batch. Shape [batch_size].
            guidance_grad (torch.Tensor, optional): Precomputed guidance gradient.
                                                    Shape [batch_size, input_dim]. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Predicted numerical features (pred_x0_num) and
                                               predicted categorical logits (pred_x0_cat_flat_logits).
        """
        batch_size = x_t.shape[0]
        
        # Process time embedding
        t_emb = self.time_mlp(t)
        
        # Process guidance if provided
        guidance_emb = None
        if guidance_grad is not None:
            guidance_emb = self.guidance_proj(guidance_grad)
        
        # Initial projection
        x = self.input_proj(x_t)
        
        # --- Encoder Path ---
        # Layer 1: Apply UNet layer with time and guidance
        enc1 = self.enc_layer1(x, t_emb, guidance_emb)
        
        # Layer 2: Apply UNet layer with time only
        enc2 = self.enc_layer2(enc1, t_emb)
        
        # Bottleneck: Apply UNet layer with time only
        bottleneck = self.bottleneck(enc2, t_emb)
        
        # --- Decoder Path (with skip connections) ---
        # Layer 1: Apply UNet layer with time and guidance
        dec1 = self.dec_layer1(bottleneck, t_emb, guidance_emb)
        
        # Layer 2: Concatenate with skip connection, apply UNet layer with time only
        dec2_input = torch.cat([dec1, enc2], dim=1)
        dec2 = self.dec_layer2(dec2_input, t_emb)
        
        # Final Layer: Concatenate with skip connection, apply UNet layer with time only
        dec_final_input = torch.cat([dec2, enc1], dim=1)
        dec_final = self.dec_final_layer(dec_final_input, t_emb)
        
        # Final projection back to input space
        pred_x0 = self.final_proj(dec_final)
        
        # Split output into numerical and categorical parts
        pred_x0_num = None
        pred_x0_cat_flat_logits = None
        
        if self.d_numerical > 0:
            pred_x0_num = pred_x0[:, :self.d_numerical]
            
        if self.d_categorical > 0:
            pred_x0_cat_flat_logits = pred_x0[:, self.d_numerical:]
            
        return pred_x0_num, pred_x0_cat_flat_logits


# --- DiffAE Layer ---
class DiffAELayer(nn.Module):
    def __init__(self, input_dim, output_dim, time_emb_dim, guidance_dim, dropout=0.2):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        # Project time embedding to be added to layer output
        self.time_proj = nn.Linear(time_emb_dim, output_dim)
        self.guidance_proj = nn.Linear(guidance_dim, output_dim)
        
        # Optional residual connection if dimensions match
        self.has_residual = (input_dim == output_dim)
        
    def forward(self, x, t_emb, guidance_emb=None):
        out = self.linear(x)
        out = self.norm(out)
        
        # Add time condition
        out = out + self.time_proj(t_emb)
        
        # Add guidance if provided
        if guidance_emb is not None:
            out = out + self.guidance_proj(guidance_emb)
            
        out = self.act(out)
        out = self.dropout(out)
        
        # Add residual connection if dimensions match
        if self.has_residual:
            out = out + x
            
        return out


# --- Main DiffAE Model ---
class TabularDiffAE(nn.Module):
    def __init__(self, input_dim, time_emb_dim=256, hidden_dim=256, latent_dim=64, 
                 d_numerical=0, d_categorical=0, actual_cat_sizes=None, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.d_numerical = d_numerical
        self.d_categorical = d_categorical
        self.actual_cat_sizes = actual_cat_sizes
        
        # Time embedding
        self.time_mlp = TimeEmbedMLP(time_emb_dim, hidden_dim)
        
        # Guidance projection layer
        self.guidance_proj = nn.Linear(self.input_dim, hidden_dim)
        
        # Encoder
        self.encoder = nn.ModuleDict({
            'input_proj': nn.Linear(input_dim, hidden_dim),
            'layer1': DiffAELayer(hidden_dim, hidden_dim, hidden_dim, hidden_dim, dropout),
            'layer2': DiffAELayer(hidden_dim, hidden_dim // 2, hidden_dim, hidden_dim, dropout),
            'bottleneck': DiffAELayer(hidden_dim // 2, latent_dim, hidden_dim, hidden_dim, dropout),
        })
        
        # Decoder
        self.decoder = nn.ModuleDict({
            'layer1': DiffAELayer(latent_dim, hidden_dim // 2, hidden_dim, hidden_dim, dropout),
            'layer2': DiffAELayer(hidden_dim // 2, hidden_dim, hidden_dim, hidden_dim, dropout),
            'final_layer': DiffAELayer(hidden_dim, hidden_dim, hidden_dim, hidden_dim, dropout),
            'final_proj': nn.Linear(hidden_dim, input_dim)
        })
        
        print(f"Initialized TabularDiffAE: input_dim={input_dim}, "
              f"hidden_dim={hidden_dim}, latent_dim={latent_dim}")
        
    def forward(self, x_t, t, guidance_grad=None):
        batch_size = x_t.shape[0]
        
        # Process time embedding
        t_emb = self.time_mlp(t)
        
        # Process guidance if provided
        guidance_emb = None
        if guidance_grad is not None:
            guidance_emb = self.guidance_proj(guidance_grad)
        
        # Encode
        x = self.encoder['input_proj'](x_t)
        x = self.encoder['layer1'](x, t_emb, guidance_emb)
        x = self.encoder['layer2'](x, t_emb, guidance_emb)
        x = self.encoder['bottleneck'](x, t_emb, guidance_emb)
        
        # Decode
        x = self.decoder['layer1'](x, t_emb, guidance_emb)
        x = self.decoder['layer2'](x, t_emb, guidance_emb)
        x = self.decoder['final_layer'](x, t_emb, guidance_emb)
        pred_x0 = self.decoder['final_proj'](x)
        
        # Split output into numerical and categorical parts
        if self.d_numerical > 0 and self.d_categorical > 0:
            pred_x0_num = pred_x0[:, :self.d_numerical]
            pred_x0_cat_flat_logits = pred_x0[:, self.d_numerical:]
            return pred_x0_num, pred_x0_cat_flat_logits
        elif self.d_numerical > 0:
            # Only numerical features
            return pred_x0, None
        elif self.d_categorical > 0:
            # Only categorical features
            return None, pred_x0
        else:
            # Should not happen, but for completeness
            return pred_x0, None

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

# --- Diffusion Scheduler Functions ---
def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    """
    Linear beta schedule
    """
    return torch.linspace(beta_start, beta_end, timesteps)


# --- Main Diffusion Utilities Class ---
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
        # Skip if no categorical features
        if not actual_cat_sizes or all(size == 0 for size in actual_cat_sizes):
            self.cat_marginals = None
            return None
        
        # Initialize counters for each category in each feature
        cat_counts = [torch.zeros(size, device=self.device) for size in actual_cat_sizes if size > 0]
        
        # Count occurrences of each category
        for batch in dataloader:
            # Unpack batch - get categorical indices
            _, x_cat, _, _, _ = batch
            
            if x_cat is not None:
                # Update counts for each categorical feature
                cat_idx = 0
                for i, size in enumerate(actual_cat_sizes):
                    if size > 0:  # Skip features with no categories
                        # Count each category
                        for c in range(size):
                            cat_counts[cat_idx][c] += (x_cat[:, i] == c).sum().float()
                        cat_idx += 1
        
        # Convert counts to probabilities (normalize)
        self.cat_marginals = [counts / counts.sum() for counts in cat_counts]
        return self.cat_marginals

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
        if x_0 is not None:
            # Original behavior: combined numerical and one-hot categorical
            batch_size = x_0.shape[0]
            
            # Generate noise if not provided
            if noise is None:
                noise = torch.randn_like(x_0)
                
            # Get diffusion parameters for current timesteps
            sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1)
            sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
            
            # Sample from q(x_t | x_0)
            x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
            
            return x_t
        
        else:
            # New behavior: separate numerical and categorical features
            assert x_0_num is not None or x_0_cat_int is not None, "Either x_0 or (x_0_num, x_0_cat_int) must be provided"
            
            # Handle numerical features
            num_noise = None
            if x_0_num is not None:
                batch_size = x_0_num.shape[0]
                
                # Generate noise for numerical features
                if noise is None:
                    noise = torch.randn_like(x_0_num)
                
                # Get diffusion parameters for current timesteps
                sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1)
                sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod_t[t].view(-1, 1)
                
                # Sample from q(x_t | x_0) for numerical features
                num_noise = sqrt_alphas_cumprod_t * x_0_num + sqrt_one_minus_alphas_cumprod_t * noise
            
            # Handle categorical features
            cat_noise = None
            cat_one_hot_noise = None
            if x_0_cat_int is not None:
                assert actual_cat_sizes is not None, "actual_cat_sizes must be provided for categorical features"
                batch_size = x_0_cat_int.shape[0]
                
                # Initialize array to store noisy categorical indices
                cat_noise = torch.zeros_like(x_0_cat_int)
                
                # Get transition probabilities for each timestep
                transition_probs = []
                for i, t_i in enumerate(t):
                    # Get diffusion parameters for current timestep
                    alpha_cumprod = self.alphas_cumprod[t_i]
                    
                    # For each categorical feature
                    cat_idx = 0
                    for j, n_cats in enumerate(actual_cat_sizes):
                        if n_cats == 0:  # Skip features with no categories
                            continue
                        
                        # Create transition matrix (n_cats x n_cats)
                        # Probability of staying in same category is higher
                        transition_matrix = torch.ones(n_cats, n_cats, device=self.device)
                        
                        # Fill diagonal with higher probability (controlled by alpha_cumprod)
                        # and off-diagonal with uniform probability over other categories
                        transition_matrix = transition_matrix * (1 - alpha_cumprod) / (n_cats - 1)
                        transition_matrix.fill_diagonal_(alpha_cumprod)
                        
                        # Get the original category index for this feature
                        original_cat = x_0_cat_int[i, cat_idx].item()
                        
                        # Sample from transition matrix for this category
                        cat_probs = transition_matrix[original_cat]
                        cat_noise[i, cat_idx] = torch.multinomial(cat_probs, 1).squeeze()
                        
                        cat_idx += 1
                
                # Convert categorical indices to one-hot format if needed
                cat_one_hot = []
                cat_idx = 0
                for j, n_cats in enumerate(actual_cat_sizes):
                    if n_cats == 0:  # Skip features with no categories
                        continue
                    
                    # Convert indices to one-hot
                    one_hot = F.one_hot(cat_noise[:, cat_idx].long(), n_cats).float()
                    cat_one_hot.append(one_hot)
                    cat_idx += 1
                
                cat_one_hot_noise = torch.cat(cat_one_hot, dim=1) if cat_one_hot else None
            
            # Combine if both numerical and categorical are present
            combined_noise = None
            if num_noise is not None and cat_one_hot_noise is not None:
                combined_noise = torch.cat([num_noise, cat_one_hot_noise], dim=1)
                
            return num_noise, cat_noise, combined_noise

    def p_sample(self, model, x_t, t, d_numerical, actual_cat_sizes, guidance_grad=None):
        """
        Reverse step: sample p(x_{t-1} | x_t) using the model
        For t > 0, add noise according to the posterior variance
        For t = 0, just return the prediction (no added noise)
        """
        with torch.no_grad():
            # Get model prediction
            pred_x0_num, pred_x0_cat_flat_logits = model(x_t, t, guidance_grad)
            
            # Extract batch_size
            batch_size = x_t.shape[0]
            
            # Handle t = 0 case (no noise, just return prediction)
            if t[0] == 0:
                # Process categorical predictions to get indices
                cat_indices = []
                if pred_x0_cat_flat_logits is not None:
                    cat_start = 0
                    for n_cats in actual_cat_sizes:
                        if n_cats > 0:
                            cat_logits = pred_x0_cat_flat_logits[:, cat_start:cat_start + n_cats]
                            cat_probs = F.softmax(cat_logits, dim=1)
                            cat_idx = torch.argmax(cat_probs, dim=1)
                            cat_indices.append(cat_idx)
                            cat_start += n_cats
                    
                # Return the prediction directly
                return pred_x0_num, torch.stack(cat_indices, dim=1) if cat_indices else None
            
            # For t > 0, sample p(x_{t-1} | x_t, x_0)
            # Compute posterior mean
            posterior_mean = (
                self.posterior_mean_coef1[t].view(-1, 1) * pred_x0_num +
                self.posterior_mean_coef2[t].view(-1, 1) * x_t[:, :d_numerical]
            )
            
            # Sample from posterior
            noise = torch.randn_like(posterior_mean)
            posterior_variance = self.posterior_variance[t].view(-1, 1)
            posterior_log_variance = self.posterior_log_variance_clipped[t].view(-1, 1)
            
            # Add noise scaled by posterior variance
            x_t_1_num = posterior_mean + torch.exp(0.5 * posterior_log_variance) * noise
            
            # Process categorical predictions to get indices (same as t=0 case)
            cat_indices = []
            if pred_x0_cat_flat_logits is not None:
                cat_start = 0
                for n_cats in actual_cat_sizes:
                    if n_cats > 0:
                        cat_logits = pred_x0_cat_flat_logits[:, cat_start:cat_start + n_cats]
                        cat_probs = F.softmax(cat_logits, dim=1)
                        cat_idx = torch.argmax(cat_probs, dim=1)
                        cat_indices.append(cat_idx)
                        cat_start += n_cats
            
            return x_t_1_num, torch.stack(cat_indices, dim=1) if cat_indices else None

    def p_sample_loop(self, model, shape, d_numerical, actual_cat_sizes, guidance_grad=None, verbose=True):
        """
        Full sampling loop: start from x_T ~ N(0, I) and sample backwards to x_0
        """
        batch_size = shape[0]
        device = self.device
        
        # Start with Gaussian noise (only for numerical features)
        x_num = torch.randn(batch_size, d_numerical, device=device)
        
        # Initialize categorical features with random values
        # We'll create one-hot tensors for model input
        cat_indices = []
        if actual_cat_sizes and any(n_cats > 0 for n_cats in actual_cat_sizes):
            for n_cats in actual_cat_sizes:
                if n_cats > 0:
                    # Sample from marginal distribution if available, otherwise uniform
                    cat_idx = torch.randint(0, n_cats, (batch_size,), device=device)
                    cat_indices.append(cat_idx)
        
        x_cat_int = torch.stack(cat_indices, dim=1) if cat_indices else None
        
        # Prepare for sampling loop
        progress_bar = tqdm(reversed(range(0, self.num_timesteps)), desc="Sampling", total=self.num_timesteps) if verbose else reversed(range(0, self.num_timesteps))
        
        for t in progress_bar:
            # Convert timestep to tensor
            timesteps = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Convert categorical indices to one-hot for model input
            cat_one_hot = []
            if x_cat_int is not None:
                cat_idx = 0
                for i, n_cats in enumerate(actual_cat_sizes):
                    if n_cats > 0:
                        one_hot = F.one_hot(x_cat_int[:, cat_idx].long(), n_cats).float()
                        cat_one_hot.append(one_hot)
                        cat_idx += 1
            
            x_cat_one_hot = torch.cat(cat_one_hot, dim=1) if cat_one_hot else None
            
            # Combine numerical and one-hot categorical for model input
            model_input = torch.cat([x_num, x_cat_one_hot], dim=1) if x_cat_one_hot is not None else x_num
            
            # Sample next state
            x_num, x_cat_int = self.p_sample(model, model_input, timesteps, d_numerical, actual_cat_sizes, guidance_grad)
        
        return x_num, x_cat_int

from datasets.dataset import TabularDataset
from models.diffae.core import TabularAE, DiffusionUtils, train_diffae, denoise_data

# In your main file where you choose which approach to use:
def train_model(model_type, train_df, test_df, processor, config):
    if model_type == "diffae":
        # Setup dimensions
        d_numerical = processor.d_numerical
        d_categorical = processor.d_categorical
        cat_sizes = processor.categories
        actual_cat_sizes = [c - 1 for c in cat_sizes]
        d_onehot_categorical = sum(actual_cat_sizes)
        total_feature_dim = d_numerical + d_onehot_categorical
        
        # Initialize DiffAE and diffusion utils
        model = TabularAE(
            input_dim=total_feature_dim,
            time_emb_dim=config['time_emb_dim'],
            hidden_dim=config['hidden_dim'],
            latent_dim=config['latent_dim'],
            d_numerical=d_numerical,
            d_categorical=d_categorical,
            actual_cat_sizes=actual_cat_sizes,
            dropout=config['dropout']
        ).to(config['device'])
        
        diffusion_utils = DiffusionUtils(
            num_timesteps=config['num_timesteps'],
            schedule=config['schedule'],
            device=config['device']
        )
        
        # Create dataloader from TabularDataset
        train_dataset = TabularDataset(train_df, processor, device=config['device'])
        train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=False)
        
        # Train DiffAE
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
        model = train_diffae(
            model, diffusion_utils, train_dataloader, optimizer, config['device'], 
            config['epochs'], actual_cat_sizes, config['lambda_cat'], config['log_interval']
        )
        
        return model, diffusion_utils
    else:
        # Your existing model training code
        pass

# Then in your test function
def test_model(model_type, model, test_df, processor, config, **kwargs):
    if model_type == "diffae":
        diffusion_utils = kwargs.get('diffusion_utils')
        d_numerical = processor.d_numerical
        cat_sizes = processor.categories
        actual_cat_sizes = [c - 1 for c in cat_sizes]
        
        # Test the denoising capabilities
        denoised_df, original_df = test_diffae(
            model, diffusion_utils, test_df, processor, config['device'],
            d_numerical, actual_cat_sizes, config['batch_size'], 
            t_eval=config['t_eval'], verbose=config['verbose']
        )
        
        # Return denoised data for downstream tasks
        return denoised_df
    else:
        # Your existing model testing code
        pass 
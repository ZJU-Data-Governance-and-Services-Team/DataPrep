"""
Data Processing Helper - Consolidates data preparation and conversion logic
"""
import torch
import numpy as np
import pandas as pd
from models.imputation.standard_pipeline import combine_features


class DataProcessorHelper:
    """Helper class for data processing operations"""
    
    def __init__(self, processor, device):
        self.processor = processor
        self.device = device
    
    def prepare_features_and_labels(self, df, dataset_name=None):
        """
        Unified method for preparing features and labels from DataFrame
        
        Args:
            df: Input DataFrame
            dataset_name: Dataset identifier for caching
            
        Returns:
            tuple: (x_combined, y_tensor, target_values)
        """
        if len(df) == 0:
            return None, None, None
            
        # Convert features
        x_num, x_cat_onehot = self.processor.transform_onehot(df, dataset_name=dataset_name)
        x_combined = combine_features(x_num, x_cat_onehot).to(self.device)
        
        # Process labels
        target_col = self.processor.target_feature[0]
        target_values = df[target_col].values
        
        # Handle nested array values
        if isinstance(target_values[0], (list, np.ndarray)):
            target_values = np.array([
                val[0] if isinstance(val, (list, np.ndarray)) else val 
                for val in target_values
            ])
        
        # Encode labels
        encoded_labels = np.array([
            self.processor.label_encoder.get(val, 0) for val in target_values
        ])
        y_tensor = torch.LongTensor(encoded_labels).to(self.device)
        
        return x_combined, y_tensor, target_values
    
    def separate_numerical_categorical(self, x_combined, d_numerical, actual_cat_sizes):
        """
        Separate combined features back into numerical and categorical components
        
        Args:
            x_combined: Combined feature tensor
            d_numerical: Number of numerical features
            actual_cat_sizes: List of categorical feature sizes
            
        Returns:
            tuple: (x_num, x_cat_onehot, x_cat_indices)
        """
        batch_size = x_combined.shape[0]
        
        if d_numerical > 0 and len(actual_cat_sizes) > 0:
            cat_onehot_dim = sum(actual_cat_sizes)
            x_num = x_combined[:, :d_numerical]
            x_cat_onehot = x_combined[:, d_numerical:d_numerical + cat_onehot_dim]
            x_cat_indices = self._onehot_to_indices(x_cat_onehot, actual_cat_sizes)
        elif d_numerical > 0:
            x_num = x_combined
            x_cat_onehot = None
            x_cat_indices = torch.empty(batch_size, 0, dtype=torch.long, device=self.device)
        else:
            x_num = None
            x_cat_onehot = x_combined
            x_cat_indices = self._onehot_to_indices(x_cat_onehot, actual_cat_sizes)
        
        return x_num, x_cat_onehot, x_cat_indices
    
    def _onehot_to_indices(self, x_cat_onehot, actual_cat_sizes):
        """Convert one-hot encoded categorical features to indices"""
        if x_cat_onehot is None or x_cat_onehot.shape[1] == 0:
            return torch.empty((x_cat_onehot.shape[0] if x_cat_onehot is not None else 0, 0), 
                             dtype=torch.long, device=self.device)
        
        indices = []
        start_idx = 0
        
        for cat_size in actual_cat_sizes:
            end_idx = start_idx + cat_size
            cat_onehot = x_cat_onehot[:, start_idx:end_idx]
            cat_indices = torch.argmax(cat_onehot, dim=1)
            indices.append(cat_indices)
            start_idx = end_idx
        
        return torch.stack(indices, dim=1) if indices else torch.empty(
            (x_cat_onehot.shape[0], 0), dtype=torch.long, device=self.device
        )
    
    def move_to_device(self, *tensors):
        """Move tensors to device, handling None values"""
        result = []
        for tensor in tensors:
            if tensor is not None:
                result.append(tensor.to(self.device))
            else:
                result.append(None)
        return result if len(result) > 1 else result[0]
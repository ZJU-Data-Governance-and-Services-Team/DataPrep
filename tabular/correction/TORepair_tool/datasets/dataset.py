import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from datasets.data_processor import DataProcessor

class TabularDataset(Dataset):
    """Dataset class for tabular data"""
    
    def __init__(self, 
                 df: pd.DataFrame,
                 processor: DataProcessor,
                 neigh_dict=None,
                 device: str = 'cpu',
                 dataset_name: str = None,
                 error_mask_df: pd.DataFrame = None,
                 return_mask: bool = True):
        """
        Initialize dataset
        
        Args:
            df: Input dataframe
            processor: Fitted DataProcessor instance
            neigh_dict: Dictionary mapping each data point to its k-nearest neigh
            device: Device to store tensors
            dataset_name: Dataset name for caching
            error_mask_df: Pre-computed error mask DataFrame (same index as df)
            return_mask: Whether to return error mask in __getitem__
        """
        self.df = df
        self.processor = processor
        self.neigh_dict = neigh_dict
        self.device = device
        self.num_features = processor.num_features
        self.cat_features = processor.cat_features
        self.return_mask = return_mask
        self.error_mask_df = error_mask_df
        
        # Pre-process all data
        self.x_num, self.x_cat, self.labels = processor.transform(df, dataset_name=dataset_name)
        self.num_mask = ~self.error_mask_df[self.num_features].values.astype(bool)
        self.cat_mask = ~self.error_mask_df[self.cat_features].values.astype(bool)
        self.num_mask = torch.FloatTensor(self.num_mask.astype(np.float32))
        self.cat_mask = torch.FloatTensor(self.cat_mask.astype(np.float32))
        
        if self.x_num is not None and self.x_num.shape[0] > 0:
            self.x_num = self.x_num.to(device)
            if self.num_mask is None:
                self.num_mask = torch.ones(self.x_num.shape, device=device)
            else:
                self.num_mask = self.num_mask.to(device)

        if self.x_cat is not None and len(self.x_cat) > 0:
            self.x_cat = self.x_cat.to(device)
            if self.cat_mask is None:
                self.cat_mask = torch.ones(self.x_cat.shape, dtype=torch.float, device=device)
            else:
                self.cat_mask = self.cat_mask.to(device)

        if self.labels is not None:
            self.labels = self.labels.to(device)
        
        # 处理外部错误掩码
        if self.return_mask and self.error_mask_df is not None:
            # 转换为tensor格式的错误掩码
            feature_cols = self.num_features + self.cat_features
            mask_data = self.error_mask_df[feature_cols].values
            self.error_mask_tensor = torch.tensor(mask_data, dtype=torch.bool, device=device)
        else:
            self.error_mask_tensor = None
        

    def __len__(self) -> int:
        """Get dataset length"""
        return len(self.df)
    
    def __getitem__(self, idx: int):
        """Get item by index"""
        if self.neigh_dict is not None:
            base_return = (
                self.x_num[idx] if self.x_num is not None and self.x_num.shape[0] > 0 else torch.tensor([], dtype=torch.float, device=self.device),
                self.x_cat[idx] if self.x_cat is not None and self.x_cat.shape[0] > 0 else torch.tensor([], dtype=torch.long, device=self.device),
                self.labels[idx],
                self.neigh_dict[idx][0],
                self.neigh_dict[idx][1]
            )
        else:
            base_return = (
                self.x_num[idx] if self.x_num is not None and self.x_num.shape[0] > 0 else torch.tensor([], dtype=torch.float, device=self.device),
                self.x_cat[idx] if self.x_cat is not None and self.x_cat.shape[0] > 0 else torch.tensor([], dtype=torch.long, device=self.device),
                self.labels[idx],
                self.num_mask[idx] if self.num_mask is not None and self.num_mask.shape[0] > 0 else torch.tensor([1.0], device=self.device),
                self.cat_mask[idx] if self.cat_mask is not None and self.cat_mask.shape[0] > 0 else torch.tensor([1.0], device=self.device)
            )
        
        # 如果需要返回错误掩码，添加到返回值中
        if self.return_mask and self.error_mask_tensor is not None:
            return base_return + (self.error_mask_tensor[idx],)
        else:
            return base_return
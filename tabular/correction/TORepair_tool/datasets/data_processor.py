import pandas as pd
import numpy as np
import torch
from typing import Tuple, List, Dict, Union
from sklearn.preprocessing import StandardScaler
from utils.utility import calculate_missing_rate, perform_simple_repair
from utils.error_detection import ErrorDetector, calculate_error_rate
from utils.mask_cache_manager import get_global_mask_cache_manager

class DataProcessor:
    """Data processor for handling numerical and categorical features"""
    
    def __init__(self, 
                 numerical_features: List[str] = None,
                 categorical_features: List[str] = None,
                 target_feature: List[str] = None,
                 target: str = None,
                 only_det_missing: bool = True,
                 error_detector: ErrorDetector = None,
                 args=None):
        """
        Initialize the data processor
        
        Args:
            numerical_features: List of numerical column names
            categorical_features: List of categorical column names 
            target: Name of target column
            args: 参数对象
        """
        self.num_features = numerical_features or []
        self.cat_features = categorical_features or []
        self.target = target
        self.target_feature = target_feature or []
        self.only_det_missing = only_det_missing
        if not only_det_missing:
            self.error_detector = error_detector
        self.args = args

        self.num_scaler = StandardScaler()
        self.cat_encoders = {}
        self.categories = []  # include mask token
        self.mask_indices = {}  # Store mask index for each categorical feature
        self.label_encoder = {}  
        self.transformed = False
        
        # 使用全局统一缓存管理器
        self.mask_cache_manager = get_global_mask_cache_manager()
        
    def set_error_mask_cache(self, dataset_name: str, error_mask: pd.DataFrame, df: pd.DataFrame = None):
        """设置错误掩码缓存（兼容性方法）"""
        # 为了兼容现有代码，这里计算dummy的column_rates和overall_rate
        total_cells = error_mask.size
        error_cells_per_column = error_mask.sum()
        column_rates = (error_cells_per_column / error_mask.shape[0]).to_dict()
        overall_rate = error_cells_per_column.sum() / total_cells
        
        self.mask_cache_manager.set_dataset_cache(
            dataset_name=dataset_name,
            error_mask=error_mask,
            column_rates=column_rates,
            overall_rate=overall_rate,
            df=df
        )
    
    def get_error_mask(self, df: pd.DataFrame, dataset_name: str = None) -> pd.DataFrame:
        """
        获取错误掩码，优先使用缓存
        
        Args:
            df: 输入数据框
            dataset_name: 数据集名称（如'train', 'val', 'test'）
            
        Returns:
            错误掩码DataFrame
        """
        # 1. 尝试从统一缓存管理器获取
        cached_mask = self.mask_cache_manager.get_error_mask_pandas(df, dataset_name)
        if cached_mask is not None:
            return cached_mask
        
        # 2. 缓存未命中，重新计算
        if self.only_det_missing:
            error_info = calculate_missing_rate(df)
        else:
            error_info = calculate_error_rate(df, self.error_detector, dataset_name=dataset_name)
        
        error_mask = error_info['error_mask']
        
        # 3. 将结果加入缓存（通过统一缓存管理器）
        self.mask_cache_manager.set_dataset_cache(
            dataset_name=dataset_name or f"temp_{id(df)}",
            error_mask=error_mask,
            column_rates=error_info['column_rates'],
            overall_rate=error_info['overall_rate'],
            df=df
        )
        
        return error_mask
        
    def fit(self, df: pd.DataFrame) -> None:
        """
        Fit the processor on training data
        
        Args:
            df: Input dataframe
        """
        if self.num_features:
            # Use only non-missing values to fit the scaler
            num_data = df[self.num_features].copy()
            self.num_scaler.fit(num_data.fillna(0))
            
        for col in self.cat_features:
            unique_cats = df[col].dropna().unique().tolist()
            # Add special MASK token
            unique_cats.append("MASK")
            self.cat_encoders[col] = {cat: idx for idx, cat in enumerate(unique_cats)}
            self.mask_indices[col] = len(unique_cats) - 1
            self.categories.append(len(unique_cats))
        
        self.cat_sizes_without_mask = [cat_len - 1 for cat_len in self.categories]

        if self.target_feature:
            target_col = self.target_feature[0]  
            unique_labels = df[target_col].dropna().unique().tolist()
            self.label_encoder = {label: idx for idx, label in enumerate(unique_labels)}
            
    def transform(self, df: pd.DataFrame, is_denoised=False, dataset_name: str = None) -> Union[Tuple[torch.Tensor, torch.Tensor], 
                                                                    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Transform data into numerical and categorical tensors with masks for missing values
        
        Args:
            df: Input dataframe
            is_denoised: If True, only return x_num and x_cat without labels and masks
            dataset_name: 数据集名称，用于缓存优化
            
        Returns:
            For is_denoised=True:
                x_num: Tensor of numerical features
                x_cat: Tensor of categorical features
            For is_denoised=False:
                x_num: Tensor of numerical features
                x_cat: Tensor of categorical features
                labels: Tensor of target labels
                num_mask: Mask for numerical features (1=observed, 0=missing)
                cat_mask: Mask for categorical features (1=observed, 0=missing)
        """
        self.transformed = True
        df_placeholder = perform_simple_repair(self.args, df, self.num_features, self.cat_features, dataset_name=dataset_name, placeholder=True)
        
        # num_mask = None
        # df_mask = self.get_error_mask(df, dataset_name)
        # if self.num_features:
        #     # 使用缓存优化的错误掩码获取
        #     combined_missing_mask = df_mask[self.num_features]
            
        #     # 反转掩码（True表示非缺失）
        #     num_mask = ~combined_missing_mask.values
        #     num_mask = torch.FloatTensor(num_mask.astype(np.float32))
            
        # cat_mask = None
        # if self.cat_features:
        #     # 使用缓存优化的错误掩码获取
        #     combined_missing_mask = df_mask[self.cat_features]
            
        #     # 反转掩码（True表示非error）
        #     cat_mask = ~combined_missing_mask.values
        #     cat_mask = torch.FloatTensor(cat_mask.astype(np.float32))
        
        x_num = None
        if self.num_features:
            df_num = df_placeholder[self.num_features].copy()
            x_num = self.num_scaler.transform(df_num)
            x_num = torch.FloatTensor(x_num)

        x_cat = None
        if self.cat_features:
            cat_data = []
            for i, col in enumerate(self.cat_features):
                col_data = df_placeholder[col].copy()
                encoded = col_data.map(lambda val: self.cat_encoders[col].get(val, self.mask_indices[col]))
                cat_data.append(encoded.values)
            x_cat = torch.LongTensor(np.stack(cat_data, axis=1))
        
        if is_denoised:
            return x_num, x_cat
        else:
            # Handle the target
            if self.target_feature:
                target_col = self.target_feature[0]  
                target_values = df[target_col].values
                encoded_labels = np.array([self.label_encoder.get(val, 0) for val in target_values])
                labels = torch.LongTensor(encoded_labels)
            else:
                labels = torch.zeros(len(df), dtype=torch.long)
                
            return x_num, x_cat, labels
    
    def inverse_transform(self, x_num: torch.Tensor = None, x_cat: torch.Tensor = None) -> pd.DataFrame:
        """
        Inverse transform encoded data back to original format
        
        Args:
            x_num: Numerical feature tensor
            x_cat: Categorical feature tensor
            
        Returns:
            Dataframe with original format
        """
        data = {}
        if isinstance(x_num, torch.Tensor):
            if x_num.device.type == "cuda":
                x_num = x_num.cpu().detach().numpy()
            elif x_num.device.type == "cpu":
                x_num = x_num.detach().numpy()
        if isinstance(x_cat, torch.Tensor):
            if x_cat.device.type == "cuda":
                x_cat = x_cat.cpu().detach().numpy()
            elif x_cat.device.type == "cpu":
                x_cat = x_cat.detach().numpy()

        if x_num is not None and x_num.shape[0] > 0 and x_num.size > 0:
            x_num_np = self.num_scaler.inverse_transform(x_num)
            for i, col in enumerate(self.num_features):
                data[col] = x_num_np[:, i]
                
        if x_cat is not None:
            for i, col in enumerate(self.cat_features):
                inv_map = {v: k for k, v in self.cat_encoders[col].items()}
                data[col] = [inv_map[val] for val in x_cat[:, i]]
                # Replace MASK with None for better pandas compatibility
                data[col] = [None if val == "MASK" else val for val in data[col]]
                
        return pd.DataFrame(data)

    def transform_onehot(self, df: pd.DataFrame, dataset_name: str = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transform data into numerical and one-hot encoded categorical tensors
        
        Args:
            df: Input dataframe
            dataset_name: 数据集名称，用于缓存优化
            
        Returns:
            x_num: Tensor of numerical features
            x_cat_onehot: Tensor of one-hot encoded categorical features
        """
        x_num, x_cat = self.transform(df, is_denoised=True, dataset_name=dataset_name)
            
        x_cat_onehot = None
        if x_cat is not None:
            # 预计算总的one-hot维度
            total_cats = sum(self.categories)
            x_cat_onehot = torch.zeros((len(df), total_cats), dtype=torch.float32)
            
            # 使用one_hot并直接填充到预分配的tensor中
            start_idx = 0
            for i, n_cats in enumerate(self.categories):
                end_idx = start_idx + n_cats
                x_cat_onehot[:, start_idx:end_idx] = torch.nn.functional.one_hot(x_cat[:, i], n_cats).float()
                start_idx = end_idx
            
        return x_num, x_cat_onehot
    
    def get_mask_indices(self) -> Dict[str, int]:
        """Get mask indices for all categorical features"""
        return self.mask_indices

    @property 
    def d_numerical(self) -> int:
        """Get number of numerical features"""
        return len(self.num_features)
    
    @property
    def d_categorical(self) -> int:
        """Get number of categorical features"""
        return len(self.cat_features)
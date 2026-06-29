import pandas as pd
import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union
import hashlib
from dataclasses import dataclass


@dataclass
class CachedMaskResult:
    """缓存的掩码结果数据结构"""
    error_mask: pd.DataFrame  # 原始pandas格式的错误掩码
    column_rates: Dict[str, float]  # 每列错误率
    overall_rate: float  # 总体错误率
    detection_details: Dict[str, Any] = None  # 详细检测信息
    tensor_cache: Dict[str, torch.Tensor] = None  # tensor格式缓存
    
    def __post_init__(self):
        if self.tensor_cache is None:
            self.tensor_cache = {}


class MaskCacheManager:
    """
    统一的掩码缓存管理器
    
    负责管理ErrorDetector、DataProcessor和BiLevelTrainer之间的掩码缓存，
    避免重复计算并确保一致性。
    """
    
    def __init__(self):
        # 数据集级别的缓存 - 存储完整数据集的错误检测结果
        self.dataset_cache: Dict[str, CachedMaskResult] = {}
        
        # DataFrame hash到数据集名称的映射
        self.df_hash_mapping: Dict[str, str] = {}
        
        # 批次级别的tensor缓存 - 用于快速访问
        self.batch_tensor_cache: Dict[str, torch.Tensor] = {}
        
        # 配置信息
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _get_df_hash(self, df: pd.DataFrame) -> str:
        """
        获取DataFrame的唯一标识符
        
        Args:
            df: 输入DataFrame
            
        Returns:
            DataFrame的hash字符串
        """
        # 使用shape、列名和数据内容的hash作为唯一标识
        shape_str = str(df.shape)
        columns_str = str(sorted(list(df.columns)))
        
        # 使用更稳定的方式来生成内容hash
        # 将DataFrame转换为字符串表示，处理NaN值
        if len(df) <= 5:
            # 对于小数据集，使用所有行
            content_df = df.fillna('__NAN__')  # 用字符串替换NaN以确保一致性
            content_str = content_df.to_string(index=True)
        else:
            # 对于大数据集，使用前3行和后2行以及一些统计信息
            head_df = df.head(3).fillna('__NAN__')
            tail_df = df.tail(2).fillna('__NAN__')
            content_str = (
                head_df.to_string(index=True) + 
                tail_df.to_string(index=True) +
                str(sorted(list(df.index[:5]))) +
                str(sorted(list(df.index[-3:])))
            )

        # 组合所有信息并计算hash
        combined_str = f"{shape_str}_{columns_str}_{content_str}"
        return hashlib.md5(combined_str.encode()).hexdigest()
    
    def _get_batch_cache_key(self, df: pd.DataFrame, mask_type: str = "error") -> str:
        """
        为批次DataFrame生成缓存键
        
        Args:
            df: 输入DataFrame
            mask_type: 掩码类型（"error", "observation", "missing"等）
            
        Returns:
            批次缓存键
        """
        df_hash = self._get_df_hash(df)
        return f"{mask_type}_{df_hash}"
    
    def set_dataset_cache(self, 
                         dataset_name: str, 
                         error_mask: pd.DataFrame, 
                         column_rates: Dict[str, float], 
                         overall_rate: float,
                         detection_details: Dict[str, Any] = None,
                         df: pd.DataFrame = None) -> None:
        """
        设置数据集级别的缓存
        
        Args:
            dataset_name: 数据集名称（如'train', 'val', 'test'）
            error_mask: 错误掩码DataFrame
            column_rates: 每列错误率
            overall_rate: 总体错误率
            detection_details: 详细检测信息
            df: 原始DataFrame（用于建立hash映射）
        """
        cached_result = CachedMaskResult(
            error_mask=error_mask,
            column_rates=column_rates,
            overall_rate=overall_rate,
            detection_details=detection_details
        )
        
        self.dataset_cache[dataset_name] = cached_result
        
        # 建立DataFrame hash到数据集名称的映射
        if df is not None:
            df_hash = self._get_df_hash(df)
            self.df_hash_mapping[df_hash] = dataset_name
    
    def get_dataset_cache(self, 
                         df: pd.DataFrame, 
                         dataset_name: str = None) -> Optional[CachedMaskResult]:
        """
        获取数据集级别的缓存
        
        Args:
            df: 输入DataFrame
            dataset_name: 数据集名称
            
        Returns:
            缓存的掩码结果，如果未找到返回None
        """
        # 1. 优先使用dataset_name直接查找
        if dataset_name and dataset_name in self.dataset_cache:
            cached_result = self.dataset_cache[dataset_name]
            # 验证行数是否匹配（列数可能不同，因为我们可能只查询特定列）
            if cached_result.error_mask.shape[0] == df.shape[0]:
                return cached_result
        
        # 如果显式提供了 dataset_name 但未命中缓存，则不再尝试通过 df_hash 回退，以防止不同检测方法之间的结果被错误复用
        # 只有当用户未指定 dataset_name 时，才允许通过 hash 进行模糊匹配。
        if dataset_name:
            return None
        
        # 2. 通过DataFrame hash查找
        df_hash = self._get_df_hash(df)
        if df_hash in self.df_hash_mapping:
            mapped_dataset_name = self.df_hash_mapping[df_hash]
            if mapped_dataset_name in self.dataset_cache:
                cached_result = self.dataset_cache[mapped_dataset_name]
                # 再次验证行数
                if cached_result.error_mask.shape[0] == df.shape[0]:
                    return cached_result
        
        return None
    
    def get_error_mask_pandas(self, 
                             df: pd.DataFrame, 
                             dataset_name: str = None) -> Optional[pd.DataFrame]:
        """
        获取pandas格式的错误掩码
        
        Args:
            df: 输入DataFrame
            dataset_name: 数据集名称
            
        Returns:
            错误掩码DataFrame，如果未找到返回None
        """
        cached_result = self.get_dataset_cache(df, dataset_name)
        if cached_result is not None:
            return cached_result.error_mask
        return None
    
    def get_error_mask_tensor(self, 
                             df: pd.DataFrame, 
                             feature_cols: list,
                             actual_cat_sizes: list = None,
                             num_features: list = None,
                             cat_features: list = None,
                             mask_type: str = "error",
                             dataset_name: str = None,
                             device: torch.device = None) -> Optional[torch.Tensor]:
        """
        获取tensor格式的错误掩码
        
        Args:
            df: 输入DataFrame
            feature_cols: 特征列名列表
            actual_cat_sizes: 实际分类特征大小列表
            num_features: 数值特征列名列表
            cat_features: 分类特征列名列表
            mask_type: 掩码类型（"error"表示错误掩码，"observation"表示观测掩码）
            dataset_name: 数据集名称
            device: 目标设备
            
        Returns:
            错误掩码tensor，如果未找到返回None
        """
        if device is None:
            device = self.device
        
        # 生成tensor缓存键
        tensor_cache_key = f"{mask_type}_{self._get_df_hash(df[feature_cols])}"
        
        # 1. 先检查数据集缓存中是否有对应的tensor缓存
        cached_result = self.get_dataset_cache(df, dataset_name)
        if cached_result is not None and tensor_cache_key in cached_result.tensor_cache:
            cached_tensor = cached_result.tensor_cache[tensor_cache_key]
            if cached_tensor.device != device:
                cached_tensor = cached_tensor.to(device)
                cached_result.tensor_cache[tensor_cache_key] = cached_tensor
            return cached_tensor
        
        # 2. 检查批次级别的tensor缓存
        batch_cache_key = self._get_batch_cache_key(df[feature_cols], mask_type)
        if batch_cache_key in self.batch_tensor_cache:
            cached_tensor = self.batch_tensor_cache[batch_cache_key]
            if cached_tensor.device != device:
                cached_tensor = cached_tensor.to(device)
                self.batch_tensor_cache[batch_cache_key] = cached_tensor
            return cached_tensor
        
        # 3. 如果有pandas格式的错误掩码，转换为tensor格式
        if cached_result is not None:
            tensor_mask = self._convert_pandas_mask_to_tensor(
                pandas_mask=cached_result.error_mask[feature_cols],
                mask_type=mask_type,
                actual_cat_sizes=actual_cat_sizes,
                num_features=num_features,
                cat_features=cat_features,
                device=device
            )
            
            # 缓存转换后的tensor
            cached_result.tensor_cache[tensor_cache_key] = tensor_mask
            self.batch_tensor_cache[batch_cache_key] = tensor_mask
            
            return tensor_mask
        
        return None
    
    def set_tensor_cache(self, 
                        df: pd.DataFrame, 
                        tensor_mask: torch.Tensor,
                        mask_type: str = "error") -> None:
        """
        设置tensor格式的缓存
        
        Args:
            df: 输入DataFrame
            tensor_mask: tensor格式的掩码
            mask_type: 掩码类型
        """
        cache_key = self._get_batch_cache_key(df, mask_type)
        self.batch_tensor_cache[cache_key] = tensor_mask
    
    def _convert_pandas_mask_to_tensor(self,
                                     pandas_mask: pd.DataFrame,
                                     mask_type: str,
                                     actual_cat_sizes: list = None,
                                     num_features: list = None,
                                     cat_features: list = None,
                                     device: torch.device = None) -> torch.Tensor:
        """
        将pandas错误掩码转换为tensor格式
        
        Args:
            pandas_mask: pandas DataFrame掩码
            mask_type: 掩码类型（"error"或"observation"）
            actual_cat_sizes: 实际分类特征大小列表
            num_features: 数值特征列名列表
            cat_features: 分类特征列名列表
            device: 目标设备
            
        Returns:
            转换后的tensor掩码
        """
        if device is None:
            device = self.device
        
        if mask_type == "error":
            # 错误掩码：True表示缺失/错误
            return self._build_error_tensor_mask(pandas_mask, actual_cat_sizes, num_features, cat_features, device)
        else:  # observation mask
            # 观测掩码：True表示观测值（非缺失）
            observation_mask = ~pandas_mask  # 反转pandas错误掩码
            return self._build_observation_tensor_mask(observation_mask, actual_cat_sizes, num_features, cat_features, device)
    
    def _build_error_tensor_mask(self, pandas_mask: pd.DataFrame, actual_cat_sizes: list, 
                                num_features: list, cat_features: list, device: torch.device) -> torch.Tensor:
        """构建错误张量掩码"""
        mask_parts = []
        
        # 处理数值特征
        if num_features:
            num_mask = pandas_mask[num_features].values
            num_mask_tensor = torch.tensor(num_mask, dtype=torch.bool, device=device)
            mask_parts.append(num_mask_tensor)
        
        # 处理分类特征（扩展为独热编码形式）
        if cat_features and actual_cat_sizes:
            cat_mask = pandas_mask[cat_features].values
            cat_mask_tensor = torch.tensor(cat_mask, dtype=torch.bool, device=device)
            
            expanded_cat_masks = []
            for i, cat_size in enumerate(actual_cat_sizes):
                feature_mask = cat_mask_tensor[:, i:i+1]
                expanded_feature_mask = feature_mask.repeat(1, cat_size)
                expanded_cat_masks.append(expanded_feature_mask)
            
            if expanded_cat_masks:
                cat_mask_expanded = torch.cat(expanded_cat_masks, dim=1)
                mask_parts.append(cat_mask_expanded)
        
        # 合并所有掩码
        if mask_parts:
            return torch.cat(mask_parts, dim=1)
        else:
            return torch.empty((pandas_mask.shape[0], 0), dtype=torch.bool, device=device)
    
    def _build_observation_tensor_mask(self, observation_mask: pd.DataFrame, actual_cat_sizes: list,
                                      num_features: list, cat_features: list, device: torch.device) -> torch.Tensor:
        """构建观测张量掩码"""
        mask_parts = []
        
        # 处理数值特征
        if num_features:
            num_mask = observation_mask[num_features].values
            num_mask_tensor = torch.tensor(num_mask, dtype=torch.bool, device=device)
            mask_parts.append(num_mask_tensor)
        
        # 处理分类特征
        if cat_features:
            cat_mask = observation_mask[cat_features].values
            cat_mask_tensor = torch.tensor(cat_mask, dtype=torch.bool, device=device)
            mask_parts.append(cat_mask_tensor)
        
        # 合并所有掩码
        if mask_parts:
            return torch.cat(mask_parts, dim=1)
        else:
            return torch.empty((observation_mask.shape[0], 0), dtype=torch.bool, device=device)
    
    def clear_cache(self, dataset_name: str = None) -> None:
        """
        清理缓存
        
        Args:
            dataset_name: 如果指定，只清理特定数据集的缓存；否则清理所有缓存
        """
        if dataset_name:
            if dataset_name in self.dataset_cache:
                del self.dataset_cache[dataset_name]
            # 清理相关的hash映射
            to_remove = [k for k, v in self.df_hash_mapping.items() if v == dataset_name]
            for k in to_remove:
                del self.df_hash_mapping[k]
        else:
            self.dataset_cache.clear()
            self.df_hash_mapping.clear()
            self.batch_tensor_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        return {
            "dataset_cache_size": len(self.dataset_cache),
            "df_hash_mapping_size": len(self.df_hash_mapping),
            "batch_tensor_cache_size": len(self.batch_tensor_cache),
            "dataset_names": list(self.dataset_cache.keys())
        }


# 全局缓存管理器实例
_global_mask_cache_manager = None


def get_global_mask_cache_manager() -> MaskCacheManager:
    """获取全局缓存管理器实例"""
    global _global_mask_cache_manager
    if _global_mask_cache_manager is None:
        _global_mask_cache_manager = MaskCacheManager()
    return _global_mask_cache_manager


def reset_global_mask_cache_manager():
    """重置全局缓存管理器（主要用于测试）"""
    global _global_mask_cache_manager
    _global_mask_cache_manager = None 
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Tuple, Any
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.neighbors import LocalOutlierFactor
try:
    from gensim.models import Word2Vec
except ImportError:
    Word2Vec = None
from collections import Counter
import warnings
from utils.constants import ADDITIONAL_MISSING_VALUES
from utils.mask_cache_manager import get_global_mask_cache_manager

warnings.filterwarnings('ignore')

class ErrorDetector:
    """
    面向下游任务的表格数据错误检测器
    
    支持多种常用的错误检测算法：
    1. 统计异常值检测 (Z-score, IQR)
    2. 孤立森林 (Isolation Forest)
    3. 局部异常因子 (Local Outlier Factor)
    4. DBSCAN聚类异常检测
    5. 基于规则的数据质量检测
    6. 基于模式的分类特征异常检测
    """
    
    def __init__(self, 
                 numerical_features: List[str] = None,
                 categorical_features: List[str] = None,
                 target_feature: List[str] = None,
                 detection_methods: List[str] = None,
                 random_state: int = 42,
                 additional_missing_values: List[str] = ADDITIONAL_MISSING_VALUES,
                 w2v_vector_size: int = 10,
                 contamination='auto'):
        """
        初始化错误检测器
        
        Args:
            numerical_features: 数值特征列名列表
            categorical_features: 分类特征列名列表  
            target_feature: 目标列名列表
            detection_methods: 检测方法列表，默认使用多种方法集成
            contamination: 异常比例，影响某些算法的异常阈值
            random_state: 随机种子
            additional_missing_values: 额外的缺失值表示列表
            w2v_vector_size: Word2Vec向量维度，用于semantic_isolation_forest方法
        """
        self.num_features = numerical_features or []
        self.cat_features = categorical_features or []
        self.target_feature = target_feature or []
        self.contamination = contamination
        self.random_state = random_state
        self.additional_missing_values = additional_missing_values
        self.detection_methods = detection_methods or ['statistical', 'isolation_forest', 'lof', 'rule_based', 'pattern_based', 'missing_values']
        self.w2v_vector_size = w2v_vector_size
        self.verbose = False
        
        # 初始化各种检测器
        self.detectors = {}
        self.fitted = False
        
        # 用于存储训练数据的统计信息
        self.num_stats = {}
        self.cat_patterns = {}
        
        # 新增：全局统计量用于一致性预处理
        self.global_num_medians = {}  # 数值特征的全局中位数
        self.global_num_means = {}    # 数值特征的全局均值
        
        # Word2Vec和语义特征相关属性
        self.w2v_model = None
        self.feature_mapping = {}
        self.scaler = None
        
        # 使用全局统一缓存管理器
        self.mask_cache_manager = get_global_mask_cache_manager()
    
    def _compute_global_statistics(self, df: pd.DataFrame):
        """
        计算全局统计量，用于一致性预处理
        
        Args:
            df: 训练数据框
        """
        print("  - 计算全局统计量...")
        
        # 计算数值特征的全局统计量
        if self.num_features:
            num_data = df[self.num_features]
            for col in self.num_features:
                valid_values = num_data[col].dropna()
                if len(valid_values) > 0:
                    self.global_num_medians[col] = valid_values.median()
                    self.global_num_means[col] = valid_values.mean()
                else:
                    # 如果没有有效值，使用默认值
                    self.global_num_medians[col] = 0.0
                    self.global_num_means[col] = 0.0
        
        print(f"  - 全局中位数: {self.global_num_medians}")
    
    def _fill_missing_with_global_stats(self, num_data: pd.DataFrame, strategy='median') -> pd.DataFrame:
        """
        使用全局统计量填充缺失值，确保一致性
        
        Args:
            num_data: 数值特征数据
            strategy: 填充策略，'median' 或 'mean'
            
        Returns:
            填充后的数值特征数据
        """
        if num_data.empty:
            return num_data
            
        num_filled = num_data.copy()
        
        if strategy == 'median':
            fill_values = self.global_num_medians
        else:  # strategy == 'mean'
            fill_values = self.global_num_means
        
        for col in num_data.columns:
            if col in fill_values:
                num_filled[col] = num_filled[col].fillna(fill_values[col])
            else:
                # 回退到当前数据的统计量（不应该发生）
                num_filled[col] = num_filled[col].fillna(num_data[col].median())
                
        return num_filled
        
    def get_cached_error_results(self, df: pd.DataFrame, dataset_name: str = None) -> Dict[str, Any]:
        """
        获取缓存的错误检测结果
        
        Args:
            df: 输入数据框
            dataset_name: 数据集名称（如'train', 'val', 'test'）
            
        Returns:
            错误检测结果字典，如果缓存未命中则返回None
        """
        cached_result = self.mask_cache_manager.get_dataset_cache(df, dataset_name)
        if cached_result is not None:
            return {
                "error_mask": cached_result.error_mask,
                "column_rates": cached_result.column_rates,
                "overall_rate": cached_result.overall_rate,
                "detection_details": cached_result.detection_details
            }
        return None
    
    def fit(self, df: pd.DataFrame):
        """
        在训练数据上拟合错误检测器
        
        Args:
            df: 训练数据框
        """
        print("拟合错误检测器...")
        
        # 首先计算全局统计量
        self._compute_global_statistics(df)
        
        # 分离数值和分类特征
        num_data = df[self.num_features] if self.num_features else pd.DataFrame()
        cat_data = df[self.cat_features] if self.cat_features else pd.DataFrame()
        
        # 1. 统计异常值检测 - 计算数值特征的统计信息
        if 'statistical' in self.detection_methods and not num_data.empty:
            self.num_stats = {}
            for col in self.num_features:
                values = num_data[col].dropna()
                if len(values) > 0:
                    self.num_stats[col] = {
                        'mean': values.mean(),
                        'std': values.std(),
                        'q1': values.quantile(0.25),
                        'q3': values.quantile(0.75),
                        'iqr': values.quantile(0.75) - values.quantile(0.25),
                        'min': values.min(),
                        'max': values.max()
                    }
        
        # 2. 孤立森林 - 使用全局统计量填充
        if 'isolation_forest' in self.detection_methods and not num_data.empty:
            # 使用全局中位数填补缺失值
            num_filled = self._fill_missing_with_global_stats(num_data, 'median')
            self.detectors['isolation_forest'] = IsolationForest(
                contamination=self.contamination,
                random_state=self.random_state,
            ).fit(num_filled)
        
        # 2b. 基于语义的孤立森林 (新增方法)
        if 'semantic_isolation_forest' in self.detection_methods:
            print("训练基于语义的孤立森林检测器...")
            self._fit_semantic_isolation_forest(df)
        
        # 3. 局部异常因子（支持混合数据类型）
        if 'lof' in self.detection_methods and (not num_data.empty or not cat_data.empty):
            print("训练基于Gower距离的LOF检测器...")
            self._fit_lof_detector(df)
        
        # 4. OPTICS聚类异常检测
        if 'dbscan' in self.detection_methods and (not num_data.empty or not cat_data.empty):
            print("训练OPTICS聚类异常检测器...")
            self._fit_optics_detector(df)
        
        # 5. 基于模式的分类特征检测
        if 'pattern_based' in self.detection_methods and not cat_data.empty:
            self.cat_patterns = {}
            for col in self.cat_features:
                values = cat_data[col].dropna()
                if len(values) > 0:
                    # 计算值的频率分布
                    value_counts = values.value_counts()
                    
                    self.cat_patterns[col] = {
                        'value_frequencies': value_counts.to_dict(),
                        'rare_threshold': 0.01,  # 低于1%频率的值被认为是罕见的
                        'unique_values': set(values.unique()),
                        'most_common': value_counts.index[0] if len(value_counts) > 0 else None
                    }
        
        self.fitted = True
        print(f"错误检测器拟合完成，使用方法: {self.detection_methods}")
    
    def detect_errors(self, df: pd.DataFrame, dataset_name: str = None) -> Dict[str, Any]:
        """
        检测数据中的错误，支持缓存机制以提高性能
        
        Args:
            df: 待检测的数据框
            dataset_name: 数据集名称，用于缓存优化
            
        Returns:
            包含错误检测结果的字典，格式与calculate_missing_rate兼容:
            - "error_mask": 错误掩码DataFrame（True表示错误）
            - "column_rates": 每列错误率字典
            - "overall_rate": 总体错误率
            - "detection_details": 详细检测信息
        """
        if not self.fitted:
            raise ValueError("错误检测器尚未拟合，请先调用fit()方法")
        
        # 尝试从缓存获取结果
        cached_results = self.get_cached_error_results(df, dataset_name)
        if cached_results is not None:
            if self.verbose:
                print(f"使用缓存的错误检测结果，数据形状: {df.shape}")
            return cached_results
        
        if self.verbose:
            print(f"开始错误检测，数据形状: {df.shape}")
        
        # 初始化错误掩码
        error_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
        detection_details = {}
        
        # 分离数值和分类特征
        num_data = df[self.num_features] if self.num_features else pd.DataFrame()
        cat_data = df[self.cat_features] if self.cat_features else pd.DataFrame()
        
        # 1. 统计异常值检测
        if 'statistical' in self.detection_methods and not num_data.empty:
            stat_errors = self._detect_statistical_outliers(num_data)
            error_mask[self.num_features] |= stat_errors
            detection_details['statistical'] = stat_errors.sum().to_dict()
        
        # 2. 孤立森林检测
        if 'isolation_forest' in self.detection_methods and 'isolation_forest' in self.detectors:
            if_errors = self._detect_isolation_forest_outliers(num_data)
            error_mask[self.num_features] |= if_errors
            detection_details['isolation_forest'] = if_errors.sum().to_dict()
        
        # 2b. 基于语义的孤立森林检测 (新增方法)
        if 'semantic_isolation_forest' in self.detection_methods and 'semantic_isolation_forest' in self.detectors:
            semantic_if_errors = self._detect_semantic_isolation_forest_outliers(df)
            error_mask |= semantic_if_errors
            detection_details['semantic_isolation_forest'] = semantic_if_errors.sum().to_dict()
        
        # 3. 局部异常因子检测（支持混合数据类型）
        if 'lof' in self.detection_methods and (not num_data.empty or not cat_data.empty):
            lof_errors = self._detect_lof_outliers(df)
            error_mask |= lof_errors  # 支持所有特征类型
            detection_details['lof'] = lof_errors.sum().to_dict()
        
        # 4. OPTICS聚类异常检测
        if 'dbscan' in self.detection_methods and (not num_data.empty or not cat_data.empty):
            optics_errors = self._detect_optics_outliers(df)
            error_mask |= optics_errors  # 使用|=而不是仅限于数值特征
            detection_details['dbscan'] = optics_errors.sum().to_dict()
        
        # 5. 基于规则的检测
        if 'rule_based' in self.detection_methods:
            rule_errors = self._detect_rule_based_errors(df)
            error_mask |= rule_errors
            detection_details['rule_based'] = rule_errors.sum().to_dict()
        
        # 6. 基于模式的分类特征检测
        if 'pattern_based' in self.detection_methods and not cat_data.empty:
            pattern_errors = self._detect_pattern_based_errors(cat_data)
            error_mask[self.cat_features] |= pattern_errors
            detection_details['pattern_based'] = pattern_errors.sum().to_dict()
        
        # 7. 缺失值检测（集成calculate_missing_rate的逻辑）
        if 'missing_values' in self.detection_methods:
            missing_errors = self._detect_missing_values(df)
            error_mask |= missing_errors
            detection_details['missing_values'] = missing_errors.sum().to_dict()
        
        # 计算错误率
        error_cells_per_column = error_mask.sum()
        total_cells = df.size
        
        if total_cells == 0:
            raise ValueError("无法为空数据框计算错误率")
        
        # 计算列级别错误率
        column_rates = (error_cells_per_column / df.shape[0]).to_dict()
        
        # 计算总体错误率
        overall_rate = error_cells_per_column.sum() / total_cells
        
        result = {
            "error_mask": error_mask,  # 保持与calculate_missing_rate相同的键名
            "column_rates": column_rates,
            "overall_rate": overall_rate,
            "detection_details": detection_details
        }
        
        # 将结果保存到统一缓存管理器
        self.mask_cache_manager.set_dataset_cache(
            dataset_name=dataset_name,
            error_mask=error_mask,
            column_rates=column_rates,
            overall_rate=overall_rate,
            detection_details=detection_details,
            df=df
        )
        
        # 打印检测结果摘要
        if self.verbose:
            print(f"错误检测完成:")
            print(f"  总体错误率: {overall_rate:.4f}")
            print(f"  检测到错误的列:")
            for col, rate in column_rates.items():
                if rate > 0:
                    print(f"    {col}: {rate:.4f}")
        
        return result
    
    def _detect_statistical_outliers(self, num_data: pd.DataFrame) -> pd.DataFrame:
        """基于统计方法检测数值异常值"""
        outliers = pd.DataFrame(False, index=num_data.index, columns=num_data.columns)
        
        for col in num_data.columns:
            if col in self.num_stats:
                stats = self.num_stats[col]
                values = num_data[col]
                
                # Z-score方法（|z| > 3为异常）
                z_scores = np.abs((values - stats['mean']) / (stats['std'] + 1e-8))
                z_outliers = z_scores > 3
                
                # IQR方法
                iqr_lower = stats['q1'] - 1.5 * stats['iqr']
                iqr_upper = stats['q3'] + 1.5 * stats['iqr']
                iqr_outliers = (values < iqr_lower) | (values > iqr_upper)
                
                # 组合两种方法
                outliers[col] = z_outliers | iqr_outliers
        
        return outliers
    
    def _detect_isolation_forest_outliers(self, num_data: pd.DataFrame) -> pd.DataFrame:
        """使用孤立森林检测异常值"""
        outliers = pd.DataFrame(False, index=num_data.index, columns=num_data.columns)
        
        if 'isolation_forest' in self.detectors:
            # 使用全局统计量填补缺失值，确保一致性
            num_filled = self._fill_missing_with_global_stats(num_data, 'median')
            
            # 预测异常值
            predictions = self.detectors['isolation_forest'].predict(num_filled)
            anomalies = predictions == -1
            
            # 对所有数值列标记异常
            for col in num_data.columns:
                outliers[col] = anomalies
        
        return outliers
    
    def _detect_lof_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """使用基于Gower距离的LOF检测异常值（支持混合数据类型）"""
        error_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
        
        if 'lof' not in self.detectors:
            return error_mask
        
        # 分离数值和分类特征
        num_data = df[self.num_features] if self.num_features else pd.DataFrame()
        cat_data = df[self.cat_features] if self.cat_features else pd.DataFrame()
        
        if num_data.empty and cat_data.empty:
            return error_mask
        
        # 预处理待检测数据
        processed_data = self._preprocess_test_data_for_lof(num_data, cat_data)
        
        if processed_data is None:
            return error_mask
        
        # 获取训练时的配置
        lof_config = self.detectors['lof']
        train_processed_data = lof_config['processed_data']
        train_lof_scores = lof_config['negative_outlier_factor']
        n_neighbors = lof_config['n_neighbors']
        
        # 计算与训练数据的距离
        test_distances = self._compute_test_distances(processed_data, train_processed_data, num_data, cat_data)
        
        # 对测试数据执行LOF式的异常检测
        if self.verbose:    
            print("  - 执行基于Gower距离的LOF单元格级别异常检测...")
        cell_level_errors = self._detect_cell_level_lof_anomalies(
            df, test_distances, train_lof_scores, n_neighbors
        )
        
        # 更新错误掩码
        for col in cell_level_errors:
            if col in error_mask.columns:
                error_mask[col] = cell_level_errors[col]
        
        return error_mask

    def _preprocess_test_data_for_lof(self, num_data, cat_data):
        """预处理测试数据用于LOF"""
        processed_parts = []
        
        # 处理数值特征 - 使用全局统计量确保一致性
        if not num_data.empty and hasattr(self, '_lof_num_scaler'):
            num_filled = self._fill_missing_with_global_stats(num_data, 'median')
            scaled_num = self._lof_num_scaler.transform(num_filled)
            processed_parts.append(scaled_num)
        
        # 处理分类特征
        if not cat_data.empty and hasattr(self, '_lof_cat_encoders'):
            cat_filled = cat_data.fillna('MISSING')
            encoded_cat_parts = []
            
            for col in cat_data.columns:
                if col in self._lof_cat_encoders:
                    encoder = self._lof_cat_encoders[col]
                    # 处理未见过的类别
                    encoded_values = []
                    for val in cat_filled[col].astype(str):
                        if val in encoder.classes_:
                            encoded_val = encoder.transform([val])[0]
                        else:
                            # 未见过的类别设为最大值+1
                            encoded_val = len(encoder.classes_)
                    
                        encoded_values.append(encoded_val)
                    
                    encoded_values = np.array(encoded_values)
                    # 标准化
                    max_val = max(len(encoder.classes_), encoded_values.max())
                    if max_val > 0:
                        encoded_values = encoded_values / max_val
                    
                    encoded_cat_parts.append(encoded_values.reshape(-1, 1))
            
            if encoded_cat_parts:
                cat_encoded = np.hstack(encoded_cat_parts)
                processed_parts.append(cat_encoded)
        
        if processed_parts:
            return np.hstack(processed_parts)
        return None

    def _detect_cell_level_lof_anomalies(self, df, distances, train_lof_scores, n_neighbors):
        """
        基于LOF原理进行单元格级别的异常检测
        
        Args:
            df: 待检测的数据框
            distances: 测试数据与训练数据的距离矩阵
            train_lof_scores: 训练数据的LOF得分
            n_neighbors: 邻居数量
            
        Returns:
            字典，包含每个特征的异常标记
        """
        n_test = distances.shape[0]
        cell_errors = {}
        
        # 初始化错误掩码
        for col in self.num_features + self.cat_features:
            if col in df.columns:
                cell_errors[col] = pd.Series(False, index=df.index)
        
        # 计算LOF异常阈值（基于训练数据的LOF得分分布）
        lof_threshold = np.percentile(train_lof_scores, 10)  # 使用10%分位数作为阈值
        
        anomaly_count = 0
        
        for i in range(n_test):
            # 计算测试点的近似LOF得分
            # 找到k个最近邻
            k_nearest_indices = np.argsort(distances[i, :])[:n_neighbors]
            k_nearest_distances = distances[i, k_nearest_indices]
            
            # 计算局部可达密度的近似值
            if len(k_nearest_distances) > 0:
                # 简化的LOF计算：使用k近邻距离的倒数作为密度估计
                local_density = 1.0 / (np.mean(k_nearest_distances) + 1e-8)
                
                # 估计邻居的平均密度（使用训练数据的LOF得分）
                neighbor_lof_scores = train_lof_scores[k_nearest_indices]
                avg_neighbor_density = np.mean(-neighbor_lof_scores)  # LOF得分是负值
                
                # 计算LOF得分的近似值
                if avg_neighbor_density > 0:
                    approx_lof_score = -(avg_neighbor_density / local_density)
                else:
                    approx_lof_score = -1.0
            else:
                approx_lof_score = -2.0  # 极端异常值
            
            # 如果LOF得分低于阈值，认为是异常
            if approx_lof_score < lof_threshold:
                anomaly_count += 1
                df_idx = df.index[i]
                
                # 使用特征重要性分析来确定哪些单元格是异常的
                suspicious_features = self._analyze_feature_contribution_for_lof(
                    df.iloc[i], distances[i, :], k_nearest_distances
                )
                
                # 标记可疑的单元格
                for feature in suspicious_features:
                    if feature in cell_errors:
                        cell_errors[feature].loc[df_idx] = True
        
        if self.verbose:
            print(f"  - LOF单元格级检测: 发现 {anomaly_count} 个异常行")
        
        return cell_errors

    def _analyze_feature_contribution_for_lof(self, data_row, row_distances, k_nearest_distances):
        """
        分析每个特征对LOF异常的贡献度 - 使用"同行评议"策略
        
        Args:
            data_row: 当前数据行
            row_distances: 当前行到所有训练点的距离数组
            k_nearest_distances: k个最近邻的距离
            
        Returns:
            可疑特征列表
        """
        suspicious_features = []
        
        # 获取训练数据和K个最近邻的索引
        if 'lof' not in self.detectors:
            return suspicious_features
            
        lof_config = self.detectors['lof']
        train_num_data = lof_config.get('train_num_data', pd.DataFrame())
        train_cat_data = lof_config.get('train_cat_data', pd.DataFrame())
        
        # 获取K个最近邻的索引
        k_nearest_indices = np.argsort(row_distances)[:len(k_nearest_distances)]
        
        # 数值特征的"同行评议"分析
        if not train_num_data.empty:
            for col in self.num_features:
                if col in data_row.index and col in train_num_data.columns and not pd.isna(data_row[col]):
                    outlier_val = data_row[col]
                    
                    # 获取K个近邻在该特征上的值
                    neighbor_vals = train_num_data.iloc[k_nearest_indices][col].dropna()
                    
                    if len(neighbor_vals) > 0:
                        # 计算局部Z-score
                        neighbor_mean = neighbor_vals.mean()
                        neighbor_std = neighbor_vals.std()
                        
                        if neighbor_std > 1e-6:  # 避免除以零
                            local_z_score = abs((outlier_val - neighbor_mean) / neighbor_std)
                            if local_z_score > 2.5:  # 阈值可调
                                suspicious_features.append(col)
        
        # 分类特征的"同行评议"分析
        if not train_cat_data.empty:
            for col in self.cat_features:
                if col in data_row.index and col in train_cat_data.columns and not pd.isna(data_row[col]):
                    outlier_val = str(data_row[col])
                    
                    # 获取K个近邻在该特征上的值
                    neighbor_vals = train_cat_data.iloc[k_nearest_indices][col].dropna().astype(str)
                    
                    if len(neighbor_vals) > 0:
                        # 计算众数（最常见的值）
                        from scipy.stats import mode
                        try:
                            neighbor_mode = mode(neighbor_vals, keepdims=False).mode
                            # 如果异常点的值与近邻的众数不同，则可疑
                            if outlier_val != neighbor_mode:
                                suspicious_features.append(col)
                        except:
                            # 如果计算众数失败，检查是否存在于近邻值中
                            if outlier_val not in neighbor_vals.values:
                                suspicious_features.append(col)
        
        # 改进的兜底策略：如果没有找到可疑特征，只返回空列表而不是全部特征
        # 这避免了将所有特征都标记为错误的问题
        return suspicious_features
    
    def _detect_rule_based_errors(self, df: pd.DataFrame) -> pd.DataFrame:
        """基于规则的错误检测"""
        errors = pd.DataFrame(False, index=df.index, columns=df.columns)
        
        # 规则1: 检测不合理的数值范围
        for col in self.num_features:
            if col in df.columns:
                values = df[col]
                
                # 年龄相关字段的合理性检查
                if 'age' in col.lower():
                    errors[col] |= (values < 0) | (values > 150)
                
                # 收入相关字段的合理性检查
                if 'income' in col.lower() or 'salary' in col.lower():
                    errors[col] |= (values < 0) | (values > 1000000)  # 假设最大收入100万
                
                # 百分比字段的合理性检查
                if 'percent' in col.lower() or 'rate' in col.lower():
                    errors[col] |= (values < 0) | (values > 100)
                
                # 检测极端值（超出合理范围）
                if col in self.num_stats:
                    stats = self.num_stats[col]
                    # 超出训练数据范围10倍的值被认为是错误
                    range_width = stats['max'] - stats['min']
                    extended_min = stats['min'] - 10 * range_width
                    extended_max = stats['max'] + 10 * range_width
                    errors[col] |= (values < extended_min) | (values > extended_max)
        
        # 规则2: 检测分类特征的不一致性
        for col in self.cat_features:
            if col in df.columns:
                values = df[col].astype(str)
                
                # 检测明显的编码错误（长度异常）
                if col in self.cat_patterns:
                    # 检测异常长的字符串
                    max_length = max(len(str(v)) for v in self.cat_patterns[col]['unique_values'])
                    errors[col] |= values.str.len() > max_length * 3
                    
                    # 检测包含数字的分类字段（如果训练数据中没有数字）
                    has_digit_in_training = any(any(c.isdigit() for c in str(v)) 
                                              for v in self.cat_patterns[col]['unique_values'])
                    if not has_digit_in_training:
                        errors[col] |= values.str.contains(r'\d', regex=True, na=False)
        
        return errors
    
    def _detect_pattern_based_errors(self, cat_data: pd.DataFrame) -> pd.DataFrame:
        """基于模式的分类特征错误检测"""
        errors = pd.DataFrame(False, index=cat_data.index, columns=cat_data.columns)
        
        for col in cat_data.columns:
            if col in self.cat_patterns:
                patterns = self.cat_patterns[col]
                values = cat_data[col].astype(str)
                
                # 检测在训练数据中未出现的值
                unknown_values = ~values.isin(patterns['unique_values'])
                errors[col] |= unknown_values
                
                # 检测极其罕见的值（频率低于阈值）
                value_freq = values.map(patterns['value_frequencies']).fillna(0)
                total_training_samples = sum(patterns['value_frequencies'].values())
                rare_threshold = patterns['rare_threshold'] * total_training_samples
                extremely_rare = value_freq < rare_threshold
                errors[col] |= extremely_rare
        
        return errors
    
    def _detect_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        检测缺失值（集成calculate_missing_rate的逻辑）
        
        考虑NaN、None以及额外指定的缺失值表示
        """
        # 检测NaN和None
        missing_nan_mask = df.isna()
        missing_additional_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
        
        # 检测额外的缺失值表示
        if self.additional_missing_values:
            missing_additional_mask = df.isin(self.additional_missing_values)

        # 检查是否需要处理空白字符串
        check_whitespace = '' in self.additional_missing_values or ' ' in self.additional_missing_values
        
        if check_whitespace:
            for col in df.select_dtypes(include=['object', 'string', 'int64', 'float64']).columns:
                is_whitespace_empty = df[col].astype(str).str.strip().isin(['']) 
                missing_additional_mask[col] = missing_additional_mask[col] | is_whitespace_empty

        # 合并所有缺失值掩码
        combined_missing_mask = missing_nan_mask | missing_additional_mask
        
        return combined_missing_mask

    def _fit_optics_detector(self, df: pd.DataFrame):
        """
        拟合OPTICS聚类异常检测器 - 支持混合数据类型和单元格级别检测
        
        OPTICS算法的优势：
        1. 不需要预先指定eps参数
        2. 通过可达性距离自动识别异常值
        3. 处理数值和分类特征的混合数据
        4. 使用Gower距离处理混合数据类型
        
        Args:
            df: 包含数值和分类特征的完整数据框
        """
        print("  - 准备混合数据类型的OPTICS...")
        
        # 分离数值和分类特征
        num_data = df[self.num_features] if self.num_features else pd.DataFrame()
        cat_data = df[self.cat_features] if self.cat_features else pd.DataFrame()
        
        if num_data.empty and cat_data.empty:
            print("  - 警告: 没有可用于OPTICS的特征")
            return
        
        # 预处理数据
        processed_data = self._preprocess_mixed_data_for_optics(num_data, cat_data)
        
        if processed_data is None:
            return
        
        # 计算混合距离矩阵
        print("  - 计算Gower距离矩阵...")
        distance_matrix = self._compute_gower_distance_matrix(processed_data, num_data, cat_data)
        
        # 自动估计min_samples参数
        min_samples = self._estimate_optics_min_samples(distance_matrix)
        
        print(f"  - OPTICS参数: min_samples={min_samples}")
        
        # 拟合OPTICS模型
        print("  - 训练OPTICS模型...")
        optics_model = OPTICS(
            min_samples=min_samples,
            metric='precomputed',
            cluster_method='dbscan'
        )
        optics_model.fit(distance_matrix)
        
        # 计算异常阈值（基于可达性距离）
        outlier_threshold = self._compute_optics_outlier_threshold(optics_model)
        
        print(f"  - 异常阈值: {outlier_threshold:.3f}")
        
        # 存储OPTICS配置
        self.detectors['dbscan'] = {  # 保持原来的key名以兼容现有代码
            'model': optics_model,
            'outlier_threshold': outlier_threshold,
            'processed_data': processed_data,
            'distance_matrix': distance_matrix,
            'num_scaler': getattr(self, '_optics_num_scaler', None),
            'cat_encoders': getattr(self, '_optics_cat_encoders', None),
            # 新增：存储原始训练数据用于特征贡献分析
            'train_num_data': num_data.copy() if not num_data.empty else pd.DataFrame(),
            'train_cat_data': cat_data.copy() if not cat_data.empty else pd.DataFrame()
        }
        
        print("  - 混合数据OPTICS检测器训练完成")

    def _fit_lof_detector(self, df: pd.DataFrame):
        """
        拟合基于Gower距离的LOF检测器 - 支持混合数据类型
        
        Args:
            df: 包含数值和分类特征的完整数据框
        """
        print("  - 准备混合数据类型的LOF...")
        
        # 分离数值和分类特征
        num_data = df[self.num_features] if self.num_features else pd.DataFrame()
        cat_data = df[self.cat_features] if self.cat_features else pd.DataFrame()
        
        if num_data.empty and cat_data.empty:
            print("  - 警告: 没有可用于LOF的特征")
            return
        
        # 预处理数据（重用已有的预处理方法）
        processed_data = self._preprocess_mixed_data_for_lof(num_data, cat_data)
        
        if processed_data is None:
            return
        
        # 计算混合距离矩阵
        print("  - 计算LOF用的Gower距离矩阵...")
        distance_matrix = self._compute_gower_distance_matrix(processed_data, num_data, cat_data)
        
        # 估计LOF参数
        n_neighbors = self._estimate_lof_n_neighbors(distance_matrix)
        
        print(f"  - LOF参数: n_neighbors={n_neighbors}")
        
        # 使用距离矩阵训练LOF
        lof_model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=self.contamination,
            metric='precomputed',
            novelty=False  # 用于训练数据的异常检测
        )
        
        # 计算LOF得分
        lof_scores = lof_model.fit_predict(distance_matrix)
        negative_outlier_factor = lof_model.negative_outlier_factor_
        
        # 存储LOF配置
        self.detectors['lof'] = {
            'model': lof_model,
            'processed_data': processed_data,
            'distance_matrix': distance_matrix,
            'negative_outlier_factor': negative_outlier_factor,
            'n_neighbors': n_neighbors,
            'num_scaler': getattr(self, '_lof_num_scaler', None),
            'cat_encoders': getattr(self, '_lof_cat_encoders', None),
            # 新增：存储原始训练数据用于特征贡献分析
            'train_num_data': num_data.copy() if not num_data.empty else pd.DataFrame(),
            'train_cat_data': cat_data.copy() if not cat_data.empty else pd.DataFrame()
        }
        
        print("  - 混合数据LOF检测器训练完成")

    def _preprocess_mixed_data_for_lof(self, num_data: pd.DataFrame, cat_data: pd.DataFrame):
        """预处理混合数据用于LOF"""
        processed_parts = []
        
        # 处理数值特征 - 使用全局统计量确保一致性
        if not num_data.empty:
            print("  - 标准化数值特征（LOF）...")
            num_filled = self._fill_missing_with_global_stats(num_data, 'median')
            from sklearn.preprocessing import StandardScaler
            self._lof_num_scaler = StandardScaler()
            scaled_num = self._lof_num_scaler.fit_transform(num_filled)
            processed_parts.append(scaled_num)
        
        # 处理分类特征
        if not cat_data.empty:
            print("  - 编码分类特征（LOF）...")
            cat_filled = cat_data.fillna('MISSING')
            
            # 使用标签编码
            from sklearn.preprocessing import LabelEncoder
            self._lof_cat_encoders = {}
            encoded_cat_parts = []
            
            for col in cat_data.columns:
                encoder = LabelEncoder()
                encoded_values = encoder.fit_transform(cat_filled[col].astype(str))
                # 将编码值标准化到[0,1]范围
                if len(encoder.classes_) > 1:
                    encoded_values = encoded_values / (len(encoder.classes_) - 1)
                encoded_cat_parts.append(encoded_values.reshape(-1, 1))
                self._lof_cat_encoders[col] = encoder
            
            if encoded_cat_parts:
                cat_encoded = np.hstack(encoded_cat_parts)
                processed_parts.append(cat_encoded)
        
        if processed_parts:
            return np.hstack(processed_parts)
        return None

    def _estimate_lof_n_neighbors(self, distance_matrix):
        """估计LOF的n_neighbors参数"""
        n_samples = distance_matrix.shape[0]
        # LOF的n_neighbors通常设为数据点数的平方根或log(n)，但不能太大
        n_neighbors = min(20, max(5, int(np.sqrt(n_samples))))
        # 确保n_neighbors小于样本数
        n_neighbors = min(n_neighbors, n_samples - 1)
        return n_neighbors

    def _preprocess_mixed_data_for_optics(self, num_data: pd.DataFrame, cat_data: pd.DataFrame):
        """预处理混合数据用于OPTICS（重用DBSCAN的逻辑）"""
        return self._preprocess_mixed_data_for_dbscan(num_data, cat_data)

    def _preprocess_mixed_data_for_dbscan(self, num_data: pd.DataFrame, cat_data: pd.DataFrame):
        """预处理混合数据用于DBSCAN"""
        processed_parts = []
        
        # 处理数值特征 - 使用全局统计量确保一致性
        if not num_data.empty:
            print("  - 标准化数值特征...")
            num_filled = self._fill_missing_with_global_stats(num_data, 'median')
            from sklearn.preprocessing import StandardScaler
            self._dbscan_num_scaler = StandardScaler()
            self._optics_num_scaler = StandardScaler()
            scaled_num = self._dbscan_num_scaler.fit_transform(num_filled)
            self._optics_num_scaler.fit(num_filled)  # 同时训练OPTICS的scaler
            processed_parts.append(scaled_num)
        
        # 处理分类特征
        if not cat_data.empty:
            print("  - 编码分类特征...")
            cat_filled = cat_data.fillna('MISSING')
            
            # 使用标签编码
            from sklearn.preprocessing import LabelEncoder
            # 同时支持DBSCAN和OPTICS的编码器
            self._dbscan_cat_encoders = {}
            self._optics_cat_encoders = {}
            encoded_cat_parts = []
            
            for col in cat_data.columns:
                encoder = LabelEncoder()
                encoded_values = encoder.fit_transform(cat_filled[col].astype(str))
                # 将编码值标准化到[0,1]范围
                if len(encoder.classes_) > 1:
                    encoded_values = encoded_values / (len(encoder.classes_) - 1)
                encoded_cat_parts.append(encoded_values.reshape(-1, 1))
                self._dbscan_cat_encoders[col] = encoder
                self._optics_cat_encoders[col] = encoder
            
            if encoded_cat_parts:
                cat_encoded = np.hstack(encoded_cat_parts)
                processed_parts.append(cat_encoded)
        
        if processed_parts:
            return np.hstack(processed_parts)
        return None

    def _compute_gower_distance_matrix(self, processed_data, num_data, cat_data):
        """计算Gower距离矩阵以处理混合数据类型"""
        n_samples = processed_data.shape[0]
        distance_matrix = np.zeros((n_samples, n_samples))
        
        # 计算特征权重
        n_num_features = num_data.shape[1] if not num_data.empty else 0
        n_cat_features = cat_data.shape[1] if not cat_data.empty else 0
        total_features = n_num_features + n_cat_features
        
        if total_features == 0:
            return distance_matrix
        
        # 为每种类型的特征分配权重
        num_weight = n_num_features / total_features if n_num_features > 0 else 0
        cat_weight = n_cat_features / total_features if n_cat_features > 0 else 0
        
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                distance = 0.0
                
                # 数值特征的欧氏距离
                if n_num_features > 0:
                    num_dist = np.linalg.norm(processed_data[i, :n_num_features] - 
                                            processed_data[j, :n_num_features])
                    distance += num_weight * num_dist
                
                # 分类特征的汉明距离
                if n_cat_features > 0:
                    cat_start = n_num_features
                    cat_dist = np.mean(processed_data[i, cat_start:] != processed_data[j, cat_start:])
                    distance += cat_weight * cat_dist
                
                distance_matrix[i, j] = distance_matrix[j, i] = distance
        
        return distance_matrix

    def _estimate_dbscan_parameters(self, distance_matrix):
        """基于距离矩阵估计DBSCAN参数"""
        # 对于每个点，找到第k近邻的距离
        n_samples = distance_matrix.shape[0]
        k = min(5, max(2, n_samples // 10))
        
        k_distances = []
        for i in range(n_samples):
            # 获取第i个点到所有其他点的距离，排序后取第k个
            distances_from_i = np.sort(distance_matrix[i, :])[1:k+1]  # 排除自己(距离为0)
            if len(distances_from_i) >= k:
                k_distances.append(distances_from_i[k-1])
            else:
                k_distances.append(distances_from_i[-1])
        
        # eps设为k距离的75%分位数
        eps = np.percentile(k_distances, 75)
        
        # min_samples通常设为k
        min_samples = k
        
        return eps, min_samples

    def _estimate_optics_min_samples(self, distance_matrix):
        # OPTICS的min_samples通常设为数据维度的2倍或log(n)，这里采用保守策略
        min_samples = 5
        return min_samples

    def _compute_optics_outlier_threshold(self, optics_model):
        """
        基于OPTICS模型的可达性距离计算异常阈值
        
        Args:
            optics_model: 训练好的OPTICS模型
            
        Returns:
            异常阈值，超过此值的点被认为是异常值
        """
        reachability = optics_model.reachability_
        
        # 过滤掉无穷大的值（这些通常是边界点）
        finite_reachability = reachability[np.isfinite(reachability)]
        
        if len(finite_reachability) == 0:
            # 如果所有点都是无穷大，使用一个默认阈值
            return 1.0
        
        # 使用统计方法确定阈值
        # 方法1: 使用上四分位数 + 1.5 * IQR (类似箱线图的异常值检测)
        q75 = np.percentile(finite_reachability, 75)
        q25 = np.percentile(finite_reachability, 25)
        iqr = q75 - q25
        threshold_iqr = q75 + 1.5 * iqr
        
        # 方法2: 使用90%分位数作为阈值
        threshold_percentile = np.percentile(finite_reachability, 90)
        
        # 选择更保守的阈值（较大的那个），以减少误报
        threshold = max(threshold_iqr, threshold_percentile)
        
        return threshold

    def _detect_optics_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        使用OPTICS进行单元格级别的异常检测
        
        Args:
            df: 待检测的数据框
            
        Returns:
            错误掩码DataFrame，True表示错误单元格
        """
        error_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
        
        if 'dbscan' not in self.detectors:  # 保持原key名以兼容
            return error_mask
        
        # 分离数值和分类特征
        num_data = df[self.num_features] if self.num_features else pd.DataFrame()
        cat_data = df[self.cat_features] if self.cat_features else pd.DataFrame()
        
        if num_data.empty and cat_data.empty:
            return error_mask
        
        # 预处理待检测数据
        processed_data = self._preprocess_test_data_for_optics(num_data, cat_data)
        
        if processed_data is None:
            return error_mask
        
        # 获取训练时的配置
        optics_config = self.detectors['dbscan']  # 保持原key名以兼容
        train_processed_data = optics_config['processed_data']
        optics_model = optics_config['model']
        outlier_threshold = optics_config['outlier_threshold']
        
        # 计算与训练数据的距离
        test_distances = self._compute_test_distances(processed_data, train_processed_data, num_data, cat_data)
        
        # 对测试数据执行OPTICS式的异常检测
        if self.verbose:
            print("  - 执行基于OPTICS的单元格级别异常检测...")
        cell_level_errors = self._detect_cell_level_optics_anomalies(
            df, test_distances, optics_model, outlier_threshold
        )
        
        # 更新错误掩码
        for col in cell_level_errors:
            if col in error_mask.columns:
                error_mask[col] = cell_level_errors[col]
        
        return error_mask

    def _detect_dbscan_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        使用增强版DBSCAN进行单元格级别的异常检测 (保留以兼容旧代码)
        
        Args:
            df: 待检测的数据框
            
        Returns:
            错误掩码DataFrame，True表示错误单元格
        """
        # 现在重定向到OPTICS方法
        return self._detect_optics_outliers(df)

    def _preprocess_test_data_for_optics(self, num_data, cat_data):
        """预处理测试数据用于OPTICS（重用DBSCAN的逻辑）"""
        processed_parts = []
        
        # 处理数值特征 - 使用全局统计量确保一致性
        if not num_data.empty and hasattr(self, '_optics_num_scaler'):
            num_filled = self._fill_missing_with_global_stats(num_data, 'median')
            scaled_num = self._optics_num_scaler.transform(num_filled)
            processed_parts.append(scaled_num)
        
        # 处理分类特征
        if not cat_data.empty and hasattr(self, '_optics_cat_encoders'):
            cat_filled = cat_data.fillna('MISSING')
            encoded_cat_parts = []
            
            for col in cat_data.columns:
                if col in self._optics_cat_encoders:
                    encoder = self._optics_cat_encoders[col]
                    # 处理未见过的类别
                    encoded_values = []
                    for val in cat_filled[col].astype(str):
                        if val in encoder.classes_:
                            encoded_val = encoder.transform([val])[0]
                        else:
                            # 未见过的类别设为最大值+1
                            encoded_val = len(encoder.classes_)
                    
                        encoded_values.append(encoded_val)
                    
                    encoded_values = np.array(encoded_values)
                    # 标准化
                    max_val = max(len(encoder.classes_), encoded_values.max())
                    if max_val > 0:
                        encoded_values = encoded_values / max_val
                    
                    encoded_cat_parts.append(encoded_values.reshape(-1, 1))
            
            if encoded_cat_parts:
                cat_encoded = np.hstack(encoded_cat_parts)
                processed_parts.append(cat_encoded)
        
        if processed_parts:
            return np.hstack(processed_parts)
        return None

    def _preprocess_test_data_for_dbscan(self, num_data, cat_data):
        """预处理测试数据"""
        processed_parts = []
        
        # 处理数值特征 - 使用全局统计量确保一致性
        if not num_data.empty and hasattr(self, '_dbscan_num_scaler'):
            num_filled = self._fill_missing_with_global_stats(num_data, 'median')
            scaled_num = self._dbscan_num_scaler.transform(num_filled)
            processed_parts.append(scaled_num)
        
        # 处理分类特征
        if not cat_data.empty and hasattr(self, '_dbscan_cat_encoders'):
            cat_filled = cat_data.fillna('MISSING')
            encoded_cat_parts = []
            
            for col in cat_data.columns:
                if col in self._dbscan_cat_encoders:
                    encoder = self._dbscan_cat_encoders[col]
                    # 处理未见过的类别
                    encoded_values = []
                    for val in cat_filled[col].astype(str):
                        if val in encoder.classes_:
                            encoded_val = encoder.transform([val])[0]
                        else:
                            # 未见过的类别设为最大值+1
                            encoded_val = len(encoder.classes_)
                    
                        encoded_values.append(encoded_val)
                    
                    encoded_values = np.array(encoded_values)
                    # 标准化
                    max_val = max(len(encoder.classes_), encoded_values.max())
                    if max_val > 0:
                        encoded_values = encoded_values / max_val
                    
                    encoded_cat_parts.append(encoded_values.reshape(-1, 1))
            
            if encoded_cat_parts:
                cat_encoded = np.hstack(encoded_cat_parts)
                processed_parts.append(cat_encoded)
        
        if processed_parts:
            return np.hstack(processed_parts)
        return None

    def _compute_test_distances(self, test_data, train_data, num_data, cat_data):
        """计算测试数据与训练数据的距离"""
        n_test = test_data.shape[0]
        n_train = train_data.shape[0]
        
        # 计算特征权重
        n_num_features = num_data.shape[1] if not num_data.empty else 0
        n_cat_features = cat_data.shape[1] if not cat_data.empty else 0
        total_features = n_num_features + n_cat_features
        
        if total_features == 0:
            return np.zeros((n_test, n_train))
        
        num_weight = n_num_features / total_features if n_num_features > 0 else 0
        cat_weight = n_cat_features / total_features if n_cat_features > 0 else 0
        
        distances = np.zeros((n_test, n_train))
        
        for i in range(n_test):
            for j in range(n_train):
                distance = 0.0
                
                # 数值特征距离
                if n_num_features > 0:
                    num_dist = np.linalg.norm(test_data[i, :n_num_features] - 
                                            train_data[j, :n_num_features])
                    distance += num_weight * num_dist
                
                # 分类特征距离
                if n_cat_features > 0:
                    cat_start = n_num_features
                    cat_dist = np.mean(test_data[i, cat_start:] != train_data[j, cat_start:])
                    distance += cat_weight * cat_dist
                
                distances[i, j] = distance
        
        return distances

    def _detect_cell_level_dbscan_anomalies(self, df, distances, eps, min_samples):
        """基于DBSCAN原理进行单元格级别的异常检测"""
        n_test = distances.shape[0]
        cell_errors = {}
        
        # 初始化错误掩码
        for col in self.num_features + self.cat_features:
            if col in df.columns:
                cell_errors[col] = pd.Series(False, index=df.index)
        
        anomaly_count = 0
        
        for i in range(n_test):
            # 找到eps邻域内的点数量
            neighbors = np.sum(distances[i, :] <= eps)
            
            # 如果邻域内的点少于min_samples，认为是异常
            if neighbors < min_samples:
                anomaly_count += 1
                df_idx = df.index[i]
                
                # 使用特征重要性分析来确定哪些单元格是异常的
                suspicious_features = self._analyze_feature_contribution_for_dbscan(
                    df.iloc[i], distances[i, :], eps
                )
                
                # 标记可疑的单元格
                for feature in suspicious_features:
                    if feature in cell_errors:
                        cell_errors[feature].loc[df_idx] = True
        
        print(f"  - DBSCAN单元格级检测: 发现 {anomaly_count} 个异常行")
        
        return cell_errors

    def _analyze_feature_contribution_for_dbscan(self, data_row, row_distances, eps):
        """
        分析每个特征对DBSCAN异常的贡献度 - 使用"同行评议"策略
        由于DBSCAN实际使用OPTICS实现，这里重定向到OPTICS的分析方法
        """
        # 由于DBSCAN实际使用OPTICS实现，我们重用OPTICS的特征贡献分析
        # 将eps作为伪阈值传递（实际不会用到）
        return self._analyze_feature_contribution_for_optics(data_row, row_distances, eps)

    def _detect_cell_level_optics_anomalies(self, df, distances, optics_model, outlier_threshold):
        """
        基于OPTICS原理进行单元格级别的异常检测
        
        Args:
            df: 待检测的数据框
            distances: 测试数据与训练数据的距离矩阵
            optics_model: 训练好的OPTICS模型
            outlier_threshold: 异常阈值
            
        Returns:
            字典，包含每个特征的异常标记
        """
        n_test = distances.shape[0]
        cell_errors = {}
        
        # 初始化错误掩码
        for col in self.num_features + self.cat_features:
            if col in df.columns:
                cell_errors[col] = pd.Series(False, index=df.index)
        
        anomaly_count = 0
        
        for i in range(n_test):
            # 计算测试点的近似可达性距离
            # 使用k近邻距离作为可达性距离的近似
            min_samples = getattr(optics_model, 'min_samples', 5)
            k_nearest_distances = np.sort(distances[i, :])[:min_samples]
            
            if len(k_nearest_distances) > 0:
                # 使用k近邻中的最大距离作为可达性距离
                reachability_dist = k_nearest_distances[-1]
            else:
                reachability_dist = np.inf
            
            # 如果可达性距离超过阈值，认为是异常
            if reachability_dist > outlier_threshold:
                anomaly_count += 1
                df_idx = df.index[i]
                
                # 使用特征重要性分析来确定哪些单元格是异常的
                suspicious_features = self._analyze_feature_contribution_for_optics(
                    df.iloc[i], distances[i, :], outlier_threshold
                )
                
                # 标记可疑的单元格
                for feature in suspicious_features:
                    if feature in cell_errors:
                        cell_errors[feature].loc[df_idx] = True
        
        if self.verbose:
            print(f"  - OPTICS单元格级检测: 发现 {anomaly_count} 个异常行")
        
        return cell_errors

    def _analyze_feature_contribution_for_optics(self, data_row, row_distances, outlier_threshold):
        """
        分析每个特征对OPTICS异常的贡献度 - 使用"同行评议"策略
        
        Args:
            data_row: 当前数据行
            row_distances: 当前行到所有训练点的距离数组
            outlier_threshold: 异常阈值
            
        Returns:
            可疑特征列表
        """
        suspicious_features = []
        
        # 获取训练数据
        if 'dbscan' not in self.detectors:  # OPTICS存储在'dbscan'键中
            return suspicious_features
            
        optics_config = self.detectors['dbscan']
        train_num_data = optics_config.get('train_num_data', pd.DataFrame())
        train_cat_data = optics_config.get('train_cat_data', pd.DataFrame())
        
        # 找到最近的K个邻居（这里使用10个）
        k = 5
        k_nearest_indices = np.argsort(row_distances)[:k]
        
        # 数值特征的"同行评议"分析
        if not train_num_data.empty:
            for col in self.num_features:
                if col in data_row.index and col in train_num_data.columns and not pd.isna(data_row[col]):
                    outlier_val = data_row[col]
                    
                    # 获取K个近邻在该特征上的值
                    neighbor_vals = train_num_data.iloc[k_nearest_indices][col].dropna()
                    
                    if len(neighbor_vals) > 0:
                        # 计算局部Z-score
                        neighbor_mean = neighbor_vals.mean()
                        neighbor_std = neighbor_vals.std()
                        
                        if neighbor_std > 1e-6:  # 避免除以零
                            local_z_score = abs((outlier_val - neighbor_mean) / neighbor_std)
                            if local_z_score > 2.5:  # 阈值可调
                                suspicious_features.append(col)
        
        # 分类特征的"同行评议"分析
        if not train_cat_data.empty:
            for col in self.cat_features:
                if col in data_row.index and col in train_cat_data.columns and not pd.isna(data_row[col]):
                    outlier_val = str(data_row[col])
                    
                    # 获取K个近邻在该特征上的值
                    neighbor_vals = train_cat_data.iloc[k_nearest_indices][col].dropna().astype(str)
                    
                    if len(neighbor_vals) > 0:
                        # 计算众数（最常见的值）
                        from scipy.stats import mode
                        try:
                            neighbor_mode = mode(neighbor_vals, keepdims=False).mode
                            # 如果异常点的值与近邻的众数不同，则可疑
                            if outlier_val != neighbor_mode:
                                suspicious_features.append(col)
                        except:
                            # 如果计算众数失败，检查是否存在于近邻值中
                            if outlier_val not in neighbor_vals.values:
                                suspicious_features.append(col)
        
        # 改进的兜底策略：如果没有找到可疑特征，只返回空列表而不是全部特征
        # 这避免了将所有特征都标记为错误的问题
        return suspicious_features

    def _fit_semantic_isolation_forest(self, df: pd.DataFrame):
        """
        拟合基于语义的孤立森林检测器
        
        这个方法使用Word2Vec对分类特征进行语义编码，结合数值特征的标准化，
        训练一个能够检测混合数据类型异常的IsolationForest模型。
        
        Args:
            df: 训练数据框
        """
        print("  - 准备语义特征工程...")
        
        # 1. 处理数值特征 - 使用全局统计量确保一致性
        if self.num_features:
            num_data = self._fill_missing_with_global_stats(df[self.num_features], 'median')
            self.scaler = StandardScaler()
            scaled_num_data = self.scaler.fit_transform(num_data)
        else:
            scaled_num_data = np.empty((len(df), 0))
        
        # 2. 处理分类特征 - 使用Word2Vec
        if self.cat_features:
            print("  - 训练Word2Vec模型...")
            cat_data = df[self.cat_features].fillna('MISSING')  # 将缺失值替换为特殊标记
            
            # 将每行转换为句子（值的列表）
            sentences = [list(row.astype(str)) for _, row in cat_data.iterrows()]
            
            # 训练Word2Vec模型
            self.w2v_model = Word2Vec(
                sentences, 
                vector_size=self.w2v_vector_size,
                window=5,
                min_count=1,
                workers=4,
                seed=self.random_state
            )
            
            # 将分类数据转换为向量
            w2v_features = self._transform_categorical_to_vectors(cat_data)
        else:
            w2v_features = np.empty((len(df), 0))
        
        # 3. 建立特征映射（地址簿）
        print("  - 建立特征映射...")
        self.feature_mapping = {}
        feature_index = 0
        
        # 数值特征映射
        for col in self.num_features:
            self.feature_mapping[feature_index] = col
            feature_index += 1
        
        # 分类特征映射（每个分类特征对应w2v_vector_size个向量维度）
        for col in self.cat_features:
            for i in range(self.w2v_vector_size):
                self.feature_mapping[feature_index] = col
                feature_index += 1
        
        # 4. 合并特征矩阵
        if scaled_num_data.shape[1] > 0 and w2v_features.shape[1] > 0:
            X_train = np.hstack([scaled_num_data, w2v_features])
        elif scaled_num_data.shape[1] > 0:
            X_train = scaled_num_data
        elif w2v_features.shape[1] > 0:
            X_train = w2v_features
        else:
            raise ValueError("至少需要一个数值或分类特征来训练语义孤立森林")
        
        print(f"  - 特征矩阵形状: {X_train.shape}")
        
        # 5. 训练IsolationForest模型
        print("  - 训练IsolationForest模型...")
        self.detectors['semantic_isolation_forest'] = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
        )
        self.detectors['semantic_isolation_forest'].fit(X_train)
        
        print("  - 语义孤立森林检测器训练完成")

    def _transform_categorical_to_vectors(self, cat_data: pd.DataFrame) -> np.ndarray:
        """
        将分类数据转换为Word2Vec向量矩阵
        
        Args:
            cat_data: 分类数据框
            
        Returns:
            向量矩阵，形状为 (n_samples, n_cat_features * w2v_vector_size)
        """
        if self.w2v_model is None:
            raise ValueError("Word2Vec模型尚未训练")
        
        vectors_list = []
        
        for _, row in cat_data.iterrows():
            row_vectors = []
            for col in self.cat_features:
                word = str(row[col])
                # 如果词在模型词汇表中，获取其向量；否则使用零向量
                if word in self.w2v_model.wv:
                    row_vectors.append(self.w2v_model.wv[word])
                else:
                    row_vectors.append(np.zeros(self.w2v_vector_size))
            
            # 将该行所有特征的向量拼接
            if row_vectors:
                vectors_list.append(np.concatenate(row_vectors))
            else:
                vectors_list.append(np.array([]))
        
        if vectors_list and len(vectors_list[0]) > 0:
            return np.array(vectors_list)
        else:
            return np.empty((len(cat_data), 0))

    def _transform_data_for_semantic_detection(self, df: pd.DataFrame) -> np.ndarray:
        """
        将待检测数据转换为语义特征矩阵
        
        Args:
            df: 待检测的数据框
            
        Returns:
            特征矩阵，与训练时使用的格式一致
        """
        # 1. 处理数值特征 - 使用全局统计量确保一致性
        if self.num_features and self.scaler is not None:
            num_data = self._fill_missing_with_global_stats(df[self.num_features], 'median')
            scaled_num_data = self.scaler.transform(num_data)
        else:
            scaled_num_data = np.empty((len(df), 0))
        
        # 2. 处理分类特征
        if self.cat_features and self.w2v_model is not None:
            cat_data = df[self.cat_features].fillna('MISSING')
            w2v_features = self._transform_categorical_to_vectors(cat_data)
        else:
            w2v_features = np.empty((len(df), 0))
        
        # 3. 合并特征矩阵
        if scaled_num_data.shape[1] > 0 and w2v_features.shape[1] > 0:
            return np.hstack([scaled_num_data, w2v_features])
        elif scaled_num_data.shape[1] > 0:
            return scaled_num_data
        elif w2v_features.shape[1] > 0:
            return w2v_features
        else:
            return np.empty((len(df), 0))

    def _detect_semantic_isolation_forest_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        使用语义孤立森林检测异常，并通过决策路径分析定位错误单元格
        
        Args:
            df: 待检测的数据框
            
        Returns:
            错误掩码DataFrame，True表示错误单元格
        """
        error_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
        
        if 'semantic_isolation_forest' not in self.detectors:
            return error_mask
        
        # 1. 转换数据
        X_test = self._transform_data_for_semantic_detection(df)
        
        if X_test.shape[1] == 0:
            return error_mask
        
        # 2. 检测异常行
        model = self.detectors['semantic_isolation_forest']
        predictions = model.predict(X_test)
        outlier_indices = np.where(predictions == -1)[0]
        
        if len(outlier_indices) == 0:
            return error_mask  # 没有检测到异常
        
        if self.verbose:
            print(f"  - 检测到 {len(outlier_indices)} 个异常行，开始单元格级别分析...")
        
        # 3. 对每个异常行进行单元格级别的错误定位
        for row_idx in outlier_indices:
            outlier_vector = X_test[row_idx:row_idx+1]  # 保持2D形状
            
            # 使用决策路径分析找出可疑的特征
            suspicious_columns = self._locate_error_cells_by_decision_path(
                model, outlier_vector
            )
            
            # 在error_mask中标记可疑的单元格
            df_row_idx = df.index[row_idx]  # 获取原始DataFrame中的行索引
            for col in suspicious_columns:
                if col in df.columns:
                    error_mask.loc[df_row_idx, col] = True
        
        return error_mask

    def _locate_error_cells_by_decision_path(self, iso_forest_model, outlier_vector: np.ndarray) -> List[str]:
        """
        通过分析IsolationForest的决策路径来定位导致异常的特征/单元格
        使用按分离深度加权的策略 - 浅层分裂的特征权重更高
        
        Args:
            iso_forest_model: 训练好的IsolationForest模型
            outlier_vector: 异常数据点的特征向量 (1, n_features)
            
        Returns:
            导致异常的列名列表
        """
        feature_weighted_scores = {}  # 存储特征的加权得分
        
        # IsolationForest由多个决策树组成
        for tree in iso_forest_model.estimators_:
            # 获取异常点在当前树上的决策路径
            leaf_id = tree.apply(outlier_vector)[0]  # 获取叶子节点ID
            path_nodes = tree.decision_path(outlier_vector).toarray()[0]
            
            # 遍历决策路径上的每个节点，记录深度
            current_depth = 0
            for node_id in range(len(path_nodes)):
                if path_nodes[node_id] == 1:  # 节点被访问过
                    # 检查是否为内部节点（有分裂特征）
                    if (tree.tree_.children_left[node_id] != tree.tree_.children_right[node_id]):
                        # 获取该节点的分裂特征索引
                        feature_index = tree.tree_.feature[node_id]
                        
                        # 计算深度权重：越浅的分裂权重越高
                        # 使用 1 / (depth + 1) 作为权重函数
                        depth_weight = 1.0 / (current_depth + 1)
                        
                        # 累加该特征的加权得分
                        if feature_index not in feature_weighted_scores:
                            feature_weighted_scores[feature_index] = 0.0
                        feature_weighted_scores[feature_index] += depth_weight
                        
                        current_depth += 1
        
        # 将特征索引转换为列名，并按加权得分排序
        column_weighted_scores = {}
        for feature_idx, weighted_score in feature_weighted_scores.items():
            if feature_idx in self.feature_mapping:
                column_name = self.feature_mapping[feature_idx]
                if column_name not in column_weighted_scores:
                    column_weighted_scores[column_name] = 0.0
                column_weighted_scores[column_name] += weighted_score
        
        # 按加权得分排序
        sorted_columns = sorted(column_weighted_scores.items(), key=lambda x: x[1], reverse=True)
        
        if not sorted_columns:
            return []

        # 改进的阈值策略：使用加权得分进行筛选
        max_score = sorted_columns[0][1]
        
        # 动态阈值：选择得分不低于最高得分50%的特征
        # 但至少要求得分大于1.0（意味着至少在根节点附近有贡献）
        threshold = max(1.0, max_score * 0.4)
        
        suspicious_columns = [col for col, score in sorted_columns if score >= threshold]

        # 限制返回的特征数量，避免标记过多特征
        # 最多返回前5个最可疑的特征
        max_features = 5
        if len(suspicious_columns) > max_features:
            suspicious_columns = suspicious_columns[:max_features]

        # 如果没有特征满足阈值，返回得分最高的一个特征
        if not suspicious_columns and sorted_columns:
            suspicious_columns = [sorted_columns[0][0]]

        return suspicious_columns


def _display_error_details(df: pd.DataFrame, error_mask: pd.DataFrame, detection_methods: List[str], max_rows: int = 10, output_dir: str = "error_detection_results"):
    """
    将检测到的错误行的具体情况输出到txt文件
    
    Args:
        df: 原始数据框
        error_mask: 错误掩码（True表示错误）
        detection_methods: 使用的检测方法列表
        max_rows: 最多显示的错误行数
        output_dir: 输出目录
    """
    import os
    from datetime import datetime
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    methods_str = "_".join(detection_methods)
    filename = f"error_details_{methods_str}_{timestamp}.txt"
    filepath = os.path.join(output_dir, filename)
    
    # 找到有错误的行
    error_rows = error_mask.any(axis=1)
    error_row_indices = df[error_rows].index[:max_rows]
    
    with open(filepath, 'w', encoding='utf-8') as f:
        if len(error_row_indices) == 0:
            f.write("没有检测到错误行\n")
            print(f"\n  没有检测到错误行，结果已保存到: {filepath}")
            return
        
        # 写入头部信息
        f.write("=" * 80 + "\n")
        f.write("错误检测详细结果\n")
        f.write("=" * 80 + "\n")
        f.write(f"检测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"检测方法: {detection_methods}\n")
        f.write(f"数据形状: {df.shape}\n")
        f.write(f"总计检测到 {error_rows.sum()} 个错误行\n")
        f.write(f"显示前 {min(max_rows, len(error_row_indices))} 个错误行\n")
        f.write("=" * 80 + "\n\n")
        
        # 为每个错误行创建显示内容
        for idx, row_idx in enumerate(error_row_indices):
            f.write(f"错误行 #{idx+1} (原始索引: {row_idx})\n")
            f.write("-" * 60 + "\n")
            
            # 获取该行的原始数据和错误掩码
            original_row = df.loc[row_idx]
            error_row_mask = error_mask.loc[row_idx]
            
            # 创建显示用的DataFrame
            display_data = []
            for col in df.columns:
                original_value = original_row[col]
                is_error = error_row_mask[col]
                
                # 如果是错误单元格，标记为"ERROR"
                if is_error:
                    display_value = f"{original_value} (ERROR)"
                else:
                    display_value = str(original_value)
                
                display_data.append({
                    '属性': col,
                    '值': display_value,
                    '错误状态': '✓' if is_error else ''
                })
            
            # 创建并写入DataFrame
            display_df = pd.DataFrame(display_data)
            f.write(display_df.to_string(index=False) + "\n\n")
            
            # 统计该行的错误情况
            error_count = error_row_mask.sum()
            total_columns = len(df.columns)
            error_rate = error_count / total_columns
            f.write(f"该行错误单元格数: {error_count}/{total_columns} ({error_rate:.2%})\n")
            
            if idx < len(error_row_indices) - 1:  # 不是最后一行
                f.write("\n" + "=" * 60 + "\n\n")
        
        # 写入汇总统计
        f.write("\n" + "=" * 80 + "\n")
        f.write("汇总统计\n")
        f.write("=" * 80 + "\n")
        
        # 计算每列的错误统计
        column_error_counts = error_mask.sum()
        f.write("各列错误统计:\n")
        for col in df.columns:
            error_count = column_error_counts[col]
            error_rate = error_count / len(df)
            if error_count > 0:
                f.write(f"  {col}: {error_count} 个错误 ({error_rate:.2%})\n")
        
        # 整体错误统计
        total_errors = error_mask.sum().sum()
        total_cells = df.size
        overall_error_rate = total_errors / total_cells
        f.write(f"\n总体错误统计:\n")
        f.write(f"  总错误单元格数: {total_errors}\n")
        f.write(f"  总单元格数: {total_cells}\n")
        f.write(f"  总体错误率: {overall_error_rate:.4f} ({overall_error_rate:.2%})\n")
        f.write(f"  错误行数: {error_rows.sum()}\n")
        f.write(f"  总行数: {len(df)}\n")
        f.write(f"  错误行比例: {error_rows.sum() / len(df):.2%}\n")
    
    print(f"\n  错误检测详情已保存到: {filepath}")
    print(f"  检测到 {error_rows.sum()} 个错误行，显示了前 {min(max_rows, len(error_row_indices))} 行")

def calculate_error_rate(df: pd.DataFrame, 
                        error_detector: ErrorDetector,
                        dataset_name: str = None) -> Dict[str, Any]:
    """
    计算数据中的错误率，使用已配置的ErrorDetector，支持缓存机制
    
    Args:
        df: 待检测的数据框
        error_detector: 已配置并拟合的ErrorDetector实例
        dataset_name: 数据集名称，用于缓存优化
        
    Returns:
        dict: 包含错误检测结果的字典，格式与calculate_missing_rate完全兼容
    """
    if not error_detector.fitted:
        raise ValueError("ErrorDetector尚未拟合，请先调用fit()方法")
    
    # 检测错误，使用缓存机制
    results = error_detector.detect_errors(df, dataset_name=dataset_name)
    
    return results 

def main():
    """
    测试错误检测功能的主函数
    """
    import argparse
    import os
    import sys
    
    # 添加项目根目录到路径
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from utils.utility import load_config_json, split_dataframe
    from datasets.data_processor import DataProcessor
    
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="表格数据错误检测测试")
    parser.add_argument("--dataset", type=str, default="Titanic", help="数据集名称")
    parser.add_argument("--test_split", type=float, default=0.2, help="测试集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    
    print("="*60)
    print("表格数据错误检测测试")
    print("="*60)
    print(f"数据集: {args.dataset}")
    print(f"随机种子: {args.seed}")
    
    # 加载数据配置
    data_config_path = f"data/{args.dataset}/data_config.json"
    if not os.path.exists(data_config_path):
        print(f"错误: 找不到数据配置文件 {data_config_path}")
        print("请确保数据集配置文件存在")
        return
    
    data_config = load_config_json(data_config_path)
    
    # 提取特征配置
    numerical_features = data_config["numerical_features"]
    categorical_features = data_config["categorical_features"]
    target_feature = data_config["target_feature"]
    
    print(f"\n特征配置:")
    print(f"  数值特征 ({len(numerical_features)}): {numerical_features}")
    print(f"  分类特征 ({len(categorical_features)}): {categorical_features}")
    print(f"  目标特征: {target_feature}")
    
    # 加载数据
    df_path = data_config["data_path"]
    if not os.path.exists(df_path):
        print(f"错误: 找不到数据文件 {df_path}")
        return
    
    df = pd.read_csv(df_path)
    print(f"\n数据集信息:")
    print(f"  数据形状: {df.shape}")
    print(f"  列名: {list(df.columns)}")
    
    # 数据划分
    train_df, test_df = split_dataframe(
        df, 
        test_size=args.test_split, 
        random_state=args.seed, 
        stratify_column=target_feature[0] if len(target_feature) > 0 else None
    )
    
    print(f"\n数据划分:")
    print(f"  训练集大小: {len(train_df)}")
    print(f"  测试集大小: {len(test_df)}")
    
    # 初始化数据处理器
    processor = DataProcessor(
        numerical_features=numerical_features, 
        categorical_features=categorical_features,
        target_feature=target_feature
    )
    processor.fit(train_df)
    
    print(f"\n数据处理器信息:")
    print(f"  标签编码器: {len(processor.label_encoder)} 个类别")
    print(f"  分类特征大小: {processor.categories}")
    
    # 测试1: 使用特定检测方法
    print("\n" + "="*50)
    print("测试1: 使用特定检测方法（semantic_isolation_forest+missing_values）")
    print("="*50)
    
    detection_methods = ['semantic_isolation_forest', "missing_values"]
    
    # 创建并拟合特定方法检测器
    detector_specific = ErrorDetector(
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        target_feature=target_feature,
        detection_methods=detection_methods,
    )
    detector_specific.fit(train_df)
    
    error_results_specific = calculate_error_rate(
        df=test_df,
        error_detector=detector_specific,
        dataset_name=f"{args.dataset}_{'_'.join(detection_methods)}"
    )
    
    print(f"特定方法检测结果 ({detection_methods}):")
    print(f"  总体错误率: {error_results_specific['overall_rate']:.4f}")
    print(f"  各列错误率:")
    for col, rate in error_results_specific['column_rates'].items():
        if rate > 0:
            print(f"    {col}: {rate:.4f}")
    
    # 显示错误行的具体情况
    _display_error_details(test_df, error_results_specific['error_mask'], detection_methods)
    
    # 测试2: 使用特定检测方法
    print("\n" + "="*50)
    print("测试2: 使用特定检测方法（dbscan+missing_values）")
    print("="*50)
    
    detection_methods = ['dbscan', "missing_values"]
    
    # 创建并拟合特定方法检测器
    detector_specific = ErrorDetector(
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        target_feature=target_feature,
        detection_methods=detection_methods,
    )
    detector_specific.fit(train_df)
    
    error_results_specific = calculate_error_rate(
        df=test_df,
        error_detector=detector_specific,
        dataset_name=f"{args.dataset}_{'_'.join(detection_methods)}"
    )
    
    print(f"特定方法检测结果 ({detection_methods}):")
    print(f"  总体错误率: {error_results_specific['overall_rate']:.4f}")
    print(f"  各列错误率:")
    for col, rate in error_results_specific['column_rates'].items():
        if rate > 0:
            print(f"    {col}: {rate:.4f}")
    
    # 显示错误行的具体情况
    _display_error_details(test_df, error_results_specific['error_mask'], detection_methods)
    
    # 测试3: 使用特定检测方法
    print("\n" + "="*50)
    print("测试3: 使用特定检测方法（lof+missing_values）")
    print("="*50)
    
    detection_methods = ['lof', "missing_values"]
    
    # 创建并拟合特定方法检测器
    detector_specific = ErrorDetector(
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        target_feature=target_feature,
        detection_methods=detection_methods,
    )
    detector_specific.fit(train_df)
    
    error_results_specific = calculate_error_rate(
        df=test_df,
        error_detector=detector_specific,
        dataset_name=f"{args.dataset}_{'_'.join(detection_methods)}"
    )
    
    print(f"特定方法检测结果 ({detection_methods}):")
    print(f"  总体错误率: {error_results_specific['overall_rate']:.4f}")
    print(f"  各列错误率:")
    for col, rate in error_results_specific['column_rates'].items():
        if rate > 0:
            print(f"    {col}: {rate:.4f}")
    
    # 显示错误行的具体情况
    _display_error_details(test_df, error_results_specific['error_mask'], detection_methods)
    

if __name__ == "__main__":
    main()


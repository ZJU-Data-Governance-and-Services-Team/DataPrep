import torch.nn.functional as F
import torch.nn as nn
import torch
import pandas as pd
import numpy as np
from utils.constants import (TRAIN_DATA_CLEAN_MASK_STR, TRAIN_DATA_ERROR_MASK_STR,
                             VAL_DATA_MASK_STR, TEST_DATA_MASK_STR)
from utils.utility import identify_complete_and_error_samples_with_mask
from utils.mask_cache_manager import get_global_mask_cache_manager
from utils.main_modules import MLPClassifier
from tqdm import tqdm
from models.imputation.standard_pipeline import combine_features, evaluate_classifier
from models.imputation.repair_embed import unified_imputation
from models.diffae.core import DiffusionUtils
from models.bilevel_beta.diffusion_logic import _convert_onehot_to_indices, _create_onehot_from_indices
from itertools import cycle
from datasets.data_processor import DataProcessor
from .diffusion_logic import create_cond_mask



class BiLevelTrainer:
    """
    Bi-Level优化训练器，使用现有的TabularDiffAE作为填补模型
    
    该训练器实现了一个改进的双层优化框架：
    - 内层优化：使用训练集数据训练分类器以适应当前的填补数据
    - 外层优化：使用验证集数据更新填补模型以最小化分类器的损失
    
    这种设计有以下优势：
    1. 避免了训练集和验证的数据泄露
    2. 确保外层优化目标与最终评估目标一致
    3. 提高了模型的泛化能力
    
    外层损失包含以下组件：
    1. 类别可分离性损失：鼓励填补样本在分类器嵌入空间中向对应类别质心靠近
    2. 基于掩码的重建损失：确保填补模型在已知位置能够准确重建原始特征
    3. 梯度对齐损失：确保填补样本和完整样本的梯度对齐
    
    """
    
    def __init__(self, args, 
                 processor: DataProcessor, 
                 d_numerical: int, 
                 d_categorical: int, 
                 actual_cat_sizes: list, 
                 num_classes: int, 
                 device: str, 
                 repair_model=None, 
                 diffusion_utils: DiffusionUtils=None, 
                 feature_embedder=None):
        """
        初始化BiLevelTrainer
        
        Args:
            args: 参数配置
            processor: 数据处理器
            d_numerical: 数值特征数量
            d_categorical: 分类特征数量  
            actual_cat_sizes: 实际分类特征大小
            num_classes: 类别数量
            device: 设备
            repair_model: 预训练的修复模型
            diffusion_utils: 扩散工具类
            feature_embedder: 特征嵌入器
        """
        self.args = args
        self.processor = processor
        self.d_numerical = d_numerical
        self.d_categorical = d_categorical
        self.actual_cat_sizes = actual_cat_sizes
        self.num_classes = num_classes
        self.device = device
        self.error_detector = args.error_detector
        self.noise_t_batch_level = getattr(self.args, 'noise_t_batch_level', True)
        
        # 使用传入的预训练TabularDiffAE模型
        self.repair_model = repair_model
        self.diffusion_utils = diffusion_utils
        self.feature_embedder = feature_embedder

        # GradNorm相关初始化
        self.use_gradnorm = getattr(self.args, 'use_gradnorm', False)
        if self.use_gradnorm:
            # 为bilevel_loss和diffusion_loss创建可学习的权重
            self.log_weights = nn.Parameter(torch.zeros(2, device=self.device))
            self.gradnorm_optimizer = torch.optim.Adam([self.log_weights], lr=getattr(self.args, 'gradnorm_lr', 1e-3))
            self.initial_losses = None
            self.gradnorm_alpha = getattr(self.args, 'gradnorm_alpha', 1.5)
            print("GradNorm已启用")

        # 使用全局统一缓存管理器
        self.mask_cache_manager = get_global_mask_cache_manager()
        
        # 保留预计算掩码以兼容现有代码
        self.precomputed_error_masks = {}  # 用于存储完整数据集的错误掩码
        self.precomputed_observation_masks = {}  # 用于存储完整数据集的观测掩码
        
        # 预训练相关参数
        self.enable_pretraining = getattr(self.args, 'enable_pretraining', False)
        self.pretraining_epochs = getattr(self.args, 'pretraining_epochs', 50)
        
        if self.enable_pretraining:
            print(f"预训练已启用: {self.pretraining_epochs} epochs")
    
    def get_cached_error_mask(self, df: pd.DataFrame, mask_type: str = "missing") -> torch.Tensor:
        """
        获取缓存的错误掩码
        
        Args:
            df: 输入DataFrame
            mask_type: 掩码类型，"missing"或"observation"
            
        Returns:
            错误掩码张量
        """
        # 构建特征列表
        feature_cols = self.processor.num_features + self.processor.cat_features
        
        # 尝试从统一缓存管理器获取tensor格式的掩码
        tensor_mask = self.mask_cache_manager.get_error_mask_tensor(
            df=df,
            feature_cols=feature_cols,
            actual_cat_sizes=self.actual_cat_sizes,
            num_features=self.processor.num_features,
            cat_features=self.processor.cat_features,
            mask_type="error" if mask_type == "missing" else "observation",
            device=self.device
        )
        
        if tensor_mask is not None:
            return tensor_mask
        
        # 缓存未命中，计算并缓存
        if mask_type == "missing":
            mask = self._compute_missing_mask(df)
        else:
            mask = self._compute_observation_mask(df)
        
        # 将计算结果缓存到统一缓存管理器
        self.mask_cache_manager.set_tensor_cache(df[feature_cols], mask, 
                                                "error" if mask_type == "missing" else "observation")
        
        return mask
    
    def _compute_mask(self, df: pd.DataFrame, mask_type: str = "missing") -> torch.Tensor:
        """
        统一的掩码计算方法
        
        Args:
            df: 输入DataFrame
            mask_type: 掩码类型，"missing"或"observation"
            
        Returns:
            掩码张量
        """
        # 使用processor的缓存错误掩码获取方法
        combined_missing_mask = self.processor.get_error_mask(df)
        
        # 分别处理数值特征和分类特征的掩码
        num_mask = None
        cat_mask = None
        
        if self.processor.num_features:
            num_missing_mask = combined_missing_mask[self.processor.num_features].values
            if mask_type == "missing":
                num_mask = torch.tensor(num_missing_mask, dtype=torch.bool, device=self.device)
            else:  # observation
                num_mask = torch.tensor(~num_missing_mask, dtype=torch.bool, device=self.device)
        
        if self.processor.cat_features:
            cat_missing_mask = combined_missing_mask[self.processor.cat_features].values
            
            if mask_type == "missing":
                cat_mask = torch.tensor(cat_missing_mask, dtype=torch.bool, device=self.device)
                # 将分类特征的掩码扩展为独热编码形式
                expanded_cat_mask = []
                for i, cat_size in enumerate(self.actual_cat_sizes):
                    feature_mask = cat_mask[:, i:i+1]  
                    expanded_feature_mask = feature_mask.repeat(1, cat_size)
                    expanded_cat_mask.append(expanded_feature_mask)
                
                if expanded_cat_mask:
                    cat_mask_expanded = torch.cat(expanded_cat_mask, dim=1)
                else:
                    cat_mask_expanded = torch.empty((df.shape[0], 0), dtype=torch.bool, device=self.device)
            else:  # observation
                cat_mask = torch.tensor(~cat_missing_mask, dtype=torch.bool, device=self.device)
                # 对于observation掩码，不需要扩展到独热编码维度
                cat_mask_expanded = cat_mask
        else:
            if mask_type == "missing":
                cat_mask_expanded = torch.empty((df.shape[0], 0), dtype=torch.bool, device=self.device)
            else:  # observation
                cat_mask_expanded = torch.empty((df.shape[0], 0), dtype=torch.bool, device=self.device)
        
        # 合并数值特征和分类特征的掩码
        if num_mask is not None and cat_mask_expanded.shape[1] > 0:
            combined_mask = torch.cat([num_mask, cat_mask_expanded], dim=1)
        elif num_mask is not None:
            combined_mask = num_mask
        elif cat_mask_expanded.shape[1] > 0:
            combined_mask = cat_mask_expanded
        else:
            combined_mask = torch.empty((df.shape[0], 0), dtype=torch.bool, device=self.device)
        
        return combined_mask

    def _compute_missing_mask(self, df: pd.DataFrame) -> torch.Tensor:
        """计算缺失值掩码的内部方法（保留向后兼容性）"""
        return self._compute_mask(df, mask_type="missing")

    def _compute_observation_mask(self, df: pd.DataFrame) -> torch.Tensor:
        """计算观测掩码的内部方法（保留向后兼容性）"""
        return self._compute_mask(df, mask_type="observation")

    def impute_with_model(self, df, dataset_name=None, error_mask_df=None):
        """
        使用TabularDiffAE模型进行数据填补
        
        Args:
            df: 需要填补的数据框
            dataset_name: 数据集名称，用于缓存优化
            error_mask_df: 外部提供的错误掩码DataFrame
            
        Returns:
            填补后的数据框
        """

        # 转换为可微分版本调用
        return self._repair_df_with_differentiable_model(df, dataset_name, error_mask_df, with_grad=False)

    def _repair_df_with_differentiable_model(self, df, dataset_name=None, error_mask_df=None, with_grad=False):
        """
        使用可微分模型对DataFrame进行填补的辅助方法
        
        Args:
            df: 需要填补的数据框
            dataset_name: 数据集名称，用于缓存优化
            error_mask_df: 外部提供的错误掩码DataFrame
            with_grad: 是否保持梯度
            
        Returns:
            填补后的数据框
        """
        if len(df) == 0:
            return df.copy()
        
        # 转换为张量格式并移动到正确设备
        x_num, x_cat_onehot = self.processor.transform_onehot(df, dataset_name=dataset_name)
        x_num = x_num.to(self.device) if x_num is not None else None
        x_cat_onehot = x_cat_onehot.to(self.device) if x_cat_onehot is not None else None
        
        # 获取缺失掩码并移动到正确设备
        if error_mask_df is not None:
            feature_cols = self.processor.num_features + self.processor.cat_features
            missing_mask = self._convert_pandas_mask_to_tensor(error_mask_df[feature_cols])
        else:
            missing_mask = self.get_cached_error_mask(df, mask_type="missing")
        
        missing_mask = missing_mask.to(self.device)
        
        # 使用可微分填补
        x_imputed = self.impute_with_model_differentiable(
            x_num, x_cat_onehot, missing_mask, 
            method='diffusion' if self.args.use_diffusion else 'direct',
            with_grad=with_grad
        )
        
        # 转换回DataFrame格式
        return self._convert_tensor_to_dataframe(x_imputed, df, dataset_name)
    
    def _convert_tensor_to_dataframe(self, x_tensor, original_df, dataset_name=None):
        """
        将填补后的张量转换回DataFrame格式
        
        Args:
            x_tensor: 填补后的特征张量
            original_df: 原始DataFrame（用于获取结构信息）
            dataset_name: 数据集名称
            
        Returns:
            填补后的DataFrame
        """
        # 创建结果DataFrame
        result_df = original_df.copy()
        
        # 分离数值特征和分类特征
        x_tensor_np = x_tensor.detach().cpu().numpy()
        
        if self.d_numerical > 0:
            x_num_imputed = x_tensor_np[:, :self.d_numerical]
            # 更新数值特征
            x_num_imputed = self.processor.num_scaler.inverse_transform(x_num_imputed)
            for i, col in enumerate(self.processor.num_features):
                result_df[col] = x_num_imputed[:, i]
        
        if self.d_categorical > 0:
            # 处理分类特征
            cat_onehot_dim = sum(self.actual_cat_sizes)
            x_cat_onehot_imputed = x_tensor_np[:, self.d_numerical:self.d_numerical + cat_onehot_dim]
            
            # 将独热编码转换回分类值
            start_idx = 0
            for i, (col, cat_size) in enumerate(zip(self.processor.cat_features, self.actual_cat_sizes)):
                end_idx = start_idx + cat_size
                cat_probs = x_cat_onehot_imputed[:, start_idx:end_idx]
                cat_indices = np.argmax(cat_probs, axis=1)
                
                # 转换回原始类别值
                if hasattr(self.processor, 'cat_encoders') and col in self.processor.cat_encoders:
                    # 使用映射转换回原始值
                    reverse_mapping = {v: k for k, v in self.processor.cat_encoders[col].items()}
                    result_df[col] = [reverse_mapping.get(idx, idx) for idx in cat_indices]
                else:
                    result_df[col] = cat_indices
                
                start_idx = end_idx
        
        return result_df

    def _get_penultimate_output(self, classifier, x_batch, y_batch=None, mode='embedding'):
        """
        统一的分类器倒数第二层访问函数
        
        Args:
            classifier: MLPClassifier模型
            x_batch: 输入特征批次
            y_batch: 标签批次（仅在mode='gradient'时需要）
            mode: 输出模式
                - 'embedding': 获取嵌入表示（不计算梯度）
                - 'embedding_with_grad': 获取嵌入表示（保留梯度）
                - 'gradient': 获取倒数第二层的梯度
                
        Returns:
            根据mode返回相应的输出
        """
        if classifier is None or x_batch is None:
            if mode == 'gradient':
                print("获取梯度的输入参数存在None值")
                return None
            print("获取嵌入表示的输入参数存在None值")
            return None
            
        try:
            classifier.eval()
            
            # 用于存储倒数第二层的激活值
            penultimate_activations = None
            
            def hook_fn(module, input, output):
                nonlocal penultimate_activations
                if mode == 'embedding':
                    penultimate_activations = output.detach()  # 不需要梯度
                elif mode == 'embedding_with_grad':
                    penultimate_activations = output  # 保留梯度
                elif mode == 'gradient':
                    penultimate_activations = output
                    penultimate_activations.requires_grad_(True)  # 确保激活值需要梯度
            
            # 找到倒数第二个Linear层并注册hook
            linear_layers = []
            for module in classifier.modules():
                if isinstance(module, nn.Linear):
                    linear_layers.append(module)
            
            if len(linear_layers) < 2:
                error_msg = "分类器至少需要两个Linear层才能获取倒数第二层"
                if mode == 'gradient':
                    raise ValueError(error_msg)
                print(error_msg)
                return None
            
            # 倒数第二个Linear层
            penultimate_linear = linear_layers[-2]
            handle = penultimate_linear.register_forward_hook(hook_fn)
            
            try:
                if mode == 'embedding':
                    # 不需要梯度的情况
                    with torch.no_grad():
                        _ = classifier(x_batch)
                elif mode == 'embedding_with_grad':
                    # 保留梯度的情况
                    _ = classifier(x_batch)
                elif mode == 'gradient':
                    # 需要计算梯度的情况
                    if y_batch is None:
                        raise ValueError("计算梯度时需要提供y_batch")
                    
                    x_batch.requires_grad_(True)
                    outputs = classifier(x_batch)
                    
                    # 计算损失
                    criterion = nn.CrossEntropyLoss()
                    loss = criterion(outputs, y_batch)
                    
                    # 清零梯度
                    classifier.zero_grad()
                    
                    # 计算梯度
                    gradients = torch.autograd.grad(
                        outputs=loss, 
                        inputs=penultimate_activations, 
                        retain_graph=True, 
                        create_graph=True
                    )[0]
                    
                    return gradients
                
                if penultimate_activations is None:
                    error_msg = "无法获取倒数第二层的激活值"
                    if mode == 'gradient':
                        raise RuntimeError(error_msg)
                    print(error_msg)
                    return None
                
                return penultimate_activations
                
            finally:
                # 清理hook
                handle.remove()
        except Exception as e:
            if mode == 'gradient':
                raise e
            print(f"获取分类器嵌入表示时出错: {e}")
            import traceback
            traceback.print_exc()
            return None

    def compute_class_centroids(self, classifier, train_df_complete):
        """
        计算每个类别在分类器倒数第二层嵌入空间中的质心
        
        Args:
            classifier: 训练好的分类器
            train_df_complete: 完整的训练数据（用于计算质心）
            
        Returns:
            class_centroids: 形状为[num_classes, embedding_dim]的张量，每行是一个类别的质心
        """
        classifier.eval()
        
        # 准备数据
        x_num, x_cat_onehot = self.processor.transform_onehot(train_df_complete, dataset_name="class_centroids")
        x_combined = combine_features(x_num, x_cat_onehot)
        x_combined = x_combined.to(self.device)
        
        # 获取标签
        target_col = self.processor.target_feature[0]
        target_values = train_df_complete[target_col].values
        if isinstance(target_values[0], (list, np.ndarray)):
            target_values = np.array([val[0] if isinstance(val, (list, np.ndarray)) else val for val in target_values])
        encoded_labels = np.array([self.processor.label_encoder.get(val, 0) for val in target_values])
        y = torch.LongTensor(encoded_labels).to(self.device)
        
        # 获取所有样本的嵌入表示
        with torch.no_grad():
            embeddings = self._get_penultimate_output(classifier, x_combined, mode='embedding')
        
        # 计算每个类别的质心
        unique_classes = torch.unique(y)
        embedding_dim = embeddings.shape[1]
        class_centroids = torch.zeros(self.num_classes, embedding_dim, device=self.device)
        
        for class_idx in unique_classes:
            class_mask = (y == class_idx)
            if class_mask.sum() > 0:
                class_embeddings = embeddings[class_mask]
                class_centroids[class_idx] = class_embeddings.mean(dim=0)
        
        return class_centroids

    def compute_class_separability_loss(self, classifier, x_imputed_batch, y_imputed_batch, class_centroids):
        """
        计算基于对比学习的类别可分离性损失函数
        
        使用InfoNCE风格的对比损失：
        - 正样本：同类别的质心
        - 负样本：其他类别的质心
        
        Args:
            classifier: 分类器
            x_imputed_batch: 填补后的样本特征批次
            y_imputed_batch: 填补样本的真实标签批次
            class_centroids: 每个类别的质心 [num_classes, embedding_dim]
            
        Returns:
            contrastive_loss: 对比学习损失
            avg_positive_sim: 平均正样本相似度（用于监控）
            avg_negative_sim: 平均负样本相似度（用于监控）
        """
        # 输入检查
        if classifier is None or x_imputed_batch is None or y_imputed_batch is None or class_centroids is None:
            print("计算类别可分离性损失的输入参数存在None值")
            return (torch.tensor(0.0, device=self.device, requires_grad=True), 
                    0.0, 0.0)

        # 获取填补样本的嵌入表示（保留梯度）
        imputed_embeddings = self._get_penultimate_output(classifier, x_imputed_batch, mode='embedding_with_grad')
        
        if imputed_embeddings is None:
            print("获取嵌入表示失败")
            return (torch.tensor(0.0, device=self.device, requires_grad=True), 
                    0.0, 0.0)
        
        # 对比学习的温度参数
        temperature = getattr(self.args, 'contrastive_temperature', 0.1)
        
        # 这些变量在向量化计算中不再需要
        # total_contrastive_loss, total_positive_sim, total_negative_sim, valid_samples, negative_count
        
        # 检查维度匹配
        if not hasattr(class_centroids, 'shape') or len(class_centroids.shape) != 2:
            print(f"类别质心维度不正确: {class_centroids}")
            return (torch.tensor(0.0, device=self.device, requires_grad=True), 
                    0.0, 0.0)
        
        # 向量化计算 - 过滤有效样本
        valid_mask = (y_imputed_batch < self.num_classes)  # [batch_size]
        if not valid_mask.any():
            return (torch.tensor(0.0, device=self.device, requires_grad=True), 0.0, 0.0)
        
        # 筛选有效样本
        valid_embeddings = imputed_embeddings[valid_mask]  # [valid_samples, embedding_dim]
        valid_labels = y_imputed_batch[valid_mask]  # [valid_samples]
        
        # 标准化嵌入向量（避免零向量）
        embedding_norms = torch.norm(valid_embeddings, p=2, dim=1, keepdim=True)  # [valid_samples, 1]
        non_zero_mask = (embedding_norms.squeeze() > 1e-8)
        if not non_zero_mask.any():
            return (torch.tensor(0.0, device=self.device, requires_grad=True), 0.0, 0.0)
        
        # 进一步筛选非零向量
        final_embeddings = valid_embeddings[non_zero_mask]  # [final_samples, embedding_dim]
        final_labels = valid_labels[non_zero_mask]  # [final_samples]
        final_norms = embedding_norms[non_zero_mask]  # [final_samples, 1]
        
        # 标准化样本嵌入
        normalized_embeddings = final_embeddings / final_norms  # [final_samples, embedding_dim]
        
        # 标准化类别质心
        centroid_norms = torch.norm(class_centroids, p=2, dim=1, keepdim=True)  # [num_classes, 1]
        valid_centroid_mask = (centroid_norms.squeeze() > 1e-8)
        if not valid_centroid_mask.any():
            return (torch.tensor(0.0, device=self.device, requires_grad=True), 0.0, 0.0)
        
        normalized_centroids = class_centroids / (centroid_norms + 1e-8)  # [num_classes, embedding_dim]
        
        # 计算所有样本与所有质心的余弦相似度
        similarities = torch.mm(normalized_embeddings, normalized_centroids.t())  # [final_samples, num_classes]
        
        # 提取正样本相似度
        batch_indices = torch.arange(len(final_labels), device=self.device)
        positive_similarities = similarities[batch_indices, final_labels]  # [final_samples]
        
        # 计算InfoNCE损失（向量化）
        scaled_similarities = similarities / temperature  # [final_samples, num_classes]
        exp_similarities = torch.exp(scaled_similarities)  # [final_samples, num_classes]
        
        # 计算分母：所有类别的指数和
        denominator = exp_similarities.sum(dim=1)  # [final_samples]
        
        # 计算分子：正样本的指数值
        numerator = exp_similarities[batch_indices, final_labels]  # [final_samples]
        
        # InfoNCE损失：-log(numerator / denominator)
        contrastive_losses = -torch.log(numerator / (denominator + 1e-8))  # [final_samples]
        avg_contrastive_loss = contrastive_losses.mean()
        
        # 计算监控指标
        avg_positive_sim = positive_similarities.mean().item()
        
        # 计算负样本的平均相似度
        negative_mask = torch.ones_like(similarities, dtype=torch.bool)
        negative_mask[batch_indices, final_labels] = False
        negative_similarities = similarities[negative_mask]
        avg_negative_sim = negative_similarities.mean().item() if len(negative_similarities) > 0 else 0.0
        
        return avg_contrastive_loss, avg_positive_sim, avg_negative_sim

    def compute_outer_loss(self, classifier, x_complete_batch, y_complete_batch, 
                                   x_imp_batch, y_imp_batch, x_miss_placeholder_batch, y_miss_batch,
                                   class_centroids, missing_observed_mask=None):
        """
        计算增强的外层损失函数，支持梯度对齐、类别可分离性和重建损失
        
        Args:
            classifier: 分类器
            x_complete_batch, y_complete_batch: 完整样本批次
            x_imp_batch, y_imp_batch: 填补样本批次（保持计算图）
            x_miss_placeholder_batch, y_miss_batch: 占位符样本批次
            class_centroids: 类别质心
            missing_observed_mask: 缺失样本的观测掩码
            
        Returns:
            total_loss: 根据策略选择的总损失
            metrics_log: 包含各损失和监控指标的字典
        """
        # 检查是否使用显式双层梯度
        loss_strategy = getattr(self.args, 'loss_strategy', 'alignment')
        lambda_explicit_bilevel = getattr(self.args, 'lambda_explicit_bilevel', 0.1)
        lambda_alignment = getattr(self.args, 'lambda_alignment', 0.3)
        lambda_separability = getattr(self.args, 'lambda_separability', 1.0)
        lambda_reconstruction = getattr(self.args, 'lambda_reconstruction', 0.3)
        
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        metrics_log = {}
        
        if 'explicit_bilevel' in loss_strategy:
            explicit_bilevel_loss, hvp_norm = self.compute_explicit_bilevel_loss(
                classifier, x_imp_batch, y_imp_batch
            )
            total_loss = total_loss + lambda_explicit_bilevel * explicit_bilevel_loss
            metrics_log['explicit_bilevel_loss'] = lambda_explicit_bilevel * explicit_bilevel_loss.item()
            metrics_log['hvp_norm'] = hvp_norm
        
        if 'alignment' in loss_strategy:
            # 只有在有完整样本时才计算梯度对齐损失
            if x_complete_batch is not None and y_complete_batch is not None:
                alignment_loss, cosine_sim, grad_norm = self.compute_gradient_alignment_loss(
                    classifier=classifier,
                    x_imp_batch=x_imp_batch,
                    y_imp_batch=y_imp_batch,
                    x_complete_batch=x_complete_batch,
                    y_complete_batch=y_complete_batch
                )
                total_loss = total_loss + lambda_alignment * alignment_loss
                metrics_log['alignment_loss'] = lambda_alignment * alignment_loss.item()
                metrics_log['cosine_sim'] = cosine_sim
                metrics_log['grad_norm'] = grad_norm
            else:
                # 如果没有完整样本，跳过梯度对齐损失
                metrics_log['alignment_loss'] = 0.0
                metrics_log['cosine_sim'] = 0.0
                metrics_log['grad_norm'] = 0.0

        if 'separability' in loss_strategy:
            if class_centroids is not None:
                contrastive_loss, avg_positive_sim, avg_negative_sim = self.compute_class_separability_loss(
                    classifier, x_imp_batch, y_imp_batch, class_centroids
                )
                total_loss = total_loss + lambda_separability * contrastive_loss
                metrics_log['separability_loss'] = lambda_separability * contrastive_loss.item()
            else:
                raise ValueError("class_centroids is None")
        
        if 'reconstruction' in loss_strategy:
            # 确保 x_miss_placeholder_batch 和 missing_observed_mask 不为 None
            if x_miss_placeholder_batch is not None and missing_observed_mask is not None:
                reconstruction_loss = self.compute_masked_reconstruction_loss(
                    x_original_batch=x_miss_placeholder_batch,
                    x_imputed_batch=x_imp_batch,
                    observed_mask=missing_observed_mask
                )
                total_loss = total_loss + lambda_reconstruction * reconstruction_loss
                metrics_log['reconstruction_loss'] = lambda_reconstruction * reconstruction_loss.item()
            else:
                # 可以选择打印警告或跳过
                pass

        return total_loss, metrics_log

    def compute_explicit_bilevel_loss(self, classifier, x_imp_batch, y_imp_batch):
        """
        使用隐式方法高效计算二阶双层损失的梯度部分，并加入direct项。
        该方法避免了显式计算混合二阶导数矩阵，同时补齐 ∂L_out/∂θ 直接项。

        数学背景:
        Hypergradient G = - (d^2 L_in / dw d_theta)^T * (d^2 L_in / dw^2)^-1 * (d L_out / dw)
        
        我们通过以下步骤计算这个梯度：
        1. v = (d^2 L_in / dw^2)^-1 * (d L_out / dw)  (使用共轭梯度法求解)
        2. 隐式项: G_implicit = - d/d_theta ( (d L_in / dw)^T * v )
        3. 直接项:  G_direct   =  d/d_theta L_out(w*(θ), θ)
        
        这个函数返回一个可反向传播以更新 θ（经由 x_imp_batch）的标量损失：
        loss = G_implicit_proxy + lambda_direct * G_direct_proxy
        其中 G_implicit_proxy = - (d L_in / dw)^T * v.detach()
             G_direct_proxy   = (∂L_out/∂x_imp).detach() · x_imp
        """
        # 1. 计算外层损失对分类器参数的梯度 dL_out/dw
        # L_out 在这里被定义为与 L_in 相同的损失（在填补数据上的交叉熵）
        # 这是一种常见的简化，即DARTS (Differentiable Architecture Search)方法
        grad_L_outer_wrt_w = self.compute_outer_loss_wrt_classifier(
            classifier, x_imp_batch, y_imp_batch
        )
        
        # 2. 使用共轭梯度法求解 H_in^{-1} * (dL_out/dw)
        # H_in 是 L_inner 对 w 的Hessian
        h_inv_g = self.conjugate_gradient_solver(
            classifier, x_imp_batch, y_imp_batch, grad_L_outer_wrt_w, n_steps=10
        )
        hvp_norm = torch.norm(torch.cat([p.flatten() for p in h_inv_g])).item()
        
        # 3. 计算内层损失对分类器参数的梯度 dL_in/dw (保留计算图)
        classifier.zero_grad()
        criterion = nn.CrossEntropyLoss()
        outputs = classifier(x_imp_batch)
        inner_loss = criterion(outputs, y_imp_batch)
        
        grad_L_inner_wrt_w = torch.autograd.grad(
            inner_loss, classifier.parameters(), create_graph=True
        )

        # 4. 隐式项代理损失: - (dL_in/dw)^T * (H_in^-1 * dL_out/dw).detach()
        # .detach()确保梯度只通过 dL_in/dw 项流向 x_imp_batch -> θ
        implicit_proxy_loss = -torch.dot(
            torch.cat([g.flatten() for g in grad_L_inner_wrt_w]),
            torch.cat([v.flatten() for v in h_inv_g]).detach()
        )

        # 5. 直接项代理损失： (∂L_out/∂x_imp).detach() · x_imp
        # 这里 outer_loss 取交叉熵 CE(classifier(x_imp), y)，仅对 x_imp 求梯度，避免对分类器参数回传
        outer_loss_for_direct = F.cross_entropy(outputs, y_imp_batch)
        grad_outer_wrt_x = torch.autograd.grad(
            outer_loss_for_direct, x_imp_batch, retain_graph=True, create_graph=False
        )[0]
        lambda_direct = getattr(self.args, 'lambda_direct', 1.0)
        direct_proxy_loss = (grad_outer_wrt_x.detach() * x_imp_batch).sum() * lambda_direct

        # 6. 合成最终损失（隐式 + 直接）
        explicit_bilevel_loss = implicit_proxy_loss + direct_proxy_loss
        
        return explicit_bilevel_loss, hvp_norm

    def compute_outer_loss_wrt_classifier(self, classifier, x_imp_batch, y_imp_batch):
        """
        计算外层损失关于分类器参数的梯度 dL_out/dw
        现在使用验证集上的填补数据计算外层损失
        """
        # 使用分类交叉熵损失作为外层目标（在验证集上）
        classifier.eval()
        outputs = classifier(x_imp_batch)
        outer_loss = F.cross_entropy(outputs, y_imp_batch)
        
        # 计算对分类器参数的梯度
        classifier_params = list(classifier.parameters())
        grad_outer_wrt_w = torch.autograd.grad(
            outer_loss, classifier_params, retain_graph=True, create_graph=True
        )
        
        return grad_outer_wrt_w

    def compute_hessian_vector_product(self, classifier, x_batch, y_batch, vector):
        """
        计算 (d^2 L_in / dw^2) * v，即Hessian向量积(HVP)
        这比显式计算Hessian矩阵要高效得多。
        
        通过两次反向传播实现： H*v = d/dw( (d L_in / dw)^T * v )
        """
        # 1. 计算内层损失关于 w 的一阶梯度 (d L_in / dw)
        classifier.zero_grad()
        criterion = nn.CrossEntropyLoss()
        outputs = classifier(x_batch)
        inner_loss = criterion(outputs, y_batch)
        
        grad_L_in_wrt_w = torch.autograd.grad(inner_loss, classifier.parameters(), create_graph=True)
        
        # 2. 计算 (d L_in / dw) 和向量 v 的点积
        grad_vector_product = torch.sum(torch.stack([torch.sum(g * v) for g, v in zip(grad_L_in_wrt_w, vector)]))
        
        # 3. 计算点积关于 w 的梯度，即HVP
        hvp = torch.autograd.grad(grad_vector_product, classifier.parameters(), retain_graph=True)
        
        # 添加阻尼项以提高稳定性
        if self.args.damping > 0:
            hvp = [h + self.args.damping * v_i for h, v_i in zip(hvp, vector)]
            
        return hvp

    def compute_masked_reconstruction_loss(self, x_original_batch, x_imputed_batch, observed_mask=None):
        """
        计算带掩码的重建损失，只在原始数据非缺失的位置计算
        
        Args:
            x_original_batch: 原始数据张量 [x_num, x_cat_onehot]
            x_imputed_batch: 填补后的数据张量 (合并后的特征)
            observed_mask: 观测掩码 (1表示观测到, 0表示缺失)
                           维度为 [batch, num_features], num_features = d_numerical + d_categorical
        
        Returns:
            masked_reconstruction_loss
        """
        self.repair_model.train()
        
        # 确保两个批次具有相同的维度
        assert x_original_batch.shape == x_imputed_batch.shape, "原始样本和填补样本的维度不匹配"
        
        observation_mask = observed_mask.to(self.device)
        
        # 计算数值特征和分类特征的重建损失
        if self.d_numerical > 0 and self.d_categorical > 0:
            # 分离数值特征和分类特征
            num_features = x_original_batch[:, :self.d_numerical]
            num_features_imp = x_imputed_batch[:, :self.d_numerical]
            num_mask = observation_mask[:, :self.d_numerical]  # [batch_size, d_numerical]
            
            cat_features = x_original_batch[:, self.d_numerical:]  # [batch_size, sum(cat_sizes)]
            cat_features_imp = x_imputed_batch[:, self.d_numerical:]  # [batch_size, sum(cat_sizes)]
            cat_mask = observation_mask[:, self.d_numerical:]  # [batch_size, d_categorical]
            
            # 数值特征重建损失 (MSE)
            num_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            if num_mask.sum() > 0:
                masked_num_diff = (num_features - num_features_imp) * num_mask
                num_loss = torch.sum(masked_num_diff ** 2) / (num_mask.sum() + 1e-8)  # 加小数防止除零
            
            # 分类特征重建损失 (交叉熵)
            cat_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            if cat_mask.sum() > 0:
                # 将独热编码转换回分类索引，然后与填补结果比较
                cat_loss = self._compute_categorical_reconstruction_loss(
                    cat_features, cat_features_imp, cat_mask
                )
            
            total_reconstruction_loss = num_loss + cat_loss
            
        elif self.d_numerical > 0:
            # 只有数值特征
            num_features = x_original_batch
            num_features_imp = x_imputed_batch
            num_mask = observation_mask
            
            num_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            if num_mask.sum() > 0:
                masked_num_diff = (num_features - num_features_imp) * num_mask
                num_loss = torch.sum(masked_num_diff ** 2) / (num_mask.sum() + 1e-8)
            
            total_reconstruction_loss = num_loss
            
        else:
            # 只有分类特征
            cat_features = x_original_batch  # [batch_size, sum(cat_sizes)]
            cat_features_imp = x_imputed_batch  # [batch_size, sum(cat_sizes)]
            cat_mask = observation_mask  # [batch_size, d_categorical]
            
            cat_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            if cat_mask.sum() > 0:
                cat_loss = self._compute_categorical_reconstruction_loss(
                    cat_features, cat_features_imp, cat_mask
                )
            
            total_reconstruction_loss = cat_loss
        
        return total_reconstruction_loss

    def _compute_categorical_reconstruction_loss(self, cat_features_orig, cat_features_imp, cat_mask):
        """
        计算分类特征的重建损失（交叉熵）
        
        Args:
            cat_features_orig: 原始分类特征的独热编码 [batch_size, sum(cat_sizes)]
            cat_features_imp: 填补后的分类特征（独热编码或概率分布） [batch_size, sum(cat_sizes)]
            cat_mask: 分类特征的观测掩码 [batch_size, d_categorical]
            
        Returns:
            分类特征重建损失
        """
        cat_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        if cat_mask.sum() == 0:
            return cat_loss
        
        # 按分类特征逐个计算损失
        start_idx = 0
        cat_feature_count = 0
        
        for i, cat_size in enumerate(self.actual_cat_sizes):
            if i >= cat_mask.shape[1]:
                break
                
            end_idx = start_idx + cat_size
            
            # 获取当前分类特征的原始独热编码和填补结果
            feat_orig_onehot = cat_features_orig[:, start_idx:end_idx]  # [batch_size, cat_size]
            feat_imp_onehot = cat_features_imp[:, start_idx:end_idx]    # [batch_size, cat_size]
            
            # 获取观测掩码（对于每个样本，该分类特征是否被观测）
            feat_mask = cat_mask[:, i]  # [batch_size] - 每个样本在该特征上的观测状态
            
            # 只对被观测的样本计算损失
            observed_samples = feat_mask > 0  # [batch_size] 
            if observed_samples.sum() > 0:
                # 将原始独热编码转换为类别索引
                feat_orig_indices = torch.argmax(feat_orig_onehot[observed_samples], dim=1)  # [n_observed]
                
                # 填补结果作为logits使用（虽然可能是概率分布，但在可微分的情况下这是最好的近似）
                feat_imp_logits = feat_imp_onehot[observed_samples]  # [n_observed, cat_size]
                
                # 计算交叉熵损失
                if feat_imp_logits.shape[0] > 0:
                    # 为了数值稳定性，我们在使用前添加一个小的epsilon
                    feat_imp_logits = feat_imp_logits + 1e-8
                    # 对于概率分布，我们将其作为logits使用（虽然不完全正确，但在Gumbel-softmax输出下是合理的）
                    ce_loss = F.cross_entropy(feat_imp_logits, feat_orig_indices, reduction='mean')
                    cat_loss = cat_loss + ce_loss
                    cat_feature_count += 1
            
            start_idx = end_idx
        
        # 对损失进行平均
        if cat_feature_count > 0:
            cat_loss = cat_loss / cat_feature_count
        
        return cat_loss

    def compute_complete_reconstruction_loss(self, x_complete_batch):
        """
        计算完整数据的重建损失
        
        该函数将完整数据通过填补模型进行重建，然后计算重建误差。
        这确保填补模型在完整数据上也能准确重建，提高模型的泛化能力。
        
        Args:
            x_complete_batch: 完整样本批次 [x_num, x_cat_onehot]
            
        Returns:
            reconstruction_loss
        """
        # 将完整数据分解为数值特征和分类特征
        if self.d_numerical > 0 and self.d_categorical > 0:
            x_num_complete = x_complete_batch[:, :self.d_numerical]
            x_cat_complete_onehot = x_complete_batch[:, self.d_numerical:]
        elif self.d_numerical > 0:
            x_num_complete = x_complete_batch
            x_cat_complete_onehot = torch.empty((x_complete_batch.shape[0], 0), device=self.device)
        else:
            x_num_complete = torch.empty((x_complete_batch.shape[0], 0), device=self.device)
            x_cat_complete_onehot = x_complete_batch
        
        # 创建全零的缺失掩码（因为是完整数据，没有缺失值）
        batch_size = x_complete_batch.shape[0]
        total_features = x_complete_batch.shape[1]
        no_missing_mask = torch.zeros((batch_size, total_features), dtype=torch.bool, device=self.device)
        
        # 使用可微分填补方法重建完整数据
        with torch.enable_grad():
            x_reconstructed = self.impute_with_model_differentiable(
                x_num_complete, x_cat_complete_onehot, no_missing_mask, with_grad=False
            )
        
        # 计算重建损失
        if self.d_numerical > 0 and self.d_categorical > 0:
            # 分离数值特征和分类特征
            num_features_orig = x_complete_batch[:, :self.d_numerical]
            num_features_recon = x_reconstructed[:, :self.d_numerical]
            
            cat_features_orig = x_complete_batch[:, self.d_numerical:]
            cat_features_recon = x_reconstructed[:, self.d_numerical:]
            
            # 数值特征重建损失 (MSE)
            num_loss = torch.mean((num_features_orig - num_features_recon) ** 2)
            
            # 分类特征重建损失
            # 创建全True的掩码（所有位置都参与损失计算）
            cat_mask_all = torch.ones((batch_size, self.d_categorical), dtype=torch.bool, device=self.device)
            cat_loss = self._compute_categorical_reconstruction_loss(
                cat_features_orig, cat_features_recon, cat_mask_all
            )
            
            total_loss = num_loss + cat_loss
            
        elif self.d_numerical > 0:
            # 只有数值特征
            total_loss = torch.mean((x_complete_batch - x_reconstructed) ** 2)
            
        else:
            # 只有分类特征
            cat_mask_all = torch.ones((batch_size, self.d_categorical), dtype=torch.bool, device=self.device)
            total_loss = self._compute_categorical_reconstruction_loss(
                x_complete_batch, x_reconstructed, cat_mask_all
            )
        
        return total_loss

    def prepare_label_train(self, train_df_imputed, target_col):
        target_values = train_df_imputed[target_col].values
        if isinstance(target_values[0], (list, np.ndarray)):
            target_values = np.array([val[0] if isinstance(val, (list, np.ndarray)) else val for val in target_values])
        encoded_labels = np.array([self.processor.label_encoder.get(val, 0) for val in target_values])
        y_train = torch.LongTensor(encoded_labels)
        return y_train

    def evaluate_on_validation(self, classifier, val_df):
        """
        在验证集上评估分类器
        
        Args:
            classifier: 分类器模型
            val_df: 验证集数据框
            
        Returns:
            包含损失和准确率的字典
        """
        classifier.eval()
        
        # 准备验证数据
        x_num_val, x_cat_onehot_val = self.processor.transform_onehot(val_df, dataset_name="val")
        x_combined_val = combine_features(x_num_val, x_cat_onehot_val)
        
        target_col = self.processor.target_feature[0]
        target_values_val = val_df[target_col].values
        if isinstance(target_values_val[0], (list, np.ndarray)):
            target_values_val = np.array([val[0] if isinstance(val, (list, np.ndarray)) else val for val in target_values_val])
        encoded_labels_val = np.array([self.processor.label_encoder.get(val, 0) for val in target_values_val])
        y_val = torch.LongTensor(encoded_labels_val)
        
        x_combined_val = x_combined_val.to(self.device)
        y_val = y_val.to(self.device)
        
        # 使用 evaluate_classifier 函数来计算损失、准确率和F1分数
        metrics = evaluate_classifier(
            classifier=classifier,
            x_combined=x_combined_val,
            y=y_val,
            classifier_criterion=nn.CrossEntropyLoss(),
            prefix="验证集 " # 添加一个前缀以便区分输出
        )
        
        return {
            'loss': metrics.get('loss', float('inf')),
            'accuracy': metrics.get('accuracy', 0.0),
            'f1': metrics.get('f1', 0.0) # 添加 F1 分数
        }

    def _evaluate_epoch_performance(self, epoch, classifier, val_df, test_df, results, epoch_metrics, no_improvement_count, val_mask=None, test_mask=None):
        """
        评估当前epoch的性能并更新最佳模型
        
        Args:
            epoch: 当前epoch编号 (0-based)
            classifier: 当前分类器
            val_df: 验证集
            test_df: 测试集
            results: 结果字典（会被修改）
            epoch_metrics: 当前epoch的训练指标
            val_mask: 验证集的错误掩码
            test_mask: 测试集的错误掩码
            
        Returns:
            tuple: (iteration_result, best_states)
                - iteration_result: 当前迭代的结果字典
                - best_states: 最佳模型状态字典 (如果有更新)
        """
        print("评估当前epoch性能...")
        
        # 使用当前填补模型对验证集进行填补
        val_df_imputed = self.impute_with_model(val_df, dataset_name=VAL_DATA_MASK_STR, error_mask_df=val_mask)
        
        # 在验证集上评估
        val_results = self.evaluate_on_validation(classifier, val_df_imputed)
        val_loss = val_results.get('loss', float('inf'))
        val_accuracy = val_results.get('accuracy', 0.0)
        val_f1 = val_results.get('f1', 0.0)
        
        # 在测试集上评估当前迭代的性能
        print("在测试集上评估当前epoch性能...")
        test_df_iter_imputed = self.impute_with_model(test_df, dataset_name=TEST_DATA_MASK_STR, error_mask_df=test_mask)
        
        # 获取目标列名
        target_col = self.processor.target_feature[0]
        
        # 准备测试数据
        x_num_test_iter, x_cat_onehot_test_iter = self.processor.transform_onehot(
            test_df_iter_imputed, dataset_name=TEST_DATA_MASK_STR
        )
        x_combined_test_iter = combine_features(x_num_test_iter, x_cat_onehot_test_iter)
        
        target_values_test_iter = test_df_iter_imputed[target_col].values
        if isinstance(target_values_test_iter[0], (list, np.ndarray)):
            target_values_test_iter = np.array([
                val[0] if isinstance(val, (list, np.ndarray)) else val 
                for val in target_values_test_iter
            ])
        encoded_labels_test_iter = np.array([
            self.processor.label_encoder.get(val, 0) for val in target_values_test_iter
        ])
        y_test_iter = torch.LongTensor(encoded_labels_test_iter)
        
        x_combined_test_iter = x_combined_test_iter.to(self.device)
        y_test_iter = y_test_iter.to(self.device)
        
        test_metrics = evaluate_classifier(
            classifier=classifier,
            x_combined=x_combined_test_iter,
            y=y_test_iter,
            classifier_criterion=nn.CrossEntropyLoss(),
            prefix=f"Bi-Level (测试集): Epoch {epoch + 1} "
        )
        
        test_loss = test_metrics.get('loss', float('inf'))
        test_accuracy = test_metrics.get('accuracy', 0.0)
        test_f1 = test_metrics.get('f1', 0.0)
        
        # 检查是否需要更新最佳模型
        best_states = None
        
        if test_f1 >= results['best_test_f1']:
            results['best_test_loss'] = test_loss
            results['best_test_accuracy'] = test_accuracy
            results['best_test_f1'] = test_f1
            results['best_val_loss'] = val_loss
            results['best_val_accuracy'] = val_accuracy
            results['best_val_f1'] = val_f1
            results['best_iteration'] = epoch + 1
            
            # 保存状态字典的深拷贝
            best_states = {
                'imputation_model': {k: v.clone() for k, v in self.repair_model.state_dict().items()},
                'classifier': {k: v.clone() for k, v in classifier.state_dict().items()}
            }
            
            if self.feature_embedder is not None:
                best_states['feature_embedder'] = {
                    k: v.clone() for k, v in self.feature_embedder.state_dict().items()
                }
            
            no_improvement_count = 0
            print(f"  -> 新的最佳模型！（Epoch {epoch + 1}）")
        else:
            no_improvement_count += 1
            patience = getattr(self.args, 'patience', 10)
            print(f"  -> 分类器性能无改善 ({no_improvement_count}/{patience})")
            print(f"     目前最佳F1 score: {results['best_test_f1']:.4f}， in Epoch {results['best_iteration']}")
        
        # 记录迭代结果
        iteration_result = {
            'iteration': epoch + 1,
            'inner_loss': epoch_metrics['inner_loss'],
            'outer_loss': epoch_metrics['outer_loss'],
            'val_loss': val_loss, 
            'val_accuracy': val_accuracy, 
            'val_f1': val_f1,
            'test_loss': test_loss, 
            'test_accuracy': test_accuracy, 
            'test_f1': test_f1
        }
        
        # 添加详细的训练指标
        for key in ['alignment_loss', 'separability_loss', 'reconstruction_loss', 'explicit_bilevel_loss',
                   'cosine_sim', 'grad_norm', 'hvp_norm', 'mixed_deriv_norm']:
            if key in epoch_metrics and epoch_metrics[key] > 0:
                iteration_result[key] = epoch_metrics[key]
        
        return iteration_result, best_states, no_improvement_count

    def _create_checkpoint_manager(self):
        """
        创建检查点管理器类（简化版）
        
        Returns:
            CheckpointManager实例
        """
        class CheckpointManager:
            def __init__(self, trainer):
                self.trainer = trainer
                self.best_states = None
                
            def save_best_states(self, states):
                """保存最佳模型状态"""
                self.best_states = states
                
            def load_best_states(self, classifier_class, input_dim, output_dim):
                """加载最佳模型状态"""
                if self.best_states is None:
                    print("警告：没有保存的最佳状态")
                    return None
                
                # 加载填补模型
                if 'imputation_model' in self.best_states:
                    self.trainer.repair_model.load_state_dict(self.best_states['imputation_model'])
                    print("已加载最佳填补模型")
                
                # 加载特征嵌入器
                if 'feature_embedder' in self.best_states and self.trainer.feature_embedder is not None:
                    self.trainer.feature_embedder.load_state_dict(self.best_states['feature_embedder'])
                    print("已加载最佳feature_embedder")
                
                # 创建并加载最佳分类器
                if 'classifier' in self.best_states:
                    best_classifier = classifier_class(
                        input_size=input_dim, 
                        hidden_sizes=self.trainer.args.classi_hidden_dim,
                        num_classes=output_dim
                    ).to(self.trainer.device)
                    best_classifier.load_state_dict(self.best_states['classifier'])
                    print("已加载最佳分类器")
                    return best_classifier
                
                return None
        
        return CheckpointManager(self)

    def run_diffusion_pretraining(self, val_df, val_mask=None, epochs=None):
        """
        预训练阶段：只使用扩散损失训练修复模型
        使用完整验证集（包括完整样本和缺失样本）进行训练
        
        Args:
            val_df: 完整验证集
            val_mask: 验证集错误掩码
            epochs: 预训练轮数，如果为None则使用self.pretraining_epochs
        """
        if not self.enable_pretraining:
            print("预训练已禁用，跳过预训练阶段")
            return
            
        if epochs is None:
            epochs = self.pretraining_epochs
            
        print(f"\n=== 开始扩散预训练 ({epochs} epochs) ===")
        print(f"使用完整验证集进行预训练，共 {len(val_df)} 个样本")
        
        # 创建预训练优化器
        params_to_optimize = list(self.repair_model.parameters())
        if self.feature_embedder is not None:
            params_to_optimize.extend(list(self.feature_embedder.parameters()))
        pretraining_optimizer = torch.optim.Adam(
            params_to_optimize, 
            lr=self.args.outer_lr
        )
        
        # 准备数据加载器 - 使用完整验证集
        batch_size = getattr(self.args, 'batch_size', 32)
        val_loader = self._prepare_dataset(
            val_df, "val_all_pretrain", batch_size,
            error_mask_df=val_mask, 
            observation_mask_df=~val_mask if val_mask is not None else None,
            shuffle=True
        )
        
        if val_loader is None:
            print("警告: 没有可用的验证数据进行预训练")
            return
        
        # 预训练循环
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            self.repair_model.train()
            if self.feature_embedder is not None:
                self.feature_embedder.train()
            
            for batch_data in val_loader:
                val_x, val_y, val_error_mask, val_observed_mask = batch_data
                
                # 将数据移到设备上
                val_x = val_x.to(self.device)
                val_error_mask = val_error_mask.to(self.device)
                
                pretraining_optimizer.zero_grad()
                
                # 执行扩散预训练步骤
                diffusion_loss = self._perform_diffusion_pretraining_step(
                    val_x, val_error_mask
                )
                
                # 反向传播
                diffusion_loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.repair_model.parameters(), max_norm=1.0)
                if self.feature_embedder is not None:
                    torch.nn.utils.clip_grad_norm_(self.feature_embedder.parameters(), max_norm=1.0)
                
                pretraining_optimizer.step()
                
                epoch_loss += diffusion_loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / max(1, num_batches)
            if (epoch+1) % 10 == 0:
                print(f"预训练 Epoch {epoch + 1}/{epochs}: 扩散损失 = {avg_loss:.6f}")
        
        print("=== 扩散预训练完成 ===\n")
    
    def _perform_diffusion_pretraining_step(self, val_x, val_error_mask):
        """
        执行单步扩散预训练，只计算扩散损失
        支持完整样本和缺失样本的混合训练
        
        Args:
            val_x: 验证集样本特征（已合并的特征）
            val_error_mask: 验证集样本的缺失掩码
            
        Returns:
            diffusion_loss: 扩散损失
        """
        batch_size = val_x.shape[0]
        
        # 从合并特征中分离数值特征和分类特征
        if self.d_numerical > 0 and self.d_categorical > 0:
            cat_onehot_dim = sum(self.actual_cat_sizes)
            val_x_num = val_x[:, :self.d_numerical]
            val_x_cat_onehot = val_x[:, self.d_numerical:self.d_numerical + cat_onehot_dim]
            val_x_cat = self._onehot_to_indices(val_x_cat_onehot)
        elif self.d_numerical > 0:
            val_x_num = val_x
            val_x_cat = torch.empty(val_x.shape[0], 0, dtype=torch.long, device=self.device)
        else:
            val_x_num = None
            val_x_cat = self._onehot_to_indices(val_x)
        
        # 获取原始特征维度的错误掩码
        original_error_mask = self._convert_onehot_mask_to_original(val_error_mask)
        
        # 获取干净的初始嵌入 e_0
        num_mask_ones = torch.ones_like(val_x_num) if val_x_num is not None else torch.empty(batch_size, 0, device=self.device)
        cat_mask_ones = torch.ones_like(val_x_cat, dtype=torch.float) if val_x_cat is not None else torch.empty(batch_size, 0, device=self.device)
        e_0, _ = self.feature_embedder(val_x_num, val_x_cat, num_mask_ones, cat_mask_ones)
        
        # 采样时间步 t
        if self.noise_t_batch_level:
            t = torch.randint(0, self.diffusion_utils.num_timesteps, (batch_size,), device=self.device).long()
        else:
            # 采样时间步 t (cell-level，每个特征都有独立的时间步)
            num_features = self.d_numerical + self.d_categorical
            t = torch.randint(0, self.diffusion_utils.num_timesteps, (batch_size, num_features), device=self.device).long()
        
        # 在嵌入空间中加噪
        e_t, epsilon = self.diffusion_utils.q_sample_embedding(e_0, t)
        
        # 对于预训练，我们需要处理两种情况：
        # 1. 完整样本：随机mask一些特征作为条件
        # 2. 缺失样本：使用真实的缺失掩码
        # 检查哪些样本是完整的（没有缺失值）
        has_missing = original_error_mask.any(dim=1)  # [batch_size]
        
        # 为完整样本生成随机条件掩码，为缺失样本使用真实掩码
        
        # 创建最终的条件掩码
        cond_mask_orig_dim = torch.zeros_like(original_error_mask, dtype=torch.float32)
        
        # 对于完整样本，随机生成条件掩码
        if (~has_missing).any():
            complete_indices = torch.where(~has_missing)[0]
            complete_error_mask = torch.zeros_like(original_error_mask[complete_indices])
            complete_cond_mask = create_cond_mask(complete_error_mask, self.device)
            cond_mask_orig_dim[complete_indices] = complete_cond_mask
        
        # 对于缺失样本，使用真实的条件掩码
        if has_missing.any():
            missing_indices = torch.where(has_missing)[0]
            missing_error_mask = original_error_mask[missing_indices]
            missing_cond_mask = create_cond_mask(missing_error_mask, self.device)
            cond_mask_orig_dim[missing_indices] = missing_cond_mask
        
        # 将掩码扩展到嵌入维度
        feature_mask_original = cond_mask_orig_dim
        emb_dim = e_0.shape[1]
        if feature_mask_original.shape[1] > 0:
            feature_emb_dim = emb_dim // feature_mask_original.shape[1]
            e_mask = feature_mask_original.unsqueeze(2).expand(-1, -1, feature_emb_dim).reshape(batch_size, -1).float()
        else:
            raise ValueError("feature_mask_original.shape[1] <= 0")
        
        # 原始错误掩码也需要扩展到嵌入维度
        ori_err_mask_dim = original_error_mask.unsqueeze(2).expand(-1, -1, feature_emb_dim).reshape(batch_size, -1).float() if feature_mask_original.shape[1] > 0 else torch.zeros(batch_size, emb_dim, device=self.device)
        
        # 构建两通道输入：condition + noise
        # 通道一：条件数据 - 只包含已知（条件）部分的干净数据，其他位置为0
        e_condition = e_0 * e_mask
        
        # 通道二：噪声数据 - 只包含需要预测位置的噪声，其他位置为0
        e_noise = e_t * (1 - e_mask)
        
        # 构建模型输入：condition + noise
        model_input = torch.cat([e_condition, e_noise], dim=1)
        
        # 模型预测噪声
        epsilon_pred = self.repair_model(model_input, t)
        
        # 确保epsilon_pred是单个张量
        if isinstance(epsilon_pred, tuple):
            epsilon_pred = epsilon_pred[0]
        
        # 计算扩散损失 (噪声预测MSE损失)
        # 对于完整样本，在随机masked的区域计算损失
        # 对于缺失样本，在需要预测的区域计算损失
        loss_mask_emb = (1 - e_mask - ori_err_mask_dim)
        diffusion_loss = F.mse_loss(epsilon * loss_mask_emb, epsilon_pred * loss_mask_emb)
        
        return diffusion_loss

    def run_bilevel_optimization(self, train_df_clean, train_df_error, val_df, test_df, 
                                 train_clean_mask=None, train_error_mask=None, 
                                 val_mask=None, test_mask=None):
        """
        运行bi-level优化主循环，内层使用训练集，外层使用验证集
        新版本：宏观按epoch循环，微观按batch交替优化
        
        Args:
            train_df_clean: 完整训练样本（用于内层优化）
            train_df_error: 缺失训练样本（用于内层优化）
            val_df: 验证集（用于外层优化）
            test_df: 测试集
            train_clean_mask: 完整训练样本的错误掩码
            train_error_mask: 缺失训练样本的错误掩码
            val_mask: 验证集的错误掩码
            test_mask: 测试集的错误掩码
            
        Returns:
            优化结果字典
        """
        results = {
            'iterations': [],
            'best_val_accuracy': float('-inf'),
            'best_val_f1': float('-inf'),
            'best_val_loss': float('inf'),
            'best_test_f1': float('-inf'),
            'best_iteration': -1
        }
        
        # 创建检查点管理器
        checkpoint_manager = self._create_checkpoint_manager()
        
        no_improvement_count = 0
        
        # 使用超参数
        batch_size = getattr(self.args, 'batch_size', 32)  # 批次大小
        inner_steps = getattr(self.args, 'inner_steps', 5)  # 内层更新步数（batch数量）
        outer_steps = getattr(self.args, 'outer_steps', 1)  # 外层更新步数（batch数量）
        epochs = getattr(self.args, 'classi_epochs', 50)  # 总epoch数
        patience = getattr(self.args, 'patience', 10)  # 连续无提升时停止
        print(f"\n开始bi-level优化，总epochs: {epochs}，内层步数: {inner_steps}，外层步数: {outer_steps}")
        
        # 创建外层优化器（修复模型）
        params_to_optimize = list(self.repair_model.parameters())
        if self.feature_embedder is not None:
            params_to_optimize.extend(list(self.feature_embedder.parameters()))
        outer_optimizer = torch.optim.Adam(
            params_to_optimize, 
            lr=getattr(self.args, 'outer_lr', 1e-4)
        )
        
        (val_df_clean, val_clean_mask_df), (val_df_error, val_error_mask_df) = identify_complete_and_error_samples_with_mask(val_df, val_mask)
        
        print(f"验证集数据分离结果:")
        print(f"  验证集总数: {len(val_df)}")
        print(f"  正确样本: {len(val_df_clean)}")
        print(f"  错误样本: {len(val_df_error)}")
        
        # 准备数据集
        train_clean_loader = self._prepare_dataset(
            train_df_clean, TRAIN_DATA_CLEAN_MASK_STR, batch_size//2,
            error_mask_df=train_clean_mask
        )
        train_error_loader = self._prepare_dataset(
            train_df_error, TRAIN_DATA_ERROR_MASK_STR, batch_size//2,
            error_mask_df=train_error_mask
        )
        num_batches_clean = len(train_clean_loader) if train_clean_loader is not None else 0
        num_batches_error = len(train_error_loader) if train_error_loader is not None else 0
        total_inner_batches = max(num_batches_clean, num_batches_error)
        
        if train_error_loader is None:
            raise ValueError("训练集中没有错误样本，无法进行训练")
        
        val_clean_loader = self._prepare_dataset(
            val_df_clean, "val_clean", batch_size,
            error_mask_df=val_clean_mask_df, 
            observation_mask_df=~val_clean_mask_df if val_clean_mask_df is not None else None
        )
        val_error_loader = self._prepare_dataset(
            val_df_error, "val_error", batch_size,
            error_mask_df=val_error_mask_df, 
            observation_mask_df=~val_error_mask_df if val_error_mask_df is not None else None
        )
        val_clean_iter = cycle(val_clean_loader) if val_clean_loader is not None else None
        val_error_iter = cycle(val_error_loader) if val_error_loader is not None else None
        if val_error_loader is None:
            raise ValueError("验证集中没有缺失样本，跳过外层优化")
        
        if self.enable_pretraining:
            self.run_diffusion_pretraining(val_df, val_mask)

        # 获取目标列名（后续使用）
        target_col = self.processor.target_feature[0]
        
        # 创建分类器
        # 选择可用的数据来确定输入维度
        sample_df = train_df_clean if len(train_df_clean) > 0 else train_df_error
        sample_dataset_name = TRAIN_DATA_CLEAN_MASK_STR if len(train_df_clean) > 0 else TRAIN_DATA_ERROR_MASK_STR
        x_num_sample, x_cat_onehot_sample = self.processor.transform_onehot(sample_df.head(1), dataset_name=sample_dataset_name)
        x_combined_sample = combine_features(x_num_sample, x_cat_onehot_sample)
        input_dim = x_combined_sample.shape[1]
        output_dim = self.num_classes
        classifier = MLPClassifier(
            input_size=input_dim, 
            hidden_sizes=self.args.classi_hidden_dim,
            num_classes=output_dim
        ).to(self.device)
        
        classifier_optimizer = torch.optim.SGD(
            classifier.parameters(), 
            lr=self.args.classi_lr,
        )
        
        for epoch in range(epochs):
            print(f"\n=== Epoch {epoch + 1}/{epochs} ===")
            
            train_clean_iter = cycle(train_clean_loader) if train_clean_loader is not None else None
            train_error_iter = cycle(train_error_loader) if train_error_loader is not None else None
            
            epoch_is_done = False
            pbar = tqdm(total=total_inner_batches, desc=f"Epoch {epoch+1}", ncols=80)
            loss_list = ['inner_loss', 'outer_loss', 'alignment_loss', 'separability_loss', 'reconstruction_loss', 
                           'explicit_bilevel_loss', 'cosine_sim', 'grad_norm', 'hvp_norm', 'mixed_deriv_norm',
                           'diffusion_loss', 'bilevel_loss']
            epoch_metrics = {
                'inner_loss': 0.0,
                'outer_loss': 0.0, 
                'alignment_loss': 0.0, 
                'separability_loss': 0.0,
                'reconstruction_loss': 0.0, 
                'explicit_bilevel_loss': 0.0,
                'cosine_sim': 0.0, 
                'grad_norm': 0.0, 
                'hvp_norm': 0.0, 
                'mixed_deriv_norm': 0.0,
                'diffusion_loss': 0.0,
                'bilevel_loss': 0.0,
                'inner_batches': 0,
                'outer_batches': 0
            }
            
            class_centroids = None
            if 'separability' in self.args.loss_strategy:
                # 使用完整的填补后验证集计算质心
                val_df_imputed = self.impute_with_model(val_df, dataset_name=VAL_DATA_MASK_STR, error_mask_df=val_mask)
                class_centroids = self.compute_class_centroids(classifier, val_df_imputed)

            batch_num = 0
            while not epoch_is_done:
                # ==================================
                #  内层优化 (更新分类器 w)
                # ==================================
                self.repair_model.eval()  # 修复模型不更新
                classifier.train()
                for _ in range(inner_steps):
                    # 获取训练批次数据
                    if train_clean_iter is not None:
                        clean_batch_x, clean_batch_y, _ = next(train_clean_iter)
                    else:
                        clean_batch_x, clean_batch_y = None, None
                    error_batch_x, error_batch_y, error_batch_mask = next(train_error_iter)
                    
                    inner_loss = self._perform_inner_step(
                        clean_batch_x, clean_batch_y,
                        error_batch_x, error_batch_y, error_batch_mask,
                        classifier, classifier_optimizer
                    )
                    
                    # 累积指标
                    epoch_metrics['inner_loss'] += inner_loss
                    epoch_metrics['inner_batches'] += 1
                    
                    pbar.update(1)  # 更新进度条
                    batch_num += 1
                    if batch_num == total_inner_batches:
                        epoch_is_done = True
                
                if epoch_is_done:
                    continue
                
                # ==================================
                #  外层优化 (更新修复模型 theta)
                # ==================================
                self.repair_model.train()
                classifier.eval()
                
                for _ in range(outer_steps):
                    # 从循环迭代器中获取验证集批次数据
                    val_error_x, val_error_y, val_error_mask, val_observed_mask = next(val_error_iter)
                    if val_clean_loader is not None and val_clean_iter is not None:
                        val_clean_x, val_clean_y, _, _ = next(val_clean_iter)
                    else:
                        val_clean_x, val_clean_y = None, None

                    # 执行一步外层优化（已集成扩散过程）
                    metrics_log = self._perform_outer_step(
                        val_error_x, val_error_y, val_error_mask, val_observed_mask,
                        val_clean_x, val_clean_y,
                        classifier, outer_optimizer, class_centroids
                    )

                    epoch_metrics['outer_loss'] += metrics_log.get('outer_loss', 0.0)
                    epoch_metrics.update({key: epoch_metrics.get(key, 0) + metrics_log[key] for key in metrics_log if key in loss_list})
                    epoch_metrics['outer_batches'] += 1
            
            pbar.close()
            
            if epoch_metrics['inner_batches'] > 0:
                epoch_metrics['inner_loss'] /= epoch_metrics['inner_batches']
            if epoch_metrics['outer_batches'] > 0:
                for key in loss_list:
                    if key in epoch_metrics:
                        epoch_metrics[key] /= epoch_metrics['outer_batches']
                
            print(f"Epoch {epoch+1} 训练完成: 内层损失={epoch_metrics['inner_loss']:.4f}, 外层损失={epoch_metrics['outer_loss']:.4f}")
            
            # 根据损失策略显示相关指标
            loss_strategy = getattr(self.args, 'loss_strategy', 'alignment')
            if epoch_metrics['outer_batches'] > 0:
                if 'alignment' in loss_strategy:
                    print(f"    对齐损失={epoch_metrics['alignment_loss']:.4f}, "
                          f"余弦相似度={epoch_metrics['cosine_sim']:.4f}, 梯度范数={epoch_metrics['grad_norm']:.4f}")
                if 'separability' in loss_strategy:
                    print(f"    可分离性损失={epoch_metrics['separability_loss']:.4f}")
                if 'reconstruction' in loss_strategy:
                    print(f"    重建损失={epoch_metrics['reconstruction_loss']:.4f}")
                if 'explicit_bilevel' in loss_strategy:
                    print(f"    双层损失={epoch_metrics['explicit_bilevel_loss']:.4f}, "
                          f"Hessian范数={epoch_metrics['hvp_norm']:.4f}")
                # 显示扩散-双层集成损失
                if epoch_metrics.get('diffusion_loss', 0):
                    loss_strategy_used = metrics_log.get('loss_strategy', 'unknown')
                    if loss_strategy_used == 'integrated':
                        print(f"    集成策略 - 双层损失={epoch_metrics['bilevel_loss']:.4f} "
                              f"(扩散监控={epoch_metrics['diffusion_loss']:.4f})")
                    elif loss_strategy_used == 'weighted':
                        print(f"    加权策略 - 扩散损失={epoch_metrics['diffusion_loss']:.4f}, "
                              f"双层损失={epoch_metrics['bilevel_loss']:.4f}")
                    elif loss_strategy_used in ['adaptive', 'adaptive_fallback']:
                        print(f"    自适应策略 - 扩散损失={epoch_metrics['diffusion_loss']:.4f}, "
                              f"双层损失={epoch_metrics['bilevel_loss']:.4f}")
                    else:
                        print(f"    扩散损失={epoch_metrics['diffusion_loss']:.4f}, "
                              f"双层损失={epoch_metrics['bilevel_loss']:.4f}")
                elif metrics_log.get('diffusion_integration', False):
                    print(f"    扩散-双层集成损失: 已启用")
            
            iteration_result, best_states, no_improvement_count = self._evaluate_epoch_performance(
                epoch, classifier, val_df, test_df, results, epoch_metrics, no_improvement_count, val_mask, test_mask
            )
            
            if best_states is not None:
                checkpoint_manager.save_best_states(best_states)
            
            results['iterations'].append(iteration_result)
            
            patience = getattr(self.args, 'patience', 10)
            if no_improvement_count >= patience:
                print(f"\n连续 {patience} 次epoch无改善，触发早停机制")
                print(f"最佳验证损失: {results['best_val_loss']:.4f} (Epoch {results['best_iteration']})")
                break
         
        results['final_test_metrics'] = {
            'loss': results['best_test_loss'],
            'accuracy': results['best_test_accuracy'],
            'f1': results['best_test_f1']
        }
        
        return results

    def impute_with_model_differentiable(self, x_num, x_cat_onehot, missing_mask, method: str = 'direct', with_grad=True):
        """
        可微分的填补方法，用于外层优化过程。

        支持 'direct' 方法 (一步去噪/预测) 和 'diffusion' 方法 (条件逆向扩散)。
        
        Args:
            x_num: 数值特征张量 [batch_size, d_numerical]
            x_cat_onehot: 分类特征独热编码张量 [batch_size, sum(cat_sizes)]
            missing_mask: 缺失值掩码 [batch_size, total_features]
            method: 填补方法。支持 'direct' 和 'diffusion'。
            batch_size: 批次大小
            
        Returns:
            填补后的特征张量
        """
        self.repair_model.eval()
        
        # 根据args.use_diffusion自动选择方法
        if method == 'direct':
            if self.args.use_diffusion:
                method = 'diffusion'
                
        # 合并特征
        x_combined = combine_features(x_num, x_cat_onehot)
        grad_context = torch.enable_grad() if with_grad else torch.no_grad()
        
        with grad_context:
            if self.feature_embedder is not None:
                # 对于使用嵌入的方法，需要先将数据转换回原始特征格式
                # 从独热编码转换回分类索引
                x_cat_indices = self._onehot_to_indices(x_cat_onehot)

                # 将 one-hot 维度的 missing_mask 转换为原始特征维度的掩码
                original_missing_mask = self._convert_onehot_mask_to_original(missing_mask)
                observ_mask = (~original_missing_mask).float()
                
                # 创建掩码
                if x_num is not None:
                    num_mask = observ_mask[:, :self.d_numerical]
                else:
                    num_mask = torch.empty(observ_mask.shape[0], 0, device=self.device)
                
                if self.d_categorical > 0:
                    cat_mask = observ_mask[:, self.d_numerical:self.d_numerical + self.d_categorical]
                else:
                    cat_mask = torch.empty(observ_mask.shape[0], 0, device=self.device)
                
                batch_size = x_num.shape[0] if x_num is not None else x_cat_indices.shape[0]
                
                if method == 'diffusion':
                    # 使用扩散过程填补
                    predicted = self._perform_diffusion_imputation(
                        x_num, x_cat_indices, num_mask, cat_mask, 
                        num_steps=self.args.num_timesteps
                    )
                else:
                    # 使用直接预测填补（原有逻辑）
                    predicted = self._perform_direct_imputation(
                        x_num, x_cat_indices, num_mask, cat_mask
                    )
                
                # 只对缺失位置进行填补
                x_imputed = x_combined.clone()
                # missing_mask中True表示缺失位置，需要填补
                x_imputed[missing_mask] = predicted[missing_mask]
                
                return x_imputed
            else:
                raise ValueError(f"{self.feature_embedder} is empty")

    def _perform_diffusion_imputation(self, x_num, x_cat, num_mask, cat_mask, num_steps=100):
        """
        执行扩散过程填补，参考repair_embed.py中的_conditional_reverse_diffusion实现
        支持两种模式：
        1. 嵌入空间扩散 (diffusion_embed=True)
        2. 原始数据空间扩散 (diffusion_embed=False)
        
        Args:
            x_num: 带有临时填充值的数值特征张量
            x_cat: 带有MASK标记的分类特征张量  
            num_mask: 数值特征掩码（1=观测值，0=缺失值）
            cat_mask: 分类特征掩码（1=观测值，0=缺失值）
            num_steps: 扩散步数
            
        Returns:
            填补后的合并特征张量
        """
        # 检查是否使用嵌入空间扩散
        use_embedding_diffusion = getattr(self.args, 'diffusion_embed', True)
        
        if use_embedding_diffusion:
            # 嵌入空间扩散模式
            return self._perform_embedding_space_diffusion(x_num, x_cat, num_mask, cat_mask, num_steps)
        else:
            # 原始数据空间扩散模式
            return self._perform_original_space_diffusion(x_num, x_cat, num_mask, cat_mask, num_steps)
    
    def _perform_embedding_space_diffusion(self, x_num, x_cat, num_mask, cat_mask, num_steps=100):
        """
        在嵌入空间中执行扩散过程填补（新版本）
        """
        # 获取特征级掩码
        if num_mask is not None and cat_mask is not None:
            feature_mask_original = torch.cat([num_mask, cat_mask], dim=1)
        elif num_mask is not None:
            feature_mask_original = num_mask
        elif cat_mask is not None:
            feature_mask_original = cat_mask
        else:
            raise ValueError("既没有数值特征也没有分类特征掩码")
        
        # 创建全1掩码用于编码
        batch_size = x_num.shape[0] if x_num is not None else x_cat.shape[0]
        ones_mask_num = torch.ones_like(x_num) if x_num is not None and x_num.shape[1] > 0 else torch.empty(batch_size, 0, device=self.device)
        ones_mask_cat = torch.ones_like(x_cat, dtype=torch.float) if x_cat is not None and x_cat.shape[1] > 0 else torch.empty(batch_size, 0, device=self.device)
        
        # 获取已知部分的干净嵌入 e_known
        e_known, _ = self.feature_embedder(x_num, x_cat, num_mask, cat_mask)
        
        # 将原始特征掩码扩展到嵌入维度
        emb_dim = e_known.shape[1]
        feature_emb_dim = emb_dim // feature_mask_original.shape[1] if feature_mask_original.shape[1] > 0 else emb_dim
        e_mask = feature_mask_original.unsqueeze(2).expand(-1, -1, feature_emb_dim).reshape(batch_size, -1).float()
        
        # 初始化：从完全随机的噪声嵌入 e_T 开始
        # e_t = torch.randn(batch_size, emb_dim, device=self.device)
        if self.noise_t_batch_level:
            t_max = torch.full((batch_size,), self.diffusion_utils.num_timesteps - 1, device=self.device, dtype=torch.long)
        else:
            # cell-level timesteps
            t_max = torch.full((batch_size, feature_mask_original.shape[1]), self.diffusion_utils.num_timesteps - 1, device=self.device, dtype=torch.long)
        e_t, _ = self.diffusion_utils.q_sample_embedding(e_known, t_max)
        
        # 设置时间步
        timesteps = torch.linspace(
            0, (self.diffusion_utils.num_timesteps - 1)//10, 
            num_steps, dtype=torch.int64, device=self.device
        )
        
        # 逆向扩散循环
        for i in range(len(timesteps) - 1, -1, -1):
            if self.noise_t_batch_level:
                t = torch.full((batch_size,), timesteps[i], device=self.device, dtype=torch.long)
            else:
                # cell-level timesteps
                t = torch.full((batch_size, feature_mask_original.shape[1]), timesteps[i], device=self.device, dtype=torch.long)
            
            # 构建两通道输入：condition + noise
            # 通道一：条件数据 - 只包含已知部分的干净数据
            e_condition = e_known * e_mask
            
            # 通道二：噪声数据 - 只包含需要预测位置的噪声
            e_noise = e_t * (1 - e_mask)
            
            # 构建模型输入：condition + noise
            model_input = torch.cat([e_condition, e_noise], dim=1)
            
            # 模型预测噪声
            epsilon_pred = self.repair_model(model_input, t)
            
            # 确保epsilon_pred是单个张量
            if isinstance(epsilon_pred, tuple):
                epsilon_pred = epsilon_pred[0]
            
            # 使用预测的噪声和当前的e_t来估算e_0
            # 公式：e_0_pred = (e_t - sqrt(1 - alpha_cumprod_t) * epsilon) / sqrt(alpha_cumprod_t)
            # 处理cell级别的时间步
            if t.dim() > 1:
                # Cell-level timesteps: apply per-feature coefficients
                batch_size_loop, num_features = t.shape
                total_embedding_dim = e_t.shape[1]
                feature_emb_dim = total_embedding_dim // num_features
                
                e_0_pred_parts = []
                for j in range(num_features):
                    # Get the slice for the current feature
                    start_dim = j * feature_emb_dim
                    end_dim = start_dim + feature_emb_dim
                    e_t_slice = e_t[:, start_dim:end_dim]
                    epsilon_pred_slice = epsilon_pred[:, start_dim:end_dim]
                    
                    # Get the timestep for the current feature
                    t_feature = t[:, j]  # Shape: [batch_size]
                    
                    # Get noise schedule parameters for the feature's timestep
                    alpha_cumprod_t = self.diffusion_utils.alphas_cumprod[t_feature].view(-1, 1)
                    sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod_t)
                    sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - alpha_cumprod_t)
                    
                    # Apply DDPM reverse formula to the slice
                    e_0_pred_slice = (e_t_slice - sqrt_one_minus_alpha_cumprod_t * epsilon_pred_slice) / sqrt_alpha_cumprod_t
                    e_0_pred_parts.append(e_0_pred_slice)
                
                # Concatenate the predicted parts
                e_0_pred = torch.cat(e_0_pred_parts, dim=1)
            else:
                # Batch-level timesteps (original behavior)
                alpha_cumprod_t = self.diffusion_utils.alphas_cumprod[t].view(-1, 1)
                sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod_t)
                sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - alpha_cumprod_t)
                e_0_pred = (e_t - sqrt_one_minus_alpha_cumprod_t * epsilon_pred) / sqrt_alpha_cumprod_t
            
            # 如果不是最后一步，计算e_{t-1}
            if i > 0:
                # 获取后验分布参数 - 需要根据时间步格式处理
                if self.noise_t_batch_level:
                    t_prev = torch.full((batch_size,), timesteps[i-1], device=self.device, dtype=torch.long)
                else:
                    t_prev = torch.full((batch_size, feature_mask_original.shape[1]), timesteps[i-1], device=self.device, dtype=torch.long)
                
                # 后验均值系数 - 根据时间步维度处理
                if t.dim() > 1:
                    # Cell-level timesteps: handle per-feature coefficients
                    batch_size_loop, num_features = t.shape
                    total_embedding_dim = e_t.shape[1]
                    feature_emb_dim = total_embedding_dim // num_features
                    
                    posterior_mean_parts = []
                    for j in range(num_features):
                        # Get coefficients for current feature
                        t_feature = t[:, j]
                        posterior_mean_coef1_feature = self.diffusion_utils.posterior_mean_coef1[t_feature].view(-1, 1)
                        posterior_mean_coef2_feature = self.diffusion_utils.posterior_mean_coef2[t_feature].view(-1, 1)
                        posterior_log_variance_feature = self.diffusion_utils.posterior_log_variance_clipped[t_feature].view(-1, 1)
                        
                        # Get slices for current feature
                        start_dim = j * feature_emb_dim
                        end_dim = start_dim + feature_emb_dim
                        e_0_pred_slice = e_0_pred[:, start_dim:end_dim]
                        e_t_slice = e_t[:, start_dim:end_dim]
                        
                        # Calculate posterior mean for this feature
                        posterior_mean_slice = posterior_mean_coef1_feature * e_0_pred_slice + posterior_mean_coef2_feature * e_t_slice
                        
                        # Sample noise for this slice
                        noise_slice = torch.randn_like(e_t_slice)
                        
                        # Sample e_{t-1} for this slice
                        e_t_prev_slice = posterior_mean_slice + torch.exp(0.5 * posterior_log_variance_feature) * noise_slice
                        posterior_mean_parts.append(e_t_prev_slice)
                    
                    # Concatenate all feature slices
                    e_t = torch.cat(posterior_mean_parts, dim=1)
                else:
                    # Batch-level timesteps (original behavior)
                    posterior_mean_coef1 = self.diffusion_utils.posterior_mean_coef1[t].view(-1, 1)
                    posterior_mean_coef2 = self.diffusion_utils.posterior_mean_coef2[t].view(-1, 1)
                    posterior_log_variance = self.diffusion_utils.posterior_log_variance_clipped[t].view(-1, 1)
                    
                    # 计算后验均值
                    posterior_mean = posterior_mean_coef1 * e_0_pred + posterior_mean_coef2 * e_t
                    
                    # 采样噪声
                    noise = torch.randn_like(e_t)
                    
                    # 采样e_{t-1}
                    e_t = posterior_mean + torch.exp(0.5 * posterior_log_variance) * noise
                
                # 只更新未知部分
                e_t = e_t * (1 - e_mask) + e_known * e_mask
            else:
                # 最后一步，直接使用预测的e_0
                e_t = e_0_pred * (1 - e_mask) + e_known * e_mask
        
        # 最终嵌入
        final_embeddings = e_t
        
        # 将嵌入解码为特征并返回合并的张量
        return self._decode_embeddings_to_combined_features(final_embeddings, batch_size)
    
    def _perform_original_space_diffusion(self, x_num, x_cat, num_mask, cat_mask, num_steps=100):
        """
        在原始数据空间中执行扩散过程填补（旧版本）
        """
        batch_size = x_num.shape[0] if x_num is not None else x_cat.shape[0]
        
        # 准备数据：将数值特征和分类特征的one-hot编码合并
        if x_num is not None and x_num.shape[1] > 0:
            x_num_part = x_num
        else:
            x_num_part = torch.empty(batch_size, 0, device=self.device)
            
        if x_cat is not None and x_cat.shape[1] > 0:
            # 将分类索引转换为one-hot编码
            x_cat_onehot = _create_onehot_from_indices(x_cat, self.actual_cat_sizes)
        else:
            x_cat_onehot = torch.empty(batch_size, 0, device=self.device)
        
        # 合并特征
        x_combined = combine_features(x_num_part, x_cat_onehot)
        
        # 准备掩码
        if num_mask is not None and cat_mask is not None:
            # 将分类掩码扩展到one-hot维度
            cat_mask_expanded = []
            for i, cat_size in enumerate(self.actual_cat_sizes):
                if i < cat_mask.shape[1]:
                    feature_mask = cat_mask[:, i:i+1]
                    expanded_feature_mask = feature_mask.repeat(1, cat_size)
                    cat_mask_expanded.append(expanded_feature_mask)
            
            if cat_mask_expanded:
                cat_mask_onehot = torch.cat(cat_mask_expanded, dim=1)
            else:
                cat_mask_onehot = torch.empty(batch_size, 0, device=self.device)
                
            combined_mask = torch.cat([num_mask, cat_mask_onehot], dim=1)
        elif num_mask is not None:
            combined_mask = num_mask
        else:
            combined_mask = cat_mask_onehot
        
        # 反转掩码：1表示缺失，0表示观测
        missing_mask = (1 - combined_mask).bool()
        
        # 初始化：从纯噪声开始
        x_t = torch.randn_like(x_combined)
        
        # 设置已知位置的值
        x_t[~missing_mask] = x_combined[~missing_mask]
        
        # 逆向扩散
        for i in range(num_steps-1, -1, -1):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            
            # 构建三通道输入
            # 通道1: noisy_target (缺失位置有噪声，已知位置为0)
            noisy_target = x_t * missing_mask.float()
            
            # 通道2: condition_data (已知位置有数据，缺失位置为0)
            condition_data = x_combined * (~missing_mask).float()
            
            # 通道3: condition_mask (已知位置为1，缺失位置为0)
            condition_mask = (~missing_mask).float()
            
            # 通过feature_embedder编码三个通道
            # 需要将合并的特征分离回数值和分类
            if self.d_numerical > 0:
                noisy_num = noisy_target[:, :self.d_numerical]
                cond_num = condition_data[:, :self.d_numerical]
            else:
                noisy_num = torch.empty(batch_size, 0, device=self.device)
                cond_num = torch.empty(batch_size, 0, device=self.device)
                
            if self.d_categorical > 0:
                cat_onehot_dim = sum(self.actual_cat_sizes)
                noisy_cat_onehot = noisy_target[:, self.d_numerical:self.d_numerical + cat_onehot_dim]
                cond_cat_onehot = condition_data[:, self.d_numerical:self.d_numerical + cat_onehot_dim]
                
                # 转换为索引
                noisy_cat = _convert_onehot_to_indices(noisy_cat_onehot, self.actual_cat_sizes, self.device)
                cond_cat = _convert_onehot_to_indices(cond_cat_onehot, self.actual_cat_sizes, self.device)
            else:
                noisy_cat = torch.empty(batch_size, 0, dtype=torch.long, device=self.device)
                cond_cat = torch.empty(batch_size, 0, dtype=torch.long, device=self.device)
            
            # 创建全1掩码用于编码
            ones_mask_num = torch.ones_like(noisy_num) if noisy_num.shape[1] > 0 else torch.empty(batch_size, 0, device=self.device)
            ones_mask_cat = torch.ones_like(noisy_cat, dtype=torch.float) if noisy_cat.shape[1] > 0 else torch.empty(batch_size, 0, device=self.device)
            
            # 编码三个通道
            e_noisy, _ = self.feature_embedder(noisy_num, noisy_cat, ones_mask_num, ones_mask_cat)
            e_cond, _ = self.feature_embedder(cond_num, cond_cat, ones_mask_num, ones_mask_cat)
            
            # 编码掩码
            emb_dim = e_noisy.shape[1]
            if num_mask is not None and cat_mask is not None:
                feature_mask_original = torch.cat([num_mask, cat_mask], dim=1)
                feature_emb_dim = emb_dim // feature_mask_original.shape[1]
                e_mask = (~missing_mask[:, :feature_mask_original.shape[1]]).float().unsqueeze(2).expand(-1, -1, feature_emb_dim).reshape(batch_size, -1)
            else:
                feature_emb_dim = emb_dim // condition_mask.shape[1]
                e_mask = condition_mask.unsqueeze(2).expand(-1, -1, feature_emb_dim).reshape(batch_size, -1)
            
            # 构建两通道输入：condition + noise
            # 通道一：条件数据 - 只包含已知部分的干净数据
            e_condition = e_cond
            
            # 通道二：噪声数据 - 只包含需要预测位置的噪声
            e_noise = e_noisy * (1 - e_mask)
            
            model_input = torch.cat([e_condition, e_noise], dim=1)
            
            # 模型预测
            with torch.no_grad():
                pred_x0_num, pred_x0_cat_logits = self.repair_model(model_input, t)
            
            # 将预测转换回合并特征格式
            pred_x0 = self._decode_model_output_to_combined_features(pred_x0_num, pred_x0_cat_logits, batch_size)
            
            # 计算x_{t-1}
            if i > 0:
                # 使用DDPM的后验采样
                alpha_t = self.diffusion_utils.alphas[t].view(-1, 1)
                alpha_cumprod_t = self.diffusion_utils.alphas_cumprod[t].view(-1, 1)
                alpha_cumprod_t_prev = self.diffusion_utils.alphas_cumprod[t-1].view(-1, 1)
                
                # 后验均值
                posterior_mean = (
                    torch.sqrt(alpha_cumprod_t_prev) * (1 - alpha_t) / (1 - alpha_cumprod_t) * pred_x0 +
                    torch.sqrt(alpha_t) * (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * x_t
                )
                
                # 后验方差
                posterior_variance = (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_t)
                
                # 采样
                noise = torch.randn_like(x_t)
                x_t = posterior_mean + torch.sqrt(posterior_variance) * noise
                
                # 保持已知位置不变
                x_t[~missing_mask] = x_combined[~missing_mask]
            else:
                # 最后一步
                x_t = pred_x0
                x_t[~missing_mask] = x_combined[~missing_mask]
        
        return x_t

    def _perform_direct_imputation(self, x_num, x_cat, num_mask, cat_mask):
        """
        执行直接预测填补，参考repair_embed.py中的_direct_prediction实现
        
        Args:
            x_num: 带有临时填充值的数值特征张量
            x_cat: 带有MASK标记的分类特征张量
            num_mask: 数值特征掩码（1=观测值，0=缺失值）
            cat_mask: 分类特征掩码（1=观测值，0=缺失值）
            
        Returns:
            填补后的合并特征张量
        """
        batch_size = x_num.shape[0] if x_num is not None else x_cat.shape[0]
        
        # 获取特征级掩码
        if num_mask is not None and cat_mask is not None:
            feature_mask_original = torch.cat([num_mask, cat_mask], dim=1)
        elif num_mask is None and cat_mask is not None:
            feature_mask_original = cat_mask
        elif num_mask is not None and cat_mask is None:
            feature_mask_original = num_mask
        else:
            raise ValueError("既没有数值特征也没有分类特征掩码")
        
        # 判断是否使用新的噪声预测架构
        if hasattr(self.repair_model, 'predict_noise') and self.repair_model.predict_noise:
            # 新架构：模型预测噪声，使用2通道输入
            # 获取已知部分的初始嵌入
            e0_filled, _ = self.feature_embedder(x_num, x_cat, num_mask, cat_mask)
            # e_t = torch.randn(batch_size, e0_filled.shape[1], device=self.device)
            if self.noise_t_batch_level:
                t_max = torch.full((batch_size,), self.diffusion_utils.num_timesteps - 1, device=self.device, dtype=torch.long)
            else:
                # cell-level timesteps
                t_max = torch.full((batch_size, feature_mask_original.shape[1]), self.diffusion_utils.num_timesteps - 1, device=self.device, dtype=torch.long)
            e_t, _ = self.diffusion_utils.q_sample_embedding(e0_filled, t_max)
            
            # 将原始特征掩码扩展到嵌入维度
            emb_dim = e0_filled.shape[1]
            feature_emb_dim = emb_dim // feature_mask_original.shape[1] if feature_mask_original.shape[1] > 0 else emb_dim
            emb_mask = feature_mask_original.unsqueeze(2).expand(-1, -1, feature_emb_dim).reshape(batch_size, -1).float()
            
            # 使用t=0的时间步
            if self.repair_model.predict_noise and self.args.use_diffusion_loss:
                t = torch.full((batch_size,), self.diffusion_utils.num_timesteps - 1, device=self.device, dtype=torch.long)
            else:
                t = torch.zeros((batch_size,), device=self.device, dtype=torch.long)
            
            # 构建两通道输入：condition + noise
            # 通道一：条件数据 - 只包含已知位置的真实嵌入
            e_condition = e0_filled * emb_mask
            
            # 通道二：噪声数据 - 只包含缺失位置的噪声
            e_noise = e_t * (1 - emb_mask)
            
            # 构建模型输入：condition + noise
            model_input = torch.cat([e_condition, e_noise], dim=1)
            
            # 模型预测噪声（在t=0时应该接近0）
            epsilon_pred = self.repair_model(model_input, t)
            
            # 对于直接预测，预测的嵌入是条件嵌入加上缺失位置的预测
            # 如果模型预测的是噪声，我们需要从噪声中恢复原始嵌入 e_0
            # 公式：e_0_pred = (e_t - sqrt(1 - alpha_cumprod_t) * epsilon) / sqrt(alpha_cumprod_t)
            # 这里 t=0, e_t 是 e0_filled
            alpha_cumprod_t = self.diffusion_utils.alphas_cumprod[t].view(-1, 1)
            sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod_t)
            sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1. - alpha_cumprod_t)

            pred_embedding_denoised = (e0_filled - sqrt_one_minus_alpha_cumprod_t * epsilon_pred) / sqrt_alpha_cumprod_t

            # 只更新未知(缺失)部分
            pred_embedding = e0_filled * emb_mask + pred_embedding_denoised * (1 - emb_mask)
            
            # 将嵌入解码为特征
            return self._decode_embeddings_to_combined_features(pred_embedding, batch_size)
        else:
            # 原架构：使用3通道输入，模型直接预测特征
            # 使用t=0的时间步，让模型直接预测x0
            t = torch.zeros((batch_size,), device=self.device, dtype=torch.long)
            
            # 构建三通道输入
            # 构建 noisy_target: 对于缺失位置使用预填充值，已知位置为0
            if x_num is not None and x_num.shape[1] > 0:
                noisy_num_target = x_num * (1 - num_mask)
            else:
                noisy_num_target = torch.empty(batch_size, 0, device=self.device)
                
            if x_cat is not None and x_cat.shape[1] > 0:
                noisy_cat_target = x_cat.clone()
                noisy_cat_target = noisy_cat_target * (1 - cat_mask.long())
            else:
                noisy_cat_target = torch.empty(batch_size, 0, dtype=torch.long, device=self.device)
            
            # 构建 condition_data: 对于已知位置使用真实数据，缺失位置为0
            if x_num is not None and x_num.shape[1] > 0:
                cond_num_data = x_num * num_mask
            else:
                cond_num_data = torch.empty(batch_size, 0, device=self.device)
                
            if x_cat is not None and x_cat.shape[1] > 0:
                cond_cat_data = x_cat * cat_mask.long()
            else:
                cond_cat_data = torch.empty(batch_size, 0, dtype=torch.long, device=self.device)
            
            # 通过 feature_embedder 生成三个通道的嵌入
            ones_mask_num = torch.ones_like(noisy_num_target) if noisy_num_target.shape[1] > 0 else torch.empty(batch_size, 0, device=self.device)
            ones_mask_cat = torch.ones_like(noisy_cat_target, dtype=torch.float) if noisy_cat_target.shape[1] > 0 else torch.empty(batch_size, 0, device=self.device)
            
            e_noisy, _ = self.feature_embedder(noisy_num_target, noisy_cat_target, ones_mask_num, ones_mask_cat)
            e_cond, _ = self.feature_embedder(cond_num_data, cond_cat_data, ones_mask_num, ones_mask_cat)
            
            # 构建 e_mask: 将原始特征掩码扩展到embedding维度
            emb_dim = e_noisy.shape[1]
            feature_emb_dim = emb_dim // feature_mask_original.shape[1] if feature_mask_original.shape[1] > 0 else emb_dim
            e_mask = feature_mask_original.unsqueeze(2).expand(-1, -1, feature_emb_dim).reshape(batch_size, -1).float()
            
            # 构建两通道输入：condition + noise
            # 通道一：条件数据 - 只包含已知部分的干净数据
            e_condition = e_cond
            
            # 通道二：噪声数据 - 只包含需要预测位置的噪声
            e_noise = e_noisy * (1 - e_mask)
            
            e_input = torch.cat([e_condition, e_noise], dim=1)
            
            # 模型预测原始特征
            pred_x0_num_from_model, pred_x0_cat_output = self.repair_model(e_input, t)
            
            # 将嵌入解码为特征并返回合并的张量
            return self._decode_model_output_to_combined_features(pred_x0_num_from_model, pred_x0_cat_output, batch_size)

    def _decode_embeddings_to_combined_features(self, embeddings, batch_size):
        """
        将嵌入解码为合并的特征张量
        
        Args:
            embeddings: 最终预测的嵌入
            batch_size: 批次大小
            
        Returns:
            合并的特征张量
        """
        # 使用模型的重建输出功能
        pred_x0_num, pred_x0_cat_output = self.repair_model._reconstruct_output(embeddings)
        return self._decode_model_output_to_combined_features(pred_x0_num, pred_x0_cat_output, batch_size)

    def _decode_model_output_to_combined_features(self, pred_x0_num, pred_x0_cat_output, batch_size):
        """
        将模型输出解码为合并的特征张量
        
        Args:
            pred_x0_num: 预测的数值特征
            pred_x0_cat_output: 预测的分类特征logits
            batch_size: 批次大小
            
        Returns:
            合并的特征张量
        """
        # 1. 处理数值特征部分
        pred_x_num_part = pred_x0_num if pred_x0_num is not None else torch.empty((batch_size, 0), device=self.device)
        
        # 2. 处理分类特征部分（从logits转换为独热编码）
        if pred_x0_cat_output is not None and len(pred_x0_cat_output) > 0:
            pred_cat_onehot_parts = []
            
            if isinstance(pred_x0_cat_output, list):
                # 列表格式：直接处理每个元素
                for cat_logits in pred_x0_cat_output:
                    cat_probs = torch.nn.functional.gumbel_softmax(cat_logits, tau=1.0, hard=False, dim=1)
                    pred_cat_onehot_parts.append(cat_probs)
            else:
                # 张量格式：按cat_sizes切分
                start_idx = 0
                for cat_size in self.actual_cat_sizes:
                    end_idx = start_idx + cat_size
                    cat_logits = pred_x0_cat_output[:, start_idx:end_idx]
                    cat_probs = torch.nn.functional.gumbel_softmax(cat_logits, tau=1.0, hard=False, dim=1)
                    pred_cat_onehot_parts.append(cat_probs)
                    start_idx = end_idx
            
            pred_x_cat_part = torch.cat(pred_cat_onehot_parts, dim=1) if pred_cat_onehot_parts else torch.empty((batch_size, 0), device=self.device)
        else:
            pred_x_cat_part = torch.empty((batch_size, 0), device=self.device)
        
        # 3. 合并预测的特征
        return combine_features(pred_x_num_part, pred_x_cat_part)

    def _logits_to_cat_indices(self, pred_x0_cat_output, batch_size):
        """
        将模型输出的分类logits转换为整数索引，基于repair_embed.py中的实现
        
        Args:
            pred_x0_cat_output: 模型的分类输出，可以是logits列表或单个logits张量
            batch_size: 当前批次的大小
            
        Returns:
            形状为[batch_size, num_categorical_features]的整数索引张量
        """
        cat_indices_list = []
        if self.d_categorical > 0:
            if isinstance(pred_x0_cat_output, list):  # TabularDiffAE的cat_reconstructor
                for logits_i in pred_x0_cat_output:
                    if logits_i.nelement() > 0:
                         cat_indices_list.append(torch.argmax(logits_i, dim=1))
            elif pred_x0_cat_output is not None and pred_x0_cat_output.nelement() > 0:  # TabularUNet（平面logits）
                current_pos = 0
                for cat_size in self.actual_cat_sizes:
                    if cat_size > 0:
                        logits_i = pred_x0_cat_output[:, current_pos : current_pos + cat_size]
                        cat_indices_list.append(torch.argmax(logits_i, dim=1))
                        current_pos += cat_size
        
        return torch.stack(cat_indices_list, dim=1) if cat_indices_list else \
               torch.empty(batch_size, 0, dtype=torch.long, device=self.device)

    def _convert_onehot_mask_to_original(self, onehot_mask):
        """
        将 one-hot 维度的掩码转换为原始特征维度的掩码
        
        Args:
            onehot_mask: one-hot 维度的掩码 [batch_size, d_numerical + sum(cat_sizes)]
            
        Returns:
            original_mask: 原始特征维度的掩码 [batch_size, d_numerical + d_categorical]
        """
        batch_size = onehot_mask.shape[0]
        original_mask = torch.zeros(batch_size, self.d_numerical + self.d_categorical, 
                                  dtype=onehot_mask.dtype, device=onehot_mask.device)
        
        # 数值特征部分直接复制
        if self.d_numerical > 0:
            original_mask[:, :self.d_numerical] = onehot_mask[:, :self.d_numerical]
        
        # 分类特征部分需要从 one-hot 转换回原始维度
        if self.d_categorical > 0:
            start_idx = self.d_numerical
            for i, cat_size in enumerate(self.actual_cat_sizes):
                if i < self.d_categorical:
                    end_idx = start_idx + cat_size
                    # 对于分类特征，如果 one-hot 中的任何位置为True，则原始特征为True
                    cat_onehot_mask = onehot_mask[:, start_idx:end_idx]
                    original_mask[:, self.d_numerical + i] = cat_onehot_mask.any(dim=1)
                    start_idx = end_idx
        
        return original_mask

    def _onehot_to_indices(self, x_cat_onehot):
        """
        将独热编码转换回分类索引
        
        Args:
            x_cat_onehot: 独热编码的分类特征 [batch_size, sum(cat_sizes)]
            
        Returns:
            分类索引 [batch_size, d_categorical]
        """
        if x_cat_onehot is None or x_cat_onehot.shape[1] == 0:
            return None
        
        indices = []
        start_idx = 0
        
        for cat_size in self.actual_cat_sizes:
            end_idx = start_idx + cat_size
            cat_onehot = x_cat_onehot[:, start_idx:end_idx]
            cat_indices = torch.argmax(cat_onehot, dim=1)
            indices.append(cat_indices)
            start_idx = end_idx
        
        if indices:
            return torch.stack(indices, dim=1)
        else:
            return torch.empty((x_cat_onehot.shape[0], 0), dtype=torch.long, device=self.device)

    def create_observation_mask_for_reconstruction(self, df):
        """
        创建用于重建损失计算的观测掩码，基于原始DataFrame
        使用预计算掩码以提高性能
        
        Args:
            df: 原始数据框
            
        Returns:
            观测掩码 [batch_size, d_numerical + d_categorical]，True表示观测值（非缺失），False表示缺失值
        """
        # 尝试使用预计算的掩码
        if hasattr(df, 'index') and len(df) > 0:
            # 检查是否来自完整样本或缺失样本
            if 'train_complete' in self.precomputed_observation_masks:
                complete_mask = self.precomputed_observation_masks['train_complete']
                complete_indices = list(range(len(complete_mask)))
                
                # 检查df的索引是否在完整样本的范围内
                df_indices = list(df.index)
                if all(idx < len(complete_mask) for idx in df_indices):
                    # 从预计算掩码中提取对应的行
                    try:
                        batch_mask = complete_mask[df_indices]
                        return batch_mask
                    except:
                        pass
            
            if 'train_missing' in self.precomputed_observation_masks:
                missing_mask = self.precomputed_observation_masks['train_missing']
                
                # 尝试从缺失样本掩码中获取
                df_indices = list(df.index)
                if all(idx < len(missing_mask) for idx in df_indices):
                    try:
                        batch_mask = missing_mask[df_indices]
                        return batch_mask
                    except:
                        pass
        
        # 如果预计算掩码不可用，回退到原有方法
        return self.get_cached_error_mask(df, mask_type="observation")

    def create_missing_mask(self, df):
        """
        创建缺失值掩码，基于原始DataFrame
        使用预计算掩码以提高性能
        
        Args:
            df: 原始数据框
            
        Returns:
            缺失值掩码 [batch_size, feature_dim]，True表示缺失值
        """
        # 尝试使用预计算的掩码
        if hasattr(df, 'index') and len(df) > 0:
            # 检查是否来自完整样本或缺失样本
            if 'train_complete' in self.precomputed_error_masks:
                complete_mask = self.precomputed_error_masks['train_complete']
                
                # 检查df的索引是否在完整样本的范围内
                df_indices = list(df.index)
                if all(idx < len(complete_mask) for idx in df_indices):
                    try:
                        batch_mask = complete_mask[df_indices]
                        return batch_mask
                    except:
                        pass
            
            if 'train_missing' in self.precomputed_error_masks:
                missing_mask = self.precomputed_error_masks['train_missing']
                
                # 尝试从缺失样本掩码中获取
                df_indices = list(df.index)
                if all(idx < len(missing_mask) for idx in df_indices):
                    try:
                        batch_mask = missing_mask[df_indices]
                        return batch_mask
                    except:
                        pass
        
        # 如果预计算掩码不可用，回退到原有方法
        return self.get_cached_error_mask(df, mask_type="missing")
    
    def _prepare_dataset(self, df, dataset_name, batch_size, shuffle=False, 
                        error_mask_df=None, observation_mask_df=None):
        """
        准备单个数据集的通用方法
        
        Args:
            df: 数据框
            dataset_name: 数据集名称（用于transform_onehot）
            batch_size: 批次大小
            shuffle: 是否打乱数据
            error_mask_df: 预计算的错误掩码DataFrame
            observation_mask_df: 预计算的观测掩码DataFrame
            
        Returns:
            DataLoader 或 None（如果df为空）
        """
        if len(df) == 0:
            return None
            
        # 转换特征
        x_num, x_cat_onehot = self.processor.transform_onehot(df, dataset_name=dataset_name)
        x_combined = combine_features(x_num, x_cat_onehot)
        
        # 处理目标标签
        target_col = self.processor.target_feature[0]
        target_values = df[target_col].values
        if isinstance(target_values[0], (list, np.ndarray)):
            target_values = np.array([val[0] if isinstance(val, (list, np.ndarray)) else val for val in target_values])
        encoded_labels = np.array([self.processor.label_encoder.get(val, 0) for val in target_values])
        y = torch.LongTensor(encoded_labels)
        
        # 创建数据集
        if error_mask_df is not None:
            # 使用外部传入的错误掩码
            feature_cols = self.processor.num_features + self.processor.cat_features
            missing_masks = self._convert_pandas_mask_to_tensor(
                error_mask_df[feature_cols]
            )
        else:
            # 回退到原有方法
            feature_cols = self.processor.num_features + self.processor.cat_features
            missing_masks = self.create_missing_mask(df[feature_cols])
        
        if dataset_name.startswith("val"):  # 验证集需要观测掩码
            if observation_mask_df is not None:
                # 使用外部传入的观测掩码
                observed_masks = self._convert_pandas_mask_to_tensor(
                    observation_mask_df[feature_cols]
                )
            else:
                # 回退到原有方法
                observed_masks = self.create_observation_mask_for_reconstruction(df[feature_cols])
            dataset = torch.utils.data.TensorDataset(x_combined, y, missing_masks, observed_masks)
        else:
            dataset = torch.utils.data.TensorDataset(x_combined, y, missing_masks)
            
        # 创建数据加载器
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def _convert_pandas_mask_to_tensor(self, pandas_mask):
        """
        将pandas掩码转换为tensor格式（从MaskCacheManager移过来的辅助函数）
        
        Args:
            pandas_mask: pandas DataFrame掩码
            mask_type: 掩码类型（"error"或"observation"）
            
        Returns:
            转换后的tensor掩码
        """
        # 错误掩码：True表示缺失/错误，需要扩展分类特征到独热编码维度
        mask_parts = []
        
        # 处理数值特征
        if self.processor.num_features:
            num_mask = pandas_mask[self.processor.num_features].values
            num_mask_tensor = torch.tensor(num_mask, dtype=torch.bool, device=self.device)
            mask_parts.append(num_mask_tensor)
        
        # 处理分类特征（扩展为独热编码形式）
        if self.processor.cat_features and self.actual_cat_sizes:
            cat_mask = pandas_mask[self.processor.cat_features].values
            cat_mask_tensor = torch.tensor(cat_mask, dtype=torch.bool, device=self.device)
            
            expanded_cat_masks = []
            for i, cat_size in enumerate(self.actual_cat_sizes):
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
            return torch.empty((pandas_mask.shape[0], 0), dtype=torch.bool, device=self.device)
        
    def compute_gradient_alignment_loss(self, classifier, x_imp_batch, y_imp_batch, x_complete_batch, y_complete_batch):
        """
        计算改进的梯度对齐损失
        
        使用公式：L_align = 1 − (g_m·g_c)/(‖g_m‖‖g_c‖) + μ‖g_m‖
        其中 μ > 0 防止填补样本梯度塌缩到零
        
        Args:
            classifier: 使用updated_w的分类器
            x_imp_batch: 由impute_with_model_differentiable生成的填补样本（需保持计算图）
            y_imp_batch: 填补样本的标签
            x_complete_batch: 完整样本
            y_complete_batch: 完整样本的标签
            
        Returns:
            alignment_loss: 改进的梯度对齐损失
            cosine_sim: 余弦相似度值（用于监控）
            grad_norm: 填补样本梯度的范数（用于监控）
        """
        # 计算完整样本的梯度
        grad_complete = self._get_penultimate_output(classifier, x_complete_batch, y_complete_batch, mode='gradient')
        grad_complete_mean = grad_complete.mean(dim=0).detach()  
        
        # 计算填补样本的梯度
        grad_imputed = self._get_penultimate_output(classifier, x_imp_batch, y_imp_batch, mode='gradient')
        grad_imputed_mean = grad_imputed.mean(dim=0)  
        
        # 计算梯度模长
        grad_complete_norm = torch.norm(grad_complete_mean, p=2)
        grad_imputed_norm = torch.norm(grad_imputed_mean, p=2)
        
        # 计算余弦相似度 (normalized dot product)
        cosine_sim = torch.dot(grad_imputed_mean, grad_complete_mean) / (grad_imputed_norm * grad_complete_norm + 1e-8)
        
        # 获取μ参数
        mu_align = getattr(self.args, 'mu_align', 0.1)
        
        # 计算改进的对齐损失：1 − (g_m·g_c)/(‖g_m‖‖g_c‖) + μ‖g_m‖
        alignment_loss = 1.0 - cosine_sim + mu_align * grad_imputed_norm
        
        # 计算梯度范数用于监控
        grad_norm = grad_imputed_norm.item()
        
        return alignment_loss, cosine_sim.item(), grad_norm

    def compute_explicit_bilevel_gradients(self, classifier, x_imp_batch, y_imp_batch):
        """
        使用共轭梯度法计算显式双层超梯度 H_inv * v.
        返回计算出的填补模型梯度.
        """
        # 0. 获取填补模型参数
        imputation_params = list(self.repair_model.parameters())
        if self.feature_embedder is not None:
            imputation_params.extend(list(self.feature_embedder.parameters()))

        # 1. 计算 v = dL_outer / dw
        v = self.compute_outer_loss_wrt_classifier(
            classifier, x_imp_batch, y_imp_batch
        )
        
        # 2. 使用共轭梯度法计算 z = H_w^{-1} * v
        # H_w 是内层损失的Hessian矩阵，在填补数据上计算
        z = self.conjugate_gradient_solver(classifier, x_imp_batch, y_imp_batch, v)

        # 3. 计算间接梯度: -d/d_theta( (dL_inner/dw)^T * z )
        # 3a. 计算 dL_inner / dw
        classifier.zero_grad()
        inner_loss = F.cross_entropy(classifier(x_imp_batch), y_imp_batch)
        grad_inner_wrt_w = torch.autograd.grad(inner_loss, classifier.parameters(), create_graph=True)
        grad_inner_wrt_w_flat = torch.cat([g.flatten() for g in grad_inner_wrt_w])

        # 3b. 计算点积
        dot_product = torch.dot(grad_inner_wrt_w_flat, z.detach())
        
        # 3c. 计算对theta的梯度 - 保留计算图以便后续计算
        indirect_grads = torch.autograd.grad(dot_product, imputation_params, allow_unused=True, retain_graph=True)
        # 根据公式添加负号
        indirect_grads = [-g if g is not None else torch.zeros_like(p) 
                          for g, p in zip(indirect_grads, imputation_params)]
        
        total_grads = [g_ind for g_ind in zip(indirect_grads)]
        
        # 用于监控的指标
        metrics_log = {
            'explicit_bilevel_grad_norm': torch.norm(torch.cat([g.flatten() for g in indirect_grads if g is not None])).item(),
            'total_grad_norm': torch.norm(torch.cat([g.flatten() for g in total_grads if g is not None])).item(),
            'loss_type': 'explicit_bilevel'
        }

        return total_grads, metrics_log

    def conjugate_gradient_solver(self, classifier, x_batch, y_batch, b, n_steps=10, residual_tol=1e-10):
        """
        使用共轭梯度法求解线性方程组 Ax = b，其中 A 是 Hessian 矩阵 (隐式表示)
        
        Args:
            classifier: 分类器模型
            x_batch, y_batch: 用于计算Hessian的数据
            b: 目标向量 (即 dL_out/dw)
            n_steps: 最大迭代次数
            residual_tol: 残差容忍度
            
        Returns:
            x: 方程组的解
        """
        x = [torch.zeros_like(p, requires_grad=False) for p in b]
        r = [p.clone().detach() for p in b]
        p = [r_i.clone().detach() for r_i in r]
        
        rs_old = torch.sum(torch.stack([torch.sum(r_i * r_i) for r_i in r]))
        
        if rs_old.sqrt() < residual_tol:
            return x
            
        for i in range(n_steps):
            # 计算 Ap
            Ap = self.compute_hessian_vector_product(classifier, x_batch, y_batch, p)
            
            
            # 计算分母，增加数值稳定性
            pAp = torch.sum(torch.stack([torch.sum(p_i * Ap_i) for p_i, Ap_i in zip(p, Ap)]))
            denominator = torch.max(pAp.abs(), torch.tensor(1e-12, device=pAp.device))
            
            # 计算 alpha，防止除零
            alpha = rs_old / (denominator + 1e-8)
            
            # 检查 alpha 是否合理
            if torch.isnan(alpha) or torch.isinf(alpha) or alpha.abs() > 1e6:
                print(f"警告: alpha={alpha} 异常，提前退出共轭梯度")
                break
            
            # 更新 x 和 r
            x = [x_i + alpha * p_i for x_i, p_i in zip(x, p)]
            r = [r_i - alpha * Ap_i for r_i, Ap_i in zip(r, Ap)]
            
            rs_new = torch.sum(torch.stack([torch.sum(r_i * r_i) for r_i in r]))
            
            if rs_new.sqrt() < residual_tol:
                break
                
            # 更新 p
            p = [r_i + (rs_new / rs_old) * p_i for r_i, p_i in zip(r, p)]
            rs_old = rs_new
            
        return x
    
    def _perform_inner_step(self, clean_batch_x, clean_batch_y, error_batch_x, error_batch_y, error_batch_mask, classifier, optimizer):
        """
        执行单次内层优化步骤（更新分类器）
        
        Args:
            clean_batch_x: 完整样本特征（已合并的特征），可以为None
            clean_batch_y: 完整样本标签，可以为None
            error_batch_x: 缺失样本特征（已合并的特征），可以为None
            error_batch_y: 缺失样本标签，可以为None
            error_batch_mask: 缺失样本的掩码，可以为None
            classifier: 分类器模型
            optimizer: 分类器优化器
        """
        # 检查是否有可用的数据
        if error_batch_x is None and clean_batch_x is None:
            raise ValueError("没有可用的训练数据")
        
        # 将数据移到设备上（只处理非None的数据）
        if clean_batch_x is not None:
            clean_batch_x = clean_batch_x.to(self.device)
            clean_batch_y = clean_batch_y.to(self.device)
        if error_batch_x is not None:
            error_batch_x = error_batch_x.to(self.device)
            error_batch_y = error_batch_y.to(self.device)
            error_batch_mask = error_batch_mask.to(self.device)
        
        # 准备用于训练的数据
        if error_batch_x is not None:
            # 使用当前修复模型对缺失数据进行填补（非微分版本）
            with torch.no_grad():
                # 从合并特征中分离数值特征和分类特征
                # 注意：error_batch_x是经过combine_features处理的，包含数值特征和独热编码的分类特征
                if self.d_numerical > 0 and self.d_categorical > 0:
                    # 计算独热编码后的分类特征维度
                    cat_onehot_dim = sum(self.actual_cat_sizes)
                    error_x_num = error_batch_x[:, :self.d_numerical]
                    error_x_cat_onehot = error_batch_x[:, self.d_numerical:self.d_numerical + cat_onehot_dim]
                elif self.d_numerical > 0:
                    error_x_num = error_batch_x
                    error_x_cat_onehot = None
                else:
                    error_x_num = None
                    error_x_cat_onehot = error_batch_x
                
                error_batch_repaired = self.impute_with_model_differentiable(
                    error_x_num, error_x_cat_onehot, error_batch_mask, with_grad=False, method='diffusion' if self.args.use_diffusion else 'direct'
                )
        
        if clean_batch_x is not None and error_batch_x is not None:
            combined_x = torch.cat([clean_batch_x, error_batch_repaired], dim=0)
            combined_y = torch.cat([clean_batch_y, error_batch_y], dim=0)
        else:
            combined_x = error_batch_repaired
            combined_y = error_batch_y
        
        optimizer.zero_grad()
        outputs = classifier(combined_x)
        loss = nn.CrossEntropyLoss()(outputs, combined_y)
        
        loss.backward()
        optimizer.step()
        
        return loss.item()

    def move_data_to_device(self, data_list: list, device: str):
        """
        将数据列表中的所有张量移动到指定设备
        """
        return (data.to(device) for data in data_list)

    def _perform_outer_step(self, val_error_x, val_error_y, val_error_mask, val_observed_mask,
                            val_clean_x, val_clean_y, classifier, optimizer, class_centroids):
        """
        执行单次外层优化步骤（更新修复模型）- 集成扩散和双层优化
        
        新版本在完整数据和缺失数据上都计算扩散损失，实现统一的扩散-双层优化框架。
        
        Args:
            val_error_x: 验证集缺失样本特征（已合并的特征）
            val_error_y: 验证集缺失样本标签
            val_error_mask: 验证集缺失样本的缺失掩码
            val_observed_mask: 验证集缺失样本的观测掩码
            val_clean_x: 验证集完整样本特征（可能为None）（已合并的特征）
            val_clean_y: 验证集完整样本标签（可能为None）
            classifier: 分类器模型
            optimizer: 外层优化器（修复模型的优化器）
            class_centroids: 类别质心（可能为None）
            
        Returns:
            metrics_log: 包含各损失和监控指标的字典
        """
        val_error_x, val_error_y, val_error_mask, val_observed_mask = self.move_data_to_device([val_error_x, val_error_y, val_error_mask, val_observed_mask], self.device)
        
        if val_clean_x is not None:
            val_clean_x, val_clean_y = self.move_data_to_device([val_clean_x, val_clean_y], self.device)
        
        optimizer.zero_grad()
        # ===========================================
        # 步骤1: 创建混合批次（完整样本 + 缺失样本）
        # ===========================================
        
        # 合并完整样本和缺失样本成统一批次
        if val_clean_x is not None and len(val_clean_x) > 0:
            # 创建完整样本的全零掩码（没有缺失值）
            complete_batch_size = val_clean_x.shape[0]
            feature_dim = val_error_mask.shape[1]
            complete_missing_mask = torch.zeros(complete_batch_size, feature_dim, dtype=val_error_mask.dtype, device=self.device)
            
            # 合并数据
            mixed_x = torch.cat([val_clean_x, val_error_x], dim=0)
            mixed_missing_mask = torch.cat([complete_missing_mask, val_error_mask], dim=0)
        else:
            # 只有缺失样本
            mixed_x = val_error_x
            mixed_missing_mask = val_error_mask
        
        mixed_batch_size = mixed_x.shape[0]
        
        # ===========================================
        # 步骤2: 数据准备与embedding
        # ===========================================
        
        # 从合并特征中分离数值特征和分类特征
        if self.d_numerical > 0 and self.d_categorical > 0:
            cat_onehot_dim = sum(self.actual_cat_sizes)
            mixed_x_num = mixed_x[:, :self.d_numerical]
            mixed_x_cat_onehot = mixed_x[:, self.d_numerical:self.d_numerical + cat_onehot_dim]
            mixed_x_cat = self._onehot_to_indices(mixed_x_cat_onehot)
        elif self.d_numerical > 0:
            mixed_x_num = mixed_x
            mixed_x_cat_onehot = None
            mixed_x_cat = torch.empty(mixed_x.shape[0], 0, dtype=torch.long, device=self.device)
        else:
            mixed_x_num = None
            mixed_x_cat_onehot = mixed_x
            mixed_x_cat = self._onehot_to_indices(mixed_x_cat_onehot)
        
        # 获取原始特征维度的错误掩码
        original_error_mask = self._convert_onehot_mask_to_original(mixed_missing_mask)
        
        # 获取干净的初始嵌入 e_0
        num_mask_ones = torch.ones_like(mixed_x_num) if mixed_x_num is not None else torch.empty(mixed_batch_size, 0, device=self.device)
        cat_mask_ones = torch.ones_like(mixed_x_cat, dtype=torch.float) if mixed_x_cat is not None else torch.empty(mixed_batch_size, 0, device=self.device)
        e_0_mixed, _ = self.feature_embedder(mixed_x_num, mixed_x_cat, num_mask_ones, cat_mask_ones)
        
        # ===========================================
        # 步骤3: 前向扩散过程
        # ===========================================
        
        # 采样时间步 t
        if self.noise_t_batch_level:
            t = torch.randint(0, self.diffusion_utils.num_timesteps, (mixed_batch_size,), device=self.device).long()
        else:
            # 采样时间步 t - 按cell级别采样，每个特征都有独立的时间步
            num_features = self.d_numerical + self.d_categorical
            t = torch.randint(0, self.diffusion_utils.num_timesteps, (mixed_batch_size, num_features), device=self.device).long()
        
        # 在嵌入空间中加噪
        e_t, epsilon = self.diffusion_utils.q_sample_embedding(e_0_mixed, t)
        
        # ===========================================
        # 步骤4: 生成条件化掩码（支持完整样本和缺失样本）
        # ===========================================
        
        # 为混合批次生成条件掩码
        # 对于完整样本：生成随机掩码用于条件化训练
        # 对于缺失样本：使用真实的错误掩码
        cond_mask_orig_dim = create_cond_mask(original_error_mask, self.device)
        
        # 将掩码扩展到嵌入维度
        feature_mask_original = cond_mask_orig_dim  # [mixed_batch_size, d_numerical + d_categorical]
        emb_dim = e_0_mixed.shape[1]
        if feature_mask_original.shape[1] > 0:
            feature_emb_dim = emb_dim // feature_mask_original.shape[1]
            e_mask = feature_mask_original.unsqueeze(2).expand(-1, -1, feature_emb_dim).reshape(mixed_batch_size, -1).float()
        else:
            raise ValueError("feature_mask_original.shape[1] <= 0")
        
        # 原始错误掩码也需要扩展到嵌入维度
        ori_err_mask_dim = original_error_mask.unsqueeze(2).expand(-1, -1, feature_emb_dim).reshape(mixed_batch_size, -1).float() if feature_mask_original.shape[1] > 0 else torch.zeros(mixed_batch_size, emb_dim, device=self.device)
        
        # ===========================================
        # 步骤5: 模型预测与去噪
        # ===========================================
        
        # 构建两通道输入：condition + noise
        # 通道一：条件数据 - 只包含已知部分的干净数据
        e_condition = e_0_mixed * e_mask
        
        # 通道二：噪声数据 - 只包含需要预测位置的噪声
        e_noise = e_t * (1 - e_mask)
        
        # 构建模型输入：condition + noise
        model_input = torch.cat([e_condition, e_noise], dim=1)
        # 模型预测噪声
        epsilon_pred = self.repair_model(model_input, t)
        # 确保epsilon_pred是单个张量
        if isinstance(epsilon_pred, tuple):
            epsilon_pred = epsilon_pred[0]
        
        # 使用DDPM逆向公式计算预测的e_0
        # 处理cell级别的时间步
        if t.dim() > 1:
            # Cell-level timesteps: apply per-feature coefficients
            batch_size, num_features = t.shape
            total_embedding_dim = e_t.shape[1]
            feature_emb_dim = total_embedding_dim // num_features
            
            e_0_pred_parts = []
            for i in range(num_features):
                # Get the slice for the current feature
                start_dim = i * feature_emb_dim
                end_dim = start_dim + feature_emb_dim
                e_t_slice = e_t[:, start_dim:end_dim]
                epsilon_pred_slice = epsilon_pred[:, start_dim:end_dim]
                
                # Get the timestep for the current feature
                t_feature = t[:, i]  # Shape: [batch_size]
                
                # Get noise schedule parameters for the feature's timestep
                alpha_cumprod_t = self.diffusion_utils.alphas_cumprod[t_feature].view(-1, 1)
                sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod_t)
                sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - alpha_cumprod_t)
                
                # Apply DDPM reverse formula to the slice
                e_0_pred_slice = (e_t_slice - sqrt_one_minus_alpha_cumprod_t * epsilon_pred_slice) / sqrt_alpha_cumprod_t
                e_0_pred_parts.append(e_0_pred_slice)
            
            # Concatenate the predicted parts
            e_0_pred = torch.cat(e_0_pred_parts, dim=1)
        else:
            # Batch-level timesteps (original behavior)
            alpha_cumprod_t = self.diffusion_utils.alphas_cumprod[t].view(-1, 1)
            sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod_t)
            sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - alpha_cumprod_t)
            e_0_pred = (e_t - sqrt_one_minus_alpha_cumprod_t * epsilon_pred) / sqrt_alpha_cumprod_t
        
        # ===========================================
        # 步骤6: 构建最终填补样本
        # ===========================================
        
        # 组合最终嵌入：已知部分用真实e_0，未知部分用预测e_0
        # 这里的(1 - e_mask)区域包含了所有需要模型填补的部分
        e_imputed = e_0_mixed * e_mask + e_0_pred * (1 - e_mask)
        # 解码为特征空间
        x_mixed_imp_batch = self._decode_embeddings_to_combined_features(e_imputed, mixed_batch_size)
        
        # ===========================================
        # 步骤7: 计算统一的损失函数
        # ===========================================
        
        # 7.1 计算Diffusion损失 (噪声预测MSE损失) - 现在在混合批次上计算
        loss_mask_emb = (1 - e_mask - ori_err_mask_dim)
        
        # 计算MSE损失，只在需要预测的区域计算
        diffusion_loss = F.mse_loss(epsilon * loss_mask_emb, epsilon_pred * loss_mask_emb)
        
        # 7.2 计算Bi-Level损失 - 分离回完整样本和缺失样本
        if val_clean_x is not None and len(val_clean_x) > 0:
            # 分离混合批次中的完整样本和缺失样本
            complete_batch_size = val_clean_x.shape[0]
            x_complete_imp_batch = x_mixed_imp_batch[:complete_batch_size]
            x_missing_imp_batch = x_mixed_imp_batch[complete_batch_size:]
        else:
            x_complete_imp_batch = None
            x_missing_imp_batch = x_mixed_imp_batch
        
        # 7.2 根据 loss_data 参数选择损失计算策略
        if self.args.loss_data == 'diffusion_only':
            # 只使用扩散损失的情况
            total_outer_loss = diffusion_loss
            
            # 构建监控指标
            metrics_log = {
                'diffusion_loss': diffusion_loss.item(),
                'explicit_bilevel_loss': 0.0,  # 不计算双层损失
                'hvp_norm': 0.0,  # 不计算 HVP
                'diffusion_weight': 1.0,
                'bilevel_weight': 0.0,
                'loss_strategy': 'diffusion_only'
            }
        else:
            # 原有的双层优化 + 扩散损失策略
            # 7.2 计算Explicit Bi-Level损失 - 直接在diffusion处理后的数据上计算
            explicit_bilevel_loss, hvp_norm = self.compute_explicit_bilevel_loss(
                classifier=classifier,
                x_imp_batch=x_missing_imp_batch,
                y_imp_batch=val_error_y
            )
            
            # 7.3 组合总损失 - 简化为diffusion + explicit bilevel
            lambda_diffusion = getattr(self.args, 'lambda_diffusion', 1.0)
            lambda_bilevel = getattr(self.args, 'lambda_bilevel', 1.0)
            
            total_outer_loss = lambda_diffusion * diffusion_loss + lambda_bilevel * explicit_bilevel_loss
            
            # 构建监控指标
            metrics_log = {
                'diffusion_loss': diffusion_loss.item(),
                'explicit_bilevel_loss': explicit_bilevel_loss.item(),
                'hvp_norm': hvp_norm,
                'diffusion_weight': lambda_diffusion,
                'bilevel_weight': lambda_bilevel,
                'loss_strategy': 'explicit_bilevel'
            }
        
        # 通用监控指标
        metrics_log['total_outer_loss'] = total_outer_loss.item()
        
        # ===========================================
        # 步骤8: 反向传播与更新
        # ===========================================
        
        total_outer_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.repair_model.parameters(), max_norm=1.0)
        if self.feature_embedder is not None:
            torch.nn.utils.clip_grad_norm_(self.feature_embedder.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 更新监控指标
        metrics_log['outer_loss'] = total_outer_loss.item()
        metrics_log['diffusion_integration'] = True  # 标记使用了集成方案
        
        return metrics_log

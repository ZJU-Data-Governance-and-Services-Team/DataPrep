import torch
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Union
from torch.utils.data import DataLoader, TensorDataset
from models.diffae.core import DiffusionUtils, TabularAE, TabularUNet
from datasets.data_processor import DataProcessor
from utils.feature_embedder import FeatureEmbedder
from tqdm import tqdm


class UnifiedImputation:
    """
    统一的填补方法类，支持基于扩散的填补和直接预测填补
    
    这个类实现了条件填补方法，可以选择使用训练好的扩散模型
    进行完整的扩散逆向过程填补，或者直接预测填补（不使用扩散过程）。
    """
    
    def __init__(
        self,
        model: Union[TabularAE, TabularUNet],
        feature_embedder: FeatureEmbedder,
        processor: DataProcessor,
        diffusion_utils: Optional[DiffusionUtils] = None,
        method: str = "diffusion",  # "diffusion" 或 "direct"
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        masked_label_value: int = 0,
        diffusion_embed: bool = True
    ):
        """
        初始化UnifiedImputation类
        
        Args:
            model: 训练好的扩散模型（TabularDiffAE或TabularUNet）
            feature_embedder: 训练好的特征嵌入器
            processor: 数据处理器实例
            diffusion_utils: DiffusionUtils实例，当method="diffusion"时必需
            method: 填补方法，"diffusion"表示使用扩散过程，"direct"表示直接预测
            device: 运行设备（"cuda"或"cpu"）
            masked_label_value: 掩码标签的值
            diffusion_embed: 是否在嵌入空间中进行扩散（True:嵌入空间，False:原始数据空间）
        """
        self.model = model
        self.feature_embedder = feature_embedder
        self.processor = processor
        self.diffusion_utils = diffusion_utils
        self.method = method
        self.device = device
        self.masked_label_value = masked_label_value
        self.diffusion_embed = diffusion_embed
        
        if method == "diffusion" and diffusion_utils is None:
            raise ValueError("当method='diffusion'时，必须提供diffusion_utils参数")
        
        # 设置模型为评估模式
        self.model.eval()
        self.feature_embedder.eval()
        
        # 获取维度和类别大小
        self.d_numerical = processor.d_numerical
        self.d_categorical = processor.d_categorical
        self.actual_cat_sizes = processor.categories
        self.d_embed = feature_embedder.d_embed
        
    def _logits_to_cat_indices(
        self, 
        pred_x0_cat_output: Union[list, torch.Tensor], 
        batch_size: int
    ) -> torch.Tensor:
        """
        将模型输出的分类logits转换为整数索引
        
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

    def _prepare_initial_data(
        self, 
        df: pd.DataFrame,
        dataset_name: str = None,
        external_error_mask: pd.DataFrame = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, pd.DataFrame]:
        """
        为填补准备初始数据，通过对缺失值进行特殊处理来转换数据
        
        Args:
            df: 包含缺失值的输入数据框
            dataset_name: 数据集名称，用于缓存优化
            external_error_mask: 外部提供的错误掩码DataFrame (True表示错误/缺失)
            
        Returns:
            x_num: 带有临时填充值的数值特征张量
            x_cat: 带有MASK标记的分类特征张量
            labels: 标签张量
            num_mask: 数值特征掩码（1=观测值，0=缺失值）
            cat_mask: 分类特征掩码（1=观测值，0=缺失值）
            df_temp: 带有填充值的临时数据框（供参考）
        """
        df_temp = df.copy()
        
        if external_error_mask is not None:
            # 使用外部提供的错误掩码
            x_num, x_cat, labels = self.processor.transform(df_temp, dataset_name=dataset_name)
            
            # 从外部错误掩码生成观测掩码
            if self.processor.num_features:
                num_error_mask = external_error_mask[self.processor.num_features].values
                num_mask = torch.tensor(~num_error_mask, dtype=torch.float32, device=self.device)
            else:
                num_mask = torch.empty((len(df_temp), 0), dtype=torch.float32, device=self.device)
                
            if self.processor.cat_features:
                cat_error_mask = external_error_mask[self.processor.cat_features].values
                cat_mask = torch.tensor(~cat_error_mask, dtype=torch.float32, device=self.device)
            else:
                cat_mask = torch.empty((len(df_temp), 0), dtype=torch.float32, device=self.device)
        else:
            raise ValueError("external_error_mask is required")
        
        
        # 将数据移动到设备
        if x_num is not None:
            x_num = x_num.to(self.device)
        if x_cat is not None:
            x_cat = x_cat.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        
        return x_num, x_cat, labels, num_mask, cat_mask, df_temp
    
    def _conditional_reverse_diffusion(
        self,
        x_num: torch.Tensor,
        x_cat: torch.Tensor,
        num_mask: torch.Tensor,
        cat_mask: torch.Tensor,
        y_labels: torch.Tensor = None,
        num_steps: int = 100,
        verbose: bool = False
    ) -> torch.Tensor:
        """
        执行条件逆向扩散以生成填补的嵌入
        支持两种模式：
        1. 嵌入空间扩散 (diffusion_embed=True)
        2. 原始数据空间扩散 (diffusion_embed=False)
        
        Args:
            x_num: 带有临时填充值的数值特征张量
            x_cat: 带有MASK标记的分类特征张量
            num_mask: 数值特征掩码（1=观测值，0=缺失值）
            cat_mask: 分类特征掩码（1=观测值，0=缺失值）
            y_labels: 用于条件化的标签张量（可选）
            num_steps: 扩散步数
            verbose: 是否显示进度条
            
        Returns:
            最终预测的嵌入或特征
        """
        if self.diffusion_embed:
            # 嵌入空间扩散模式
            return self._embedding_space_diffusion(
                x_num, x_cat, num_mask, cat_mask, y_labels, num_steps, verbose
            )
        else:
            # 原始数据空间扩散模式
            return self._original_space_diffusion(
                x_num, x_cat, num_mask, cat_mask, y_labels, num_steps, verbose
            )
    
    def _embedding_space_diffusion(
        self,
        x_num: torch.Tensor,
        x_cat: torch.Tensor,
        num_mask: torch.Tensor,
        cat_mask: torch.Tensor,
        y_labels: torch.Tensor = None,
        num_steps: int = 100,
        verbose: bool = False
    ) -> torch.Tensor:
        """
        在嵌入空间中执行条件逆向扩散（新版本）
        """
        # 获取特征级掩码（1=观测值，0=缺失值）用于原始特征
        if num_mask is not None and cat_mask is not None:
            feature_mask_original = torch.cat([num_mask, cat_mask], dim=1)
        elif num_mask is not None:
            feature_mask_original = num_mask
        elif cat_mask is not None:
            feature_mask_original = cat_mask
        else:
            raise ValueError("既没有数值特征也没有分类特征掩码")
        
        batch_size = x_num.shape[0] if x_num is not None and x_num.shape[0] > 0 else x_cat.shape[0]
        
        # 获取已知部分的干净嵌入 e_known
        e_known, _ = self.feature_embedder(x_num, x_cat, num_mask, cat_mask)
        
        # 将原始特征掩码扩展到嵌入维度
        emb_dim = e_known.shape[1]
        feature_emb_dim = emb_dim // feature_mask_original.shape[1] if feature_mask_original.shape[1] > 0 else emb_dim
        e_mask = feature_mask_original.unsqueeze(2).expand(-1, -1, feature_emb_dim).reshape(batch_size, -1).float()
        
        # 如果模型支持噪声预测模式，使用新的扩散过程
        if hasattr(self.model, 'predict_noise') and self.model.predict_noise:
            # 初始化：从完全随机的噪声嵌入 e_T 开始
            e_t = torch.randn(batch_size, emb_dim, device=self.device)
            
            # 设置时间步
            timesteps = torch.linspace(
                0, self.diffusion_utils.num_timesteps - 1, 
                num_steps, dtype=torch.int64, device=self.device
            )
            
            time_range = range(len(timesteps) - 1, -1, -1)
            
            for i in time_range:
                t = torch.full((batch_size,), timesteps[i], device=self.device, dtype=torch.long)
                
                # 构建条件化输入
                e_conditional_input = e_known * e_mask + e_t * (1 - e_mask)
                
                # 构建模型输入：[e_conditional, e_mask]
                model_input = torch.cat([e_conditional_input, e_mask], dim=1)
                
                with torch.no_grad():
                    # 模型预测噪声
                    epsilon_pred = self.model(model_input, t, y_labels=y_labels)
                    
                    # 确保epsilon_pred是单个张量
                    if isinstance(epsilon_pred, tuple):
                        epsilon_pred = epsilon_pred[0]
                
                # 使用预测的噪声和当前的e_t来估算e_0
                # 公式：e_0_pred = (e_t - sqrt(1 - alpha_cumprod_t) * epsilon) / sqrt(alpha_cumprod_t)
                alpha_cumprod_t = self.diffusion_utils.alphas_cumprod[t].view(-1, 1)
                sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod_t)
                sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - alpha_cumprod_t)
                
                e_0_pred = (e_t - sqrt_one_minus_alpha_cumprod_t * epsilon_pred) / sqrt_alpha_cumprod_t
                
                # 如果不是最后一步，计算e_{t-1}
                if i > 0:
                    # 获取后验分布参数
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
            
            return e_t
        
        # 将原始特征掩码扩展到嵌入维度以便最终组合
        emb_mask = self.feature_embedder.expand_emb_mask(feature_mask_original)
        
        timesteps = torch.linspace(
            0, self.diffusion_utils.num_timesteps - 1, 
            num_steps, dtype=torch.int64, device=self.device
        )
        
        if x_num is not None:
            batch_size = x_num.shape[0]
        elif x_cat is not None:
            batch_size = x_cat.shape[0]
        else:
            raise ValueError("没有数据进行填补")

        if self.d_numerical > 0:
            noisy_num_target = torch.randn_like(x_num) * (1 - num_mask)
        else:
            noisy_num_target = torch.empty(batch_size, 0, device=self.device)
        
        if self.d_categorical > 0:
            noisy_cat_target = x_cat.clone()
            # 对于缺失的分类特征位置，使用MASK标记
            for j in range(self.d_categorical):
                if j < cat_mask.shape[1]:
                    missing_mask = (cat_mask[:, j] == 0)
                    if missing_mask.any():
                        # 使用MASK索引（每个分类特征的最后一个索引）
                        cat_size = self.actual_cat_sizes[j] if j < len(self.actual_cat_sizes) else 2
                        mask_idx = cat_size - 1  # MASK是最后一个索引
                        noisy_cat_target[missing_mask, j] = mask_idx
            # 将已知位置置零，确保noisy通道只包含目标区域的信息
            noisy_cat_target = noisy_cat_target * (1 - cat_mask.long())
        else:
            noisy_cat_target = torch.empty(batch_size, 0, dtype=torch.long, device=self.device)
        
        # 为 e_noisy 的计算提前定义掩码
        ones_mask_num = torch.ones_like(noisy_num_target)
        ones_mask_cat = torch.ones_like(noisy_cat_target, dtype=torch.float)

        with torch.no_grad():
            e_noisy, _ = self.feature_embedder(noisy_num_target, noisy_cat_target, ones_mask_num, ones_mask_cat)
            e0_filled, _ = self.feature_embedder(x_num, x_cat, num_mask, cat_mask)
            e_cond = e0_filled * emb_mask
        
        emb_dim = e_noisy.shape[1]
        feature_emb_dim = emb_dim // feature_mask_original.shape[1] if feature_mask_original.shape[1] > 0 else emb_dim
        e_mask = feature_mask_original.unsqueeze(2).expand(-1, -1, feature_emb_dim).reshape(batch_size, -1).float()

        time_range = range(len(timesteps) - 1, -1, -1)
        
        for i in time_range:
            t = torch.full((batch_size,), timesteps[i], device=self.device, dtype=torch.long)
            t_idx = t.long()

            e_noisy_input = e_noisy * e_mask + e_cond * (1-e_mask)
            e_input = torch.cat([e_noisy_input, e_mask], dim=1)
            with torch.no_grad():
                pred_x0_num_from_model, pred_x0_cat_output = self.model(e_input, t, y_labels=y_labels)
            
            # 将预测的分类logits转换为索引
            pred_x0_cat_indices_from_model = self._logits_to_cat_indices(pred_x0_cat_output, batch_size)

            # 处理数值预测
            if pred_x0_num_from_model is None or pred_x0_num_from_model.nelement() == 0:
                pred_x0_num_from_model = torch.empty(batch_size, 0, device=self.device)

            # 4. 重新嵌入预测的干净原始特征
            ones_mask_num = torch.ones_like(pred_x0_num_from_model, device=self.device)
            ones_mask_cat = torch.ones_like(pred_x0_cat_indices_from_model, dtype=torch.float32, device=self.device)

            with torch.no_grad():
                e0_pred_from_model, _ = self.feature_embedder(
                    pred_x0_num_from_model,
                    pred_x0_cat_indices_from_model,
                    ones_mask_num,
                    ones_mask_cat
                )
            
            # 5. 执行扩散步骤
            posterior_mean_coef1 = self.diffusion_utils.posterior_mean_coef1.to(self.device)[t_idx].view(-1, 1)
            posterior_mean_coef2 = self.diffusion_utils.posterior_mean_coef2.to(self.device)[t_idx].view(-1, 1)
            posterior_log_variance = self.diffusion_utils.posterior_log_variance_clipped.to(self.device)[t_idx].view(-1, 1)
            
            # 1. 定义当前时间步 t 的完整嵌入 e_t
            # e_t 由已知的干净部分 e_cond 和未知的含噪部分 e_noisy 组成
            e_t = e_cond * emb_mask + e_noisy * (1 - emb_mask)

            # 2. 计算后验均值，使用完整的 e_t
            # DDPM的后验公式需要完整的含噪输入 e_t
            posterior_mean_pred = posterior_mean_coef1 * e0_pred_from_model + posterior_mean_coef2 * e_t
            
            # 3. 采样与 e_t 相同形状的噪声
            noise = torch.randn_like(e_t) if t[0] > 0 else torch.zeros_like(e_t)
            
            # 采样e_{t-1}
            e_t_minus_1_pred = posterior_mean_pred + torch.exp(0.5 * posterior_log_variance) * noise
            
            # 6. 使用嵌入掩码组合预测部分，更新 e_noisy 以用于下一次迭代
            e_noisy = e_t_minus_1_pred * (1 - emb_mask)

        return e_noisy * (1 - emb_mask) + e0_filled * emb_mask
    
    def _original_space_diffusion(
        self,
        x_num: torch.Tensor,
        x_cat: torch.Tensor,
        num_mask: torch.Tensor,
        cat_mask: torch.Tensor,
        y_labels: torch.Tensor = None,
        num_steps: int = 100,
        verbose: bool = False
    ) -> torch.Tensor:
        """
        在原始数据空间中执行条件逆向扩散（旧版本）
        """
        # 获取特征级掩码（1=观测值，0=缺失值）用于原始特征
        if num_mask is not None and cat_mask is not None:
            feature_mask_original = torch.cat([num_mask, cat_mask], dim=1)
        elif num_mask is not None:
            feature_mask_original = num_mask
        elif cat_mask is not None:
            feature_mask_original = cat_mask
        else:
            raise ValueError("既没有数值特征也没有分类特征掩码")
        
        batch_size = x_num.shape[0] if x_num is not None and x_num.shape[0] > 0 else x_cat.shape[0]
        
        # 获取已知部分的干净嵌入 e_known (用于条件化)
        e_known, _ = self.feature_embedder(x_num, x_cat, num_mask, cat_mask)
        
        # 将原始特征掩码扩展到嵌入维度
        emb_dim = e_known.shape[1]
        feature_emb_dim = emb_dim // feature_mask_original.shape[1] if feature_mask_original.shape[1] > 0 else emb_dim
        emb_mask = self.feature_embedder.expand_emb_mask(feature_mask_original)
        
        # 设置时间步
        timesteps = torch.linspace(
            0, self.diffusion_utils.num_timesteps - 1, 
            num_steps, dtype=torch.int64, device=self.device
        )
        
        # 准备初始的噪声数据（对于缺失位置）
        if self.d_numerical > 0:
            noisy_num_target = torch.randn_like(x_num) * (1 - num_mask)
        else:
            noisy_num_target = torch.empty(batch_size, 0, device=self.device)
        
        if self.d_categorical > 0:
            noisy_cat_target = x_cat.clone()
            # 对于缺失的分类特征位置，使用MASK标记
            for j in range(self.d_categorical):
                if j < cat_mask.shape[1]:
                    missing_mask = (cat_mask[:, j] == 0)
                    if missing_mask.any():
                        # 使用MASK索引（每个分类特征的最后一个索引）
                        cat_size = self.actual_cat_sizes[j] if j < len(self.actual_cat_sizes) else 2
                        mask_idx = cat_size - 1  # MASK是最后一个索引
                        noisy_cat_target[missing_mask, j] = mask_idx
            # 将已知位置置零，确保noisy通道只包含目标区域的信息
            noisy_cat_target = noisy_cat_target * (1 - cat_mask.long())
        else:
            noisy_cat_target = torch.empty(batch_size, 0, dtype=torch.long, device=self.device)
        
        # 为 e_noisy 的计算提前定义掩码
        ones_mask_num = torch.ones_like(noisy_num_target)
        ones_mask_cat = torch.ones_like(noisy_cat_target, dtype=torch.float)

        with torch.no_grad():
            e_noisy, _ = self.feature_embedder(noisy_num_target, noisy_cat_target, ones_mask_num, ones_mask_cat)
            e0_filled, _ = self.feature_embedder(x_num, x_cat, num_mask, cat_mask)
            e_cond = e0_filled * emb_mask
        
        # 计算e_mask（用于模型输入的第三个通道）
        e_mask = feature_mask_original.unsqueeze(2).expand(-1, -1, feature_emb_dim).reshape(batch_size, -1).float()

        time_range = range(len(timesteps) - 1, -1, -1)
        
        if verbose:
            time_range = tqdm(time_range, desc="逆向扩散")
        
        for i in time_range:
            t = torch.full((batch_size,), timesteps[i], device=self.device, dtype=torch.long)
            t_idx = t.long()

            # 构建三通道输入
            e_noisy_input = e_noisy * e_mask + e_cond * (1-e_mask)
            e_input = torch.cat([e_noisy_input, e_mask], dim=1)
            with torch.no_grad():
                pred_x0_num_from_model, pred_x0_cat_output = self.model(e_input, t, y_labels=y_labels)
            
            # 将预测的分类logits转换为索引
            pred_x0_cat_indices_from_model = self._logits_to_cat_indices(pred_x0_cat_output, batch_size)

            # 处理数值预测
            if pred_x0_num_from_model is None or pred_x0_num_from_model.nelement() == 0:
                pred_x0_num_from_model = torch.empty(batch_size, 0, device=self.device)

            # 重新嵌入预测的干净原始特征
            ones_mask_num = torch.ones_like(pred_x0_num_from_model, device=self.device)
            ones_mask_cat = torch.ones_like(pred_x0_cat_indices_from_model, dtype=torch.float32, device=self.device)

            with torch.no_grad():
                e0_pred_from_model, _ = self.feature_embedder(
                    pred_x0_num_from_model,
                    pred_x0_cat_indices_from_model,
                    ones_mask_num,
                    ones_mask_cat
                )
            
            # 执行DDPM后验采样步骤
            if i > 0:
                posterior_mean_coef1 = self.diffusion_utils.posterior_mean_coef1.to(self.device)[t_idx].view(-1, 1)
                posterior_mean_coef2 = self.diffusion_utils.posterior_mean_coef2.to(self.device)[t_idx].view(-1, 1)
                posterior_log_variance = self.diffusion_utils.posterior_log_variance_clipped.to(self.device)[t_idx].view(-1, 1)
                
                # 定义当前时间步 t 的完整嵌入 e_t
                e_t = e_cond * emb_mask + e_noisy * (1 - emb_mask)

                # 计算后验均值，使用完整的 e_t
                posterior_mean_pred = posterior_mean_coef1 * e0_pred_from_model + posterior_mean_coef2 * e_t
                
                # 采样噪声
                noise = torch.randn_like(e_t)
                
                # 采样e_{t-1}
                e_t_minus_1_pred = posterior_mean_pred + torch.exp(0.5 * posterior_log_variance) * noise
                
                # 使用嵌入掩码组合预测部分，更新 e_noisy 以用于下一次迭代
                e_noisy = e_t_minus_1_pred * (1 - emb_mask)
            else:
                # 最后一步，直接使用预测的e_0
                e_noisy = e0_pred_from_model * (1 - emb_mask)

        # 返回最终的填补嵌入
        return e_noisy * (1 - emb_mask) + e0_filled * emb_mask
    
    def _direct_prediction(
        self,
        x_num: torch.Tensor,
        x_cat: torch.Tensor,
        num_mask: torch.Tensor,
        cat_mask: torch.Tensor,
        y_labels: torch.Tensor = None,
        verbose: bool = False
    ) -> torch.Tensor:
        """
        直接预测填补嵌入，不使用扩散过程
        
        Args:
            x_num: 带有临时填充值的数值特征张量
            x_cat: 带有MASK标记的分类特征张量
            num_mask: 数值特征掩码（1=观测值，0=缺失值）
            cat_mask: 分类特征掩码（1=观测值，0=缺失值）
            y_labels: 用于条件化的标签张量（可选）
            verbose: 是否显示进度条
            
        Returns:
            最终预测的嵌入
        """
        # 获取已知部分的初始嵌入
        with torch.no_grad():
            e0_filled, _ = self.feature_embedder(x_num, x_cat, num_mask, cat_mask)
        
        # 获取特征级掩码
        feature_mask_original = None
        if num_mask is not None and cat_mask is not None:
            feature_mask_original = torch.cat([num_mask, cat_mask], dim=1)
        elif num_mask is None and cat_mask is not None:
            feature_mask_original = cat_mask
        elif num_mask is not None and cat_mask is None:
            feature_mask_original = num_mask
        else:
            raise ValueError("既没有数值特征也没有分类特征掩码，无法进行填补")
        
        # 将原始特征掩码扩展到嵌入维度
        emb_mask = self.feature_embedder.expand_emb_mask(feature_mask_original)
        
        if x_num is not None and x_num.shape[0] > 0:
            batch_size = x_num.shape[0]
        elif x_cat is not None and x_cat.shape[0] > 0:
            batch_size = x_cat.shape[0]
        else:
            raise ValueError("既没有数值数据也没有分类数据，无法进行填补")
        
        # 判断是否使用新的噪声预测架构
        if hasattr(self.model, 'predict_noise') and self.model.predict_noise:
            # 新架构：模型预测噪声，使用2通道输入
            # 对于直接预测，我们使用t=0，此时模型应该预测接近0的噪声
            t = torch.zeros((batch_size,), device=self.device, dtype=torch.long)
            
            # 构建条件嵌入：已知位置使用真实嵌入，缺失位置使用零
            # e_conditional = e0_filled * emb_mask
            
            # 构建2通道输入
            model_input = torch.cat([e0_filled, emb_mask], dim=1)
            
            # 模型预测噪声（在t=0时应该接近0）
            with torch.no_grad():
                epsilon_pred = self.model(model_input, t, y_labels=y_labels)
            
            # 对于直接预测，我们假设没有噪声，所以预测的嵌入就是条件嵌入
            # 在缺失位置，模型会学习如何填补
            pred_embedding = e0_filled*emb_mask + (1 - emb_mask) * epsilon_pred
            
            return pred_embedding
        else:
            # 原架构：使用3通道输入，模型直接预测特征
            # 使用t=0的时间步，让模型直接预测x0
            t = torch.zeros((batch_size,), device=self.device, dtype=torch.long)
        
        # 构建三通道输入，按照compute_diffusion_loss的语义
        # 构建 noisy_target: 对于缺失位置使用预填充值（而不是零），已知位置为0
        if x_num is not None and x_num.shape[1] > 0:
            noisy_num_target = x_num * (1 - num_mask)
        else:
            noisy_num_target = torch.empty(batch_size, 0, device=self.device)
            
        if x_cat is not None and x_cat.shape[1] > 0:
            noisy_cat_target = x_cat.clone()
            # 对于缺失的分类特征位置，保持原值（因为是直接预测模式），已知部分置零
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
        # 为所有输入创建全1的掩码（因为我们要嵌入所有位置）
        ones_mask_num = torch.ones_like(noisy_num_target) if noisy_num_target.shape[1] > 0 else torch.empty(batch_size, 0, device=self.device)
        ones_mask_cat = torch.ones_like(noisy_cat_target, dtype=torch.float) if noisy_cat_target.shape[1] > 0 else torch.empty(batch_size, 0, device=self.device)
        
        with torch.no_grad():
            e_noisy, _ = self.feature_embedder(noisy_num_target, noisy_cat_target, ones_mask_num, ones_mask_cat)
            e_cond, _ = self.feature_embedder(cond_num_data, cond_cat_data, ones_mask_num, ones_mask_cat)
        
        # 构建 e_mask: 将原始特征掩码扩展到embedding维度
        emb_dim = e_noisy.shape[1]
        feature_emb_dim = emb_dim // feature_mask_original.shape[1] if feature_mask_original.shape[1] > 0 else emb_dim
        e_mask = feature_mask_original.unsqueeze(2).expand(-1, -1, feature_emb_dim).reshape(batch_size, -1).float()
        
        # 拼接三个通道的嵌入
        e_noisy_input = e_noisy * e_mask + e_cond * (1-e_mask)
        e_input = torch.cat([e_noisy_input, e_mask], dim=1)
        
        # 模型预测原始特征
        with torch.no_grad():
            pred_x0_num_from_model, pred_x0_cat_output = self.model(e_input, t, y_labels=y_labels)
        
        # 将预测的分类logits转换为索引
        pred_x0_cat_indices_from_model = self._logits_to_cat_indices(pred_x0_cat_output, batch_size)

        # 处理数值预测
        if pred_x0_num_from_model is None or pred_x0_num_from_model.nelement() == 0:
            pred_x0_num_from_model = torch.empty(batch_size, 0, device=self.device)

        # 重新嵌入预测的特征
        ones_mask_num = torch.ones_like(pred_x0_num_from_model, device=self.device)
        ones_mask_cat = torch.ones_like(pred_x0_cat_indices_from_model, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            e0_pred_from_model, _ = self.feature_embedder(
                pred_x0_num_from_model,
                pred_x0_cat_indices_from_model,
                ones_mask_num,
                ones_mask_cat
            )
        
        # 组合已知和预测部分
        final_embeddings = e0_filled * emb_mask + e0_pred_from_model * (1 - emb_mask)
        
        return final_embeddings
    
    def _decode_embeddings(
        self,
        embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将嵌入解码回数值和分类特征
        
        Args:
            embeddings: 最终预测的嵌入
            
        Returns:
            数值和分类特征张量
        """
        with torch.no_grad():
            return self.feature_embedder.decode_embeddings(embeddings)
    
    def _impute_batch(
        self,
        df_batch: pd.DataFrame,
        y_labels: torch.Tensor = None,
        num_steps: int = 100,
        verbose: bool = False,
        dataset_name: str = None,
        external_error_mask: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        为一批数据填补缺失值
        
        Args:
            df_batch: 带有缺失值的数据框批次
            y_labels: 用于条件化的标签张量（可选）
            num_steps: 扩散步数（仅对diffusion方法有效）
            verbose: 是否显示进度条
            dataset_name: 数据集名称，用于缓存优化
            external_error_mask: 外部提供的错误掩码DataFrame
            
        Returns:
            带有填补值的数据框
        """
        x_num, x_cat, labels, num_mask, cat_mask, _ = self._prepare_initial_data(
            df_batch, dataset_name=dataset_name, external_error_mask=external_error_mask
        )
        
        # 准备条件标签
        if y_labels is None and labels is not None:
            y_cond = labels
        else:
            y_cond = y_labels
            
        # 如果使用掩码标签
        if y_cond is not None and hasattr(self, 'masked_label_value') and self.masked_label_value is not None:
            y_cond = torch.full_like(y_cond, self.masked_label_value)
        
        # 根据方法选择填补策略
        if self.method == "diffusion":
            embeddings = self._conditional_reverse_diffusion(
                x_num, x_cat, num_mask, cat_mask, y_cond, num_steps, verbose
            )
        elif self.method == "direct":
            embeddings = self._direct_prediction(
                x_num, x_cat, num_mask, cat_mask, y_cond, verbose
            )
        else:
            raise ValueError(f"不支持的方法: {self.method}")
        
        x_num_imputed, x_cat_imputed = self.model._reconstruct_output(embeddings)
        
        # 对于direct方法，需要转换分类logits为索引
        x_cat_imputed = self._logits_to_cat_indices(x_cat_imputed, len(x_cat_imputed))
        
        imputed_df = self.processor.inverse_transform(x_num_imputed, x_cat_imputed)
        
        # 为观测特征复制原始值
        result_df = df_batch.copy()
        
        # if self.method == "diffusion":
        #     # diffusion方法：只替换缺失值
        #     for i, col in enumerate(self.processor.num_features):
        #         mask = result_df[col].isna()
        #         result_df.loc[mask, col] = imputed_df.loc[mask, col]
                
        #     for i, col in enumerate(self.processor.cat_features):
        #         mask = result_df[col].isna()
        #         result_df.loc[mask, col] = imputed_df.loc[mask, col]
        # else:
        #     # direct方法：使用掩码
        if num_mask is not None:
            missing_num_mask = (1 - num_mask).cpu().numpy().astype(bool)
            for i, col in enumerate(self.processor.num_features):
                mask = missing_num_mask[:, i]
                result_df.loc[mask, col] = imputed_df.loc[mask, col]
        
        if cat_mask is not None:
            missing_cat_mask = (1 - cat_mask).cpu().numpy().astype(bool)
            for i, col in enumerate(self.processor.cat_features):
                mask = missing_cat_mask[:, i]
                result_df.loc[mask, col] = imputed_df.loc[mask, col]
            
        return result_df
    
    def impute(
        self,
        df: pd.DataFrame,
        batch_size: int = 32,
        num_steps: int = 100,
        verbose: bool = False,
        use_masked_labels: bool = True,
        external_labels: torch.Tensor = None,
        dataset_name: str = None,
        external_error_mask: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        为整个数据框填补缺失值
        
        Args:
            df: 带有缺失值的数据框
            batch_size: 处理的批次大小
            num_steps: 扩散步数（仅对diffusion方法有效）
            verbose: 是否显示进度
            use_masked_labels: 如果为True，对所有样本使用掩码标签值
            external_labels: 可选的外部标签，用于代替数据框中的标签
            dataset_name: 数据集名称，用于缓存优化
            external_error_mask: 外部提供的错误掩码DataFrame
            
        Returns:
            带有填补值的数据框
        """
        if len(df) == 0:
            raise ValueError("空数据框")
        
        result_dfs = []
        
        for i in range(0, len(df), batch_size):
                
            df_batch = df.iloc[i:i+batch_size].reset_index(drop=True)
            
            # 准备外部错误掩码的批次
            if external_error_mask is not None:
                batch_error_mask = external_error_mask.iloc[i:i+batch_size].reset_index(drop=True)
            else:
                batch_error_mask = None
            
            # 准备外部标签
            if external_labels is not None:
                batch_labels = external_labels[i:i+batch_size].to(self.device)
            else:
                batch_labels = None
                
            # 如果使用掩码标签
            if use_masked_labels and hasattr(self.model, 'masked_label_value') and self.model.masked_label_value is not None:
                batch_size_actual = len(df_batch)
                batch_labels = torch.full((batch_size_actual,), self.model.masked_label_value, device=self.device)
            
            imputed_batch = self._impute_batch(df_batch, batch_labels, num_steps, verbose, dataset_name=dataset_name, external_error_mask=batch_error_mask)
            result_dfs.append(imputed_batch)
            
        result_df = pd.concat(result_dfs, ignore_index=True)
        
        return result_df


def unified_imputation(
    df: pd.DataFrame,
    model: Union[TabularAE, TabularUNet],
    feature_embedder: FeatureEmbedder,
    processor: DataProcessor,
    method: str = "diffusion",
    diffusion_utils: Optional[DiffusionUtils] = None,
    num_steps: int = 100,
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = False,
    use_masked_labels: bool = True,
    external_labels: torch.Tensor = None,
    masked_label_value: int = 0,
    dataset_name: str = None,
    diffusion_embed: bool = True,
    external_error_mask: pd.DataFrame = None
) -> pd.DataFrame:
    """
    使用统一接口填补数据框中的缺失值的便利函数
    
    Args:
        df: 包含缺失值的输入数据框
        model: 训练好的扩散模型（TabularDiffAE或TabularUNet）
        feature_embedder: 训练好的特征嵌入器
        processor: 用于数据转换的DataProcessor实例
        method: 填补方法，"diffusion"表示使用扩散过程，"direct"表示直接预测
        diffusion_utils: DiffusionUtils实例，当method="diffusion"时必需
        num_steps: 扩散步数（仅对diffusion方法有效）
        batch_size: 处理的批次大小
        device: 运行模型的设备（"cuda"或"cpu"）
        verbose: 是否打印进度
        use_masked_labels: 如果为True，对所有样本使用掩码标签值
        external_labels: 可选的外部标签，用于代替数据框中的标签
        masked_label_value: 用于掩码标签的值
        dataset_name: 数据集名称，用于缓存优化
        diffusion_embed: 是否在嵌入空间中进行扩散（True:嵌入空间，False:原始数据空间）
        external_error_mask: 外部提供的错误掩码DataFrame
        
    Returns:
        带有填补值的数据框
    """
    imputer = UnifiedImputation(
        model=model,
        feature_embedder=feature_embedder,
        processor=processor,
        diffusion_utils=diffusion_utils,
        method=method,
        device=device,
        masked_label_value=masked_label_value,
        diffusion_embed=diffusion_embed
    )
    
    return imputer.impute(
        df=df,
        batch_size=batch_size,
        num_steps=num_steps,
        verbose=verbose,
        use_masked_labels=use_masked_labels,
        external_labels=external_labels,
        dataset_name=dataset_name,
        external_error_mask=external_error_mask
    ) 
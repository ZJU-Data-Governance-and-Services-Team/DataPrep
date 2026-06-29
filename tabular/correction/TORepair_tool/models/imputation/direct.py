import torch
import pandas as pd
from utils.feature_embedder import FeatureEmbedder
from typing import Tuple, Union
from models.diffae.core import TabularAE, TabularUNet
from datasets.data_processor import DataProcessor


class DirectImputation:
    """
    直接预测式填补方法，不使用扩散过程
    
    这个类实现了一个条件填补方法，使用训练好的扩散模型直接预测缺失值，
    而不经过扩散逆向过程。这用于评估扩散机制的有效性。
    """
    
    def __init__(
        self,
        model: Union[TabularAE, TabularUNet],
        feature_embedder: FeatureEmbedder,
        processor: DataProcessor,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        masked_label_value: int = 0
    ):
        """
        初始化DirectImputation类
        
        Args:
            model: 训练好的扩散模型（TabularDiffAE或TabularUNet）
            feature_embedder: 训练好的特征嵌入器
            processor: 数据处理器实例
            device: 运行设备（"cuda"或"cpu"）
            masked_label_value: 掩码标签的值
        """
        self.model = model
        self.feature_embedder = feature_embedder
        self.processor = processor
        self.device = device
        self.masked_label_value = masked_label_value
        
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
        将模型输出的分类logits转换为整数索引。
        
        Args:
            pred_x0_cat_output: 模型的分类输出。可以是logits列表（来自TabularDiffAE）
                                或单个logits张量（来自TabularUNet）。
            batch_size: 当前批次的大小。
            
        Returns:
            一个形状为[batch_size, num_categorical_features]的整数索引张量。
        """
        cat_indices_list = []
        if self.d_categorical > 0:
            if isinstance(pred_x0_cat_output, list):  # 例如，带有cat_reconstructor的TabularDiffAE
                for logits_i in pred_x0_cat_output:
                    if logits_i.nelement() > 0:
                         cat_indices_list.append(torch.argmax(logits_i, dim=1))
            elif pred_x0_cat_output is not None and pred_x0_cat_output.nelement() > 0:  # 例如，TabularUNet（平面logits）
                current_pos = 0
                for cat_size in self.actual_cat_sizes:  # 来自processor的self.actual_cat_sizes
                    if cat_size > 0:
                        logits_i = pred_x0_cat_output[:, current_pos : current_pos + cat_size]
                        cat_indices_list.append(torch.argmax(logits_i, dim=1))
                        current_pos += cat_size
        
        return torch.stack(cat_indices_list, dim=1) if cat_indices_list else \
               torch.empty(batch_size, 0, dtype=torch.long, device=self.device)

    def _prepare_initial_data(
        self, 
        df: pd.DataFrame,
        dataset_name: str = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, pd.DataFrame]:
        """
        为填补准备初始数据，通过对缺失值进行特殊处理来转换数据
        
        Args:
            df: 包含缺失值的输入数据框
            dataset_name: 数据集名称，用于缓存优化
            
        Returns:
            x_num: 带有临时填充值的数值特征张量
            x_cat: 带有MASK标记的分类特征张量  
            labels: 标签张量
            num_mask: 数值特征掩码（1=观测值，0=缺失值）
            cat_mask: 分类特征掩码（1=观测值，0=缺失值）
            df_temp: 带有填充值的临时数据框（供参考）
        """
        df_temp = df.copy()
        x_num, x_cat, labels, num_mask, cat_mask = self.processor.transform(df_temp, dataset_name=dataset_name)
        
        # 将数据移动到设备
        if x_num is not None:
            x_num = x_num.to(self.device)
        if x_cat is not None:
            x_cat = x_cat.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        if num_mask is not None:
            num_mask = num_mask.to(self.device)
        if cat_mask is not None:
            cat_mask = cat_mask.to(self.device)
        
        return x_num, x_cat, labels, num_mask, cat_mask, df_temp
    
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
        
        feature_mask_original = None
        # 获取特征级掩码（1=观测值，0=缺失值）用于原始特征
        # 这个掩码将用于在每步结束时组合已知和预测部分
        if num_mask is not None and cat_mask is not None:
            feature_mask_original = torch.cat([num_mask, cat_mask], dim=1)
        elif num_mask is None and cat_mask is not None:
            feature_mask_original = cat_mask
        elif num_mask is not None and cat_mask is None:
            feature_mask_original = num_mask
        else:
            raise ValueError("既没有数值特征也没有分类特征掩码，无法进行填补")
        
        # 将原始特征掩码扩展到嵌入维度以便最终组合
        emb_mask = self.feature_embedder.expand_emb_mask(feature_mask_original)
        
        if x_num is not None and x_num.shape[0] > 0:
            batch_size = x_num.shape[0]
        elif x_cat is not None and x_cat.shape[0] > 0:
            batch_size = x_cat.shape[0]
        else:
            raise ValueError("既没有数值数据也没有分类数据，无法进行填补")
        
        # 使用t=0的时间步，让模型直接预测x0（无噪声状态）
        t = torch.zeros((batch_size,), device=self.device, dtype=torch.long)
        
        # 模型预测原始特征（x0_num, x0_cat_logits/list）从初始嵌入
        with torch.no_grad():
            # 将标签条件传递给模型
            pred_x0_num_from_model, pred_x0_cat_output = self.model(e0_filled, t, y_labels=y_labels)
        
        # 将预测的分类logits转换为索引
        pred_x0_cat_indices_from_model = self._logits_to_cat_indices(pred_x0_cat_output, batch_size)

        # 处理数值预测（确保它是张量，即使为空）
        if pred_x0_num_from_model is None or pred_x0_num_from_model.nelement() == 0:
            pred_x0_num_from_model = torch.empty(batch_size, 0, device=self.device)

        # 重新嵌入预测的清洁原始特征以获得e0_pred_from_model
        # 为重新嵌入创建全1掩码，因为这些是模型对清洁数据的最佳猜测
        ones_mask_num = torch.ones_like(pred_x0_num_from_model, device=self.device)
        ones_mask_cat = torch.ones_like(pred_x0_cat_indices_from_model, dtype=torch.float32, device=self.device)  # 掩码应该是float

        with torch.no_grad():
            e0_pred_from_model, _ = self.feature_embedder(
                pred_x0_num_from_model,
                pred_x0_cat_indices_from_model,
                ones_mask_num,
                ones_mask_cat
            )
        
        # 使用嵌入掩码组合已知和预测部分
        final_embeddings = e0_filled * emb_mask + e0_pred_from_model * (1 - emb_mask)
        
        # 返回最终填补的嵌入
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
        verbose: bool = False,
        dataset_name: str = None
    ) -> pd.DataFrame:
        """
        为一批数据填补缺失值
        
        Args:
            df_batch: 带有缺失值的数据框批次
            y_labels: 用于条件化的标签张量（可选）
            verbose: 是否显示进度条
            dataset_name: 数据集名称，用于缓存优化
            
        Returns:
            带有填补值的数据框
        """
        # 在处理前获取原始缺失值掩码
        # 这里num_mask里1表示观测值，0表示缺失值
        x_num, x_cat, labels, num_mask, cat_mask, _ = self._prepare_initial_data(df_batch, dataset_name=dataset_name)
        
        # 如果没有提供外部标签，使用处理后的内部标签
        if y_labels is None and labels is not None:
            # 这里可以选择使用实际标签或者掩码标签
            # 对于训练数据，我们可能想用实际标签；对于测试数据，使用掩码标签
            y_cond = labels
        else:
            y_cond = y_labels
            
        # 如果掩码所有标签，且掩码值存在
        if y_cond is not None and hasattr(self, 'masked_label_value') and self.masked_label_value is not None:
            # 全部替换为掩码值
            y_cond = torch.full_like(y_cond, self.masked_label_value)
        
        embeddings = self._direct_prediction(
            x_num, x_cat, num_mask, cat_mask, y_cond, verbose
        )
        
        x_num_imputed, x_cat_imputed = self.model._reconstruct_output(embeddings)
        x_cat_imputed = self._logits_to_cat_indices(x_cat_imputed, len(x_cat_imputed))
        
        imputed_df = self.processor.inverse_transform(x_num_imputed, x_cat_imputed)
        
        # 为观测特征复制原始值，使用原始缺失值掩码
        # 在这里需要反转num_mask和cat_mask
        num_mask = 1 - num_mask if num_mask is not None else None
        cat_mask = 1 - cat_mask if cat_mask is not None else None
        result_df = df_batch.copy()
        for i, col in enumerate(self.processor.num_features):
            mask = num_mask[:, i].cpu().numpy().astype(bool)
            result_df.loc[mask, col] = imputed_df.loc[mask, col]
        
        for i, col in enumerate(self.processor.cat_features):
            mask = cat_mask[:, i].cpu().numpy().astype(bool)
            result_df.loc[mask, col] = imputed_df.loc[mask, col]
            
        return result_df
    
    def impute(
        self,
        df: pd.DataFrame,
        batch_size: int = 32,
        verbose: bool = False,
        use_masked_labels: bool = True,
        external_labels: torch.Tensor = None,
        dataset_name: str = None
    ) -> pd.DataFrame:
        """
        为整个数据框填补缺失值
        
        Args:
            df: 带有缺失值的数据框
            batch_size: 处理的批次大小
            verbose: 是否显示进度
            use_masked_labels: 如果为True，对所有样本使用掩码标签值
            external_labels: 可选的外部标签，用于代替数据框中的标签
            dataset_name: 数据集名称，用于缓存优化
            
        Returns:
            带有填补值的数据框
        """
        if len(df) == 0:
            raise ValueError("空数据框")
        
        result_dfs = []
        
        for i in (range(0, len(df), batch_size)):
                
            df_batch = df.iloc[i:i+batch_size].reset_index(drop=True)
            
            # 准备外部标签（如果有）
            if external_labels is not None:
                batch_labels = external_labels[i:i+batch_size].to(self.device)
            else:
                batch_labels = None
                
            # 如果使用掩码标签，将所有标签设为掩码值
            if use_masked_labels and hasattr(self.model, 'masked_label_value') and self.model.masked_label_value is not None:
                # 这将覆盖batch_labels
                batch_size = len(df_batch)
                batch_labels = torch.full((batch_size,), self.model.masked_label_value, device=self.device)
            
            imputed_batch = self._impute_batch(df_batch, batch_labels, verbose, dataset_name=dataset_name)
            result_dfs.append(imputed_batch)
            
        result_df = pd.concat(result_dfs, ignore_index=True)
        
        return result_df


def rep_embeds_direct(
    df: pd.DataFrame,
    model: Union[TabularAE, TabularUNet],
    feature_embedder: FeatureEmbedder,
    processor: DataProcessor,
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = False,
    use_masked_labels: bool = True,
    external_labels: torch.Tensor = None,
    masked_label_value: int = 0,
    dataset_name: str = None
) -> pd.DataFrame:
    """
    使用直接预测方法填补数据框中的缺失值的便利函数（不使用扩散过程）
    
    Args:
        df: 包含缺失值的输入数据框
        model: 训练好的扩散模型（TabularDiffAE或TabularUNet）
        feature_embedder: 训练好的特征嵌入器
        processor: 用于数据转换的DataProcessor实例
        batch_size: 处理的批次大小
        device: 运行模型的设备（"cuda"或"cpu"）
        verbose: 是否打印进度
        use_masked_labels: 如果为True，对所有样本使用掩码标签值
        external_labels: 可选的外部标签，用于代替数据框中的标签
        masked_label_value: 用于掩码标签的值
        dataset_name: 数据集名称，用于缓存优化
        
    Returns:
        带有填补值的数据框
    """
    imputer = DirectImputation(
        model=model,
        feature_embedder=feature_embedder,
        processor=processor,
        device=device,
        masked_label_value=masked_label_value
    )
    
    return imputer.impute(
        df=df,
        batch_size=batch_size,
        verbose=verbose,
        use_masked_labels=use_masked_labels,
        external_labels=external_labels,
        dataset_name=dataset_name
    ) 
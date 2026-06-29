try:
    from dataprep.base import BaseEstimator
except ModuleNotFoundError:
    from base import BaseEstimator
from abc import abstractmethod
import pandas as pd
import numpy as np

class BaseImputer(BaseEstimator):

    @abstractmethod
    def train(self, data: pd.DataFrame, missing_mask: np.ndarray, **kwargs):
        """
        使⽤带有缺失位置信息的数据训练补全模型。
         :param data: 包含缺失值的 pandas DataFrame。
         :param missing_mask: 标记缺失位置的 numpy 数组 (1=missing, 0=not missing)。
         :return: self, 训练好的模型实例。
         """
        raise NotImplementedError

    @abstractmethod
    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
         对输⼊的数据框进⾏数据补全。通常，这⾥的输⼊data和训练时的data是同⼀个。
         :param data: 包含缺失值的 pandas DataFrame。
         :return: 已补全的 pandas DataFrame，且形状与输⼊⼀致。
         """
        raise NotImplementedError

    def train_and_predict(self, data, missing_mask):
        """
        标准流程：先训练模型，然后立即返回填补结果。

        Args:
            data: 包含缺失值的原始数据
            missing_mask: 掩码矩阵 (0表示缺失)

        Returns:
            imputed_data: 填补后的数据
        """
        self.train(data, missing_mask)

        return self.predict(data)

    def estimate(self, ground_truth, imputed_data, missing_mask):
        """
        计算 RMSE 和 MAE 指标。
        只在原始数据缺失（missing_mask == 0）的地方计算误差。

        Args:
            ground_truth: 完整的真实数据 (Ground Truth)，不应包含 NaN
            imputed_data: 算法填补后的数据
            missing_mask: 掩码矩阵，1表示观测到，0表示缺失 (用于定位哪里需要计算误差)

        Returns:
            dict: 包含 'rmse' 和 'mae' 的字典
        """
        ground_truth = np.array(ground_truth)
        imputed_data = np.array(imputed_data)
        missing_mask = np.array(missing_mask)

        # 确保形状一致
        if ground_truth.shape != imputed_data.shape or ground_truth.shape != missing_mask.shape:
            raise ValueError("Input shapes of ground_truth, imputed_data, and missing_mask must match.")

        # 1. 提取缺失位置的索引 (mask == 0 的地方)
        missing_indices = np.where(missing_mask == 0)

        # 如果没有缺失值，直接返回 0
        if len(missing_indices[0]) == 0:
            return {'rmse': 0.0, 'mae': 0.0}

        # 2. 提取对应位置的真实值和预测值
        y_true = ground_truth[missing_indices]
        y_pred = imputed_data[missing_indices]

        # 3. 计算 RMSE
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)

        # 4. 计算 MAE
        mae = np.mean(np.abs(y_true - y_pred))

        print(f"Evaluation Metrics (on missing part): RMSE={rmse:.4f}, MAE={mae:.4f}")
        return {'rmse': rmse, 'mae': mae}


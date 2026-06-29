import numpy as np
import torch

try:
    import dataprep.tabular.imputation.EDIT_modules as em
    from dataprep.tabular.imputation.base import BaseImputer
except ModuleNotFoundError:
    import tabular.imputation.EDIT_modules as em
    from tabular.imputation.base import BaseImputer


class EDIT(BaseImputer):
    """
    EDIT Imputer.

    Reference:
        Miao et al. "Efficient and effective data imputation with influence functions."
        VLDB 2021.

    思路:
        在 GAIN 框架上用影响函数 (Hessian^-1 · grad) 估计每个样本对最终
        填补质量的贡献，挑出 Top-k 个最有用的样本做重训练，比直接训练
        全部数据效果更好且更省时间。
    """

    def __init__(self,
                 batch_size: int = 8,
                 hint_rate: float = 0.9,
                 alpha: float = 10,
                 epoch: int = 10,
                 initial_size: int = 6000,
                 validation_size: int = 6000,
                 damping: float = 1e-2,
                 device: str = None):
        """
        Args:
            batch_size       : Mini-batch 大小 (默认 8)
            hint_rate        : GAIN 的 hint rate
            alpha            : 生成器损失中 MSE 项的权重 (默认 10)
            epoch            : 初始训练 / 重训练阶段各自的 epoch 数 (默认 10)
            initial_size     : 初始训练集大小
            validation_size  : 验证集大小 (用于影响函数)
            damping          : Generator L2 正则项系数 (默认 1e-2)
            device           : 'cuda' or 'cpu'；不传则自动选
        """
        self.batch_size = batch_size
        self.hint_rate = hint_rate
        self.alpha = alpha
        self.epoch = epoch
        self.initial_size = initial_size
        self.validation_size = validation_size
        self.damping = damping
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        # 内部状态
        self.norm_parameters = None
        self.generator = None
        self.discriminator = None

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    def train(self, data: np.ndarray, missing_mask: np.ndarray = None) -> None:
        """
        Args:
            data         : np.ndarray, 原始数据
            missing_mask : np.ndarray, 1=观测, 0=缺失。不传时按 NaN 自动生成。
        """
        if hasattr(self, '_create_temp_dir'):
            self._create_temp_dir(prefix="edit_train_")

        data = np.array(data, dtype=np.float64)
        if missing_mask is None:
            missing_mask = 1. - np.isnan(data)
        missing_mask = np.array(missing_mask, dtype=np.float64)

        no, dim = data.shape
        h_dim = int(dim)

        # 1. 归一化
        data_for_train = data.copy()
        data_for_train[missing_mask == 0] = np.nan
        mask_f32 = missing_mask.astype(np.float32)

        # 2. 初始化网络
        self.generator = em.EditGenerator(dim, h_dim).to(self.device)
        self.discriminator = em.EditDiscriminator(dim, h_dim).to(self.device)

        # 3. 调用 module 中的核心算法
        params = {
            'batch_size': self.batch_size,
            'epoch': self.epoch,
            'hint_rate': self.hint_rate,
            'alpha': self.alpha,
            'damping': self.damping,
            'initial_size': self.initial_size,
            'validation_size': self.validation_size,
        }

        print(f"Starting EDIT training on {self.device}...")
        self.norm_parameters = em.train_edit_algorithm(
            self.generator,
            self.discriminator,
            data_for_train,
            mask_f32,
            params,
            self.device,
        )

        if hasattr(self, '_save_checkpoint'):
            self._save_checkpoint("edit_imputer_complete.pkl")

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict(self, data: np.ndarray, missing_mask: np.ndarray = None) -> np.ndarray:
        """
        Args:
            data : np.ndarray
            missing_mask : np.ndarray, 1=观测, 0=缺失。
        Returns:
            imputed_data : 填补后的完整数据
        """
        if self.generator is None:
            raise RuntimeError("Model needs to be trained first. Call .train() first.")

        self.generator.eval()

        data = np.array(data, dtype=np.float64)

        if missing_mask is None:
            missing_mask = 1. - np.isnan(data)
        else:
            missing_mask = np.array(missing_mask, dtype=np.float64)

        missing_mask = missing_mask.astype(np.float32)
        observed_mask = missing_mask.astype(bool)

        no, dim = data.shape

        # 1. 归一化
        data_for_norm = data.copy()
        data_for_norm[missing_mask == 0] = np.nan

        norm_data = em.normalization_with_parameter(data_for_norm, self.norm_parameters)
        norm_data_x = np.nan_to_num(norm_data, 0).astype(np.float32)

        # 2. 把缺失位置注入噪声 (和训练时一致)
        z_mb = em.sample_Z(no, dim).astype(np.float32)
        x_mb = missing_mask * norm_data_x + (1 - missing_mask) * z_mb

        x_torch = torch.tensor(x_mb, dtype=torch.float32).to(self.device)
        m_torch = torch.tensor(missing_mask, dtype=torch.float32).to(self.device)

        # 3. 用 Generator 出补全值
        with torch.no_grad():
            imputed_norm_prob = self.generator(x_torch, m_torch).cpu().numpy()

        # 4. 观测值保留，缺失位置用生成值
        imputed_data_norm = missing_mask * norm_data_x + (1 - missing_mask) * imputed_norm_prob

        # 5. 反归一化
        imputed_data = em.renormalization(imputed_data_norm, self.norm_parameters)


        # 6. 对类别型变量做 rounding
        data_for_rounding = data.copy()
        data_for_rounding[missing_mask == 0] = np.nan
        imputed_data = em.rounding(imputed_data, data_for_rounding)

        # rounding 只保留在缺失填补位置，观测值严格恢复为原始值
        imputed_data[observed_mask] = data[observed_mask]

        return imputed_data

    def train_and_predict(self, data: np.ndarray, missing_mask: np.ndarray = None) -> np.ndarray:
        self.train(data, missing_mask)
        return self.predict(data, missing_mask=missing_mask)

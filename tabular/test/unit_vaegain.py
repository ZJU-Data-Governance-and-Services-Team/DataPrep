import sys
import os
import unittest
import numpy as np
import torch
from unittest.mock import MagicMock, patch

# ==========================================
# 1. 导入路径设置
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
# 假设结构是 dataprep/tabular/imputation/
project_root = os.path.abspath(os.path.join(current_dir, '../../..'))

if project_root not in sys.path:
    sys.path.append(project_root)

# ==========================================
# 2. 尝试导入模块
# ==========================================
try:
    from dataprep.tabular.imputation.VAEGAIN import VAEGAIN
    import dataprep.tabular.imputation.VAEGAIN_modules as vm
except ImportError as e:
    raise ImportError(f"导入失败，请检查文件位置。\n详细错误: {e}")


# ==========================================
# 3. 测试类定义
# ==========================================

class TestVAEGAINModules(unittest.TestCase):
    """测试 VAEGAIN_modules.py 中的组件"""

    def setUp(self):
        self.input_dim = 4
        self.latent_dim = 2
        self.batch_size = 5
        self.h1, self.h2 = 8, 4

    def test_normalization_renormalization(self):
        """验证数据归一化与反归一化的可逆性"""
        data = np.array([[10, 100], [20, 200], [30, 300]], dtype=float)
        # 1. 归一化
        norm_data, params = vm.normalization(data)
        self.assertTrue((norm_data >= 0).all() and (norm_data <= 1).all())

        # 2. 反归一化
        recon_data = vm.renormalization(norm_data, params)
        np.testing.assert_array_almost_equal(data, recon_data)

    def test_encoder_shape(self):
        """测试编码器输出"""
        net = vm.Encoder(self.input_dim, self.h1, self.h2, self.latent_dim)
        x = torch.randn(self.batch_size, self.input_dim)

        z_mean, z_log_var = net(x)

        # 输出应为 (batch, latent_dim)
        self.assertEqual(z_mean.shape, (self.batch_size, self.latent_dim))
        self.assertEqual(z_log_var.shape, (self.batch_size, self.latent_dim))

    def test_decoder_shape(self):
        """测试解码器输出"""
        net = vm.Decoder(self.latent_dim, self.h1, self.h2, self.input_dim)
        z = torch.randn(self.batch_size, self.latent_dim)

        x_mean, x_log_sigma = net(z)

        # 输出应为 (batch, input_dim)
        self.assertEqual(x_mean.shape, (self.batch_size, self.input_dim))
        self.assertEqual(x_log_sigma.shape, (self.batch_size, self.input_dim))

    def test_discriminator_shape(self):
        """测试判别器输出"""
        net = vm.Discriminator(self.input_dim, self.h1, self.h2)
        x = torch.randn(self.batch_size, self.input_dim)
        h = torch.randn(self.batch_size, self.input_dim)  # hint vector

        prob = net(x, h)

        # 输出应为 (batch, input_dim)，因为是对每个维度判断真假
        self.assertEqual(prob.shape, (self.batch_size, self.input_dim))
        # 输出经过 Sigmoid，应在 [0, 1]
        self.assertTrue((prob >= 0).all() and (prob <= 1).all())

    def test_gaussian_log_likelihood(self):
        """测试高斯对数似然计算"""
        x = torch.zeros(self.batch_size, self.input_dim)
        mean = torch.zeros(self.batch_size, self.input_dim)
        # log_sigma = 0 -> sigma = 1
        log_sigma = torch.zeros(self.batch_size, self.input_dim)

        ll = vm.gaussian_log_likelihood(x, mean, log_sigma)

        # 验证形状
        self.assertEqual(ll.shape, (self.batch_size, self.input_dim))

        # 标准正态分布在 0 处的 log_prob 是 -0.5 * log(2pi)
        expected = -0.5 * np.log(2 * np.pi)

        # 修复点：强制将 expected 转为 float32 以匹配模型输出
        self.assertTrue(torch.allclose(ll, torch.tensor(expected, dtype=torch.float32), atol=1e-4))


class TestVAEGAINMain(unittest.TestCase):
    """测试 VAEGAIN.py 主类逻辑"""

    def setUp(self):
        # 构造含有 NaN 的数据
        self.raw_data = np.array([
            [1.0, 10.0],
            [2.0, np.nan],
            [3.0, 30.0],
            [np.nan, 40.0]
        ])
        self.mask = 1 - np.isnan(self.raw_data)

        self.imputer = VAEGAIN(batch_size=2, epoch=1, device='cpu', latent_size=2)

        # Mock 基类方法
        self.imputer._create_temp_dir = MagicMock()
        self.imputer._save_checkpoint = MagicMock()

    @patch('dataprep.tabular.imputation.VAEGAIN.vm.train_vaegain')
    def test_train_pipeline(self, mock_train_loop):
        """测试 VAEGAIN 训练流程"""

        self.imputer.train(self.raw_data, self.mask)

        # 1. 验证组件初始化
        self.assertIsNotNone(self.imputer.encoder)
        self.assertIsNotNone(self.imputer.decoder)
        self.assertIsNotNone(self.imputer.discriminator)

        # 2. 验证参数传递
        mock_train_loop.assert_called_once()
        args, kwargs = mock_train_loop.call_args

        # 检查传入的 params 字典
        passed_params = args[5]  # params 是第6个参数
        self.assertEqual(passed_params['learning_rate'], 0.002)
        self.assertEqual(passed_params['discriminator_number'], 1)

    def test_predict_pipeline(self):
        """测试 VAEGAIN 预测流程"""
        # 手动初始化网络，模拟“已训练”状态
        dim = 2
        latent = 2
        # 使用随机初始化的网络即可，我们只测试数据流，不关心预测准确度
        self.imputer.encoder = vm.Encoder(dim, 10, 10, latent)
        self.imputer.decoder = vm.Decoder(latent, 10, 10, dim)

        # 设置归一化参数
        self.imputer.norm_parameters = {
            'min': np.nanmin(self.raw_data, axis=0),
            'max': np.nanmax(self.raw_data, axis=0),
            'den': np.nanmax(self.raw_data, axis=0) - np.nanmin(self.raw_data, axis=0)
        }

        # 运行预测
        imputed_data = self.imputer.predict(self.raw_data)

        # 1. 检查形状
        self.assertEqual(imputed_data.shape, self.raw_data.shape)

        # 2. 检查 NaN 是否被移除
        self.assertFalse(np.isnan(imputed_data).any())

        # 3. 检查观测值一致性
        # VAEGAIN 同样采用 M * X + (1-M) * G 的策略
        # 因此观测值必须被保留 (允许浮点误差)
        mask_bool = self.mask.astype(bool)
        np.testing.assert_array_almost_equal(
            imputed_data[mask_bool],
            self.raw_data[mask_bool],
            decimal=5
        )

    def test_predict_without_train(self):
        """验证未训练时报错"""
        # 确保网络为 None
        self.imputer.encoder = None
        with self.assertRaises(RuntimeError):
            self.imputer.predict(self.raw_data)


if __name__ == '__main__':
    unittest.main()
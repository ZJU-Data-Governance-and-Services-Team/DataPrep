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
project_root = os.path.abspath(os.path.join(current_dir, '../../..'))

if project_root not in sys.path:
    sys.path.append(project_root)

# ==========================================
# 2. 尝试导入模块
# ==========================================
try:
    from dataprep.tabular.imputation.GAIN import GAIN
    import dataprep.tabular.imputation.GAIN_modules as gm
except ImportError as e:
    raise ImportError(f"导入失败，请检查文件位置。\n详细错误: {e}")


# ==========================================
# 3. 测试类定义
# ==========================================

class TestGAINModules(unittest.TestCase):
    """测试 GAIN_modules.py 中的底层函数和网络结构"""

    def setUp(self):
        self.data = np.array([
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0]
        ])
        self.dim = 2
        self.h_dim = 4

    def test_normalization_renormalization(self):
        """测试数据的归一化和反归一化过程"""
        # 1. Normalization
        norm_data, params = gm.normalization(self.data)

        # 检查范围是否在 [0, 1]
        self.assertTrue((norm_data >= 0).all() and (norm_data <= 1).all())
        self.assertTrue(np.isclose(np.min(norm_data), 0))
        self.assertTrue(np.isclose(np.max(norm_data), 1))

        # 2. Renormalization
        renorm_data = gm.renormalization(norm_data, params)

        # 检查是否还原回原始数据
        np.testing.assert_array_almost_equal(self.data, renorm_data)

    def test_generator_shape(self):
        """测试生成器网络的输入输出形状"""
        net = gm.GainGenerator(self.dim, self.h_dim)
        batch_size = 5

        # 输入: data + mask (dim * 2)
        x = torch.randn(batch_size, self.dim)
        m = torch.randn(batch_size, self.dim)

        output = net(x, m)

        # 输出形状应为 (batch_size, dim)
        self.assertEqual(output.shape, (batch_size, self.dim))
        # 输出经过 Sigmoid，应在 [0, 1] 之间
        self.assertTrue((output >= 0).all() and (output <= 1).all())

    def test_discriminator_shape(self):
        """测试判别器网络的输入输出形状"""
        net = gm.GainDiscriminator(self.dim, self.h_dim)
        batch_size = 5

        # 输入: reconstructed_data + hint (dim * 2)
        x = torch.randn(batch_size, self.dim)
        h = torch.randn(batch_size, self.dim)

        output = net(x, h)

        # 输出形状应为 (batch_size, dim)
        self.assertEqual(output.shape, (batch_size, self.dim))


class TestGAINMain(unittest.TestCase):
    """测试 GAIN.py 中的主类逻辑"""

    def setUp(self):
        # 构造含有缺失值的假数据
        self.raw_data = np.array([
            [1.0, 10.0],
            [2.0, np.nan],
            [3.0, 30.0],
            [np.nan, 40.0]
        ])
        self.mask = 1 - np.isnan(self.raw_data)  # 1 for observed, 0 for missing

        # 实例化 GAIN
        self.imputer = GAIN(batch_size=2, epoch=1, device='cpu')

        # Mock 掉 BaseImputer 的方法，防止因缺少父类实现或文件系统操作报错
        self.imputer._create_temp_dir = MagicMock()
        self.imputer._save_checkpoint = MagicMock()

    @patch('dataprep.tabular.imputation.GAIN.gm.train_gain_model')
    def test_train_pipeline(self, mock_train_loop):
        """测试训练流程 (Mock 掉实际的训练循环)"""

        # 运行训练
        self.imputer.train(self.raw_data, self.mask)

        # 1. 验证模型是否被初始化
        self.assertIsNotNone(self.imputer.generator)
        self.assertIsNotNone(self.imputer.discriminator)
        self.assertIsNotNone(self.imputer.norm_parameters)

        # 2. 验证是否调用了底层训练循环
        mock_train_loop.assert_called_once()

        # 3. 验证是否调用了保存 checkpoint
        self.imputer._save_checkpoint.assert_called_once()

    def test_predict_without_train(self):
        """测试未训练直接预测是否抛出异常"""
        with self.assertRaises(RuntimeError):
            self.imputer.predict(self.raw_data)

    def test_predict_pipeline(self):
        """测试预测流程"""
        # 1. 手动设置已训练状态 (绕过 train 方法)
        dim = 2
        h_dim = 2
        # 初始化真实的网络结构用于推理测试
        self.imputer.generator = gm.GainGenerator(dim, h_dim)
        self.imputer.norm_parameters = {
            'min': np.array([1.0, 10.0]),
            'max': np.array([4.0, 40.0]),
            'den': np.array([3.0, 30.0])
        }

        # 2. 运行预测
        imputed_data = self.imputer.predict(self.raw_data)

        # 3. 验证输出
        # 形状必须相同
        self.assertEqual(imputed_data.shape, self.raw_data.shape)

        # 结果中不应包含 NaN
        self.assertFalse(np.isnan(imputed_data).any())

        # 观测值应该保持不变 (或者非常接近，取决于反归一化精度)
        # 第0行全是观测值
        np.testing.assert_array_almost_equal(imputed_data[0], self.raw_data[0])

        # 第1行第1列是缺失值，应该被填补
        self.assertNotEqual(imputed_data[1, 1], np.nan)


if __name__ == '__main__':
    unittest.main()
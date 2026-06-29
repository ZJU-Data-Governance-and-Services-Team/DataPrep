import sys
import os
import unittest
import numpy as np
import torch
import torch.nn as nn
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
    from dataprep.tabular.imputation.SCIS import SCIS
    import dataprep.tabular.imputation.SCIS_modules as sm
except ImportError as e:
    raise ImportError(f"导入失败，请检查文件位置。\n详细错误: {e}")


# ==========================================
# 3. 测试类定义
# ==========================================

class TestSCISModules(unittest.TestCase):
    """测试 SCIS_modules.py 中的底层数学函数和网络特性"""

    def setUp(self):
        self.dim = 3
        self.h_dim = 5
        self.batch_size = 4
        self.device = 'cpu'

    def test_sinkhorn_loss(self):
        """测试 Sinkhorn Loss 计算是否正常"""
        x = torch.randn(self.batch_size, self.dim)
        y = torch.randn(self.batch_size, self.dim)

        # 运行 Sinkhorn
        loss = sm.sinkhorn_loss_torch(x, y, epsilon=0.1, niter=5)

        # 验证返回的是否是标量 tensor
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.numel(), 1)
        self.assertFalse(torch.isnan(loss))

    def test_generator_forward_with_params(self):
        """核心测试: 验证 Generator 是否支持手动参数注入 (用于 Hessian 计算)"""
        net = sm.GainGenerator(self.dim, self.h_dim)
        x = torch.randn(self.batch_size, self.dim)
        m = torch.randn(self.batch_size, self.dim)

        # 1. 正常 Forward
        out_normal = net(x, m)

        # 2. 提取参数
        params = [p for p in net.parameters()]

        # 3. 手动参数 Forward
        out_manual = net.forward_with_params(x, m, params)

        # 4. 验证两者结果是否一致
        # 由于浮点数精度，使用 allclose
        self.assertTrue(torch.allclose(out_normal, out_manual, atol=1e-6))

    def test_compute_hessian_diag(self):
        """测试 Hessian 对角近似计算流程"""
        # 构建一个简单的可导过程
        model = nn.Linear(2, 1)
        inputs = torch.randn(4, 2)
        target = torch.randn(4, 1)

        # Forward pass
        output = model(inputs)
        loss = torch.mean((output - target) ** 2)

        # 运行 Hessian 计算
        try:
            diags = sm.compute_hessian_diag(loss, list(model.parameters()), self.device)
            self.assertEqual(len(diags), len(list(model.parameters())))
            # 验证输出形状与参数形状一致
            for diag, param in zip(diags, model.parameters()):
                self.assertEqual(diag.shape, param.shape)
        except Exception as e:
            self.fail(f"Hessian 计算抛出异常: {e}")


class TestSCISMain(unittest.TestCase):
    """测试 SCIS.py 中的主类逻辑"""

    def setUp(self):
        # 构造数据
        self.raw_data = np.array([
            [0.1, 0.5, 0.9],
            [0.2, np.nan, 0.8],
            [np.nan, 0.4, 0.7],
            [0.4, 0.6, 0.6]
        ])
        self.mask = 1 - np.isnan(self.raw_data)

        # 实例化 SCIS
        self.imputer = SCIS(batch_size=2, epoch=1, device='cpu')

        # Mock 掉 BaseImputer 的文件操作
        self.imputer._create_temp_dir = MagicMock()
        self.imputer._save_checkpoint = MagicMock()

    @patch('dataprep.tabular.imputation.SCIS.sm.train_scis_algorithm')
    def test_train_pipeline(self, mock_train_algo):
        """测试 SCIS 训练接口调用"""

        self.imputer.train(self.raw_data, self.mask)

        # 1. 验证网络初始化
        self.assertIsNotNone(self.imputer.generator)
        self.assertIsNotNone(self.imputer.discriminator)

        # 2. 验证参数传递
        mock_train_algo.assert_called_once()
        args, kwargs = mock_train_algo.call_args

        # 验证传入的 params 字典包含了 SCIS 特有的参数
        passed_params = args[4]
        self.assertIn('value', passed_params)  # Sinkhorn 权重
        self.assertIn('epsilon', passed_params)  # Sinkhorn 参数
        self.assertIn('guarantee', passed_params)  # Hessian 搜索参数

    def test_predict_pipeline(self):
        """测试 SCIS 预测接口"""
        # 手动注入“已训练”的模型
        dim = 3
        h_dim = 3
        self.imputer.generator = sm.GainGenerator(dim, h_dim)
        # 手动注入归一化参数
        self.imputer.norm_parameters = {
            'min': np.array([0.1, 0.4, 0.6]),
            'max': np.array([0.4, 0.6, 0.9]),
            'den': np.array([0.3, 0.2, 0.3])
        }

        # 运行预测
        imputed_data = self.imputer.predict(self.raw_data)

        # 验证
        self.assertEqual(imputed_data.shape, self.raw_data.shape)
        self.assertFalse(np.isnan(imputed_data).any())

        # 验证观测值未变 (第0行全是观测值)
        # 注意：由于 SCIS 同样使用了 M*X + (1-M)*G 的逻辑，观测值应被保留
        np.testing.assert_array_almost_equal(
            imputed_data[0], self.raw_data[0], decimal=5
        )

    def test_predict_not_trained(self):
        """验证未训练报错"""
        with self.assertRaises(RuntimeError):
            self.imputer.predict(self.raw_data)


if __name__ == '__main__':
    unittest.main()
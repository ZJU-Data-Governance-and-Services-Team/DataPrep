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

try:
    from dataprep.tabular.imputation.EDIT import EDIT
    import dataprep.tabular.imputation.EDIT_modules as em
except ImportError as e:
    raise ImportError(f"导入失败，请检查文件位置。\n详细错误: {e}")


# ==========================================
# 2. 测试 EDIT_modules
# ==========================================

class TestEDITModules(unittest.TestCase):
    """测试 EDIT_modules.py 中的底层函数和网络结构"""

    def setUp(self):
        self.data = np.array([
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
        ])
        self.dim = 2
        self.h_dim = 4

    def test_normalization_renormalization(self):
        """归一化和反归一化应该可逆"""
        norm_data, params = em.normalization(self.data)
        self.assertTrue((norm_data >= 0).all() and (norm_data <= 1).all())
        self.assertTrue(np.isclose(np.min(norm_data), 0))
        self.assertTrue(np.isclose(np.max(norm_data), 1))
        renorm_data = em.renormalization(norm_data, params)
        np.testing.assert_array_almost_equal(self.data, renorm_data)

    def test_normalization_with_parameter(self):
        """测试使用已有参数进行归一化"""
        norm_data, params = em.normalization(self.data)

        new_data = np.array([
            [1.0, 10.0],
            [2.5, 25.0],
            [4.0, 40.0],
        ])

        norm_new = em.normalization_with_parameter(new_data, params)

        expected = np.array([
            [0.0, 0.0],
            [0.5, 0.5],
            [1.0, 1.0],
        ])

        np.testing.assert_array_almost_equal(norm_new, expected)

    def test_generator_shape(self):
        """生成器输入输出形状"""
        net = em.EditGenerator(self.dim, self.h_dim)
        bs = 5
        x = torch.randn(bs, self.dim)
        m = torch.randn(bs, self.dim)
        out = net(x, m)
        self.assertEqual(out.shape, (bs, self.dim))
        # Sigmoid 输出在 [0, 1]
        self.assertTrue((out >= 0).all() and (out <= 1).all())

    def test_discriminator_shape(self):
        """判别器输入输出形状"""
        net = em.EditDiscriminator(self.dim, self.h_dim)
        bs = 5
        x = torch.randn(bs, self.dim)
        h = torch.randn(bs, self.dim)
        out = net(x, h)
        self.assertEqual(out.shape, (bs, self.dim))

    def test_select_top_k_by_influence(self):
        """累积选 Top-k 的工具函数"""
        scores = np.array([5.0, 1.0, 4.0, -2.0, 3.0], dtype=np.float32)
        # sum = 11.0, 降序: idx [0, 2, 4, 1, 3] -> 5, 4, 3, 1, -2
        # 累加: 5(>11? no), 9(>11? no), 12(>11? yes, break)
        # 至少应该选了 [0, 2, 4]
        top_k = em.select_top_k_by_influence(scores)
        self.assertIn(0, top_k)
        self.assertIn(2, top_k)
        # 兜底逻辑保证至少 10%
        self.assertGreaterEqual(len(top_k), 1)

    def test_sample_Z(self):
        #测试随机噪声 Z 的形状和取值范围
        z = em.sample_Z(batch_size=5, dim=3)

        self.assertEqual(z.shape, (5, 3))
        self.assertTrue((z >= 0).all())
        self.assertTrue((z <= 0.01).all())

    def test_sample_M(self):
        #测试 mask M 的形状和二值性
        m = em.sample_M(batch_size=5, dim=3, p=0.5)

        self.assertEqual(m.shape, (5, 3))
        self.assertTrue(np.isin(m, [0, 1]).all())

    def test_sample_M_random_and_fixed(self):
        """训练阶段 sample_M 应随机；influence 阶段 fixed sampler 应可复现"""
        a = em.sample_M(batch_size=50, dim=20, p=0.5)
        b = em.sample_M(batch_size=50, dim=20, p=0.5)

        # 极小概率会相等，但 50*20 规模下基本可忽略
        self.assertFalse(np.array_equal(a, b))

        fa = em.sample_M_fixed(batch_size=10, dim=4, p=0.5)
        fb = em.sample_M_fixed(batch_size=10, dim=4, p=0.5)
        np.testing.assert_array_equal(fa, fb)

    def test_rounding(self):
        """类别型列应被四舍五入"""
        data_x = np.array([
            [0.0, 10.0],
            [1.0, 20.0],
            [0.0, np.nan],
            [1.0, 40.0],
        ])

        imputed = np.array([
            [0.2, 10.4],
            [0.8, 20.5],
            [0.6, 30.6],
            [1.2, 40.1],
        ])

        rounded = em.rounding(imputed, data_x)

        np.testing.assert_array_equal(rounded[:, 0], np.round(imputed[:, 0]))

# ==========================================
# 3. 测试 EDIT 主类
# ==========================================

class TestEDITMain(unittest.TestCase):
    """测试 EDIT.py 中的主类行为"""

    def setUp(self):
        np.random.seed(0)
        torch.manual_seed(0)

        # 构造一份带缺失的小数据
        self.raw_data = np.array([
            [1.0, 10.0],
            [2.0, np.nan],
            [3.0, 30.0],
            [np.nan, 40.0],
            [5.0, 50.0],
            [6.0, 60.0],
        ])
        self.mask = 1 - np.isnan(self.raw_data).astype(float)

        self.imputer = EDIT(batch_size=2, epoch=1,initial_size=3, validation_size=2,device='cpu')

        # Mock 掉文件系统相关方法
        self.imputer._create_temp_dir = MagicMock()
        self.imputer._save_checkpoint = MagicMock()

    @patch('dataprep.tabular.imputation.EDIT.em.train_edit_algorithm')
    def test_train_pipeline(self, mock_train_algo):
        """训练流程: mock 掉底层训练循环, 只验证主类逻辑"""
        self.imputer.train(self.raw_data, self.mask)

        # 1. 模型和归一化参数应被初始化
        self.assertIsNotNone(self.imputer.generator)
        self.assertIsNotNone(self.imputer.discriminator)
        self.assertIsNotNone(self.imputer.norm_parameters)

        # 2. 底层算法应被调用一次
        mock_train_algo.assert_called_once()

        # 3. checkpoint 应被保存
        self.imputer._save_checkpoint.assert_called_once()

    def test_predict_without_train(self):
        """未训练直接 predict 必须报错"""
        with self.assertRaises(RuntimeError):
            self.imputer.predict(self.raw_data)

    def test_predict_pipeline(self):
        """predict 形状/取值检查"""
        dim = 2
        h_dim = 2
        self.imputer.generator = em.EditGenerator(dim, h_dim)
        self.imputer.norm_parameters = {
            'min': np.array([1.0, 10.0]),
            'max': np.array([6.0, 60.0]),
            'den': np.array([5.0, 50.0]),
        }

        imputed = self.imputer.predict(self.raw_data)

        # 形状一致
        self.assertEqual(imputed.shape, self.raw_data.shape)
        # 输出无 NaN
        self.assertFalse(np.isnan(imputed).any())
        # 所有观测位置都应保持不变
        observed = self.mask.astype(bool)
        np.testing.assert_array_almost_equal(imputed[observed],self.raw_data[observed],decimal=4)

    def test_predict_uses_missing_mask_not_nan_only(self):
        """predict 应优先使用传入的 missing_mask，而不是只依赖 np.isnan(data)。"""

        class ConstantGenerator(torch.nn.Module):
            def forward(self, x, m):
                return torch.zeros_like(x)

        data = np.array([
            [1.0, 999.0],
            [2.0, 20.0],
        ])
        mask = np.array([
            [1.0, 0.0],
            [1.0, 1.0],
        ])

        self.imputer.generator = ConstantGenerator()
        self.imputer.norm_parameters = {
            'min': np.array([1.0, 10.0]),
            'max': np.array([2.0, 20.0]),
            'den': np.array([1.0, 10.0]),
        }

        imputed = self.imputer.predict(data, missing_mask=mask)

        observed = mask.astype(bool)
        np.testing.assert_array_almost_equal(imputed[observed], data[observed])
        self.assertNotEqual(imputed[0, 1], 999.0)

    @patch.object(EDIT, 'train')
    @patch.object(EDIT, 'predict')
    def test_train_and_predict_passes_missing_mask(self, mock_predict, mock_train):
        """train_and_predict 应把外部 missing_mask 继续传给 predict。"""
        data = np.array([[1.0, 999.0], [2.0, 20.0]])
        mask = np.array([[1.0, 0.0], [1.0, 1.0]])
        expected = np.array([[1.0, 10.0], [2.0, 20.0]])
        mock_predict.return_value = expected

        imputer = EDIT(device='cpu')
        result = imputer.train_and_predict(data, missing_mask=mask)

        self.assertIs(result, expected)
        self.assertIs(mock_train.call_args.args[0], data)
        self.assertIs(mock_train.call_args.args[1], mask)
        self.assertIs(mock_predict.call_args.args[0], data)
        self.assertIs(mock_predict.call_args.kwargs['missing_mask'], mask)

if __name__ == '__main__':
    unittest.main()

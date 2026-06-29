import sys
import os
import unittest
import pandas as pd
import numpy as np
import json
from unittest.mock import MagicMock, patch, mock_open

# ==========================================
# 1. 导入路径设置
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
# 假设结构是 dataprep/tabular/detection/
project_root = os.path.abspath(os.path.join(current_dir, '../../..'))

if project_root not in sys.path:
    sys.path.append(project_root)

# ==========================================
# 2. 尝试导入模块
# ==========================================
try:
    from dataprep.tabular.detection.ZeroED import ZeroED
    import dataprep.tabular.detection.ZeroED_modules as mo
except ImportError as e:
    raise ImportError(f"导入失败，请检查文件位置。\n详细错误: {e}")


# ==========================================
# 3. 测试类定义
# ==========================================

class TestZeroEDModules(unittest.TestCase):
    """测试 ZeroED_modules.py 中的底层工具函数"""

    def test_extract_single_feature_vector(self):
        """测试特征提取逻辑 (9维特征)"""
        # 测试数字字符串
        val = "123.45"
        feat = mo.extract_single_feature_vector(val)
        self.assertEqual(len(feat), 9)
        self.assertEqual(feat[0], 123.45)  # Numerical value
        self.assertEqual(feat[8], 0.0)  # Is Missing

        # 测试文本
        val = "abc"
        feat = mo.extract_single_feature_vector(val)
        self.assertEqual(feat[0], 0.0)  # 非数字转0
        self.assertEqual(feat[3], 3)  # 3个字母

        # 测试缺失值
        val = np.nan
        feat = mo.extract_single_feature_vector(val)
        self.assertEqual(feat[8], 1.0)  # Is Missing = 1

    def test_batch_extract_features(self):
        """测试批量特征提取"""
        s = pd.Series(["1", "a", np.nan])
        feats = mo.batch_extract_features(s)
        self.assertEqual(feats.shape, (3, 9))

    def test_extract_json_from_text(self):
        """测试从 LLM 返回文本中提取 JSON"""
        text_1 = 'Here is the json: ```json\n{"key": "value"}\n```'
        self.assertEqual(mo.extract_json_from_text(text_1), {"key": "value"})

        text_2 = '{"key": "value"}'
        self.assertEqual(mo.extract_json_from_text(text_2), {"key": "value"})

    def test_extract_python_code(self):
        """测试从 LLM 返回文本中提取 Python 代码"""
        text = "Here is code: ```python\ndef foo(): pass\n```"
        codes = mo.extract_python_code(text)
        self.assertEqual(len(codes), 1)
        self.assertIn("def foo():", codes[0])

    def test_execute_cleaning_func(self):
        """测试生成的 Python 代码能否被正确执行"""
        # 模拟一个判断函数：长度大于3为 Clean
        code = """
def is_clean_len(row, attr):
    val = str(row[attr])
    return len(val) > 3
"""
        row_clean = pd.Series({'col': 'abcd'})
        row_dirty = pd.Series({'col': 'ab'})

        # 测试 Clean
        res_clean = mo._execute_cleaning_func(code, row_clean, 'col')
        self.assertTrue(res_clean)

        # 测试 Dirty
        res_dirty = mo._execute_cleaning_func(code, row_dirty, 'col')
        self.assertFalse(res_dirty)

        # 测试代码报错情况 (应默认返回 True)
        bad_code = "def is_clean_err(row, attr): raise ValueError"
        self.assertTrue(mo._execute_cleaning_func(bad_code, row_dirty, 'col'))


class TestZeroEDFlow(unittest.TestCase):
    """测试 ZeroED 主类及 Pipeline 流程"""

    def setUp(self):
        self.df = pd.DataFrame({
            'age': ['25', '30', '150', 'nan', '20'],
            'name': ['Alice', 'Bob', 'C', 'Dave', 'Eve']
        })

        # 禁用真实 API 调用
        self.detector = ZeroED(
            api_use=False,
            local_model_use=True,
            result_dir='./temp/test_zeroed_results',
            verbose=False
        )
        # Mock 基类方法
        self.detector._create_temp_dir = MagicMock()
        self.detector._save_checkpoint = MagicMock()

        # Mock Logger 防止写文件
        self.detector.logger = MagicMock()

    @patch('dataprep.tabular.detection.ZeroED_modules.get_ans_from_llm')
    def test_train_pipeline_mocked_llm(self, mock_llm):
        """
        测试 train() 流程。
        我们 Mock LLM 的返回值，让 Pipeline 跑通。
        """

        # 1. 模拟 LLM 返回
        # 第一次调用 (Phase 4 Guide): 返回随意文本
        # 第二次调用 (Phase 5 Labeling): 返回 json 格式
        # 第三次调用 (Phase 6 ErrGen): 返回 json 数组
        # 第四次调用 (Phase 7 FuncGen): 返回 python 代码

        def side_effect(*args, **kwargs):
            prompt = args[0]
            if "distribution analysis" in prompt.lower():
                return "Analysis: Data looks ok."
            elif "entries" in prompt and "has_error" in prompt:
                # Labeling Prompt: 让包含 '150' 的被标记为 error
                if '150' in prompt:
                    return '```json\n{"has_error": true}\n```'
                return '```json\n{"has_error": false}\n```'
            elif "generate realistic errors" in prompt:
                # Error Generation Prompt
                return '```json\n[["age", "200", "Reason: Too high", {}]]\n```'
            elif "create precise judge functions" in prompt:
                # Function Generation Prompt
                return '```python\ndef is_clean_range(row, attr):\n    try:\n        return float(row[attr]) < 100\n    except:\n        return True\n```'
            return ""

        mock_llm.side_effect = side_effect

        # 2. 运行训练
        self.detector.train(self.df)

        # 3. 验证状态
        self.assertTrue(self.detector.is_trained_)
        # 验证是否对 'age' 列生成了模型或函数
        # 由于我们 Mock 了 labeling，'150' 被标为 dirty，应该能训练出模型或保留函数
        has_model = 'age' in self.detector.local_models
        has_func = 'age' in self.detector.generated_funcs
        self.assertTrue(has_model or has_func, "Should generate model or functions for 'age'")

    def test_predict_pipeline(self):
        """测试 predict() 流程"""
        # 1. 手动注入已训练状态
        self.detector.is_trained_ = True

        # 注入一个 Python 规则函数到 'age' 列
        # 规则: 长度 > 2 视为 Dirty (例如 '150', 'nan')
        func_code = """
def is_clean_len(row, attr):
    return len(str(row[attr])) <= 2
"""
        self.detector.generated_funcs = {'age': [func_code]}

        # 注入一个 Mock 的 sklearn 模型到 'name' 列
        mock_model = MagicMock()
        # 假设 predict 返回 [0, 0, 1, 0, 0] (1=Dirty)
        mock_model.predict.return_value = np.array([0, 0, 1, 0, 0])
        self.detector.local_models = {'name': mock_model}

        # 2. 运行预测
        result = self.detector.predict(self.df)

        # 3. 验证结果
        # 'age' 列: '150' (len 3) -> Dirty(True), 'nan' (len 3) -> Dirty(True), others -> Clean(False)
        self.assertTrue(result.loc[2, 'age'])  # 150
        self.assertTrue(result.loc[3, 'age'])  # nan
        self.assertFalse(result.loc[0, 'age'])  # 25

        # 'name' 列: Mock 模型返回第2个索引为 Dirty
        self.assertTrue(result.loc[2, 'name'])

        # 验证是否调用了本地模型提取特征
        mock_model.predict.assert_called()

    def test_predict_not_trained(self):
        """测试未训练报错"""
        self.detector.is_trained_ = False
        with self.assertRaises(RuntimeError):
            self.detector.predict(self.df)

    def test_phase_1_correlation(self):
        """独立测试 Phase 1 相关性计算"""
        params = {
            'related_attrs': True,
            'rel_top': 1
        }
        logger = MagicMock()

        # 构建完全相关的两列
        df_corr = pd.DataFrame({
            'A': ['1', '2', '3', '4'],
            'B': ['1', '2', '3', '4'],  # B 完全由 A 决定
            'C': ['x', 'x', 'y', 'y']
        })

        res = mo.run_phase_1_correlation(df_corr, params, logger)

        # A 和 B 的相关性应该很高
        self.assertIn('B', res['A'])
        self.assertIn('A', res['B'])


if __name__ == '__main__':
    unittest.main()
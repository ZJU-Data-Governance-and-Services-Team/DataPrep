import sys
import os
import unittest
import json
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch, mock_open

# ==========================================
# 1. 导入路径设置
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../..'))

if project_root not in sys.path:
    sys.path.append(project_root)

# ==========================================
# 2. 导入被测模块
# ==========================================
try:
    from dataprep.tabular.correction.ZeroEC import ZeroEC
    import dataprep.tabular.correction.ZeroEC_modules as modules
except ImportError as e:
    raise ImportError(f"导入失败，请检查文件是否在 dataprep/tabular/correction/ 目录下。\n详细错误: {e}")


# ==========================================
# 3. 测试类定义
# ==========================================

class TestZeroECModules(unittest.TestCase):
    """测试 ZeroEC_modules 中的独立工具函数"""

    def test_get_folder_name(self):
        """测试文件夹创建逻辑"""
        base_path = 'test_runs'
        # 修复点 1: 增加 os.path.isdir 的 Mock
        with patch('os.path.exists') as mock_exists, \
                patch('os.makedirs') as mock_makedirs, \
                patch('os.listdir') as mock_listdir, \
                patch('os.path.isdir') as mock_isdir:
            # 情况1: 基础文件夹不存在
            mock_exists.return_value = False
            # 即使 isdir 为 True，因为 exists 为 False (或 listdir 为空)，逻辑应能处理
            path = modules.get_folder_name(base_path)
            self.assertTrue(path.endswith('run-1'))

            # 情况2: 存在 run-1, run-2
            mock_exists.return_value = True
            mock_listdir.return_value = ['run-1', 'run-2', 'misc_file']
            # 关键修复: 告诉代码这些“假”文件夹确实是文件夹
            mock_isdir.return_value = True

            path = modules.get_folder_name(base_path)
            self.assertTrue(path.endswith('run-3'))

    def test_clean_json_string(self):
        """测试 JSON 字符串清理"""
        raw_str = "```json\n{\"key\": \"value\"}\n```"
        cleaned = modules._clean_json_string(raw_str)
        self.assertEqual(cleaned, "{\"key\": \"value\"}")
        self.assertIsNone(modules._clean_json_string(None))

    def test_format_row(self):
        """测试行数据格式化为 JSON 字符串"""
        header = ['Name', 'Age']
        row = ['Alice', '30']
        result = modules.format_row(row, header)
        result_dict = json.loads(result)
        self.assertEqual(result_dict, {'Name': 'Alice', 'Age': '30'})

    def test_train_val_split(self):
        """测试数据集切分"""
        data = {i: f"item_{i}" for i in range(10)}
        train, val = modules.train_val_split(data)
        self.assertEqual(len(train), 5)
        self.assertEqual(len(val), 5)
        self.assertTrue(set(train.keys()).isdisjoint(set(val.keys())))


class TestZeroECMain(unittest.TestCase):
    """测试 ZeroEC 主类的流程逻辑"""

    def setUp(self):
        """每个测试前准备基础 Mock 数据"""
        self.mock_df = pd.DataFrame({'col1': ['a', 'b'], 'col2': ['1', '2']})

        self.mock_params = {
            'dirty_data': self.mock_df,
            'clean_data': self.mock_df,
            'detection': pd.DataFrame({'col1': [0, 0], 'col2': [0, 1]}),
            'prompt_dir': 'dummy_prompts',
            'output_path': 'dummy_runs/run-1',
            'human_repair_num': 10,
            'max_workers': 1,
            'corrections': self.mock_df.copy(),
            'logs': [],
            'row_count': 2,
            'column_count': 2,
            # 添加缺失的 params 键，防止其他地方报错
            'retriever_dict': {},
            'indices_dict': {},
            'CoE_dict': {},
            'codes': {},
            'fds': {},
            'train_data': {},
            'val_data': {}
        }

    # 修复点 2: 增加 JsonOutputParser 的 Mock，绕过 Pydantic 验证
    @patch('dataprep.tabular.correction.ZeroEC.JsonOutputParser')
    @patch('dataprep.tabular.correction.ZeroEC.modules')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.join', return_value='dummy/path')
    def test_initialization(self, mock_path, mock_file, mock_modules, mock_parser_cls):
        """测试 ZeroEC 的初始化 (__init__)"""
        mock_modules.get_folder_name.return_value = 'dummy_runs/run-1'
        mock_modules.load_all_prompts.return_value = {'sys': 'prompt'}
        mock_modules.load_datasets.return_value = self.mock_params
        mock_modules.init_all_models.return_value = {'llm': MagicMock()}

        zeroec = ZeroEC(output_dir='test_output')

        mock_modules.get_folder_name.assert_called()
        mock_modules.load_all_prompts.assert_called()
        mock_modules.load_datasets.assert_called()
        # 验证是否创建了解析器
        mock_parser_cls.assert_called()

        self.assertIn('llm', zeroec.params)

    # 修复点 2: 同样 Mock JsonOutputParser
    @patch('dataprep.tabular.correction.ZeroEC.JsonOutputParser')
    @patch('dataprep.tabular.correction.ZeroEC.modules')
    @patch('builtins.open', new_callable=mock_open)
    def test_train_pipeline(self, mock_file, mock_modules, mock_parser_cls):
        """测试 train() 流程是否按顺序调用模块"""
        mock_modules.get_folder_name.return_value = 'run-1'
        mock_modules.load_datasets.return_value = self.mock_params
        mock_modules.init_all_models.return_value = {}

        zeroec = ZeroEC()
        # 手动注入 params，覆盖初始化中的内容
        zeroec.params.update(self.mock_params)

        zeroec.train()

        mock_modules.run_embedding_and_selection.assert_called_once_with(zeroec.params)
        mock_modules.simulate_human_repair.assert_called_once_with(zeroec.params)
        mock_modules.run_auto_cot_generation.assert_called_once_with(zeroec.params)
        mock_modules.run_code_fd_generation.assert_called_once_with(zeroec.params)

    # 修复点 2: 同样 Mock JsonOutputParser
    @patch('dataprep.tabular.correction.ZeroEC.JsonOutputParser')
    @patch('dataprep.tabular.correction.ZeroEC.modules')
    @patch('builtins.open', new_callable=mock_open)
    def test_predict_pipeline(self, mock_file, mock_modules, mock_parser_cls):
        """测试 predict() 流程是否按顺序调用模块"""
        mock_modules.get_folder_name.return_value = 'run-1'
        mock_modules.load_datasets.return_value = self.mock_params

        zeroec = ZeroEC()
        zeroec.start_time = 0
        zeroec.params.update(self.mock_params)

        zeroec.predict()

        mock_modules.run_code_fd_execution.assert_called_once_with(zeroec.params)
        mock_modules.run_retriever_update.assert_called_once_with(zeroec.params)
        mock_modules.run_retrieval.assert_called_once_with(zeroec.params)
        mock_modules.run_llm_repair.assert_called_once_with(zeroec.params)
        mock_modules.run_evaluation.assert_called_once()


if __name__ == '__main__':
    unittest.main()
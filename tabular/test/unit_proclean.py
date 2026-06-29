import os
import sys
import tempfile
import unittest
from types import ModuleType
from unittest.mock import MagicMock, patch

import pandas as pd


# ==========================================
# 1. 导入路径与可选依赖 Mock
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))

if project_root not in sys.path:
    sys.path.append(project_root)


def _optional_dependency_modules():
    """Patch heavy optional dependencies so unit tests can import proclean modules quickly."""
    torch_module = ModuleType('torch')
    torch_module.float16 = 'float16'
    torch_module.cuda = MagicMock()
    torch_module.cuda.is_available = MagicMock(return_value=False)

    transformers_module = ModuleType('transformers')
    transformers_module.AutoModelForCausalLM = MagicMock()
    transformers_module.AutoTokenizer = MagicMock()

    fasttext_module = ModuleType('fasttext')
    fasttext_module.load_model = MagicMock()

    class FastTextAPI:
        @staticmethod
        def eprint(message):
            return None

    fasttext_module.FastText = FastTextAPI

    sklearn_module = ModuleType('sklearn')
    sklearn_metrics_module = ModuleType('sklearn.metrics')
    sklearn_ensemble_module = ModuleType('sklearn.ensemble')
    sklearn_preprocessing_module = ModuleType('sklearn.preprocessing')
    sklearn_decomposition_module = ModuleType('sklearn.decomposition')
    sklearn_cluster_module = ModuleType('sklearn.cluster')

    sklearn_metrics_module.mutual_info_score = lambda x, y: 1.0
    sklearn_ensemble_module.IsolationForest = MagicMock()
    sklearn_preprocessing_module.MinMaxScaler = MagicMock()
    sklearn_decomposition_module.PCA = MagicMock()
    sklearn_cluster_module.HDBSCAN = MagicMock()

    scipy_module = ModuleType('scipy')
    scipy_stats_module = ModuleType('scipy.stats')
    scipy_stats_module.entropy = lambda values: 1.0

    levenshtein_module = ModuleType('Levenshtein')
    levenshtein_module.distance = lambda left, right: 0

    openai_module = ModuleType('openai')
    openai_module.OpenAI = MagicMock()

    return {
        'torch': torch_module,
        'transformers': transformers_module,
        'fasttext': fasttext_module,
        'sklearn': sklearn_module,
        'sklearn.metrics': sklearn_metrics_module,
        'sklearn.ensemble': sklearn_ensemble_module,
        'sklearn.preprocessing': sklearn_preprocessing_module,
        'sklearn.decomposition': sklearn_decomposition_module,
        'sklearn.cluster': sklearn_cluster_module,
        'scipy': scipy_module,
        'scipy.stats': scipy_stats_module,
        'Levenshtein': levenshtein_module,
        'openai': openai_module,
    }


def _install_optional_dependency_patches():
    """Keep heavy optional dependencies mocked for the whole test module."""
    for name, module in _optional_dependency_modules().items():
        sys.modules[name] = module


_install_optional_dependency_patches()


# ==========================================
# 2. 测试类定义
# ==========================================

class TestAnomalyTupleSelectorUnits(unittest.TestCase):
    """测试 AnomalyTupleSelector.py 中的底层评分和选择函数。"""

    def _import_selector(self):
        with patch.dict(sys.modules, _optional_dependency_modules()):
            from AnomalyTupleSelector import AnomalyTupleSelector
            return AnomalyTupleSelector

    def test_pattern_normalization(self):
        """验证格式模式归一化规则。"""
        AnomalyTupleSelector = self._import_selector()
        selector = AnomalyTupleSelector.__new__(AnomalyTupleSelector)

        from_chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
        to_chars = ('A' * 26) + ('a' * 26) + ('9' * 10)
        selector.fine_trans_table = str.maketrans(from_chars, to_chars)

        self.assertEqual(selector._get_pattern('AB-123'), 'AA-999')
        self.assertEqual(selector._get_pattern('12.0 oz'), '99.9 aa')
        self.assertEqual(selector._get_pattern('nan'), 'aaa')
        self.assertEqual(selector._get_pattern(None), 'NULL')

    def test_column_format_consistency_score(self):
        """验证格式一致性分数计算。"""
        AnomalyTupleSelector = self._import_selector()
        selector = AnomalyTupleSelector.__new__(AnomalyTupleSelector)

        score_one_dominant_pattern = selector._calculate_column_format_consistency(
            {'A': 0.8, 'B': 0.05}, min_freq_threshold=0.1
        )
        self.assertAlmostEqual(score_one_dominant_pattern, 0.8)

        score_two_high_patterns = selector._calculate_column_format_consistency(
            {'A': 0.6, 'B': 0.4}, min_freq_threshold=0.1
        )
        self.assertAlmostEqual(score_two_high_patterns, 1.0 / (2 ** 0.5))

        self.assertEqual(
            selector._calculate_column_format_consistency({'A': 1.0}, min_freq_threshold=0.1),
            0.0,
        )

    def test_theils_u_perfect_dependency(self):
        """验证 Theil's U 在完美函数依赖下返回 1。"""
        AnomalyTupleSelector = self._import_selector()
        selector = AnomalyTupleSelector(pd.DataFrame({
            'entity': ['A', 'A', 'B', 'B'],
            'target': ['X', 'X', 'Y', 'Y'],
        }))

        score = selector._calculate_theils_u(selector.raw_df['entity'], selector.raw_df['target'])
        self.assertAlmostEqual(score, 1.0)

    def test_select_coverage_aware_picks_highest_uncovered_columns(self):
        """验证 coverage-aware 选择会优先覆盖异常分数最高的列。"""
        AnomalyTupleSelector = self._import_selector()
        selector = AnomalyTupleSelector.__new__(AnomalyTupleSelector)

        score_matrix = pd.DataFrame({
            'col_a': [0.1, 0.9, 0.2],
            'col_b': [0.8, 0.1, 0.3],
        })
        selected = selector._select_coverage_aware(score_matrix, [0, 1, 2], k=2)

        self.assertEqual(selected, [1, 0])


class TestFormatCleanerUnits(unittest.TestCase):
    """测试 FormatCleaner.py 中的模式、解析和指标函数。"""

    def _import_format_cleaner(self):
        with patch.dict(sys.modules, _optional_dependency_modules()):
            from FormatCleaner import FormatCleaner
            return FormatCleaner

    def test_pattern_normalization(self):
        """验证格式模式归一化。"""
        FormatCleaner = self._import_format_cleaner()
        cleaner = FormatCleaner.__new__(FormatCleaner)

        from_chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
        to_chars = ('A' * 26) + ('a' * 26) + ('9' * 10)
        cleaner.fine_trans_table = str.maketrans(from_chars, to_chars)

        self.assertEqual(cleaner._get_pattern('12.0 oz'), '99.9 aa')
        self.assertEqual(cleaner._get_pattern('AB-123'), 'AA-999')
        self.assertEqual(cleaner._get_pattern('nan'), 'NULL')

    def test_extract_decision_and_code(self):
        """验证从 LLM 响应中提取 decision、code 和 reason。"""
        FormatCleaner = self._import_format_cleaner()
        cleaner = FormatCleaner.__new__(FormatCleaner)

        response = '''是```python
def detect_format_error(value):
    return True
```
<reason>格式不一致</reason>'''

        decision, code, reason = cleaner._extract_decision_and_code(response, 'detect_format_error')

        self.assertEqual(decision, '是')
        self.assertIn('def detect_format_error', code)
        self.assertEqual(reason, '格式不一致')

    def test_evaluate_detection_metrics(self):
        """验证格式检测指标计算。"""
        FormatCleaner = self._import_format_cleaner()
        cleaner = FormatCleaner.__new__(FormatCleaner)
        cleaner.dirty_df = pd.DataFrame({'city': ['Paris', 'Paris', 'Berlin']})
        cleaner.clean_df = pd.DataFrame({'city': ['Paris', 'Rome', 'Berlin']})

        results = {}

        def detect_func(value):
            return value == 'Paris'

        metrics = cleaner._evaluate_detection('city', detect_func, results)

        self.assertEqual(metrics['all_need_detect'], 1)
        self.assertEqual(metrics['all_detected'], 2)
        self.assertEqual(metrics['correctly_detect'], 1)
        self.assertEqual(metrics['wrongly_detect'], 1)
        self.assertEqual(metrics['missing_errors'], 0)
        self.assertIn('detection_metrics', results)

    def test_evaluate_repair_metrics(self):
        """验证格式修复指标计算。"""
        FormatCleaner = self._import_format_cleaner()
        cleaner = FormatCleaner.__new__(FormatCleaner)
        cleaner.dirty_df = pd.DataFrame({'city': ['Paris', 'Rome', 'Berlin']})
        cleaner.clean_df = pd.DataFrame({'city': ['Paris', 'Rome', 'London']})

        results = {}

        def detect_func(value):
            return value in ['Paris', 'Berlin']

        def repair_func(value):
            if value == 'Paris':
                return 'Paris'
            if value == 'Berlin':
                return 'London'
            return value

        metrics = cleaner._evaluate_repair('city', detect_func, repair_func, results)

        self.assertEqual(metrics['all_need_repair'], 1)
        self.assertEqual(metrics['all_repaired'], 1)
        self.assertEqual(metrics['wrong_2_right (TP)'], 1)
        self.assertEqual(metrics['wrong_not_change (FN)'], 0)
        self.assertAlmostEqual(metrics['Precision'], 1.0)
        self.assertIn('repair_metrics', results)


class TestFDCleanerUnits(unittest.TestCase):
    """测试 FDCleaner.py 中的 XML 解析函数。"""

    def _import_fd_cleaner(self):
        with patch.dict(sys.modules, _optional_dependency_modules()):
            from FDCleaner import FDCleaner
            return FDCleaner

    def test_extract_judgment(self):
        """验证 FD 判断结果解析。"""
        FDCleaner = self._import_fd_cleaner()
        cleaner = FDCleaner.__new__(FDCleaner)

        decision, reason = cleaner._extract_judgment(
            '<decision>Yes</decision><reason>同一实体必须一致</reason>'
        )

        self.assertEqual(decision, '是')
        self.assertEqual(reason, '同一实体必须一致')

    def test_extract_fallback(self):
        """验证 FD fallback 建议值解析。"""
        FDCleaner = self._import_fd_cleaner()
        cleaner = FDCleaner.__new__(FDCleaner)

        suggested_value, reason = cleaner._extract_fallback(
            '<suggested_value>BAPTIST MEDICAL CENTER</suggested_value><reason>多数上下文支持</reason>'
        )

        self.assertEqual(suggested_value, 'BAPTIST MEDICAL CENTER')
        self.assertEqual(reason, '多数上下文支持')


class TestSemanticCleanerUnits(unittest.TestCase):
    """测试 SemanticCleaner.py 中的解析和检测评估函数。"""

    def _import_semantic_cleaner(self):
        with patch.dict(sys.modules, _optional_dependency_modules()):
            from SemanticCleaner import SemanticCleaner
            return SemanticCleaner

    def test_extract_probe_decision(self):
        """验证语义探针 decision 解析。"""
        SemanticCleaner = self._import_semantic_cleaner()
        cleaner = SemanticCleaner.__new__(SemanticCleaner)

        self.assertEqual(cleaner._extract_probe_decision('<decision>否</decision>'), '否')
        self.assertEqual(cleaner._extract_probe_decision('<decision>YES</decision>'), '是')
        self.assertIsNone(cleaner._extract_probe_decision('invalid response'))

    def test_extract_repair_response(self):
        """验证语义修复 XML 解析。"""
        SemanticCleaner = self._import_semantic_cleaner()
        cleaner = SemanticCleaner.__new__(SemanticCleaner)

        response = '''<repairs>
    <repair row="2" col="city">
        <value>Rome</value>
        <reason>根据同组上下文修正。</reason>
    </repair>
</repairs>'''

        repairs = cleaner._extract_repair_response(response)

        self.assertEqual(len(repairs), 1)
        self.assertEqual(repairs[0], (2, 'city', 'Rome', '根据同组上下文修正。'))

    def test_evaluate_detection_metrics(self):
        """验证语义检测指标计算。"""
        SemanticCleaner = self._import_semantic_cleaner()
        cleaner = SemanticCleaner.__new__(SemanticCleaner)
        cleaner.current_dirty_df = pd.DataFrame({'city': ['Paris', 'Paris', 'Berlin']})
        cleaner.clean_df = pd.DataFrame({'city': ['Paris', 'Rome', 'Berlin']})

        with tempfile.TemporaryDirectory() as temp_dir:
            cleaner.phase_semantic_dir = temp_dir
            evaluation = cleaner._evaluate_detection({'city': [0]})

        city_metrics = evaluation['Column_Metrics']['city']
        global_metrics = evaluation['Global_Metrics']

        self.assertEqual(city_metrics['all_need_detect'], 1)
        self.assertEqual(city_metrics['all_detected'], 1)
        self.assertEqual(city_metrics['correctly_detect'], 0)
        self.assertEqual(city_metrics['wrongly_detect'], 1)
        self.assertEqual(city_metrics['missing_errors'], 1)
        self.assertEqual(global_metrics['Precision'], 0.0)
        self.assertTrue(os.path.exists(os.path.join(temp_dir, 'merged_semantic_detection_evaluation.json')))


class TestProCleanUnits(unittest.TestCase):
    """测试 ProClean.py 中的项目级入口函数。"""

    def _import_proclean(self):
        with patch.dict(sys.modules, _optional_dependency_modules()):
            from ProClean import ProClean, run_project, run_proclean
            return ProClean, run_project, run_proclean

    def test_run_proclean_returns_config_metadata(self):
        """验证 run_proclean 返回语义清洗结果和配置元数据。"""
        _, _, run_proclean = self._import_proclean()

        with tempfile.TemporaryDirectory() as temp_dir:
            dirty_path = os.path.join(temp_dir, 'dirty.csv')
            clean_path = os.path.join(temp_dir, 'clean.csv')
            semantic_model_path = os.path.join(temp_dir, 'semantic_model')
            fasttext_model_path = os.path.join(temp_dir, 'fasttext_model.bin')

            for path in [dirty_path, clean_path, semantic_model_path, fasttext_model_path]:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write('ok')

            with patch('ProClean.run_semantic_cleaner', return_value={'base_dir': temp_dir}) as mock_runner:
                result = run_proclean(
                    dataset='hospital',
                    dirty_path=dirty_path,
                    clean_path=clean_path,
                    result_root=os.path.join(temp_dir, 'results'),
                    debug_mode=False,
                    llm_base_url='http://localhost:8000/v1',
                    llm_api_key='test-key',
                    llm_model='mock-model',
                    semantic_model_path=semantic_model_path,
                    fasttext_model_path=fasttext_model_path,
                )

            mock_runner.assert_called_once()
            self.assertEqual(result['dataset'], 'hospital')
            self.assertEqual(result['base_dir'], temp_dir)
            self.assertEqual(result['config']['result_root'], os.path.join(temp_dir, 'results'))
            self.assertFalse(result['config']['debug_mode'])


if __name__ == '__main__':
    unittest.main()

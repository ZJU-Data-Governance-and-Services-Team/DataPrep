import os
import sys
import tempfile
import unittest
from types import ModuleType
from unittest.mock import MagicMock, patch


# ==========================================
# 1. 导入路径与可选依赖 Mock
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))

if project_root not in sys.path:
    sys.path.append(project_root)


def _optional_dependency_modules():
    """Patch heavy optional dependencies so ProClean tests do not load LLM/model files."""
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

class TestProCleanEntryPoints(unittest.TestCase):
    """测试 ProClean.py 中的项目级入口和参数校验逻辑。"""

    def _import_proclean(self):
        with patch.dict(sys.modules, _optional_dependency_modules()):
            from ProClean import ProClean, run_project, run_proclean
            return ProClean, run_project, run_proclean

    def test_require_text_and_path_helpers(self):
        """验证必需文本和必需路径校验是否正常。"""
        with patch.dict(sys.modules, _optional_dependency_modules()):
            from ProClean import _require_path, _require_text

        with tempfile.TemporaryDirectory() as temp_dir:
            existing_path = os.path.join(temp_dir, 'dirty.csv')
            missing_path = os.path.join(temp_dir, 'missing.csv')
            with open(existing_path, 'w', encoding='utf-8') as f:
                f.write('a\n1\n')

            self.assertIsNone(_require_text('dataset', 'beers'))
            self.assertIsNone(_require_path('dirty_path', existing_path))

            with self.assertRaises(ValueError):
                _require_text('dataset', '')

            with self.assertRaises(FileNotFoundError):
                _require_path('dirty_path', missing_path)

    def test_run_proclean_delegates_to_semantic_cleaner(self):
        """测试 run_proclean 是否按预期调用 SemanticCleaner 的入口函数。"""
        with patch.dict(sys.modules, _optional_dependency_modules()):
            from ProClean import run_proclean

        with tempfile.TemporaryDirectory() as temp_dir:
            dirty_path = os.path.join(temp_dir, 'dirty.csv')
            clean_path = os.path.join(temp_dir, 'clean.csv')
            semantic_model_path = os.path.join(temp_dir, 'semantic_model')
            fasttext_model_path = os.path.join(temp_dir, 'fasttext_model.bin')

            for path in [dirty_path, clean_path, semantic_model_path, fasttext_model_path]:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write('ok')

            expected_result = {
                'base_dir': os.path.join(temp_dir, 'run'),
                'phase_semantic_dir': os.path.join(temp_dir, 'run', 'phase_semantic'),
            }

            with patch('ProClean.run_semantic_cleaner', return_value=expected_result) as mock_runner:
                result = run_proclean(
                    dataset='beers',
                    dirty_path=dirty_path,
                    clean_path=clean_path,
                    result_root=os.path.join(temp_dir, 'results'),
                    debug_mode=True,
                    llm_base_url='http://localhost:8000/v1',
                    llm_api_key='test-key',
                    llm_model='mock-model',
                    semantic_model_path=semantic_model_path,
                    fasttext_model_path=fasttext_model_path,
                )

            mock_runner.assert_called_once_with(
                dataset='beers',
                original_dirty_path=dirty_path,
                clean_path=clean_path,
                result_root=os.path.join(temp_dir, 'results'),
                debug_mode=True,
                llm_base_url='http://localhost:8000/v1',
                llm_api_key='test-key',
                llm_model='mock-model',
                model_path=semantic_model_path,
                fasttext_model_path=fasttext_model_path,
            )

            self.assertEqual(result['base_dir'], expected_result['base_dir'])
            self.assertEqual(result['phase_semantic_dir'], expected_result['phase_semantic_dir'])
            self.assertEqual(result['config']['dataset'], 'beers')
            self.assertEqual(result['config']['dirty_path'], dirty_path)
            self.assertEqual(result['config']['clean_path'], clean_path)
            self.assertTrue(result['config']['debug_mode'])
            self.assertEqual(result['config']['llm_base_url'], 'http://localhost:8000/v1')
            self.assertEqual(result['config']['llm_model'], 'mock-model')
            self.assertEqual(result['config']['semantic_model_path'], semantic_model_path)
            self.assertEqual(result['config']['fasttext_model_path'], fasttext_model_path)

    def test_run_proclean_validates_required_inputs(self):
        """验证 run_proclean 对缺失路径和缺失配置的检查。"""
        with patch.dict(sys.modules, _optional_dependency_modules()):
            from ProClean import run_proclean

        with tempfile.TemporaryDirectory() as temp_dir:
            dirty_path = os.path.join(temp_dir, 'dirty.csv')
            clean_path = os.path.join(temp_dir, 'clean.csv')
            semantic_model_path = os.path.join(temp_dir, 'semantic_model')
            fasttext_model_path = os.path.join(temp_dir, 'fasttext_model.bin')

            for path in [dirty_path, clean_path, semantic_model_path, fasttext_model_path]:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write('ok')

            with self.assertRaises(ValueError):
                run_proclean(
                    dataset='',
                    dirty_path=dirty_path,
                    clean_path=clean_path,
                    semantic_model_path=semantic_model_path,
                    fasttext_model_path=fasttext_model_path,
                    llm_base_url='http://localhost:8000/v1',
                    llm_api_key='test-key',
                    llm_model='mock-model',
                )

            with self.assertRaises(ValueError):
                run_proclean(
                    dataset='beers',
                    dirty_path=dirty_path,
                    clean_path=clean_path,
                    semantic_model_path=semantic_model_path,
                    fasttext_model_path=fasttext_model_path,
                    llm_base_url='',
                    llm_api_key='test-key',
                    llm_model='mock-model',
                )

            with self.assertRaises(FileNotFoundError):
                run_proclean(
                    dataset='beers',
                    dirty_path=os.path.join(temp_dir, 'missing.csv'),
                    clean_path=clean_path,
                    semantic_model_path=semantic_model_path,
                    fasttext_model_path=fasttext_model_path,
                    llm_base_url='http://localhost:8000/v1',
                    llm_api_key='test-key',
                    llm_model='mock-model',
                )

    def test_proclean_instance_run_delegates_to_run_proclean(self):
        """测试 ProClean 实例的 run() 是否转发到 run_proclean。"""
        ProClean, _, run_proclean = self._import_proclean()

        with tempfile.TemporaryDirectory() as temp_dir:
            dirty_path = os.path.join(temp_dir, 'dirty.csv')
            clean_path = os.path.join(temp_dir, 'clean.csv')
            semantic_model_path = os.path.join(temp_dir, 'semantic_model')
            fasttext_model_path = os.path.join(temp_dir, 'fasttext_model.bin')

            for path in [dirty_path, clean_path, semantic_model_path, fasttext_model_path]:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write('ok')

            cleaner = ProClean(
                dataset='beers',
                dirty_path=dirty_path,
                clean_path=clean_path,
                result_root=os.path.join(temp_dir, 'results'),
                debug_mode=True,
                llm_base_url='http://localhost:8000/v1',
                llm_api_key='test-key',
                llm_model='mock-model',
                semantic_model_path=semantic_model_path,
                fasttext_model_path=fasttext_model_path,
            )

            with patch('ProClean.run_proclean', return_value={'ok': True}) as mock_runner:
                result = cleaner.run()

            mock_runner.assert_called_once_with(
                dataset='beers',
                dirty_path=dirty_path,
                clean_path=clean_path,
                result_root=os.path.join(temp_dir, 'results'),
                debug_mode=True,
                llm_base_url='http://localhost:8000/v1',
                llm_api_key='test-key',
                llm_model='mock-model',
                semantic_model_path=semantic_model_path,
                fasttext_model_path=fasttext_model_path,
            )
            self.assertEqual(result, {'ok': True})

    def test_run_project_alias(self):
        """验证 run_project 是 run_proclean 的别名。"""
        _, run_project, run_proclean = self._import_proclean()
        self.assertIs(run_project, run_proclean)


if __name__ == '__main__':
    unittest.main()

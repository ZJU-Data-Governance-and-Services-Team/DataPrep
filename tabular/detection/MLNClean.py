"""
MLNClean Detector
=================
把 MLNClean 用作"错误检测器": 跑完整 MLN 清洗管线, 对比"清洗前 vs 清洗后",
凡是值变了的格子, 都视为"算法检测到的错误", 输出布尔 mask.

用法:
    from dataprep.tabular.detection.MLNClean import MLNClean

    # 读规则
    with open('rules.txt') as f:
        rules = f.readlines()

    detector = MLNClean(
        rules=rules,
        evidence_df=evidence_df,
        partition_number=1,
        mcmc_samples=20,
    )
    error_mask = detector.train_and_predict(dirty_df)  # bool DataFrame, True=有错
"""
import os
import pandas as pd

try:
    from dataprep.base import BaseEstimator
    import dataprep.tabular.detection.MLNClean_modules as mo
except ModuleNotFoundError:
    from base import BaseEstimator
    import tabular.detection.MLNClean_modules as mo


class MLNClean(BaseEstimator):
    """
    MLNClean as Detector.

    Reference:
        "Cleaning Uncertain Data via Markov Logic Networks."
    """

    def __init__(self,
                 rules=None,
                 rules_path: str = None,
                 evidence_df: pd.DataFrame = None,
                 evidence_path: str = None,
                 partition_number: int = 1,
                 agp_threshold: int = 2,
                 mcmc_samples: int = 20,
                 mcmc_warmup: int = 20,
                 verbose: bool = True,
                 **kwargs):
        """
        Args:
            rules           : list[str], MLN 规则字符串 (一行一条)
            rules_path      : str, 规则文件路径 (二选一)
            evidence_df     : pd.DataFrame, 学权重用的证据数据
            evidence_path   : str, 证据 CSV 路径
            partition_number: int, 数据分区数; >1 时触发原 MLNClean 启发式分区
            agp_threshold   : int, AGP 组大小阈值
            mcmc_samples    : int, Pyro MCMC 采样数
            mcmc_warmup     : int, Pyro MCMC warmup 步数
            verbose         : bool, 是否打印进度
        """
        # 规则: 优先用 rules, 否则从 rules_path 读
        if rules is None and rules_path is None:
            raise ValueError("必须指定 rules 或 rules_path 之一")
        if rules is None:
            with open(rules_path, 'r', encoding='utf-8') as f:
                rules = f.readlines()
        self.rules = list(rules)

        # 证据: 优先 evidence_df, 否则 evidence_path
        self.evidence_df = evidence_df
        if self.evidence_df is None and evidence_path is not None:
            self.evidence_df = pd.read_csv(evidence_path)

        self.partition_number = partition_number
        self.agp_threshold = agp_threshold
        self.mcmc_samples = mcmc_samples
        self.mcmc_warmup = mcmc_warmup
        self.verbose = verbose

        # 内部状态
        self.cleaned_df_ = None
        self.is_trained_ = False

    # ------------------------------------------------------------------
    # Train: 跑完整管线, 把清洗后 DataFrame 缓存到 self
    # ------------------------------------------------------------------
    def train(self, dirty_df: pd.DataFrame, **kwargs):
        if hasattr(self, '_create_temp_dir'):
            self._create_temp_dir(prefix="mlnclean_det_")

        if self.verbose:
            print(f"[MLNClean-Det] Training on dataset shape: {dirty_df.shape}")

        self.cleaned_df_ = mo.run_mln_clean_pipeline(
            dirty_df=dirty_df,
            rules_text_list=self.rules,
            evidence_df=self.evidence_df,
            partition_number=self.partition_number,
            agp_threshold=self.agp_threshold,
            mcmc_samples=self.mcmc_samples,
            mcmc_warmup=self.mcmc_warmup,
            verbose=self.verbose,
        )
        self.is_trained_ = True

        if hasattr(self, '_save_checkpoint'):
            self._save_checkpoint("mlnclean_det_complete.pkl")

    # ------------------------------------------------------------------
    # Predict: dirty 对比 cleaned, 凡是变了的格子标 True
    # ------------------------------------------------------------------
    def predict(self, dirty_df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_trained_:
            raise RuntimeError("Model is not trained. Call .train() first.")

        mask = mo.compute_diff_mask(dirty_df, self.cleaned_df_)
        if self.verbose:
            n_errors = int(mask.values.sum())
            total = mask.size
            print(f"[MLNClean-Det] Found {n_errors} / {total} dirty cells ({n_errors / total:.2%})")
        return mask

    def train_and_predict(self, dirty_df: pd.DataFrame) -> pd.DataFrame:
        self.train(dirty_df)
        return self.predict(dirty_df)


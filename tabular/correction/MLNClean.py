import pandas as pd

try:
    from dataprep.base import BaseEstimator
    # 共享 modules: 复用 detection 子包下的 MLNClean_modules
    import dataprep.tabular.detection.MLNClean_modules as mo
except ModuleNotFoundError:
    from base import BaseEstimator
    import tabular.detection.MLNClean_modules as mo


class MLNClean(BaseEstimator):
    """
    MLNClean as Corrector.

    Reference:
        "Cleaning Uncertain Data via Markov Logic Networks."
    """

    def __init__(self,
                 rules=None,
                 rules_path: str = None,
                 evidence_df: pd.DataFrame = None,
                 evidence_path: str = None,
                 detection_mask: pd.DataFrame = None,
                 detection_path: str = None,
                 cleaned_df: pd.DataFrame = None,
                 partition_number: int = 1,
                 agp_threshold: int = 2,
                 mcmc_samples: int = 20,
                 mcmc_warmup: int = 20,
                 verbose: bool = True,
                 **kwargs):
        """
        Args:
            rules           : list[str], MLN 规则字符串
            rules_path      : str, 规则文件路径 (二选一)
            evidence_df     : pd.DataFrame, 学权重用的证据数据
            evidence_path   : str, 证据 CSV 路径
            detection_mask  : pd.DataFrame, 错误检测 mask (True=要修).
                              不传则代表"全量修复"
            detection_path  : str, mask CSV 路径 (二选一)
            partition_number: int, 数据分区数
            agp_threshold   : int, AGP 组大小阈值
            mcmc_samples    : int, Pyro MCMC 采样数
            mcmc_warmup     : int, Pyro MCMC warmup 步数
            verbose         : bool, 是否打印进度
        """
        if rules is None and rules_path is None:
            raise ValueError("必须指定 rules 或 rules_path 之一")
        if rules is None:
            with open(rules_path, 'r', encoding='utf-8') as f:
                rules = f.readlines()
        self.rules = list(rules)

        self.evidence_df = evidence_df
        if self.evidence_df is None and evidence_path is not None:
            self.evidence_df = pd.read_csv(evidence_path)

        # detection mask 可选
        self.detection_mask = detection_mask
        if self.detection_mask is None and detection_path is not None:
            self.detection_mask = pd.read_csv(detection_path)

        if self.detection_mask is not None:
            self.detection_mask = mo.to_bool_mask(self.detection_mask)

        self.partition_number = partition_number
        self.agp_threshold = agp_threshold
        self.mcmc_samples = mcmc_samples
        self.mcmc_warmup = mcmc_warmup
        self.verbose = verbose

        # 内部状态
        self.cleaned_df_ = cleaned_df.copy() if cleaned_df is not None else None
        self.is_trained_ = cleaned_df is not None

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    def train(self, dirty_df: pd.DataFrame, **kwargs):
        if hasattr(self, '_create_temp_dir'):
            self._create_temp_dir(prefix="mlnclean_cor_")

        if self.verbose:
            print(f"[MLNClean-Cor] Training on dataset shape: {dirty_df.shape}")

        if self.cleaned_df_ is not None:
            self.is_trained_ = True
            return

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
            self._save_checkpoint("mlnclean_cor_complete.pkl")

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict(self,dirty_df: pd.DataFrame,detection_mask: pd.DataFrame = None) -> pd.DataFrame:
        """
        Args:
            dirty_df       : 原始脏 DataFrame
            detection_mask : 可选; 不传则用 self.detection_mask;
                             如果两者都是 None, 则做"全量替换" (cleaned 直接当输出)
        Returns:
            fixed_df : 修复后的 DataFrame
        """
        if not self.is_trained_:
            raise RuntimeError("Model is not trained. Call .train() first.")

        mask = detection_mask if detection_mask is not None else self.detection_mask

        if mask is None:
            # 全量修复
            if self.verbose:
                print("[MLNClean-Cor] No detection mask: full replacement.")
            return self.cleaned_df_.copy()
        mask = mo.to_bool_mask(mask)

        fixed = mo.apply_corrections_with_mask(dirty_df, self.cleaned_df_, mask)
        if self.verbose:
            n_fixed = int(mask.values.sum())
            print(f"[MLNClean-Cor] Applied corrections to {n_fixed} cells.")
        return fixed

    def train_and_predict(self,
                          dirty_df: pd.DataFrame,
                          detection_mask: pd.DataFrame = None) -> pd.DataFrame:
        self.train(dirty_df)
        return self.predict(dirty_df, detection_mask=detection_mask)



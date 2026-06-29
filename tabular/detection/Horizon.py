# -*- coding: utf-8 -*-
import pandas as pd

try:
    from dataprep.base import BaseEstimator
except ModuleNotFoundError:
    from base import BaseEstimator

try:
    from dataprep.tabular.correction.Horizon import Horizon as HorizonRepair
except ModuleNotFoundError:
    from tabular.correction.Horizon import Horizon as HorizonRepair


class Horizon(BaseEstimator):
    """
    Horizon detection module.

    Horizon 原始算法本质是 repair-first：
        dirty_df + FD rules -> repaired_df

    detection 的做法：
        repaired_df 与 dirty_df 不同的位置，就是检测出的错误位置。
    """

    def __init__(self, rule_path=None, **kwargs):
        self.rule_path = rule_path
        self.kwargs = kwargs
        self.is_trained_ = False
        self.repaired_df_ = None

    def train(self, dirty_csv: pd.DataFrame, **kwargs):
        self.is_trained_ = True
        return self

    def predict(self, dirty_csv: pd.DataFrame) -> pd.DataFrame:
        dirty_df = dirty_csv.copy().astype(str).fillna("nan")

        repair_model = HorizonRepair(
            rule_path=self.rule_path,
            **self.kwargs
        )

        repaired_df = repair_model.train_and_predict(dirty_df)
        repaired_df = repaired_df.astype(str).fillna("nan")

        self.repaired_df_ = repaired_df

        error_mask = repaired_df != dirty_df
        return error_mask

    def train_and_predict(self, dirty_csv: pd.DataFrame) -> pd.DataFrame:
        self.train(dirty_csv)
        return self.predict(dirty_csv)
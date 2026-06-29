# -*- coding: utf-8 -*-
"""DataPrep wrapper for SCAREd detection.

SCAREd is repair-first. Detection is derived from changed cells:
    error_mask = repaired_df != dirty_df
"""

from __future__ import annotations

from typing import Any, Optional, Sequence
import pandas as pd

try:
    from dataprep.base import BaseEstimator
except ModuleNotFoundError:
    from base import BaseEstimator

try:
    from dataprep.tabular.correction.SCAREd import SCAREd as SCAREdRepair
except ModuleNotFoundError:
    from tabular.correction.SCAREd import SCAREd as SCAREdRepair


class SCAREd(BaseEstimator):
    def __init__(
        self,
        clean_df: Optional[pd.DataFrame] = None,
        detection_mask: Optional[pd.DataFrame] = None,
        reliable_attrs: Optional[Sequence[str]] = None,
        n_reliable_attrs: int = 2,
        perfected: bool = False,
        use_perfect_detection_if_clean: bool = False,
        repair_attrs: Optional[Sequence[str]] = None,
        min_partition_size: int = 1,
        max_partition_values: Optional[int] = None,
        use_index_partition: bool = True,
        **kwargs: Any,
    ):
        self.clean_df = clean_df
        self.detection_mask = detection_mask
        self.reliable_attrs = list(reliable_attrs) if reliable_attrs is not None else None
        self.n_reliable_attrs = n_reliable_attrs
        self.perfected = perfected
        self.use_perfect_detection_if_clean = use_perfect_detection_if_clean
        self.repair_attrs = list(repair_attrs) if repair_attrs is not None else None
        self.min_partition_size = min_partition_size
        self.max_partition_values = max_partition_values
        self.use_index_partition = use_index_partition
        self.kwargs = kwargs

        self.is_trained_ = False
        self.repaired_df_ = None
        self.repair_mask_ = None
        self.debug_info_ = None

    def train(self, dirty_csv: pd.DataFrame, **kwargs: Any):
        if "clean_df" in kwargs:
            self.clean_df = kwargs["clean_df"]
        if "detection_mask" in kwargs:
            self.detection_mask = kwargs["detection_mask"]
        if "repair_attrs" in kwargs:
            self.repair_attrs = list(kwargs["repair_attrs"]) if kwargs["repair_attrs"] is not None else None
        self.is_trained_ = True
        return self

    def predict(self, dirty_csv: pd.DataFrame) -> pd.DataFrame:
        if not self.is_trained_:
            raise RuntimeError("Model is not trained. Run .train() first!")

        repair_model = SCAREdRepair(
            clean_df=self.clean_df,
            detection_mask=self.detection_mask,
            reliable_attrs=self.reliable_attrs,
            n_reliable_attrs=self.n_reliable_attrs,
            perfected=self.perfected,
            use_perfect_detection_if_clean=self.use_perfect_detection_if_clean,
            apply_only_detected=False,
            repair_attrs=self.repair_attrs,
            min_partition_size=self.min_partition_size,
            max_partition_values=self.max_partition_values,
            use_index_partition=self.use_index_partition,
            **self.kwargs,
        )
        repaired = repair_model.train_and_predict(dirty_csv)
        self.repaired_df_ = repaired
        self.repair_mask_ = repair_model.repair_mask_
        self.debug_info_ = repair_model.debug_info_

        error_mask = repaired.astype(str).ne(dirty_csv.copy().fillna("null").astype(str))
        error_mask.index = dirty_csv.index
        error_mask.columns = dirty_csv.columns
        return error_mask.astype(bool)

    def train_and_predict(self, dirty_csv: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        self.train(dirty_csv, **kwargs)
        return self.predict(dirty_csv)

# -*- coding: utf-8 -*-
"""DataPrep wrapper for SCAREd repair/correction.

This wrapper keeps DataPrep's BaseEstimator-style API, while the core logic in
SCAREd_modules.py follows the teacher-provided scared.py script.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence
import pandas as pd

try:
    from dataprep.base import BaseEstimator
except ModuleNotFoundError:
    from base import BaseEstimator

try:
    import dataprep.tabular.correction.SCAREd_modules as modules
except ModuleNotFoundError:
    import tabular.correction.SCAREd_modules as modules


class SCAREd(BaseEstimator):
    def __init__(
        self,
        clean_df: Optional[pd.DataFrame] = None,
        detection_mask: Optional[pd.DataFrame] = None,
        reliable_attrs: Optional[Sequence[str]] = None,
        n_reliable_attrs: int = 2,
        perfected: bool = False,
        use_perfect_detection_if_clean: bool = False,
        apply_only_detected: bool = False,
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
        self.apply_only_detected = apply_only_detected
        self.repair_attrs = list(repair_attrs) if repair_attrs is not None else None
        self.min_partition_size = min_partition_size
        self.max_partition_values = max_partition_values
        self.use_index_partition = use_index_partition
        self.kwargs = kwargs

        self.is_trained_ = False
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

        repaired, mask, debug = modules.generate_repairs(
            dirty_df=dirty_csv,
            clean_df=self.clean_df,
            detection_mask=self.detection_mask,
            reliable_attrs=self.reliable_attrs,
            n_reliable_attrs=self.n_reliable_attrs,
            perfected=self.perfected,
            use_perfect_detection_if_clean=self.use_perfect_detection_if_clean,
            apply_only_detected=self.apply_only_detected,
            repair_attrs=self.repair_attrs,
            min_partition_size=self.min_partition_size,
            max_partition_values=self.max_partition_values,
            use_index_partition=self.use_index_partition,
            return_mask=True,
            return_debug=True,
        )
        self.repair_mask_ = mask
        self.debug_info_ = debug
        return repaired

    def train_and_predict(self, dirty_csv: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        self.train(dirty_csv, **kwargs)
        return self.predict(dirty_csv)

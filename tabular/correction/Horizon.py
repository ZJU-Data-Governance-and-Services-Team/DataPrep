# -*- coding: utf-8 -*-
"""DataPrep wrapper for Horizon repair/correction."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple, Union, Dict
import pandas as pd

try:
    from dataprep.base import BaseEstimator
except ModuleNotFoundError:
    from base import BaseEstimator

try:
    import dataprep.tabular.correction.Horizon_modules as modules
except ModuleNotFoundError:
    import tabular.correction.Horizon_modules as modules


class Horizon(BaseEstimator):
    """Horizon FD-based data repair.

    Parameters
    ----------
    rule_path:
        Path to FD rules.  Text rules such as `A ⇒ B` / `A -> B` and DataPrep
        JSON dict rules are both supported.
    rules:
        Optional in-memory FD rules, e.g. `[('provider_id', 'provider_address')]`.
    detection_mask:
        Optional boolean mask.  Used only if `apply_only_detected=True`.
    apply_only_detected:
        If True, only replace cells marked True in `detection_mask`; otherwise
        return Horizon's full repaired table.
    """

    def __init__(
        self,
        rule_path: Optional[str] = None,
        rules: Optional[Sequence[Union[modules.FDRule, Tuple[Union[str, Sequence[str]], str], Dict[str, Union[str, Sequence[str]]]]]] = None,
        detection_mask: Optional[pd.DataFrame] = None,
        apply_only_detected: bool = False,
        **kwargs: Any,
    ):
        self.rule_path = rule_path
        self.rules = rules
        self.detection_mask = detection_mask
        self.apply_only_detected = apply_only_detected
        self.kwargs = kwargs
        self.is_trained_ = False
        self.pattern_expressions_ = None

    def train(self, dirty_csv: pd.DataFrame, **kwargs: Any):
        # Horizon is rule-based and does not learn parameters across calls.
        if "detection_mask" in kwargs:
            self.detection_mask = kwargs["detection_mask"]
        self.is_trained_ = True
        return self

    def predict(self, dirty_csv: pd.DataFrame) -> pd.DataFrame:
        if not self.is_trained_:
            raise RuntimeError("Model is not trained. Run .train() first!")
        repaired, pattern_expressions = modules.generate_repairs(
            dirty_df=dirty_csv,
            rule_path=self.rule_path,
            rules=self.rules,
            detection_mask=self.detection_mask,
            apply_only_detected=self.apply_only_detected,
            return_pattern_expressions=True,
        )
        self.pattern_expressions_ = pattern_expressions
        return repaired

    def train_and_predict(self, dirty_csv: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        self.train(dirty_csv, **kwargs)
        return self.predict(dirty_csv)

# -*- coding: utf-8 -*-
"""
Strict DataPrep refactor of teacher's SCAREd/scared.py.

This file keeps the teacher script's algorithmic order as much as possible:
    partition -> get_model -> get_all_preds/_get_single_preds -> get_final_pred/khs_solution -> repaired_df

Differences from the script are only interface/engineering changes:
    1. Accept DataFrame inputs instead of csv paths.
    2. Remove argparse, global stdout redirection and fixed output paths.
    3. Make clean_df / detection_mask optional. When clean_df is provided,
       an external detection_mask can be passed explicitly. clean_df is kept for
       evaluation/debug and no longer creates a perfect mask by default.
    4. Keep a DataPrep-style return: repaired_df and repair_mask.

Important:
    The original script depends on a global detection_dictionary to select
    additional reliable attributes. This refactor preserves that behavior by
    deriving the detection dictionary from detection_mask or clean_df.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

NAN_TOKEN = "null"
INDEX_COL = "Index"


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Match the teacher script's string-based processing."""
    return df.copy().replace({np.nan: NAN_TOKEN}).fillna(NAN_TOKEN).astype(str)


def _make_one_hot_encoder() -> OneHotEncoder:
    """Compatible with both old and new scikit-learn."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _align_mask(mask: Optional[pd.DataFrame], df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if mask is None:
        return None
    aligned = mask.copy()
    aligned = aligned.reindex(index=df.index, columns=df.columns).fillna(False)
    return aligned.replace({"True": True, "False": False, 1: True, 0: False}).astype(bool)


def build_mask_from_clean(dirty_df: pd.DataFrame, clean_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Build perfect detection mask from dirty/clean, equivalent to PERFECTED branch."""
    if clean_df is None:
        return None
    d = _normalize_df(dirty_df)
    c = _normalize_df(clean_df).reindex(index=d.index, columns=d.columns)
    return d.ne(c).astype(bool)


def mask_to_detection_dictionary(mask: Optional[pd.DataFrame]) -> Dict[Tuple[int, int], str]:
    """Convert a boolean mask into the teacher script's detection_dictionary format."""
    detection_dictionary: Dict[Tuple[int, int], str] = {}
    if mask is None:
        return detection_dictionary
    mask_bool = mask.astype(bool)
    for i in range(mask_bool.shape[0]):
        for j, col in enumerate(mask_bool.columns):
            if bool(mask_bool.iat[i, j]):
                detection_dictionary[(i, j)] = "dummy"
    return detection_dictionary


def _extra_reliable_attrs_from_detection(
    columns: Sequence[str],
    detection_dictionary: Dict[Tuple[int, int], str],
    n_reliable_attrs: int = 2,
) -> List[str]:
    """Choose extra reliable attributes from a detection dictionary.

    Difference from the early refactor:
    - The early code only ranked columns that appeared in detection_dictionary.
      This ignored columns with zero detected errors, which are actually better
      reliable-attribute candidates.
    - Here we rank *all* original columns by detected-error count.
    - If no detection dictionary is available, return [] to avoid arbitrary
      automatic reliable-attribute choices.
    """
    from collections import Counter

    if not detection_dictionary or n_reliable_attrs <= 0:
        return []

    det_count = Counter([key[1] for key in detection_dictionary.keys()])
    ranked = []
    for j, col in enumerate(columns):
        ranked.append((j, col, int(det_count.get(j, 0))))

    ranked.sort(key=lambda x: (x[2], x[0]))
    return [col for _, col, _ in ranked[:n_reliable_attrs]]


@dataclass
class Candidate:
    """Candidate tuple repair stored in RS."""
    com_prob: float
    values: pd.Series
    weight: float
    source: Tuple[int, int]


class _ConstantModel:
    """Fallback for partitions where one target has a single class."""
    def __init__(self, value: str):
        self.value = str(value)

    def predict(self, X):
        return np.array([self.value] * len(X))

    def predict_proba(self, X):
        return np.ones((len(X), 1), dtype=float)


class SCAREdCleaner:
    """
    DataFrame version of teacher's SCAREd class.

    The key method correspondence is:
        teacher __init__      -> this __init__
        partition             -> partition
        get_model             -> get_model
        get_ori_prob          -> get_ori_prob
        get_all_preds         -> get_all_preds
        _get_single_preds     -> _get_single_preds
        get_final_pred        -> get_final_pred
        khs_solution          -> khs_solution
        run                   -> run
    """

    def __init__(
        self,
        dirty_df: pd.DataFrame,
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
    ):
        self.original_index = dirty_df.index
        self.original_columns = list(dirty_df.columns)

        self.dirty_csv = _normalize_df(dirty_df).reset_index(drop=True)
        self.rep_csv = copy.deepcopy(self.dirty_csv)
        self.clean_csv = _normalize_df(clean_df).reset_index(drop=True) if clean_df is not None else None

        # Teacher script uses clean_csv to compute wrong_cells for evaluation/perfected branch.
        self.wrong_cells: List[Tuple[int, int]] = []
        if self.clean_csv is not None:
            clean_aligned = self.clean_csv.reindex(index=self.dirty_csv.index, columns=self.dirty_csv.columns)
            diff = self.dirty_csv.ne(clean_aligned)
            for i in range(diff.shape[0]):
                # Teacher script skipped column 0. We do the same if there is at least one column.
                for j in range(1, diff.shape[1]):
                    if bool(diff.iat[i, j]):
                        self.wrong_cells.append((i, j))

        # Detection dictionary source:
        # 1. external detection_mask, or
        # 2. perfect mask from clean_df if requested.
        self.detection_mask = _align_mask(detection_mask, self.dirty_csv)
        if self.detection_mask is None and self.clean_csv is not None and use_perfect_detection_if_clean:
            self.detection_mask = build_mask_from_clean(self.dirty_csv, self.clean_csv)

        self.detection_dictionary = mask_to_detection_dictionary(self.detection_mask)
        self.perfected = bool(perfected)
        self.apply_only_detected = bool(apply_only_detected)
        self.min_partition_size = int(min_partition_size)
        self.max_partition_values = max_partition_values
        self.use_index_partition = bool(use_index_partition)

        # Teacher script:
        #   self.reliable_attrs = reliable_attrs
        #   self.reliable_attrs.append('Index')
        #   self.reliable_attrs.extend(re_attrs from detection_dictionary)
        base_reliable = [c for c in (list(reliable_attrs) if reliable_attrs is not None else []) if c in self.dirty_csv.columns]
        extra_reliable = _extra_reliable_attrs_from_detection(
            columns=self.original_columns,
            detection_dictionary=self.detection_dictionary,
            n_reliable_attrs=n_reliable_attrs,
        )

        self.reliable_attrs: List[str] = []
        if self.use_index_partition:
            self.reliable_attrs.append(INDEX_COL)
        for c in base_reliable + extra_reliable:
            if c not in self.reliable_attrs and c in self.dirty_csv.columns:
                self.reliable_attrs.append(c)

        self.dirty_csv.insert(0, INDEX_COL, list(range(len(self.dirty_csv))))
        self.rep_work = copy.deepcopy(self.dirty_csv)

        # Repair field control:
        # - repair_attrs=None keeps the original SCAREd behavior: every non-reliable
        #   original data column is flexible and may be predicted/repaired.
        # - repair_attrs=[...] limits prediction/repair to specific columns, which
        #   is safer for DataPrep experiments and reduces over-repair.
        if repair_attrs is None:
            self.flexible_attrs = [
                attr for attr in self.original_columns
                if attr not in self.reliable_attrs and attr != INDEX_COL
            ]
        else:
            self.flexible_attrs = []
            for attr in repair_attrs:
                if attr in self.original_columns and attr not in self.reliable_attrs and attr not in self.flexible_attrs:
                    self.flexible_attrs.append(attr)

        self.rs: Dict[int, Dict[Tuple[int, int], Candidate]] = {}
        self.rep_cells: List[Tuple[int, int]] = []
        self.debug_info: Dict[str, Any] = {
            "reliable_attrs": list(self.reliable_attrs),
            "flexible_attrs": list(self.flexible_attrs),
            "num_detection_cells": len(self.detection_dictionary),
            "num_wrong_cells": len(self.wrong_cells),
            "num_partitions": 0,
            "num_rs_records": 0,
            "num_candidate_entries": 0,
            "num_changed_candidates": 0,
            "repair_attrs": list(repair_attrs) if repair_attrs is not None else None,
            "use_perfect_detection_if_clean": bool(use_perfect_detection_if_clean),
            "apply_only_detected_without_mask": bool(apply_only_detected and self.detection_mask is None),
        }

    def find_order(self):
        return list(self.original_columns)

    def partition(self) -> List[List[pd.DataFrame]]:
        """
        Strictly follow teacher script:
            for attr in reliable_attrs:
                for every value of attr:
                    create a partition with dirty_csv[attr] == value
        """
        df_partitions: List[List[pd.DataFrame]] = []
        for attr in self.reliable_attrs:
            if attr not in self.dirty_csv.columns:
                continue
            attr_card = list(dict(self.dirty_csv[attr].value_counts(dropna=False)).keys())
            if self.max_partition_values is not None:
                attr_card = attr_card[: self.max_partition_values]
            df_partition = []
            for val in attr_card:
                part = self.dirty_csv[self.dirty_csv[attr] == val].copy()
                if len(part) >= self.min_partition_size:
                    df_partition.append(part)
            df_partitions.append(df_partition)
        self.debug_info["num_partitions"] = sum(len(x) for x in df_partitions)
        return df_partitions

    def get_model(self, data: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
        """
        Teacher script trains a chain of MultinomialNB models:
            model_i predicts flexible_attrs[i]
            input features = reliable_attrs + previously predicted flexible attrs.
        """
        models: Dict[int, Dict[str, Any]] = {}
        r_attrs = copy.deepcopy(self.reliable_attrs)
        f_attrs = copy.deepcopy(self.flexible_attrs)
        for i, f_attr in enumerate(f_attrs):
            model: Dict[str, Any] = {}
            enc_x = _make_one_hot_encoder()
            enc_y = LabelEncoder()

            X = enc_x.fit_transform(data[r_attrs].astype(str).values)
            y_raw = data[f_attr].astype(str).values
            classes = pd.unique(y_raw)

            model["xenc"] = enc_x
            model["yenc"] = enc_y.fit(y_raw if len(y_raw) else [NAN_TOKEN])

            if len(classes) <= 1:
                # sklearn versions differ on fitting single-class NB. This fallback preserves behavior.
                value = str(classes[0]) if len(classes) else NAN_TOKEN
                model["constant"] = True
                model["value"] = value
                model["model"] = _ConstantModel(value)
            else:
                y = model["yenc"].transform(y_raw)
                clf = MultinomialNB(fit_prior=True)
                clf.fit(X, y)
                model["constant"] = False
                model["model"] = clf

            r_attrs.append(f_attr)
            models[i] = model
        return models

    def _predict_one(self, model: Dict[str, Any], x_data: pd.Series) -> Tuple[str, float]:
        X = model["xenc"].transform(x_data.astype(str).values.reshape(1, -1))
        if model.get("constant", False):
            return str(model["value"]), 1.0
        pred = model["model"].predict(X)
        proba = model["model"].predict_proba(X)[0]
        y_pred = str(model["yenc"].inverse_transform(pred)[0])
        y_prob = float(np.max(proba))
        return y_pred, y_prob

    def _prob_of_value(self, model: Dict[str, Any], x_data: pd.Series, y_value: str) -> float:
        X = model["xenc"].transform(x_data.astype(str).values.reshape(1, -1))
        y_value = str(y_value)
        if model.get("constant", False):
            return 1.0 if y_value == str(model["value"]) else 1e-12
        # If unseen label, the original script would error. For DataPrep robustness, assign tiny prob.
        try:
            y_idx = model["yenc"].transform(np.array([y_value]))[0]
        except ValueError:
            return 1e-12
        proba = model["model"].predict_proba(X)[0]
        if y_idx >= len(proba):
            return 1e-12
        return float(proba[y_idx])

    def get_ori_prob(
        self,
        data: pd.DataFrame,
        i: int,
        ori_prob: float,
        models: Dict[int, Dict[str, Any]],
        r_set: List[str],
        f_set: List[str],
        k_cur: int,
    ) -> float:
        """Teacher script's recursive original probability."""
        if len(f_set) == k_cur:
            return float(ori_prob)
        f_attr = f_set[k_cur]
        x_data = data[r_set].iloc[i]
        y_data = data[f_attr].iloc[i]
        y_prob = self._prob_of_value(models[k_cur], x_data, y_data) * float(ori_prob)
        r_set_temp = copy.deepcopy(r_set)
        r_set_temp.append(f_attr)
        return self.get_ori_prob(data, i, y_prob, models, r_set_temp, f_set, k_cur + 1)

    def get_all_preds(self, data: pd.DataFrame, models: Dict[int, Dict[str, Any]], i_idx: int, j_idx: int) -> None:
        for i in range(len(data)):
            idx = int(data[self.reliable_attrs].iloc[i, 0])
            if idx not in self.rs:
                self.rs[idx] = {}
            tuple_prob = self.get_ori_prob(data, i, 1.0, models, self.reliable_attrs, self.flexible_attrs, 0)
            self._get_single_preds(data, i, 1.0, models, self.reliable_attrs, self.flexible_attrs, 0, i_idx, j_idx, tuple_prob)

    def _is_detected_wrong_in_partition(self, data: pd.DataFrame, row_pos: int, f_attr: str) -> bool:
        if self.detection_mask is None:
            return False
        global_idx = int(data[INDEX_COL].iloc[row_pos]) if INDEX_COL in data.columns else row_pos
        if f_attr not in self.detection_mask.columns:
            return False
        return bool(self.detection_mask.reset_index(drop=True).at[global_idx, f_attr])

    def _get_single_preds(
        self,
        data: pd.DataFrame,
        i: int,
        ori_prob: float,
        models: Dict[int, Dict[str, Any]],
        r_set: List[str],
        f_set: List[str],
        k_cur: int,
        i_idx: int,
        j_idx: int,
        tuple_prob: float,
    ) -> None:
        """
        Preserve teacher script recursion:
            1. recurse with current/original path
            2. if predicted value differs, recurse with predicted path
               - if perfected=True, only allow predicted path for detected/wrong cells
        """
        if len(f_set) == k_cur:
            com_prob = tuple_prob
            weight = float(ori_prob) * len(data) / max(len(self.dirty_csv), 1)
            storage = Candidate(com_prob=com_prob, values=data[r_set].iloc[i].copy(), weight=weight, source=(i_idx, j_idx))
            idx = int(data[r_set].iloc[i, 0])
            old = self.rs.setdefault(idx, {}).get((i_idx, j_idx))
            if old is None or weight > old.weight:
                self.rs[idx][(i_idx, j_idx)] = storage
            return None

        f_attr = f_set[k_cur]
        x_data = data[r_set].iloc[i]
        y_data = str(data[f_attr].iloc[i])
        y_pred, y_prob = self._predict_one(models[k_cur], x_data)

        r_set_temp = copy.deepcopy(r_set)
        r_set_temp.append(f_attr)
        next_k = k_cur + 1

        # Teacher script passes y_prob, not ori_prob * y_prob.
        self._get_single_preds(data, i, y_prob, models, r_set_temp, f_set, next_k, i_idx, j_idx, tuple_prob)

        if y_pred != y_data:
            allow_predicted_branch = True
            if self.perfected:
                allow_predicted_branch = self._is_detected_wrong_in_partition(data, i, f_attr)
            if allow_predicted_branch:
                data_copy = copy.deepcopy(data)
                data_copy.iloc[i, list(data_copy.columns).index(f_attr)] = y_pred
                self.debug_info["num_changed_candidates"] += 1
                self._get_single_preds(data_copy, i, y_prob, models, r_set_temp, f_set, next_k, i_idx, j_idx, tuple_prob)

    def get_final_pred(self, assign: Dict[Tuple[int, int], Candidate]) -> Dict[str, str]:
        """
        Teacher script's KHS/KPG candidate aggregation:
            build edges among candidate values across flexible attributes,
            then keep one value per flexible attribute.
        """
        attr_val_edge: Dict[str, Dict[str, Dict[Tuple[str, str], float]]] = {}
        attr_val_val: Dict[str, Dict[str, float]] = {}
        for attr in self.flexible_attrs:
            attr_val_edge[attr] = {}
            attr_val_val[attr] = {}

        for key in assign.keys():
            for i in range(len(self.flexible_attrs)):
                data = assign[key].values[self.flexible_attrs]
                attr_i = self.flexible_attrs[i]
                val_i = str(data[attr_i])
                node_i = (attr_i, val_i)

                # Ensure singleton flexible case also has a candidate vote.
                attr_val_val.setdefault(attr_i, {})
                attr_val_val[attr_i][val_i] = attr_val_val[attr_i].get(val_i, 0.0) + float(assign[key].weight)

                for j in range(i + 1, len(self.flexible_attrs)):
                    attr_j = self.flexible_attrs[j]
                    val_j = str(data[attr_j])
                    node_j = (attr_j, val_j)
                    weight = float(assign[key].weight)

                    if val_i not in attr_val_edge[attr_i]:
                        attr_val_edge[attr_i][val_i] = {}
                    attr_val_edge[attr_i][val_i][node_j] = attr_val_edge[attr_i][val_i].get(node_j, 0.0) + weight

                    if val_j not in attr_val_edge[attr_j]:
                        attr_val_edge[attr_j][val_j] = {}
                    attr_val_edge[attr_j][val_j][node_i] = attr_val_edge[attr_j][val_j].get(node_i, 0.0) + weight

        # Teacher script computes weighted degree from edges.
        for attr in attr_val_edge.keys():
            for val in attr_val_edge[attr].keys():
                attr_val_val.setdefault(attr, {})
                attr_val_val[attr][val] = attr_val_val[attr].get(val, 0.0)
                for node in attr_val_edge[attr][val].keys():
                    attr_val_val[attr][val] = attr_val_val[attr][val] + attr_val_edge[attr][val][node]

        if not any(attr_val_val.values()):
            return {}

        final_select = self.khs_solution(attr_val_val, attr_val_edge)
        update: Dict[str, str] = {}
        for key in final_select.keys():
            update[key[0]] = key[1]
        return update

    def khs_solution(
        self,
        attr_val_val: Dict[str, Dict[str, float]],
        attr_val_edge: Dict[str, Dict[str, Dict[Tuple[str, str], float]]],
    ) -> Dict[Tuple[str, str], float]:
        attr_val_dict: Dict[Tuple[str, str], float] = {}
        for attr in attr_val_val.keys():
            for val in attr_val_val[attr].keys():
                node = (attr, val)
                attr_val_dict[node] = attr_val_val[attr][val]

        guard = 0
        while not self.end_condition(attr_val_edge) and guard < 100000:
            guard += 1
            sorted_node = sorted(attr_val_dict.items(), key=lambda kv: (kv[1], kv[0]))
            del_node = None
            for node in sorted_node:
                # Keep at least one value per attribute.
                if len(attr_val_val.get(node[0][0], {})) == 1:
                    continue
                del_node = copy.deepcopy(node[0])
                break

            if del_node is None:
                break

            del_attr, del_val = del_node
            for node in list(attr_val_edge.get(del_attr, {}).get(del_val, {}).keys()):
                other_attr, other_val = node
                edge_weight = attr_val_edge[del_attr][del_val][node]
                if other_val in attr_val_edge.get(other_attr, {}):
                    attr_val_dict[(other_attr, other_val)] = attr_val_dict.get((other_attr, other_val), 0.0) - edge_weight
                    attr_val_edge[other_attr][other_val].pop((del_attr, del_val), None)

            attr_val_edge.get(del_attr, {}).pop(del_val, None)
            attr_val_val.get(del_attr, {}).pop(del_val, None)
            attr_val_dict.pop((del_attr, del_val), None)

        return attr_val_dict

    def end_condition(self, attr_val_edge: Dict[str, Dict[str, Dict[Tuple[str, str], float]]]) -> bool:
        for key in attr_val_edge:
            if len(attr_val_edge[key]) != 1:
                return False
        return True

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df_partitions = self.partition()

        # Phase 1: candidate storage generation.
        for i in range(len(df_partitions)):
            for j in range(len(df_partitions[i])):
                part = df_partitions[i][j]
                if len(part) == 0:
                    continue
                models = self.get_model(part)
                self.get_all_preds(part, models, i, j)

        self.debug_info["num_rs_records"] = len(self.rs)
        self.debug_info["num_candidate_entries"] = sum(len(v) for v in self.rs.values())

        # Phase 2: conduct repair.
        repair_mask = pd.DataFrame(False, index=self.dirty_csv.index, columns=self.dirty_csv.columns)
        for i in range(len(self.dirty_csv)):
            assign = self.rs.get(i, {})
            if not assign:
                continue
            fin_select = self.get_final_pred(assign)
            for key in fin_select.keys():
                if key not in self.rep_work.columns:
                    continue

                if self.apply_only_detected:
                    # Safe behavior: if the user asks to repair only detected cells
                    # but no detection mask is supplied, no cell is considered detected.
                    # This prevents accidentally falling back to full-table repair.
                    if self.detection_mask is None:
                        continue
                    if key not in self.detection_mask.columns or not bool(self.detection_mask.reset_index(drop=True).at[i, key]):
                        continue

                old_val = str(self.dirty_csv.iloc[i, list(self.rep_work.columns).index(key)])
                new_val = str(fin_select[key])
                self.rep_work.iloc[i, list(self.rep_work.columns).index(key)] = new_val

                if old_val != new_val:
                    # Map work-column index to output-column index.
                    if key in self.original_columns:
                        self.rep_cells.append((i, self.original_columns.index(key)))
                        repair_mask.at[i, key] = True

        repaired = self.rep_work.drop(columns=[INDEX_COL], errors="ignore")
        repaired = repaired[self.original_columns]
        repaired.index = self.original_index

        repair_mask = repair_mask.drop(columns=[INDEX_COL], errors="ignore")
        repair_mask = repair_mask.reindex(columns=self.original_columns, fill_value=False)
        repair_mask.index = self.original_index

        self.debug_info["changed_cells"] = int(repair_mask.values.sum())
        return repaired.astype(str), repair_mask.astype(bool)


def generate_repairs(
    dirty_df: pd.DataFrame,
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
    return_mask: bool = False,
    return_debug: bool = False,
):
    cleaner = SCAREdCleaner(
        dirty_df=dirty_df,
        clean_df=clean_df,
        detection_mask=detection_mask,
        reliable_attrs=reliable_attrs,
        n_reliable_attrs=n_reliable_attrs,
        perfected=perfected,
        use_perfect_detection_if_clean=use_perfect_detection_if_clean,
        apply_only_detected=apply_only_detected,
        repair_attrs=repair_attrs,
        min_partition_size=min_partition_size,
        max_partition_values=max_partition_values,
        use_index_partition=use_index_partition,
    )
    repaired, mask = cleaner.run()

    if return_debug and return_mask:
        return repaired, mask, cleaner.debug_info
    if return_debug:
        return repaired, cleaner.debug_info
    if return_mask:
        return repaired, mask
    return repaired


def detect_errors(
    dirty_df: pd.DataFrame,
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
) -> pd.DataFrame:
    repaired, mask = generate_repairs(
        dirty_df=dirty_df,
        clean_df=clean_df,
        detection_mask=detection_mask,
        reliable_attrs=reliable_attrs,
        n_reliable_attrs=n_reliable_attrs,
        perfected=perfected,
        use_perfect_detection_if_clean=use_perfect_detection_if_clean,
        apply_only_detected=False,
        repair_attrs=repair_attrs,
        min_partition_size=min_partition_size,
        max_partition_values=max_partition_values,
        use_index_partition=use_index_partition,
        return_mask=True,
    )
    diff = _normalize_df(repaired).ne(_normalize_df(dirty_df))
    diff.index = dirty_df.index
    diff.columns = dirty_df.columns
    return diff.astype(bool)


# ---------------------------------------------------------------------------
# Compatibility utilities kept from the teacher's original scared.py
# ---------------------------------------------------------------------------


def check_string(string: str):
    """Keep the original dataset-noise suffix helper for script compatibility."""
    string = str(string)
    import re
    if re.search(r"-inner_error-", string):
        return "-inner_error-" + string[-6:-4]
    if re.search(r"-outer_error-", string):
        return "-outer_error-" + string[-6:-4]
    if re.search(r"-inner_outer_error-", string):
        return "-inner_outer_error-" + string[-6:-4]
    if re.search(r"-dirty-original_error-", string):
        return "-original_error-" + string[-9:-4]
    return ""


def handler(signum, frame):
    raise TimeoutError("Time exceeded")


def _scared_evaluation(self) -> Dict[str, float]:
    """DataPrep-safe version of the original evaluation method.

    It returns metrics instead of writing fixed-path result files and redirecting
    stdout.  The method is attached to SCAREdCleaner for original-name coverage.
    """
    rep_cells = set(self.rep_cells)
    wrong_cells = set(self.wrong_cells)
    det_right = len(rep_cells & wrong_cells)
    detection_precision = det_right / (len(rep_cells) + 1e-10)
    detection_recall = det_right / (len(wrong_cells) + 1e-10)
    detection_f1 = 2 * detection_precision * detection_recall / (detection_precision + detection_recall + 1e-10)

    repair_right = 0
    recall_right = 0
    if self.clean_csv is not None:
        rep_no_index = self.rep_work.drop(columns=[INDEX_COL], errors="ignore")
        rep_no_index = rep_no_index.reindex(columns=self.original_columns)
        clean = self.clean_csv.reindex(index=rep_no_index.index, columns=rep_no_index.columns)
        for i, j in rep_cells:
            if j < len(self.original_columns):
                col = self.original_columns[j]
                if str(rep_no_index.at[i, col]) == str(clean.at[i, col]):
                    repair_right += 1
        for i, j in wrong_cells:
            if j < len(self.original_columns):
                col = self.original_columns[j]
                if str(rep_no_index.at[i, col]) == str(clean.at[i, col]):
                    recall_right += 1
    repair_precision = repair_right / (len(rep_cells) + 1e-10)
    repair_recall = recall_right / (len(wrong_cells) + 1e-10)
    repair_f1 = 2 * repair_precision * repair_recall / (repair_precision + repair_recall + 1e-10)
    return {
        "detection_precision": detection_precision,
        "detection_recall": detection_recall,
        "detection_f1": detection_f1,
        "repair_precision": repair_precision,
        "repair_recall": repair_recall,
        "repair_f1": repair_f1,
        "changed_cells": len(rep_cells),
        "true_errors": len(wrong_cells),
    }


def _scared_model_quality(self, data, model, x_data, y_data):
    """Original model-quality stub: partition representativeness."""
    return len(data) / max(len(self.dirty_csv), 1)


def _scared_get_cond_entropy(self, data, xname, yname):
    xs = data[xname].unique()
    p_x = data[xname].value_counts() / max(data.shape[0], 1)
    ce = 0.0
    for x in xs:
        ce += float(p_x[x]) * float(self._getEntropy(data[data[xname] == x][yname]))
    return ce


def _scared_get_entropy(self, data):
    if not isinstance(data, pd.Series):
        data = pd.Series(data)
    if len(data) == 0:
        return 0.0
    prt_ary = data.groupby(data).count().values / float(len(data))
    return float(sum(-(np.log2(prt_ary) * prt_ary)))


# Attach original method names that were evaluation/debug utilities in the script.
SCAREdCleaner.evaluation = _scared_evaluation
SCAREdCleaner._model_quality = _scared_model_quality
SCAREdCleaner._getCondEntropy = _scared_get_cond_entropy
SCAREdCleaner._getEntropy = _scared_get_entropy


class SCAREd(SCAREdCleaner):
    """Compatibility class with the teacher's original class name.

    Accepts either DataFrames or CSV paths.  New DataPrep code should use the
    wrapper in ``tabular/correction/SCAREd.py``; this class exists so every
    original class/function name remains available after refactoring.
    """
    def __init__(self, csv_d, csv_c=None, reliable_attrs=None, **kwargs):
        dirty_df = pd.read_csv(csv_d) if isinstance(csv_d, (str, bytes)) else csv_d
        clean_df = pd.read_csv(csv_c) if isinstance(csv_c, (str, bytes)) else csv_c
        super().__init__(dirty_df=dirty_df, clean_df=clean_df, reliable_attrs=reliable_attrs or [], **kwargs)


__all__ = [
    "check_string", "handler", "Candidate", "SCAREd", "SCAREdCleaner",
    "build_mask_from_clean", "mask_to_detection_dictionary", "generate_repairs",
    "detect_errors",
]

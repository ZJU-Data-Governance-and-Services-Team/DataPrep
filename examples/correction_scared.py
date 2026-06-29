# -*- coding: utf-8 -*-
"""
Example script for SCAREd correction in DataPrep.

Recommended usage after the SCAREd refactor:
1. clean_path is used for evaluation only by default. It is NOT passed into the model.
2. repair_attrs can restrict which columns are allowed to be repaired.
3. apply_only_detected should be used together with --detection_mask_path or
   --use_perfect_detection_mask. Otherwise, no cells will be repaired in the safe refactor.
"""

import os
import sys
import time
import argparse
import inspect
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from dataprep.tabular.correction.SCAREd import SCAREd
except ModuleNotFoundError:
    from tabular.correction.SCAREd import SCAREd


def parse_columns(text):
    """Parse comma-separated columns. Empty string means None."""
    if text is None:
        return None
    text = str(text).strip()
    if text == "":
        return None
    text = text.replace("，", ",")
    cols = [x.strip() for x in text.split(",") if x.strip()]
    return cols if cols else None


def normalize_df(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.fillna("null").astype(str)
    for col in df.columns:
        df[col] = df[col].map(lambda x: str(x).strip())
    return df


def read_csv_safely(path, index_col=None):
    for enc in ["utf-8-sig", "utf-8", "gbk"]:
        try:
            return pd.read_csv(path, encoding=enc, index_col=index_col)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, index_col=index_col)


def build_perfect_detection_mask(df_dirty, df_clean):
    """dirty != clean. This is oracle/perfect mask; use only for upper-bound experiments."""
    if df_clean is None:
        return None
    d = normalize_df(df_dirty)
    c = normalize_df(df_clean).reindex(index=d.index, columns=d.columns)
    return d.ne(c).astype(bool)


def read_detection_mask(path, df_dirty):
    """Read a boolean detection mask csv and align it to dirty_df."""
    mask = read_csv_safely(path)
    mask.columns = [str(c).strip() for c in mask.columns]
    mask = mask.reindex(index=df_dirty.index, columns=df_dirty.columns).fillna(False)

    def to_bool(x):
        s = str(x).strip().lower()
        return s in {"true", "1", "yes", "y", "t"}

    return mask.applymap(to_bool).astype(bool)


def evaluate_correction(name, df_clean, df_dirty, df_corrected, time_cost):
    df_clean = normalize_df(df_clean)
    df_dirty = normalize_df(df_dirty)
    df_corrected = normalize_df(df_corrected)

    common_idx = df_clean.index.intersection(df_dirty.index).intersection(df_corrected.index)
    common_col = df_clean.columns.intersection(df_dirty.columns).intersection(df_corrected.columns)

    gt = df_clean.loc[common_idx, common_col]
    dirty = df_dirty.loc[common_idx, common_col]
    pred = df_corrected.loc[common_idx, common_col]

    gt_error_mask = dirty.ne(gt)
    repaired_error_mask = pred.ne(gt)
    changed_mask = pred.ne(dirty)

    total_errors = int(gt_error_mask.values.sum())
    fixed_count = int((gt_error_mask & pred.eq(gt)).values.sum())
    changed_count = int(changed_mask.values.sum())
    new_errors = int((changed_mask & ~gt_error_mask & pred.ne(gt)).values.sum())

    repair_precision = fixed_count / changed_count if changed_count > 0 else 0.0
    repair_recall = fixed_count / total_errors if total_errors > 0 else 0.0
    repair_f1 = (
        2 * repair_precision * repair_recall / (repair_precision + repair_recall)
        if (repair_precision + repair_recall) > 0
        else 0.0
    )

    return {
        "Model": name,
        "Repair Precision": f"{repair_precision:.2%}",
        "Repair Recall": f"{repair_recall:.2%}",
        "Repair F1": f"{repair_f1:.2%}",
        "Fixed/Total": f"{fixed_count}/{total_errors}",
        "Dirty Error Cells": total_errors,
        "Repaired Error Cells": int(repaired_error_mask.values.sum()),
        "Changed Cells": changed_count,
        "New Errors": new_errors,
        "Time(s)": round(time_cost, 2),
    }


def save_changed_detail(df_clean, df_dirty, df_corrected, save_path):
    df_clean = normalize_df(df_clean)
    df_dirty = normalize_df(df_dirty)
    df_corrected = normalize_df(df_corrected)

    common_idx = df_clean.index.intersection(df_dirty.index).intersection(df_corrected.index)
    common_col = df_clean.columns.intersection(df_dirty.columns).intersection(df_corrected.columns)

    gt = df_clean.loc[common_idx, common_col]
    dirty = df_dirty.loc[common_idx, common_col]
    pred = df_corrected.loc[common_idx, common_col]

    records = []
    for i in common_idx:
        for col in common_col:
            dirty_val = str(dirty.loc[i, col]).strip()
            pred_val = str(pred.loc[i, col]).strip()
            gt_val = str(gt.loc[i, col]).strip()

            if dirty_val != pred_val:
                records.append({
                    "row": i,
                    "column": col,
                    "dirty_value": dirty_val,
                    "repaired_value": pred_val,
                    "clean_value": gt_val,
                    "is_dirty_error": dirty_val != gt_val,
                    "is_correct": pred_val == gt_val,
                    "is_new_error": dirty_val == gt_val and pred_val != gt_val,
                })

    detail_df = pd.DataFrame(records)
    detail_df.to_csv(save_path, index=False, encoding="utf-8-sig")
    return detail_df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirty_path", required=True, help="Path to dirty.csv")
    parser.add_argument("--clean_path", default=None, help="Path to clean.csv, used for evaluation only by default")
    parser.add_argument("--output_dir", default="./temp/runs_scared_correction", help="Output directory")
    parser.add_argument("--reliable_attrs", default="", help="Comma-separated reliable attrs, e.g. provider_number,measure_code")
    parser.add_argument("--repair_attrs", default="", help="Comma-separated repairable attrs. Empty means all non-reliable attrs")
    parser.add_argument("--detection_mask_path", default=None, help="Optional external detection mask csv, e.g. Horizon error mask")
    parser.add_argument("--use_perfect_detection_mask", action="store_true", help="Build oracle mask from clean.csv and pass it as detection_mask")
    parser.add_argument("--use_clean_in_model", action="store_true", help="Pass clean_df into SCAREd internals. Not recommended for fair experiments")
    parser.add_argument("--index_col", default=None, help="If CSV first column is index, set to 0")
    parser.add_argument("--apply_only_detected", action="store_true", help="Only modify cells marked by detection mask")
    parser.add_argument("--no_index_partition", action="store_true", help="Disable Index partition")
    # Kept for backward compatibility. Default is already no perfect detection.
    parser.add_argument("--no_perfect_detection", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    index_col = None
    if args.index_col is not None:
        try:
            index_col = int(args.index_col)
        except ValueError:
            index_col = args.index_col

    repaired_path = os.path.join(args.output_dir, "repaired_scared.csv")
    result_path = os.path.join(args.output_dir, "scared_correction_result.csv")
    detail_path = os.path.join(args.output_dir, "scared_changed_detail.csv")
    repair_mask_path = os.path.join(args.output_dir, "scared_repair_mask.csv")
    used_detection_mask_path = os.path.join(args.output_dir, "scared_used_detection_mask.csv")
    debug_path = os.path.join(args.output_dir, "scared_debug_info.csv")

    print("Loading Data...")
    df_dirty = normalize_df(read_csv_safely(args.dirty_path, index_col=index_col)).reset_index(drop=True)

    df_clean = None
    if args.clean_path is not None:
        df_clean = normalize_df(read_csv_safely(args.clean_path, index_col=index_col)).reset_index(drop=True)
        min_len = min(len(df_clean), len(df_dirty))
        df_clean = df_clean.iloc[:min_len].reset_index(drop=True)
        df_dirty = df_dirty.iloc[:min_len].reset_index(drop=True)

    reliable_attrs = parse_columns(args.reliable_attrs)
    repair_attrs = parse_columns(args.repair_attrs)

    for name, cols in [("reliable_attr", reliable_attrs), ("repair_attr", repair_attrs)]:
        if cols is not None:
            for attr in cols:
                if attr not in df_dirty.columns:
                    raise ValueError(f"{name} not found in dirty.csv columns: {attr}")

    detection_mask = None
    if args.detection_mask_path is not None:
        detection_mask = read_detection_mask(args.detection_mask_path, df_dirty)
    elif args.use_perfect_detection_mask:
        if df_clean is None:
            raise ValueError("--use_perfect_detection_mask requires --clean_path")
        detection_mask = build_perfect_detection_mask(df_dirty, df_clean)

    if detection_mask is not None:
        detection_mask.to_csv(used_detection_mask_path, index=False, encoding="utf-8-sig")

    if args.apply_only_detected and detection_mask is None:
        print("[WARN] --apply_only_detected is set, but no detection mask is provided. Safe refactor will repair 0 cells.")

    print(f"Data ready. Shape: {df_dirty.shape}")
    print(f"Reliable attrs from CLI: {reliable_attrs}")
    print(f"Repair attrs from CLI  : {repair_attrs}")
    print(f"Clean df provided      : {df_clean is not None}")
    print(f"Clean passed to model  : {args.use_clean_in_model}")
    print(f"Detection mask provided: {detection_mask is not None}")
    print(f"Apply only detected    : {args.apply_only_detected}")

    print("\n>>> Running SCAREd (Correction)...")
    start_time = time.time()

    init_params = inspect.signature(SCAREd.__init__).parameters
    kwargs = {}
    if "reliable_attrs" in init_params:
        kwargs["reliable_attrs"] = reliable_attrs
    if "apply_only_detected" in init_params:
        kwargs["apply_only_detected"] = args.apply_only_detected
    if "repair_attrs" in init_params:
        kwargs["repair_attrs"] = repair_attrs
    if "detection_mask" in init_params and detection_mask is not None:
        kwargs["detection_mask"] = detection_mask
    if "clean_df" in init_params and args.use_clean_in_model and df_clean is not None:
        kwargs["clean_df"] = df_clean
    if "use_perfect_detection_if_clean" in init_params:
        # Default false. Perfect mask should be passed explicitly as detection_mask.
        kwargs["use_perfect_detection_if_clean"] = False
    if "perfected" in init_params:
        kwargs["perfected"] = False
    if "use_index_partition" in init_params:
        kwargs["use_index_partition"] = (not args.no_index_partition)

    model = SCAREd(**kwargs)

    try:
        # Fair default: do not pass clean_df into model training.
        repaired_df = model.train_and_predict(df_dirty)
    except TypeError:
        train_kwargs = {}
        if detection_mask is not None:
            train_kwargs["detection_mask"] = detection_mask
        if repair_attrs is not None:
            train_kwargs["repair_attrs"] = repair_attrs
        if args.use_clean_in_model and df_clean is not None:
            train_kwargs["clean_df"] = df_clean
        repaired_df = model.train_and_predict(df_dirty, **train_kwargs)

    cost = time.time() - start_time

    if not isinstance(repaired_df, pd.DataFrame):
        repaired_df = pd.DataFrame(repaired_df, columns=df_dirty.columns)

    repaired_df = normalize_df(repaired_df).reset_index(drop=True)
    repaired_df.to_csv(repaired_path, index=False, encoding="utf-8-sig")

    if hasattr(model, "repair_mask_") and model.repair_mask_ is not None:
        repair_mask = model.repair_mask_
        if not isinstance(repair_mask, pd.DataFrame):
            repair_mask = pd.DataFrame(repair_mask, columns=df_dirty.columns)
        repair_mask = repair_mask.reindex(index=df_dirty.index, columns=df_dirty.columns).fillna(False).astype(bool)
    else:
        repair_mask = repaired_df.astype(str).ne(df_dirty.astype(str))

    repair_mask.to_csv(repair_mask_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print(" SCAREd Correction Result")
    print("=" * 80)

    if df_clean is not None:
        result = evaluate_correction(
            name="SCAREd",
            df_clean=df_clean,
            df_dirty=df_dirty,
            df_corrected=repaired_df,
            time_cost=cost,
        )
        result_df = pd.DataFrame([result])
        result_df.to_csv(result_path, index=False, encoding="utf-8-sig")
        print(result_df.to_string(index=False))

        detail_df = save_changed_detail(
            df_clean=df_clean,
            df_dirty=df_dirty,
            df_corrected=repaired_df,
            save_path=detail_path,
        )
        print("-" * 80)
        print(f"Result saved to       : {result_path}")
        print(f"Changed detail saved  : {detail_path}")
        print(f"Changed cells detail  : {len(detail_df)}")
    else:
        changed_cells = int(repair_mask.values.sum())
        print(f"Changed Cells: {changed_cells}")
        print(f"Time(s): {round(cost, 2)}")
        print("clean_path not provided, skip repair accuracy evaluation.")

    print("-" * 80)
    print(f"Repaired data saved to: {repaired_path}")
    print(f"Repair mask saved to  : {repair_mask_path}")
    if detection_mask is not None:
        print(f"Used detection mask   : {used_detection_mask_path}")

    if hasattr(model, "debug_info_") and model.debug_info_ is not None:
        print("-" * 80)
        print("Debug info:")
        print(model.debug_info_)
        pd.DataFrame([model.debug_info_]).to_csv(debug_path, index=False, encoding="utf-8-sig")
        print(f"Debug info saved to   : {debug_path}")

    print("-" * 80)
    print("\nSCAREd correction finished.")

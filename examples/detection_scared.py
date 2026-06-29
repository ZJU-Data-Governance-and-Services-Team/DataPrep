# -*- coding: utf-8 -*-
"""
Example script for SCAREd detection in DataPrep.

SCAREd is repair-first:
    detection mask = repaired_df != dirty_df

After the refactor:
1. clean_path is used for evaluation only by default.
2. repair_attrs can restrict SCAREd's repair/detection search space.
3. clean_df is not passed into the model unless --use_clean_in_model is explicitly set.
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
    from dataprep.tabular.detection.SCAREd import SCAREd as SCAREdDetector
except ModuleNotFoundError:
    try:
        from tabular.detection.SCAREd import SCAREd as SCAREdDetector
    except ModuleNotFoundError:
        SCAREdDetector = None

try:
    from dataprep.tabular.correction.SCAREd import SCAREd as SCAREdRepair
except ModuleNotFoundError:
    from tabular.correction.SCAREd import SCAREd as SCAREdRepair


def parse_columns(text):
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
    if df_clean is None:
        return None
    d = normalize_df(df_dirty)
    c = normalize_df(df_clean).reindex(index=d.index, columns=d.columns)
    return d.ne(c).astype(bool)


def read_detection_mask(path, df_dirty):
    mask = read_csv_safely(path)
    mask.columns = [str(c).strip() for c in mask.columns]
    mask = mask.reindex(index=df_dirty.index, columns=df_dirty.columns).fillna(False)

    def to_bool(x):
        s = str(x).strip().lower()
        return s in {"true", "1", "yes", "y", "t"}

    return mask.applymap(to_bool).astype(bool)


def evaluate_detection(df_clean, df_dirty, pred_mask):
    df_clean = normalize_df(df_clean)
    df_dirty = normalize_df(df_dirty)
    pred_mask = pred_mask.reindex(index=df_dirty.index, columns=df_dirty.columns).fillna(False).astype(bool)

    gt_mask = df_dirty.ne(df_clean.reindex(index=df_dirty.index, columns=df_dirty.columns)).astype(bool)

    tp = int((pred_mask & gt_mask).values.sum())
    fp = int((pred_mask & ~gt_mask).values.sum())
    fn = int((~pred_mask & gt_mask).values.sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "Model": "SCAREd",
        "Precision": f"{precision:.2%}",
        "Recall": f"{recall:.2%}",
        "F1": f"{f1:.2%}",
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Dirty Error Cells": int(gt_mask.values.sum()),
        "Detected Cells": int(pred_mask.values.sum()),
    }


def save_detected_detail(df_clean, df_dirty, pred_mask, save_path):
    df_clean = normalize_df(df_clean) if df_clean is not None else None
    df_dirty = normalize_df(df_dirty)
    pred_mask = pred_mask.reindex(index=df_dirty.index, columns=df_dirty.columns).fillna(False).astype(bool)

    records = []
    for i in pred_mask.index:
        for col in pred_mask.columns:
            if bool(pred_mask.loc[i, col]):
                dirty_val = str(df_dirty.loc[i, col])
                clean_val = str(df_clean.loc[i, col]) if df_clean is not None and col in df_clean.columns else ""
                is_true_error = dirty_val != clean_val if df_clean is not None else None
                records.append({
                    "row": i,
                    "column": col,
                    "dirty_value": dirty_val,
                    "clean_value": clean_val,
                    "is_true_error": is_true_error,
                })
    detail_df = pd.DataFrame(records)
    detail_df.to_csv(save_path, index=False, encoding="utf-8-sig")
    return detail_df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirty_path", required=True, help="Path to dirty.csv")
    parser.add_argument("--clean_path", default=None, help="Path to clean.csv, used for evaluation only by default")
    parser.add_argument("--output_dir", default="./temp/runs_scared_detection", help="Output directory")
    parser.add_argument("--reliable_attrs", default="", help="Comma-separated reliable attrs, e.g. provider_number,measure_code")
    parser.add_argument("--repair_attrs", default="", help="Comma-separated repairable attrs. Empty means all non-reliable attrs")
    parser.add_argument("--detection_mask_path", default=None, help="Optional external detection mask csv, used only if wrapper supports detection_mask")
    parser.add_argument("--use_perfect_detection_mask", action="store_true", help="Build oracle mask from clean.csv and pass it as detection_mask")
    parser.add_argument("--use_clean_in_model", action="store_true", help="Pass clean_df into SCAREd internals. Not recommended for fair experiments")
    parser.add_argument("--index_col", default=None, help="If CSV first column is index, set to 0")
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

    mask_path = os.path.join(args.output_dir, "scared_detection_mask.csv")
    result_path = os.path.join(args.output_dir, "scared_detection_result.csv")
    detail_path = os.path.join(args.output_dir, "scared_detected_detail.csv")
    repaired_path = os.path.join(args.output_dir, "scared_detection_repaired.csv")
    used_detection_mask_path = os.path.join(args.output_dir, "scared_used_detection_mask.csv")
    debug_path = os.path.join(args.output_dir, "scared_detection_debug_info.csv")

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

    print(f"Data ready. Shape: {df_dirty.shape}")
    print(f"Reliable attrs from CLI: {reliable_attrs}")
    print(f"Repair attrs from CLI  : {repair_attrs}")
    print(f"Clean df provided      : {df_clean is not None}")
    print(f"Clean passed to model  : {args.use_clean_in_model}")
    print(f"Detection mask provided: {detection_mask is not None}")

    print("\n>>> Running SCAREd (Detection)...")
    start_time = time.time()

    model_cls = SCAREdDetector if SCAREdDetector is not None else SCAREdRepair
    init_params = inspect.signature(model_cls.__init__).parameters

    kwargs = {}
    if "reliable_attrs" in init_params:
        kwargs["reliable_attrs"] = reliable_attrs
    if "repair_attrs" in init_params:
        kwargs["repair_attrs"] = repair_attrs
    if "detection_mask" in init_params and detection_mask is not None:
        kwargs["detection_mask"] = detection_mask
    if "clean_df" in init_params and args.use_clean_in_model and df_clean is not None:
        kwargs["clean_df"] = df_clean
    if "use_perfect_detection_if_clean" in init_params:
        kwargs["use_perfect_detection_if_clean"] = False
    if "perfected" in init_params:
        kwargs["perfected"] = False
    if "use_index_partition" in init_params:
        kwargs["use_index_partition"] = (not args.no_index_partition)

    model = model_cls(**kwargs)

    try:
        output = model.train_and_predict(df_dirty)
    except TypeError:
        train_kwargs = {}
        if detection_mask is not None:
            train_kwargs["detection_mask"] = detection_mask
        if repair_attrs is not None:
            train_kwargs["repair_attrs"] = repair_attrs
        if args.use_clean_in_model and df_clean is not None:
            train_kwargs["clean_df"] = df_clean
        output = model.train_and_predict(df_dirty, **train_kwargs)

    repaired_df = None
    if isinstance(output, pd.DataFrame) and output.shape == df_dirty.shape:
        values = set(str(x).strip().lower() for x in pd.unique(output.astype(str).values.ravel()))
        if values.issubset({"true", "false", "0", "1"}):
            pred_mask = output.replace({"True": True, "False": False, "true": True, "false": False, "1": True, "0": False}).astype(bool)
            repaired_df = getattr(model, "repaired_df_", None)
        else:
            repaired_df = normalize_df(output).reset_index(drop=True)
            pred_mask = repaired_df.astype(str).ne(df_dirty.astype(str))
    else:
        repaired_df = getattr(model, "repaired_df_", None)
        pred_mask = pd.DataFrame(output, index=df_dirty.index, columns=df_dirty.columns).astype(bool)

    if repaired_df is None and hasattr(model, "repaired_df_") and model.repaired_df_ is not None:
        repaired_df = normalize_df(model.repaired_df_).reset_index(drop=True)

    elapsed = time.time() - start_time

    pred_mask = pred_mask.reindex(index=df_dirty.index, columns=df_dirty.columns).fillna(False).astype(bool)
    pred_mask.to_csv(mask_path, index=False, encoding="utf-8-sig")

    if repaired_df is not None:
        normalize_df(repaired_df).to_csv(repaired_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print(" SCAREd Detection Result")
    print("=" * 80)

    if df_clean is not None:
        result = evaluate_detection(df_clean, df_dirty, pred_mask)
        result["Time(s)"] = round(elapsed, 2)
        result_df = pd.DataFrame([result])
        result_df.to_csv(result_path, index=False, encoding="utf-8-sig")
        print(result_df.to_string(index=False))

        detail_df = save_detected_detail(df_clean, df_dirty, pred_mask, detail_path)
        print("-" * 80)
        print(f"Result saved to       : {result_path}")
        print(f"Detected detail saved : {detail_path}")
        print(f"Detected cells detail : {len(detail_df)}")
    else:
        print(f"Detected Cells: {int(pred_mask.values.sum())}")
        print(f"Time(s): {round(elapsed, 2)}")
        print("clean_path not provided, skip detection evaluation.")

    print("-" * 80)
    print(f"Detection mask saved to: {mask_path}")
    if repaired_df is not None:
        print(f"Repaired data saved to : {repaired_path}")
    if detection_mask is not None:
        print(f"Used detection mask    : {used_detection_mask_path}")

    if hasattr(model, "debug_info_") and model.debug_info_ is not None:
        print("-" * 80)
        print("Debug info:")
        print(model.debug_info_)
        pd.DataFrame([model.debug_info_]).to_csv(debug_path, index=False, encoding="utf-8-sig")
        print(f"Debug info saved to    : {debug_path}")

    print("-" * 80)
    print("\nSCAREd detection finished.")

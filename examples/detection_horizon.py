# -*- coding: utf-8 -*-
import os
import sys
import time
import argparse
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from dataprep.tabular.detection.Horizon import Horizon
except ModuleNotFoundError:
    from tabular.detection.Horizon import Horizon


# 1. 自动发现简单 FD：只生成 Horizon 当前支持的 A⇒B
def discover_simple_fds(df_clean, rule_path, exclude_cols=None, allow_unique_lhs=False):
    df = df_clean.fillna("nan").astype(str)
    exclude_cols = set(exclude_cols or ["index", "Unnamed: 0"])

    cols = [c for c in df.columns if c not in exclude_cols]
    fds = []

    for lhs in cols:
        lhs_unique_num = df[lhs].nunique(dropna=False)

        # 默认跳过唯一值列，避免 id⇒所有列 这种无意义 FD
        if not allow_unique_lhs and lhs_unique_num == len(df):
            continue

        # 跳过常量列作为左部
        if lhs_unique_num <= 1:
            continue

        for rhs in cols:
            if lhs == rhs:
                continue

            # 跳过常量列作为右部
            if df[rhs].nunique(dropna=False) <= 1:
                continue

            grouped = df.groupby(lhs, dropna=False)[rhs].nunique(dropna=False)

            if grouped.max() <= 1:
                fds.append((lhs, rhs))

    fds = sorted(set(fds))

    os.makedirs(os.path.dirname(rule_path), exist_ok=True)
    with open(rule_path, "w", encoding="utf-8") as f:
        for lhs, rhs in fds:
            f.write(f"{lhs}⇒{rhs}\n")

    return fds


# 2. 评估函数：用于计算检测 Precision / Recall / F1
def evaluate_detection(name, df_clean, df_dirty, pred_mask, time_cost):
    df_clean = df_clean.fillna("nan").astype(str)
    df_dirty = df_dirty.fillna("nan").astype(str)

    common_idx = df_clean.index.intersection(df_dirty.index).intersection(pred_mask.index)
    common_col = df_clean.columns.intersection(df_dirty.columns).intersection(pred_mask.columns)

    clean = df_clean.loc[common_idx, common_col]
    dirty = df_dirty.loc[common_idx, common_col]
    pred = pred_mask.loc[common_idx, common_col].astype(bool)

    # 真实错误位置
    true_mask = dirty != clean

    tp = int((pred & true_mask).sum().sum())
    fp = int((pred & ~true_mask).sum().sum())
    fn = int((~pred & true_mask).sum().sum())
    tn = int((~pred & ~true_mask).sum().sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "Model": name,
        "Precision": f"{precision:.2%}",
        "Recall": f"{recall:.2%}",
        "F1": f"{f1:.2%}",
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Detected Cells": int(pred.sum().sum()),
        "True Error Cells": int(true_mask.sum().sum()),
        "Time(s)": round(time_cost, 2)
    }


# 3. 保存检测明细
def save_detected_detail(df_dirty, pred_mask, save_path, df_clean=None):
    df_dirty = df_dirty.fillna("nan").astype(str)
    pred_mask = pred_mask.astype(bool)

    if df_clean is not None:
        df_clean = df_clean.fillna("nan").astype(str)

    common_idx = df_dirty.index.intersection(pred_mask.index)
    common_col = df_dirty.columns.intersection(pred_mask.columns)

    records = []

    for i in common_idx:
        for col in common_col:
            if pred_mask.loc[i, col]:
                record = {
                    "row": i,
                    "column": col,
                    "dirty_value": df_dirty.loc[i, col]
                }

                if df_clean is not None and col in df_clean.columns and i in df_clean.index:
                    clean_value = str(df_clean.loc[i, col]).strip()
                    dirty_value = str(df_dirty.loc[i, col]).strip()

                    record["clean_value"] = clean_value
                    record["is_true_error"] = dirty_value != clean_value

                records.append(record)

    detail_df = pd.DataFrame(records)
    detail_df.to_csv(save_path, index=False, encoding="utf-8-sig")

    return detail_df


# 4. 主程序
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dirty_path", required=True, help="dirty.csv 路径")
    parser.add_argument("--clean_path", default=None, help="clean.csv 路径；用于自动发现 FD 和评价 detection")
    parser.add_argument("--rule_path", default=None, help="FD 约束文件路径；不传则从 clean.csv 自动生成")
    parser.add_argument("--output_dir", default="./temp/runs_horizon_detection", help="结果输出目录")
    parser.add_argument("--allow_unique_lhs", action="store_true", help="是否允许唯一值列作为 FD 左部")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.rule_path is None:
        args.rule_path = os.path.join(args.output_dir, "constraints_auto.txt")

    mask_path = os.path.join(args.output_dir, "error_mask_horizon.csv")
    result_path = os.path.join(args.output_dir, "horizon_detection_result.csv")
    detail_path = os.path.join(args.output_dir, "horizon_detected_detail.csv")

    # 加载数据
    print("Loading Data...")
    df_dirty = pd.read_csv(args.dirty_path).fillna("nan").astype(str)

    df_clean = None
    if args.clean_path is not None:
        df_clean = pd.read_csv(args.clean_path).fillna("nan").astype(str)

        min_len = min(len(df_clean), len(df_dirty))
        df_clean = df_clean.iloc[:min_len].reset_index(drop=True)
        df_dirty = df_dirty.iloc[:min_len].reset_index(drop=True)
    else:
        df_dirty = df_dirty.reset_index(drop=True)

    print(f"Data ready. Shape: {df_dirty.shape}")

    # 自动生成 FD
    if not os.path.exists(args.rule_path):
        if df_clean is None:
            raise ValueError("未提供 rule_path 时，必须提供 clean_path 用于自动发现 FD。")

        print("\n>>> Discovering FD Constraints...")
        fds = discover_simple_fds(
            df_clean=df_clean,
            rule_path=args.rule_path,
            allow_unique_lhs=args.allow_unique_lhs
        )

        print(f"FD rules generated: {len(fds)}")
        print(f"Rule file: {args.rule_path}")

        for lhs, rhs in fds[:30]:
            print(f"{lhs}⇒{rhs}")

        if len(fds) > 30:
            print(f"... {len(fds) - 30} more rules")
    else:
        print(f"\n>>> Using existing FD rule file: {args.rule_path}")

    # 运行 Horizon detection
    print("\n>>> Running Horizon (FD-based Detection)...")
    start_time = time.time()

    detector = Horizon(rule_path=args.rule_path)
    error_mask = detector.train_and_predict(df_dirty)

    cost = time.time() - start_time

    # 统一 mask 格式
    error_mask = error_mask.astype(bool)
    error_mask = error_mask.reset_index(drop=True)

    # 保存 error mask
    error_mask.to_csv(mask_path, index=False, encoding="utf-8-sig")

    # 保存检测明细
    detail_df = save_detected_detail(
        df_dirty=df_dirty,
        pred_mask=error_mask,
        save_path=detail_path,
        df_clean=df_clean
    )

    print("\n" + "=" * 80)
    print(" Horizon Detection Result")
    print("=" * 80)

    if df_clean is not None:
        result = evaluate_detection(
            "Horizon",
            df_clean,
            df_dirty,
            error_mask,
            cost
        )

        result_df = pd.DataFrame([result])
        result_df.to_csv(result_path, index=False, encoding="utf-8-sig")

        print(result_df.to_string(index=False))
        print("-" * 80)
        print(f"Detection result saved to: {result_path}")
    else:
        print("clean_path not provided, skip detection evaluation.")
        print(f"Detected cells: {int(error_mask.sum().sum())}")
        print(f"Time(s): {round(cost, 2)}")
        print("-" * 80)

    print(f"Error mask saved to     : {mask_path}")
    print(f"Detected detail saved to: {detail_path}")
    print(f"Detected detail number  : {len(detail_df)}")
    print("-" * 80)

    print("\nHorizon detection finished.")
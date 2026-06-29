# -*- coding: utf-8 -*-
import os
import sys
import time
import argparse
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))

# 如果脚本放在 examples/ 下，项目根目录就是 examples 的上一层
project_root = os.path.abspath(os.path.join(current_dir, ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from dataprep.tabular.correction.Horizon import Horizon
except ModuleNotFoundError:
    from tabular.correction.Horizon import Horizon


# 1. 自动发现简单 FD：只生成 Horizon 当前支持的 A⇒B
def discover_simple_fds(df_clean, rule_path, exclude_cols=None, allow_unique_lhs=False):
    df = df_clean.fillna("nan").astype(str)
    exclude_cols = set(exclude_cols or ["index", "Unnamed: 0"])

    cols = [c for c in df.columns if c not in exclude_cols]
    fds = []

    for lhs in cols:
        # 默认跳过唯一值比例过高的列，避免 id⇒所有列 这类无意义 FD
        if not allow_unique_lhs and df[lhs].nunique(dropna=False) == len(df):
            continue

        # 跳过常量列作为左部
        if df[lhs].nunique(dropna=False) <= 1:
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


# 2. 评估函数：用于计算修复准确率
def evaluate_correction(name, df_clean, df_dirty, df_corrected, time_cost):
    df_clean = df_clean.fillna("nan").astype(str)
    df_dirty = df_dirty.fillna("nan").astype(str)
    df_corrected = df_corrected.fillna("nan").astype(str)

    common_idx = df_clean.index.intersection(df_dirty.index).intersection(df_corrected.index)
    common_col = df_clean.columns.intersection(df_dirty.columns).intersection(df_corrected.columns)

    gt = df_clean.loc[common_idx, common_col]
    dirty = df_dirty.loc[common_idx, common_col]
    pred = df_corrected.loc[common_idx, common_col]

    # 真实错误位置：dirty 和 clean 不一致的位置
    mask = dirty != gt

    y_true_vals = gt.values[mask.values]
    y_pred_vals = pred.values[mask.values]

    total_errors = len(y_true_vals)
    correct_count = 0

    for t, p in zip(y_true_vals, y_pred_vals):
        if str(t).strip() == str(p).strip():
            correct_count += 1

    accuracy = correct_count / total_errors if total_errors > 0 else 0

    dirty_error_cells = int((dirty != gt).sum().sum())
    repaired_error_cells = int((pred != gt).sum().sum())
    changed_cells = int((pred != dirty).sum().sum())

    return {
        "Model": name,
        "Accuracy (Repair)": f"{accuracy:.2%}",
        "Fixed/Total": f"{correct_count}/{total_errors}",
        "Dirty Error Cells": dirty_error_cells,
        "Repaired Error Cells": repaired_error_cells,
        "Changed Cells": changed_cells,
        "Time(s)": round(time_cost, 2)
    }


# 3. 保存修复明细
def save_changed_detail(df_clean, df_dirty, df_corrected, save_path):
    df_clean = df_clean.fillna("nan").astype(str)
    df_dirty = df_dirty.fillna("nan").astype(str)
    df_corrected = df_corrected.fillna("nan").astype(str)

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
                    "is_correct": pred_val == gt_val
                })

    detail_df = pd.DataFrame(records)
    detail_df.to_csv(save_path, index=False, encoding="utf-8-sig")
    return detail_df


# 4. 主程序
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--clean_path", required=True, help="clean.csv 路径")
    parser.add_argument("--dirty_path", required=True, help="dirty.csv 路径")
    parser.add_argument("--rule_path", default=None, help="FD 约束文件路径；不传则自动生成")
    parser.add_argument("--output_dir", default="./temp/runs_horizon", help="结果输出目录")
    parser.add_argument("--allow_unique_lhs", action="store_true", help="是否允许唯一值列作为 FD 左部")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.rule_path is None:
        args.rule_path = os.path.join(args.output_dir, "constraints_auto.txt")

    repaired_path = os.path.join(args.output_dir, "repaired_horizon.csv")
    result_path = os.path.join(args.output_dir, "horizon_result.csv")
    detail_path = os.path.join(args.output_dir, "horizon_changed_detail.csv")

    # 加载数据
    print("Loading Data...")
    df_clean = pd.read_csv(args.clean_path).fillna("nan").astype(str)
    df_dirty = pd.read_csv(args.dirty_path).fillna("nan").astype(str)

    min_len = min(len(df_clean), len(df_dirty))
    df_clean = df_clean.iloc[:min_len].reset_index(drop=True)
    df_dirty = df_dirty.iloc[:min_len].reset_index(drop=True)

    print(f"Data ready. Shape: {df_dirty.shape}")

    # 自动生成 FD
    if not os.path.exists(args.rule_path):
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

    # 运行 Horizon
    print("\n>>> Running Horizon (FD-based Correction)...")
    start_time = time.time()

    horizon = Horizon(rule_path=args.rule_path)
    df_fixed_horizon = horizon.train_and_predict(df_dirty)

    cost = time.time() - start_time

    # 保存修复结果
    df_fixed_horizon.to_csv(repaired_path, index=False, encoding="utf-8-sig")

    # 评估结果
    result = evaluate_correction(
        "Horizon",
        df_clean,
        df_dirty,
        df_fixed_horizon,
        cost
    )

    result_df = pd.DataFrame([result])
    result_df.to_csv(result_path, index=False, encoding="utf-8-sig")

    # 保存修复明细
    detail_df = save_changed_detail(
        df_clean,
        df_dirty,
        df_fixed_horizon,
        detail_path
    )

    print("\n" + "=" * 80)
    print(" Horizon Correction Result")
    print("=" * 80)
    print(result_df.to_string(index=False))
    print("-" * 80)

    print(f"Repaired data saved to: {repaired_path}")
    print(f"Result saved to       : {result_path}")
    print(f"Changed detail saved  : {detail_path}")
    print(f"Changed cells detail  : {len(detail_df)}")
    print("-" * 80)
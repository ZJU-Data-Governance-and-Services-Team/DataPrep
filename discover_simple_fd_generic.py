# -*- coding: utf-8 -*-
"""
通用版：从 clean.csv 自动发现简单函数依赖 FD，并生成 Horizon 可读取的约束文件。

适用范围：
1. 任意 CSV 文件；
2. 只发现单属性 FD：A⇒B；
3. 不写死具体字段名；
4. 自动跳过无意义 FD，例如：
   - 常量列作为左部；
   - 行号 / 唯一 ID 类字段默认不作为左部；
   - 常量列作为右部默认跳过；
5. 输出格式兼容 Horizon：
   A⇒B

基本用法：
python discover_simple_fd_generic.py --clean clean.csv --out constraints_auto.txt

查看所有候选 FD 统计：
python discover_simple_fd_generic.py --clean clean.csv --out constraints_auto.txt --report fd_report.csv

如果你确实想允许唯一 ID 作为左部：
python discover_simple_fd_generic.py --clean clean.csv --out constraints_auto.txt --allow_unique_lhs
"""

import argparse
import os
import pandas as pd


def parse_col_list(text):
    """
    把命令行中的列名字符串转成列表。
    例如：
        "index,Unnamed: 0,id"
    转成：
        ["index", "Unnamed: 0", "id"]
    """
    if text is None or str(text).strip() == "":
        return []
    return [x.strip() for x in str(text).split(",") if x.strip()]


def read_csv_safely(path):
    """
    尽量稳妥地读取 CSV。
    """
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        try:
            return pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="gbk")


def normalize_dataframe(df, nan_token="nan"):
    """
    统一数据格式：
    1. 列名转字符串并去掉首尾空格；
    2. 缺失值填充；
    3. 所有值转字符串；
    4. 每个单元格去掉首尾空格。
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.fillna(nan_token).astype(str)

    for col in df.columns:
        df[col] = df[col].map(lambda x: str(x).strip())

    return df


def is_fd_holds(df, lhs, rhs, max_violation_ratio=0.0):
    """
    判断 lhs⇒rhs 是否成立。

    严格 FD：
        对每个 lhs 取值，rhs 只能有一个取值。

    近似 FD：
        如果 max_violation_ratio > 0，允许少量 lhs 分组违反。
    """
    nunique_per_lhs = df.groupby(lhs, dropna=False)[rhs].nunique(dropna=False)

    total_groups = len(nunique_per_lhs)
    violating_groups = int((nunique_per_lhs > 1).sum())

    if total_groups == 0:
        return False, 0, 0, 1.0

    violation_ratio = violating_groups / total_groups
    holds = violation_ratio <= max_violation_ratio

    return holds, violating_groups, total_groups, violation_ratio


def discover_simple_fds(
    df,
    exclude_cols=None,
    include_cols=None,
    skip_constant_lhs=True,
    skip_constant_rhs=True,
    allow_unique_lhs=False,
    max_lhs_unique_ratio=0.95,
    min_lhs_repeat_groups=2,
    max_violation_ratio=0.0,
):
    """
    通用单属性 FD 发现。

    参数说明：
    exclude_cols:
        不参与 FD 发现的列，比如 index、Unnamed: 0。

    include_cols:
        如果指定，则只在这些列中发现 FD。

    skip_constant_lhs:
        是否跳过常量列作为 FD 左部。

    skip_constant_rhs:
        是否跳过常量列作为 FD 右部。

    allow_unique_lhs:
        是否允许唯一值列作为左部。
        默认 False，因为唯一 ID 会导致 ID⇒所有列，这通常没有清洗意义。

    max_lhs_unique_ratio:
        左部列唯一值比例过高时跳过。
        默认 0.95，表示如果某列 95% 以上值都不同，则默认不作为左部。

    min_lhs_repeat_groups:
        左部至少要有多少个重复分组。
        例如 A 列中至少有 2 个取值重复出现，才认为 A 有发现 FD 的意义。

    max_violation_ratio:
        允许的违反比例。
        0 表示严格 FD。
    """

    exclude_cols = set(exclude_cols or [])
    include_cols = set(include_cols) if include_cols else None

    all_cols = list(df.columns)

    cols = []
    for col in all_cols:
        if col in exclude_cols:
            continue
        if include_cols is not None and col not in include_cols:
            continue
        cols.append(col)

    n_rows = len(df)

    if n_rows == 0:
        return [], pd.DataFrame()

    nunique_map = {
        col: int(df[col].nunique(dropna=False))
        for col in cols
    }

    constant_cols = {
        col for col in cols
        if nunique_map[col] <= 1
    }

    fds = []
    report_records = []

    for lhs in cols:
        lhs_nunique = nunique_map[lhs]
        lhs_unique_ratio = lhs_nunique / max(n_rows, 1)

        if skip_constant_lhs and lhs in constant_cols:
            continue

        if not allow_unique_lhs:
            if lhs_nunique == n_rows:
                continue
            if lhs_unique_ratio > max_lhs_unique_ratio:
                continue

        lhs_group_sizes = df.groupby(lhs, dropna=False).size()
        repeat_group_count = int((lhs_group_sizes >= 2).sum())

        if repeat_group_count < min_lhs_repeat_groups:
            continue

        for rhs in cols:
            if lhs == rhs:
                continue

            rhs_nunique = nunique_map[rhs]
            rhs_unique_ratio = rhs_nunique / max(n_rows, 1)

            if skip_constant_rhs and rhs in constant_cols:
                continue

            holds, violating_groups, total_groups, violation_ratio = is_fd_holds(
                df=df,
                lhs=lhs,
                rhs=rhs,
                max_violation_ratio=max_violation_ratio
            )

            if holds:
                fds.append((lhs, rhs))

                report_records.append({
                    "lhs": lhs,
                    "rhs": rhs,
                    "lhs_nunique": lhs_nunique,
                    "rhs_nunique": rhs_nunique,
                    "lhs_unique_ratio": round(lhs_unique_ratio, 6),
                    "rhs_unique_ratio": round(rhs_unique_ratio, 6),
                    "lhs_repeat_group_count": repeat_group_count,
                    "total_lhs_groups": total_groups,
                    "violating_groups": violating_groups,
                    "violation_ratio": round(violation_ratio, 6),
                })

    report_df = pd.DataFrame(report_records)

    if not report_df.empty:
        report_df = report_df.sort_values(
            by=[
                "violation_ratio",
                "lhs_unique_ratio",
                "lhs_repeat_group_count",
                "rhs_nunique"
            ],
            ascending=[True, True, False, False]
        ).reset_index(drop=True)

    fds = [(row["lhs"], row["rhs"]) for _, row in report_df.iterrows()] if not report_df.empty else []

    return fds, report_df


def write_fd_file(fds, out_path):
    """
    写出 Horizon 可读取的 FD 文件。
    注意：中间必须是 ⇒
    """
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for lhs, rhs in fds:
            f.write(f"{lhs}⇒{rhs}\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--clean",
        required=True,
        help="clean.csv 路径"
    )

    parser.add_argument(
        "--out",
        required=True,
        help="输出 FD 约束文件路径，例如 constraints_auto.txt"
    )

    parser.add_argument(
        "--report",
        default=None,
        help="可选：输出 FD 统计报告 CSV，例如 fd_report.csv"
    )

    parser.add_argument(
        "--exclude_cols",
        default="index,Unnamed: 0",
        help="不参与 FD 发现的列，逗号分隔。默认：index,Unnamed: 0"
    )

    parser.add_argument(
        "--include_cols",
        default="",
        help="只在指定列中发现 FD，逗号分隔。默认空，表示使用所有列"
    )

    parser.add_argument(
        "--allow_unique_lhs",
        action="store_true",
        help="允许唯一值列作为 FD 左部。默认不允许"
    )

    parser.add_argument(
        "--keep_constant_rhs",
        action="store_true",
        help="保留常量列作为 FD 右部。默认跳过常量右部"
    )

    parser.add_argument(
        "--max_lhs_unique_ratio",
        type=float,
        default=0.95,
        help="左部列唯一值比例阈值，超过则默认跳过。默认 0.95"
    )

    parser.add_argument(
        "--min_lhs_repeat_groups",
        type=int,
        default=2,
        help="左部列至少需要多少个重复取值分组。默认 2"
    )

    parser.add_argument(
        "--max_violation_ratio",
        type=float,
        default=0.0,
        help="允许的 FD 违反分组比例。默认 0，表示严格 FD"
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=0,
        help="只输出前 K 条 FD。默认 0 表示全部输出"
    )

    args = parser.parse_args()

    print("========== Loading Clean Data ==========")
    df = read_csv_safely(args.clean)
    df = normalize_dataframe(df)

    print("clean path:", args.clean)
    print("shape:", df.shape)

    exclude_cols = parse_col_list(args.exclude_cols)
    include_cols = parse_col_list(args.include_cols)

    print("\n========== Discovering Simple FDs ==========")

    fds, report_df = discover_simple_fds(
        df=df,
        exclude_cols=exclude_cols,
        include_cols=include_cols,
        skip_constant_lhs=True,
        skip_constant_rhs=not args.keep_constant_rhs,
        allow_unique_lhs=args.allow_unique_lhs,
        max_lhs_unique_ratio=args.max_lhs_unique_ratio,
        min_lhs_repeat_groups=args.min_lhs_repeat_groups,
        max_violation_ratio=args.max_violation_ratio,
    )

    if args.top_k and args.top_k > 0:
        fds = fds[:args.top_k]
        report_df = report_df.iloc[:args.top_k].reset_index(drop=True)

    write_fd_file(fds, args.out)

    if args.report:
        report_dir = os.path.dirname(os.path.abspath(args.report))
        if report_dir and not os.path.exists(report_dir):
            os.makedirs(report_dir, exist_ok=True)
        report_df.to_csv(args.report, index=False, encoding="utf-8-sig")

    print("发现 FD 数量:", len(fds))
    print("FD 文件已保存:", args.out)

    if args.report:
        print("FD 报告已保存:", args.report)

    print("\n========== Preview ==========")
    preview_num = min(30, len(fds))
    for lhs, rhs in fds[:preview_num]:
        print(f"{lhs}⇒{rhs}")

    if len(fds) > preview_num:
        print(f"... 还有 {len(fds) - preview_num} 条未显示")


if __name__ == "__main__":
    main()
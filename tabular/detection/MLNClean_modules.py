"""
MLNClean 共享底层模块

来源:
    Zhao et al. "Cleaning Uncertain Data via Markov Logic Networks."
    原始代码: rules_partition/{Rule, Heap, index_construction}.py
              data_cleaning/{AGP, RSC, FCSR}.py
              weight_learning/Weight.py

依赖:
    - pandas, numpy           (必装)
    - python-Levenshtein      (AGP 阶段做异常组合并要用)
    - pyro-ppl, torch         (MLN 权重学习要用 MCMC)
"""
import math
import sys
import random
import re
from collections.abc import Iterable
from functools import partial

import numpy as np
import pandas as pd


# =============================================================================
# 1. 二叉堆 (来自原 rules_partition/Heap.py, 用于 data_partition)
# =============================================================================

class BinaryHeap(object):
    """小顶堆, 堆中元素是 [value, payload] 形式"""

    def __init__(self, max_size=math.inf):
        self._heap = [[-math.inf, 0]]
        self.max_size = max_size

    def __len__(self):
        return len(self._heap) - 1

    def insert(self, *data):
        if isinstance(data[0], Iterable):
            if len(data) > 1:
                return
            data = data[0]
        if not len(self) + len(data) < self.max_size:
            return
        for x in data:
            self._heap.append(x)
            self._siftup()

    def _siftup(self):
        pos = len(self)
        x = self._heap[-1]
        while x[0] < self._heap[pos >> 1][0]:
            self._heap[pos] = self._heap[pos >> 1]
            pos >>= 1
        self._heap[pos] = x

    def delete_min(self):
        if not len(self):
            return
        _min = self._heap[1]
        last = self._heap.pop()
        if len(self):
            self._heap[1] = last
            self._siftdown(1)
        return _min

    def _siftdown(self, idx):
        temp = self._heap[idx]
        length = len(self)
        while True:
            child_idx = idx << 1
            if child_idx > length:
                break
            if child_idx != length and self._heap[child_idx][0] > self._heap[child_idx + 1][0]:
                child_idx += 1
            if temp[0] > self._heap[child_idx][0]:
                self._heap[idx] = self._heap[child_idx]
            else:
                break
            idx = child_idx
        self._heap[idx] = temp

    def get_min(self):
        return self._heap[1][0] if len(self) else None

    def get_size(self):
        return bool(len(self))


# =============================================================================
# 2. 规则解析 (来自原 rules_partition/index_construction.py)
# =============================================================================

def parse_rules(rules_text_list):
    """
    把 MLN 规则文本解析成 [(reason_cols, result_cols), ...] 列表

    规则格式: "!flight(...) v act_arr_time(...)"
    意思: 如果 flight 同, 则 act_arr_time 同; 前者带 ! 是 reason, 后者是 result.

    Args:
        rules_text_list: list[str], 每行一条规则
    Returns:
        block_pair: [(reason_list, result_list), ...]
    """
    block_pair = []
    for rule in rules_text_list:
        rule = rule.strip()
        if not rule:
            continue
        rule_list = rule.split(' v ')
        reason_list = []
        result_list = []
        for attribute in rule_list:
            idx = attribute.index('(')
            if attribute[0] == '!':
                reason_list.append(attribute[1:idx])
            else:
                result_list.append(attribute[:idx])
        block_pair.append([reason_list, result_list])
    return block_pair


def block_into_group(data, pair_cols, reason_pair):
    """按 reason_pair 列对 data 分组, 给每行打 'group' 编号. 返回选取 pair_cols 列后的 DataFrame."""
    data = data.copy()
    data["group"] = data.groupby(reason_pair).ngroup()
    return data[pair_cols]


# =============================================================================
# 3. 数据分区 (来自原 rules_partition/Rule.py)
# =============================================================================

def _dist(a, b):
    """两行间不相等元素数 (汉明距离)"""
    return int(np.sum(a != b))


def data_partition(df, partition_num=1, random_seed=None):
    """
    把 df 按汉明距离启发式平分成 partition_num 组.
        1. 随机选 partition_num 个质心；
        2. 每个分区最多 max_per_pkg 行；
        3. 每行优先放到距离最近的质心分区；
        4. 如果最近分区已满，且当前行比该分区中最远的旧行更近，则替换旧行，并把旧行重新分配；
        5. 所有行必须且只能出现一次。
    """
    if partition_num <= 1:
        return [df.copy()]

    if random_seed is not None:
        random.seed(random_seed)

    rows = df.values
    n = rows.shape[0]
    max_per_pkg = n // partition_num + 1

    centroid_idx = random.sample(range(n), partition_num)
    heaps = []
    for i in centroid_idx:
        h = BinaryHeap()
        h.insert([[0, i]])
        heaps.append(h)

    for i in range(n):
        if i in centroid_idx:
            continue
        cur_i = i
        while cur_i is not None:
            dist_to_c = [_dist(rows[cur_i], rows[j]) for j in centroid_idx]
            placed = False
            while not placed:
                min_dist = min(dist_to_c)
                min_idx = dist_to_c.index(min_dist)
                if len(heaps[min_idx]) < max_per_pkg:
                    heaps[min_idx].insert([[-min_dist, cur_i]])
                    placed = True
                    cur_i = None
                else:
                    top_element = heaps[min_idx].get_min()
                    # 堆里存的是 [-distance, row_idx]，
                    if top_element is not None and (-top_element) > min_dist:
                        _, old_i = heaps[min_idx].delete_min()
                        heaps[min_idx].insert([[-min_dist, cur_i]])
                        cur_i = old_i
                        placed = True
                    else:
                        dist_to_c[min_idx] = rows.shape[1] + 1

    result = []
    for h in heaps:
        row_positions = []
        while h.get_size():
            min_pair = h.delete_min()
            row_positions.append(min_pair[1])

        result.append(df.iloc[row_positions].copy())

    return result


# =============================================================================
# 4. AGP - Abnormal Group Processing (来自原 data_cleaning/AGP.py)
# =============================================================================

def process_by_AGP(data, reason_column, threshold_count=2):
    """
    把"小到不正常"的组合并到与它最相似 (编辑距离最小) 的正常组里.

    Args:
        data: 含 reason_column 和 'group' 列的 DataFrame
        reason_column: 用作 reason 的列名列表
        threshold_count: 组大小阈值, 小于此值视为异常组
    Returns:
        data: 修改后的 DataFrame (相同对象)
    """
    try:
        from Levenshtein import distance as _levenshtein_distance
    except ImportError:
        raise ImportError(
            "MLNClean AGP 阶段需要 python-Levenshtein. 安装命令: "
            "pip install python-Levenshtein"
        )

    data = data.copy()
    group_counts = data["group"].value_counts()
    abnormal_groups = group_counts[group_counts < threshold_count].index
    abnormal_groups_field = data[data["group"].isin(abnormal_groups)][reason_column]

    # 正常组中提取的待匹配模板 (reason 拼接字符串, group_id)
    result_column = list(reason_column) + ["group"]
    normal_groups = group_counts[group_counts >= threshold_count].index
    normal_rows = data[data["group"].isin(normal_groups)][result_column].values.tolist()
    # 去重
    normal_rows = list(set(tuple(r) for r in normal_rows))
    normal_rows = [list(t) for t in normal_rows]
    normal_str = [[''.join(map(str, r[:-1])), r[-1]] for r in normal_rows]

    if not normal_str:
        # 没有任何"正常组"可参照, 跳过 AGP
        return data

    # 每个异常组找最近的正常组并合并
    for idx, row in abnormal_groups_field.iterrows():
        abn_str = row.astype(str).str.cat()
        min_dist = sys.float_info.max
        min_group = None
        for gs in normal_str:
            d = _levenshtein_distance(abn_str, gs[0])
            if d < min_dist:
                min_dist = d
                min_group = gs[1]
        if min_group is None:
            continue
        # 用匹配到的正常组的 reason 值覆盖异常行
        match_row = [r for r in normal_rows if r[-1] == min_group][0]
        data.at[idx, "group"] = min_group
        for i, col in enumerate(reason_column):
            data.at[idx, col] = match_row[i]
    return data


# =============================================================================
# 5. RSC - 同组内冲突消解 + MLN 权重学习 (来自原 data_cleaning/RSC.py)
# =============================================================================

def _flatten(nested):
    """递归展平嵌套列表"""
    out = []
    for x in nested:
        if isinstance(x, list):
            out.extend(_flatten(x))
        else:
            out.append(x)
    return out


def generate_evidence(evidence_df, rules_pair):
    """
    从 evidence_df 中统计每个属性的取值分布作为 MLN 的 evidence.
    Returns:
        evidence_data:  list[list], 每个属性的取值列表
        evidence_value: list[list[float]], 对应的概率分布
        rules_set:      list[str], 涉及的所有列名(去重)
    """
    rules_set = list(set(_flatten(rules_pair)))
    n = evidence_df.shape[0]
    evidence_data, evidence_value = [], []
    for col in rules_set:
        counts = evidence_df[col].value_counts().to_dict()
        evidence_data.append(list(counts.keys()))
        evidence_value.append([v / n for v in counts.values()])
    return evidence_data, evidence_value, rules_set


def generate_rule(evidence_df, rules_pair):
    """
    从 evidence_df 中抽出每个规则涉及列的去重组合.
    Returns:
        list[pd.DataFrame], 每个 DataFrame 是该规则下属性取值的所有去重组合.
    """
    rules = []
    for pair in rules_pair:
        cols = [c for sub in pair for c in sub]
        rules.append(evidence_df[cols].drop_duplicates())
    return rules


def _find_location(var_name, var_data, evidence_data, data_header):
    indexes = [i for i, v in enumerate(data_header) if v == var_name]
    selected = evidence_data[indexes[0]]
    return selected[selected[0] == var_data].index.tolist()[0]


def _generate_weight_model(evidence, evidence_value, rules, list_header):
    """Pyro 概率模型: 联合采样所有变量和规则权重."""
    import pyro
    import pyro.distributions as dist
    import torch

    data = []
    variables = {}
    for i, name in enumerate(list_header):
        data.append(pd.DataFrame(evidence[i]))
        probs = torch.tensor(evidence_value[i])
        variables[name] = pyro.sample(name, dist.Categorical(probs=probs))

    weights = {}
    str_time3 = 0
    for str_time1, rule in enumerate(rules):
        var_name = rule.columns.tolist()
        for str_time2, (_, row) in enumerate(rule.iterrows()):
            str_time3 += 1
            key = f"f{str_time3}"
            weights[key] = pyro.sample(
                f"{str_time1}_{str_time2}",
                dist.Normal(0.99, 0.01)
            )
            var_data = row.tolist()
            reason_eq = weights[key].unsqueeze(0)
            for i, vn in enumerate(var_name):
                loc = _find_location(vn, var_data[i], data, list_header)
                if i < len(var_name) - 1:
                    reason_eq = reason_eq * (loc != variables[vn])
                else:
                    reason_eq = reason_eq * (loc == variables[vn])
            pyro.factor(key, reason_eq)


def train_mln_weights(evidence, evidence_value, rules, list_header,num_samples=20, warmup=20):
    """
    用 Pyro NUTS-MCMC 学每条规则的权重.
    Returns:
        dict, key 是规则取值的 tuple, value 是权重 tensor
    """
    try:
        import pyro
        import torch
        from pyro.infer import MCMC, NUTS
    except ImportError:
        raise ImportError(
            "MLNClean 权重学习需要 pyro-ppl 和 torch. 安装命令: "
            "pip install pyro-ppl torch"
        )

    # 避免上一次 MCMC 异常退出后 Pyro 状态残留
    pyro.clear_param_store()
    try:
        from pyro.poutine.runtime import _PYRO_STACK
        _PYRO_STACK.clear()
    except Exception:
        pass

    fn = partial(
        _generate_weight_model,
        evidence=evidence, evidence_value=evidence_value,
        rules=rules, list_header=list_header
    )
    kernel = NUTS(fn)
    mcmc = MCMC(kernel, num_samples=num_samples, warmup_steps=warmup)
    mcmc.run()
    samples = mcmc.get_samples()

    avg_weights = {}
    for key, values in samples.items():
        # key 形如 "{block}_{group}", 转回 (col_values, ...) 的 tuple
        parts = key.split('_')
        if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
            continue
        block_idx = int(parts[0])
        group_idx = int(parts[1])
        rule_row = rules[block_idx].iloc[group_idx]
        avg_weights[tuple(rule_row)] = torch.mean(values, dim=0)
    return avg_weights


def process_by_RSC(data, reason_column, result_column, weight_data):
    """
    同组内冲突消解 (R-Score Cleaning):
        对每个 group, 列出所有 (reason, result) 组合, 选权重最大的那个作为该组的 result.
    Args:
        data: 经过 AGP 处理的 block_data
        reason_column / result_column: 列名 list (注意 reason_column 最后一位通常是 'group')
        weight_data: train_mln_weights 返回的权重字典
    """
    data = data.copy()
    grouped_reason = data.groupby('group')[reason_column]
    grouped_result = data.groupby('group')[result_column]

    reasons = []
    results = []
    for _, group_data in grouped_reason:
        reasons.append(group_data[reason_column].iloc[0])
    for _, group_data in grouped_result:
        tuples = group_data[result_column].values.tolist()
        results.append(list(set(tuple(t) for t in tuples)))

    for i in range(len(reasons)):
        reason = reasons[i].tolist()
        candidates = results[i]
        if len(candidates) <= 1:
            continue

        # 选权重最大的 result
        max_w = 0
        max_tuple = None
        for single_result in candidates:
            combined = tuple(reason[:-1]) + single_result    # reason 最后一位是 group, 跳过
            w = weight_data.get(combined)
            if w is not None and w > max_w:
                max_tuple = single_result
                max_w = w

        if max_tuple is not None:
            group_idx = reason[-1]
            data.loc[data['group'] == group_idx, result_column] = max_tuple
    return data


# =============================================================================
# 6. FCSR - 跨 block 冲突聚合 (来自原 data_cleaning/FCSR.py)
# =============================================================================

def process_by_FCSR(split_data, data, weight_data, rules):
    """
    把每个规则 block 处理后的数据聚合, 解决跨 block 冲突.

    Args:
        split_data: list[pd.DataFrame], 每个规则下 AGP+RSC 后的结果
        data:       原始 DataFrame (作为输出的载体)
        rules:      generate_rule 的返回
    Returns:
        修复后的 DataFrame (drop 掉 'group' 列, 不改 ID 顺序)
    """
    data = data.copy()
    merged = pd.concat(split_data)
    sorted_data = merged.sort_values(by='ID')
    sorted_no_group = sorted_data.drop(columns='group')
    grouped = sorted_no_group.groupby('ID')

    for id_value, group_line in grouped:
        for col in group_line.columns:
            type_values = group_line[col].dropna().unique()
            type_dupes = group_line[col].dropna()

            if len(type_values) == 1:
                data.loc[data['ID'] == id_value, col] = type_values[0]
            elif len(type_values) > 1:
                # 跨 block 冲突: 按规则优先级决定
                times = 0
                for index, rule in enumerate(rules):
                    if col not in rule.columns:
                        continue
                    times += 1
                    block_values = rule[col].dropna().unique()
                    if index >= len(group_line):
                        break
                    if group_line.isnull().iloc[index, group_line.columns.get_loc(col)]:
                        continue
                    if all(elem in block_values for elem in type_values):
                        right_data = group_line.iloc[index, group_line.columns.get_loc(col)]
                        data.loc[data['ID'] == id_value, col] = right_data
                        break
                    elif times < len(type_dupes):
                        pass
                    elif times == len(type_dupes):
                        right_data = group_line.iloc[index, group_line.columns.get_loc(col)]
                        data.loc[data['ID'] == id_value, col] = right_data
                        break

    if 'group' in data.columns:
        data = data.drop(columns=['group'])
    return data


# =============================================================================
# 7. 顶层管线 (detection 和 correction 共享)
# =============================================================================

def _clean_single_partition(data_name,processed_pair,rules,weight_data,original_columns,agp_threshold=2,verbose=True,partition_idx=None):
    """
    对单个 partition 执行原版 main.py 中从 block_into_group 到 FCSR 的流程。

    对应原 main.py 逻辑：
        total_data = []
        for pair in processed_pair:
            single_pair = ...
            block_data = block_into_group(data_name, single_pair, reason_column)
            deleted_abnormal_group_data = process_by_AGP(...)
            deleted_abnormal_line_data = process_by_RSC(...)
            total_data.append(deleted_abnormal_line_data)
        result = process_by_FCSR(total_data, data_name, weight_data, rules)

    Args:
        data_name        : 当前 partition 的 DataFrame
        processed_pair   : parse_rules() / constructing_data() 的结果
        rules            : generate_rule() 的结果
        weight_data      : train_mln_weights() / mln() 的结果
        original_columns : 原始 dirty_df 的列顺序
    """
    total_data = []

    for r_idx, pair in enumerate(processed_pair):
        single_pair = [item for sublist in pair for item in sublist]
        single_pair.insert(0, "ID")
        single_pair.append("group")

        reason_column = list(pair[0])
        result_column = list(pair[1])

        block_data = block_into_group(
            data_name,
            single_pair,
            reason_column
        )

        deleted_abnormal_group_data = process_by_AGP(
            block_data,
            list(reason_column),
            threshold_count=agp_threshold
        )

        deleted_abnormal_line_data = process_by_RSC(
            deleted_abnormal_group_data,
            list(reason_column) + ["group"],
            result_column,
            weight_data
        )

        total_data.append(deleted_abnormal_line_data)

        if verbose:
            prefix = (
                f"[Partition {partition_idx}] "
                if partition_idx is not None
                else ""
            )
            print(
                f"  {prefix}[Rule {r_idx + 1}/{len(processed_pair)}] "
                f"reason={pair[0]} -> result={pair[1]}: "
                f"processed {len(deleted_abnormal_line_data)} rows"
            )

    if not total_data:
        return data_name[original_columns].copy()

    data_for_fcsr = data_name.copy()

    # 原 process_by_FCSR 最后会 drop group；
    # 当前 DataFrame 如果没有 group，补一个临时 group，保持和现有 FCSR 接口兼容。
    if "group" not in data_for_fcsr.columns:
        data_for_fcsr["group"] = 0

    result = process_by_FCSR(
        total_data,
        data_for_fcsr,
        weight_data,
        rules
    )

    return result[original_columns]

def run_mln_clean_pipeline(dirty_df, rules_text_list,evidence_df=None,partition_number=1,agp_threshold=2,mcmc_samples=20,mcmc_warmup=20,verbose=True):
    """
    完整的 MLNClean 管线：输入脏 DataFrame，输出 MLN 清洗后的 DataFrame。

    对齐原版 main.py：
        1. 读取 rules.txt
        2. data_partition
        3. constructing_data
        4. generate_evidence
        5. generate_rule
        6. mln 学 weight_data
        7. 对每个 partition：
              for pair in processed_pair:
                  block_into_group
                  AGP
                  RSC
              FCSR
        8. concat 所有 partition 的结果
    """
    if not isinstance(dirty_df, pd.DataFrame):
        raise TypeError("dirty_df 必须是 pandas DataFrame。")

    if "ID" not in dirty_df.columns:
        raise ValueError("MLNClean 期望输入 DataFrame 含 'ID' 列。")

    if rules_text_list is None:
        raise ValueError("rules_text_list 不能为空。")

    if not isinstance(rules_text_list, (list, tuple)):
        raise TypeError("rules_text_list 必须是规则字符串列表。")

    if len(rules_text_list) == 0:
        raise ValueError("rules_text_list 不能为空列表。")

    if evidence_df is None:
        raise ValueError("必须指定 evidence_df 或 evidence_path。")

    if partition_number < 1:
        raise ValueError("partition_number 必须 >= 1。")

    if verbose:
        print(
            f"[MLNClean] Input shape: {dirty_df.shape}, "
            f"rules: {len(rules_text_list)}, "
            f"partition_number: {partition_number}"
        )

    # 1. 对应原 constructing_data(rules1_list)
    processed_pair = parse_rules(rules_text_list)

    if not processed_pair:
        raise ValueError("解析后规则为空，请检查 rules_text_list 格式。")

    # 2. 检查 dirty_df / evidence_df 是否包含规则涉及列
    required_cols = list(dict.fromkeys(_flatten(processed_pair)))

    missing_dirty_cols = [
        col for col in required_cols
        if col not in dirty_df.columns
    ]
    if missing_dirty_cols:
        raise ValueError(
            f"dirty_df 缺少规则涉及列：{missing_dirty_cols}。"
        )

    missing_evidence_cols = [
        col for col in required_cols
        if col not in evidence_df.columns
    ]
    if missing_evidence_cols:
        raise ValueError(
            f"evidence_df 缺少规则涉及列：{missing_evidence_cols}。"
            "请确认 evidence_path 指向原版 rules_data.csv。"
        )

    # 3. 对应原 generate_evidence / generate_rule / mln
    #    注意：这些只依赖 rules_data.csv，不依赖某个 partition，所以只做一次。
    if verbose:
        print(
            f"[MLNClean] Phase 1: Learning MLN weights via MCMC "
            f"({mcmc_samples} samples, {mcmc_warmup} warmup)..."
        )

    evidence, evidence_value, set1 = generate_evidence(
        evidence_df,
        processed_pair
    )

    rules = generate_rule(
        evidence_df,
        processed_pair
    )

    weight_data = train_mln_weights(
        evidence,
        evidence_value,
        rules,
        set1,
        num_samples=mcmc_samples,
        warmup=mcmc_warmup,
    )

    original_columns = list(dirty_df.columns)

    # 4. 对应原 data_partition(args.data_file, args.partition_number)
    #    但 DataPrep 版本输入已经是 DataFrame，所以这里传 dirty_df。
    partitions = data_partition(
        dirty_df,
        partition_num=partition_number
    )

    cleaned_parts = []

    if verbose:
        print(
            f"[MLNClean] Phase 2: Cleaning {len(partitions)} partition(s) "
            f"over {len(processed_pair)} rule(s)..."
        )

    # 5. 按原版注释补循环：
    #    “如果分组数量大于一，将要在这里加上一个循环”
    for p_idx, part_df in enumerate(partitions):
        if part_df.empty:
            continue

        if verbose:
            print(
                f"[MLNClean] Cleaning partition {p_idx + 1}/{len(partitions)}, "
                f"shape={part_df.shape}"
            )

        cleaned_part = _clean_single_partition(
            data_name=part_df.copy(),
            processed_pair=processed_pair,
            rules=rules,
            weight_data=weight_data,
            original_columns=original_columns,
            agp_threshold=agp_threshold,
            verbose=verbose,
            partition_idx=p_idx
        )

        cleaned_parts.append(cleaned_part)

    if not cleaned_parts:
        raise RuntimeError("所有 partition 均为空，无法生成 cleaned_df。")

    # 6. 合并所有 partition 的结果
    cleaned_df = pd.concat(cleaned_parts, axis=0)

    # 如果 data_partition 保留了原始 index，则按原始 index 恢复顺序；
    # 如果当前 data_partition 重置了 index，则这里可能失败。
    # 所以下面配合 data_partition 修改为 df.iloc[row_positions].copy()。
    cleaned_df = cleaned_df.loc[dirty_df.index]

    cleaned_df = cleaned_df[original_columns]

    if verbose:
        print("[MLNClean] Pipeline complete.")

    return cleaned_df


# =============================================================================
# 8. Diff 工具 (detection 用) + 局部覆盖 (correction 用)
# =============================================================================

def compute_diff_mask(dirty_df, cleaned_df, exclude_columns=("ID",)):
    """
    比对脏数据和清洗后数据, 凡是变了的格子都标 True. 用于把 MLNClean 用作"检测器".

    Args:
        dirty_df    : 原始脏 DataFrame
        cleaned_df  : run_mln_clean_pipeline 输出
        exclude_columns: 这些列即使变了也不算 "错" (默认 'ID')
    Returns:
        mask_df : 同形 bool DataFrame, True=该格子被算法判定为错误
    """
    mask = pd.DataFrame(False, index=dirty_df.index, columns=dirty_df.columns)
    sentinel = object()

    for col in dirty_df.columns:
        if col in exclude_columns:
            continue

        a = dirty_df[col].astype(object)
        b = cleaned_df[col].astype(object)

        both_nan = a.isna() & b.isna()
        changed = ~(a.eq(b) | both_nan)

        mask[col] = changed.values

    return mask

def to_bool_mask(mask_df):
    """
    将 DataPrep detection mask 转为 bool
        1 = 错误，需要修复
        0 = 正常，不修复
    """
    mask_df = mask_df.loc[:, ~mask_df.columns.astype(str).str.startswith("Unnamed:")]

    return mask_df.replace({
        "True": 1,
        "False": 0,
        "true": 1,
        "false": 0,
    }).astype(int).astype(bool)

def apply_corrections_with_mask(dirty_df, cleaned_df, mask_df):
    """
    只在 mask=True 的格子用 cleaned 的值覆盖 dirty, 其它格子保留原值.
    用于把 MLNClean 用作"修复器".

    Args:
        dirty_df  : 原始脏 DataFrame
        cleaned_df: MLN 清洗后 DataFrame
        mask_df   : 与 dirty_df 同形的 bool DataFrame (True = 该格子要被修)
    Returns:
        fixed_df : 在 mask 位置应用了 cleaned 值的 DataFrame
    """
    fixed = dirty_df.copy()
    mask_df = to_bool_mask(mask_df)
    for col in dirty_df.columns:
        if col not in mask_df.columns or col not in cleaned_df.columns:
            continue
        m = mask_df[col].values
        fixed.loc[m, col] = cleaned_df[col].values[m]
    return fixed

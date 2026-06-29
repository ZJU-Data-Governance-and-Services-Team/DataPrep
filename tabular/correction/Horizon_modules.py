# -*- coding: utf-8 -*-
"""
Horizon repair core for DataPrep.

This module refactors the original script-style `horizon.py` into reusable
functions that accept/return pandas DataFrames.  It follows the Horizon paper's
pipeline:

    BuildFDPatternGraph -> ComputePatternQuality -> BuildSCCGraphAndSort
    -> OrderFDs -> GeneratePatternPreservingRepairs -> repaired DataFrame

Supported FD rule formats
-------------------------
1. Text, one FD per line:
       A ⇒ B
       A -> B
       A → B
   Composite LHS is accepted with commas, e.g.:
       A,B -> C

2. JSON dict:
       {"A": ["B", "C"]}
   interpreted as A -> B and A -> C.

The original Horizon code only handled single-attribute LHS rules.  This
implementation keeps that behavior as the main path and adds best-effort support
for composite LHS by treating the tuple of LHS values as one graph node.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict, deque
import json
import os
import re
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

NAN_TOKEN = "nan"
LHS_JOINER = "||"
VALUE_JOINER = "\u241f"  # visible unit separator for composite LHS values


@dataclass(frozen=True)
class FDRule:
    """Functional dependency rule: lhs attributes determine rhs attribute."""

    lhs: Tuple[str, ...]
    rhs: str

    @property
    def left(self) -> str:
        """Compatibility with the original tmporder.left for single-LHS code."""
        return self.lhs[0] if len(self.lhs) == 1 else LHS_JOINER.join(self.lhs)

    @property
    def right(self) -> str:
        return self.rhs

    @property
    def lhs_attr_key(self) -> str:
        return self.left


@dataclass
class OrderedFD:
    """FD rule with SCC order metadata."""

    lhs: Tuple[str, ...]
    rhs: str
    lnum: int = 0
    rnum: int = 0

    @property
    def left(self) -> str:
        return self.lhs[0] if len(self.lhs) == 1 else LHS_JOINER.join(self.lhs)

    @property
    def right(self) -> str:
        return self.rhs


class Vertex:
    """Node in FD pattern graph.

    A node corresponds to an (attribute, value) pair.  The original script used
    only value as key, which may merge values from different columns.  Here the
    graph key is `(attr, value)` to avoid collisions while keeping `.id` as the
    raw value for compatibility.
    """

    def __init__(self, key: Tuple[str, str], attr: str, value: str, node_type: int):
        self.key = key
        self.id = value
        self.attr = attr
        self.type = node_type  # 0: bound/root-like attr; 1: free/RHS-like attr
        self.connectedTo: Dict["Vertex", float] = {}
        self.connectedQLT: Dict["Vertex", float] = {}

    def addNeighbor(self, nbr: "Vertex") -> None:
        self.connectedTo[nbr] = self.connectedTo.get(nbr, 0.0) + 1.0
        self.connectedQLT.setdefault(nbr, 0.0)

    def getConnections(self) -> List["Vertex"]:
        return list(self.connectedTo.keys())

    def getId(self) -> str:
        return self.id

    def getAttr(self) -> str:
        return self.attr

    def getType(self) -> int:
        return self.type

    def getweight(self, nbr: "Vertex") -> float:
        return self.connectedTo[nbr]

    def __repr__(self) -> str:
        return f"Vertex(attr={self.attr!r}, id={self.id!r})"


class Graph:
    """FD pattern graph."""

    def __init__(self):
        self.vertList: Dict[Tuple[str, str], Vertex] = {}
        self.numVertices = 0

    def addVertex(self, attr: str, value: str, node_type: int) -> Vertex:
        key = (attr, str(value))
        if key not in self.vertList:
            self.numVertices += 1
            self.vertList[key] = Vertex(key, attr, str(value), node_type)
        return self.vertList[key]

    def getVertex(self, attr: str, value: str) -> Optional[Vertex]:
        return self.vertList.get((attr, str(value)))

    def addEdge(self, from_attr: str, from_value: str, to_attr: str, to_value: str) -> None:
        f_key = (from_attr, str(from_value))
        t_key = (to_attr, str(to_value))
        if f_key not in self.vertList or t_key not in self.vertList:
            raise KeyError(f"Cannot add edge {f_key} -> {t_key}: vertex missing")
        self.vertList[f_key].addNeighbor(self.vertList[t_key])

    def getVertices(self) -> Iterable[Tuple[str, str]]:
        return self.vertList.keys()

    def __iter__(self):
        return iter(self.vertList.values())


# ---------------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------------


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Use the same string-oriented representation as the original code."""
    return df.copy().replace({np.nan: NAN_TOKEN}).fillna(NAN_TOKEN).astype(str)


def _split_lhs(lhs: str) -> Tuple[str, ...]:
    lhs = lhs.strip()
    # Accept common separators for composite LHS.
    if not lhs:
        raise ValueError("Empty LHS in FD rule")
    if "," in lhs:
        return tuple(x.strip() for x in lhs.split(",") if x.strip())
    if "+" in lhs:
        return tuple(x.strip() for x in lhs.split("+") if x.strip())
    if "&" in lhs:
        return tuple(x.strip() for x in lhs.split("&") if x.strip())
    return (lhs,)


def parse_fd_rules(
    rule_path: Optional[str] = None,
    rules: Optional[Sequence[Union[FDRule, Tuple[Union[str, Sequence[str]], str], Dict[str, Union[str, Sequence[str]]]]]] = None,
) -> List[FDRule]:
    """Parse FD rules from a path or directly supplied rule objects."""
    parsed: List[FDRule] = []

    if rules is not None:
        for item in rules:
            if isinstance(item, FDRule):
                parsed.append(item)
            elif isinstance(item, tuple) and len(item) == 2:
                lhs, rhs = item
                lhs_tuple = tuple(lhs) if isinstance(lhs, (list, tuple)) else _split_lhs(str(lhs))
                parsed.append(FDRule(tuple(str(x).strip() for x in lhs_tuple), str(rhs).strip()))
            elif isinstance(item, dict):
                for lhs, rhs_values in item.items():
                    rhs_list = rhs_values if isinstance(rhs_values, (list, tuple)) else [rhs_values]
                    for rhs in rhs_list:
                        parsed.append(FDRule(_split_lhs(str(lhs)), str(rhs).strip()))
            else:
                raise ValueError(f"Unsupported FD rule object: {item!r}")

    if rule_path is not None:
        if not os.path.exists(rule_path):
            raise FileNotFoundError(f"FD rule file not found: {rule_path}")
        with open(rule_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if content:
            # JSON dictionary used by several DataPrep datasets.
            if content.startswith("{"):
                obj = json.loads(content)
                for lhs, rhs_values in obj.items():
                    rhs_list = rhs_values if isinstance(rhs_values, list) else [rhs_values]
                    for rhs in rhs_list:
                        parsed.append(FDRule(_split_lhs(str(lhs)), str(rhs).strip()))
            else:
                for raw_line in content.splitlines():
                    line = raw_line.strip()
                    if not line or line.startswith("#"):
                        continue
                    # Remove trailing punctuation sometimes present in rule files.
                    line = line.rstrip(";。")
                    parts = re.split(r"\s*(?:⇒|->|→|=>)\s*", line, maxsplit=1)
                    if len(parts) != 2:
                        # Accept a light-weight A: B,C style fallback.
                        if ":" in line:
                            lhs, rhs_text = line.split(":", 1)
                            for rhs in re.split(r"[,，]", rhs_text):
                                rhs = rhs.strip()
                                if rhs:
                                    parsed.append(FDRule(_split_lhs(lhs), rhs))
                            continue
                        raise ValueError(f"Cannot parse FD rule line: {raw_line!r}")
                    lhs_text, rhs_text = parts
                    for rhs in re.split(r"[,，]", rhs_text):
                        rhs = rhs.strip()
                        if rhs:
                            parsed.append(FDRule(_split_lhs(lhs_text), rhs))

    # Deduplicate while preserving order.
    seen = set()
    unique_rules = []
    for r in parsed:
        if not r.lhs or not r.rhs:
            continue
        key = (r.lhs, r.rhs)
        if key not in seen:
            seen.add(key)
            unique_rules.append(r)
    if not unique_rules:
        raise ValueError("No FD rules were provided. Pass `rule_path` or `rules`.")
    return unique_rules


def _check_columns(df: pd.DataFrame, rules: Sequence[FDRule]) -> None:
    missing: Set[str] = set()
    cols = set(map(str, df.columns))
    for r in rules:
        for a in r.lhs:
            if a not in cols:
                missing.add(a)
        if r.rhs not in cols:
            missing.add(r.rhs)
    if missing:
        raise ValueError(f"FD rules reference columns not present in dirty_df: {sorted(missing)}")


def _lhs_attr_key(lhs: Tuple[str, ...]) -> str:
    return lhs[0] if len(lhs) == 1 else LHS_JOINER.join(lhs)


def _lhs_value(row: pd.Series, lhs: Tuple[str, ...]) -> str:
    if len(lhs) == 1:
        return str(row[lhs[0]])
    return VALUE_JOINER.join(str(row[a]) for a in lhs)


# ---------------------------------------------------------------------------
# Horizon paper steps
# ---------------------------------------------------------------------------


def BuildFDPatternGraph(dirty_df: pd.DataFrame, fd_rules: Sequence[FDRule]) -> Graph:
    """Build the FD pattern graph from dirty data and FDs.

    Paper concept: Section 3.2, FD Pattern Graph.  Every concrete FD pattern
    LHS-value -> RHS-value is encoded as a graph edge whose weight starts as the
    occurrence count and is normalized to support.
    """
    data = _normalize_df(dirty_df)
    _check_columns(data, fd_rules)
    g = Graph()

    lhs_attrs = {a for r in fd_rules for a in r.lhs}
    rhs_attrs = {r.rhs for r in fd_rules}
    bound_attrs = lhs_attrs - rhs_attrs

    n = max(len(data), 1)

    for rule in fd_rules:
        left_attr_key = _lhs_attr_key(rule.lhs)
        left_is_bound = all(a in bound_attrs for a in rule.lhs)
        left_type = 0 if left_is_bound else 1
        right_type = 1

        for _, row in data.iterrows():
            lv = _lhs_value(row, rule.lhs)
            rv = str(row[rule.rhs])
            g.addVertex(left_attr_key, lv, left_type)
            g.addVertex(rule.rhs, rv, right_type)
            g.addEdge(left_attr_key, lv, rule.rhs, rv)

    # Convert counts into supports.
    for v in g:
        for w in v.getConnections():
            v.connectedTo[w] = v.connectedTo[w] / n
    return g


def _dfs_quality(g: Graph, root: Vertex, visited_attrs: Dict[str, int]) -> Tuple[float, int]:
    if not root.getConnections():
        return 0.0, 0
    if visited_attrs.get(root.attr, 0) == 1:
        return 0.0, 0

    sup = 0.0
    num = 0
    visited_attrs[root.attr] = 1
    for child in root.getConnections():
        if visited_attrs.get(child.attr, 0) == 0:
            edge_sup = root.connectedTo[child]
            child_sup, child_num = _dfs_quality(g, child, visited_attrs)
            sup += edge_sup + child_sup
            num += 1 + child_num
            root.connectedQLT[child] = (child_sup + edge_sup) / (child_num + 1)
        else:
            # Back-edge / cyclic FD fallback: keep local support as quality.
            root.connectedQLT[child] = root.connectedTo[child]
    visited_attrs[root.attr] = 0
    return sup, num


def ComputePatternQulity(g: Graph) -> Graph:
    """Compute edge quality scores by DFS propagation.

    Name kept as `Qulity` to remain compatible with the original script's typo.
    """
    attrs = {v.attr for v in g}
    roots = [v for v in g if v.getType() == 0]
    if not roots:
        roots = list(g)
    for root in roots:
        visited_attrs = {attr: 0 for attr in attrs}
        _dfs_quality(g, root, visited_attrs)

    # Ensure every edge has a non-zero computed quality; use support fallback.
    for v in g:
        for w in v.getConnections():
            if v.connectedQLT.get(w, 0.0) == 0.0:
                v.connectedQLT[w] = v.connectedTo[w]
    return g


def _tarjan_scc(graph: Dict[str, Set[str]]) -> List[List[str]]:
    index = 0
    stack: List[str] = []
    on_stack: Set[str] = set()
    indices: Dict[str, int] = {}
    lowlink: Dict[str, int] = {}
    comps: List[List[str]] = []

    def strongconnect(v: str) -> None:
        nonlocal index
        indices[v] = index
        lowlink[v] = index
        index += 1
        stack.append(v)
        on_stack.add(v)

        for w in graph.get(v, set()):
            if w not in indices:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], indices[w])

        if lowlink[v] == indices[v]:
            comp = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                comp.append(w)
                if w == v:
                    break
            comps.append(sorted(comp))

    for v in graph:
        if v not in indices:
            strongconnect(v)
    return comps


def BuildSCCGraghAndSort(fd_rules: Sequence[FDRule]):
    """Build the FD attribute graph, SCC graph, and topological component order."""
    attrs: Set[str] = set()
    G: Dict[str, Set[str]] = defaultdict(set)
    for rule in fd_rules:
        attrs.update(rule.lhs)
        attrs.add(rule.rhs)
        for a in rule.lhs:
            G[a].add(rule.rhs)
        G.setdefault(rule.rhs, set())
    for a in attrs:
        G.setdefault(a, set())

    scc = _tarjan_scc(G)
    comp_id: Dict[str, int] = {}
    for i, comp in enumerate(scc):
        for attr in comp:
            comp_id[attr] = i

    dag: Dict[int, Set[int]] = {i: set() for i in range(len(scc))}
    indegree = {i: 0 for i in range(len(scc))}
    for u, vs in G.items():
        for v in vs:
            cu, cv = comp_id[u], comp_id[v]
            if cu != cv and cv not in dag[cu]:
                dag[cu].add(cv)
                indegree[cv] += 1

    q = deque([i for i in dag if indegree[i] == 0])
    order: List[int] = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in sorted(dag[u]):
            indegree[v] -= 1
            if indegree[v] == 0:
                q.append(v)
    if len(order) != len(dag):
        # Should not happen for SCC DAG, but keep a safe fallback.
        order = list(range(len(scc)))

    order_rank = {comp: rank for rank, comp in enumerate(order)}
    tar = {attr: order_rank[comp_id[attr]] for attr in comp_id}
    return order, tar, scc, dict(G)


def OrderFDs(fd_rules: Sequence[FDRule], order=None, tar=None, scc=None, G=None) -> List[OrderedFD]:
    """Order FDs using SCC topological order."""
    if tar is None:
        order, tar, scc, G = BuildSCCGraghAndSort(fd_rules)
    ordered = []
    for rule in fd_rules:
        lhs_rank = min(tar.get(a, 0) for a in rule.lhs)
        rhs_rank = tar.get(rule.rhs, lhs_rank)
        ordered.append(OrderedFD(lhs=rule.lhs, rhs=rule.rhs, lnum=lhs_rank, rnum=rhs_rank))
    ordered.sort(key=lambda x: (x.lnum, x.rnum, x.left, x.right))
    return ordered


def GeneratePatternPreservingRepairs(
    dirty_df: pd.DataFrame,
    rule_path: Optional[str] = None,
    rules: Optional[Sequence[Union[FDRule, Tuple[Union[str, Sequence[str]], str], Dict[str, Union[str, Sequence[str]]]]]] = None,
    return_pattern_expressions: bool = True,
) -> Union[List[Dict[str, str]], Tuple[List[Dict[str, str]], Graph, List[OrderedFD]]]:
    """Generate one pattern expression per input row.

    This is the DataFrame version of the original script function.  It does not
    use clean data, PERFECTED/ONLYED flags, or write files.
    """
    fd_rules = parse_fd_rules(rule_path=rule_path, rules=rules)
    data = _normalize_df(dirty_df)
    _check_columns(data, fd_rules)

    g = BuildFDPatternGraph(data, fd_rules)
    ComputePatternQulity(g)
    order, tar, scc, G = BuildSCCGraghAndSort(fd_rules)
    ordered_fds = OrderFDs(fd_rules, order, tar, scc, G)

    lhs_attrs = {a for r in fd_rules for a in r.lhs}
    rhs_attrs = {r.rhs for r in fd_rules}
    bound_attrs = lhs_attrs - rhs_attrs
    if not bound_attrs:
        # Cyclic-only rules: choose attributes in the earliest SCC as anchors.
        min_rank = min(tar.get(a, 0) for a in lhs_attrs) if lhs_attrs else 0
        bound_attrs = {a for a in lhs_attrs if tar.get(a, 0) == min_rank}

    pattern_expressions: List[Dict[str, str]] = []
    for _, row in data.iterrows():
        rtable: Dict[str, str] = {}
        for attr in bound_attrs:
            if attr in data.columns:
                rtable[attr] = str(row[attr])

        for fd in ordered_fds:
            # Ensure LHS values are available.  If a predecessor FD did not set
            # them, fall back to the tuple's current dirty value.
            for attr in fd.lhs:
                rtable.setdefault(attr, str(row[attr]))

            if fd.rhs in rtable:
                continue

            left_attr_key = _lhs_attr_key(fd.lhs)
            lval = VALUE_JOINER.join(rtable[a] for a in fd.lhs) if len(fd.lhs) > 1 else rtable[fd.lhs[0]]
            left_vertex = g.getVertex(left_attr_key, lval)

            best_value: Optional[str] = None
            best_score = -1.0
            if left_vertex is not None:
                for candidate in left_vertex.getConnections():
                    if candidate.attr == fd.rhs:
                        score = left_vertex.connectedQLT.get(candidate, left_vertex.connectedTo.get(candidate, 0.0))
                        if score > best_score:
                            best_score = score
                            best_value = candidate.id
            if best_value is None:
                best_value = str(row[fd.rhs])
            rtable[fd.rhs] = best_value

        # Fill attributes not involved in the FD chain with original values so
        # apply_pattern_expressions can reconstruct a full table.
        for col in data.columns:
            rtable.setdefault(col, str(row[col]))
        pattern_expressions.append(rtable)

    if return_pattern_expressions:
        return pattern_expressions, g, ordered_fds
    return pattern_expressions


def apply_pattern_expressions(dirty_df: pd.DataFrame, pattern_expressions: Sequence[Dict[str, str]]) -> pd.DataFrame:
    """Convert Horizon pattern expressions into a repaired DataFrame."""
    repaired = _normalize_df(dirty_df)
    for row_idx, expr in enumerate(pattern_expressions):
        if row_idx >= len(repaired):
            break
        for attr, value in expr.items():
            if attr in repaired.columns:
                repaired.iat[row_idx, repaired.columns.get_loc(attr)] = value
    repaired.index = dirty_df.index
    return repaired


def generate_repairs(
    dirty_df: pd.DataFrame,
    rule_path: Optional[str] = None,
    rules: Optional[Sequence[Union[FDRule, Tuple[Union[str, Sequence[str]], str], Dict[str, Union[str, Sequence[str]]]]]] = None,
    detection_mask: Optional[pd.DataFrame] = None,
    apply_only_detected: bool = False,
    return_pattern_expressions: bool = False,
):
    """Main repair API used by DataPrep wrappers.

    Parameters
    ----------
    dirty_df:
        Dirty input table.
    rule_path / rules:
        Functional dependencies.
    detection_mask:
        Optional boolean mask.  If `apply_only_detected=True`, only cells marked
        True are replaced by Horizon's proposed values.
    apply_only_detected:
        Whether to restrict updates to provided detection mask.
    return_pattern_expressions:
        If True, return `(repaired_df, pattern_expressions)`.
    """
    pattern_expressions, _, _ = GeneratePatternPreservingRepairs(
        dirty_df=dirty_df,
        rule_path=rule_path,
        rules=rules,
        return_pattern_expressions=True,
    )
    proposed = apply_pattern_expressions(dirty_df, pattern_expressions)

    if apply_only_detected:
        if detection_mask is None:
            raise ValueError("`detection_mask` is required when apply_only_detected=True")
        mask = detection_mask.reindex(index=dirty_df.index, columns=dirty_df.columns).fillna(False).astype(bool)
        repaired = _normalize_df(dirty_df)
        repaired[mask] = proposed[mask]
    else:
        repaired = proposed

    if return_pattern_expressions:
        return repaired, pattern_expressions
    return repaired


def detect_errors(
    dirty_df: pd.DataFrame,
    rule_path: Optional[str] = None,
    rules: Optional[Sequence[Union[FDRule, Tuple[Union[str, Sequence[str]], str], Dict[str, Union[str, Sequence[str]]]]]] = None,
) -> pd.DataFrame:
    """Detect errors by comparing Horizon repair output with dirty input."""
    repaired = generate_repairs(dirty_df=dirty_df, rule_path=rule_path, rules=rules)
    mask = repaired.astype(str).ne(_normalize_df(dirty_df).astype(str))
    mask.index = dirty_df.index
    mask.columns = dirty_df.columns
    return mask


# ---------------------------------------------------------------------------
# Compatibility utilities kept from the teacher's original horizon.py
# ---------------------------------------------------------------------------


def check_string(string: str):
    """Keep the original dataset-noise suffix helper for script compatibility."""
    string = str(string)
    if re.search(r"-inner_error-", string):
        return "-inner_error-" + string[-6:-4]
    if re.search(r"-outer_error-", string):
        return "-outer_error-" + string[-6:-4]
    if re.search(r"-inner_outer_error-", string):
        return "-inner_outer_error-" + string[-6:-4]
    if re.search(r"-dirty-original_error-", string):
        return "-original_error-" + string[-9:-4]
    return ""


def dfs(g: Graph, root: Vertex, vis: Dict[str, int]):
    """Original public DFS helper, implemented through the refactored graph."""
    return _dfs_quality(g, root, vis)


def dfs1(g: Graph, root: Vertex, vis: Dict[str, int]):
    """Debug DFS from the original script; returns visited node ids instead of printing only."""
    visited = []
    def _walk(v):
        visited.append(v.id)
        if not v.getConnections() or vis.get(v.attr, 0) == 1:
            return
        vis[v.attr] = 1
        for child in v.getConnections():
            visited.append(v.connectedQLT.get(child, v.connectedTo.get(child, 0.0)))
            _walk(child)
        vis[v.attr] = 0
    _walk(root)
    return visited


def tr(G: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    """Transpose a directed graph represented as adjacency sets."""
    GT: Dict[str, Set[str]] = {u: set() for u in G.keys()}
    for u, vs in G.items():
        for v in vs:
            GT.setdefault(v, set()).add(u)
    return GT


def topoSort(G: Dict[str, Set[str]]) -> List[str]:
    """Topological-like DFS finishing order used by the original SCC code."""
    res: List[str] = []
    seen: Set[str] = set()
    def _dfs(u: str):
        if u in seen:
            return
        seen.add(u)
        for v in G.get(u, set()):
            _dfs(v)
        res.append(u)
    for u in G.keys():
        _dfs(u)
    res.reverse()
    return res


def walk(G: Dict[str, Set[str]], s: str, S: Optional[Set[str]] = None) -> Dict[str, Optional[str]]:
    """Reachability helper kept for compatibility with the original SCC routine."""
    if S is None:
        S = set()
    Q = [s]
    P: Dict[str, Optional[str]] = {s: None}
    while Q:
        u = Q.pop()
        for v in G.get(u, set()):
            if v in P or v in S:
                continue
            Q.append(v)
            P[v] = u
    return P


class tmporder(OrderedFD):
    """Original order container name retained as an alias of OrderedFD."""
    def __init__(self):
        super().__init__(lhs=("",), rhs="", lnum=0, rnum=0)


def dirty_cells(dirty_file, clean_file) -> List[Tuple[int, int]]:
    """Return ground-truth dirty cell coordinates for DataFrames or CSV paths."""
    dirty_df = pd.read_csv(dirty_file) if isinstance(dirty_file, (str, os.PathLike)) else dirty_file.copy()
    clean_df = pd.read_csv(clean_file) if isinstance(clean_file, (str, os.PathLike)) else clean_file.copy()
    d = _normalize_df(dirty_df)
    c = _normalize_df(clean_df).reindex(index=d.index, columns=d.columns)
    cells: List[Tuple[int, int]] = []
    diff = d.ne(c)
    for i in range(diff.shape[0]):
        for j in range(diff.shape[1]):
            if bool(diff.iat[i, j]):
                cells.append((i, j))
    return cells


def _read_df_for_metric(path_or_df) -> pd.DataFrame:
    return pd.read_csv(path_or_df) if isinstance(path_or_df, (str, os.PathLike)) else path_or_df.copy()


def _patterns_to_df(pattern_expressions, dirty_df: pd.DataFrame) -> pd.DataFrame:
    return apply_pattern_expressions(dirty_df, pattern_expressions)


def calDetPrecRec(pattern_expressions, dirty_path, clean_path):
    """Compatibility metric: precision/recall for cells changed by Horizon patterns."""
    dirty_df = _normalize_df(_read_df_for_metric(dirty_path))
    clean_df = _normalize_df(_read_df_for_metric(clean_path)).reindex(index=dirty_df.index, columns=dirty_df.columns)
    repaired = _patterns_to_df(pattern_expressions, dirty_df)
    pred_mask = repaired.ne(dirty_df)
    true_mask = dirty_df.ne(clean_df)
    tp = int((pred_mask & true_mask).values.sum())
    pred = int(pred_mask.values.sum())
    true = int(true_mask.values.sum())
    precision = tp / (pred + 1e-10)
    recall = tp / (true + 1e-10)
    return precision, recall


def calRepPrec(pattern_expressions, dirty_path, clean_path):
    """Compatibility repair precision: changed cells that match clean values."""
    dirty_df = _normalize_df(_read_df_for_metric(dirty_path))
    clean_df = _normalize_df(_read_df_for_metric(clean_path)).reindex(index=dirty_df.index, columns=dirty_df.columns)
    repaired = _patterns_to_df(pattern_expressions, dirty_df)
    changed = repaired.ne(dirty_df)
    correct = changed & repaired.eq(clean_df)
    return int(correct.values.sum()) / (int(changed.values.sum()) + 1e-10)


def calRepRec(pattern_expressions, dirty_path, clean_path):
    """Compatibility repair recall: true dirty cells corrected to clean values."""
    dirty_df = _normalize_df(_read_df_for_metric(dirty_path))
    clean_df = _normalize_df(_read_df_for_metric(clean_path)).reindex(index=dirty_df.index, columns=dirty_df.columns)
    repaired = _patterns_to_df(pattern_expressions, dirty_df)
    true_mask = dirty_df.ne(clean_df)
    corrected = true_mask & repaired.eq(clean_df)
    return int(corrected.values.sum()) / (int(true_mask.values.sum()) + 1e-10)


def export_res(pattern_expressions, dirty_path, res_path: Optional[str] = None) -> pd.DataFrame:
    """Build repaired DataFrame and optionally export it to CSV."""
    dirty_df = _read_df_for_metric(dirty_path)
    repaired = _patterns_to_df(pattern_expressions, dirty_df)
    if res_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(res_path)), exist_ok=True)
        repaired.to_csv(res_path, index=False)
    return repaired


def calF1(precision, recall):
    return 2 * precision * recall / (precision + recall + 1e-10)


def _vertex_str(self):
    return str(self.id) + "connectedTo" + str([x.id for x in self.connectedTo])


def _graph_contains(self, n):
    """Compatibility with original ``n in Graph`` usage.

    Accepts either the refactored ``(attr, value)`` key or a raw value.  Raw
    value matching is best-effort because the refactor avoids cross-column value
    collisions by storing graph keys as ``(attr, value)``.
    """
    if n in self.vertList:
        return True
    return any(v.id == str(n) for v in self.vertList.values())


Vertex.__str__ = _vertex_str
Graph.__contains__ = _graph_contains


__all__ = [
    "FDRule", "OrderedFD", "tmporder", "Vertex", "Graph",
    "check_string", "parse_fd_rules", "BuildFDPatternGraph", "dfs", "dfs1",
    "ComputePatternQulity", "tr", "topoSort", "walk", "BuildSCCGraghAndSort",
    "OrderFDs", "GeneratePatternPreservingRepairs", "apply_pattern_expressions",
    "generate_repairs", "detect_errors", "calDetPrecRec", "calRepPrec", "calRepRec",
    "export_res", "calF1", "dirty_cells",
]

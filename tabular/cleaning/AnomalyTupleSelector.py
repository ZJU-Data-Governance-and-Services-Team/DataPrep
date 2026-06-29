import pandas as pd
import numpy as np
import string
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import fasttext

try:
    from .project_config import DEFAULT_FASTTEXT_MODEL_PATH
except ImportError:
    from project_config import DEFAULT_FASTTEXT_MODEL_PATH


class AnomalyTupleSelector:
    # Initialize selector state and load the input data.
    def __init__(self, df: pd.DataFrame, fasttext_model_path: str = DEFAULT_FASTTEXT_MODEL_PATH):
        self.raw_df = df
        self.fasttext_model_path = fasttext_model_path
        self.anomaly_scores = pd.DataFrame(0.0, index=df.index, columns=df.columns)
        self.format_scores = pd.DataFrame(0.0, index=df.index, columns=df.columns)
        self.fd_scores = pd.DataFrame(0.0, index=df.index, columns=df.columns)
        self.semantic_scores = pd.DataFrame(0.0, index=df.index, columns=df.columns)
        
        from_chars = string.ascii_uppercase + string.ascii_lowercase + string.digits
        to_chars = ('A' * len(string.ascii_uppercase)) + \
                   ('a' * len(string.ascii_lowercase)) + \
                   ('9' * len(string.digits))
        self.fine_trans_table = str.maketrans(from_chars, to_chars)
        
        self.ft_model = None
        
        self.entity_col = None

    # Normalize a value into a compact pattern label.
    def _get_pattern(self, text):
        if pd.isna(text):
            return "NULL"
        return str(text).translate(self.fine_trans_table)
    
    # Score how consistently a column follows frequent patterns.
    def _calculate_column_format_consistency(self, freq_dict, min_freq_threshold):
        n_unique = len(freq_dict)
        
        if n_unique <= 1:
            return 0.0
            
        high_freq_probs = [p for p in freq_dict.values() if p > min_freq_threshold]
        n_high = len(high_freq_probs)
        
        if n_high == 0:
            return 0.0
            
        sum_high_probs = sum(high_freq_probs)
        denominator = np.sqrt(n_high)
        
        return float(sum_high_probs / denominator)

    # Find columns with stable format patterns.
    def _detect_format_consistent_columns(self, min_freq_threshold, consistency_threshold):
        print(f"Detecting Format Consistent Columns (Threshold: {min_freq_threshold})...")
        results = {}
        
        for col in self.raw_df.columns:
            series = self.raw_df[col]
            
            patterns = series.map(self._get_pattern)
            
            freq_dict = patterns.value_counts(normalize=True).to_dict()
            
            score = self._calculate_column_format_consistency(freq_dict, min_freq_threshold)
            
            results[col] = {
                'score': score,
                'original_data': series,
                'patterns': patterns,
                'frequency': freq_dict
            }
            
        print(f"{'Column':<20} | {'Score':<8} | {'High-Freq Patterns'}")
        print("-" * 80)
        
        for col in self.raw_df.columns:
            info = results[col]
            
            high_freq_patterns = {pat: freq for pat, freq in info['frequency'].items() if freq > min_freq_threshold}
            if high_freq_patterns:
                pat_str = ", ".join([f"{pat} ({freq:.1%})" for pat, freq in high_freq_patterns.items()])
            else:
                pat_str = "None"
            pat_str += f' | Total patterns: {len(info["frequency"])} '
                
            col_display = f"*{col}" if info['score'] > consistency_threshold else col
                
            print(f"{col_display:<20} | {info['score']:.4f}   | {pat_str}")
            
        return results

    # Compute format anomaly scores for all cells.
    def calculate_format_anomaly(self, min_freq_threshold=0.10, consistency_threshold=0.5):
        print(f"Calculating Format Anomaly Scores (Consistency > {consistency_threshold})...")
        
        format_info_dict = self._detect_format_consistent_columns(min_freq_threshold, consistency_threshold)
        
        format_score_matrix = pd.DataFrame(0.0, index=self.raw_df.index, columns=self.raw_df.columns)
        
        for col in self.raw_df.columns:
            col_info = format_info_dict[col]
            consistency_score = col_info['score']
            
            if consistency_score > consistency_threshold:
                patterns = col_info['patterns']
                freq_map = col_info['frequency']
                
                max_freq = max(freq_map.values())
                
                probs = patterns.map(freq_map)
                
                scores = 1.0 - (probs / max_freq).fillna(0)
                
                format_score_matrix[col] = scores
            else:
                format_score_matrix[col] = 0.0
                
        row_sums = format_score_matrix.sum(axis=1)
        max_row_sum = row_sums.max()
        
        if max_row_sum > 0:
            format_score_matrix /= max_row_sum
            
        self.format_scores += format_score_matrix
        print(f"Format Anomaly Calculation Done. (Max Row Score Normalized to 1.0, Raw Max was {max_row_sum:.4f})")
        print("*" * 65)
        
    # Calculate entropy for a column or series.
    def _calculate_entropy(self, series):
        probs = series.value_counts(normalize=True).values
        return entropy(probs)

    # Compute Theil's U dependency score between two columns.
    def _calculate_theils_u(self, source_col, target_col):
        s_clean = source_col.fillna("NULL").astype(str)
        t_clean = target_col.fillna("NULL").astype(str)
        h_target = self._calculate_entropy(t_clean)
        
        if h_target == 0: 
            return 0.0
            
        mi = mutual_info_score(s_clean, t_clean)
        return mi / h_target

    # Identify the strongest global entity identifier column.
    def _detect_global_entity_identifier(self):
        n_rows = len(self.raw_df)
        columns = self.raw_df.columns
        
        best_col = None
        max_score = -float('inf')
        
        debug_info = []
        
        for candidate in columns:
            n_unique = self.raw_df[candidate].nunique(dropna=False)
            unique_ratio = n_unique / n_rows
            
            total_u = 0.0
            valid_targets = 0
            
            for target in columns:
                if candidate == target: continue
                
                u_score = self._calculate_theils_u(self.raw_df[candidate], self.raw_df[target])
                total_u += u_score
                valid_targets += 1
            
            if valid_targets > 0:
                avg_u = total_u / valid_targets
                
                final_score = avg_u - unique_ratio
                
                debug_info.append({
                    'col': candidate,
                    'avg_u': avg_u,
                    'n_unique': n_unique,
                    'penalty': unique_ratio,
                    'final_score': final_score
                })
                
                if final_score > max_score:
                    max_score = final_score
                    best_col = candidate
        
        print(f"Global Entity Identifier Selection:")
        print(f"{'Column':<20} | {'Avg U (Power)':<12} | {'Unique Ratio':<12} | {'Final Score'}")
        print("-" * 65)
        
        for item in debug_info:
            col_display = f"*{item['col']}" if item['col'] == best_col else item['col']
            print(f"{col_display:<20} | {item['avg_u']:.4f}       | {item['penalty']:.4f}       | {item['final_score']:.4f}")
            
        if best_col:
            print(f">>> Winner: {best_col}")
            self.entity_col = best_col
            
        best_col_score = max_score
        return best_col, best_col_score

    # Detect columns functionally dependent on the entity identifier.
    def _detect_fd_dependent_columns(self, entity_col, fd_threshold):
        print(f"\nDetecting FD Dependent Columns on Entity '{entity_col}' (Threshold: {fd_threshold})...")
        results = {}

        for target_col in self.raw_df.columns:
            if target_col == entity_col:
                results[target_col] = 1.0
                continue

            u_score = self._calculate_theils_u(self.raw_df[entity_col], self.raw_df[target_col])
            results[target_col] = u_score

        print(f"{'Column':<20} | {'U(Target|Entity) Score'}")
        print("-" * 50)
        
        for col in self.raw_df.columns:
            score = results[col]
            
            if col == entity_col:
                col_display = f"[{col}]"
            elif score > fd_threshold:
                col_display = f"*{col}"
            else:
                col_display = col
                
            print(f"{col_display:<20} | {score:.4f}")

        return results

    # Compute functional-dependency anomaly scores for all cells.
    def calculate_fd_anomaly(self, fd_threshold=0.5):
        print(f"Calculating FD Anomaly Scores (Global Entity Strategy, FD Confidence > {fd_threshold})...")
        
        entity_col, _ = self._detect_global_entity_identifier()
        
        if entity_col is None:
            print("No suitable entity identifier found. Skipping FD check.")
            return

        fd_score_matrix = pd.DataFrame(0.0, index=self.raw_df.index, columns=self.raw_df.columns)
        
        fd_dependence_scores = self._detect_fd_dependent_columns(entity_col, fd_threshold)
        
        for target_col in self.raw_df.columns:
            
            if target_col == entity_col:
                fd_score_matrix[target_col] = 0.0
                continue
            
            dependence_score = fd_dependence_scores[target_col]
            
            if dependence_score > fd_threshold:
                temp_df = self.raw_df[[entity_col, target_col]].fillna("NULL_VAL")
                
                pair_counts = temp_df.groupby([entity_col, target_col])[target_col].transform('count')
                
                temp_df['__cnt__'] = pair_counts
                mode_counts = temp_df.groupby(entity_col)['__cnt__'].transform('max')
                
                scores = 1.0 - (pair_counts / mode_counts)
                
                fd_score_matrix[target_col] = scores
            else:
                fd_score_matrix[target_col] = 0.0
                
        row_sums = fd_score_matrix.sum(axis=1)
        max_row_sum = row_sums.max()
        
        if max_row_sum > 0:
            fd_score_matrix /= max_row_sum

        self.fd_scores += fd_score_matrix
        print(f"FD Anomaly Calculation Done. (Max Row Score Normalized to 1.0, Raw Max was {max_row_sum:.4f})")
        print("*" * 65)
        
    # Load and cache the FastText model.
    def _get_fasttext_model(self):
        if self.ft_model is None:
            print(f"Loading FastText Model ({self.fasttext_model_path})...")
            fasttext.FastText.eprint = lambda x: None
            self.ft_model = fasttext.load_model(self.fasttext_model_path)
            
        return self.ft_model

    # Compute semantic anomaly scores using FastText embeddings.
    def calculate_semantic_anomaly(self):
        print("Calculating Semantic Anomaly Scores (using FastText)...")
        
        semantic_score_matrix = pd.DataFrame(0.0, index=self.raw_df.index, columns=self.raw_df.columns)

        model = None 
        
        for col in self.raw_df.columns:
            series = self.raw_df[col].fillna("NULL").astype(str)
            
            # --- Part 1: Value Frequency ---
            freq_map = series.value_counts(normalize=True).to_dict()
            freq_scores = 1.0 - series.map(freq_map)
            
            # --- Part 2: Embedding Isolation Forest ---
            n_unique = series.nunique()
            
            emb_scores = pd.Series(0.0, index=self.raw_df.index)
            emb_success = False
            
            if n_unique > 2:
                try:
                    if model is None:
                        model = self._get_fasttext_model()
                        
                    unique_vals = list(freq_map.keys())
                    
                    embeddings = [model.get_sentence_vector(str(v)) for v in unique_vals]
                    embeddings = np.array(embeddings)
                    
                    clf = IsolationForest(n_estimators=100, contamination='auto', random_state=42, n_jobs=-1)
                    clf.fit(embeddings)
                    
                    raw_scores = -clf.decision_function(embeddings)
                    scaler = MinMaxScaler()
                    norm_scores = scaler.fit_transform(raw_scores.reshape(-1, 1)).flatten()
                    
                    val_to_score = dict(zip(unique_vals, norm_scores))
                    emb_scores = series.map(val_to_score)
                    
                    emb_success = True
                    
                except Exception as e:
                    print(f"  [Warn] FastText failed for '{col}': {e}. Fallback to freq only.")
            
            if emb_success:
                total_semantic_score = (0.5 * freq_scores) + (0.5 * emb_scores)
            else:
                total_semantic_score = freq_scores
                
            semantic_score_matrix[col] = total_semantic_score
            
        row_sums = semantic_score_matrix.sum(axis=1)
        max_row_sum = row_sums.max()
        
        if max_row_sum > 0:
            semantic_score_matrix /= max_row_sum
            
        self.semantic_scores += semantic_score_matrix
        print(f"Semantic Anomaly Calculation Done. (Max Row Score Normalized to 1.0, Raw Max was {max_row_sum:.4f})")
    
    # Select top anomalous tuples using a merged score.
    def select_top_k_tuples_merge(self, k=10, weight_format=1.0, weight_fd=1.0, weight_semantic=1.0):
        if weight_format:
            self.calculate_format_anomaly(min_freq_threshold=0.10, consistency_threshold=0.5)
        if weight_fd:
            self.calculate_fd_anomaly(fd_threshold=0.5)
        if weight_semantic:
            self.calculate_semantic_anomaly()
        self.anomaly_scores = (weight_format * self.format_scores + 
                              weight_fd * self.fd_scores + 
                              weight_semantic * self.semantic_scores
                              ) / (weight_format + weight_fd + weight_semantic)
        row_total_scores = self.anomaly_scores.sum(axis=1)
        
        top_k_indices = row_total_scores.nlargest(k).index
        
        selected_data = self.raw_df.loc[top_k_indices].copy()
        selected_data['__Anomaly_Score__'] = row_total_scores.loc[top_k_indices]
        
        return selected_data
    
    # Select diverse high-scoring tuples while respecting coverage.
    def _select_coverage_aware(self, score_matrix, available_indices, k):
        selected_indices = []
        covered_cols = set()
        
        col_max_scores = score_matrix.loc[available_indices].max()
        anomalous_cols = col_max_scores[col_max_scores > 0].sort_values(ascending=False).index.tolist()
        
        for col in anomalous_cols:
            if len(selected_indices) >= k:
                break
                
            if col not in covered_cols:
                best_row = score_matrix.loc[available_indices, col].idxmax()
                
                if score_matrix.loc[best_row, col] > 0:
                    selected_indices.append(best_row)
                    
                    available_indices = available_indices.difference([best_row])
                    
                    for c in anomalous_cols:
                        if c in score_matrix.columns and score_matrix.loc[best_row, c] > 0:
                            covered_cols.add(c)
                            
        if len(selected_indices) < k:
            remaining_k = k - len(selected_indices)
            row_sums = score_matrix.loc[available_indices].sum(axis=1)
            valid_remaining = row_sums[row_sums > 0].nlargest(remaining_k).index.tolist()
            selected_indices.extend(valid_remaining)
            available_indices = available_indices.difference(valid_remaining)
            
        return selected_indices

    # Select top anomalous tuples separately by anomaly type.
    def select_top_k_tuples_sep(self, num_fd=4, num_format=3, num_semantic=3, coverage_aware=True, random_label=False):
        selected_indices = set()
        results = []
        
        # --- Step 1: Calculate Scores Independently ---
        print(f"\n--- AnomalyTupleSelector Step 1: Calculating Scores for Selection (Coverage Aware: {coverage_aware}, Random: {random_label}) ---")
        print("*" * 65)
        self.calculate_fd_anomaly(fd_threshold=0.5)
        self.calculate_format_anomaly(min_freq_threshold=0.10, consistency_threshold=0.5)
        self.calculate_semantic_anomaly()
        
        if random_label:
            total_k = num_fd + num_format + num_semantic
            print(f"\n--- AnomalyTupleSelector Random Baseline: Selecting {total_k} tuples randomly ---")
            
            sample_size = min(total_k, len(self.raw_df))
            
            random_indices = self.raw_df.sample(n=sample_size).index.tolist()

            selected_df = self.raw_df.loc[random_indices].copy()
            selected_df['__Selection_Reason__'] = 'Random_Baseline'
            
            total_scores = (self.format_scores + self.fd_scores + self.semantic_scores).sum(axis=1)
            selected_df['__Dimension_Score__'] = total_scores.loc[random_indices]
            
            print(f'\nSuccessfully selected {sample_size} Random tuples for baseline testing.')
            return selected_df

        # --- Step 2: FD Selection (Group-based Strategy) ---
        strategy_str = "Group + Coverage Aware Strategy" if coverage_aware else "Group Strategy"
        print(f"\n--- AnomalyTupleSelector Step 2: Selecting {num_fd} tuples based on FD ({strategy_str}) ---")
        
        fd_row_scores = self.fd_scores.sum(axis=1)
        
        if self.entity_col is not None:
            entity_col = self.entity_col
            print(f"  Using Entity Column: '{entity_col}' for grouping.")
            
            group_stats = fd_row_scores.groupby(self.raw_df[entity_col]).agg(['mean', 'count', 'std'])
            
            valid_groups = group_stats[group_stats['count'] >= num_fd]
            
            target_group_indices = []
            
            if not valid_groups.empty:
                best_group_key = valid_groups['mean'].idxmax()
                print(f"  Best Group Found: {best_group_key} (Mean Score: {valid_groups.loc[best_group_key, 'mean']:.4f})")
                
                group_indices = self.raw_df[self.raw_df[entity_col] == best_group_key].index
                available_group_indices = pd.Index(group_indices)
                
                n_abnormal = num_fd - 1
                
                if coverage_aware:
                    top_abnormal = self._select_coverage_aware(self.fd_scores, available_group_indices, n_abnormal)
                else:
                    group_scores = fd_row_scores.loc[group_indices].sort_values(ascending=False)
                    top_abnormal = group_scores.head(n_abnormal).index.tolist()
                
                remaining_group_indices = available_group_indices.difference(top_abnormal)
                if not remaining_group_indices.empty:
                    bottom_normal = [fd_row_scores.loc[remaining_group_indices].idxmin()]
                else:
                    bottom_normal = []
                
                target_group_indices = top_abnormal + bottom_normal
                
                for idx in target_group_indices:
                    results.append((idx, 'FD_Group_Strategy', fd_row_scores.loc[idx]))
                    selected_indices.add(idx)
                    
            else:
                print(f"  [Warn] No group satisfies size requirement. Fallback to {'Coverage-Aware ' if coverage_aware else ''}Top-K for FD.")
                if coverage_aware:
                    fallback_indices = self._select_coverage_aware(self.fd_scores, self.fd_scores.index, num_fd)
                else:
                    fallback_indices = fd_row_scores.nlargest(num_fd).index
                    
                for idx in fallback_indices:
                    results.append((idx, 'FD_Fallback', fd_row_scores.loc[idx]))
                    selected_indices.add(idx)
        else:
             print(f"  [Warn] No Entity Identifier found. Fallback to {'Coverage-Aware ' if coverage_aware else ''}Top-K for FD.")
             if coverage_aware:
                 fallback_indices = self._select_coverage_aware(self.fd_scores, self.fd_scores.index, num_fd)
             else:
                 fallback_indices = fd_row_scores.nlargest(num_fd).index
                 
             for idx in fallback_indices:
                results.append((idx, 'FD_Fallback', fd_row_scores.loc[idx]))
                selected_indices.add(idx)
                
        # --- Step 3: Format Selection ---
        print(f"\n--- AnomalyTupleSelector Step 3: Selecting {num_format} tuples based on Format ---")
        format_row_scores = self.format_scores.sum(axis=1)
        available_indices = format_row_scores.index.difference(list(selected_indices))
        
        if coverage_aware:
            top_format_indices = self._select_coverage_aware(self.format_scores, available_indices, num_format)
        else:
            top_format_indices = format_row_scores.loc[available_indices].nlargest(num_format).index
            
        for idx in top_format_indices:
            results.append((idx, 'Format_Anomaly', format_row_scores.loc[idx]))
            selected_indices.add(idx)
            
        # --- Step 4: Semantic Selection ---
        print(f"\n--- AnomalyTupleSelector Step 4: Selecting {num_semantic} tuples based on Semantic ---")
        semantic_row_scores = self.semantic_scores.sum(axis=1)
        available_indices = semantic_row_scores.index.difference(list(selected_indices))
        
        if coverage_aware:
            top_semantic_indices = self._select_coverage_aware(self.semantic_scores, available_indices, num_semantic)
        else:
            top_semantic_indices = semantic_row_scores.loc[available_indices].nlargest(num_semantic).index
            
        for idx in top_semantic_indices:
            results.append((idx, 'Semantic_Anomaly', semantic_row_scores.loc[idx]))
            selected_indices.add(idx)
            
        # --- Assemble Final DataFrame ---
        final_indices = [r[0] for r in results]
        final_reasons = [r[1] for r in results]
        final_scores = [r[2] for r in results]
        
        selected_df = self.raw_df.loc[final_indices].copy()
        selected_df['__Selection_Reason__'] = final_reasons
        selected_df['__Dimension_Score__'] = final_scores
        
        print(f'\nSuccessfully selected {num_fd} FD-based, {num_format} Format-based, and {num_semantic} Semantic-based tuples (Total: {len(selected_df)})')
        
        return selected_df
    

if __name__ == "__main__":
    dataset = 'jobs'  # 'flights' 'hospital' 'beers' 'jobs' 'tax'
    dataset_root = 'data/datasets'
    df = pd.read_csv(f'{dataset_root}/{dataset}/{dataset}_error-01.csv', dtype=str).fillna('nan')
    
    selector = AnomalyTupleSelector(df)
    
    # top_10_rows = selector.select_top_k_tuples_merge()
    top_10_rows = selector.select_top_k_tuples_sep(4, 3, 3, coverage_aware=True, random_label=False)
    
    print("\n=== Top 10 Tuples for Manual Annotation ===")
    print(top_10_rows)
    

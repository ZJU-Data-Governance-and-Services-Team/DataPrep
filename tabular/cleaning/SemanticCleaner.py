import pandas as pd
import numpy as np
import Levenshtein
import os
import json
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import sys
from datetime import datetime
import fasttext
fasttext.FastText.eprint = lambda x: None
from sklearn.decomposition import PCA
from sklearn.cluster import HDBSCAN
import shutil
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from .LLM_response import call_llm
    from .FDCleaner import FDCleaner
    from .measure import measure_repair
    from .project_config import DEFAULT_FASTTEXT_MODEL_PATH, DEFAULT_LLM_API_KEY, DEFAULT_LLM_BASE_URL, DEFAULT_LLM_MODEL, DEFAULT_RESULT_ROOT, DEFAULT_SEMANTIC_MODEL_PATH
except ImportError:
    from LLM_response import call_llm
    from FDCleaner import FDCleaner
    from measure import measure_repair
    from project_config import DEFAULT_FASTTEXT_MODEL_PATH, DEFAULT_LLM_API_KEY, DEFAULT_LLM_BASE_URL, DEFAULT_LLM_MODEL, DEFAULT_RESULT_ROOT, DEFAULT_SEMANTIC_MODEL_PATH

class TypoDetector:
    # Initialize typo detector settings and load dirty data.
    def __init__(self, dirty_path: str, clean_path: str, save_dir: str,
                 min_dict_ratio: float = 0.2, 
                 max_suspect_ratio: float = 0.005, 
                 max_distance_ratio: float = 0.25):
        self.dirty_path = dirty_path
        self.clean_path = clean_path
        self.save_dir = save_dir
        self.min_dict_ratio = min_dict_ratio
        self.max_suspect_ratio = max_suspect_ratio
        self.max_distance_ratio = max_distance_ratio
        
        
        self.dirty_df = pd.read_csv(dirty_path, dtype=str).fillna('nan')
        
        self.null_values = ['nan', 'none', 'null', 'empty', '']

    # Detect likely typos and null-like values by column.
    def detect(self) -> dict:
        result_dict = {}
        
        for col in self.dirty_df.columns:
            series = self.dirty_df[col]
            val_series = series.str.lower()
            
            null_mask = val_series.isin(self.null_values)
            null_indices = val_series[null_mask].index.tolist()
            
            valid_mask = ~null_mask
            filtered_series = val_series[valid_mask]
            
            typo_indices = []
            if len(filtered_series) > 0:
                val_ratios = filtered_series.value_counts(normalize=True)
                valid_dict = val_ratios[val_ratios >= self.min_dict_ratio].index.tolist()
                suspects = val_ratios[val_ratios <= self.max_suspect_ratio].index.tolist()
                
                known_typos = set()
                for suspect in suspects:
                    allowed_dist = int(len(suspect) * self.max_distance_ratio)
                    
                    if allowed_dist > 0:
                        for valid_word in valid_dict:
                            dist = Levenshtein.distance(suspect, valid_word)
                            if 0 < dist <= allowed_dist:
                                known_typos.add(suspect)
                                break
                
                if known_typos:
                    typo_indices = val_series[val_series.isin(known_typos)].index.tolist()
            
            combined_error_indices = list(set(null_indices + typo_indices))
            
            if combined_error_indices:
                combined_error_indices.sort()
                result_dict[col] = combined_error_indices
        
        phase_semantic_dir = os.path.join(self.save_dir, "phase_semantic")
        os.makedirs(phase_semantic_dir, exist_ok=True)
        
        save_path = os.path.join(phase_semantic_dir, "typo_detect.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=4)
            
        return result_dict


class SemanticDetector:
    # Initialize semantic detector settings, model paths, and data frames.
    def __init__(self, dirty_path: str, clean_path: str, model_path: str, save_dir: str,
                 median_multiplier: float = 2.0, min_k_ratio: float = 0.01,
                 fasttext_model_path: str = DEFAULT_FASTTEXT_MODEL_PATH,
                 llm_base_url: str = DEFAULT_LLM_BASE_URL, llm_api_key: str = DEFAULT_LLM_API_KEY, llm_model: str = DEFAULT_LLM_MODEL):
        self.dirty_path = dirty_path
        self.clean_path = clean_path
        self.model_path = model_path
        self.model_path = model_path
        self.save_dir = save_dir
        self.median_multiplier = median_multiplier
        self.min_k_ratio = min_k_ratio
        self.fasttext_model_path = fasttext_model_path
        self.llm_base_url = llm_base_url
        self.llm_api_key = llm_api_key
        self.llm_model = llm_model

        self.LLM_sleep_time = 0
        self.LLM_time = 0
        
        self.phase_semantic_dir = os.path.join(self.save_dir, "phase_semantic")
        os.makedirs(self.phase_semantic_dir, exist_ok=True)
        
        self.dirty_df = pd.read_csv(dirty_path, dtype=str).fillna('nan')
        self.loss_matrix = pd.DataFrame(0.0, index=self.dirty_df.index, columns=self.dirty_df.columns)
        self.null_values = ['nan', 'none', 'null', 'empty', 'n-r', '']
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None

    # Load the semantic model and tokenizer when needed.
    def _load_model(self):
        if self.model is not None:
            return 
            
        print(f"Loading Model '{self.model_path}' to {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, 
            torch_dtype=torch.float16, 
            trust_remote_code=True
        ).to(self.device)
        self.model.eval()

    # Serialize a row into text and track value spans.
    def _serialize_row_and_get_all_spans(self, row: pd.Series):
        text = ""
        spans_dict = {}
        for col in row.index:
            if str(col).startswith("__"): 
                continue
            val = str(row[col])
            prefix = f"The {col} is "
            text += prefix
            start_char = len(text)
            text += val
            end_char = len(text)
            if val.lower() not in self.null_values:
                spans_dict[col] = (start_char, end_char)
            text += ". "
        return text.strip(), spans_dict

    # Calculate average token losses for each value span.
    def _calculate_row_losses(self, text: str, spans_dict: dict):
        if not spans_dict:
            return {}
        inputs = self.tokenizer(text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=True)
        input_ids = inputs["input_ids"].to(self.device)
        offsets = inputs["offset_mapping"][0].cpu().numpy()
        
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits
            
        shift_logits = logits[0, :-1, :].contiguous()
        shift_labels = input_ids[0, 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        token_losses = loss_fct(shift_logits, shift_labels)
        
        row_losses = {}
        for col, (char_start, char_end) in spans_dict.items():
            target_token_indices = []
            for idx, (start, end) in enumerate(offsets):
                if start < char_end and end > char_start:
                    target_token_indices.append(idx)
                    
            local_losses = []
            for idx in target_token_indices:
                shifted_idx = idx - 1
                if 0 <= shifted_idx < len(token_losses):
                    local_losses.append(token_losses[shifted_idx].item())
                    
            if local_losses:
                row_losses[col] = sum(local_losses) / len(local_losses)
            else:
                row_losses[col] = 0.0
        return row_losses

    # Compute semantic perplexity losses for all rows and save the matrix.
    def _generate_loss_matrix(self):
        self._load_model()
        print("Computing Semantic PPL Matrix...")
        
        for idx, row in tqdm(self.dirty_df.iterrows(), total=len(self.dirty_df), desc="Forward Pass"):
            text, spans_dict = self._serialize_row_and_get_all_spans(row)
            if not spans_dict:
                continue
            row_losses = self._calculate_row_losses(text, spans_dict)
            for col, local_loss in row_losses.items():
                self.loss_matrix.loc[idx, col] = local_loss
                
        save_path = os.path.join(self.phase_semantic_dir, "semantic_loss_matrix.csv")
        self.loss_matrix.to_csv(save_path, index=True)
        print(f"Matrix saved to: {save_path}")

    def _call_llm(self, prompt):
        llm_args = self._get_llm_args()
        if llm_args is None:
            return "No LLM specified or LLM not recognized."

        base_url, model, api_key = llm_args
        return call_llm(prompt=prompt, base_url=base_url, api_key=api_key, model=model)

    def _get_llm_args(self):
        if not (self.llm_base_url and self.llm_api_key and self.llm_model):
            return None
        return self.llm_base_url, self.llm_model, self.llm_api_key

    # Build an LLM prompt for verifying suspicious semantic values.
    def _build_semantic_probe_prompt(self, col: str, suspicious_tuples_str: str, all_dirty_rows_str: str, all_clean_rows_str: str, col_labeled_list_str: str):
        prompt = f"""[Role] Data Quality Expert
[Task] Anomaly Verification for column '{col}'

[Global Context (All Sampled Tuples)]
Original Dirty Tuples:
{all_dirty_rows_str}
Clean Tuples (Ground Truth):
{all_clean_rows_str}

[Labeled Errors for Target Column '{col}']
{col_labeled_list_str}

[Suspicious Tuples for Verification]
The following tuples contain values in the target column '{col}' that have been flagged as highly suspicious by a semantic perplexity scorer.
Please focus specifically on the target column '{col}' within these tuples:
{suspicious_tuples_str}

[Instruction]
Based on the expected semantics of the column '{col}', the global context, and the labeled errors, do any of the flagged values in the target column contain obvious typos or semantic errors?
Answer strictly with XML tags:
<decision>是</decision> (if you are absolutely certain that at least one flagged value is an obvious error) 
OR 
<decision>否</decision> (if all flagged values appear normal and semantically plausible)
<reason>Briefly explain your judgment.</reason>
"""
        return prompt

    # Extract a semantic probe decision from an LLM response.
    def _extract_probe_decision(self, response: str):
        if not response:
            return None
        decision = None
        match_decision = re.search(r'<decision>(.*?)</decision>', response, re.DOTALL | re.IGNORECASE)
        if match_decision:
            d_text = match_decision.group(1).strip()
            if "是" in d_text or "yes" in d_text.lower():
                decision = "是"
            elif "否" in d_text or "no" in d_text.lower():
                decision = "否"
        return decision

    # Save an LLM prompt and response for audit logs.
    def _save_prompt_response(self, phase: str, col: str, attempt: int, prompt: str, response: str, pr_dir: str):
        safe_col = str(col).replace("/", "_").replace("\\", "_")
        filename = os.path.join(pr_dir, f"{phase}_{safe_col}_attempt_{attempt}.txt")
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=== PROMPT ===\n")
            f.write(prompt + "\n\n")
            f.write("=== RESPONSE ===\n")
            f.write(str(response) + "\n")

    # Detect semantic anomalies and save the result.
    def detect(self) -> dict:
        self._generate_loss_matrix()
        
        k_threshold = min(max(1, int(len(self.dirty_df) * self.min_k_ratio)), 30)
        result_dict = {}
        
        print("Running detection logic...")
        
        pr_dir = os.path.join(self.phase_semantic_dir, 'prompt_response')
        os.makedirs(pr_dir, exist_ok=True)
        
        phase_label_dir = os.path.join(self.save_dir, 'phase_label')
        with open(os.path.join(phase_label_dir, 'labeled_data.json'), 'r', encoding='utf-8') as f:
            labeled_data = json.load(f)
        with open(os.path.join(phase_label_dir, 'tuples_data.json'), 'r', encoding='utf-8') as f:
            tuples_data = json.load(f)
            
        all_dirty_rows_str = "\n".join([f"Row {k}: {v}" for k, v in tuples_data.get("dirty", {}).items()])
        all_clean_rows_str = "\n".join([f"Row {k}: {v}" for k, v in tuples_data.get("clean", {}).items()])

        for col in self.dirty_df.columns:
            print(f"Processing semantic detection for column: {col}")
            valid_col_losses = self.loss_matrix[col].replace(0.0, np.nan).dropna()
            
            if valid_col_losses.empty:
                continue

            sorted_losses = valid_col_losses.sort_values(ascending=False)
            top_3_idx = sorted_losses.head(3).index.tolist()
            
            suspicious_lines = []
            for idx in top_3_idx:
                row_data = {k: v for k, v in self.dirty_df.loc[idx].to_dict().items() if not str(k).startswith("__")}
                suspicious_lines.append(f"Row {idx}: {row_data}")
            suspicious_tuples_str = "\n".join(suspicious_lines)
            
            col_labels = [item for item in labeled_data if item[1] == col]
            col_labeled_list_str = "\n".join([f"Row {item[0]}: '{item[2]}' -> '{item[3]}' (Reason: {item[4]})" for item in col_labels])
            if not col_labeled_list_str: 
                col_labeled_list_str = "None"

            prompt = self._build_semantic_probe_prompt(
                col=col, 
                suspicious_tuples_str=suspicious_tuples_str, 
                all_dirty_rows_str=all_dirty_rows_str, 
                all_clean_rows_str=all_clean_rows_str, 
                col_labeled_list_str=col_labeled_list_str
            )
            
            decision = None
            max_retries = 3
            for attempt in range(max_retries):
                response = self._call_llm(prompt) 
                
                self._save_prompt_response("semantic_detect", col, attempt + 1, prompt, response, pr_dir)
                
                decision = self._extract_probe_decision(response)
                
                if decision is not None:
                    break
                else:
                    print(f"    Attempt {attempt + 1}: Failed to extract <decision> from LLM response for column '{col}'. Retrying...")
            
            if decision is None:
                print(f"  [Judgment] Failed to get valid judgment after {max_retries} attempts. Skipping column '{col}'.")
                continue

            if decision == "否":
                continue

            col_median_loss = valid_col_losses.median()
            threshold_loss = col_median_loss * self.median_multiplier
            detected_idx = valid_col_losses[valid_col_losses > threshold_loss].index.tolist()

            if len(detected_idx) < k_threshold:
                detected_idx = sorted_losses.head(k_threshold).index.tolist()
                
            if len(detected_idx) > len(self.dirty_df) * 0.1:
                detected_idx = sorted_losses.head(k_threshold).index.tolist()

            if detected_idx:
                detected_idx.sort()
                result_dict[col] = detected_idx
        
        save_path = os.path.join(self.phase_semantic_dir, "semantic_detect.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=4)
            
        print(f"Detection completed. Result saved to: {save_path}")
        return result_dict, self.LLM_time, self.LLM_sleep_time

class SemanticCleaner:
    # Initialize semantic cleaning configuration and load clean data.
    def __init__(self, dataset: str, original_dirty_path: str, clean_path: str,
                 debug_mode=True, model_path='',
                 result_root=DEFAULT_RESULT_ROOT,
                 fasttext_model_path: str = DEFAULT_FASTTEXT_MODEL_PATH,
                 llm_base_url: str = DEFAULT_LLM_BASE_URL, llm_api_key: str = DEFAULT_LLM_API_KEY, llm_model: str = DEFAULT_LLM_MODEL):
        self.dataset = dataset
        self.original_dirty_path = original_dirty_path
        self.clean_path = clean_path
        self.debug_mode = debug_mode
        self.model_path = model_path
        self.result_root = result_root
        self.fasttext_model_path = fasttext_model_path
        self.llm_base_url = llm_base_url
        self.llm_api_key = llm_api_key
        self.llm_model = llm_model
        self.LLM_sleep_time = 0
        self.LLM_time = 0
        
        self.clean_df = pd.read_csv(clean_path, dtype=str).fillna('nan')
        self.phase_semantic_dir = None
        

    # Run typo and semantic detectors, then merge their findings.
    def _run_detectors_and_merge(self, fd_cleaned_csv_path: str) -> dict:
        print("\n--- Running TypoDetector and SemanticDetector ---")
        
        typo_detector = TypoDetector(
            dirty_path=fd_cleaned_csv_path, 
            clean_path=self.clean_path, 
            save_dir=os.path.dirname(self.phase_semantic_dir)
        )
        print("Executing TypoDetector...")
        typo_result = typo_detector.detect()
        
        semantic_detector = SemanticDetector(
            dirty_path=fd_cleaned_csv_path,
            clean_path=self.clean_path,
            model_path=self.model_path,
            save_dir=os.path.dirname(self.phase_semantic_dir),
            fasttext_model_path=self.fasttext_model_path,
            llm_base_url=self.llm_base_url,
            llm_api_key=self.llm_api_key,
            llm_model=self.llm_model,
        )
        print("Executing SemanticDetector...")
        semantic_result, temp_time3, temp_time4 = semantic_detector.detect()
        self.LLM_time += temp_time3
        self.LLM_sleep_time += temp_time4
        
        print("Merging detection results...")
        merged_result = {}
        all_cols = set(typo_result.keys()).union(set(semantic_result.keys()))
        
        for col in all_cols:
            typo_indices = typo_result.get(col, [])
            semantic_indices = semantic_result.get(col, [])
            merged_indices = sorted(list(set(typo_indices + semantic_indices)))
            merged_result[col] = merged_indices
            
        merged_save_path = os.path.join(self.phase_semantic_dir, "merged_semantic_detect.json")
        with open(merged_save_path, "w", encoding="utf-8") as f:
            json.dump(merged_result, f, ensure_ascii=False, indent=4)
            
        print(f"Merged detection result saved to: {merged_save_path}")
        return merged_result

    # Evaluate merged semantic detection results against clean data.
    def _evaluate_detection(self, merged_detect_dict: dict) -> dict:
        print("\n--- Evaluating Semantic Detection ---")
        
        global_metrics = {
            "all_need_detect": 0,
            "all_detected": 0,
            "correctly_detect": 0,
            "wrongly_detect": 0,
            "missing_errors": 0
        }
        
        col_metrics_dict = {}
        
        for col in self.current_dirty_df.columns:
            if str(col).startswith("__"): 
                continue
                
            col_need_detect = 0
            col_detected = 0
            col_correct = 0
            col_wrong = 0
            
            wrongly_detect_log = []
            missing_errors_log = []
            
            detected_indices = set(merged_detect_dict.get(col, []))
            
            for idx, current_val in self.current_dirty_df[col].items():
                clean_val = self.clean_df.loc[idx, col]
                
                is_true_error = (str(current_val) != str(clean_val)) or (str(clean_val).lower() in ['nan', 'none', 'null', 'empty', ''])
                is_detected = idx in detected_indices
                
                if is_true_error:
                    col_need_detect += 1
                if is_detected:
                    col_detected += 1
                    if is_true_error:
                        col_correct += 1
                    else:
                        col_wrong += 1
                        wrongly_detect_log.append(f"Row {idx}: {current_val}")
                else:
                    if is_true_error:
                        missing_errors_log.append(f"Row {idx}: {current_val} -> {clean_val}")
                        
            col_missing = col_need_detect - col_correct
            
            global_metrics["all_need_detect"] += col_need_detect
            global_metrics["all_detected"] += col_detected
            global_metrics["correctly_detect"] += col_correct
            global_metrics["wrongly_detect"] += col_wrong
            global_metrics["missing_errors"] += col_missing
            
            if col_need_detect > 0 or col_detected > 0:
                pre = col_correct / (col_detected + 1e-8)
                rec = col_correct / (col_need_detect + 1e-8)
                f1 = 2 * pre * rec / (pre + rec + 1e-8)
                
                col_metrics_dict[col] = {
                    "all_need_detect": col_need_detect,
                    "all_detected": col_detected,
                    "correctly_detect": col_correct,
                    "wrongly_detect": col_wrong,
                    "missing_errors": col_missing,
                    "Precision": round(pre, 4),
                    "Recall": round(rec, 4),
                    "F1": round(f1, 4),
                    "wrongly_detect_logs": wrongly_detect_log[:20],
                    "missing_errors_logs": missing_errors_log[:20]
                }
                
        g_pre = global_metrics["correctly_detect"] / (global_metrics["all_detected"] + 1e-8)
        g_rec = global_metrics["correctly_detect"] / (global_metrics["all_need_detect"] + 1e-8)
        g_f1 = 2 * g_pre * g_rec / (g_pre + g_rec + 1e-8)
        
        global_metrics["Precision"] = round(g_pre, 4)
        global_metrics["Recall"] = round(g_rec, 4)
        global_metrics["F1"] = round(g_f1, 4)
        
        final_evaluation = {
            "Global_Metrics": global_metrics,
            "Column_Metrics": col_metrics_dict
        }
        
        eval_save_path = os.path.join(self.phase_semantic_dir, "merged_semantic_detection_evaluation.json")
        with open(eval_save_path, "w", encoding="utf-8") as f:
            json.dump(final_evaluation, f, ensure_ascii=False, indent=4)
            
        print(f"Global Detection Metrics -> F1: {global_metrics['F1']}, Precision: {global_metrics['Precision']}, Recall: {global_metrics['Recall']}")
        print(f"Evaluation saved to: {eval_save_path}")
        
        return final_evaluation
    
    # Extract FastText embeddings for rows.
    def _get_row_embeddings(self) -> np.ndarray:
        print(f"\n--- Extracting Cell-wise Embeddings for Clustering ---")
        print(f"Loading FastText Model from {self.fasttext_model_path}...")
        ft_model = fasttext.load_model(self.fasttext_model_path)
        
        embeddings = []
        valid_cols = [c for c in self.current_dirty_df.columns if not str(c).startswith("__")]
        
        for idx, row in self.current_dirty_df.iterrows():
            row_embs = []
            for col in valid_cols:
                val = str(row[col]).strip()
                
                if val.lower() in ['nan', 'none', 'null', 'empty', '']:
                    cell_emb = np.zeros(ft_model.get_dimension())
                else:
                    cell_emb = ft_model.get_sentence_vector(val)
                    
                row_embs.append(cell_emb)
            
            full_row_emb = np.concatenate(row_embs)
            embeddings.append(full_row_emb)
            
        return np.array(embeddings)
    
    # Cluster rows and group similar tuples for repair.
    def _cluster_for_repair(self, merged_detect_dict: dict) -> dict:
        embeddings = self._get_row_embeddings()
        
        target_dim = len(self.current_dirty_df.columns) * 10
        n_components = min(target_dim, embeddings.shape[0], embeddings.shape[1])
        print(f"\nApplying PCA to reduce dimensions from {embeddings.shape[1]} to {n_components}...")
        
        pca = PCA(n_components=n_components, random_state=42)
        reduced_embs = pca.fit_transform(embeddings)

        print("Executing HDBSCAN clustering...")
        hdbscan_model = HDBSCAN(min_cluster_size=5, metric='euclidean', cluster_selection_epsilon=0.0)
        cluster_labels = hdbscan_model.fit_predict(reduced_embs)
        
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        print(f"-> HDBSCAN initially found {n_clusters} clusters and {n_noise} noise points (-1).")

        unique_clusters = set(cluster_labels) - {-1}
        if n_noise > 0 and len(unique_clusters) > 0:
            print("Re-assigning noise points to their nearest valid clusters...")
            centroids = {c: reduced_embs[cluster_labels == c].mean(axis=0) for c in unique_clusters}
            
            for i in range(len(cluster_labels)):
                if cluster_labels[i] == -1:
                    point = reduced_embs[i]
                    closest_c = min(centroids.keys(), key=lambda c: np.linalg.norm(point - centroids[c]))
                    cluster_labels[i] = closest_c
                    
            print("Re-assignment complete. No more -1 labels.")
            
        elif len(unique_clusters) == 0:
            print("Warning: HDBSCAN found ONLY noise. Grouping all into cluster 0.")
            cluster_labels = np.zeros(len(cluster_labels), dtype=int)
        
        clustered_df = self.current_dirty_df.copy()
        clustered_df['clusterID'] = cluster_labels
        
        all_error_indices = set()
        for indices in merged_detect_dict.values():
            all_error_indices.update(indices)
        clustered_df['has_error'] = clustered_df.index.map(lambda x: 1 if x in all_error_indices else 0)
        
        save_path = os.path.join(self.phase_semantic_dir, f"{self.dataset}_semantic_clustered.csv")
        clustered_df.to_csv(save_path, index=True)
        print(f"Clustered data saved to: {save_path}")
        
        cluster_mapping = {}
        for idx, cluster_id in zip(clustered_df.index, cluster_labels):
            if cluster_id not in cluster_mapping:
                cluster_mapping[cluster_id] = []
            cluster_mapping[cluster_id].append(idx)
            
        clusters_with_errors = 0
        for cluster_id, indices in cluster_mapping.items():
            if any(idx in all_error_indices for idx in indices):
                clusters_with_errors += 1
                
        print(f"-> Statistic: {clusters_with_errors} out of {len(cluster_mapping)} clusters contain errors.")
            
        return cluster_mapping

    # Call the configured LLM with the given prompt.
    def _call_llm(self, prompt):
        if self.debug_mode:
            return 'debug'

        llm_args = self._get_llm_args()
        if llm_args is None:
            return "No LLM specified or LLM not recognized."

        base_url, model, api_key = llm_args
        return call_llm(prompt=prompt, base_url=base_url, api_key=api_key, model=model)

    # Return configured LLM arguments when available.
    def _get_llm_args(self):
        if not (self.llm_base_url and self.llm_api_key and self.llm_model):
            return None
        return self.llm_base_url, self.llm_model, self.llm_api_key

    # Save an LLM prompt and response for audit logs.
    def _save_prompt_response(self, phase: str, cluster_id: str, attempt: int, prompt: str, response: str, pr_dir: str):
        filename = os.path.join(pr_dir, f"{phase}_cluster_{cluster_id}_attempt_{attempt}.txt")
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=== PROMPT ===\n")
            f.write(prompt + "\n\n")
            f.write("=== RESPONSE ===\n")
            f.write(str(response) + "\n")

    # Build an LLM prompt for repairing semantic anomalies.
    def _build_semantic_repair_prompt(self, cluster_context_str: str, error_instructions_str: str,
                                      all_dirty_rows_str: str, all_clean_rows_str: str, col_labeled_list_str: str):
        return f"""[Role] Data Quality Expert
[Task] Semantic Anomaly Repair

[Global Context (All Sampled Tuples)]
Original Dirty Tuples:
{all_dirty_rows_str}
Clean Tuples (Ground Truth):
{all_clean_rows_str}

[Labeled Semantic Errors for Reference]
{col_labeled_list_str}

[Current Cluster Data]
Here is a cluster of structurally and semantically similar tuples. Some tuples are healthy (provided as clean context), and some contain specific semantic errors that need to be repaired.
{cluster_context_str}

[Specific Repair Tasks]
The following cells have been detected as containing semantic errors (e.g., typos, contradictory values, out-of-domain terms). 
{error_instructions_str}

[Instruction]
Analyze the contextual tuples in the cluster and the global context. Determine the correct, intended value for the erroneous cells.
Provide your repair results strictly using the following XML format. Do not change the row index or column name.

Important Guidelines:
1. The error detection results are not absolutely accurate. If you believe a flagged value is actually correct, you can choose not to repair it.
2. If you are not highly confident about what a value should be repaired to, you can also choose not to repair it.
3. For values that were NOT flagged as errors, if you spot an obvious error and clearly know what it should be modified to, go ahead and repair it.
4. Quoting Rule: Do NOT wrap the corrected value in extra quotes (like "value" or 'value') unless the actual string is legitimately supposed to contain quotes.
5. Provide a brief `<reason>` for each repair you make.
6. Reply strictly in the following XML format: 
[Output Format Example]
<repairs>
    <repair row="15" col="city">
        <value>Los Angeles</value>
        <reason>Corrected typo 'Los Angele' to match the clean context pattern in this cluster.</reason>
    </repair>
    <repair row="42" col="zip_code">
        <value>90001</value>
        <reason>Filled missing value based on the city 'Los Angeles' and state 'CA' in the same tuple.</reason>
    </repair>
</repairs>
"""

    # Extract semantic repair actions from an LLM response.
    def _extract_repair_response(self, response: str):
        repairs = []
        
        if not response:
            return repairs
            
        pattern = r'<repair\s+row="(\d+)"\s+col="([^"]+)">\s*<value>(.*?)</value>\s*<reason>(.*?)</reason>\s*</repair>'
        matches = re.finditer(pattern, response, re.IGNORECASE | re.DOTALL)
        
        for match in matches:
            row_idx = int(match.group(1))
            col_name = match.group(2).strip()
            repaired_val = match.group(3).strip()
            repair_reason = match.group(4).strip()
            repairs.append((row_idx, col_name, repaired_val, repair_reason))
            
        return repairs
    
    # Evaluate semantic repair results against clean data.
    def _evaluate_repair(self, repair_log: list, repair_reasons_dict: dict) -> dict:
        print("\n--- Evaluating Semantic Repair ---")
        
        global_metrics = {
            "all_need_repair": 0, "all_repaired": 0, "wrong_2_right": 0,
            "wrong_2_wrong": 0, "right_2_wrong": 0, "wrong_not_change": 0
        }
        
        col_metrics_dict = {}
        
        wrong_2_right_logs = []
        wrong_2_wrong_logs = []
        right_2_wrong_logs = []
        wrong_not_change_logs = []

        for col in self.current_dirty_df.columns:
            if str(col).startswith("__"): 
                continue
                
            col_metrics = { "all_need_repair": 0, "all_repaired": 0, "wrong_2_right": 0, 
                            "wrong_2_wrong": 0, "right_2_wrong": 0, "wrong_not_change": 0 }
            
            for idx, dirty_val in self.current_dirty_df[col].items():
                clean_val = self.clean_df.loc[idx, col]
                repaired_val = self.cleaned_df.loc[idx, col]
                
                is_true_error = (str(dirty_val) != str(clean_val))
                if is_true_error:
                    col_metrics["all_need_repair"] += 1
                    
                if str(repaired_val) != str(dirty_val):
                    col_metrics["all_repaired"] += 1
                    
                    reason_str = repair_reasons_dict.get((idx, col), "No reason recorded")
                    log_str = f"Row {idx} [{col}]: '{dirty_val}' -> '{repaired_val}' (Truth: '{clean_val}') | Reason: {reason_str}"
                    
                    if str(repaired_val) == str(clean_val):
                        col_metrics["wrong_2_right"] += 1
                        wrong_2_right_logs.append(log_str)
                    else:
                        if is_true_error:
                            col_metrics["wrong_2_wrong"] += 1
                            wrong_2_wrong_logs.append(log_str)
                        else:
                            col_metrics["right_2_wrong"] += 1
                            right_2_wrong_logs.append(log_str)
                else:
                    if is_true_error:
                        col_metrics["wrong_not_change"] += 1
                        wrong_not_change_logs.append(f"Row {idx} [{col}]: '{dirty_val}' -> '{repaired_val}' (Truth: '{clean_val}')")

            for k in global_metrics.keys():
                global_metrics[k] += col_metrics[k]

            if col_metrics["all_need_repair"] > 0 or col_metrics["all_repaired"] > 0:
                pre = col_metrics["wrong_2_right"] / (col_metrics["all_repaired"] + 1e-8)
                rec = col_metrics["wrong_2_right"] / (col_metrics["all_need_repair"] + 1e-8)
                f1 = 2 * pre * rec / (pre + rec + 1e-8)
                col_metrics["Precision"] = round(pre, 4)
                col_metrics["Recall"] = round(rec, 4)
                col_metrics["F1"] = round(f1, 4)
                col_metrics_dict[col] = col_metrics

        g_pre = global_metrics["wrong_2_right"] / (global_metrics["all_repaired"] + 1e-8)
        g_rec = global_metrics["wrong_2_right"] / (global_metrics["all_need_repair"] + 1e-8)
        g_f1 = 2 * g_pre * g_rec / (g_pre + g_rec + 1e-8)
        
        global_metrics["Precision"] = round(g_pre, 4)
        global_metrics["Recall"] = round(g_rec, 4)
        global_metrics["F1"] = round(g_f1, 4)
        
        final_evaluation = {
            "Global_Metrics": global_metrics,
            "Column_Metrics": col_metrics_dict,
            "Logs": {
                "wrong_2_right": wrong_2_right_logs,
                "wrong_2_wrong": wrong_2_wrong_logs,
                "right_2_wrong": right_2_wrong_logs,
                "wrong_not_change": wrong_not_change_logs,
                "llm_repair_attempts": repair_log
            }
        }
        
        eval_save_path = os.path.join(self.phase_semantic_dir, "semantic_repair_evaluation.json")
        with open(eval_save_path, "w", encoding="utf-8") as f:
            json.dump(final_evaluation, f, ensure_ascii=False, indent=4)
            
        print(f"Global Repair Metrics -> F1: {global_metrics['F1']}, Precision: {global_metrics['Precision']}, Recall: {global_metrics['Recall']}")
        print(f"Repair evaluation saved to: {eval_save_path}")
        return final_evaluation
    
    # Run the full semantic cleaning pipeline and save outputs.
    def run(self):
        
        start_time = time.time()
        
        fd_cleaner = FDCleaner(
            dataset=self.dataset,
            original_dirty_path=self.original_dirty_path,
            clean_path=self.clean_path,
            debug_mode=self.debug_mode,
            llm_base_url=self.llm_base_url,
            llm_api_key=self.llm_api_key,
            llm_model=self.llm_model,
            fasttext_model_path=self.fasttext_model_path,
            result_root=self.result_root
        )
        base_dir, temp_time5, temp_time6 = fd_cleaner.run()
        self.LLM_time += temp_time5
        self.LLM_sleep_time += temp_time6
        phase_fd_dir = os.path.join(base_dir, 'phase_fd')
        phase_label_dir = os.path.join(base_dir, 'phase_label')
        self.phase_semantic_dir = os.path.join(base_dir, 'phase_semantic')
        os.makedirs(self.phase_semantic_dir, exist_ok=True)
        fd_cleaned_csv = os.path.join(phase_fd_dir, f"{self.dataset}_fd_cleaned.csv")

        pr_dir = os.path.join(self.phase_semantic_dir, 'prompt_response')
        os.makedirs(pr_dir, exist_ok=True)
        
        self.current_dirty_df = pd.read_csv(fd_cleaned_csv, dtype=str).fillna('nan')
        self.cleaned_df = self.current_dirty_df.copy()
        
        with open(os.path.join(phase_label_dir, 'labeled_data.json'), 'r', encoding='utf-8') as f:
            labeled_data = json.load(f)
        with open(os.path.join(phase_label_dir, 'tuples_data.json'), 'r', encoding='utf-8') as f:
            tuples_data = json.load(f)
            
        all_dirty_rows_str = "\n".join([f"Row {k}: {v}" for k, v in tuples_data.get("dirty", {}).items()])
        all_clean_rows_str = "\n".join([f"Row {k}: {v}" for k, v in tuples_data.get("clean", {}).items()])
        col_labeled_list_str = "\n".join([f"Row {item[0]} [{item[1]}]: '{item[2]}' -> '{item[3]}' (Reason: {item[4]})" for item in labeled_data])
        if not col_labeled_list_str: col_labeled_list_str = "None"

        merged_detect_dict = self._run_detectors_and_merge(fd_cleaned_csv)
        self._evaluate_detection(merged_detect_dict)
        cluster_mapping = self._cluster_for_repair(merged_detect_dict)
        
        row_to_error_cols = {}
        for col, indices in merged_detect_dict.items():
            for idx in indices:
                if idx not in row_to_error_cols:
                    row_to_error_cols[idx] = []
                row_to_error_cols[idx].append(col)

        print("\n--- Starting LLM Semantic Repair ---")
        repair_log = []
        repair_reasons_dict = {}
        
        for cluster_id, indices in cluster_mapping.items():
            error_rows = [idx for idx in indices if idx in row_to_error_cols]
            clean_rows = [idx for idx in indices if idx not in row_to_error_cols]
            
            if not error_rows:
                continue
            
            if len(error_rows) > 100:
                continue
                
            print(f"> Processing Cluster {cluster_id} ({len(error_rows)} errors)...")
            
            context_indices = error_rows + clean_rows[:5]
            cluster_context_lines = []
            for idx in context_indices:
                row_data = {k: v for k, v in self.current_dirty_df.loc[idx].to_dict().items() if not str(k).startswith("__")}
                tag = "[ERROR ROW]" if idx in error_rows else "[CLEAN CONTEXT]"
                cluster_context_lines.append(f"Row {idx} {tag}: {row_data}")
            cluster_context_str = "\n".join(cluster_context_lines)
            
            error_instructions = []
            for idx in error_rows:
                for col in row_to_error_cols[idx]:
                    val = self.current_dirty_df.loc[idx, col]
                    error_instructions.append(f"- Row {idx}, Column '{col}': Current suspicious value is '{val}'")
            error_instructions_str = "\n".join(error_instructions)
            
            prompt = self._build_semantic_repair_prompt(
                cluster_context_str, error_instructions_str,
                all_dirty_rows_str, all_clean_rows_str, col_labeled_list_str
            )
            
            repairs = []
            for attempt in range(3):
                response = self._call_llm(prompt)
                self._save_prompt_response("semantic_repair", str(cluster_id), attempt + 1, prompt, response, pr_dir)
                
                repairs = self._extract_repair_response(response)
                if repairs:
                    break
                print(f"  Attempt {attempt + 1}: Failed to extract <repair> tags. Retrying...")
                
            for row_idx, col_name, rep_val, rep_reason in repairs:
                try:
                    self.cleaned_df.loc[row_idx, col_name] = rep_val
                    repair_reasons_dict[(row_idx, col_name)] = rep_reason
                except Exception as e:
                    print(f"  [Warning] Could not apply repair for Row {row_idx}, Col '{col_name}': {e}")
                    
            repair_log.append({
                "cluster_id": str(cluster_id),
                "num_errors_detected": len(error_rows),
                "num_repairs_extracted": len(repairs)
            })

        cleaned_csv_path = os.path.join(self.phase_semantic_dir, f"{self.dataset}_semantic_cleaned.csv")
        self.cleaned_df.to_csv(cleaned_csv_path, index=False)
        
        self._evaluate_repair(repair_log, repair_reasons_dict)
        
        print(f"\nSaved Phase 3 cleaned data to: {cleaned_csv_path}")
        print("\n=== Final Global Evaluation for Full Pipeline ===")
        measure_repair(self.clean_path, self.original_dirty_path, cleaned_csv_path)
        
        base_dir = os.path.dirname(self.phase_semantic_dir)
        
        phases_txt = [
            ('Format', os.path.join(base_dir, 'phase_format', f"{self.dataset}_format_cleaned.txt")),
            ('FD', os.path.join(base_dir, 'phase_fd', f"{self.dataset}_fd_cleaned.txt")),
            ('Semantic', os.path.join(self.phase_semantic_dir, f"{self.dataset}_semantic_cleaned.txt"))
        ]
        
        result_txt_path = os.path.join(base_dir, "result.txt")
        try:
            with open(result_txt_path, 'w', encoding='utf-8') as out_f:
                for phase_name, txt_path in phases_txt:
                    out_f.write(f"=== {phase_name} Phase ===\n")
                    if os.path.exists(txt_path):
                        with open(txt_path, 'r', encoding='utf-8') as in_f:
                            for i, line in enumerate(in_f):
                                if i < 9:
                                    out_f.write(line)
                                else:
                                    break
                    else:
                        out_f.write(f"Warning: File not found ({txt_path})\n")
                    out_f.write("\n")
            print(f"\nAggregated evaluation results saved to: {result_txt_path}")
        except Exception as e:
            print(f"\n[Warning] Failed to aggregate result.txt: {e}")

        final_csv_dst = os.path.join(base_dir, f"{self.dataset}_cleaned.csv")
        try:
            shutil.copy2(cleaned_csv_path, final_csv_dst)
            print(f"Final fully-cleaned dataset copied to: {final_csv_dst}")
        except Exception as e:
            print(f"[Warning] Failed to copy final CSV: {e}")

        print(f"\nPhase 3 Semantic Cleaning completed. Check results in: {self.phase_semantic_dir}")
        end_time = time.time()
        with open(os.path.join(base_dir, "time.txt"), 'w', encoding='utf-8') as f:
            f.write(f"Total time: {end_time - start_time:.2f} seconds\n")
            f.write(f"LLM time: {self.LLM_time:.2f} seconds\n")
            f.write(f"LLM sleep time: {self.LLM_sleep_time:.2f} seconds\n")
            f.write(f"System time: {end_time - start_time - self.LLM_time:.2f} seconds\n")
            f.write(f"System+LLM time: {end_time - start_time - self.LLM_sleep_time:.2f} seconds")
        return base_dir

    
# Run the semantic cleaning pipeline and return output paths.
def run_semantic_cleaner(
    dataset: str,
    original_dirty_path: str,
    clean_path: str,
    result_root: str = DEFAULT_RESULT_ROOT,
    debug_mode: bool = False,
    llm_base_url: str = DEFAULT_LLM_BASE_URL,
    llm_api_key: str = DEFAULT_LLM_API_KEY,
    llm_model: str = DEFAULT_LLM_MODEL,
    model_path: str = DEFAULT_SEMANTIC_MODEL_PATH,
    fasttext_model_path: str = DEFAULT_FASTTEXT_MODEL_PATH,
):
    """Run the full semantic cleaning pipeline and return key output paths."""
    cleaner = SemanticCleaner(
        dataset=dataset,
        original_dirty_path=original_dirty_path,
        clean_path=clean_path,
        debug_mode=debug_mode,
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        llm_model=llm_model,
        model_path=model_path,
        result_root=result_root,
        fasttext_model_path=fasttext_model_path,
    )
    base_dir = cleaner.run()
    phase_semantic_dir = cleaner.phase_semantic_dir

    return {
        "dataset": dataset,
        "base_dir": base_dir,
        "phase_semantic_dir": phase_semantic_dir,
        "cleaned_csv_path": os.path.join(phase_semantic_dir, f"{dataset}_semantic_cleaned.csv"),
        "final_csv_path": os.path.join(base_dir, f"{dataset}_cleaned.csv"),
        "result_txt_path": os.path.join(base_dir, "result.txt"),
        "time_txt_path": os.path.join(base_dir, "time.txt"),
        "llm_time": cleaner.LLM_time,
        "llm_sleep_time": cleaner.LLM_sleep_time,
    }


if __name__ == "__main__":
    datasets = ['beers', 'hospital', 'flights', 'tax', 'jobs']
    dataset_root = 'data/datasets'
    result_root = DEFAULT_RESULT_ROOT
    model_path = DEFAULT_SEMANTIC_MODEL_PATH

    for dataset in datasets:
        print(f"\n\n=== Processing Dataset: {dataset} ===")
        result = run_semantic_cleaner(
            dataset=dataset,
            original_dirty_path=f'{dataset_root}/{dataset}/{dataset}_error-01.csv',
            clean_path=f'{dataset_root}/{dataset}/{dataset}_clean.csv',
            result_root=result_root,
            debug_mode=False,
            llm_base_url=DEFAULT_LLM_BASE_URL,
            llm_api_key=DEFAULT_LLM_API_KEY,
            llm_model=DEFAULT_LLM_MODEL,
            model_path=model_path,
        )
        print(result)
    
    

import pandas as pd
import numpy as np
import os
import sys
import json
import re
import shutil
from tqdm import tqdm
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from .measure import measure_repair
    from .AnomalyTupleSelector import AnomalyTupleSelector
    from .FormatCleaner import FormatCleaner
    from .LLM_response import call_llm
    from .project_config import DEFAULT_FASTTEXT_MODEL_PATH, DEFAULT_LLM_API_KEY, DEFAULT_LLM_BASE_URL, DEFAULT_LLM_MODEL, DEFAULT_RESULT_ROOT
except ImportError:
    from measure import measure_repair
    from AnomalyTupleSelector import AnomalyTupleSelector
    from FormatCleaner import FormatCleaner
    from LLM_response import call_llm
    from project_config import DEFAULT_FASTTEXT_MODEL_PATH, DEFAULT_LLM_API_KEY, DEFAULT_LLM_BASE_URL, DEFAULT_LLM_MODEL, DEFAULT_RESULT_ROOT

class PrintLogger:
    # Initialize logger output to console and file.
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')

    # Write messages to both stdout and the log file.
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    # Flush stdout and log file buffers.
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    # Delegate unknown attributes to stdout.
    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

class FDCleaner:
    # Initialize FD cleaner settings and load the clean data.
    def __init__(self, dataset: str, original_dirty_path: str,clean_path: str, debug_mode=True, result_root=DEFAULT_RESULT_ROOT, fasttext_model_path: str = DEFAULT_FASTTEXT_MODEL_PATH, llm_base_url: str = DEFAULT_LLM_BASE_URL, llm_api_key: str = DEFAULT_LLM_API_KEY, llm_model: str = DEFAULT_LLM_MODEL):
        self.dataset = dataset
        self.original_dirty_path = original_dirty_path
        self.clean_path = clean_path
        self.debug_mode = debug_mode
        self.result_root = result_root
        self.fasttext_model_path = fasttext_model_path
        self.llm_base_url = llm_base_url
        self.llm_api_key = llm_api_key
        self.llm_model = llm_model
        self.LLM_sleep_time = 0
        self.LLM_time = 0
        
        self.clean_df = pd.read_csv(clean_path, dtype=str).fillna('nan')
        self.dirty_df = None
        self.cleaned_df = None

    # Call the configured LLM with the given prompt.
    def _call_llm(self, prompt):
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

    # Return a test judgment response for debugging.
    def _call_llm_test_judgment(self, prompt):
        return '''<decision>Yes</decision>
<reason>
xxxxxxx
</reason>'''

    # Return a test fallback response for debugging.
    def _call_llm_test_fallback(self, prompt):
        return '''<suggested_value>BAPTIST MEDICAL CENTER</suggested_value>
<reason>
xxxxxxx
</reason>'''

    # Save an LLM prompt and response for audit logs.
    def _save_prompt_response(self, phase, col, attempt, prompt, response, pr_dir):
        safe_col = col.replace("/", "_").replace("\\", "_")
        filename = os.path.join(pr_dir, f"{phase}_{safe_col}_attempt_{attempt}.txt")
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=== PROMPT ===\n")
            f.write(prompt + "\n\n")
            f.write("=== RESPONSE ===\n")
            f.write(str(response) + "\n")

    # Extract an FD judgment decision and reason from an LLM response.
    def _extract_judgment(self, response: str):
        decision = None
        reason = ""
        
        match_decision = re.search(r'<decision>(.*?)</decision>', response, re.DOTALL | re.IGNORECASE)
        if match_decision:
            d_text = match_decision.group(1).strip()
            if "是" in d_text or "yes" in d_text.lower():
                decision = "是"
            elif "否" in d_text or "no" in d_text.lower():
                decision = "否"
                
        match_reason = re.search(r'<reason>(.*?)</reason>', response, re.DOTALL | re.IGNORECASE)
        if match_reason:
            reason = match_reason.group(1).strip()
            
        return decision, reason

    # Extract a fallback value and reason from an LLM response.
    def _extract_fallback(self, response: str):
        suggested_value = None
        reason = ""
        
        match_val = re.search(r'<suggested_value>(.*?)</suggested_value>', response, re.DOTALL | re.IGNORECASE)
        if match_val:
            suggested_value = match_val.group(1).strip()
            
        match_reason = re.search(r'<reason>(.*?)</reason>', response, re.DOTALL | re.IGNORECASE)
        if match_reason:
            reason = match_reason.group(1).strip()
            
        return suggested_value, reason

    # Build an LLM prompt for judging FD consistency.
    def _build_fd_judgment_prompt(self, col, entity_col, sample_groups_str, all_dirty_rows_str, all_clean_rows_str, col_labeled_list_str):
        return f"""[Role] Data Quality Expert
[Task] Determine if the column '{col}' has a strict Functional Dependency (FD) on the entity identifier '{entity_col}'.

[Global Context (All Sampled Tuples)]
Original Dirty Tuples:
{all_dirty_rows_str}
Clean Tuples (Ground Truth):
{all_clean_rows_str}

[Labeled Errors for Target Column '{col}']
{col_labeled_list_str}

[Grouped Data Samples]
The dirty data has currently undergone format cleaning, but issues such as functional dependency errors may still exist.
Below are 2 random groups of data, grouped by the entity identifier '{entity_col}'. Please focus specifically on the target column '{col}':
{sample_groups_str}

[Instruction]
Based on the semantics of '{col}' and its relationship with '{entity_col}', and by carefully analyzing both the [Grouped Data Samples] and the Clean Tuples (Ground Truth), should all rows within the same '{entity_col}' group share the EXACT same value for '{col}'?
Please analyze and answer strictly using the following XML tags:
<decision>是</decision> (if it should be strictly consistent) OR <decision>否</decision> (if variation is allowed)
<reason>Briefly explain your reasoning based on the data semantics and the provided truth samples.</reason>
"""

    # Build an LLM prompt for resolving an FD conflict.
    def _build_fd_fallback_prompt(self, col, entity_col, entity_val, specific_group_str, all_dirty_rows_str, all_clean_rows_str):
        return f"""[Role] Data Quality Expert
[Task] Resolve a data conflict in column '{col}' for a specific entity group.

[Global Context]
Original Dirty Tuples:
{all_dirty_rows_str}
Clean Tuples (Ground Truth):
{all_clean_rows_str}

[Target Group for Conflict Resolution]
The dirty data has currently undergone format cleaning, but issues such as functional dependency errors may still exist.
Entity Identifier '{entity_col}' = '{entity_val}' group data. Please focus specifically on the target column '{col}':
{specific_group_str}

[Instruction]
We have established that the column '{col}' must maintain a consistent value within the same '{entity_col}' group. However, the data above contains conflicting values and no single value has a strict majority (>50%).
Your task is to analyze the conflicting values in this group and determine the most correct, unified value to use.
Please answer strictly using the following XML tags:
<suggested_value>The exact corrected value to apply to the entire group</suggested_value>
<reason>Explain why you chose this value over the others.</reason>

[Note]
Do not include quotes around the value inside the tags unless the value itself requires them.
"""

    # Evaluate FD repair metrics against the clean data.
    def _evaluate_fd_repair(self, col: str, original_dirty_df, current_cleaned_df, results_dict):
        wrong_2_right = 0
        right_2_wrong = 0
        wrong_2_wrong = 0
        wrong_not_change = 0
        
        all_repaired = 0
        all_need_repair = 0
        
        wrong_2_right_log = []
        right_2_wrong_log = []
        wrong_2_wrong_log = []
        wrong_not_change_log = []

        for idx, dirty_val in original_dirty_df[col].items():
            clean_val = self.clean_df.loc[idx, col]
            repaired_val = current_cleaned_df.loc[idx, col]
            
            is_true_error = (str(dirty_val) != str(clean_val))
            if is_true_error:
                all_need_repair += 1
                
            if str(repaired_val) != str(dirty_val):
                all_repaired += 1
                log_str = f"Row {idx}: '{dirty_val}' -> '{repaired_val}' (Truth: '{clean_val}')"
                
                if str(repaired_val) == str(clean_val):
                    wrong_2_right += 1
                    wrong_2_right_log.append(log_str)
                else:
                    if is_true_error:
                        wrong_2_wrong += 1
                        wrong_2_wrong_log.append(log_str)
                    else:
                        right_2_wrong += 1
                        right_2_wrong_log.append(log_str)
            else:
                if is_true_error:
                    wrong_not_change += 1
                    wrong_not_change_log.append(f"Row {idx}: '{dirty_val}' -> '{repaired_val}' (Truth: '{clean_val}')")

        pre = wrong_2_right / (all_repaired + 1e-8)
        rec = wrong_2_right / (all_need_repair + 1e-8)
        f1 = 2 * pre * rec / (pre + rec + 1e-8)
        
        metrics = {
            "all_need_repair": all_need_repair,
            "all_repaired": all_repaired,
            "wrong_2_right (TP)": wrong_2_right,
            "wrong_2_wrong (FP_type1)": wrong_2_wrong,
            "right_2_wrong (FP_type2)": right_2_wrong,
            "wrong_not_change (FN)": wrong_not_change,
            "Precision": pre,
            "Recall": rec,
            "F1": f1
        }
        
        results_dict['repair_metrics'] = metrics
        results_dict['wrong_2_right_logs'] = wrong_2_right_log
        results_dict['right_2_wrong_logs'] = right_2_wrong_log
        results_dict['wrong_2_wrong_logs'] = wrong_2_wrong_log
        results_dict['wrong_not_change_logs'] = wrong_not_change_log
        
        return metrics

    # Run the FD cleaning phase and save outputs.
    def run(self):
        print("Running FormatCleaner first...")
        fc = FormatCleaner(
            self.dataset,
            self.original_dirty_path,
            self.clean_path,
            debug_mode=self.debug_mode,
            result_root=self.result_root,
            llm_base_url=self.llm_base_url,
            llm_api_key=self.llm_api_key,
            llm_model=self.llm_model,
            fasttext_model_path=self.fasttext_model_path
        )
        base_dir, temp_time3, temp_time4 = fc.run()
        self.LLM_time += temp_time3
        self.LLM_sleep_time += temp_time4
        phase_format_dir = os.path.join(base_dir, 'phase_format')
        phase_label_dir = os.path.join(os.path.dirname(phase_format_dir), 'phase_label')

        if not hasattr(self, 'phase_fd_dir') or 'phase_fd_dir' not in locals():
            phase_fd_dir = os.path.join(os.path.dirname(phase_format_dir), 'phase_fd')
            os.makedirs(phase_fd_dir, exist_ok=True)

        pr_dir = os.path.join(phase_fd_dir, 'prompt_response')
        os.makedirs(pr_dir, exist_ok=True)
        
        format_cleaned_csv = os.path.join(phase_format_dir, f"{self.dataset}_format_cleaned.csv")
        self.dirty_df = pd.read_csv(format_cleaned_csv, dtype=str).fillna('nan')
        self.cleaned_df = self.dirty_df.copy()
        
        original_stdout = sys.stdout
        log_path = os.path.join(phase_fd_dir, 'run_fd.log')
        sys.stdout = PrintLogger(log_path)
        
        start_time = datetime.now()
        
        try:
            print(f"\n--- Starting Phase 2: FD Cleaning ---")
            
            with open(os.path.join(phase_label_dir, 'values_stat.json'), 'r', encoding='utf-8') as f:
                values_stat = json.load(f)
            with open(os.path.join(phase_label_dir, 'labeled_data.json'), 'r', encoding='utf-8') as f:
                labeled_data = json.load(f)
            with open(os.path.join(phase_label_dir, 'tuples_data.json'), 'r', encoding='utf-8') as f:
                tuples_data = json.load(f)
                
            all_dirty_rows_str = "\n".join([f"Row {k}: {v}" for k, v in tuples_data.get("dirty", {}).items()])
            all_clean_rows_str = "\n".join([f"Row {k}: {v}" for k, v in tuples_data.get("clean", {}).items()])
            
            print("Detecting Entity Identifier and FD dependencies...")
            selector = AnomalyTupleSelector(self.dirty_df, fasttext_model_path=self.fasttext_model_path)
            entity_col, entity_col_score = selector._detect_global_entity_identifier()
            
            if entity_col is None:
                print("No clear Entity Identifier found. Aborting FD Phase.")
                return phase_fd_dir
                
            fd_scores = selector._detect_fd_dependent_columns(entity_col, fd_threshold=0.5)
            target_cols = [col for col, score in fd_scores.items() if score > 0.5 and col != entity_col]
            
            print(f"Target FD Columns to clean: {target_cols}")
            
            fd_judgments_dict = {}
            summary_lines = ["=== FD Cleaning Phase Summary ===\n"]
            
            for col in target_cols:
                print(f"\n>>> Processing Column: {col} (Entity: {entity_col}) <<<")
                col_results = {}
                fallback_logs = []
                safe_col = str(col).replace("/", "_").replace("\\", "_")
                
                valid_groups = [g for n, g in self.dirty_df.groupby(entity_col) if 5 <= len(g)]
                
                if len(valid_groups) >= 2:
                    sample_groups = pd.concat([valid_groups[0].head(10), valid_groups[1].head(10)])
                elif len(valid_groups) == 1:
                    sample_groups = valid_groups[0].head(10)
                else:
                    sample_groups = self.dirty_df.head(10)
                
                sample_group_lines = []
                for idx, row in sample_groups.iterrows():
                    valid_data = {k: v for k, v in row.to_dict().items() if not str(k).startswith("__")}
                    sample_group_lines.append(f"Row {idx}: {valid_data}")
                sample_groups_str = "\n".join(sample_group_lines)
                
                col_labels = [item for item in labeled_data if item[1] == col]
                col_labeled_list_str = "\n".join([f"Row {item[0]}: '{item[2]}' -> '{item[3]}' (Reason: {item[4]})" for item in col_labels])
                if not col_labeled_list_str: col_labeled_list_str = "None"
                
                judgment_prompt = self._build_fd_judgment_prompt(col, entity_col, sample_groups_str, all_dirty_rows_str, all_clean_rows_str, col_labeled_list_str)                
                
                decision = None
                reason = ""
                max_retries = 3
                
                for attempt in range(max_retries):
                    if self.debug_mode:
                        j_response = self._call_llm_test_judgment(judgment_prompt)
                    else:
                        j_response = self._call_llm(judgment_prompt)
                        
                    self._save_prompt_response("judgment", safe_col, attempt + 1, judgment_prompt, j_response, pr_dir)
                    decision, reason = self._extract_judgment(j_response)
                    
                    if decision is not None:
                        break
                    else:
                        print(f"    Attempt {attempt + 1}: Failed to extract <decision> from LLM response. Retrying...")
                
                if decision is None:
                    print(f"  [Judgment] Failed to get valid judgment after {max_retries} attempts. Skipping column '{col}'.")
                    summary_lines.append(f"Column '{col}': Skipped (Failed to extract judgment after {max_retries} attempts).\n")
                    continue
                
                fd_judgments_dict[col] = {
                    "decision": decision,
                    "reason": reason
                }
                
                if decision == "否":
                    print(f"  [Judgment] LLM evaluated: NO. Skipping column '{col}'.")
                    summary_lines.append(f"Column '{col}': Skipped (LLM reasoned it should not be strictly consistent).\n")
                    continue
                    
                print(f"  [Judgment] LLM evaluated: YES. Proceeding to majority voting & fallback.")
                
                grouped = self.dirty_df.groupby(entity_col)
                total_groups = 0
                fallback_groups_count = 0
                
                for entity_val, group in tqdm(grouped, desc=f"Cleaning {col}"):
                    if str(entity_val).lower() in ['nan', 'none', 'null', '']:
                        continue
                        
                    total_groups += 1
                    indices = group.index
                    group_vals = group[col]
                    
                    valid_vals = [v for v in group_vals if str(v).lower() not in ['nan', 'none', 'null', 'empty', '']]
                    
                    majority_val = None
                    if valid_vals:
                        val_series = pd.Series(valid_vals)
                        val_counts = val_series.value_counts()
                        max_val = val_counts.idxmax()
                        max_count = val_counts.max()
                        
                        if (max_count / len(valid_vals)) > 0.5:
                            majority_val = max_val
                            
                    if majority_val is not None:
                        self.cleaned_df.loc[indices, col] = majority_val
                    else:
                        fallback_groups_count += 1
                        
                        specific_group_lines = []
                        for idx, row in group.head(50).iterrows():
                            valid_data = {k: v for k, v in row.to_dict().items() if not str(k).startswith("__")}
                            specific_group_lines.append(f"Row {idx}: {valid_data}")
                        specific_group_str = "\n".join(specific_group_lines)
                        
                        fb_prompt = self._build_fd_fallback_prompt(col, entity_col, entity_val, specific_group_str, all_dirty_rows_str, all_clean_rows_str)
                        
                        suggested_val = None
                        fb_reason = ""
                        
                        for attempt in range(max_retries):
                            if self.debug_mode:
                                fb_response = self._call_llm_test_fallback(fb_prompt)
                            else:
                                fb_response = self._call_llm(fb_prompt)
                                
                            self._save_prompt_response(f"fallback_entity_{entity_val}", safe_col, attempt + 1, fb_prompt, fb_response, pr_dir)
                            suggested_val, fb_reason = self._extract_fallback(fb_response)
                            
                            if suggested_val is not None:
                                break
                            else:
                                print(f"    Attempt {attempt + 1}: Failed to extract <suggested_value> from Fallback response. Retrying...")

                        if suggested_val is None:
                            print(f"    [Fallback] Failed to get valid suggested value after {max_retries} attempts for entity '{entity_val}'. Skipping group.")
                            continue
                        
                        self.cleaned_df.loc[indices, col] = suggested_val
                        
                        ground_truth = self.clean_df.loc[indices, col].to_dict()
                        fallback_logs.append({
                            "entity_val": entity_val,
                            "original_data": group.to_dict(orient='index'),
                            "ground_truth": ground_truth,
                            "llm_suggested_value": suggested_val,
                            "llm_reason": fb_reason
                        })
                
                metrics = self._evaluate_fd_repair(col, self.dirty_df, self.cleaned_df, col_results)
                col_results["total_groups"] = total_groups
                col_results["fallback_groups_count"] = fallback_groups_count
                col_results["fallback_logs"] = fallback_logs
                
                with open(os.path.join(phase_fd_dir, f"{safe_col}_fd_results.json"), 'w', encoding='utf-8') as f:
                    json.dump(col_results, f, ensure_ascii=False, indent=4)
                    
                met_ordered = { 'F1': round(metrics['F1'], 4), 'Precision': round(metrics['Precision'], 4), 'Recall': round(metrics['Recall'], 4) }
                print(f"  [Metrics] FD Repair: {met_ordered} | Fallback triggers: {fallback_groups_count}/{total_groups}")
                summary_lines.append(f"Column '{col}' FD Repair -> Metrics: {met_ordered} | Fallbacks: {fallback_groups_count}/{total_groups}\n")

            with open(os.path.join(phase_fd_dir, 'fd_judgments.json'), 'w', encoding='utf-8') as f:
                json.dump(fd_judgments_dict, f, ensure_ascii=False, indent=4)
                
            with open(os.path.join(phase_fd_dir, 'fd_phase_summary.txt'), 'w', encoding='utf-8') as f:
                f.write("\n".join(summary_lines))
                end_time = datetime.now()
                total_duration = (end_time - start_time).total_seconds() - self.LLM_sleep_time
                f.write("\n=== Execution Time ===\n")
                f.write(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"LLM Sleep Time (Waiting Time): {self.LLM_sleep_time}\n")
                f.write(f"Total Duration: {total_duration}\n")

            cleaned_csv_path = os.path.join(phase_fd_dir, f"{self.dataset}_fd_cleaned.csv")
            self.cleaned_df.to_csv(cleaned_csv_path, index=False)
            
            print(f"\nSaved Phase 2 cleaned data to: {cleaned_csv_path}")
            print("\n=== Global Evaluation for FD Cleaning Phase ===")
            measure_repair(self.clean_path, self.original_dirty_path, cleaned_csv_path)

            print(f"\nPhase 2 FD Cleaning completed. Check results in: {phase_fd_dir}")
            return os.path.dirname(phase_fd_dir), self.LLM_time, self.LLM_sleep_time
            
        except Exception as e:
            print(f"\n[Fatal Error in run]: {str(e)}")
            raise e
        finally:
            sys.stdout = original_stdout


if __name__ == "__main__":
    datasets = ['beers',
                'hospital',
                'flights'
                ]
    for dataset in datasets:
        dataset_root = 'data/datasets',
        cleaner = FDCleaner(
            dataset=dataset,
            original_dirty_path=f'{dataset_root}/{dataset}/{dataset}_error-01.csv',
            clean_path=f'{dataset_root}/{dataset}/{dataset}_clean.csv',
            debug_mode=False,
            llm_base_url=DEFAULT_LLM_BASE_URL,
            llm_api_key=DEFAULT_LLM_API_KEY,
            llm_model=DEFAULT_LLM_MODEL,
            fasttext_model_path=DEFAULT_FASTTEXT_MODEL_PATH
        )
        cleaner.run()

import pandas as pd
import numpy as np
import os
import sys
import json
import re
import string
from tqdm import tqdm
from datetime import datetime
import shutil

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from .measure import measure_repair
    from .AnomalyTupleSelector import AnomalyTupleSelector
    from .AutoLabeler import AutoLabeler
    from .LLM_response import call_llm
    from .project_config import DEFAULT_FASTTEXT_MODEL_PATH, DEFAULT_LLM_API_KEY, DEFAULT_LLM_BASE_URL, DEFAULT_LLM_MODEL, DEFAULT_RESULT_ROOT
except ImportError:
    from measure import measure_repair
    from AnomalyTupleSelector import AnomalyTupleSelector
    from AutoLabeler import AutoLabeler
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

class FormatCleaner:
    # Initialize format cleaner settings and load the clean data.
    def __init__(self, dataset: str, dirty_path: str, clean_path: str, debug_mode=True, result_root=DEFAULT_RESULT_ROOT, fasttext_model_path: str = DEFAULT_FASTTEXT_MODEL_PATH, llm_base_url: str = DEFAULT_LLM_BASE_URL, llm_api_key: str = DEFAULT_LLM_API_KEY, llm_model: str = DEFAULT_LLM_MODEL):
        self.dataset = dataset
        self.dirty_path = dirty_path
        self.clean_path = clean_path
        self.debug_mode = debug_mode
        self.result_root = result_root
        self.fasttext_model_path = fasttext_model_path
        self.llm_base_url = llm_base_url
        self.llm_api_key = llm_api_key
        self.llm_model = llm_model
        self.LLM_sleep_time = 0
        self.LLM_time = 0
        
        self.dirty_df = pd.read_csv(dirty_path, dtype=str).fillna('nan')
        self.clean_df = pd.read_csv(clean_path, dtype=str).fillna('nan')
        self.cleaned_df = self.dirty_df.copy()
        
        from_chars = string.ascii_uppercase + string.ascii_lowercase + string.digits
        to_chars = ('A' * len(string.ascii_uppercase)) + \
                   ('a' * len(string.ascii_lowercase)) + \
                   ('9' * len(string.digits))
        self.fine_trans_table = str.maketrans(from_chars, to_chars)

    # Infer a simple pattern label from a value.
    def _get_pattern(self, text):
        if str(text).lower() in ['nan', 'none', '']: return "NULL"
        return str(text).translate(self.fine_trans_table)

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
            
    # Return a test detection response for debugging.
    def _call_llm_test_detect(self, prompt):
        return '''是```python
import re
# Return True when a value is empty or too short.
def detect_format_error(value):
    if str(value).lower() in ['nan', 'none', 'empty', '']:
        return False
    val_str = str(value).strip()
    pattern = r'^\d+\.\d+ oz$'
    if re.match(pattern, val_str):
        return False
    else:
        return True
```
<reason>
xxxxxx...
</reason>'''

    # Return a test repair response for debugging.
    def _call_llm_test_repair(self, prompt):
        return '''是```python
import re
# Return a placeholder cleaned value for debugging.
def repair_format_error(value):
    if str(value).lower() in ['nan', 'none', 'empty', '']:
        return value
    val_str = str(value).strip()
    match = re.search(r'(\d+(\.\d+)?)', val_str)
    if match:
        number = match.group(1)
        if '.' not in number:
            number += '.0'
        return f"{number} oz"
    return value
```
<reason>
xxxxxx...
</reason>'''

    # Save an LLM prompt and response for audit logs.
    def _save_prompt_response(self, phase, col, attempt, prompt, response, pr_dir):
        safe_col = col.replace("/", "_").replace("\\", "_")
        filename = os.path.join(pr_dir, f"{phase}_{safe_col}_attempt_{attempt}.txt")
        # print(f"    Saving prompt and response to: {filename}")
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=== PROMPT ===\n")
            f.write(prompt + "\n\n")
            f.write("=== RESPONSE ===\n")
            f.write(str(response) + "\n")

    # Extract generated code and decision from an LLM response.
    def _extract_decision_and_code(self, response: str, func_name: str):
        decision = "否"
        if "是" in response[:50] or "Yes" in response[:50] or "yes" in response[:50]:
            decision = "是"
            
        code_block = None
        pattern = re.compile(r'```python\s+(.*?)\s+```', re.DOTALL)
        match = pattern.search(response)
        if match:
            code_block = match.group(1)
        
        reason = ""
        pattern_reason = re.compile(r'<reason>(.*?)</reason>', re.DOTALL | re.IGNORECASE)
        match_reason = pattern_reason.search(response)
        if match_reason:
            reason = match_reason.group(1).strip()
            
        return decision, code_block, reason

    # Build pattern context from sampled column values.
    def _generate_pattern_context(self, series: pd.Series):
        patterns = series.map(self._get_pattern)
        pattern_counts = patterns.value_counts()
        total = len(patterns)
        
        target_patterns = pattern_counts.index.tolist()
        if len(target_patterns) > 10:
            target_patterns = target_patterns[:5] + target_patterns[-5:]
            
        context_lines = []
        for p in target_patterns:
            freq = pattern_counts[p] / total
            samples = series[patterns == p].unique()
            sample_str = ", ".join([f"'{s}'" for s in samples[:2] if str(s) != 'nan'])
            if not sample_str: sample_str = "N/A"
            context_lines.append(f"Pattern: {p} | Frequency: {freq:.2%} | Samples: [{sample_str}]")
            
        return "\n".join(context_lines)

    # Evaluate generated detection code against clean and dirty values.
    def _evaluate_detection(self, col: str, detect_func, results_dict):
        all_need_detect = 0
        all_detected = 0
        correctly_detect = 0
        wrongly_detect = 0
        missing_errors = 0
        
        missing_errors_log = []
        wrongly_detect_log = []

        for idx, dirty_val in self.dirty_df[col].items():
            clean_val = self.clean_df.loc[idx, col]
            
            is_true_error = (dirty_val != clean_val) or (clean_val == 'nan') or (clean_val == 'empty')
            
            if is_true_error:
                all_need_detect += 1
                
            try:
                is_detected = detect_func(dirty_val)
                if is_detected:
                    all_detected += 1
                    if is_true_error:
                        correctly_detect += 1
                    else:
                        wrongly_detect += 1
                        wrongly_detect_log.append(f"Row {idx}: {dirty_val}")
                else:
                    if is_true_error:
                        missing_errors_log.append(f"Row {idx}: {dirty_val} -> {clean_val}")
            except Exception as e:
                pass 
        
        missing_errors = all_need_detect - correctly_detect

        pre = correctly_detect / (all_detected + 1e-8)
        rec = correctly_detect / (all_need_detect + 1e-8)
        f1 = 2 * pre * rec / (pre + rec + 1e-8)
        
        metrics = {
            "all_need_detect": all_need_detect,
            "all_detected": all_detected,
            "correctly_detect": correctly_detect,
            "wrongly_detect": wrongly_detect,
            "missing_errors": missing_errors,
            "Precision": pre,
            "Recall": rec,
            "F1": f1
        }
        
        results_dict['detection_metrics'] = metrics
        results_dict['detection_wrongly_detect_logs'] = wrongly_detect_log[:20]
        results_dict['detection_missing_errors_logs'] = missing_errors_log[:20]
        
        return metrics

    # Evaluate generated repair code against clean and dirty values.
    def _evaluate_repair(self, col: str, detect_func, repair_func, results_dict):
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

        for idx, dirty_val in self.dirty_df[col].items():
            clean_val = self.clean_df.loc[idx, col]
            
            is_true_error = (dirty_val != clean_val)
            if is_true_error:
                all_need_repair += 1
                
            try:
                if detect_func(dirty_val):
                    repaired_val = str(repair_func(dirty_val))
                else:
                    repaired_val = dirty_val
            except Exception:
                repaired_val = dirty_val
                
            if repaired_val != dirty_val:
                all_repaired += 1
                log_str = f"Row {idx}: '{dirty_val}' -> '{repaired_val}' (Truth: '{clean_val}')"
                
                if repaired_val == clean_val:
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
    
    # Build an LLM prompt for detecting formatting errors.
    def _build_detect_prompt(self, col, pattern_context, clean_vals, dirty_vals, all_dirty_rows_str, all_clean_rows_str, col_labeled_list_str):
        return f"""[Role] Data Quality Expert
[Task] Analyze if the column '{col}' has systematic formatting errors.

[Global Context (All Sampled Tuples)]
A sample of the dataset has been manually cleaned. Below are the original tuples containing errors and their corresponding cleaned versions.
Dirty Tuples:
{all_dirty_rows_str}
Clean Tuples (Ground Truth):
{all_clean_rows_str}

[Labeled Errors and Corresponding Clean Data for Target Column '{col}']
{col_labeled_list_str}

[Data Profile for Target Column '{col}']
Pattern Abstraction Rule: 
In the patterns below, uppercase letters are replaced by 'A', lowercase letters by 'a', and digits by '9'. Other characters (like punctuation or spaces) remain unchanged.
Patterns and Frequencies:
{pattern_context}

[Known Values for Target Column '{col}']
Known Clean Values for Target Column '{col}'(Ground Truth): {clean_vals}
Known Dirty Values for Target Column '{col}' (Errors): {dirty_vals}

[Instruction]
1. Does this column suffer from formatting issues (e.g., inconsistent length, invalid characters, regex mismatch)? Answer strictly with "是" or "否" at the very beginning.
2. If "是", provide a Python function named `detect_format_error(value)` that returns True if the value has a formatting error, and False otherwise. Also, provide a brief explanation of why you designed the rule this way.

[Note]
1. Handle strings safely (e.g., value = str(value)).
2. Explicitly handle missing or empty values (e.g., 'nan', 'none', 'null', ''). You MUST consider them as formatting errors and return True immediately.
3. Do not provide FD or Semantic checks, ONLY Format checks.

[Response Format]
1. Start with "是" or "否" to indicate if formatting issues are detected.
2. If "是", wrap the code in ```python ... ``` and the reason in <reason>...</reason> tags after the code block.

"""

    # Build an LLM prompt for repairing formatting errors.
    def _build_repair_prompt(self, col, detect_code, repair_mapping_str, all_dirty_rows_str, all_clean_rows_str, col_labeled_list_str):
        return f"""[Role] Data Quality Expert
[Task] Fix formatting errors in the column '{col}'.

[Global Context (All Sampled Tuples)]
A sample of the dataset has been manually cleaned. Below are the original tuples containing errors and their corresponding cleaned versions.
Dirty Tuples:
{all_dirty_rows_str}
Clean Tuples (Ground Truth):
{all_clean_rows_str}

[Labeled Errors and Corresponding Clean Data for Target Column '{col}']
{col_labeled_list_str}

[Error Detection Function for Target Column '{col}']
You previously wrote this detection function:
```python
{detect_code}
```

[Known Values for Target Column '{col}']
Transformation examples (Dirty -> Clean):
{repair_mapping_str}

[Instruction]
1. Can these formatting errors be fixed using simple rules, string manipulation, or regex? Answer strictly with "是" or "否" at the very beginning.
2. If "是", provide a Python function named `repair_format_error(value)` that takes a dirty value and returns the cleaned string. If a value cannot be fixed, return the original value. Also, provide a brief explanation of your repair logic.

[Note]
1. Handle strings safely (e.g., value = str(value)).
2. Explicitly handle missing or empty values (e.g., 'nan', 'none', 'null', ''). If a value is missing and lacks sufficient information to be formatted, do NOT force a hardcoded guess or hallucinate a mapping. Simply return the original value.
3. Do NOT attempt to fix Functional Dependency (FD) or Semantic errors (e.g., spelling mistakes, wrong entities) at this stage. Focus STRICTLY and ONLY on repairing the formatting issues.
4. Strategy Advice: Instead of writing regex to delete noise, highly prefer an EXTRACTION and RECONSTRUCTION strategy. You can draw inspiration for this from the core logic of the detect function you previously wrote. This approach maximizes the success rate of format repairs while ensuring the underlying semantics remain strictly unchanged.

[Response Format]
1. Start with "是" or "否" to indicate if formatting issues are detected.
2. If "是", wrap the code in ```python ... ``` and the reason in <reason>...</reason> tags after the code block.
"""

    # Run the format cleaning phase and save outputs.
    def run(self):
        print("Running AutoLabeler first...")
        labeler = AutoLabeler(
            self.dataset,
            self.dirty_path,
            self.clean_path,
            result_root=self.result_root,
            llm_base_url=self.llm_base_url,
            llm_api_key=self.llm_api_key,
            llm_model=self.llm_model,
            fasttext_model_path=self.fasttext_model_path
        )
        try:
            base_dir, temp_time3, temp_time4 = labeler.run()
            self.LLM_time += temp_time3
            self.LLM_sleep_time += temp_time4
            phase_label_dir = os.path.join(base_dir, 'phase_label')
            start_time = datetime.now()
            phase_format_dir = os.path.join(os.path.dirname(phase_label_dir), 'phase_format')
            os.makedirs(phase_format_dir, exist_ok=True)
        except Exception as e:
            print(f"Error running AutoLabeler: {e}")
            raise e
        
        pr_dir = os.path.join(phase_format_dir, 'prompt_response')
        os.makedirs(pr_dir, exist_ok=True)
        
        original_stdout = sys.stdout
        log_path = os.path.join(phase_format_dir, 'run.log')
        sys.stdout = PrintLogger(log_path)
        
        try:
            print(f"\n--- Starting Phase 1: Format Cleaning ---")
            
            with open(os.path.join(phase_label_dir, 'values_stat.json'), 'r', encoding='utf-8') as f:
                values_stat = json.load(f)
            with open(os.path.join(phase_label_dir, 'labeled_data.json'), 'r', encoding='utf-8') as f:
                labeled_data = json.load(f)
            with open(os.path.join(phase_label_dir, 'tuples_data.json'), 'r', encoding='utf-8') as f:
                tuples_data = json.load(f)
                
            dirty_tuples_dict = tuples_data.get("dirty", {})
            clean_tuples_dict = tuples_data.get("clean", {})
            
            all_dirty_strs = []
            all_clean_strs = []
            for idx_str, row_data in dirty_tuples_dict.items():
                valid_data = {k: v for k, v in row_data.items() if not str(k).startswith("__")}
                all_dirty_strs.append(f"Row {idx_str}: {valid_data}")
                
                if idx_str in clean_tuples_dict:
                    c_data = {k: v for k, v in clean_tuples_dict[idx_str].items() if not str(k).startswith("__")}
                    all_clean_strs.append(f"Row {idx_str}: {c_data}")
                else:
                    all_clean_strs.append(f"Row {idx_str}: {{Data Missing}}")
                    
            all_dirty_rows_str = "\n".join(all_dirty_strs)
            all_clean_rows_str = "\n".join(all_clean_strs)
                
            selector = AnomalyTupleSelector(self.dirty_df, fasttext_model_path=self.fasttext_model_path)
            format_info_dict = selector._detect_format_consistent_columns(min_freq_threshold=0.10, consistency_threshold=0.5)
            target_cols = [col for col, info in format_info_dict.items() if info['score'] > 0.5]
            
            summary_lines = ["=== Format Cleaning Phase Summary ===\n"]
            
            for col in target_cols:
                print(f"\n>>> Processing Column: {col} <<<")
                col_results = {}
                safe_col = str(col).replace("/", "_").replace("\\", "_")
                
                pattern_context = self._generate_pattern_context(self.dirty_df[col])
                clean_vals = values_stat['clean_subset_values'].get(col, [])
                dirty_vals = values_stat['dirty_error_values'].get(col, [])
                
                col_labels = [item for item in labeled_data if item[1] == col]
                labeled_dirty_examples = [str(item[2]) for item in col_labels]
                labeled_clean_examples = [str(item[3]) for item in col_labels]
                repair_mapping_strs = []
                for d, c in zip(labeled_dirty_examples, labeled_clean_examples):
                    repair_mapping_strs.append(f"'{d}' -> '{c}'")
                repair_mapping_str = "\n".join(repair_mapping_strs)
                if not repair_mapping_str:
                    repair_mapping_str = "No specific transformation examples available."
                
                col_labeled_strs = []
                for item in col_labels:
                    col_labeled_strs.append(f"Row {item[0]}: '{item[2]}' -> '{item[3]}' (Reason: {item[4]})")
                col_labeled_list_str = "\n".join(col_labeled_strs)
                if not col_labeled_list_str:
                    col_labeled_list_str = "No specific labeled errors for this column."
                
                detect_prompt = self._build_detect_prompt(col, pattern_context, clean_vals, dirty_vals, all_dirty_rows_str, all_clean_rows_str, col_labeled_list_str)
                print(f"  --- Detection Phase for '{col}' ---")
                
                detect_func = None
                max_retries = 3
                for attempt in range(max_retries):
                    if self.debug_mode:
                        response = self._call_llm_test_detect(detect_prompt)
                    else:
                        response = self._call_llm(detect_prompt) 
                    self._save_prompt_response("detection", safe_col, attempt+1, detect_prompt, response, pr_dir)
                    decision, code, reason = self._extract_decision_and_code(response, "detect_format_error")
                    
                    col_results[f'detect_prompt_attempt_{attempt+1}'] = detect_prompt
                    col_results[f'detect_response_attempt_{attempt+1}'] = response
                    
                    if decision == "否":
                        print(f"    Attempt {attempt+1}: LLM answered '否'. Ending detection for '{col}'.")
                        summary_lines.append(f"Column '{col}': LLM determined NO format issues.\n")
                        break
                    
                    if code:
                        local_vars = {}
                        try:
                            exec(code, globals(), local_vars)
                            temp_func = local_vars['detect_format_error']
                            
                            clean_test_passed = all(not temp_func(c) for c in labeled_clean_examples if c != 'nan')
                            dirty_test_passed = any(temp_func(d) for d in labeled_dirty_examples if d != 'nan')
                            
                            if clean_test_passed and dirty_test_passed:
                                detect_func = temp_func
                                col_results['final_detect_code'] = code
                                col_results['final_detect_reason'] = reason
                                print(f"    Attempt {attempt+1}: LLM answered '是' and generated a VALID detection function.")
                                summary_lines.append(f"[Valid Detection Function]:\n{code}\n")
                                summary_lines.append(f"[Reason]: {reason}\n")
                                break
                            else:
                                print(f"    Attempt {attempt+1}: LLM answered '是', but generated function FAILED validation (flagged clean data or missed all dirty data). Retrying...")
                        except Exception as e:
                            print(f"    Attempt {attempt+1}: LLM answered '是', but code execution ERROR: {e}. Retrying...")
                    else:
                        print(f"    Attempt {attempt+1}: LLM answered '是', but FAILED to generate parseable Python code. Retrying...")
                else:
                    print(f"  [Result] Failed to generate a working detection function for '{col}' after {max_retries} attempts.")
                    summary_lines.append(f"Column '{col}': Failed to generate valid detection function after {max_retries} attempts.\n")
                            
                if detect_func is None:
                    summary_lines.append("-" * 60 + "\n")
                    continue
                    
                det_metrics = self._evaluate_detection(col, detect_func, col_results)
                det_ordered = {
                    'F1': round(det_metrics['F1'], 4),
                    'Precision': round(det_metrics['Precision'], 4),
                    'Recall': round(det_metrics['Recall'], 4),
                    **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in det_metrics.items() if k not in ['F1', 'Precision', 'Recall']}
                }
                print(f"  [Metrics] Detection: {det_ordered}")
                summary_lines.append(f"Column '{col}' Detection -> {det_ordered}\n")
                
                repair_prompt = self._build_repair_prompt(col, col_results.get('final_detect_code', ''), repair_mapping_str, all_dirty_rows_str, all_clean_rows_str, col_labeled_list_str)
                print(f"  --- Repair Phase for '{col}' ---")
                
                repair_func = None
                for attempt in range(max_retries):
                    if self.debug_mode:
                        response = self._call_llm_test_repair(repair_prompt)
                    else:
                        response = self._call_llm(repair_prompt)
                    self._save_prompt_response("repair", safe_col, attempt+1, repair_prompt, response, pr_dir)
                    decision, code, reason = self._extract_decision_and_code(response, "repair_format_error")
                    
                    col_results[f'repair_prompt_attempt_{attempt+1}'] = repair_prompt
                    col_results[f'repair_response_attempt_{attempt+1}'] = response
                    
                    if decision == "否":
                        print(f"    Attempt {attempt+1}: LLM answered '否'. Ending repair for '{col}'.")
                        summary_lines.append(f"Column '{col}': Errors not simply repairable.\n")
                        break
                        
                    if code:
                        local_vars = {}
                        try:
                            exec(code, globals(), local_vars)
                            temp_func = local_vars['repair_format_error']
                            
                            clean_safe = all(str(temp_func(c)).strip() == str(c).strip() for c in labeled_clean_examples if c != 'nan')
                            dirty_fixed = any(str(temp_func(d)).strip() == str(c).strip() for d, c in zip(labeled_dirty_examples, labeled_clean_examples) if d != 'nan')
                            
                            if clean_safe and dirty_fixed:
                                repair_func = temp_func
                                col_results['final_repair_code'] = code
                                col_results['final_repair_reason'] = reason
                                print(f"    Attempt {attempt+1}: LLM answered '是' and generated a VALID repair function.")
                                summary_lines.append(f"[Valid Repair Function]:\n{code}\n")
                                summary_lines.append(f"[Reason]: {reason}\n")
                                # Apply a generated function safely to a value.
                                def safe_apply(val):
                                    try:
                                        if detect_func(val):
                                            return str(repair_func(val))
                                        return val
                                    except:
                                        return val
                                self.cleaned_df[col] = self.cleaned_df[col].apply(safe_apply)
                                break
                            else:
                                print(f"    Attempt {attempt+1}: LLM answered '是', but generated function FAILED validation (altered clean data or failed to fix any dirty data). Retrying...")
                        except Exception as e:
                            print(f"    Attempt {attempt+1}: LLM answered '是', but code execution ERROR in repair: {e}. Retrying...")
                    else:
                        print(f"    Attempt {attempt+1}: LLM answered '是', but FAILED to generate parseable Python code. Retrying...")
                else:
                    print(f"  [Result] Failed to generate a working repair function for '{col}' after {max_retries} attempts.")
                    summary_lines.append(f"Column '{col}': Failed to generate valid repair function after {max_retries} attempts.\n")

                if repair_func:
                    rep_metrics = self._evaluate_repair(col, detect_func, repair_func, col_results)
                    rep_ordered = {
                        'F1': round(rep_metrics['F1'], 4),
                        'Precision': round(rep_metrics['Precision'], 4),
                        'Recall': round(rep_metrics['Recall'], 4),
                        **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in rep_metrics.items() if k not in ['F1', 'Precision', 'Recall']}
                    }
                    print(f"  [Metrics] Repair: {rep_ordered}")
                    summary_lines.append(f"Column '{col}' Repair -> {rep_ordered}\n")
                    
                with open(os.path.join(phase_format_dir, f"{safe_col}_format_results.json"), 'w', encoding='utf-8') as f:
                    json.dump(col_results, f, ensure_ascii=False, indent=4)
                summary_lines.append("-" * 60 + "\n")

            with open(os.path.join(phase_format_dir, 'format_phase_summary.txt'), 'w', encoding='utf-8') as f:
                f.write("\n".join(summary_lines))
                end_time = datetime.now()
                total_duration = (end_time - start_time).total_seconds() - self.LLM_sleep_time
                f.write("\n=== Execution Time ===\n")
                f.write(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"LLM Sleep Time (Waiting Time): {self.LLM_sleep_time}\n")
                f.write(f"Total Duration: {total_duration}\n")
                
            cleaned_csv_path = os.path.join(phase_format_dir, f"{self.dataset}_format_cleaned.csv")
            self.cleaned_df.to_csv(cleaned_csv_path, index=False)
            print(f"\nSaved Phase 1 cleaned data to: {cleaned_csv_path}")
            print("\n=== Global Evaluation for Format Cleaning Phase ===")
            measure_repair(self.clean_path, self.dirty_path, cleaned_csv_path)
            print(f"\nPhase 1 Format Cleaning completed. Check results in: {phase_format_dir}")
            return os.path.dirname(phase_format_dir), self.LLM_time, self.LLM_sleep_time
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
        print(f"\n=== Running FormatCleaner for dataset: {dataset} ===")
        dataset_root = 'data/datasets'
        dirty_path = f'{dataset_root}/{dataset}/{dataset}_error-01.csv'
        clean_path = f'{dataset_root}/{dataset}/{dataset}_clean.csv'
        cleaner = FormatCleaner(
            dataset=dataset,
            dirty_path=dirty_path,
            clean_path=clean_path,
            debug_mode=False,
            llm_base_url=DEFAULT_LLM_BASE_URL,
            llm_api_key=DEFAULT_LLM_API_KEY,
            llm_model=DEFAULT_LLM_MODEL,
            fasttext_model_path=DEFAULT_FASTTEXT_MODEL_PATH
        )
        cleaner.run()

import pandas as pd
import numpy as np
import sys
import os
import json
import re
from datetime import datetime
from tqdm import tqdm

try:
    from .AnomalyTupleSelector import AnomalyTupleSelector
    from .LLM_response import call_llm
    from .project_config import DEFAULT_FASTTEXT_MODEL_PATH, DEFAULT_LLM_API_KEY, DEFAULT_LLM_BASE_URL, DEFAULT_LLM_MODEL, DEFAULT_RESULT_ROOT
except ImportError:
    from AnomalyTupleSelector import AnomalyTupleSelector
    from LLM_response import call_llm
    from project_config import DEFAULT_FASTTEXT_MODEL_PATH, DEFAULT_LLM_API_KEY, DEFAULT_LLM_BASE_URL, DEFAULT_LLM_MODEL, DEFAULT_RESULT_ROOT

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

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


class AutoLabeler:
    # Initialize auto-labeling settings and load dirty and clean data.
    def __init__(self, dataset: str, dirty_path: str, clean_path: str,
                 result_root=DEFAULT_RESULT_ROOT,
                 fasttext_model_path: str = DEFAULT_FASTTEXT_MODEL_PATH,
                 llm_base_url: str = DEFAULT_LLM_BASE_URL, llm_api_key: str = DEFAULT_LLM_API_KEY, llm_model: str = DEFAULT_LLM_MODEL):
        self.dataset = dataset
        self.dirty_path = dirty_path
        self.clean_path = clean_path
        self.result_root = result_root
        self.fasttext_model_path = fasttext_model_path
        self.llm_base_url = llm_base_url
        self.llm_api_key = llm_api_key
        self.llm_model = llm_model
        self.clean_df = None
        self.dirty_df = None
        self.LLM_sleep_time = 0
        self.LLM_time = 0

    # Compare two values while treating null-like values as missing.
    def _is_different(self, val1, val2):
        if pd.isna(val1) and pd.isna(val2): return False
        if pd.isna(val1) or pd.isna(val2): return True
        return str(val1).strip() != str(val2).strip()

    # Sanitize an LLM-provided reason before saving it.
    def _sanitize_reason(self, reason):
        if not isinstance(reason, str):
            return str(reason)
        reason = reason.replace('\n', ' ').replace('\r', '')
        reason = reason.replace('"', "'")
        return reason.strip()

    # Build an LLM prompt for explaining a value correction.
    def _construct_prompt(self, all_dirty_rows_str, all_clean_rows_str, current_idx, attr, dirty_val, clean_val):
        prompt = f"""[Role Description]
You are a Data Quality Expert.

[Task Description]
I will provide you with a set of 'Dirty Tuples' (containing errors) and their corresponding 'Clean Tuples' (Ground Truth).
Dirty Tuples refer to the original data with errors, while Clean Tuples are the corresponding cleaned data after manual data cleaning.
Your task is to analyze a specific error in one of these tuples (identified by Row Index and Attribute Name).
Other data are mainly used as context information. Use them to understand the data pattern.

[Tuples]
Dirty Tuples:
{all_dirty_rows_str}

Clean Tuples:
{all_clean_rows_str}

[Current Analysis Task]
We are focusing on Row Index: {current_idx}
Attribute Name: "{attr}"
Value in Dirty Tuple: "{dirty_val}"
Value in Clean Tuple: "{clean_val}"

[Instruction]
Please analyze why the value in the Dirty Tuple is incorrect and why the value in the Clean Tuple is correct.
In your response, you need to explain the error type (e.g., format error, dependency violation, typo, semantically impossible value, missing value) and the motivation for correcting it.
Here are some examples:
'ounces': '12.0 ounce' -> '12.0 oz'. Reason: The error is a **format error** due to inconsistent unit abbreviation; 'ounce' should be abbreviated as 'oz' to match the standard format used across the dataset. The clean value '12.0 oz' corrects this inconsistency for uniformity and compliance with the expected format.
'act_dep_time': '11:58 p.m.' -> '11:25 p.m.'. Reason: The error is a **dependency violation**, because for flight 'AA-204-LAX-MCO', other sources indicate the actual departure time is '11:25 p.m.', and '11:58 p.m.' is inconsistent with them. The clean value '11:25 p.m.' is correct because it matches the consistent actual departure time across other tuples for the same flight.
'PhoneNumber': '2x6x938310' -> '2565938310'. Reason: The error is a **typo**, where 'x' characters incorrectly replace digit '5' in the phone number. The clean value restores the correct 10-digit format consistent with other valid phone numbers in the dataset.
'style': '32.0 oz.' -> 'Scotch Ale / Wee Heavy'. Reason: The error is a **semantically impossible value**, as '32.0 oz.' represents volume and incorrectly populates the 'style' attribute, which should describe beer type. The clean value 'Scotch Ale / Wee Heavy' is correct because it accurately identifies the beer style, consistent with other entries and domain knowledge.
'state': nan -> 'CO'. Reason: The error is a **missing value** in the 'state' attribute, which violates data completeness. The correct value 'CO' is inferred from the consistent pattern in other tuples where 'Oskar Blues Brewery' is associated with 'CO' (e.g., Rows 1554, 1583), ensuring referential integrity and geographic consistency.
Do not include any extra information.
Provide a concise response in 1–2 sentences.
"""
        return prompt

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

    # Label selected tuples with explanations using the LLM.
    def _label_tuples(self, selected_tuples: pd.DataFrame, phase_dir: str):
        labeled_data = []
        
        valid_cols = [c for c in selected_tuples.columns if not str(c).startswith("__")]
        
        all_dirty_strs = []
        all_clean_strs = []
        
        for idx, row in selected_tuples.iterrows():
            d_dict = row[valid_cols].to_dict()
            all_dirty_strs.append(f"Row {idx}: {d_dict}")
            
            if idx in self.clean_df.index:
                c_dict = self.clean_df.loc[idx, valid_cols].to_dict()
                all_clean_strs.append(f"Row {idx}: {c_dict}")
            else:
                all_clean_strs.append(f"Row {idx}: {{Data Missing}}")
                
        all_dirty_rows_str = "\n".join(all_dirty_strs)
        all_clean_rows_str = "\n".join(all_clean_strs)

        pr_dir = os.path.join(phase_dir, 'prompt_response')
        os.makedirs(pr_dir, exist_ok=True)
        
        for idx, dirty_row in tqdm(selected_tuples.iterrows(), total=len(selected_tuples), desc="LLM Labeling"):
            try:
                clean_row = self.clean_df.loc[idx]
            except KeyError:
                print(f"[Error] Index {idx} not found in Clean DataFrame.")
                continue
            
            for col in valid_cols:
                d_val = dirty_row[col]
                if col not in clean_row.index: 
                    continue
                    
                c_val = clean_row[col]
                
                if self._is_different(d_val, c_val):
                    prompt = self._construct_prompt(
                        all_dirty_rows_str=all_dirty_rows_str,
                        all_clean_rows_str=all_clean_rows_str,
                        current_idx=idx,
                        attr=col,
                        dirty_val=d_val,
                        clean_val=c_val
                    )
                    
                    raw_reason = ""
                    try:
                        raw_reason = self._call_llm(prompt)
                        reason = self._sanitize_reason(raw_reason)
                        
                        safe_col_name = str(col).replace("/", "_").replace("\\", "_")
                        pr_filename = os.path.join(pr_dir, f"{idx}_{safe_col_name}.txt")
                        
                        with open(pr_filename, 'w', encoding='utf-8') as pr_file:
                            pr_file.write("=== PROMPT ===\n")
                            pr_file.write(prompt + "\n\n")
                            pr_file.write("=== RAW RESPONSE ===\n")
                            pr_file.write(str(raw_reason) + "\n\n")
                            pr_file.write("=== SANITIZED REASON ===\n")
                            pr_file.write(reason + "\n")
                            
                    except Exception as e:
                        reason = f"Error: {str(e)}"
                        print(f"[Error] Exception at Row {idx}, Col {col}: {reason}")
                    
                    labeled_data.append([idx, col, d_val, c_val, reason])
                    
        return labeled_data

    class NpEncoder(json.JSONEncoder):
        # Serialize unsupported JSON objects as strings.
        def default(self, obj):
            if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                                np.int16, np.int32, np.int64, np.uint8,
                                np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            return super(AutoLabeler.NpEncoder, self).default(obj)

    # Run auto-labeling and save labeled data.
    def run(self):
        start_time = datetime.now()
        
        base_dir = self.result_root
        timestamp = start_time.strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(base_dir, f"{timestamp}_{self.dataset}_{self.llm_model}")
        
        phase_dir = os.path.join(save_dir, 'phase_label')
        
        if not os.path.exists(phase_dir):
            os.makedirs(phase_dir)
            print(f"Created phase directory: {phase_dir}")
        else:
            print(f"Phase directory exists: {phase_dir}")
            return
            
        original_stdout = sys.stdout
        log_path = os.path.join(phase_dir, 'run.log')
        sys.stdout = PrintLogger(log_path)
        
        try:
            print(f"Loading Dirty Data: {self.dirty_path}")
            self.dirty_df = pd.read_csv(self.dirty_path, dtype=str).fillna('nan')
            print(f"Loading Clean Data: {self.clean_path}")
            self.clean_df = pd.read_csv(self.clean_path, dtype=str).fillna('nan')

            if len(self.dirty_df) != len(self.clean_df):
                 raise ValueError("Shape mismatch between Dirty and Clean datasets.")

            print("Executing AnomalyTupleSelector...")
            selector = AnomalyTupleSelector(self.dirty_df, fasttext_model_path=self.fasttext_model_path)
            selected_df = selector.select_top_k_tuples_sep(num_fd=4, num_format=3, num_semantic=3, coverage_aware=True, random_label=False)
            print(f"Selected {len(selected_df)} tuples for labeling.")

            print("Executing Labeling...")
            labeled_list = self._label_tuples(selected_df, phase_dir)
            print(f"Labeling finished. Found {len(labeled_list)} attribute errors.")

            print("Preparing data for storage...")
            
            clean_subset_df = self.clean_df.loc[selected_df.index].copy()
            dfs_data = {
                "dirty": selected_df.to_dict(orient='index'),
                "clean": clean_subset_df.to_dict(orient='index')
            }
            
            with open(os.path.join(phase_dir, 'tuples_data.json'), 'w', encoding='utf-8') as f:
                json.dump(dfs_data, f, cls=self.NpEncoder, ensure_ascii=False, indent=4)

            clean_values_stat = {}
            valid_cols = [c for c in clean_subset_df.columns if not str(c).startswith("__")]

            for col in valid_cols:
                unique_vals = clean_subset_df[col].unique()
                clean_values_stat[col] = [x for x in unique_vals if not pd.isna(x)]

            dirty_values_stat = {}
            for item in labeled_list:
                col = item[1]
                d_val = item[2]
                if col not in dirty_values_stat:
                    dirty_values_stat[col] = set()
                dirty_values_stat[col].add(d_val)
            
            for col in dirty_values_stat:
                dirty_values_stat[col] = list(dirty_values_stat[col])

            values_data = {
                "clean_subset_values": clean_values_stat,
                "dirty_error_values": dirty_values_stat
            }

            with open(os.path.join(phase_dir, 'values_stat.json'), 'w', encoding='utf-8') as f:
                json.dump(values_data, f, cls=self.NpEncoder, ensure_ascii=False, indent=4)

            with open(os.path.join(phase_dir, 'labeled_data.json'), 'w', encoding='utf-8') as f:
                json.dump(labeled_list, f, cls=self.NpEncoder, ensure_ascii=False, indent=4)

            print("Generating human-readable summary...")
            summary_path = os.path.join(phase_dir, 'auto_labeler_summary.txt')
            
            summary_dirty_strs = []
            summary_clean_strs = []
            
            for idx, row in selected_df.iterrows():
                d_dict = row[valid_cols].to_dict()
                summary_dirty_strs.append(f"Row {idx}: {d_dict}")
                if idx in self.clean_df.index:
                    c_dict = self.clean_df.loc[idx, valid_cols].to_dict()
                    summary_clean_strs.append(f"Row {idx}: {c_dict}")
                else:
                    summary_clean_strs.append(f"Row {idx}: {{Data Missing}}")
            
            all_dirty_rows_str = "\n".join(summary_dirty_strs)
            all_clean_rows_str = "\n".join(summary_clean_strs)

            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("=== Global Context (All Selected Tuples) ===\n\n")
                f.write(">>> Dirty Tuples (Input):\n")
                f.write(all_dirty_rows_str + "\n\n")
                f.write(">>> Clean Tuples (Ground Truth):\n")
                f.write(all_clean_rows_str + "\n\n")
                
                f.write("=== Value Statistics ===\n")
                all_stat_cols = sorted(set(clean_values_stat.keys()) | set(dirty_values_stat.keys()))
                for col in all_stat_cols:
                    f.write(f"Column '{col}':\n")
                    f.write(f"  [Clean Set]: {clean_values_stat.get(col, [])}\n")
                    f.write(f"  [Dirty Set]: {dirty_values_stat.get(col, [])}\n")
                f.write("\n")

                f.write("=== Labeled Data (Analysis Results) ===\n")
                for item in labeled_list:
                    f.write(f"Row {item[0]} | Attribute '{item[1]}':\n")
                    f.write(f"  Dirty: {item[2]}\n")
                    f.write(f"  Clean: {item[3]}\n")
                    f.write(f"  Reason: {item[4]}\n")
                    f.write("-" * 60 + "\n")
                
                end_time = datetime.now()
                total_duration = (end_time - start_time).total_seconds() - self.LLM_sleep_time
                f.write("\n=== Execution Time ===\n")
                f.write(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"LLM Sleep Time (Waiting Time): {self.LLM_sleep_time}\n")
                f.write(f"Total Duration: {total_duration}\n")
                    
            print(f"All results (including logs and prompts) successfully saved to: {phase_dir}")
            return save_dir, self.LLM_time, self.LLM_sleep_time
            
        except Exception as e:
            print(f"\n[Fatal Error in run]: {str(e)}")
            raise e
        finally:
            sys.stdout = original_stdout

if __name__ == "__main__":
    datasets = ['hospital', 'flights']
    for dataset in datasets:
        dataset_root = 'data/datasets'
        dirty_path = f'{dataset_root}/{dataset}/{dataset}_error-01.csv'
        clean_path = f'{dataset_root}/{dataset}/{dataset}_clean.csv'
        labeler = AutoLabeler(
            dataset,
            dirty_path,
            clean_path,
            llm_base_url=DEFAULT_LLM_BASE_URL,
            llm_api_key=DEFAULT_LLM_API_KEY,
            llm_model=DEFAULT_LLM_MODEL,
            fasttext_model_path=DEFAULT_FASTTEXT_MODEL_PATH
        )
        labeler.run()

import logging
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


@dataclass
class DetectionResult:
    mask: pd.DataFrame
    records: List[Tuple[int, str]]
    reasons: List[Dict[str, Any]]
    raw_detection_responses: List[str]
    raw_verify_responses: List[str]


class Logger:
    def __init__(self, result_dir: str, verbose: bool = True):
        os.makedirs(result_dir, exist_ok=True)
        self.verbose = verbose
        self.logger = logging.getLogger(f"IterClean_Logger_{time.time()}")
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            file_handler = logging.FileHandler(os.path.join(result_dir, "iterclean_log.txt"), encoding="utf-8")
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

            if verbose:
                stream_handler = logging.StreamHandler()
                stream_handler.setFormatter(formatter)
                self.logger.addHandler(stream_handler)

    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)


def validate_params(params: Dict[str, Any]) -> None:
    if int(params.get("batch_size", 0)) <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if int(params.get("max_workers", 0)) <= 0:
        raise ValueError("max_workers must be a positive integer.")
    if params.get("batch_mode") not in {"sequential", "random", "group", "ourbat", "ranbat"}:
        raise ValueError("batch_mode must be one of: sequential, random, group, ourbat, ranbat.")
    if not params.get("prompt_detect") and not params.get("prompt_detect_path"):
        raise ValueError("IterClean requires the original detection prompt.")
    if params.get("use_verify") and not params.get("prompt_verify") and not params.get("prompt_verify_path"):
        raise ValueError("IterClean requires the original verify prompt when use_verify=True.")


def load_prompt(prompt: Optional[str], prompt_path: Optional[str], prompt_name: str) -> str:
    if prompt is not None:
        return prompt
    if prompt_path:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    raise ValueError(f"IterClean requires the original {prompt_name} prompt.")


def df_to_dirty_data(df: pd.DataFrame) -> List[str]:
    dirty_data = []
    for pos, (_, row) in enumerate(df.iterrows()):
        formatted = {col: _format_cell(row[col]) for col in df.columns}
        formatted_row = "{" + "; ".join(f"{key}: {value}" for key, value in formatted.items()) + "}"
        dirty_data.append(f"Tuple{pos + 1}{formatted_row}")
    return dirty_data


def _format_cell(value: Any) -> str:
    if pd.isna(value):
        return "nan"
    return str(value)


def build_batches(
    df: pd.DataFrame,
    batch_size: int,
    batch_mode: str = "ourbat",
    ref_column: Optional[str] = None,
    dataset_name: Optional[str] = None,
) -> List[List[int]]:
    row_positions = list(range(len(df)))
    if not row_positions:
        return []

    if batch_mode in {"random", "ranbat"}:
        row_positions = row_positions[:]
        random.shuffle(row_positions)
        return _split_list(row_positions, batch_size)

    if ref_column is None:
        ref_column = infer_ref_column(dataset_name, df.columns)

    if batch_mode in {"group", "ourbat"} and ref_column and ref_column in df.columns:
        batches: List[List[int]] = []
        ref_values = []
        reset_df = df.reset_index(drop=True)
        for i in range(len(reset_df)):
            if reset_df.at[i, ref_column] not in ref_values:
                ref_values.append(reset_df.at[i, ref_column])
        for ref_value in ref_values:
            positions = []
            for row_pos in range(len(reset_df)):
                if reset_df.at[row_pos, ref_column] == ref_value:
                    positions.append(row_pos)
            batches.extend(_split_list(positions, batch_size))
        return batches

    return _split_list(row_positions, batch_size)


def infer_ref_column(dataset_name: Optional[str], columns: Sequence[str]) -> Optional[str]:
    if not dataset_name:
        return None

    dataset = dataset_name.lower()
    if dataset == "hospital":
        candidate = "HospitalName"
    elif dataset == "flights":
        candidate = "flight"
    elif dataset == "beers":
        candidate = "brewery_id"
    else:
        candidate = "Year"
    return candidate if candidate in columns else None


def _split_list(values: Sequence[int], batch_size: int) -> List[List[int]]:
    return [list(values[i : i + batch_size]) for i in range(0, len(values), batch_size)]


def build_detect_prompt(base_prompt: str, dirty_rows: Sequence[str], batch: Sequence[int]) -> str:
    prompt = base_prompt.rstrip() + "\n"
    for idx in batch:
        prompt += f"{dirty_rows[idx]}\n"
    return prompt


def build_verify_prompt(
    base_prompt: str,
    dirty_rows: Sequence[str],
    batch: Sequence[int],
    detection_lines: Sequence[str],
) -> str:
    prompt = base_prompt.rstrip() + "\n"
    for idx in batch:
        prompt += f"{dirty_rows[idx]}\n"
    prompt += "Here are the error detection results with reasons that you need to verify:\n"
    for line in detection_lines:
        tuple_no = _extract_tuple_number(line)
        if tuple_no is not None and tuple_no - 1 in batch:
            prompt += f"{line}\n"
    return prompt


def call_llm(prompt: str, params: Dict[str, Any], task_num: int = 0) -> str:
    if not params.get("api_use", True):
        return ""

    backend = str(params.get("llm_backend", "openai")).lower()
    if backend == "ollama":
        return _call_ollama(prompt, params, task_num)
    return _call_openai_compatible(prompt, params)


def _call_openai_compatible(prompt: str, params: Dict[str, Any]) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("openai is required for llm_backend='openai'.") from exc

    api_key = params.get("api_key") or os.getenv("OPENAI_API_KEY") or "EMPTY"
    kwargs = {"api_key": api_key}
    if params.get("base_url"):
        kwargs["base_url"] = params["base_url"]

    client = OpenAI(**kwargs)
    completion = client.chat.completions.create(
        model=params.get("model_name", "gpt-4o"),
        temperature=params.get("temperature", 0),
        messages=[
            {
                "role": "system",
                "content": params.get(
                    "system_prompt",
                    "You are a world-class data engineer, proficient in cleaning dirty data.",
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    return completion.choices[0].message.content or ""


def _call_ollama(prompt: str, params: Dict[str, Any], task_num: int = 0) -> str:
    try:
        import requests
    except ImportError as exc:
        raise ImportError("requests is required for llm_backend='ollama'.") from exc

    urls = params.get("ollama_urls")
    if not urls:
        urls = [params.get("base_url") or "http://localhost:11434/api/chat"]
    url = urls[task_num % len(urls)]

    response = requests.post(
        url,
        json={
            "model": params.get("model_name", "llama3"),
            "messages": [
                {
                    "role": "system",
                    "content": params.get(
                        "system_prompt",
                        "You are a world-class data engineer, proficient in cleaning dirty data.",
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "temperature": params.get("temperature", 0),
        },
        timeout=params.get("timeout", 120),
    )
    response.raise_for_status()
    data = response.json()
    return data.get("message", {}).get("content", "")


def extract_detection_lines(text: str) -> List[str]:
    if not text:
        return []

    lines = []
    for line in text.split("\n"):
        start_index = line.find("[")
        end_index = line.find("]")
        if start_index == -1 or end_index == -1:
            continue
        content = line[start_index + 1 : end_index]
        item = f"[{content}]"
        if "no error" in item.lower():
            continue
        lines.append(item)
    return lines


def extract_information(text: str) -> List[List[str]]:
    information = []
    for line in extract_detection_lines(text):
        content = line[1:-1].replace("'", "").replace('"', "")
        information.append([item.strip() for item in content.split(",")])
    return information


def parse_detection_records(
    text_or_lines: Iterable[str] | str,
    columns: Sequence[str],
) -> Tuple[List[Tuple[int, str]], List[Dict[str, Any]]]:
    if isinstance(text_or_lines, str):
        rows = extract_information(text_or_lines)
    else:
        rows = []
        for line in text_or_lines:
            rows.extend(extract_information(line))

    records: List[Tuple[int, str]] = []
    reasons: List[Dict[str, Any]] = []
    column_set = set(columns)

    for item in rows:
        if len(item) < 2:
            continue
        tuple_no = _extract_tuple_number(item[0])
        col_name = item[1].strip()
        if tuple_no is None or col_name not in column_set:
            continue
        row_pos = tuple_no - 1
        reason = ", ".join(item[2:]).strip() if len(item) > 2 else ""
        record = (row_pos, col_name)
        if record not in records:
            records.append(record)
            reasons.append({"row": row_pos, "column": col_name, "reason": reason})
    return records, reasons


def records_to_mask(records: Sequence[Tuple[int, str]], df: pd.DataFrame) -> pd.DataFrame:
    mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    for row_pos, col_name in records:
        if 0 <= row_pos < len(df) and col_name in mask.columns:
            mask.iat[row_pos, mask.columns.get_loc(col_name)] = True
    return mask


def detect_batch(
    batch_id: int,
    batch: Sequence[int],
    dirty_rows: Sequence[str],
    prompt_detect: str,
    params: Dict[str, Any],
) -> Tuple[List[str], str]:
    prompt = build_detect_prompt(prompt_detect, dirty_rows, batch)
    response = call_llm(prompt, params, task_num=batch_id)
    _write_response_file(params, "detect", batch, prompt, response)
    return extract_detection_lines(response), response


def verify_batch(
    batch_id: int,
    batch: Sequence[int],
    dirty_rows: Sequence[str],
    detection_lines: Sequence[str],
    prompt_verify: str,
    params: Dict[str, Any],
) -> Tuple[List[str], str]:
    prompt = build_verify_prompt(prompt_verify, dirty_rows, batch, detection_lines)
    batch_detection_lines = [
        line
        for line in detection_lines
        if _extract_tuple_number(line) is not None and _extract_tuple_number(line) - 1 in batch
    ]
    if not batch_detection_lines:
        return [], ""
    response = call_llm(prompt, params, task_num=batch_id)
    _write_response_file(params, "verify", batch, prompt, response)

    extracted_verify = extract_information(response)
    verified_lines = []
    for i, original_line in enumerate(batch_detection_lines):
        try:
            if "invalid" not in ",".join(extracted_verify[i]).lower():
                verified_lines.append(original_line)
        except Exception:
            verified_lines.append(original_line)
    return verified_lines, response


def detect_dataframe(df: pd.DataFrame, params: Dict[str, Any], logger: Optional[Logger] = None) -> DetectionResult:
    validate_params(params)
    prompt_detect = load_prompt(
        params.get("prompt_detect"),
        params.get("prompt_detect_path"),
        "detection",
    )
    prompt_verify = ""
    if params.get("use_verify"):
        prompt_verify = load_prompt(
            params.get("prompt_verify"),
            params.get("prompt_verify_path"),
            "verify",
        )

    dirty_rows = df_to_dirty_data(df)
    batches = build_batches(
        df,
        int(params.get("batch_size", 5)),
        params.get("batch_mode", "ourbat"),
        params.get("ref_column"),
        params.get("dataset_name"),
    )
    if logger:
        logger.info(f"Built {len(batches)} IterClean detection batches.")

    detection_lines: List[str] = []
    raw_detection_responses: List[str] = []
    total_batches = len(batches)
    completed_batches = 0

    with ThreadPoolExecutor(max_workers=int(params.get("max_workers", 4))) as executor:
        futures = {
            executor.submit(detect_batch, batch_id, batch, dirty_rows, prompt_detect, params): batch_id
            for batch_id, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            batch_id = futures[future]
            lines, response = future.result()
            detection_lines.extend(lines)
            raw_detection_responses.append(response)
            completed_batches += 1
            if logger:
                logger.info(
                    f"Detection batch {completed_batches}/{total_batches} finished "
                    f"(batch_id={batch_id}, detected_lines={len(lines)})."
                )

    raw_verify_responses: List[str] = []
    final_lines = detection_lines
    if params.get("use_verify") and detection_lines:
        final_lines = []
        total_verify_batches = len(batches)
        completed_verify_batches = 0
        if logger:
            logger.info(f"Starting IterClean verification for {total_verify_batches} batches.")
        with ThreadPoolExecutor(max_workers=int(params.get("max_workers", 4))) as executor:
            futures = {
                executor.submit(verify_batch, batch_id, batch, dirty_rows, detection_lines, prompt_verify, params): batch_id
                for batch_id, batch in enumerate(batches)
            }
            for future in as_completed(futures):
                batch_id = futures[future]
                lines, response = future.result()
                final_lines.extend(lines)
                if response:
                    raw_verify_responses.append(response)
                completed_verify_batches += 1
                if logger:
                    logger.info(
                        f"Verify batch {completed_verify_batches}/{total_verify_batches} finished "
                        f"(batch_id={batch_id}, verified_lines={len(lines)})."
                    )

    records, reasons = parse_detection_records(final_lines, df.columns)
    mask = records_to_mask(records, df)

    return DetectionResult(
        mask=mask,
        records=records,
        reasons=reasons,
        raw_detection_responses=raw_detection_responses,
        raw_verify_responses=raw_verify_responses,
    )


def _extract_tuple_number(text: str) -> Optional[int]:
    match = re.search(r"Tuple\s*(\d+)", str(text), flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def _write_response_file(
    params: Dict[str, Any],
    prefix: str,
    batch: Sequence[int],
    prompt: str,
    response: str,
) -> None:
    if not params.get("save_responses", True):
        return
    result_dir = params.get("result_dir")
    if not result_dir:
        return
    os.makedirs(result_dir, exist_ok=True)
    file_path = os.path.join(result_dir, f"{prefix}_tuple{list(batch)}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(prompt + "\n" + response)

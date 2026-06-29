import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import dataprep.tabular.detection.iterclean_modules as det_modules
except ModuleNotFoundError:
    import tabular.detection.iterclean_modules as det_modules


@dataclass
class CorrectionResult:
    corrected_df: pd.DataFrame
    repair_records: List[Dict[str, Any]]
    detection_records: List[Tuple[int, str]]
    detection_reasons: List[Dict[str, Any]]
    raw_detection_responses: List[str]
    raw_verify_responses: List[str]
    raw_repair_responses: List[str]


class Logger:
    def __init__(self, result_dir, verbose=True):
        os.makedirs(result_dir, exist_ok=True)
        self.logger = logging.getLogger(f"IterClean_Correction_Logger_{time.time()}")
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            file_handler = logging.FileHandler(os.path.join(result_dir, "iterclean_correction_log.txt"), encoding="utf-8")
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

            if verbose:
                stream_handler = logging.StreamHandler()
                stream_handler.setFormatter(formatter)
                self.logger.addHandler(stream_handler)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)


def validate_params(params):
    if int(params.get("batch_size", 0)) <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if int(params.get("max_round", 0)) <= 0:
        raise ValueError("max_round must be a positive integer.")
    if int(params.get("max_workers", 0)) <= 0:
        raise ValueError("max_workers must be a positive integer.")
    if params.get("batch_mode") not in {"sequential", "random", "group", "ourbat", "ranbat"}:
        raise ValueError("batch_mode must be one of: sequential, random, group, ourbat, ranbat.")
    if not params.get("prompt_repair") and not params.get("prompt_repair_path"):
        raise ValueError("IterClean correction requires the original repair prompt.")


def load_prompt(prompt, prompt_path, prompt_name):
    if prompt is not None:
        return prompt
    if prompt_path:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    raise ValueError(f"IterClean correction requires the original {prompt_name} prompt.")


def repair_dataframe(
    df,
    params,
    logger=None,
    detection_mask=None,
    detection_records=None,
    detection_reasons=None,
):
    validate_params(params)
    prompt_repair = load_prompt(params.get("prompt_repair"), params.get("prompt_repair_path"), "repair")

    current_df = df.copy()
    all_detection_records: List[Tuple[int, str]] = []
    all_detection_reasons: List[Dict[str, Any]] = []
    all_repair_records: List[Dict[str, Any]] = []
    all_raw_detect: List[str] = []
    all_raw_verify: List[str] = []
    all_raw_repair: List[str] = []

    max_round = int(params.get("max_round", 1))
    base_batch_size = int(params.get("batch_size", 5))
    batch_size_step = int(params.get("batch_size_step", 0))
    first_round_has_external_detection = (
        detection_mask is not None or detection_records is not None or detection_reasons is not None
    )

    for round_idx in range(max_round):
        round_params = dict(params)
        round_params["batch_size"] = base_batch_size + round_idx * batch_size_step
        round_no = round_idx + 1

        if logger:
            logger.info(f"===== IterClean correction round {round_no}/{max_round} =====")

        use_external_detection = first_round_has_external_detection and round_idx == 0
        round_detection_mask = detection_mask if use_external_detection else None
        round_detection_records = detection_records if use_external_detection else None
        round_detection_reasons = detection_reasons if use_external_detection else None

        detection_lines, round_records, round_reasons, raw_detect, raw_verify = _resolve_detection_results(
            current_df,
            round_params,
            logger,
            detection_mask=round_detection_mask,
            detection_records=round_detection_records,
            detection_reasons=round_detection_reasons,
        )
        all_detection_records.extend(round_records)
        all_detection_reasons.extend(round_reasons)
        all_raw_detect.extend(raw_detect)
        all_raw_verify.extend(raw_verify)

        if not detection_lines:
            if logger:
                logger.info(f"Round {round_no}: no errors detected; stopping correction.")
            break

        corrected_df, repair_records, raw_repair = _run_single_repair_round(
            current_df,
            detection_lines,
            prompt_repair,
            round_params,
            logger,
            round_no,
        )
        all_repair_records.extend(repair_records)
        all_raw_repair.extend(raw_repair)
        current_df = corrected_df

        if not repair_records:
            if logger:
                logger.info(f"Round {round_no}: no repair records parsed; stopping correction.")
            break

    return CorrectionResult(
        corrected_df=current_df,
        repair_records=all_repair_records,
        detection_records=all_detection_records,
        detection_reasons=all_detection_reasons,
        raw_detection_responses=all_raw_detect,
        raw_verify_responses=all_raw_verify,
        raw_repair_responses=all_raw_repair,
    )


def _run_single_repair_round(df, detection_lines, prompt_repair, params, logger, round_no):
    dirty_rows = det_modules.df_to_dirty_data(df)
    batches = det_modules.build_batches(
        df,
        int(params.get("batch_size", 5)),
        params.get("batch_mode", "ourbat"),
        params.get("ref_column"),
        params.get("dataset_name"),
    )

    raw_repair_responses: List[str] = []
    repair_text = ""
    total_batches = len(batches)
    completed_batches = 0

    if logger:
        logger.info(f"Round {round_no}: starting IterClean repair for {total_batches} batches.")

    with ThreadPoolExecutor(max_workers=int(params.get("max_workers", 4))) as executor:
        futures = {
            executor.submit(repair_batch, batch_id, batch, dirty_rows, detection_lines, prompt_repair, params): batch_id
            for batch_id, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            batch_id = futures[future]
            response = future.result()
            if response:
                repair_text += response + "\n"
                raw_repair_responses.append(response)
            completed_batches += 1
            if logger:
                logger.info(
                    f"Round {round_no}: repair batch {completed_batches}/{total_batches} finished "
                    f"(batch_id={batch_id}, response_chars={len(response)})."
                )

    repair_records = parse_repair_records(repair_text, df.columns)
    corrected_df = apply_repair_records(df, repair_records)
    return corrected_df, repair_records, raw_repair_responses


def repair_batch(batch_id, batch, dirty_rows, detection_lines, prompt_repair, params):
    prompt = build_repair_prompt(prompt_repair, dirty_rows, batch, detection_lines)
    if prompt is None:
        return ""
    response = det_modules.call_llm(prompt, params, task_num=batch_id)
    _write_response_file(params, "repair", batch, prompt, response)
    return response + "\n"


def build_repair_prompt(prompt_repair, dirty_rows, batch, detection_lines):
    prompt = prompt_repair.rstrip() + "\n"
    for idx in batch:
        prompt += str(dirty_rows[idx]) + "\n"

    prompt += "Here are error detection results within reasons, not mentioned cell values are right: \n"
    errors_num = 0
    for line in detection_lines:
        tuple_no = _extract_tuple_number(line)
        if tuple_no is not None and tuple_no - 1 in batch:
            prompt += line + "\n"
            errors_num += 1

    if errors_num == 0:
        return None
    return prompt


def parse_repair_records(text, columns):
    repair_records = []
    column_set = set(columns)
    for item in det_modules.extract_information(text):
        if len(item) < 3:
            continue
        tuple_no = _extract_tuple_number(item[0])
        column = item[1].strip()
        if tuple_no is None or column not in column_set:
            continue
        repair_records.append({
            "row": tuple_no - 1,
            "column": column,
            "value": item[2],
        })
    return repair_records


def apply_repair_records(df, repair_records):
    corrected_df = df.copy()
    for record in repair_records:
        row = record["row"]
        column = record["column"]
        value = record["value"]
        if 0 <= row < len(corrected_df) and column in corrected_df.columns:
            corrected_df.iloc[row, corrected_df.columns.get_loc(column)] = value
    return corrected_df


def _resolve_detection_results(
    df,
    params,
    logger,
    detection_mask=None,
    detection_records=None,
    detection_reasons=None,
):
    raw_detect: List[str] = []
    raw_verify: List[str] = []

    if detection_reasons:
        records = [(int(item["row"]), item["column"]) for item in detection_reasons]
        return _reasons_to_lines(detection_reasons), records, list(detection_reasons), raw_detect, raw_verify

    if detection_records:
        reasons = [
            {"row": row, "column": column, "reason": "Reason: detected dirty cell"}
            for row, column in detection_records
        ]
        return _reasons_to_lines(reasons), list(detection_records), reasons, raw_detect, raw_verify

    if detection_mask is not None:
        records, reasons = _mask_to_records(detection_mask)
        return _reasons_to_lines(reasons), records, reasons, raw_detect, raw_verify

    if logger:
        logger.info("No external detection result supplied; running IterClean detection before repair.")
    det_result = det_modules.detect_dataframe(df, params, logger)
    lines = _reasons_to_lines(det_result.reasons)
    return (
        lines,
        det_result.records,
        det_result.reasons,
        det_result.raw_detection_responses,
        det_result.raw_verify_responses,
    )


def _mask_to_records(mask):
    mask = mask.astype(bool)
    records = []
    reasons = []
    for col in mask.columns:
        for row, is_error in mask[col].items():
            if is_error:
                row_pos = int(row)
                records.append((row_pos, col))
                reasons.append({"row": row_pos, "column": col, "reason": "Reason: detected dirty cell"})
    return records, reasons


def _reasons_to_lines(reasons):
    lines = []
    for item in reasons:
        row = int(item["row"])
        column = item["column"]
        reason = item.get("reason", "")
        if reason and not str(reason).lower().startswith("reason:"):
            reason = "Reason: " + str(reason)
        elif not reason:
            reason = "Reason: detected dirty cell"
        lines.append(f"[Tuple{row + 1}, {column}, {reason}]")
    return lines


def _extract_tuple_number(text):
    match = re.search(r"Tuple\s*(\d+)", str(text), flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def _write_response_file(params, prefix, batch, prompt, response):
    if not params.get("save_responses", True):
        return
    result_dir = params.get("result_dir")
    if not result_dir:
        return
    os.makedirs(result_dir, exist_ok=True)
    file_path = os.path.join(result_dir, f"{prefix}_tuple{list(batch)}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(prompt + "\n" + response)

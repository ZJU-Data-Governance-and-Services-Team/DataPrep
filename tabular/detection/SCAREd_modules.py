# -*- coding: utf-8 -*-
"""SCAREd detection-side helpers for DataPrep.

SCAREd is repair-first in the teacher code.  Detection is obtained from changed
cells after running the correction logic; shared candidate-generation/KHS logic
is delegated to the correction module to avoid two divergent implementations.
"""

try:
    import dataprep.tabular.correction.SCAREd_modules as repair_modules
except ModuleNotFoundError:
    import tabular.correction.SCAREd_modules as repair_modules

# Detection API
detect_errors = repair_modules.detect_errors
build_mask_from_clean = repair_modules.build_mask_from_clean
mask_to_detection_dictionary = repair_modules.mask_to_detection_dictionary
check_string = repair_modules.check_string
handler = repair_modules.handler
SCAREd = repair_modules.SCAREd
SCAREdCleaner = repair_modules.SCAREdCleaner


def generate_error_mask(
    dirty_df,
    clean_df=None,
    detection_mask=None,
    reliable_attrs=None,
    n_reliable_attrs=2,
    perfected=False,
    use_perfect_detection_if_clean=False,
    repair_attrs=None,
    min_partition_size=1,
    max_partition_values=None,
    use_index_partition=True,
):
    """Return SCAREd detection mask: repaired_df != dirty_df."""
    return repair_modules.detect_errors(
        dirty_df=dirty_df,
        clean_df=clean_df,
        detection_mask=detection_mask,
        reliable_attrs=reliable_attrs,
        n_reliable_attrs=n_reliable_attrs,
        perfected=perfected,
        use_perfect_detection_if_clean=use_perfect_detection_if_clean,
        repair_attrs=repair_attrs,
        min_partition_size=min_partition_size,
        max_partition_values=max_partition_values,
        use_index_partition=use_index_partition,
    )


__all__ = [
    "detect_errors", "generate_error_mask", "build_mask_from_clean",
    "mask_to_detection_dictionary", "check_string", "handler",
    "SCAREd", "SCAREdCleaner",
]

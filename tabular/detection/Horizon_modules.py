# -*- coding: utf-8 -*-
"""Horizon detection-side helpers for DataPrep.

The teacher's Horizon algorithm is repair-first.  The detection split therefore
keeps detection-oriented APIs here and delegates the shared FD-pattern machinery
to the correction module to avoid duplicating algorithm code.
"""

try:
    import dataprep.tabular.correction.Horizon_modules as repair_modules
except ModuleNotFoundError:
    import tabular.correction.Horizon_modules as repair_modules

# Detection API
detect_errors = repair_modules.detect_errors
calDetPrecRec = repair_modules.calDetPrecRec
dirty_cells = repair_modules.dirty_cells
check_string = repair_modules.check_string
calF1 = repair_modules.calF1


def generate_error_mask(dirty_df, rule_path=None, rules=None):
    """Return Horizon detection mask: repaired_df != dirty_df."""
    return repair_modules.detect_errors(dirty_df=dirty_df, rule_path=rule_path, rules=rules)


__all__ = [
    "detect_errors", "generate_error_mask", "calDetPrecRec", "dirty_cells",
    "check_string", "calF1",
]

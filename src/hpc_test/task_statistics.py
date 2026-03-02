"""CPU task: compute derived statistics from the matrix result."""

from typing import Annotated

import pandas as pd

from hpc_test.config import data_catalog


def task_statistics(
    matrix_result: Annotated[pd.DataFrame, data_catalog["matrix_result"]],
) -> Annotated[pd.DataFrame, data_catalog["statistics_report"]]:
    """Compute range, coefficient of variation, and total elements."""
    stats = matrix_result.copy()
    stats["range"] = stats["max"] - stats["min"]
    stats["cv"] = stats["std"] / stats["mean"].abs()
    stats["total_elements"] = 500 * 500
    return stats

"""CPU task: multiply two random matrices and return summary statistics."""

from typing import Annotated

import numpy as np
import pandas as pd
import pytask

from hpc_test.config import data_catalog


def _multiply_and_summarize(size: int, seed: int) -> pd.DataFrame:
    """Multiply two random matrices and return summary statistics."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((size, size))
    b = rng.standard_normal((size, size))
    result = a @ b
    return pd.DataFrame(
        {
            "mean": [result.mean()],
            "std": [result.std()],
            "min": [result.min()],
            "max": [result.max()],
        }
    )


def task_matrix_multiply() -> Annotated[
    pd.DataFrame, data_catalog["matrix_result"]
]:
    """Multiply two 500x500 random matrices and store summary stats."""
    return _multiply_and_summarize(size=500, seed=42)

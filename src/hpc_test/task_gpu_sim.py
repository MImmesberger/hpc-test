"""GPU task: element-wise math on a large matrix (SLURM GPU partition)."""

from typing import Annotated

import numpy as np
import pandas as pd
import pytask

from hpc_test.config import data_catalog


@pytask.mark.slurm(partition="sgpu_devel", gpus=1, mem="4G", time="00:05:00")
def task_gpu_simulation() -> Annotated[
    pd.DataFrame, data_catalog["gpu_result"]
]:
    """Element-wise sin + cos * exp on a 2000x2000 matrix."""
    rng = np.random.default_rng(123)
    matrix = rng.standard_normal((2000, 2000))
    result = np.sin(matrix) + np.cos(matrix) * np.exp(-np.abs(matrix))
    return pd.DataFrame(
        {
            "mean": [result.mean()],
            "std": [result.std()],
            "min": [result.min()],
            "max": [result.max()],
        }
    )

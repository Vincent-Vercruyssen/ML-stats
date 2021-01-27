"""

Utility functions for the classifier comparisons.

Author: Vincent Vercruyssen
Year:   2021

"""

import time
import numpy as np
import pandas as pd

from collections import OrderedDict


# -----------------------------------------
# Functions
# -----------------------------------------


def _check_input_matrix(matrix):
    """Check the input matrix and transform to suitable format.

    Input
    -----
    matrix : pd.DataFrame
        Rows are datasets, columns are methods, and values are the metric.

    Output
    ------
    matrix : np.ndarray
        Contains the metric values.
    methods : dict
        Contains the method names. Each number refers to the column index of matrix.
    datasets : dict
        Contains the dataset names. Each number refers to the row index of matrix.
    """

    # check the type
    if not (isinstance(matrix, pd.DataFrame)):
        raise TypeError("Input `matrix` can only be a DataFrame")

    # collect the methods and the datasets
    try:
        methods = OrderedDict({i: str(m) for i, m in enumerate(matrix.columns)})
        datasets = OrderedDict({i: str(d) for i, d in enumerate(matrix.index.values)})
    except:
        raise Exception("Could not extract dataset and method names from `matrix`")

    # extract the metric values
    try:
        metric_values = matrix.values.copy()
    except:
        raise Exception("Could not extract metric values from `matrix`")

    # check for nan
    nan_indices = np.where(np.isnan(metric_values))
    if len(nan_indices[0]) > 0:
        print(
            "`matrix` contains {} nan values. They will be changed to the average!".format(
                len(nan_indices[0])
            )
        )
        column_mean = np.nanmean(metric_values, axis=0)
        metric_values[nan_indices] = np.take(column_mean, nan_indices[1])

    # high precision
    matrix = metric_values.astype(np.float64)

    # info
    print("Number of compared methods:", len(methods))
    print("Number of datasets:", len(datasets))

    return matrix, methods, datasets

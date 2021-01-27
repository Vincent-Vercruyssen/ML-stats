"""

Functions for the comparison of two or more classifiers:
- compute the average ranks
- compute win/ties/losses

Author: Vincent Vercruyssen
Year:   2021

"""

import time
import numpy as np
import scipy as sp
import pandas as pd

from collections import OrderedDict

from .utils import _check_input_matrix


# -----------------------------------------
# Block-design class
# -----------------------------------------


class BlockDesign:
    """Class for the results of the experiment in the block design.

    Input
    -----
    matrix : pd.DataFrame
        Contains the results of the experiment:
        rows        = datasets/blocks
        columns     = models/groups
        values      = metric (e.g., accuracy, F1...)
    threshold : float
        The threshold to determine whether two values in matrix are equal or distinct.
    precision : int
        The precision of the values in `matrix`.
        Has an impact on the ranks, number of wins/ties/losses...
    higher_is_better : bool
        Indicates whether higher metric values in `matrix` equal a better result.
    verbose : bool
        Controls verbosity.

    Remark
    ------
    The shape of `matrix` is (N, k) with:
    k = number of methods that are compared.
    N = number of datasets on which comparison is done.
    """

    def __init__(
        self, matrix, threshold=0.001, precision=4, higher_is_better=True, verbose=True
    ):
        # check matrix
        self.matrix_array, self.methods, self.datasets = _check_input_matrix(matrix)
        self.matrix_df = matrix
        self.N = len(self.datasets)
        self.k = len(self.methods)

        # check threshold
        if 0.0 <= float(threshold):
            self.threshold = float(threshold)

        # check precision
        if 0 <= int(precision):
            self.precision = int(precision)

        # check higher_is_better
        if isinstance(higher_is_better, bool):
            self.higher_is_better = higher_is_better

        # other parameters
        self.verbose = bool(verbose)

        # uninstantiated parameters
        self.average_ranks = None
        self.wins_ties_losses = None

    def to_ranks(self):
        """ Transform the `matrix` results into average rank matrix. """

        # compute the average ranks
        self.average_ranks = _compute_average_ranks(
            self.matrix_df,
            threshold=self.threshold,
            precision=self.precision,
            higher_is_better=self.higher_is_better,
        )

        return self.average_ranks

    def to_wins_ties_losses(self):
        """ Transform the `matrix` results into win/tie/loss matrix. """

        # compute the wins/ties/losses
        self.wins_ties_losses = _compute_wins_losses(
            self.matrix_df,
            method="",
            comparison="all-v-all",
            threshold=self.threshold,
            precision=self.precision,
            higher_is_better=self.higher_is_better,
        )

        return self.wins_ties_losses


# -----------------------------------------
# Functions
# -----------------------------------------


def _compute_average_ranks(
    matrix,
    threshold=0.001,
    precision=4,
    higher_is_better=True,
):
    """Compute the average rank of each method.

    Input
    -----
    matrix : pd.DataFrame
        Rows are datasets, columns are methods, and values are the compared metric.
    threshold : float
        The threshold to consider two metric values equal or different.
    precision : int
        The precision to which to consider metric differences when computing the rank.
    higher_is_better : bool
        Higher values in the matrix are better (or not).

    Output
    ------
    average_ranks : pd.DataFrame
        Contains the average rank +/- standard deviation for each method.

    Remark
    ------
    By setting a certain precision and the threshold = 0, ranks are only compared on a precision basis.

    TODO: add input checking on parameters
    """

    # check the input
    matrix, methods, datasets = _check_input_matrix(matrix)
    M = len(methods)
    N = len(datasets)

    # convert the matrix if necessary
    if higher_is_better:
        matrix = 1.0 - matrix

    # precision warning
    if threshold > (1 / (10 ** precision)):
        print(
            "WARNING: threshold-based rank computation is not perfect. Consider lowering the precision."
        )

    # compute the ranking on each dataset
    ranks_per_dataset = np.zeros((N, M), dtype=np.float64)
    for i, scores in enumerate(matrix):
        scores = np.round(scores, precision)
        ranks_per_dataset[i, :] = _rank_single_dataset(scores, threshold)

    # compute the average ranks
    average_ranks = pd.DataFrame(
        0.0,
        columns=["average rank", "standard deviation"],
        index=[m for _, m in methods.items()],
    )
    average_ranks["average rank"] = np.mean(ranks_per_dataset, axis=0)
    average_ranks["standard deviation"] = np.std(ranks_per_dataset, axis=0)

    return average_ranks


def _compute_wins_losses(
    matrix,
    method="",
    comparison="all-v-all",
    threshold=0.001,
    precision=4,
    higher_is_better=True,
):
    """Compute the wins, ties, losses for each method.

     Input
    -----
    matrix : pd.DataFrame
        Rows are datasets, columns are methods, and values are the metric.
    method : str or int
        Name or index of the method in the one-v-all scenario.
    comparison : str
        All-v-all or one-v-all comparison.
    threshold : float
        The threshold to consider two metrics equal or different.
    precision : int
        The precision to which to consider metric differences when computing the rank.
    higher_is_better : bool
        Higher values in the matrix are better (or not).

    Output
    ------
    wins_ties_losses : pd.DataFrame
        Contains the wins/ties/losses for all pairwise comparisons (all-v-all).
        OR Contains the wins/ties/losses for one method (one-v-all).

    TODO: add input checking on parameters
    """

    # check the input
    matrix, methods, datasets = _check_input_matrix(matrix)
    M = len(methods)
    N = len(datasets)

    inv_methods = OrderedDict({v: k for k, v in methods.items()})

    # convert the matrix if necessary
    if not (higher_is_better):
        matrix = 1.0 - matrix

    # precision warning
    if threshold > (1 / (10 ** precision)):
        print(
            "WARNING: threshold-based rank computation is not perfect. Consider lowering the precision."
        )
    matrix = np.round(matrix, precision)

    # compute the wins/ties/losses
    if comparison.lower() == "all-v-all":
        wins_ties_losses = pd.DataFrame(
            "", columns=methods.values(), index=methods.values()
        )

        # add wins/ties/losses
        for _, m1 in methods.items():
            m1_scores = matrix[:, inv_methods[m1]]
            for _, m2 in methods.items():
                if m1 == m2:
                    wins_ties_losses.loc[m1, m2] = "0-{}-0".format(N)
                else:
                    m2_scores = matrix[:, inv_methods[m2]]
                    diff = m1_scores - m2_scores
                    nw = len(np.where(diff > threshold)[0])
                    nl = len(np.where(diff < -threshold)[0])
                    nt = len(np.where(abs(diff) <= threshold)[0])
                    assert (
                        N - nt == nw + nl
                    ), "Error in computing number of wins/ties/losses"
                    wins_ties_losses.loc[m1, m2] = "{}-{}-{}".format(nw, nt, nl)

    elif comparison.lower() == "one-v-all":
        wins_ties_losses = pd.DataFrame(
            0, columns=["wins", "ties", "losses"], index=methods.values()
        )

        # add wins/ties/losses
        m1_scores = matrix[:, inv_methods[method]]
        for _, m in methods.items():
            if m == method:
                wins_ties_losses.loc[m, "ties"] = N
            else:
                m2_scores = matrix[:, inv_methods[m]]
                diff = m1_scores - m2_scores
                nw = len(np.where(diff > threshold)[0])
                nl = len(np.where(diff < -threshold)[0])
                nt = len(np.where(abs(diff) <= threshold)[0])
                assert (
                    N - nt == nw + nl
                ), "Error in computing number of wins/ties/losses"
                wins_ties_losses.loc[m, :] = [nw, nt, nl]
    else:
        raise Exception("Unkown value for parameter `comparison`")

    return wins_ties_losses


# -----------------------------------------
# Helper functions
# -----------------------------------------


def _rank_single_dataset(scores, rank_threshold):
    """Compute the rank of each method on a single dataset.

    Input
    -----
    dataset_scores : array
        Scores of the methods for a single dataset. Lower is better.
    rank_threshold : float
        The threshold to consider two metrics equal or different.

    Output
    ------
    ranks : np.array
        Rank of each method on this dataset.

    Remark
    ------
    The problem of consecutive comparison of elements in the array can allow for the same rank
    between the first element and last element of a series of subsequent elements, where the distance
    between the two is larger than the allowed distance, but the increments between all the consecutive
    elements are smaller than the allowed distance. Therefore, they all get the same rank,
    although technically they shouldn't have the same rank.

    This method does not do that. Instead it anchors the score at ip0 and then compares the consecutive
    scores until a score is reached at ip1 such that: score(ip1) - score(ip0) > rank_threshold.
    At that point, all previous scores get the same rank, and the anchorpoint is moved to score at ip1.
    """

    n = len(scores)

    # if ranking is precision-based and 2 scores are equal, ranking is arbitrary
    sorter = np.argsort(scores)
    sorted_scores = np.sort(scores)
    inv_sorter = np.empty(n, dtype=np.intp)
    inv_sorter[sorter] = np.arange(n, dtype=np.intp)

    # compute the rank
    ranks = np.arange(1, n + 1, 1, dtype=np.float64)
    ip0 = 0
    c = 1  # counter
    for ip1 in range(1, n):
        if (sorted_scores[ip1] - sorted_scores[ip0]) > rank_threshold:
            # fill in the ranks of all previous values + reset index pointer
            ranks[ip0:ip1] = np.sum(ranks[ip0:ip1]) / c
            ip0 = ip1
            c = 1
        elif ip1 == n - 1:
            # end-condition necessary because ip1 only goes to last element
            ranks[ip0:] = np.sum(ranks[ip0:]) / (c + 1)
        else:
            c += 1

    # rearrange according to original array
    ranks = ranks[inv_sorter]

    return ranks

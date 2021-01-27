"""

Contains unit tests for the functions in the package.

Author: Vincent Vercruyssen
Year:   2021

"""

import time
import numpy as np
import pandas as pd

from .classifier_comparisons import _rank_single_dataset


# -----------------------------------------
# Main function
# -----------------------------------------


def main():
    """ Run the unit tests. """

    # src.classifier_comparisons._rank_single_dataset
    _test_rank_single_dataset()


# -----------------------------------------
# Unit tests
# -----------------------------------------


def _test_rank_single_dataset():
    """ Unit tests for src.classifier_comparisons._rank_single_dataset. """

    print("\nTesting: src.classifier_comparisons._rank_single_dataset:")

    # test 0
    test = np.array([0.3, 0.1, 0.04, 0.6, 1.0])
    solution = np.array([3, 2, 1, 4, 5])
    print("Expected:", solution, "-- Method:", _rank_single_dataset(test, 0.01))

    # test 1
    test = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    solution = np.array([3, 3, 3, 3, 3])
    print("Expected:", solution, "-- Method:", _rank_single_dataset(test, 0.01))

    # test 2
    test = np.array([0.0, 0.0, 0.0, 0.0, 0.1])
    solution = np.array([2.5, 2.5, 2.5, 2.5, 5])
    print("Expected:", solution, "-- Method:", _rank_single_dataset(test, 0.01))

    # test 3
    test = np.array([0.0, 0.5])
    solution = np.array([1, 2])
    print("Expected:", solution, "-- Method:", _rank_single_dataset(test, 0.01))

    # test 4
    test = np.array([0.0, 0.5])
    solution = np.array([1.5, 1.5])
    print("Expected:", solution, "-- Method:", _rank_single_dataset(test, 0.500001))

    # test 5
    test = np.array([0.0, 0.1, 0.22, 0.2, 0.33, 0.3301, 1.0, 1.0, 1.01])
    solution = np.array([1, 2, 4, 3, 5.5, 5.5, 8, 8, 8])
    print("Expected:", solution, "-- Method:", _rank_single_dataset(test, 0.0101))

    # test 6
    test = np.array([0.0, 0.1, 0.22, 0.2, 0.33, 0.3301, 1.0, 1.0, 1.01])
    solution = np.array([1, 2, 4, 3, 5.5, 5.5, 7.5, 7.5, 9])
    print("Expected:", solution, "-- Method:", _rank_single_dataset(test, 0.01))

    # test 7
    test = np.array([0.002, 0.002, 0.100, 0.101, 0.500, 0.500, 0.500])
    solution = np.array([1.5, 1.5, 3, 4, 6, 6, 6])
    print("Expected:", solution, "-- Method:", _rank_single_dataset(test, 0.0))


if __name__ == "__main__":
    main()
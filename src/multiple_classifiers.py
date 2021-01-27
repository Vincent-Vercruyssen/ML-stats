"""

Functions for the statistical comparison of multiple classifiers
1. Parametric:
2. Non-parametric:
    - Friedman
3. Post-hocs:
    - Nemenyi
    - Bonferroni-Dunn

Terminology often used in literature is:
- blocks = datasets
- groups = treatment = the classifiers/methods

Author: Vincent Vercruyssen
Year:   2021

"""

import time
import numpy as np
import pandas as pd
import scipy.stats as sps

from statsmodels.stats.libqsturng import psturng, qsturng

from .classifier_comparisons import BlockDesign


# -----------------------------------------
# GLOBAL PARAMETERS
# -----------------------------------------

G_TOL_ = 1e-8


# -----------------------------------------
# Parametric
# -----------------------------------------


# -----------------------------------------
# Non-parametric
# -----------------------------------------


def friedman_test(
    block_design,
    alpha=0.05,
    verbose=True,
):
    """Friedman test.

    This is the non-parametric equivalent of the repeated-measures ANOVA.

    H_0 : All methods perform similarly.
    H_a : At least two methods do not perform similarly.
    --> two-tailed test?

    Based on:   Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets.
                Journal of Machine learning research, 7, 1-30.

    Input
    -----
    block_design : instance of BlockDesign class
        Contains the results of the experiments.
    alpha : float
        Alpha value for the statistical test.
    verbose : bool
        Controls verbosity.

    Output
    ------
    test_results : pd.DataFrame
        Contains the test results.

    TODO: input checking + error handling etc.
    """

    # parameters
    N, k = block_design.N, block_design.k
    if not (N > 1 and k > 1):
        raise Exception("Not enough datasets and methods.")
    if not (N >= 10 and k >= 5):
        print("WARNING: exact critical values should be used [Demsar, 2006]")

    # compute the average ranks
    average_ranks = block_design.to_ranks()

    # store the results
    test_results = pd.DataFrame(
        columns=[
            "alpha",
            "critical value",
            "test statistic",
            "p value",
            "significant? (test stat > crit value)",
        ]
    )

    # 1. Friedman test: chi square distribution with k - 1 degrees of freedom
    test_stat = ((12 * N) / (k * (k + 1))) * (
        np.sum(np.power(average_ranks["average rank"].values, 2))
        - (k * ((k + 1) ** 2)) / 4
    )
    dof = k - 1

    crit_val = sps.chi2.ppf(1.0 - alpha, df=dof)
    p_val = 1.0 - sps.chi2.cdf(test_stat, df=dof) * 1.0
    if p_val < G_TOL_:
        p_val = 0.0

    # significant? (use p_val, more robust)
    sign = True if p_val <= alpha else False

    # store
    test_results.loc["Friedman test", :] = [
        alpha,
        crit_val,
        test_stat,
        p_val,
        sign,
    ]

    # 2. Iman and Davenport test (because Friedman is conservative)
    # distributed according to F-distribution with k - 1 and (k - 1)(N - 1) degrees of freedom
    test_stat = ((N - 1) * test_stat) / (N * (k - 1) - test_stat)
    dof1 = k - 1
    dof2 = (k - 1) * (N - 1)

    crit_val = sps.f.ppf(1.0 - alpha, dfn=dof1, dfd=dof2)
    p_val = 1.0 - sps.f.cdf(test_stat, dfn=dof1, dfd=dof2) * 1.0
    if p_val < G_TOL_:
        p_val = 0.0

    # significant? (use p_val, more robust)
    sign = True if p_val <= alpha else False

    # store
    test_results.loc["Iman Davenport", :] = [
        alpha,
        crit_val,
        test_stat,
        p_val,
        sign,
    ]

    return test_results


def friedman_aligned_rank_test(
    block_design,
    alpha=0.05,
    verbose=True,
):
    """Friedman aligned ranks test.

    H_0 : All methods perform similarly.
    H_a : At least two methods do not perform similarly.
    --> two-tailed test?

    Based on:   Hodges, J. L., and Lehmann, E. L. (2012). Rank methods for combination of independent experiments in analysis of variance.
                In Selected Works of EL Lehmann (pp. 403-418). Springer, Boston, MA.

    Input
    -----
    block_design : instance of BlockDesign class
        Contains the results of the experiments.
    alpha : float
        Alpha value for the statistical test.
    verbose : bool
        Controls verbosity.

    Output
    ------
    test_results : pd.DataFrame
        Contains the test results.

    TODO: input checking + error handling etc.
    """

    # parameters
    N, k = block_design.N, block_design.k
    if not (N > 1 and k > 1):
        raise Exception("Not enough datasets and methods.")
    if not (N >= 10 and k >= 5):
        print("WARNING: exact critical values should be used [Demsar, 2006]")

    # compute the average ranks
    average_ranks = block_design.to_ranks()

    # store the results
    test_results = pd.DataFrame(
        columns=[
            "alpha",
            "critical value",
            "test statistic",
            "p value",
            "significant? (test stat > crit value)",
        ]
    )

    return test_results


# -----------------------------------------
# Post-hocs
# -----------------------------------------


def nemenyi_friedman_test(
    block_design,
    alpha=0.05,
    verbose=True,
):
    """Nemenyi post-hoc test for all pairwise comparisons between different methods.

    This test is usually conducted if significant results for the Friedman test are obtained.
    It is similar to the Tukey test for ANOVA and is used when all classifiers are compared to
    each other.

    H_0 : The performance of two methods is not different.
    H_a : The performance of two methods is different.
    --> two-tailed test

    Using statsmodels.stats.libqsturng for the Studentized range statistic.
    The q and p values are extracted as in a one-tailed test!

    Based on:   Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets.
                Journal of Machine learning research, 7, 1-30.

    Input
    -----
    block_design : instance of BlockDesign class
        Contains the results of the experiments.
    alpha : float
        Alpha value for the statistical test.
    verbose : bool
        Controls verbosity.

    Output
    ------
    p_values : pd.DataFrame
        Matrix structure of the p-values of all pairwise comparisons.
    significant_differences : pd.DataFrmae
        Matrixt structure of the significant differences of all
        pairwise comparisons under the given alpha value.

    TODO: input checking + error handling etc.
    """

    # parameters
    N, k = block_design.N, block_design.k
    average_ranks = block_design.to_ranks()

    # compute the p values and significant differences
    p_values = pd.DataFrame(
        1.0,
        columns=[m for _, m in block_design.methods.items()],
        index=[m for _, m in block_design.methods.items()],
    )
    significant_differences = pd.DataFrame(
        False,
        columns=[m for _, m in block_design.methods.items()],
        index=[m for _, m in block_design.methods.items()],
    )

    correction = np.sqrt(2) / (np.sqrt((k * (k + 1)) / (6 * N)))

    for i in range(k):
        m1 = block_design.methods[i]
        for j in range(i + 1, k):
            m2 = block_design.methods[j]

            # rank difference
            r = abs(
                average_ranks.loc[m1, "average rank"]
                - average_ranks.loc[m2, "average rank"]
            )

            # compute p value under studentized range statistic
            test_stat = r * correction
            p_val = psturng(test_stat, r=k, v=np.inf) * 1.0
            if p_val < G_TOL_:
                p_val = 0.0

            # significant?
            sign = True if p_val <= alpha else False

            # alternative way to get this result
            # this is purely to doublecheck the p-value-based way
            crit_val_a = qsturng(1.0 - alpha, r=k, v=np.inf)
            test_stat_a = r * correction
            sign_a = True if test_stat_a >= crit_val_a else False
            assert (
                s_a == s
            ), "ERROR: two ways of testing the hypothesis produce different results"

            # store
            p_values.loc[m1, m2] = p_val
            p_values.loc[m2, m1] = p_val
            significant_differences.loc[m1, m2] = sign
            significant_differences.loc[m2, m1] = sign

    # return results
    p_values = p_values.style.set_caption("Two-tailed p values")
    significant_differences = significant_differences.style.set_caption(
        "Significant differences under alpha = {}".format(alpha)
    )

    return p_values, significant_differences


def bonferroni_dunn_test(
    block_design,
    control,
    alpha=0.05,
    verbose=True,
):
    """Bonferroni Dunn test for comparing all methods against a control method.

    This test is used when all methods are compared against a control method and
    controls the family-wise error in multiple hypothesis testing.

    H_0 : The performance of a method and the control are similar.
    H_a : The performance of a method and the control are different.
    --> two-tailed test

    Based on:   Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets.
                Journal of Machine learning research, 7, 1-30.

    Input
    -----
    block_design : instance of BlockDesign class
        Contains the results of the experiments.
    control : str
        Name of the control method/group
    alpha : float
        Alpha value for the statistical test.
    verbose : bool
        Controls verbosity.

    Output
    ------
    p_values : pd.DataFrame
        Matrix structure of the p-values of all pairwise comparisons.
    significant_differences : pd.DataFrmae
        Matrixt structure of the significant differences of all
        pairwise comparisons under the given alpha value.

    TODO: input checking + error handling etc.
    TODO: alternative calculation like the Nemenyi test.
    """

    # parameters
    N, k = block_design.N, block_design.k
    average_ranks = block_design.to_ranks()

    if not (control in block_design.methods.values()):
        raise Exception("`control` should be a method in the BlockDesign")

    # store the results
    p_values = pd.DataFrame(
        1.0,
        columns=[m for _, m in block_design.methods.items()],
        index=[control],
    )
    significant_differences = pd.DataFrame(
        False,
        columns=[m for _, m in block_design.methods.items()],
        index=[control],
    )

    # compute the statistical tests
    correction = np.sqrt((k * (k + 1)) / (6 * N))
    alpha = alpha / (k - 1)
    if verbose:
        print("Bonferroni-Dunn-corrected alpha value:", alpha_corrected)

    for i in range(k):
        m = block_design.methods[i]
        if m == control:
            continue

        # rank difference
        r = abs(
            average_ranks.loc[control, "average rank"]
            - average_ranks.loc[m, "average rank"]
        )

        # compute the p value under the normal distribution
        z_stat = r / correction
        p_val = sps.norm.sf(z_stat, loc=0.0, scale=1.0) * 2.0
        if p_val < G_TOL_:
            p_val = 0.0

        # TWO-TAILED TEST: significant?
        sign = True if p_val <= alpha_corrected else False

        # store
        p_values.loc[control, m] = p_val
        significant_differences.loc[control, m] = sign

    # return results
    p_values = p_values.style.set_caption("Two-tailed p values")
    significant_differences = significant_differences.style.set_caption(
        "Significant differences under alpha = {}".format(alpha_corrected)
    )

    return p_values, significant_differences

# ML-stats
A Python package with some statistical tools for evaluating Machine Learning models.

# Table of contents

<!-- TOC -->autoauto- [ML-stats](#ml-stats)auto- [Table of contents](#table-of-contents)auto- [Installation](#installation)auto- [Use the package](#use-the-package)auto- [Current contents of the package](#current-contents-of-the-package)autoauto<!-- /TOC -->

# Installation

To install the package, simply download the files for now, add the source to your path in python, and import the necessary functionality.

# Use the package

To do a statistical test, you need to:
1. Construct a matrix of the experimental results. The rows are the datasets/blocks, the columns are the methods/groups, and the values of the matrix are the recorded performance metric (i.e., what the experiment measures). For now, let's assume random results:

``` python
import pandas as pd

matrix = pd.DataFrame(
    np.random.randn(2, 2), 
    columns=['method1', 'method2'], 
    index=['dataset1', 'dataset2']
)
```

2. Create an instance of the ```BlockDesign``` class. The BlockDesign class stores the results and preprocesses them for later use. You can specify precision, threshold...

``` python
from src.classifier_comparisons import BlockDesign

block_design = BlockDesign(matrix, threshold=0.01, precision=4, higher_is_better=True)
```

3. Give this instance to the appropriate statistical test.

``` python
test_results = friedman_test(block_design, alpha=0.05)
```

# Current contents of the package

Assuming that the results are stored in a matrix (let's create a random one for now):

``` python
import pandas as pd

matrix = pd.DataFrame(
    np.random.randn(2, 2), 
    columns=['method1', 'method2'], 
    index=['dataset1', 'dataset2']
)
```

Comparing classifiers:
1. Compute the average ranks (Friedman)

    ``` python
    from src.classifier_comparisons import BlockDesign
    
    average_ranks = BlockDesign(matrix).to_ranks()
    ```

2. Compute the wins/ties/losses between different methods

    ``` python
    from src.classifier_comparisons import BlockDesign
    
    average_ranks = BlockDesign(matrix).to_wins_ties_losses()
    ```

Non-parametric tests:
1. Friedman test

    ``` python
    from src.multiple_classifiers import friedman_test
    
    block_design = BlockDesign(matrix)
    test_results = friedman_test(block_design, alpha=0.05)
    ```

Non-parametric post-hoc tests:
1. Nemenyi Friedman test

    ``` python
    from src.multiple_classifiers import nemenyi_friedman_test
    
    block_design = BlockDesign(matrix)
    p_values, sign_diffs = nemenyi_friedman_test(block_design, alpha=0.05)
    ```

2. Bonferroni-Dunn test

    ``` python
    from src.multiple_classifiers import bonferroni_dunn_test
    
    block_design = BlockDesign(matrix)
    p_values, sign_diffs = bonferroni_dunn_test(block_design, alpha=0.05)
    ```
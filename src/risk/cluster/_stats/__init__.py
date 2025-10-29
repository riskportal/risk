"""
risk/cluster/_stats
~~~~~~~~~~~~~~~~~~~
"""

from .permutation import compute_permutation_test
from .tests import (
    compute_binom_test,
    compute_chi2_test,
    compute_hypergeom_test,
)

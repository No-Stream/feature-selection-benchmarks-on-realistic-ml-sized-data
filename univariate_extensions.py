"""
Additional skl-interface univariate filters.
"""


import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_X_y, safe_sqr, safe_mask
from sklearn.feature_selection._univariate_selection import f_oneway
from sklearn.utils.extmath import safe_sparse_dot, row_norms
from scipy.stats import rankdata


class IdentityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
   
    def fit(self, input_array, y=None):
        return self
   
    def transform(self, input_array, y=None):
        return input_array
    

def f_rank_classif(X, y):
    """
    Same idea as f_classif, but computed on ranks
    """
    X, y = check_X_y(X, y, accept_sparse=["csr", "csc", "coo"])
    assert(isinstance(X, np.ndarray))
    X = np.array([rankdata(feat_vector) for feat_vector in X])
    args = [X[safe_mask(X, y == k)] for k in np.unique(y)]
    return f_oneway(*args)

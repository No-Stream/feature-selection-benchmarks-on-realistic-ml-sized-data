"""
Extract features from tree-based model SHAP values, using LinkedIn's optimized FastTreeSHAP.
"""


from sklearn.base import BaseEstimator, TransformerMixin
import fasttreeshap as ftshap
import numpy as np
import pandas as pd


class TreeShapImportances(BaseEstimator, TransformerMixin):
    """
    SKL-style transformer to select from SHAP feature importances.
    Note: unlike SKL, we will accept and return pd.DataFrame,
          but we'll need to retain np compat for sklearn interface.
    """
    def __init__(self, model, subsample=50_000, n_jobs=1, random_state=42):
        self.model = model
        self.subsample = subsample
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.feature_names_in_ = None
        self._n_cols = None
        self._explainer = None
        self.feature_importances_ = None
   
    def fit(self, X, y=None, **kwargs):
        assert(X.shape[0] == y.shape[0])
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        #self.feature_names_in_ = X.columns.tolist()
        self.n_cols = X.shape[1]
        if X.shape[0] > self.subsample:
            #print('debug: downsample')
            X = X.copy().sample(self.subsample, random_state=self.random_state)
            y = y.copy().sample(self.subsample, random_state=self.random_state)
        self.model = self.model.fit(X, y)
        # self._explainer = shap.Explainer(
        #     self.model, 
        #     feature_names=self.feature_names_in_, 
        #     algorithm='auto', 
        #     n_jobs=self.n_jobs,
        # )
        self._explainer = ftshap.Explainer(
            self.model,
        )
        shap_values = self._explainer(X, interactions=False)
        raw_importances = np.mean(np.abs(shap_values.values), axis=0)
        # Normalize [0,1]
        self.feature_importances_ = raw_importances / np.sum(raw_importances)
        return self
   
    def transform(self, input_array, y=None):
        warnings.warn("SelectFromTreeShap is intended to be used with SelectFromModel, not by itself.")
        return input_array
    
    def predict(self, X, y=None):
        warnings.warn("SelectFromTreeShap is not intended to predict.")
        return self.model.predict(X)

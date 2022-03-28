"""
Models etc. for benchmarking
"""


import lightgbm as lgb 
import xgboost as xgb
from sklearn.feature_selection import (
    SelectFdr, SelectFromModel, SelectKBest, SelectPercentile,
    chi2, f_classif, mutual_info_classif, RFECV,
    SequentialFeatureSelector, SelectFdr, VarianceThreshold
)
from sklearn.feature_selection._univariate_selection import *
from sklearn.decomposition import PCA, KernelPCA, SparsePCA
from sklearn.cluster import FeatureAgglomeration
from sklearn.model_selection import cross_val_score, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.base import BaseEstimator, TransformerMixin

from skrebate import ReliefF, MultiSURF, SURF

from eli5.sklearn import PermutationImportance

from category_encoders import OrdinalEncoder, TargetEncoder

from univariate_extensions import IdentityTransformer


SEED = 42
INNER_CV_NJOBS = 1

xgbc = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric='logloss',
    max_depth=6,
    learning_rate=0.05,
    n_estimators=400,
    subsample=.9,
    colsample_bytree=.8,
    colsample_bylevel=.8,
    colsample_bynode=.8,
    reg_alpha=1.0,             # add a small amount of L1 to simplify
    reg_lambda=3.0,
    gamma=1e-1,
    min_child_weight=10,            
    tree_method= 'hist',       # fast with less overfit
    grow_policy= 'depthwise',  # less overfit w/ hist vs. lossguide
    # fun fact: I have a 16-core cpu; if we spawn n_cv * n_xgb jobs, we get at least 10x16 jobs, which chokes the CPU; 
    #           2*16 works much better (= threads of my 16-core CPU)
    n_jobs=INNER_CV_NJOBS,                  
    use_label_encoder=False,
    random_state=SEED,
    importance_type='gain',
)

xgbc_sml = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric='logloss',
    max_depth=4,
    learning_rate=0.1,
    n_estimators=100,
    subsample=.9,
    colsample_bytree=.8,
    colsample_bylevel=.8,
    colsample_bynode=.8,
    reg_alpha=1.0,             # add a small amount of L1 to simplify
    reg_lambda=3.0,
    gamma=1e-1,
    min_child_weight=20,            
    tree_method= 'hist',       # fast with less overfit
    grow_policy= 'depthwise',  # less overfit w/ hist vs. lossguide
    n_jobs=INNER_CV_NJOBS,
    use_label_encoder=False,
    random_state=SEED + 1,
    importance_type='gain',
)

xgbc_sml_split = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric='logloss',
    max_depth=4,
    learning_rate=0.1,
    n_estimators=100,
    subsample=.9,
    colsample_bytree=.8,
    colsample_bylevel=.8,
    colsample_bynode=.8,
    reg_alpha=1.0,             # add a small amount of L1 to simplify
    reg_lambda=3.0,
    gamma=1e-1,
    min_child_weight=20,            
    tree_method= 'hist',       # fast with less overfit
    grow_policy= 'depthwise',  # less overfit w/ hist vs. lossguide
    n_jobs=INNER_CV_NJOBS,
    use_label_encoder=False,
    random_state=SEED + 1,
    importance_type='weight',
)

lgbc = lgb.LGBMClassifier(
    boosting_type='gbdt', 
    learning_rate=0.05, 
    num_leaves=31, 
    max_depth=-1, 
    n_estimators=400, 
    min_child_samples=20, 
    subsample=0.9, 
    subsample_freq=1, 
    colsample_bytree=0.8, 
    reg_alpha=1.0, 
    reg_lambda=3.0, 
    random_state=SEED, 
    n_jobs=INNER_CV_NJOBS, 
    verbose=-1,
    importance_type='gain',
)
lgbc_sml = lgb.LGBMClassifier(
    boosting_type='gbdt', 
    learning_rate=0.1,
    num_leaves=31, 
    max_depth=-1, 
    n_estimators=100, 
    min_child_samples=20, 
    subsample=0.9, 
    subsample_freq=1, 
    colsample_bytree=0.8, 
    reg_alpha=1.0, 
    reg_lambda=3.0, 
    random_state=SEED, 
    n_jobs=INNER_CV_NJOBS, 
    verbose=-1,
    importance_type='gain',
)
lgbc_tiny = lgb.LGBMClassifier(
    boosting_type='gbdt', 
    num_leaves=31, 
    max_depth=4, 
    learning_rate=0.2, 
    n_estimators=30, 
    min_child_samples=20, 
    subsample=0.9, 
    subsample_freq=1, 
    colsample_bytree=0.8, 
    reg_alpha=0.5, 
    reg_lambda=1.0, 
    random_state=SEED, 
    n_jobs=INNER_CV_NJOBS, 
    verbose=-1,
    importance_type='gain',
)

mean_imputer = SimpleImputer(strategy='mean', add_indicator=True)
robust_scaler = RobustScaler(with_centering=True, with_scaling=True)
qtile_scaler = QuantileTransformer(
    n_quantiles=1000, output_distribution='normal', subsample=20_000, random_state=SEED,
)
target_encoder = TargetEncoder(drop_invariant=True, min_samples_leaf=30, smoothing=1.0)
remove_constant = VarianceThreshold(threshold=0.0)
cv_small = StratifiedKFold(5, shuffle=False)
cv_tiny = StratifiedKFold(3, shuffle=False)
id_transf = IdentityTransformer()

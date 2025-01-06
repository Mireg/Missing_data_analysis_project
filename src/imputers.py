import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

class ImputationMethods:
    def __init__(self, random_state=42):
        self.random_state = random_state
        
    def get_imputers(self):
        return {
            'simple_mean': self._create_simple_imputer('mean'),
            'simple_median': self._create_simple_imputer('median'),
            'mice_rf': self._create_mice_imputer(RandomForestRegressor(n_estimators=100, random_state=self.random_state)),
            'mice_xgb': self._create_mice_imputer(xgb.XGBRegressor(random_state=self.random_state))
        }
    
    def _create_simple_imputer(self, strategy):
        return {
            'numeric': SimpleImputer(strategy=strategy),
            'categorical': SimpleImputer(strategy='most_frequent')
        }
    
    def _create_mice_imputer(self, estimator):
        return {
            'numeric': IterativeImputer(
                estimator=estimator,
                random_state=self.random_state,
                max_iter=10
            ),
            'categorical': SimpleImputer(strategy='most_frequent')
        }

# src/models.py
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from lightgbm import LGBMRegressor

def get_models(random_state=42):
    return {
        'random_forest': RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1),
        'xgboost': xgb.XGBRegressor(random_state=random_state, n_jobs=-1),
        'lightgbm': LGBMRegressor(random_state=random_state, n_jobs=-1)
    }
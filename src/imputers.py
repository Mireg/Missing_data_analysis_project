import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import xgboost as xgb
from lightgbm import LGBMRegressor

class ImputationMethods:
    def __init__(self, random_state=42):
        self.random_state = random_state
        
    def get_imputers(self):
        return {
            'advanced_iterative': self._create_advanced_iterative_imputer(),
            'knn': self._create_knn_imputer(),
            'ensemble': self._create_ensemble_imputer(),
            'lightgbm': self._create_lightgbm_imputer(),
            'hybrid': self._create_hybrid_imputer()
        }
    
    def _create_advanced_iterative_imputer(self):
        return {
            'numeric': IterativeImputer(
                estimator=RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    random_state=self.random_state
                ),
                initial_strategy='median',
                max_iter=20,
                random_state=self.random_state,
                skip_complete=True
            ),
            'categorical': SimpleImputer(
                strategy='most_frequent',
                add_indicator=True
            )
        }
    
    def _create_knn_imputer(self):
        return {
            'numeric': KNNImputer(
                n_neighbors=5,
                weights='distance'
            ),
            'categorical': SimpleImputer(
                strategy='most_frequent',
                add_indicator=True
            )
        }
    
    def _create_ensemble_imputer(self):
        return {
            'numeric': IterativeImputer(
                estimator=ExtraTreesRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    random_state=self.random_state
                ),
                initial_strategy='median',
                max_iter=20,
                random_state=self.random_state,
                skip_complete=True
            ),
            'categorical': SimpleImputer(
                strategy='most_frequent',
                add_indicator=True
            )
        }
    
    def _create_lightgbm_imputer(self):
        return {
            'numeric': IterativeImputer(
                estimator=LGBMRegressor(
                    n_estimators=100,
                    num_leaves=31,
                    random_state=self.random_state
                ),
                initial_strategy='median',
                max_iter=20,
                random_state=self.random_state,
                skip_complete=True
            ),
            'categorical': SimpleImputer(
                strategy='most_frequent',
                add_indicator=True
            )
        }
    
    def _create_hybrid_imputer(self):
        """Combines different imputation strategies based on missing percentage"""
        return {
            'numeric': self._create_hybrid_numeric_imputer(),
            'categorical': SimpleImputer(
                strategy='most_frequent',
                add_indicator=True
            )
        }
    
    def _create_hybrid_numeric_imputer(self):
        class HybridImputer:
            def __init__(self, random_state=42):
                self.random_state = random_state
                self.knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
                self.iterative_imputer = IterativeImputer(
                    estimator=RandomForestRegressor(
                        n_estimators=100,
                        random_state=random_state
                    ),
                    random_state=random_state,
                    max_iter=20
                )
                self.simple_imputer = SimpleImputer(strategy='median')
                
            def fit_transform(self, X):
                # Calculate missing percentages
                missing_pct = X.isnull().mean()
                
                # Create mask for different strategies
                high_missing = missing_pct > 0.3
                medium_missing = (missing_pct > 0.1) & (missing_pct <= 0.3)
                low_missing = missing_pct <= 0.1
                
                # Apply different strategies based on missing percentage
                X_imputed = X.copy()
                
                # For columns with high missing values, use simple imputation
                high_missing_cols = X.columns[high_missing]
                if len(high_missing_cols) > 0:
                    X_imputed[high_missing_cols] = self.simple_imputer.fit_transform(X[high_missing_cols])
                
                # For columns with medium missing values, use KNN
                medium_missing_cols = X.columns[medium_missing]
                if len(medium_missing_cols) > 0:
                    X_imputed[medium_missing_cols] = self.knn_imputer.fit_transform(X[medium_missing_cols])
                
                # For columns with low missing values, use iterative imputation
                low_missing_cols = X.columns[low_missing]
                if len(low_missing_cols) > 0:
                    X_imputed[low_missing_cols] = self.iterative_imputer.fit_transform(X[low_missing_cols])
                
                return X_imputed
            
            def transform(self, X):
                # Similar to fit_transform but without fitting
                return self.fit_transform(X)  # For simplicity, we're using fit_transform
        
        return HybridImputer(random_state=self.random_state)
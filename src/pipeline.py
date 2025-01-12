import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
import time
from sklearn.model_selection import RandomizedSearchCV
from src.optimization import tune_xgboost, tune_lightgbm

class ModelPipeline:
    def __init__(self, imputation_methods, prediction_models, random_state=42):
        self.random_state = random_state
        self.imputation_methods = imputation_methods
        self.prediction_models = prediction_models
        self.results = {}
        
    def impute_data(self, X, numeric_features, categorical_features, imputer_dict, fit_imputer=True):
        """Impute data using specified imputers"""
        # Get encoded categorical columns (quarter_*)
        encoded_cols = [col for col in X.columns if col.startswith('quarter_')]
        
        # Handle numeric features
        X_numeric = X[numeric_features].copy() if len(numeric_features) > 0 else pd.DataFrame()
        
        # Handle any remaining categorical features (not encoded yet)
        X_categorical = X[categorical_features].copy() if len(categorical_features) > 0 else pd.DataFrame()
        
        # Impute numeric features
        if len(numeric_features) > 0:
            if fit_imputer:
                X_numeric = pd.DataFrame(
                    imputer_dict['numeric'].fit_transform(X_numeric),
                    columns=numeric_features,
                    index=X.index
                )
            else:
                X_numeric = pd.DataFrame(
                    imputer_dict['numeric'].transform(X_numeric),
                    columns=numeric_features,
                    index=X.index
                )
                    
        # Impute categorical features (if any not yet encoded)
        if len(categorical_features) > 0:
            if fit_imputer:
                X_categorical = pd.DataFrame(
                    imputer_dict['categorical'].fit_transform(X_categorical),
                    columns=categorical_features,
                    index=X.index
                )
            else:
                X_categorical = pd.DataFrame(
                    imputer_dict['categorical'].transform(X_categorical),
                    columns=categorical_features,
                    index=X.index
                )
        
        # Add already encoded categorical columns (quarter_*) without imputation
        if encoded_cols:
            X_encoded = X[encoded_cols].copy()
            # Combine all parts: numeric features, categorical features, and encoded features
            return pd.concat([X_numeric, X_categorical, X_encoded], axis=1)
        else:
            # Return just numeric and categorical if no encoded columns
            return pd.concat([X_numeric, X_categorical], axis=1)
        
    def run_pipeline(self, X_train, y_train, X_test, numeric_features, categorical_features):
        print("\nStarting processing...")
        results = {}
        
        for imp_name, imputer in self.imputation_methods.items():
            print(f"\nProcessing with {imp_name} imputation...")
            
            # Impute training and test data
            X_train_imputed = self.impute_data(X_train, numeric_features, categorical_features, imputer, True)
            X_test_imputed = self.impute_data(X_test, numeric_features, categorical_features, imputer, False)
            
            print("Tuning XGBoost...")
            xgb_model, xgb_params = tune_xgboost(X_train_imputed, y_train)
            print("Best XGBoost params:", xgb_params)
            
            # print("Tuning LightGBM...")
            # lgb_model, lgb_params = tune_lightgbm(X_train_imputed, y_train)
            # print("Best LightGBM params:", lgb_params)
            
            # Update models with tuned versions
            self.prediction_models.update({
                'xgboost_tuned': xgb_model
                #,'lightgbm_tuned': lgb_model
            })

            for model_name, model in self.prediction_models.items():
                key = f"{imp_name}_{model_name}"
                print(f"Training {key}...")
                
                # Create pipeline
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', model)
                ])
                
                # Do KFold CV first
                cv_scores = cross_val_score(pipeline, X_train_imputed, y_train, 
                                        cv=5, scoring='neg_root_mean_squared_error')
                
                # After CV, fit on full training data
                pipeline.fit(X_train_imputed, y_train)
                
                # Make predictions on test set
                test_predictions = pipeline.predict(X_test_imputed)
                
                # Store both CV results and test predictions
                results[key] = {
                    'cv_scores': -cv_scores,  # Convert back to positive RMSE
                    'cv_rmse_mean': -cv_scores.mean(),
                    'cv_rmse_std': cv_scores.std(),
                    'test_predictions': test_predictions,
                    'model': pipeline
                }
                
                print(f"Completed {key} - CV RMSE: {-cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        return results
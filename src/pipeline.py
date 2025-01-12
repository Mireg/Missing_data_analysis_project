import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
import time

class ModelPipeline:
    def __init__(self, imputation_methods, prediction_models, random_state=42):
        self.random_state = random_state
        self.imputation_methods = imputation_methods
        self.prediction_models = prediction_models
        self.results = {}
        
    def impute_data(self, X, numeric_features, categorical_features, imputer_dict, fit_imputer=True):
        """Impute data using specified imputers"""
        X_numeric = X[numeric_features].copy()
        X_categorical = X[categorical_features].copy()
        
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
                
        return pd.concat([X_numeric, X_categorical], axis=1)
    
    def run_pipeline(self, X_train, y_train, X_test, numeric_features, categorical_features):
        """Run the complete pipeline with all approaches and enhanced cross-validation"""
        print("\nStarting processing with multiple approaches...")
        
        # Define KFold cross-validator
        kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        for imp_name, imputer in self.imputation_methods.items():
            print(f"\nProcessing with {imp_name} imputation...")
            start_time = time.time()
            
            # Impute data
            X_train_imputed = self.impute_data(X_train, numeric_features, categorical_features, imputer, True)
            X_test_imputed = self.impute_data(X_test, numeric_features, categorical_features, imputer, False)
            
            for model_name, model in self.prediction_models.items():
                print(f"Training {model_name}...")
                model_start_time = time.time()
                
                # Initialize lists to store fold results
                fold_scores = []
                fold_predictions = []
                
                # Perform k-fold cross-validation
                for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train_imputed)):
                    # Split data for this fold
                    X_fold_train = X_train_imputed.iloc[train_idx]
                    y_fold_train = y_train.iloc[train_idx]
                    X_fold_val = X_train_imputed.iloc[val_idx]
                    y_fold_val = y_train.iloc[val_idx]
                    
                    # Create and train pipeline
                    pipeline = Pipeline([
                        ('scaler', StandardScaler()),
                        ('model', model)
                    ])
                    
                    # Train on this fold
                    pipeline.fit(X_fold_train, y_fold_train)
                    
                    # Predict on validation fold
                    fold_pred = pipeline.predict(X_fold_val)
                    fold_rmse = np.sqrt(mean_squared_error(y_fold_val, fold_pred))
                    fold_scores.append(fold_rmse)
                    
                    # Store predictions
                    fold_predictions.append(pipeline.predict(X_test_imputed))
                    
                    print(f"Fold {fold_idx + 1} RMSE: {fold_rmse:.2f}")
                
                # Calculate average predictions across folds
                final_predictions = np.mean(fold_predictions, axis=0)
                
                # Store results
                approach_name = f"{imp_name}_{model_name}"
                self.results[approach_name] = {
                    'cv_rmse_mean': np.mean(fold_scores),
                    'cv_rmse_std': np.std(fold_scores),
                    'predictions': final_predictions,
                    'pipeline': pipeline,  # Store the last pipeline
                    'training_time': time.time() - model_start_time,
                    'fold_scores': fold_scores  # Store individual fold scores
                }
                
                print(f"Finished {model_name} - CV RMSE: {np.mean(fold_scores):.2f} (±{np.std(fold_scores):.2f})")
            
            print(f"Total time for {imp_name}: {time.time() - start_time:.2f} seconds")
            
        return self.results
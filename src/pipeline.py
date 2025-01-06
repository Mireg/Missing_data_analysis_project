import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
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
        """Run the complete pipeline with all approaches"""
        print("\nStarting processing with multiple approaches...")
        
        for imp_name, imputer in self.imputation_methods.items():
            print(f"\nProcessing with {imp_name} imputation...")
            start_time = time.time()
            
            # Impute data
            X_train_imputed = self.impute_data(X_train, numeric_features, categorical_features, imputer, True)
            X_test_imputed = self.impute_data(X_test, numeric_features, categorical_features, imputer, False)
            
            for model_name, model in self.prediction_models.items():
                print(f"Training {model_name}...")
                model_start_time = time.time()
                
                # Create and train pipeline
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', model)
                ])
                
                # Perform cross-validation
                cv_scores = cross_val_score(pipeline, X_train_imputed, y_train, cv=5, scoring='neg_mean_squared_error')
                rmse_scores = np.sqrt(-cv_scores)
                
                # Train final model and predict
                pipeline.fit(X_train_imputed, y_train)
                predictions = pipeline.predict(X_test_imputed)
                
                # Store results
                approach_name = f"{imp_name}_{model_name}"
                self.results[approach_name] = {
                    'cv_rmse_mean': rmse_scores.mean(),
                    'cv_rmse_std': rmse_scores.std(),
                    'predictions': predictions,
                    'pipeline': pipeline,
                    'training_time': time.time() - model_start_time
                }
                
                print(f"Finished {model_name} - CV RMSE: {rmse_scores.mean():.2f} (±{rmse_scores.std():.2f})")
            
            print(f"Total time for {imp_name}: {time.time() - start_time:.2f} seconds")
            
        return self.results
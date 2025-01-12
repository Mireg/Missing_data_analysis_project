import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
import time

def quick_evaluate_pipeline():
    # Load processed train and test data
    train_df = pd.read_csv('data/pzn-rent-train-processed.csv')
    test_df = pd.read_csv('data/pzn-rent-test-processed.csv')
    
    # Separate features and target
    X_train = train_df.drop('price', axis=1)
    y_train = train_df['price']
    X_test = test_df.drop('price', axis=1) if 'price' in test_df.columns else test_df
    y_test = test_df['price'] if 'price' in test_df.columns else None
    
    # Identify numeric and categorical columns
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = ['quarter']  # Specify the categorical column
    
    # Define simple imputation methods
    imputation_methods = {
        'mean': SimpleImputer(strategy='mean'),
        'median': SimpleImputer(strategy='median'),
        'most_frequent': SimpleImputer(strategy='most_frequent')
    }
    
    # Define models with simple configurations
    models = {
        'random_forest': RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ),
        'xgboost': xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        ),
        'lightgbm': LGBMRegressor(
            n_estimators=100,
            num_leaves=31,
            random_state=42,
            n_jobs=-1
        )
    }
    
    # Results storage
    results = []
    predictions = {}
    
    # Test each combination
    for imp_name, imputer in imputation_methods.items():
        for model_name, model in models.items():
            print(f"\nTesting {imp_name} imputation with {model_name}")
            start_time = time.time()
            
            # Create preprocessing steps
            numeric_transformer = Pipeline([
                ('imputer', imputer),
                ('scaler', StandardScaler())
            ])
            
            categorical_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(drop='first', sparse_output=False))
            ])
            
            # Combine preprocessing steps
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ])
            
            # Create full pipeline
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)
            ])
            
            # Fit and predict
            pipeline.fit(X_train, y_train)
            train_pred = pipeline.predict(X_train)
            test_pred = pipeline.predict(X_test)
            
            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            train_r2 = r2_score(y_train, train_pred)
            
            # Store results
            result = {
                'imputation': imp_name,
                'model': model_name,
                'train_rmse': train_rmse,
                'train_r2': train_r2,
                'time': time.time() - start_time
            }
            
            # Add test metrics if available
            if y_test is not None:
                test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
                test_r2 = r2_score(y_test, test_pred)
                result.update({
                    'test_rmse': test_rmse,
                    'test_r2': test_r2
                })
            
            results.append(result)
            predictions[f"{imp_name}_{model_name}"] = test_pred
            
            print(f"Train RMSE: {train_rmse:.2f}")
            print(f"Train R²: {train_r2:.4f}")
            if y_test is not None:
                print(f"Test RMSE: {test_rmse:.2f}")
                print(f"Test R²: {test_r2:.4f}")
            print(f"Time taken: {result['time']:.2f} seconds")
    
    # Create results DataFrame and sort by performance
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('train_rmse' if y_test is None else 'test_rmse')
    
    print("\nFinal Results:")
    print(results_df.to_string(index=False))
    
    # Save results and predictions
    results_df.to_csv('predictions/quick_eval/quick_evaluation_results.csv', index=False)
    
    # Save predictions for each approach
    for name, preds in predictions.items():
        pd.DataFrame({
            'ID': range(1, len(preds) + 1),
            'predicted_price': preds
        }).to_csv(f'predictions_{name}.csv', index=False)
    
    return results_df, predictions

results, predictions = quick_evaluate_pipeline()
"""
Standalone script with complete examples for price prediction.
Run this to see all models in action.

Usage:
    python examples_price_prediction.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.data.data_load import load_data
from src.strategy.predictive import (
    prepare_data_for_modeling,
    TimeSeriesSplitter,
    ModelTrainer,
    FeatureSelector,
    # Models
    LinearModel,
    RandomForestModel,
    GradientBoostingModel,
    MLPModel,
)


def example_1_basic_tree_model():
    """Example 1: Train a basic Random Forest model."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Tree Model (Random Forest)")
    print("="*80)
    
    try:
        # Load data
        print("\n1. Loading data...")
        df = load_data(
            symbols=['BTCUSDT'],
            start_date="2025-06-01",
            end_date="2025-12-31",
            interval="1m"
        )
        print(f"   Loaded {len(df)} samples")
        
        # Prepare features
        print("\n2. Engineering features...")
        df_features, feature_cols, target_col = prepare_data_for_modeling(
            df,
            horizon=1,
            lookback_window=60,
            dropna=True
        )
        print(f"   Generated {len(feature_cols)} features from {df_features.shape[0]} samples")
        
        # Prepare data
        print("\n3. Splitting data...")
        X = df_features[feature_cols].values
        y = df_features[target_col].values
        X_train, X_val, X_test, y_train, y_val, y_test = \
            TimeSeriesSplitter.train_val_test_split(X, y)
        
        # Train model
        print("\n4. Training Random Forest...")
        trainer = ModelTrainer()
        rf_model = RandomForestModel(n_estimators=50, max_depth=10)
        trainer.train_model(rf_model, X_train, y_train, X_val, y_val)
        
        # Evaluate
        print("\n5. Evaluating on test set...")
        trainer.evaluate_model(rf_model, X_test, y_test)
        
        # Feature importance
        print("\n6. Top 10 important features:")
        rf_model.feature_cols = feature_cols
        importance_df = rf_model.feature_importance()
        print(importance_df.head(10).to_string(index=False))
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def example_2_multiple_models():
    """Example 2: Train and compare multiple models."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Multiple Models Comparison")
    print("="*80)
    
    try:
        # Load and prepare data (using smaller dataset for speed)
        print("\n1. Loading and preparing data...")
        df = load_data(
            symbols=['BTCUSDT'],
            start_date="2025-09-01",
            end_date="2025-10-31",
            interval="1m"
        )
        df_features, feature_cols, target_col = prepare_data_for_modeling(
            df, horizon=1, lookback_window=60, dropna=True
        )
        
        X = df_features[feature_cols].values
        y = df_features[target_col].values
        X_train, X_val, X_test, y_train, y_val, y_test = \
            TimeSeriesSplitter.train_val_test_split(X, y)
        
        print(f"   Data ready: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test")
        
        # Initialize trainer
        trainer = ModelTrainer()
        
        # Train models
        print("\n2. Training models...")
        
        models = [
            ("Linear", LinearModel()),
            ("RandomForest", RandomForestModel(n_estimators=50, max_depth=10)),
            ("GradientBoosting", GradientBoostingModel(n_estimators=50, max_depth=5)),
        ]
        
        for name, model in models:
            print(f"\n   Training {name}...")
            trainer.train_model(model, X_train, y_train, X_val, y_val)
        
        # Evaluate all models
        print("\n3. Evaluating models on test set...")
        for name, model in models:
            print(f"\n   Evaluating {name}...")
            trainer.evaluate_model(model, X_test, y_test)
        
        # Compare
        print("\n4. Model Comparison:")
        comparison = trainer.compare_models()
        
        # Save results
        results_path = Path('/Users/qinli/Documents/GitHub/QuantTrading/results')
        results_path.mkdir(exist_ok=True)
        trainer.save_results(results_path / 'example_comparison.json')
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def example_3_mlp_model():
    """Example 3: Train a neural network (MLP) model."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Neural Network (MLP) Model")
    print("="*80)
    
    try:
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF verbose output
        
        # Load and prepare data
        print("\n1. Loading and preparing data...")
        df = load_data(
            symbols=['BTCUSDT'],
            start_date="2025-09-01",
            end_date="2025-10-31",
            interval="1m"
        )
        df_features, feature_cols, target_col = prepare_data_for_modeling(
            df, horizon=1, lookback_window=60, dropna=True
        )
        
        X = df_features[feature_cols].values
        y = df_features[target_col].values
        X_train, X_val, X_test, y_train, y_val, y_test = \
            TimeSeriesSplitter.train_val_test_split(X, y)
        
        print(f"   Data ready: {X.shape}")
        
        # Train MLP
        print("\n2. Training MLP Neural Network...")
        trainer = ModelTrainer()
        mlp_model = MLPModel(hidden_layers=[64, 32, 16], epochs=10, batch_size=32)
        trainer.train_model(mlp_model, X_train, y_train, X_val, y_val, verbose=0)
        
        # Evaluate
        print("\n3. Evaluating on test set...")
        trainer.evaluate_model(mlp_model, X_test, y_test)
        
    except ImportError:
        print("TensorFlow not installed. Install with: pip install tensorflow")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def example_4_cross_validation():
    """Example 4: Cross-validation with multiple folds."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Time Series Cross-Validation")
    print("="*80)
    
    try:
        # Load data
        print("\n1. Loading data...")
        df = load_data(
            symbols=['BTCUSDT'],
            start_date="2025-08-01",
            end_date="2025-10-31",
            interval="1m"
        )
        df_features, feature_cols, target_col = prepare_data_for_modeling(
            df, horizon=1, lookback_window=60, dropna=True
        )
        
        X = df_features[feature_cols].values
        y = df_features[target_col].values
        
        print(f"   Loaded {len(X)} samples")
        
        # Manual time series cross-validation
        print("\n2. Running 3-fold time series cross-validation...")
        
        total_len = len(X)
        fold_size = total_len // 3
        
        fold_results = []
        
        for fold in range(3):
            print(f"\n   Fold {fold + 1}/3:")
            
            # Split: all previous data for train, current fold for test
            test_start = fold * fold_size
            test_end = (fold + 1) * fold_size if fold < 2 else total_len
            
            X_train_fold = X[:test_start]
            y_train_fold = y[:test_start]
            X_test_fold = X[test_start:test_end]
            y_test_fold = y[test_start:test_end]
            
            if len(X_train_fold) == 0:
                continue
            
            # Train model
            model = RandomForestModel(n_estimators=30, max_depth=10)
            model.train(X_train_fold, y_train_fold)
            
            # Evaluate
            y_pred = model.predict(X_test_fold)
            accuracy = (y_pred == y_test_fold).mean()
            
            print(f"      Samples: {len(X_train_fold)} train, {len(X_test_fold)} test")
            print(f"      Accuracy: {accuracy:.4f}")
            
            fold_results.append(accuracy)
        
        print(f"\n   Cross-validation results:")
        print(f"   Mean Accuracy: {np.mean(fold_results):.4f} (+/- {np.std(fold_results):.4f})")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def example_5_feature_analysis():
    """Example 5: Analyze and visualize feature importance."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Feature Importance Analysis")
    print("="*80)
    
    try:
        # Load and prepare data
        print("\n1. Loading and preparing data...")
        df = load_data(
            symbols=['BTCUSDT'],
            start_date="2025-09-01",
            end_date="2025-10-31",
            interval="1m"
        )
        df_features, feature_cols, target_col = prepare_data_for_modeling(
            df, horizon=1, lookback_window=60, dropna=True
        )
        
        X = df_features[feature_cols].values
        y = df_features[target_col].values
        X_train, X_val, X_test, y_train, y_val, y_test = \
            TimeSeriesSplitter.train_val_test_split(X, y)
        
        # Train Random Forest
        print("\n2. Training Random Forest...")
        rf_model = RandomForestModel(n_estimators=50, max_depth=10)
        rf_model.train(X_train, y_train)
        
        # Get feature importance
        print("\n3. Computing feature importance...")
        rf_model.feature_cols = feature_cols
        importance_df = rf_model.feature_importance()
        
        # Display top features
        print("\n4. Top 20 Important Features:")
        top_20 = importance_df.head(20)
        print(top_20.to_string(index=False))
        
        # Show feature categories
        print("\n5. Feature distribution by category:")
        feature_categories = {}
        for feature in feature_cols:
            # Categorize by name
            if 'sma' in feature:
                cat = 'SMA'
            elif 'ema' in feature:
                cat = 'EMA'
            elif 'rsi' in feature:
                cat = 'RSI'
            elif 'macd' in feature:
                cat = 'MACD'
            elif 'bb_' in feature:
                cat = 'BBands'
            elif 'atr' in feature:
                cat = 'ATR'
            elif 'volume' in feature:
                cat = 'Volume'
            elif 'volatility' in feature:
                cat = 'Volatility'
            elif 'momentum' in feature:
                cat = 'Momentum'
            elif 'lag' in feature:
                cat = 'Lagged'
            else:
                cat = 'Other'
            
            if cat not in feature_categories:
                feature_categories[cat] = 0
            feature_categories[cat] += 1
        
        for cat, count in sorted(feature_categories.items(), key=lambda x: x[1], reverse=True):
            print(f"   {cat:12s}: {count:3d} features")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("PRICE PREDICTION EXAMPLES")
    print("="*80)
    
    examples = [
        ("Example 1: Basic Tree Model", example_1_basic_tree_model),
        ("Example 2: Multiple Models", example_2_multiple_models),
        ("Example 3: Neural Network", example_3_mlp_model),
        ("Example 4: Cross-Validation", example_4_cross_validation),
        ("Example 5: Feature Analysis", example_5_feature_analysis),
    ]
    
    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\nRunning all examples...\n")
    
    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\n✗ Error in {name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("EXAMPLES COMPLETED")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()


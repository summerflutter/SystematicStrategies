"""
Training utilities for price prediction models.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.model_selection import train_test_split
import json


class ModelTrainer:
    """Utility class for training and comparing multiple models."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
    
    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            X: Feature array
            y: Target array
            test_size: Proportion of test set
            val_size: Proportion of validation set (from remaining data)
        
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # Second split: train and val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=self.random_state
        )
        
        print(f"Data split:")
        print(f"  Train: {len(X_train)} samples")
        print(f"  Val:   {len(X_val)} samples")
        print(f"  Test:  {len(X_test)} samples")
        print(f"  Total: {len(X)} samples\n")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_model(
        self,
        model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ):
        """Train a single model."""
        print(f"Training {model.name} model...")
        
        if X_val is not None and y_val is not None:
            model.train(X_train, y_train, X_val=X_val, y_val=y_val, **kwargs)
        else:
            model.train(X_train, y_train, **kwargs)
        
        self.models[model.name] = model
        print(f"{model.name} model trained successfully!\n")
    
    def evaluate_model(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict:
        """Evaluate a model and return metrics."""
        from src.strategy.predictive.models import ModelEvaluator
        
        print(f"Evaluating {model.name} model...")
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if model.predict_proba(X_test) is not None else None
        
        metrics = ModelEvaluator.print_results(y_test, y_pred, y_proba, model.name)
        
        self.results[model.name] = {
            'metrics': metrics,
            'y_pred': y_pred,
            'y_proba': y_proba
        }
        
        return metrics
    
    def compare_models(self) -> pd.DataFrame:
        """Compare all trained models."""
        comparison_data = []
        
        for model_name, result in self.results.items():
            metrics = result['metrics']
            comparison_data.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1'],
                'AUC': metrics.get('auc', np.nan)
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison = df_comparison.sort_values('F1-Score', ascending=False)
        
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        print(df_comparison.to_string(index=False))
        print("="*80 + "\n")
        
        return df_comparison
    
    def save_results(self, filepath: str):
        """Save results to JSON file."""
        results_to_save = {}
        
        for model_name, result in self.results.items():
            results_to_save[model_name] = {
                'accuracy': float(result['metrics']['accuracy']),
                'precision': float(result['metrics']['precision']),
                'recall': float(result['metrics']['recall']),
                'f1': float(result['metrics']['f1']),
                'auc': float(result['metrics'].get('auc', np.nan))
            }
        
        with open(filepath, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        print(f"Results saved to {filepath}")


class TimeSeriesSplitter:
    """Custom time series train/val/test splitter."""
    
    @staticmethod
    def train_val_test_split(
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split time series data into train/val/test without shuffling (preserves temporal order).
        
        Args:
            X: Feature array
            y: Target array
            train_ratio: Proportion of training data
            val_ratio: Proportion of validation data
            test_ratio: Proportion of test data
        
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        total_samples = len(X)
        
        train_end = int(total_samples * train_ratio)
        val_end = train_end + int(total_samples * val_ratio)
        
        X_train = X[:train_end]
        y_train = y[:train_end]
        
        X_val = X[train_end:val_end]
        y_val = y[train_end:val_end]
        
        X_test = X[val_end:]
        y_test = y[val_end:]
        
        print(f"Time series split:")
        print(f"  Train: {len(X_train)} samples ({train_ratio*100:.1f}%)")
        print(f"  Val:   {len(X_val)} samples ({val_ratio*100:.1f}%)")
        print(f"  Test:  {len(X_test)} samples ({test_ratio*100:.1f}%)")
        print(f"  Total: {total_samples} samples\n")
        
        return X_train, X_val, X_test, y_train, y_val, y_test

    @staticmethod
    def train_test_split_by_date(
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        train_start_date: str,
        train_end_date: str,
        test_start_date: str,
        test_end_date: str,
        val_ratio: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split time series data by date ranges instead of ratios.

        Args:
            df: DataFrame with features and target
            feature_cols: List of feature column names
            target_col: Target column name
            train_start_date: Start date for training data (YYYY-MM-DD)
            train_end_date: End date for training data (YYYY-MM-DD)
            test_start_date: Start date for test data (YYYY-MM-DD)
            test_end_date: End date for test data (YYYY-MM-DD)
            val_ratio: Ratio of training data to use for validation

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Ensure we have a datetime index or open_time column
        if 'open_time' in df.columns:
            df = df.set_index('open_time')
        elif not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have 'open_time' column or DatetimeIndex")

        # Convert date strings to datetime (timezone-aware)
        train_start = pd.to_datetime(train_start_date).tz_localize('UTC')
        train_end = pd.to_datetime(train_end_date).tz_localize('UTC')
        test_start = pd.to_datetime(test_start_date).tz_localize('UTC')
        test_end = pd.to_datetime(test_end_date).tz_localize('UTC')

        # Filter data by date ranges
        train_mask = (df.index >= train_start) & (df.index <= train_end)
        test_mask = (df.index >= test_start) & (df.index <= test_end)

        df_train = df[train_mask]
        df_test = df[test_mask]

        if len(df_train) == 0:
            raise ValueError(f"No training data found between {train_start_date} and {train_end_date}")
        if len(df_test) == 0:
            raise ValueError(f"No test data found between {test_start_date} and {test_end_date}")

        # Split training data into train and validation
        train_samples = len(df_train)
        val_samples = int(train_samples * val_ratio)
        train_samples_final = train_samples - val_samples

        # Prepare arrays
        X_train_full = df_train[feature_cols].values
        y_train_full = df_train[target_col].values
        X_test = df_test[feature_cols].values
        y_test = df_test[target_col].values

        # Split training data
        X_train = X_train_full[:train_samples_final]
        y_train = y_train_full[:train_samples_final]
        X_val = X_train_full[train_samples_final:]
        y_val = y_train_full[train_samples_final:]

        print(f"Date-based time series split:")
        print(f"  Train: {len(X_train)} samples ({train_start_date} to {train_end_date}, {val_ratio*100:.1f}% for validation)")
        print(f"  Val:   {len(X_val)} samples (from training data)")
        print(f"  Test:  {len(X_test)} samples ({test_start_date} to {test_end_date})")
        print(f"  Total: {len(X_train) + len(X_val) + len(X_test)} samples\n")

        return X_train, X_val, X_test, y_train, y_val, y_test

    @staticmethod
    def train_test_split_by_period(
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        train_period_days: int,
        test_period_days: int,
        val_ratio: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split time series data by specifying training and test periods in days.

        Args:
            df: DataFrame with features and target
            feature_cols: List of feature column names
            target_col: Target column name
            train_period_days: Number of days for training data
            test_period_days: Number of days for test data
            val_ratio: Ratio of training data to use for validation

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Ensure we have a datetime index or open_time column
        if 'open_time' in df.columns:
            df = df.set_index('open_time')
        elif not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have 'open_time' column or DatetimeIndex")

        # Sort by date
        df = df.sort_index()

        # Find the latest date in the data
        end_date = df.index.max()

        # Calculate start dates
        test_start = end_date - pd.Timedelta(days=test_period_days)
        train_start = test_start - pd.Timedelta(days=train_period_days)

        print(f"Period-based split:")
        print(f"  Data end date: {end_date.strftime('%Y-%m-%d')}")
        print(f"  Train period: {train_period_days} days ({train_start.strftime('%Y-%m-%d')} to {test_start.strftime('%Y-%m-%d')})")
        print(f"  Test period:  {test_period_days} days ({test_start.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})")

        # Use the date-based splitter
        return TimeSeriesSplitter.train_test_split_by_date(
            df,
            feature_cols,
            target_col,
            train_start_date=train_start.strftime('%Y-%m-%d'),
            train_end_date=test_start.strftime('%Y-%m-%d'),
            test_start_date=test_start.strftime('%Y-%m-%d'),
            test_end_date=end_date.strftime('%Y-%m-%d'),
            val_ratio=val_ratio
        )


class FeatureSelector:
    """Feature selection utilities."""
    
    @staticmethod
    def select_top_features(
        importance_df: pd.DataFrame,
        top_n: int = 20
    ) -> List[str]:
        """
        Select top N features from importance dataframe.
        
        Args:
            importance_df: DataFrame with 'feature' and 'importance' columns
            top_n: Number of top features to select
        
        Returns:
            List of top feature names
        """
        top_features = importance_df.head(top_n)['feature'].tolist()
        print(f"Selected top {top_n} features:\n{top_features}\n")
        return top_features
    
    @staticmethod
    def get_feature_importance_plot(importance_df: pd.DataFrame, top_n: int = 20):
        """
        Create a feature importance plot.
        
        Args:
            importance_df: DataFrame with 'feature' and 'importance' columns
            top_n: Number of top features to display
        """
        try:
            import matplotlib.pyplot as plt
            
            top_df = importance_df.head(top_n)
            
            fig, ax = plt.subplots(figsize=(10, max(6, int(top_n * 0.3))))
            ax.barh(range(len(top_df)), top_df['importance'].values)
            ax.set_yticks(range(len(top_df)))
            ax.set_yticklabels(top_df['feature'].values)
            ax.set_xlabel('Importance')
            ax.set_title(f'Top {top_n} Feature Importance')
            ax.invert_yaxis()
            
            plt.tight_layout()
            return fig, ax
        
        except ImportError:
            print("matplotlib not available. Install with: pip install matplotlib")
            return None, None








"""
Price prediction models including linear, tree-based, and deep learning models.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
from abc import ABC, abstractmethod
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import torch.nn as nn
import torch


try:
    import joblib
except ImportError:
    # Fallback to pickle if joblib not available
    import pickle as joblib
    joblib.dump = lambda obj, path: pickle.dump(obj, open(path, 'wb'))
    joblib.load = lambda path: pickle.load(open(path, 'rb'))
import warnings
warnings.filterwarnings('ignore')


class BaseModel(ABC):
    """Base class for all prediction models."""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.is_trained = False
    
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs):
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass
    
    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Predict probabilities (if applicable)."""
        return None
    
    def scale_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Scale features using StandardScaler."""
        if fit:
            return self.scaler.fit_transform(X)
        return self.scaler.transform(X)
    
    def save(self, filepath: str):
        """Save model to disk."""
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model from disk."""
        self.model = joblib.load(filepath)
        self.is_trained = True
        print(f"Model loaded from {filepath}")


class LinearModel(BaseModel):
    """Logistic Regression model for binary classification."""
    
    def __init__(self):
        super().__init__("Linear")
        self.model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs):
        """Train logistic regression."""
        X_scaled = self.scale_features(X_train, fit=True)
        self.model.fit(X_scaled, y_train)
        self.is_trained = True
        print(f"{self.name} Model trained")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        X_scaled = self.scale_features(X, fit=False)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        X_scaled = self.scale_features(X, fit=False)
        return self.model.predict_proba(X_scaled)


class RandomForestModel(BaseModel):
    """Random Forest model for classification."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 15, random_state: int = 42):
        super().__init__("RandomForest")
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            verbose=0
        )
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs):
        """Train random forest."""
        X_scaled = self.scale_features(X_train, fit=True)
        self.model.fit(X_scaled, y_train)
        self.is_trained = True
        print(f"{self.name} Model trained")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        X_scaled = self.scale_features(X, fit=False)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        X_scaled = self.scale_features(X, fit=False)
        return self.model.predict_proba(X_scaled)
    
    def feature_importance(self) -> pd.DataFrame:
        """Get feature importance."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        if self.feature_cols is None:
            raise ValueError("Feature columns not set")
        
        importances = self.model.feature_importances_
        df_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return df_importance


class XGBoostModel(BaseModel):
    """XGBoost model for classification."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 6, learning_rate: float = 0.1):
        super().__init__("XGBoost")
        try:
            import xgboost as xgb
            self.model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
            self.use_xgb = True
        except ImportError:
            print("XGBoost not available, falling back to sklearn GradientBoostingClassifier")
            from sklearn.ensemble import GradientBoostingClassifier
            self.model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42
            )
            self.use_xgb = False
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs):
        """Train XGBoost or Gradient Boosting."""
        X_scaled = self.scale_features(X_train, fit=True)
        self.model.fit(X_scaled, y_train)
        self.is_trained = True
        print(f"{self.name} Model trained")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        X_scaled = self.scale_features(X, fit=False)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        X_scaled = self.scale_features(X, fit=False)
        return self.model.predict_proba(X_scaled)
    
    def feature_importance(self) -> pd.DataFrame:
        """Get feature importance."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        if self.feature_cols is None:
            raise ValueError("Feature columns not set")
        
        importances = self.model.feature_importances_
        df_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return df_importance


class GradientBoostingModel(BaseModel):
    """Gradient Boosting model for classification."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 5, learning_rate: float = 0.1):
        super().__init__("GradientBoosting")
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            verbose=0
        )
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs):
        """Train gradient boosting."""
        X_scaled = self.scale_features(X_train, fit=True)
        self.model.fit(X_scaled, y_train)
        self.is_trained = True
        print(f"{self.name} Model trained")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        X_scaled = self.scale_features(X, fit=False)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        X_scaled = self.scale_features(X, fit=False)
        return self.model.predict_proba(X_scaled)
    
    def feature_importance(self) -> pd.DataFrame:
        """Get feature importance."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        if self.feature_cols is None:
            raise ValueError("Feature columns not set")
        
        importances = self.model.feature_importances_
        df_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return df_importance


class MLPModel(BaseModel):
    """Multi-Layer Perceptron (neural network) model using PyTorch."""
    
    def __init__(self, hidden_layers: list = None, epochs: int = 50, batch_size: int = 32):
        super().__init__("MLP")
        try:
            import torch
            import torch.nn as nn
            self.torch = torch
            self.nn = nn
        except ImportError:
            raise ImportError("torch not installed. Install with: pip install torch")
        
        if hidden_layers is None:
            hidden_layers = [64, 32, 16]
        
        self.hidden_layers = hidden_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.device = self.torch.device('cuda' if self.torch.cuda.is_available() else 'cpu')
    
    def _build_model(self, input_dim: int):
        """Build the neural network architecture."""
        layers = []
        
        # Input and hidden layers
        prev_dim = input_dim
        for units in self.hidden_layers:
            layers.append(self.nn.Linear(prev_dim, units))
            layers.append(self.nn.ReLU())
            layers.append(self.nn.Dropout(0.2))
            prev_dim = units
        
        # Output layer
        layers.append(self.nn.Linear(prev_dim, 1))
        layers.append(self.nn.Sigmoid())
        
        return self.nn.Sequential(*layers).to(self.device)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None, verbose: int = 0):
        """Train the MLP model."""
        X_scaled = self.scale_features(X_train, fit=True)
        
        self.model = self._build_model(X_scaled.shape[1])
        optimizer = self.torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = self.nn.BCELoss()
        
        X_train_tensor = self.torch.FloatTensor(X_scaled).to(self.device)
        y_train_tensor = self.torch.FloatTensor(y_train).reshape(-1, 1).to(self.device)
        
        X_val_tensor = None
        y_val_tensor = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scale_features(X_val, fit=False)
            X_val_tensor = self.torch.FloatTensor(X_val_scaled).to(self.device)
            y_val_tensor = self.torch.FloatTensor(y_val).reshape(-1, 1).to(self.device)
        
        self.model.train()
        for epoch in range(self.epochs):
            for i in range(0, len(X_train_tensor), self.batch_size):
                X_batch = X_train_tensor[i:i+self.batch_size]
                y_batch = y_train_tensor[i:i+self.batch_size]
                
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.4f}")
        
        self.is_trained = True
        print(f"{self.name} Model trained")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained yet")
        X_scaled = self.scale_features(X, fit=False)
        X_tensor = self.torch.FloatTensor(X_scaled).to(self.device)
        
        self.model.eval()
        with self.torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
        
        return (predictions > 0.5).astype(int).flatten()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained yet")
        X_scaled = self.scale_features(X, fit=False)
        X_tensor = self.torch.FloatTensor(X_scaled).to(self.device)
        
        self.model.eval()
        with self.torch.no_grad():
            proba = self.model(X_tensor).cpu().numpy()
        
        return np.column_stack([1 - proba, proba])


class CNNModel(BaseModel):
    """Convolutional Neural Network model for sequence data using PyTorch."""
    
    def __init__(self, epochs: int = 50, batch_size: int = 32):
        super().__init__("CNN")
        try:
            import torch
            import torch.nn as nn
            self.torch = torch
            self.nn = nn
        except ImportError:
            raise ImportError("torch not installed. Install with: pip install torch")
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.seq_length = None
        self.device = self.torch.device('cuda' if self.torch.cuda.is_available() else 'cpu')
    
    class CNNNet(nn.Module):
        """PyTorch CNN architecture."""
        def __init__(self, input_features):
            super().__init__()
            self.conv1 = nn.Conv1d(input_features, 32, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm1d(32)
            self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm1d(64)
            self.pool1 = nn.MaxPool1d(2)
            
            self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm1d(128)
            self.pool2 = nn.MaxPool1d(2)
            
            self.fc1 = nn.Linear(128 * 5, 64)  # Adjust based on seq_length
            self.dropout = nn.Dropout(0.3)
            self.fc2 = nn.Linear(64, 1)
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, x):
            x = self.pool1(self.bn2(nn.functional.relu(self.conv2(self.bn1(nn.functional.relu(self.conv1(x)))))))
            x = self.pool2(self.bn3(nn.functional.relu(self.conv3(x))))
            x = x.view(x.size(0), -1)
            x = self.dropout(nn.functional.relu(self.fc1(x)))
            x = self.sigmoid(self.fc2(x))
            return x
    
    def _create_sequences(self, X: np.ndarray, seq_length: int) -> np.ndarray:
        """Create sequences for CNN input."""
        sequences = []
        for i in range(len(X) - seq_length + 1):
            sequences.append(X[i:i + seq_length])
        return np.array(sequences)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, seq_length: int = 20, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None, verbose: int = 0):
        """Train CNN model."""
        X_scaled = self.scale_features(X_train, fit=True)
        
        # Create sequences
        X_seq = self._create_sequences(X_scaled, seq_length)
        y_seq = y_train[seq_length - 1:]
        
        # Ensure matching lengths
        if len(X_seq) != len(y_seq):
            min_len = min(len(X_seq), len(y_seq))
            X_seq = X_seq[:min_len]
            y_seq = y_seq[:min_len]
        
        self.seq_length = seq_length
        
        # Build model with correct input shape
        self.model = self.CNNNet(X_scaled.shape[1]).to(self.device)
        optimizer = self.torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = self.nn.BCELoss()
        
        # Convert to tensors (CNN expects shape: [batch, features, seq_length])
        X_seq_tensor = self.torch.FloatTensor(X_seq.transpose(0, 2, 1)).to(self.device)
        y_seq_tensor = self.torch.FloatTensor(y_seq).reshape(-1, 1).to(self.device)
        
        self.model.train()
        for epoch in range(self.epochs):
            for i in range(0, len(X_seq_tensor), self.batch_size):
                X_batch = X_seq_tensor[i:i+self.batch_size]
                y_batch = y_seq_tensor[i:i+self.batch_size]
                
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.4f}")
        
        self.is_trained = True
        print(f"{self.name} Model trained")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained yet")
        if self.seq_length is None:
            raise ValueError("Sequence length not set")
        
        X_scaled = self.scale_features(X, fit=False)
        X_seq = self._create_sequences(X_scaled, self.seq_length)
        
        if len(X_seq) == 0:
            raise ValueError(f"Not enough data to create sequences of length {self.seq_length}")
        
        X_seq_tensor = self.torch.FloatTensor(X_seq.transpose(0, 2, 1)).to(self.device)
        
        self.model.eval()
        with self.torch.no_grad():
            predictions = self.model(X_seq_tensor).cpu().numpy()
        
        # Pad predictions to match input length
        padded_pred = np.zeros(len(X))
        padded_pred[self.seq_length - 1:self.seq_length - 1 + len(predictions)] = predictions.flatten()
        
        return (padded_pred > 0.5).astype(int)


class LSTMModel(BaseModel):
    """LSTM (Long Short-Term Memory) model for sequence prediction using PyTorch."""
    
    def __init__(self, epochs: int = 50, batch_size: int = 32, lstm_units: int = 50):
        super().__init__("LSTM")
        try:
            import torch
            import torch.nn as nn
            self.torch = torch
            self.nn = nn
        except ImportError:
            raise ImportError("torch not installed. Install with: pip install torch")
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.lstm_units = lstm_units
        self.model = None
        self.seq_length = None
        self.device = self.torch.device('cuda' if self.torch.cuda.is_available() else 'cpu')
    
    class LSTMNet(nn.Module):
        """PyTorch LSTM architecture."""
        def __init__(self, input_size, hidden_size):
            super().__init__()
            self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=0.2)
            self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True, dropout=0.2)
            self.dropout = nn.Dropout(0.2)
            self.fc1 = nn.Linear(hidden_size, 32)
            self.fc2 = nn.Linear(32, 1)
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, x):
            # x shape: [batch, seq_len, features]
            out, _ = self.lstm1(x)
            out, _ = self.lstm2(out)
            # Take the last output
            out = out[:, -1, :]  # [batch, hidden_size]
            out = self.dropout(nn.functional.relu(self.fc1(out)))
            out = self.sigmoid(self.fc2(out))
            return out
    
    def _create_sequences(self, X: np.ndarray, seq_length: int) -> np.ndarray:
        """Create sequences for LSTM input."""
        sequences = []
        for i in range(len(X) - seq_length + 1):
            sequences.append(X[i:i + seq_length])
        return np.array(sequences)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, seq_length: int = 60,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None, verbose: int = 0):
        """Train LSTM model."""
        X_scaled = self.scale_features(X_train, fit=True)
        
        # Create sequences
        X_seq = self._create_sequences(X_scaled, seq_length)
        y_seq = y_train[seq_length - 1:]
        
        # Ensure matching lengths
        if len(X_seq) != len(y_seq):
            min_len = min(len(X_seq), len(y_seq))
            X_seq = X_seq[:min_len]
            y_seq = y_seq[:min_len]
        
        self.seq_length = seq_length
        
        # Build model
        self.model = self.LSTMNet(X_scaled.shape[1], self.lstm_units).to(self.device)
        optimizer = self.torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = self.nn.BCELoss()
        
        # Convert to tensors
        X_seq_tensor = self.torch.FloatTensor(X_seq).to(self.device)
        y_seq_tensor = self.torch.FloatTensor(y_seq).reshape(-1, 1).to(self.device)
        
        self.model.train()
        for epoch in range(self.epochs):
            for i in range(0, len(X_seq_tensor), self.batch_size):
                X_batch = X_seq_tensor[i:i+self.batch_size]
                y_batch = y_seq_tensor[i:i+self.batch_size]
                
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.4f}")
        
        self.is_trained = True
        print(f"{self.name} Model trained")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained yet")
        if self.seq_length is None:
            raise ValueError("Sequence length not set")
        
        X_scaled = self.scale_features(X, fit=False)
        X_seq = self._create_sequences(X_scaled, self.seq_length)
        
        if len(X_seq) == 0:
            raise ValueError(f"Not enough data to create sequences of length {self.seq_length}")
        
        X_seq_tensor = self.torch.FloatTensor(X_seq).to(self.device)
        
        self.model.eval()
        with self.torch.no_grad():
            predictions = self.model(X_seq_tensor).cpu().numpy()
        
        # Pad predictions to match input length
        padded_pred = np.zeros(len(X))
        padded_pred[self.seq_length - 1:self.seq_length - 1 + len(predictions)] = predictions.flatten()
        
        return (padded_pred > 0.5).astype(int)


class TransformerModel(BaseModel):
    """Transformer-based model for sequence prediction using PyTorch."""
    
    def __init__(self, epochs: int = 50, batch_size: int = 32, d_model: int = 64, num_heads: int = 4):
        super().__init__("Transformer")
        try:
            import torch
            import torch.nn as nn
            self.torch = torch
            self.nn = nn
        except ImportError:
            raise ImportError("torch not installed. Install with: pip install torch")
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.model = None
        self.seq_length = None
        self.device = self.torch.device('cuda' if self.torch.cuda.is_available() else 'cpu')
    
    class TransformerNet(nn.Module):
        """PyTorch Transformer architecture."""
        def __init__(self, input_size, d_model, num_heads):
            super().__init__()
            self.embedding = nn.Linear(input_size, d_model)
            
            # Multi-head attention layer
            self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
            self.norm1 = nn.LayerNorm(d_model)
            
            # Feed-forward network
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.ReLU(),
                nn.Linear(d_model * 2, d_model)
            )
            self.norm2 = nn.LayerNorm(d_model)
            
            # Global average pooling
            self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
            
            # Output layer
            self.output = nn.Linear(d_model, 1)
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, x):
            # x shape: [batch, seq_len, features]
            
            # Embedding
            x = self.embedding(x)  # [batch, seq_len, d_model]
            
            # Multi-head attention with residual connection
            attn_output, _ = self.attention(x, x, x)
            x = self.norm1(x + attn_output)
            
            # Feed-forward network with residual connection
            ffn_output = self.ffn(x)
            x = self.norm2(x + ffn_output)
            
            # Global average pooling
            x = x.permute(0, 2, 1)  # [batch, d_model, seq_len]
            x = self.global_avg_pool(x).squeeze(-1)  # [batch, d_model]
            
            # Output
            x = self.sigmoid(self.output(x))
            return x
    
    def _create_sequences(self, X: np.ndarray, seq_length: int) -> np.ndarray:
        """Create sequences for Transformer input."""
        sequences = []
        for i in range(len(X) - seq_length + 1):
            sequences.append(X[i:i + seq_length])
        return np.array(sequences)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, seq_length: int = 60,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None, verbose: int = 0):
        """Train Transformer model."""
        X_scaled = self.scale_features(X_train, fit=True)
        
        # Create sequences
        X_seq = self._create_sequences(X_scaled, seq_length)
        y_seq = y_train[seq_length - 1:]
        
        # Ensure matching lengths
        if len(X_seq) != len(y_seq):
            min_len = min(len(X_seq), len(y_seq))
            X_seq = X_seq[:min_len]
            y_seq = y_seq[:min_len]
        
        self.seq_length = seq_length
        
        # Build model
        self.model = self.TransformerNet(X_scaled.shape[1], self.d_model, self.num_heads).to(self.device)
        optimizer = self.torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = self.nn.BCELoss()
        
        # Convert to tensors
        X_seq_tensor = self.torch.FloatTensor(X_seq).to(self.device)
        y_seq_tensor = self.torch.FloatTensor(y_seq).reshape(-1, 1).to(self.device)
        
        self.model.train()
        for epoch in range(self.epochs):
            for i in range(0, len(X_seq_tensor), self.batch_size):
                X_batch = X_seq_tensor[i:i+self.batch_size]
                y_batch = y_seq_tensor[i:i+self.batch_size]
                
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.4f}")
        
        self.is_trained = True
        print(f"{self.name} Model trained")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained yet")
        if self.seq_length is None:
            raise ValueError("Sequence length not set")
        
        X_scaled = self.scale_features(X, fit=False)
        X_seq = self._create_sequences(X_scaled, self.seq_length)
        
        if len(X_seq) == 0:
            raise ValueError(f"Not enough data to create sequences of length {self.seq_length}")
        
        X_seq_tensor = self.torch.FloatTensor(X_seq).to(self.device)
        
        self.model.eval()
        with self.torch.no_grad():
            predictions = self.model(X_seq_tensor).cpu().numpy()
        
        # Pad predictions to match input length
        padded_pred = np.zeros(len(X))
        padded_pred[self.seq_length - 1:self.seq_length - 1 + len(predictions)] = predictions.flatten()
        
        return (padded_pred > 0.5).astype(int)


class ModelEvaluator:
    """Utility class for model evaluation."""
    
    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Evaluate model predictions."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
        }
        
        if y_proba is not None:
            metrics['auc'] = roc_auc_score(y_true, y_proba[:, 1])
        
        return metrics
    
    @staticmethod
    def print_results(y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None,
                      model_name: str = "Model"):
        """Print evaluation results."""
        metrics = ModelEvaluator.evaluate(y_true, y_pred, y_proba)
        
        print(f"\n{'='*50}")
        print(f"{model_name} - Evaluation Results")
        print(f"{'='*50}")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1']:.4f}")
        if 'auc' in metrics:
            print(f"AUC-ROC:   {metrics['auc']:.4f}")
        
        cm = confusion_matrix(y_true, y_pred)
        print(f"\nConfusion Matrix:")
        print(cm)
        print(f"{'='*50}\n")
        
        return metrics


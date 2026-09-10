"""
Model Training & Evaluation Module for Black-Box & White-Box Models.
Trains XGBoost, Random Forest, Multi-Layer Perceptron (Neural Net), and Logistic Regression.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import xgboost as xgb


class ModelManager:
    """Manages training, persistence, and evaluation of black-box & white-box classifiers."""

    def __init__(self, artifacts_dir: str = 'artifacts'):
        self.artifacts_dir = artifacts_dir
        os.makedirs(self.artifacts_dir, exist_ok=True)
        self.models = {}
        self.metrics = {}

    def train_all(self, X_train, y_train, X_test, y_test, X_train_scaled=None, X_test_scaled=None):
        """Trains all models and saves metrics."""
        
        # 1. Black-Box Model 1: XGBoost Classifier
        print("[ModelManager] Training XGBoost Classifier (Black-Box)...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=120,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            eval_metric='logloss'
        )
        xgb_model.fit(X_train, y_train)
        self.models['xgboost'] = xgb_model

        # 2. Black-Box Model 2: Random Forest Classifier
        print("[ModelManager] Training Random Forest Classifier (Black-Box)...")
        rf_model = RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_split=4,
            random_state=42
        )
        rf_model.fit(X_train, y_train)
        self.models['random_forest'] = rf_model

        # 3. Black-Box Model 3: Multi-Layer Perceptron / Deep Neural Network
        print("[ModelManager] Training Multi-Layer Perceptron Neural Net (Black-Box)...")
        mlp_X_train = X_train_scaled if X_train_scaled is not None else X_train
        mlp_X_test = X_test_scaled if X_test_scaled is not None else X_test
        mlp_model = MLPClassifier(
            hidden_layer_sizes=(64, 32, 16),
            activation='relu',
            max_iter=400,
            random_state=42
        )
        mlp_model.fit(mlp_X_train, y_train)
        self.models['neural_net'] = mlp_model

        # 4. White-Box Baseline: Logistic Regression
        print("[ModelManager] Training Logistic Regression (White-Box Baseline)...")
        lr_X_train = X_train_scaled if X_train_scaled is not None else X_train
        lr_X_test = X_test_scaled if X_test_scaled is not None else X_test
        lr_model = LogisticRegression(max_iter=500, random_state=42)
        lr_model.fit(lr_X_train, y_train)
        self.models['logistic_regression'] = lr_model

        # Evaluate performance for all models
        for name, model in self.models.items():
            if name in ['neural_net', 'logistic_regression']:
                xt = mlp_X_test
            else:
                xt = X_test
                
            y_pred = model.predict(xt)
            y_proba = model.predict_proba(xt)[:, 1]

            cm = confusion_matrix(y_test, y_pred).tolist()
            self.metrics[name] = {
                'accuracy': round(float(accuracy_score(y_test, y_pred)), 4),
                'precision': round(float(precision_score(y_test, y_pred)), 4),
                'recall': round(float(recall_score(y_test, y_pred)), 4),
                'f1_score': round(float(f1_score(y_test, y_pred)), 4),
                'roc_auc': round(float(roc_auc_score(y_test, y_proba)), 4),
                'confusion_matrix': cm
            }

        self.save_models()
        return self.metrics

    def save_models(self):
        """Serializes models and metrics to artifacts directory."""
        for name, model in self.models.items():
            path = os.path.join(self.artifacts_dir, f"{name}.pkl")
            joblib.dump(model, path)
            print(f"[ModelManager] Saved {name} model to {path}")
            
        metrics_path = os.path.join(self.artifacts_dir, "metrics.json")
        import json
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"[ModelManager] Saved metrics to {metrics_path}")

    def load_models(self):
        """Loads serialized models from artifacts directory."""
        model_names = ['xgboost', 'random_forest', 'neural_net', 'logistic_regression']
        for name in model_names:
            path = os.path.join(self.artifacts_dir, f"{name}.pkl")
            if os.path.exists(path):
                self.models[name] = joblib.load(path)
                
        metrics_path = os.path.join(self.artifacts_dir, "metrics.json")
        if os.path.exists(metrics_path):
            import json
            with open(metrics_path, 'r') as f:
                self.metrics = json.load(f)
        return self.models, self.metrics


if __name__ == '__main__':
    from src.data_loader import get_prepared_data
    data = get_prepared_data()
    mm = ModelManager()
    metrics = mm.train_all(
        data['X_train'], data['y_train'],
        data['X_test'], data['y_test'],
        data['X_train_scaled'], data['X_test_scaled']
    )
    import json
    print(json.dumps(metrics, indent=2))

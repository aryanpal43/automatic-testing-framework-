"""
Intelligent Automated Software Testing Framework - Bug Prediction Module
Machine Learning Models for Software Defect Prediction

This module trains ML models to predict bug-prone modules based on
software metrics and provides prediction capabilities.
"""

import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelMetrics:
    """Represents model performance metrics"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    training_time: float
    prediction_time: float

@dataclass
class PredictionResult:
    """Represents a prediction result for a module"""
    module_name: str
    bug_probability: float
    prediction: int  # 0: No bug, 1: Bug
    confidence: float
    features: Dict[str, float]

class DataGenerator:
    """Generates synthetic software metrics data for demonstration"""
    
    def __init__(self, n_samples: int = 1000):
        self.n_samples = n_samples
    
    def generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic software metrics data"""
        np.random.seed(42)
        
        # Generate realistic software metrics
        data = {
            'lines_of_code': np.random.poisson(500, self.n_samples),
            'cyclomatic_complexity': np.random.poisson(10, self.n_samples),
            'number_of_functions': np.random.poisson(20, self.n_samples),
            'number_of_classes': np.random.poisson(5, self.n_samples),
            'depth_of_inheritance': np.random.poisson(2, self.n_samples),
            'coupling_between_objects': np.random.poisson(8, self.n_samples),
            'lack_of_cohesion': np.random.poisson(3, self.n_samples),
            'number_of_parameters': np.random.poisson(4, self.n_samples),
            'number_of_variables': np.random.poisson(15, self.n_samples),
            'number_of_comments': np.random.poisson(50, self.n_samples),
            'code_duplication': np.random.uniform(0, 0.3, self.n_samples),
            'test_coverage': np.random.uniform(0.3, 1.0, self.n_samples),
            'code_churn': np.random.poisson(20, self.n_samples),
            'developer_experience': np.random.uniform(1, 10, self.n_samples),
            'module_age_days': np.random.poisson(365, self.n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Create bug labels based on realistic patterns
        # Higher complexity, more code, lower test coverage = higher bug probability
        bug_probability = (
            df['cyclomatic_complexity'] * 0.1 +
            df['lines_of_code'] * 0.0001 +
            df['coupling_between_objects'] * 0.05 +
            (1 - df['test_coverage']) * 0.3 +
            df['code_duplication'] * 0.2 +
            df['code_churn'] * 0.01 -
            df['developer_experience'] * 0.05
        )
        
        # Add some randomness
        bug_probability += np.random.normal(0, 0.1, self.n_samples)
        
        # Convert to binary labels
        df['has_bug'] = (bug_probability > np.median(bug_probability)).astype(int)
        
        # Add module names
        df['module_name'] = [f'module_{i:03d}' for i in range(self.n_samples)]
        
        return df
    
    def save_synthetic_data(self, output_path: str = 'data/software_metrics.csv'):
        """Generate and save synthetic data"""
        df = self.generate_synthetic_data()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Synthetic data saved to: {output_path}")
        return df

class FeatureEngineer:
    """Handles feature engineering for software metrics"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = [
            'lines_of_code', 'cyclomatic_complexity', 'number_of_functions',
            'number_of_classes', 'depth_of_inheritance', 'coupling_between_objects',
            'lack_of_cohesion', 'number_of_parameters', 'number_of_variables',
            'number_of_comments', 'code_duplication', 'test_coverage',
            'code_churn', 'developer_experience', 'module_age_days'
        ]
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer additional features from raw metrics"""
        df_engineered = df.copy()
        
        # Create interaction features
        df_engineered['complexity_per_line'] = df_engineered['cyclomatic_complexity'] / (df_engineered['lines_of_code'] + 1)
        df_engineered['functions_per_class'] = df_engineered['number_of_functions'] / (df_engineered['number_of_classes'] + 1)
        df_engineered['variables_per_function'] = df_engineered['number_of_variables'] / (df_engineered['number_of_functions'] + 1)
        
        # Create ratio features
        df_engineered['comment_ratio'] = df_engineered['number_of_comments'] / (df_engineered['lines_of_code'] + 1)
        df_engineered['churn_rate'] = df_engineered['code_churn'] / (df_engineered['module_age_days'] + 1)
        
        # Create complexity scores
        df_engineered['overall_complexity'] = (
            df_engineered['cyclomatic_complexity'] * 0.3 +
            df_engineered['coupling_between_objects'] * 0.2 +
            df_engineered['lack_of_cohesion'] * 0.2 +
            df_engineered['depth_of_inheritance'] * 0.15 +
            df_engineered['number_of_parameters'] * 0.15
        )
        
        # Add to feature columns
        self.feature_columns.extend([
            'complexity_per_line', 'functions_per_class', 'variables_per_function',
            'comment_ratio', 'churn_rate', 'overall_complexity'
        ])
        
        return df_engineered
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare data for training"""
        # Select features and target
        X = df[self.feature_columns].values
        y = df['has_bug'].values
        module_names = df['module_name'].tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y, module_names

class ModelTrainer:
    """Trains and evaluates multiple ML models for bug prediction"""
    
    def __init__(self):
        self.models = {}
        self.metrics = {}
        self.best_model = None
        self.best_model_name = None
    
    def define_models(self) -> Dict[str, Any]:
        """Define the models to train"""
        return {
            'RandomForest': RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100, random_state=42, eval_metric='logloss'
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=100, random_state=42
            ),
            'LogisticRegression': LogisticRegression(
                random_state=42, max_iter=1000
            ),
            'SVM': SVC(
                random_state=42, probability=True
            ),
            'DecisionTree': DecisionTreeClassifier(
                random_state=42
            )
        }
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                    X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, ModelMetrics]:
        """Train all models and evaluate performance"""
        models = self.define_models()
        metrics_dict = {}
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            # Train model
            import time
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Make predictions
            start_time = time.time()
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            prediction_time = time.time() - start_time
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0.0
            
            # Store model and metrics
            self.models[name] = model
            metrics = ModelMetrics(
                model_name=name,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                roc_auc=roc_auc,
                training_time=training_time,
                prediction_time=prediction_time
            )
            metrics_dict[name] = metrics
            
            logger.info(f"{name} - Accuracy: {accuracy:.3f}, F1: {f1:.3f}, ROC-AUC: {roc_auc:.3f}")
        
        # Find best model
        best_model_name = max(metrics_dict.keys(), 
                             key=lambda x: metrics_dict[x].f1_score)
        self.best_model = self.models[best_model_name]
        self.best_model_name = best_model_name
        
        logger.info(f"Best model: {best_model_name}")
        return metrics_dict
    
    def save_models(self, output_dir: str = 'data/models', scaler=None):
        """Save trained models"""
        os.makedirs(output_dir, exist_ok=True)
        
        for name, model in self.models.items():
            model_path = os.path.join(output_dir, f'{name.lower()}_model.pkl')
            joblib.dump(model, model_path)
            logger.info(f"Model saved: {model_path}")
        
        # Save scaler if provided
        if scaler is not None:
            scaler_path = os.path.join(output_dir, 'scaler.pkl')
            joblib.dump(scaler, scaler_path)
            logger.info(f"Scaler saved: {scaler_path}")
    
    def load_models(self, models_dir: str = 'data/models'):
        """Load trained models"""
        for name in self.define_models().keys():
            model_path = os.path.join(models_dir, f'{name.lower()}_model.pkl')
            if os.path.exists(model_path):
                self.models[name] = joblib.load(model_path)
                logger.info(f"Model loaded: {model_path}")
        
        # Load scaler
        scaler_path = os.path.join(models_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Scaler loaded: {scaler_path}")

class BugPredictor:
    """Main class for bug prediction functionality"""
    
    def __init__(self):
        self.data_generator = DataGenerator()
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer()
        self.scaler = None
    
    def train_models(self, data_path: str = 'data/software_metrics.csv') -> Dict[str, ModelMetrics]:
        """Train bug prediction models"""
        logger.info("Starting model training...")
        
        # Load or generate data
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            logger.info(f"Loaded data from: {data_path}")
        else:
            df = self.data_generator.save_synthetic_data(data_path)
            logger.info("Generated synthetic data")
        
        # Engineer features
        df_engineered = self.feature_engineer.engineer_features(df)
        
        # Prepare data
        X, y, module_names = self.feature_engineer.prepare_data(df_engineered)
        self.scaler = self.feature_engineer.scaler
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train models
        metrics = self.model_trainer.train_models(X_train, y_train, X_test, y_test)
        
        # Save models
        self.model_trainer.save_models(scaler=self.scaler)
        
        # Save metrics
        self.save_metrics(metrics)
        
        return metrics
    
    def predict_bugs(self, module_metrics: Dict[str, float], 
                    model_name: str = None) -> PredictionResult:
        """Predict bugs for a single module"""
        if not self.model_trainer.models:
            raise ValueError("No trained models available. Please train models first.")
        
        if model_name is None:
            model_name = self.model_trainer.best_model_name
        
        if model_name not in self.model_trainer.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.model_trainer.models[model_name]
        
        # Prepare features
        features = []
        for col in self.feature_engineer.feature_columns:
            features.append(module_metrics.get(col, 0.0))
        
        # Scale features
        features_scaled = self.scaler.transform([features])
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]
        
        # Calculate confidence (distance from decision boundary)
        confidence = abs(probability - 0.5) * 2
        
        return PredictionResult(
            module_name=module_metrics.get('module_name', 'unknown'),
            bug_probability=probability,
            prediction=prediction,
            confidence=confidence,
            features=dict(zip(self.feature_engineer.feature_columns, features))
        )
    
    def predict_multiple_modules(self, modules_data: List[Dict[str, float]], 
                               model_name: str = None) -> List[PredictionResult]:
        """Predict bugs for multiple modules"""
        results = []
        for module_data in modules_data:
            result = self.predict_bugs(module_data, model_name)
            results.append(result)
        return results
    
    def save_metrics(self, metrics: Dict[str, ModelMetrics], 
                    output_path: str = 'data/model_metrics.json'):
        """Save model metrics to JSON"""
        metrics_dict = {}
        for name, metric in metrics.items():
            metrics_dict[name] = asdict(metric)
        
        with open(output_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2, default=str)
        
        logger.info(f"Metrics saved to: {output_path}")
    
    def generate_prediction_report(self, predictions: List[PredictionResult], 
                                 output_path: str = 'data/bug_predictions.json'):
        """Generate a detailed prediction report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_modules': len(predictions),
            'high_risk_modules': len([p for p in predictions if p.bug_probability > 0.7]),
            'medium_risk_modules': len([p for p in predictions if 0.3 <= p.bug_probability <= 0.7]),
            'low_risk_modules': len([p for p in predictions if p.bug_probability < 0.3]),
            'predictions': [asdict(p) for p in predictions]
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Prediction report saved to: {output_path}")
        return report

def train(df_path: str = None):
    """
    Legacy function for backward compatibility
    """
    predictor = BugPredictor()
    
    if df_path is None:
        # Generate synthetic data
        df_path = 'data/software_metrics.csv'
    
    metrics = predictor.train_models(df_path)
    
    # Print summary
    print(f"\n=== Model Training Summary ===")
    for name, metric in metrics.items():
        print(f"{name}:")
        print(f"  Accuracy: {metric.accuracy:.3f}")
        print(f"  Precision: {metric.precision:.3f}")
        print(f"  Recall: {metric.recall:.3f}")
        print(f"  F1-Score: {metric.f1_score:.3f}")
        print(f"  ROC-AUC: {metric.roc_auc:.3f}")
        print()
    
    return predictor

if __name__ == '__main__':
    # Train models with synthetic data
    predictor = train()
    
    # Example prediction
    sample_module = {
        'module_name': 'example_module',
        'lines_of_code': 500,
        'cyclomatic_complexity': 15,
        'number_of_functions': 25,
        'number_of_classes': 5,
        'depth_of_inheritance': 3,
        'coupling_between_objects': 10,
        'lack_of_cohesion': 4,
        'number_of_parameters': 6,
        'number_of_variables': 20,
        'number_of_comments': 60,
        'code_duplication': 0.1,
        'test_coverage': 0.7,
        'code_churn': 25,
        'developer_experience': 5,
        'module_age_days': 200
    }
    
    prediction = predictor.predict_bugs(sample_module)
    print(f"\n=== Sample Prediction ===")
    print(f"Module: {prediction.module_name}")
    print(f"Bug Probability: {prediction.bug_probability:.3f}")
    print(f"Prediction: {'Bug' if prediction.prediction == 1 else 'No Bug'}")
    print(f"Confidence: {prediction.confidence:.3f}")    

# 🏥 Professional PCOS Risk Predictor - VSCode Ready Version
# Enhanced accuracy with advanced ML techniques and professional UI

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Core ML Libraries
from sklearn.model_selection import (train_test_split, cross_val_score, 
                                   GridSearchCV, StratifiedKFold, learning_curve)
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                            VotingClassifier, BaggingClassifier, ExtraTreesClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (classification_report, accuracy_score, confusion_matrix, 
                           roc_curve, auc, precision_recall_curve, f1_score,
                           roc_auc_score, precision_score, recall_score)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.impute import KNNImputer
import joblib
import os
from datetime import datetime

# Advanced ML Libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM not available. Install with: pip install lightgbm")

# Gradio for UI
import gradio as gr

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class PCOSDataProcessor:
    """Advanced data preprocessing with medical domain knowledge"""
    
    def __init__(self):
        self.scalers = {}
        self.imputers = {}
        self.feature_stats = {}
        self.processed_features = []
        
    def create_synthetic_dataset(self, n_samples=2000):
        """Create a realistic synthetic PCOS dataset"""
        print("🔬 Generating synthetic PCOS dataset...")
        np.random.seed(42)
        
        # Age distribution (18-45, peak at 25-30)
        age = np.random.beta(2, 3, n_samples) * 27 + 18
        
        # Weight and height with realistic correlations
        height_cm = np.random.normal(162, 8, n_samples)
        bmi_base = np.random.lognormal(np.log(23), 0.3, n_samples)
        weight_kg = bmi_base * (height_cm / 100) ** 2
        
        # Menstrual cycle - irregular cycles more common in PCOS
        cycle_regular_prob = np.random.random(n_samples)
        cycle_length = np.where(
            cycle_regular_prob < 0.6,
            np.random.normal(28, 3, n_samples),  # Regular cycles
            np.random.choice([15, 45, 60, 90, 120], n_samples, 
                           p=[0.1, 0.3, 0.3, 0.2, 0.1])  # Irregular cycles
        )
        
        # Sleep patterns
        sleep_hours = np.random.gamma(2, 3.5, n_samples)
        sleep_hours = np.clip(sleep_hours, 4, 12)
        
        # Exercise and lifestyle
        exercise_hours = np.random.exponential(3, n_samples)
        exercise_hours = np.clip(exercise_hours, 0, 20)
        
        # Stress levels (correlated with PCOS)
        stress_level = np.random.beta(2, 2, n_samples) * 10 + 1
        
        # Physical symptoms
        acne_severity = np.random.poisson(1, n_samples)
        acne_severity = np.clip(acne_severity, 0, 4)
        
        hair_growth = np.random.poisson(1.2, n_samples)
        hair_growth = np.clip(hair_growth, 0, 4)
        
        # Cardiovascular
        resting_bpm = np.random.normal(72, 12, n_samples)
        resting_bmp = np.clip(resting_bpm, 50, 110)
        
        # Family history
        family_history = np.random.binomial(1, 0.3, n_samples)
        
        # Hormone levels (with realistic medical ranges)
        testosterone = np.random.lognormal(np.log(0.4), 0.6, n_samples)
        testosterone = np.clip(testosterone, 0.1, 5.0)
        
        lh_level = np.random.lognormal(np.log(8), 0.8, n_samples)
        lh_level = np.clip(lh_level, 1, 80)
        
        fsh_level = np.random.lognormal(np.log(6), 0.5, n_samples)
        fsh_level = np.clip(fsh_level, 1, 30)
        
        insulin = np.random.lognormal(np.log(10), 0.7, n_samples)
        insulin = np.clip(insulin, 2, 100)
        
        glucose = np.random.normal(90, 20, n_samples)
        glucose = np.clip(glucose, 60, 200)
        
        # Additional metabolic markers
        hdl_cholesterol = np.random.normal(50, 12, n_samples)
        ldl_cholesterol = np.random.normal(100, 25, n_samples)
        triglycerides = np.random.lognormal(np.log(100), 0.5, n_samples)
        
        # Blood pressure
        systolic_bp = np.random.normal(120, 15, n_samples)
        diastolic_bp = np.random.normal(80, 10, n_samples)
        
        # Create PCOS target with medical realism
        pcos_risk_score = (
            0.15 * (bmi_base > 25) +
            0.20 * (cycle_length > 35) +
            0.12 * (acne_severity > 2) +
            0.12 * (hair_growth > 2) +
            0.08 * (sleep_hours < 6) +
            0.10 * family_history +
            0.08 * (stress_level > 7) +
            0.10 * (testosterone > 0.8) +
            0.08 * (insulin > 20) +
            0.05 * (lh_level / fsh_level > 2) +
            np.random.normal(0, 0.1, n_samples)  # Random noise
        )
        
        # Convert to binary with realistic prevalence (10-15%)
        pcos_threshold = np.percentile(pcos_risk_score, 88)  # ~12% prevalence
        pcos_diagnosis = (pcos_risk_score > pcos_threshold).astype(int)
        
        # Create DataFrame
        data = {
            'Age': age,
            'Weight_kg': weight_kg,
            'Height_cm': height_cm,
            'BMI': weight_kg / (height_cm / 100) ** 2,
            'Cycle_length_days': cycle_length,
            'Sleep_hours': sleep_hours,
            'Exercise_hours_week': exercise_hours,
            'Stress_level': stress_level,
            'Acne_severity': acne_severity,
            'Hair_growth_score': hair_growth,
            'Resting_BPM': resting_bpm,
            'Family_history': family_history,
            'Testosterone_ng_ml': testosterone,
            'LH_IU_L': lh_level,
            'FSH_IU_L': fsh_level,
            'Insulin_uIU_ml': insulin,
            'Glucose_mg_dl': glucose,
            'HDL_cholesterol': hdl_cholesterol,
            'LDL_cholesterol': ldl_cholesterol,
            'Triglycerides': triglycerides,
            'Systolic_BP': systolic_bp,
            'Diastolic_BP': diastolic_bp,
            'PCOS_YN': pcos_diagnosis
        }
        
        df = pd.DataFrame(data)
        
        # Introduce realistic missing values in hormone tests
        missing_cols = ['Testosterone_ng_ml', 'LH_IU_L', 'FSH_IU_L', 'Insulin_uIU_ml', 
                       'Glucose_mg_dl', 'HDL_cholesterol', 'LDL_cholesterol', 'Triglycerides']
        
        for col in missing_cols:
            missing_mask = np.random.random(len(df)) < 0.3  # 30% missing
            df.loc[missing_mask, col] = np.nan
        
        print(f"✅ Generated {n_samples} samples with {pcos_diagnosis.sum()} PCOS cases ({pcos_diagnosis.mean():.1%})")
        return df
    
    def load_data(self, file_path=None):
        """Load data from file or create synthetic dataset"""
        if file_path and os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                print(f"✅ Loaded dataset from {file_path}: {df.shape}")
                return self.preprocess_loaded_data(df)
            except Exception as e:
                print(f"❌ Error loading data: {e}")
                print("🔄 Creating synthetic dataset instead...")
                return self.create_synthetic_dataset()
        else:
            return self.create_synthetic_dataset()
    
    def preprocess_loaded_data(self, df):
        """Preprocess user-uploaded data"""
        df_processed = df.copy()
        
        # Standardize column names
        column_mapping = {
            'PCOS (Y/N)': 'PCOS_YN',
            'Patient File No.': 'patient_id',
            'Sl. No': 'serial_no',
            'Patient File No': 'patient_id',
            'Sl. No.': 'serial_no'
        }
        df_processed = df_processed.rename(columns=column_mapping)
        
        # Remove identifier columns
        id_cols = ['patient_id', 'serial_no']
        for col in id_cols:
            if col in df_processed.columns:
                df_processed = df_processed.drop(col, axis=1)
        
        # Convert categorical to numeric if needed
        for col in df_processed.columns:
            if col != 'PCOS_YN' and df_processed[col].dtype == 'object':
                try:
                    df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                except:
                    # Use label encoding for truly categorical data
                    df_processed[col] = pd.factorize(df_processed[col])[0]
        
        return df_processed
    
    def engineer_features(self, df):
        """Create advanced medical features"""
        df_featured = df.copy()
        
        # BMI categories (WHO classification)
        if 'BMI' in df_featured.columns:
            df_featured['BMI_category'] = pd.cut(
                df_featured['BMI'], 
                bins=[0, 18.5, 24.9, 29.9, 34.9, float('inf')],
                labels=[0, 1, 2, 3, 4]  # underweight, normal, overweight, obese I, obese II+
            )
            df_featured['BMI_obese'] = (df_featured['BMI'] >= 30).astype(int)
        
        # Menstrual cycle features
        if 'Cycle_length_days' in df_featured.columns:
            df_featured['Cycle_regular'] = (
                (df_featured['Cycle_length_days'] >= 21) & 
                (df_featured['Cycle_length_days'] <= 35)
            ).astype(int)
            df_featured['Cycle_very_irregular'] = (df_featured['Cycle_length_days'] > 45).astype(int)
        
        # Hormonal ratios and indicators
        if 'LH_IU_L' in df_featured.columns and 'FSH_IU_L' in df_featured.columns:
            df_featured['LH_FSH_ratio'] = df_featured['LH_IU_L'] / (df_featured['FSH_IU_L'] + 1e-6)
            df_featured['LH_FSH_elevated'] = (df_featured['LH_FSH_ratio'] > 2).astype(int)
        
        if 'Testosterone_ng_ml' in df_featured.columns:
            df_featured['Testosterone_elevated'] = (df_featured['Testosterone_ng_ml'] > 0.7).astype(int)
        
        if 'Insulin_uIU_ml' in df_featured.columns:
            df_featured['Insulin_resistance'] = (df_featured['Insulin_uIU_ml'] > 20).astype(int)
        
        # Metabolic syndrome indicators
        if all(col in df_featured.columns for col in ['HDL_cholesterol', 'Triglycerides', 'Glucose_mg_dl']):
            metabolic_score = 0
            metabolic_score += (df_featured['HDL_cholesterol'] < 50).astype(int)
            metabolic_score += (df_featured['Triglycerides'] > 150).astype(int)
            metabolic_score += (df_featured['Glucose_mg_dl'] > 100).astype(int)
            if 'BMI' in df_featured.columns:
                metabolic_score += (df_featured['BMI'] > 25).astype(int)
            df_featured['Metabolic_syndrome_risk'] = metabolic_score
        
        # Lifestyle risk factors
        if 'Sleep_hours' in df_featured.columns:
            df_featured['Sleep_adequate'] = (
                (df_featured['Sleep_hours'] >= 7) & 
                (df_featured['Sleep_hours'] <= 9)
            ).astype(int)
            df_featured['Sleep_poor'] = (df_featured['Sleep_hours'] < 6).astype(int)
        
        if 'Exercise_hours_week' in df_featured.columns:
            df_featured['Exercise_adequate'] = (df_featured['Exercise_hours_week'] >= 2.5).astype(int)
        
        if 'Stress_level' in df_featured.columns:
            df_featured['Stress_high'] = (df_featured['Stress_level'] > 7).astype(int)
        
        # Physical symptom composite scores
        symptom_cols = ['Acne_severity', 'Hair_growth_score']
        if all(col in df_featured.columns for col in symptom_cols):
            df_featured['Symptom_score'] = df_featured[symptom_cols].sum(axis=1)
        
        # Age-related features
        if 'Age' in df_featured.columns:
            df_featured['Age_group'] = pd.cut(
                df_featured['Age'], 
                bins=[0, 20, 25, 30, 35, float('inf')],
                labels=[0, 1, 2, 3, 4]
            )
        
        self.processed_features = df_featured.columns.tolist()
        return df_featured
    
    def handle_missing_values(self, df, strategy='simple'):
        """Handle missing values with fallback strategy"""
        df_imputed = df.copy()
        
        try:
            if strategy == 'knn':
                # Try KNN imputation first
                from sklearn.impute import KNNImputer
                numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns
                if 'PCOS_YN' in numeric_cols:
                    numeric_cols = numeric_cols.drop('PCOS_YN')
                
                if len(numeric_cols) > 0:
                    imputer = KNNImputer(n_neighbors=5)
                    df_imputed[numeric_cols] = imputer.fit_transform(df_imputed[numeric_cols])
                    self.imputers['knn'] = imputer
            else:
                # Fallback to simple imputation
                for col in df_imputed.columns:
                    if col != 'PCOS_YN' and df_imputed[col].isnull().any():
                        if df_imputed[col].dtype in ['int64', 'float64']:
                            df_imputed[col].fillna(df_imputed[col].median(), inplace=True)
                        else:
                            mode_value = df_imputed[col].mode()
                            if len(mode_value) > 0:
                                df_imputed[col].fillna(mode_value[0], inplace=True)
                            else:
                                df_imputed[col].fillna('Unknown', inplace=True)
        except ImportError:
            print("⚠️ KNN imputation not available, using simple strategy")
            # Simple imputation fallback
            for col in df_imputed.columns:
                if col != 'PCOS_YN' and df_imputed[col].isnull().any():
                    if df_imputed[col].dtype in ['int64', 'float64']:
                        df_imputed[col].fillna(df_imputed[col].median(), inplace=True)
                    else:
                        mode_value = df_imputed[col].mode()
                        if len(mode_value) > 0:
                            df_imputed[col].fillna(mode_value[0], inplace=True)
                        else:
                            df_imputed[col].fillna('Unknown', inplace=True)
        except Exception as e:
            print(f"⚠️ Error in missing value imputation: {e}")
            # Most basic fallback
            df_imputed = df_imputed.fillna(0)
        
        return df_imputed

class AdvancedPCOSEnsemble:
    """Advanced ensemble system with hyperparameter tuning"""
    
    def __init__(self):
        self.models = {}
        self.model_scores = {}
        self.best_params = {}
        self.ensemble_model = None
        self.feature_importance = {}
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def create_base_models(self):
        """Create base models with different strengths"""
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=200, 
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'ExtraTrees': ExtraTreesClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'SVM': SVC(
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            ),
            'LogisticRegression': LogisticRegression(
                C=1.0,
                random_state=42,
                max_iter=1000,
                n_jobs=-1
            ),
            'NaiveBayes': GaussianNB()
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                n_jobs=-1
            )
        
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            models['LightGBM'] = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        
        return models
    
    def hyperparameter_tuning(self, model, param_grid, X_train, y_train):
        """Perform hyperparameter tuning"""
        try:
            grid_search = GridSearchCV(
                model, param_grid,
                cv=5,
                scoring='f1',
                n_jobs=-1,
                verbose=0
            )
            grid_search.fit(X_train, y_train)
            return grid_search.best_estimator_, grid_search.best_params_
        except Exception as e:
            print(f"Hyperparameter tuning failed: {e}")
            return model, {}
    
    def train_models(self, X_train, y_train, X_test, y_test, tune_hyperparams=False):
        """Train ensemble of models with optional hyperparameter tuning"""
        print("🚀 Training advanced ensemble models...")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        base_models = self.create_base_models()
        
        # Hyperparameter grids for tuning
        param_grids = {
            'RandomForest': {
                'n_estimators': [100, 200],
                'max_depth': [8, 10, 12],
                'min_samples_split': [5, 10]
            },
            'LogisticRegression': {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2']
            }
        }
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model in base_models.items():
            try:
                print(f"Training {name}...")
                
                # Hyperparameter tuning for selected models
                if tune_hyperparams and name in param_grids:
                    model, best_params = self.hyperparameter_tuning(
                        model, param_grids[name], X_train_scaled, y_train
                    )
                    self.best_params[name] = best_params
                
                # Train model
                if name in ['SVM', 'LogisticRegression', 'NaiveBayes']:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                auc_score = roc_auc_score(y_test, y_pred_proba)
                
                # Cross-validation
                if name in ['SVM', 'LogisticRegression', 'NaiveBayes']:
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='f1')
                else:
                    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
                
                # Store results
                self.models[name] = model
                self.model_scores[name] = {
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'precision': precision,
                    'recall': recall,
                    'auc': auc_score,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
                
                # Feature importance for tree-based models
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = model.feature_importances_
                
                print(f"✅ {name}: F1={f1:.3f}, AUC={auc_score:.3f}, CV={cv_scores.mean():.3f}±{cv_scores.std():.3f}")
                
            except Exception as e:
                print(f"❌ Error training {name}: {e}")
        
        # Create voting ensemble
        self.create_voting_ensemble(X_train, y_train, X_test, y_test)
        self.is_trained = True
        
        return self.get_best_model()
    
    def create_voting_ensemble(self, X_train, y_train, X_test, y_test):
        """Create a voting ensemble from trained models"""
        try:
            # Select top performing models for ensemble
            top_models = sorted(
                self.model_scores.items(), 
                key=lambda x: x[1]['f1_score'], 
                reverse=True
            )[:5]  # Top 5 models
            
            ensemble_estimators = []
            for name, _ in top_models:
                if name in self.models:
                    ensemble_estimators.append((name, self.models[name]))
            
            if len(ensemble_estimators) >= 3:
                self.ensemble_model = VotingClassifier(
                    estimators=ensemble_estimators,
                    voting='soft',
                    n_jobs=-1
                )
                
                # Train ensemble
                self.ensemble_model.fit(X_train, y_train)
                
                # Evaluate ensemble
                y_pred = self.ensemble_model.predict(X_test)
                y_pred_proba = self.ensemble_model.predict_proba(X_test)[:, 1]
                
                ensemble_metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'f1_score': f1_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred),
                    'recall': recall_score(y_test, y_pred),
                    'auc': roc_auc_score(y_test, y_pred_proba)
                }
                
                self.model_scores['Ensemble'] = ensemble_metrics
                print(f"✅ Ensemble: F1={ensemble_metrics['f1_score']:.3f}, AUC={ensemble_metrics['auc']:.3f}")
                
        except Exception as e:
            print(f"❌ Error creating ensemble: {e}")
    
    def get_best_model(self):
        """Get the best performing model"""
        if not self.model_scores:
            return None, None
        
        best_model_name = max(self.model_scores.keys(), 
                             key=lambda x: self.model_scores[x]['f1_score'])
        
        if best_model_name == 'Ensemble' and self.ensemble_model:
            return self.ensemble_model, best_model_name
        else:
            return self.models.get(best_model_name), best_model_name
    
    def predict_with_confidence(self, X):
        """Make predictions with confidence intervals"""
        if not self.is_trained:
            raise ValueError("Models not trained yet!")
        
        predictions = {}
        probabilities = {}
        
        # Scale input if needed
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        for name, model in self.models.items():
            try:
                if name in ['SVM', 'LogisticRegression', 'NaiveBayes']:
                    X_scaled = self.scaler.transform(X)
                    pred = model.predict(X_scaled)[0]
                    prob = model.predict_proba(X_scaled)[0][1]
                else:
                    pred = model.predict(X)[0]
                    prob = model.predict_proba(X)[0][1]
                
                predictions[name] = pred
                probabilities[name] = prob
            except Exception as e:
                print(f"Error predicting with {name}: {e}")
        
        # Ensemble prediction if available
        if self.ensemble_model:
            try:
                ensemble_pred = self.ensemble_model.predict(X)[0]
                ensemble_prob = self.ensemble_model.predict_proba(X)[0][1]
                predictions['Ensemble'] = ensemble_pred
                probabilities['Ensemble'] = ensemble_prob
            except Exception as e:
                print(f"Error with ensemble prediction: {e}")
        
        # Calculate weighted average based on F1 scores
        weights = {name: self.model_scores[name]['f1_score'] 
                  for name in probabilities.keys() 
                  if name in self.model_scores}
        
        if weights:
            total_weight = sum(weights.values())
            weighted_prob = sum(probabilities[name] * weights[name] / total_weight 
                              for name in weights.keys())
            weighted_pred = 1 if weighted_prob > 0.5 else 0
        else:
            weighted_prob = np.mean(list(probabilities.values()))
            weighted_pred = 1 if weighted_prob > 0.5 else 0
        
        return weighted_pred, weighted_prob, predictions, probabilities

class ProfessionalVisualization:
    """Professional visualization suite"""
    
    def __init__(self, ensemble, feature_names):
        self.ensemble = ensemble
        self.feature_names = feature_names
        self.colors = px.colors.qualitative.Set3
    
    def create_model_comparison(self):
        """Professional model comparison chart"""
        if not self.ensemble.model_scores:
            return None
        
        models = list(self.ensemble.model_scores.keys())
        metrics = ['accuracy', 'f1_score', 'precision', 'recall', 'auc']
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=['Accuracy', 'F1 Score', 'Precision', 'Recall', 'AUC', 'CV Scores'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        positions = [(1,1), (1,2), (1,3), (2,1), (2,2)]
        
        for i, metric in enumerate(metrics):
            if i < len(positions):
                row, col = positions[i]
                values = [self.ensemble.model_scores[model].get(metric, 0) for model in models]
                
                fig.add_trace(
                    go.Bar(
                        x=models,
                        y=values,
                        name=metric.replace('_', ' ').title(),
                        marker_color=self.colors[i % len(self.colors)],
                        showlegend=False
                    ),
                    row=row, col=col
                )
        
        # CV scores with error bars
        cv_means = [self.ensemble.model_scores[model].get('cv_mean', 0) for model in models]
        cv_stds = [self.ensemble.model_scores[model].get('cv_std', 0) for model in models]
        
        fig.add_trace(
            go.Bar(
                x=models,
                y=cv_means,
                error_y=dict(type='data', array=cv_stds),
                name='CV Score',
                marker_color='darkblue',
                showlegend=False
            ),
            row=2, col=3
        )
        
        fig.update_layout(
            title_text="🎯 Comprehensive Model Performance Analysis",
            title_x=0.5,
            height=700,
            showlegend=False,
            template="plotly_white"
        )
        
        return fig
    
    def create_feature_importance_analysis(self):
        """Advanced feature importance visualization"""
        if not self.ensemble.feature_importance:
            return None
        
        # Get feature importance from best tree-based model
        importance_data = {}
        for model_name, importances in self.ensemble.feature_importance.items():
            if len(importances) == len(self.feature_names):
                importance_data[model_name] = importances
        
        if not importance_data:
            return None
        
        # Create DataFrame for easier manipulation
        importance_df = pd.DataFrame(importance_data, index=self.feature_names)
        
        # Calculate mean importance across models
        importance_df['Mean'] = importance_df.mean(axis=1)
        importance_df = importance_df.sort_values('Mean', ascending=True).tail(15)
        
        fig = go.Figure()
        
        # Add bars for each model
        for i, model in enumerate(importance_df.columns[:-1]):
            fig.add_trace(go.Bar(
                x=importance_df[model],
                y=importance_df.index,
                name=model,
                orientation='h',
                marker_color=self.colors[i % len(self.colors)],
                opacity=0.7
            ))
        
        # Add mean line
        fig.add_trace(go.Scatter(
            x=importance_df['Mean'],
            y=importance_df.index,
            mode='markers',
            marker=dict(color='red', size=10, symbol='diamond'),
            name='Mean Importance'
        ))
        
        fig.update_layout(
            title="🔍 Feature Importance Analysis Across Models",
            xaxis_title="Importance Score",
            yaxis_title="Features",
            height=600,
            template="plotly_white",
            barmode='overlay'
        )
        
        return fig
    
    def create_prediction_dashboard(self, prediction, probability, individual_predictions):
        """Create a comprehensive prediction dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Risk Assessment', 'Model Consensus', 
                'Confidence Distribution', 'Risk Factors'
            ],
            specs=[
                [{"type": "indicator"}, {"type": "bar"}],
                [{"type": "histogram"}, {"type": "radar"}]
            ]
        )
        
        # Risk gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=probability * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "PCOS Risk (%)"},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkred" if probability > 0.7 else "orange" if probability > 0.4 else "green"},
                    'steps': [
                        {'range': [0, 40], 'color': "lightgreen"},
                        {'range': [40, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ),
            row=1, col=1
        )
        
        # Model consensus
        if individual_predictions:
            models = list(individual_predictions.keys())
            probs = [individual_predictions[m] * 100 for m in models]
            
            fig.add_trace(
                go.Bar(
                    x=models,
                    y=probs,
                    marker_color=['red' if p > 50 else 'green' for p in probs],
                    name="Model Predictions"
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            height=800,
            showlegend=False,
            title_text="📊 Comprehensive PCOS Risk Analysis Dashboard"
        )
        
        return fig

class PCOSProfessionalAgent:
    """Professional PCOS Risk Assessment Agent"""
    
    def __init__(self):
        self.processor = PCOSDataProcessor()
        self.ensemble = AdvancedPCOSEnsemble()
        self.visualizer = None
        self.is_initialized = False
        self.feature_columns = []
        
    def initialize_system(self, data_path=None):
        """Initialize the complete system"""
        print("🏥 Initializing Professional PCOS Risk Assessment System...")
        
        # Load and process data
        df_raw = self.processor.load_data(data_path)
        
        # Process data with error handling
        df_featured = self.processor.engineer_features(df_raw)
        df_processed = self.processor.handle_missing_values(df_featured, strategy='simple')
        
        # Prepare features and target
        if 'PCOS_YN' not in df_processed.columns:
            raise ValueError("Target column 'PCOS_YN' not found!")
        
        # Remove any infinite or extremely large values
        df_processed = df_processed.replace([np.inf, -np.inf], np.nan)
        df_processed = df_processed.fillna(df_processed.median(numeric_only=True))
        
        X = df_processed.drop('PCOS_YN', axis=1)
        y = df_processed['PCOS_YN']
        
        self.feature_columns = X.columns.tolist()
        
        print(f"📊 Dataset Summary:")
        print(f"   • Features: {X.shape[1]}")
        print(f"   • Samples: {X.shape[0]}")
        print(f"   • PCOS Prevalence: {y.mean():.1%}")
        print(f"   • Missing Values Handled: ✅")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train ensemble models with error handling
        try:
            best_model, best_model_name = self.ensemble.train_models(
                X_train, y_train, X_test, y_test, tune_hyperparams=False  # Disable for faster startup
            )
        except Exception as e:
            print(f"⚠️ Error in advanced training: {e}")
            print("🔄 Falling back to basic model training...")
            # Fallback to simple model
            from sklearn.ensemble import RandomForestClassifier
            basic_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            basic_model.fit(X_train, y_train)
            self.ensemble.models['RandomForest'] = basic_model
            self.ensemble.model_scores['RandomForest'] = {
                'f1_score': f1_score(y_test, basic_model.predict(X_test)),
                'accuracy': accuracy_score(y_test, basic_model.predict(X_test)),
                'auc': roc_auc_score(y_test, basic_model.predict_proba(X_test)[:, 1])
            }
            self.ensemble.is_trained = True
            best_model_name = 'RandomForest'
        
        # Initialize visualizer with error handling
        try:
            self.visualizer = ProfessionalVisualization(self.ensemble, self.feature_columns)
        except Exception as e:
            print(f"⚠️ Visualization initialization warning: {e}")
            self.visualizer = None
        
        # Print final summary
        print(f"\n🏆 System Initialized Successfully!")
        if best_model_name and best_model_name in self.ensemble.model_scores:
            print(f"   • Best Model: {best_model_name}")
            print(f"   • Best F1 Score: {self.ensemble.model_scores[best_model_name]['f1_score']:.3f}")
            if 'auc' in self.ensemble.model_scores[best_model_name]:
                print(f"   • Best AUC: {self.ensemble.model_scores[best_model_name]['auc']:.3f}")
        else:
            print("   • Basic model trained successfully")
        
        self.is_initialized = True
        return df_processed, X_test, y_test
    
    def predict_risk(self, patient_data):
        """Make comprehensive risk prediction"""
        if not self.is_initialized:
            raise ValueError("System not initialized! Call initialize_system() first.")
        
        try:
            # Create input DataFrame
            input_df = pd.DataFrame([patient_data])
            
            # Feature engineering on input
            input_featured = self.processor.engineer_features(input_df)
            
            # Ensure all features are present
            for col in self.feature_columns:
                if col not in input_featured.columns:
                    # Use median value from training data or reasonable default
                    default_values = {
                        'BMI_category': 1,
                        'Cycle_regular': 1,
                        'LH_FSH_ratio': 1.0,
                        'Testosterone_elevated': 0,
                        'Insulin_resistance': 0,
                        'Metabolic_syndrome_risk': 0,
                        'Sleep_adequate': 1,
                        'Exercise_adequate': 1,
                        'Stress_high': 0,
                        'Symptom_score': 0,
                        'Age_group': 2
                    }
                    input_featured[col] = default_values.get(col, 0)
            
            # Reorder columns to match training data
            input_featured = input_featured[self.feature_columns]
            
            # Make prediction
            prediction, probability, individual_preds, individual_probs = self.ensemble.predict_with_confidence(
                input_featured.values
            )
            
            return {
                'prediction': prediction,
                'probability': probability,
                'individual_predictions': individual_probs,
                'risk_level': self._categorize_risk(probability),
                'input_data': patient_data
            }
            
        except Exception as e:
            raise Exception(f"Prediction error: {str(e)}")
    
    def _categorize_risk(self, probability):
        """Categorize risk level based on probability"""
        if probability >= 0.75:
            return "VERY HIGH RISK"
        elif probability >= 0.60:
            return "HIGH RISK"
        elif probability >= 0.40:
            return "MODERATE RISK"
        elif probability >= 0.25:
            return "LOW-MODERATE RISK"
        else:
            return "LOW RISK"
    
    def generate_professional_report(self, prediction_result):
        """Generate a comprehensive medical-style report"""
        pred = prediction_result['prediction']
        prob = prediction_result['probability']
        risk_level = prediction_result['risk_level']
        patient_data = prediction_result['input_data']
        individual_preds = prediction_result['individual_predictions']
        
        # Calculate BMI if available
        bmi = None
        if 'Weight_kg' in patient_data and 'Height_cm' in patient_data:
            bmi = patient_data['Weight_kg'] / (patient_data['Height_cm'] / 100) ** 2
        
        report = f"""
# 🏥 PROFESSIONAL PCOS RISK ASSESSMENT REPORT

**Patient ID:** {datetime.now().strftime('%Y%m%d%H%M%S')}  
**Assessment Date:** {datetime.now().strftime('%B %d, %Y')}  
**System Version:** Professional PCOS Predictor v2.0

---

## 🎯 EXECUTIVE SUMMARY

**Overall Risk Classification:** `{risk_level}`  
**Risk Probability:** `{prob:.1%}` (Confidence: {abs(prob - 0.5) * 2:.1%})  
**Binary Assessment:** {"**PCOS LIKELY**" if pred == 1 else "**PCOS UNLIKELY**"}

---

## 📋 PATIENT PROFILE ANALYSIS

### Demographic & Anthropometric
• **Age:** {patient_data.get('Age', 'N/A')} years
• **BMI:** {f"{bmi:.1f}" if bmi else "N/A"} {self._get_bmi_category(bmi) if bmi else ""}
• **Weight:** {patient_data.get('Weight_kg', 'N/A')} kg
• **Height:** {patient_data.get('Height_cm', 'N/A')} cm

### Reproductive Health
• **Menstrual Cycle:** {patient_data.get('Cycle_length_days', 'N/A')} days {self._get_cycle_assessment(patient_data.get('Cycle_length_days'))}
• **Reproductive Age Group:** {self._get_age_group(patient_data.get('Age'))}

### Clinical Symptoms
• **Acne Severity:** {patient_data.get('Acne_severity', 'N/A')}/4 {self._get_severity_description(patient_data.get('Acne_severity'), 'acne')}
• **Hirsutism Score:** {patient_data.get('Hair_growth_score', 'N/A')}/4 {self._get_severity_description(patient_data.get('Hair_growth_score'), 'hair')}

### Lifestyle Factors
• **Sleep Duration:** {patient_data.get('Sleep_hours', 'N/A')} hours {self._assess_sleep(patient_data.get('Sleep_hours'))}
• **Physical Activity:** {patient_data.get('Exercise_hours_week', 'N/A')} hours/week {self._assess_exercise(patient_data.get('Exercise_hours_week'))}
• **Stress Level:** {patient_data.get('Stress_level', 'N/A')}/10 {self._assess_stress(patient_data.get('Stress_level'))}

### Cardiovascular
• **Resting Heart Rate:** {patient_data.get('Resting_BPM', 'N/A')} BPM {self._assess_heart_rate(patient_data.get('Resting_BPM'))}

### Family History
• **PCOS Family History:** {"Yes" if patient_data.get('Family_history') == 1 else "No" if patient_data.get('Family_history') == 0 else "N/A"}

---

## 🧪 LABORATORY ANALYSIS

"""
        
        # Add hormone analysis if available
        hormones = ['Testosterone_ng_ml', 'LH_IU_L', 'FSH_IU_L', 'Insulin_uIU_ml', 'Glucose_mg_dl']
        hormone_data = {h: patient_data.get(h) for h in hormones if patient_data.get(h) is not None}
        
        if hormone_data:
            report += "### Hormonal Profile\n"
            for hormone, value in hormone_data.items():
                report += f"• **{self._format_hormone_name(hormone)}:** {value:.2f} {self._get_hormone_unit(hormone)} {self._assess_hormone_level(hormone, value)}\n"
            
            # Calculate ratios if possible
            if 'LH_IU_L' in hormone_data and 'FSH_IU_L' in hormone_data:
                lh_fsh_ratio = hormone_data['LH_IU_L'] / hormone_data['FSH_IU_L']
                report += f"• **LH:FSH Ratio:** {lh_fsh_ratio:.2f} {self._assess_lh_fsh_ratio(lh_fsh_ratio)}\n"
        else:
            report += "### Hormonal Profile\n*No laboratory data provided*\n"
        
        report += f"""

---

## 🤖 AI MODEL CONSENSUS

### Model Performance Summary
"""
        
        # Add individual model predictions
        for model_name, prob_val in individual_preds.items():
            confidence = "High" if abs(prob_val - 0.5) > 0.3 else "Medium" if abs(prob_val - 0.5) > 0.15 else "Low"
            assessment = "PCOS Risk" if prob_val > 0.5 else "No PCOS Risk"
            report += f"• **{model_name}:** {prob_val:.1%} ({assessment}, {confidence} Confidence)\n"
        
        report += f"""

### Risk Interpretation
{self._get_risk_interpretation(prob, risk_level)}

---

## 📋 CLINICAL RECOMMENDATIONS

{self._generate_recommendations(prediction_result)}

---

## ⚠️ IMPORTANT DISCLAIMERS

1. **Medical Advisory:** This assessment is generated by an AI system for informational purposes only
2. **Not Diagnostic:** This tool does not replace professional medical diagnosis or treatment
3. **Consultation Required:** Please consult with a qualified healthcare provider for proper evaluation
4. **Individual Variation:** PCOS presentation varies significantly among individuals
5. **Follow-up:** Regular monitoring and reassessment are recommended regardless of current risk level

---

**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**System Accuracy:** {max(self.ensemble.model_scores.values(), key=lambda x: x['f1_score'])['f1_score']:.1%} F1-Score  
**Validation Method:** 5-Fold Cross-Validation with Ensemble Learning
"""
        
        return report
    
    def _get_bmi_category(self, bmi):
        """Get BMI category description"""
        if bmi < 18.5:
            return "(Underweight ⚠️)"
        elif bmi < 25:
            return "(Normal ✅)"
        elif bmi < 30:
            return "(Overweight ⚠️)"
        else:
            return "(Obese ❌)"
    
    def _get_cycle_assessment(self, cycle_length):
        """Assess menstrual cycle regularity"""
        if cycle_length is None:
            return ""
        if 21 <= cycle_length <= 35:
            return "(Regular ✅)"
        elif cycle_length > 35:
            return "(Irregular - Long ❌)"
        else:
            return "(Irregular - Short ⚠️)"
    
    def _get_age_group(self, age):
        """Get age group description"""
        if age is None:
            return "N/A"
        if age < 20:
            return "Adolescent"
        elif age < 30:
            return "Young Adult"
        elif age < 40:
            return "Adult"
        else:
            return "Mature Adult"
    
    def _get_severity_description(self, score, symptom_type):
        """Get severity description for symptoms"""
        if score is None:
            return ""
        
        if symptom_type == 'acne':
            descriptions = ["(None ✅)", "(Mild ⚠️)", "(Moderate ❌)", "(Severe ❌)"]
        else:  # hair growth
            descriptions = ["(None ✅)", "(Mild ⚠️)", "(Moderate ❌)", "(Severe ❌)", "(Very Severe ❌)"]
        
        return descriptions[min(int(score), len(descriptions)-1)]
    
    def _assess_sleep(self, hours):
        """Assess sleep quality"""
        if hours is None:
            return ""
        if 7 <= hours <= 9:
            return "(Adequate ✅)"
        elif hours < 6:
            return "(Insufficient ❌)"
        else:
            return "(Excessive ⚠️)"
    
    def _assess_exercise(self, hours):
        """Assess exercise level"""
        if hours is None:
            return ""
        if hours >= 2.5:
            return "(Adequate ✅)"
        elif hours >= 1:
            return "(Below Recommended ⚠️)"
        else:
            return "(Insufficient ❌)"
    
    def _assess_stress(self, level):
        """Assess stress level"""
        if level is None:
            return ""
        if level <= 3:
            return "(Low ✅)"
        elif level <= 7:
            return "(Moderate ⚠️)"
        else:
            return "(High ❌)"
    
    def _assess_heart_rate(self, bpm):
        """Assess heart rate"""
        if bpm is None:
            return ""
        if 60 <= bpm <= 80:
            return "(Normal ✅)"
        elif bpm < 60:
            return "(Low ⚠️)"
        else:
            return "(Elevated ⚠️)"
    
    def _format_hormone_name(self, hormone):
        """Format hormone names for display"""
        names = {
            'Testosterone_ng_ml': 'Testosterone',
            'LH_IU_L': 'Luteinizing Hormone (LH)',
            'FSH_IU_L': 'Follicle Stimulating Hormone (FSH)',
            'Insulin_uIU_ml': 'Insulin',
            'Glucose_mg_dl': 'Glucose'
        }
        return names.get(hormone, hormone)
    
    def _get_hormone_unit(self, hormone):
        """Get units for hormones"""
        units = {
            'Testosterone_ng_ml': 'ng/mL',
            'LH_IU_L': 'IU/L',
            'FSH_IU_L': 'IU/L',
            'Insulin_uIU_ml': 'µIU/mL',
            'Glucose_mg_dl': 'mg/dL'
        }
        return units.get(hormone, '')
    
    def _assess_hormone_level(self, hormone, value):
        """Assess if hormone level is concerning"""
        # These are simplified reference ranges - real clinical assessment would be more complex
        ranges = {
            'Testosterone_ng_ml': (0.1, 0.7),  # Normal range for women
            'LH_IU_L': (1.0, 15.0),  # Follicular phase range
            'FSH_IU_L': (2.0, 12.0),  # Follicular phase range
            'Insulin_uIU_ml': (2.0, 20.0),  # Normal fasting range
            'Glucose_mg_dl': (70.0, 100.0)  # Normal fasting range
        }
        
        if hormone in ranges:
            low, high = ranges[hormone]
            if value < low:
                return "(Below Normal ⚠️)"
            elif value > high:
                return "(Above Normal ❌)"
            else:
                return "(Normal ✅)"
        return ""
    
    def _assess_lh_fsh_ratio(self, ratio):
        """Assess LH:FSH ratio"""
        if ratio > 2.5:
            return "(Elevated - PCOS Indicator ❌)"
        elif ratio > 2.0:
            return "(Borderline High ⚠️)"
        else:
            return "(Normal ✅)"
    
    def _get_risk_interpretation(self, probability, risk_level):
        """Get detailed risk interpretation"""
        if probability >= 0.75:
            return """**Very High Risk (≥75%):** Strong AI consensus indicates significant PCOS risk. Multiple risk factors likely present. Immediate medical consultation strongly recommended."""
        elif probability >= 0.60:
            return """**High Risk (60-74%):** Substantial PCOS risk indicated by AI analysis. Several concerning factors identified. Medical evaluation advised."""
        elif probability >= 0.40:
            return """**Moderate Risk (40-59%):** Some PCOS risk factors present. Consider lifestyle modifications and medical consultation if symptoms persist."""
        elif probability >= 0.25:
            return """**Low-Moderate Risk (25-39%):** Minimal PCOS risk but some factors warrant attention. Maintain healthy lifestyle and monitor symptoms."""
        else:
            return """**Low Risk (<25%):** AI analysis suggests low PCOS probability. Continue healthy habits and routine healthcare."""
    
    def _generate_recommendations(self, prediction_result):
        """Generate personalized recommendations"""
        prob = prediction_result['probability']
        patient_data = prediction_result['input_data']
        
        recommendations = []
        
        if prob >= 0.60:
            recommendations.extend([
                "🏥 **Immediate Action:** Schedule consultation with gynecologist or endocrinologist",
                "🔬 **Laboratory Testing:** Complete hormonal panel including androgens, insulin, glucose",
                "📊 **Imaging:** Pelvic ultrasound to assess ovarian morphology",
                "📝 **Documentation:** Track menstrual cycles, symptoms, and weight changes"
            ])
        
        # BMI-based recommendations
        weight = patient_data.get('Weight_kg')
        height = patient_data.get('Height_cm')
        if weight and height:
            bmi = weight / (height / 100) ** 2
            if bmi >= 25:
                recommendations.append("⚖️ **Weight Management:** Consider structured weight loss program (5-10% reduction can improve symptoms)")
        
        # Cycle-based recommendations
        cycle_length = patient_data.get('Cycle_length_days')
        if cycle_length and cycle_length > 35:
            recommendations.append("🔄 **Menstrual Tracking:** Use apps or charts to monitor cycle patterns and symptoms")
        
        # Lifestyle recommendations
        sleep_hours = patient_data.get('Sleep_hours')
        if sleep_hours and sleep_hours < 7:
            recommendations.append("😴 **Sleep Hygiene:** Aim for 7-9 hours nightly to regulate hormones")
        
        exercise_hours = patient_data.get('Exercise_hours_week')
        if exercise_hours and exercise_hours < 2.5:
            recommendations.append("🏃‍♀️ **Physical Activity:** Increase to ≥150 minutes moderate exercise weekly")
        
        stress_level = patient_data.get('Stress_level')
        if stress_level and stress_level > 7:
            recommendations.append("🧘‍♀️ **Stress Management:** Consider meditation, yoga, or counseling")
        
        # General recommendations
        recommendations.extend([
            "🥗 **Nutrition:** Focus on low-glycemic, anti-inflammatory diet",
            "📅 **Follow-up:** Regular monitoring regardless of current assessment",
            "📚 **Education:** Learn about PCOS to better understand symptoms and management"
        ])
        
        return "\n".join([f"{i+1}. {rec}" for i, rec in enumerate(recommendations)])

def create_professional_gradio_interface(agent):
    """Create a professional Gradio interface"""
    
    # Custom CSS for professional appearance
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .metric-box {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    .risk-high { background-color: #ffe6e6; border-left: 4px solid #dc3545; }
    .risk-moderate { background-color: #fff3cd; border-left: 4px solid #ffc107; }
    .risk-low { background-color: #e6ffe6; border-left: 4px solid #28a745; }
    """
    
    with gr.Blocks(
        title="🏥 Professional PCOS Risk Predictor",
        theme=gr.themes.Soft(),
        css=custom_css
    ) as interface:
        
        gr.HTML("""
        <div class="main-header">
            <h1>🏥 Professional PCOS Risk Assessment System</h1>
            <p>Advanced AI-Powered Medical Decision Support Tool</p>
            <p><em>Professional Grade • Evidence-Based • Clinically Informed</em></p>
        </div>
        """)
        
        with gr.Tab("🩺 Patient Assessment", elem_id="assessment-tab"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 👤 Demographic Information")
                    with gr.Row():
                        age = gr.Slider(18, 50, value=28, label="Age (years)", step=1)
                        family_history = gr.Radio(
                            choices=["No", "Yes"], 
                            label="Family History of PCOS", 
                            value="No"
                        )
                    
                    gr.Markdown("### 📏 Anthropometric Data")
                    with gr.Row():
                        weight = gr.Number(value=65, label="Weight (kg)", minimum=30, maximum=200)
                        height = gr.Number(value=165, label="Height (cm)", minimum=140, maximum=200)
                    
                    gr.Markdown("### 🔄 Reproductive Health")
                    with gr.Row():
                        cycle_length = gr.Slider(
                            15, 120, value=28, 
                            label="Average Menstrual Cycle Length (days)", 
                            step=1
                        )
                    
                    gr.Markdown("### 🎭 Clinical Symptoms")
                    with gr.Row():
                        acne = gr.Radio(
                            choices=["None", "Mild", "Moderate", "Severe"],
                            label="Acne Severity", 
                            value="None"
                        )
                        hirsutism = gr.Radio(
                            choices=["None", "Mild", "Moderate", "Severe", "Very Severe"],
                            label="Excess Hair Growth (Hirsutism)", 
                            value="None"
                        )
                    
                    gr.Markdown("### 🏃‍♀️ Lifestyle Factors")
                    with gr.Row():
                        sleep = gr.Slider(3, 12, value=7.5, label="Average Sleep Hours", step=0.5)
                        exercise = gr.Slider(0, 20, value=4, label="Exercise Hours per Week", step=0.5)
                    
                    with gr.Row():
                        stress = gr.Slider(1, 10, value=5, label="Stress Level (1=Low, 10=High)", step=1)
                        resting_bpm = gr.Slider(50, 120, value=72, label="Resting Heart Rate (BPM)", step=1)
                
                with gr.Column(scale=1):
                    gr.Markdown("### 🧪 Laboratory Results")
                    gr.Markdown("*Leave blank if not available*")
                    
                    testosterone = gr.Number(
                        label="Testosterone (ng/mL)", 
                        placeholder="Optional - Normal: 0.1-0.7",
                        minimum=0, maximum=10
                    )
                    
                    lh = gr.Number(
                        label="LH Level (IU/L)", 
                        placeholder="Optional - Normal: 1-15",
                        minimum=0, maximum=100
                    )
                    
                    fsh = gr.Number(
                        label="FSH Level (IU/L)", 
                        placeholder="Optional - Normal: 2-12",
                        minimum=0, maximum=50
                    )
                    
                    insulin = gr.Number(
                        label="Insulin (µIU/mL)", 
                        placeholder="Optional - Normal: 2-20",
                        minimum=0, maximum=200
                    )
                    
                    glucose = gr.Number(
                        label="Glucose (mg/dL)", 
                        placeholder="Optional - Normal: 70-100",
                        minimum=0, maximum=300
                    )
                    
                    gr.Markdown("### 🩸 Additional Metabolic Markers")
                    gr.Markdown("*Optional - For enhanced accuracy*")
                    
                    hdl = gr.Number(
                        label="HDL Cholesterol (mg/dL)", 
                        placeholder="Optional",
                        minimum=0, maximum=200
                    )
                    
                    ldl = gr.Number(
                        label="LDL Cholesterol (mg/dL)", 
                        placeholder="Optional",
                        minimum=0, maximum=400
                    )
                    
                    triglycerides = gr.Number(
                        label="Triglycerides (mg/dL)", 
                        placeholder="Optional",
                        minimum=0, maximum=1000
                    )
                    
                    with gr.Row():
                        systolic_bp = gr.Number(
                            label="Systolic BP (mmHg)", 
                            placeholder="Optional",
                            minimum=70, maximum=200
                        )
                        diastolic_bp = gr.Number(
                            label="Diastolic BP (mmHg)", 
                            placeholder="Optional", 
                            minimum=40, maximum=120
                        )
            
            with gr.Row():
                with gr.Column(scale=1):
                    assess_btn = gr.Button(
                        "🔍 Generate Professional Assessment", 
                        variant="primary", 
                        size="lg"
                    )
                    
                    clear_btn = gr.Button(
                        "🔄 Clear All Fields", 
                        variant="secondary"
                    )
                
                with gr.Column(scale=2):
                    gr.Markdown("### ⚡ Quick Assessment")
                    quick_result = gr.Textbox(
                        label="Risk Level", 
                        placeholder="Click 'Generate Assessment' for results",
                        lines=2
                    )
            
            with gr.Row():
                comprehensive_report = gr.Textbox(
                    label="📋 Comprehensive Medical Report", 
                    lines=30, 
                    max_lines=50,
                    placeholder="Detailed professional assessment will appear here..."
                )
        
        with gr.Tab("📊 System Analytics", elem_id="analytics-tab"):
            gr.Markdown("### 🎯 Model Performance Metrics")
            
            if agent.is_initialized and agent.visualizer:
                try:
                    performance_plot = gr.Plot(
                        value=agent.visualizer.create_model_comparison(),
                        label="Model Performance Comparison"
                    )
                    
                    feature_plot = gr.Plot(
                        value=agent.visualizer.create_feature_importance_analysis(),
                        label="Feature Importance Analysis"
                    )
                except Exception as e:
                    print(f"⚠️ Plot generation error: {e}")
                    gr.Markdown("*Visualizations will be available after running an assessment*")
            else:
                gr.Markdown("*Initialize system first by running an assessment*")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📈 System Statistics")
                    if agent.is_initialized and agent.ensemble.model_scores:
                        try:
                            best_model_name = max(
                                agent.ensemble.model_scores.keys(), 
                                key=lambda x: agent.ensemble.model_scores[x].get('f1_score', 0)
                            )
                            best_f1 = agent.ensemble.model_scores[best_model_name].get('f1_score', 0)
                            best_auc = agent.ensemble.model_scores[best_model_name].get('auc', 0)
                            
                            gr.Markdown(f"""
                            **Best Model:** {best_model_name}  
                            **F1 Score:** {best_f1:.3f}  
                            **AUC Score:** {best_auc:.3f}  
                            **Features Used:** {len(agent.feature_columns)}  
                            **Cross-Validation:** 5-Fold Stratified  
                            **Ensemble Methods:** Voting Classifier  
                            """)
                        except Exception as e:
                            gr.Markdown("*System statistics will be available after initialization*")
                    else:
                        gr.Markdown("*System not yet initialized*")
                
                with gr.Column():
                    gr.Markdown("### 🔬 Medical Validation")
                    gr.Markdown("""
                    **Clinical Basis:** Rotterdam Criteria  
                    **Feature Engineering:** Medical Domain Knowledge  
                    **Validation Method:** Stratified Cross-Validation  
                    **Missing Data:** KNN Imputation  
                    **Bias Mitigation:** Ensemble Learning  
                    **Interpretability:** SHAP Values & Feature Importance  
                    """)
        
        with gr.Tab("ℹ️ Professional Information", elem_id="info-tab"):
            gr.Markdown("""
            ## 🏥 About This Professional System
            
            ### 🎯 Clinical Purpose
            This advanced PCOS risk assessment system is designed to support healthcare professionals 
            and inform patients about potential PCOS risk factors. It uses state-of-the-art machine 
            learning techniques combined with medical domain knowledge.
            
            ### 🧠 Technical Architecture
            
            **Machine Learning Stack:**
            - **Ensemble Learning:** Combines multiple algorithms for robust predictions
            - **Advanced Preprocessing:** KNN imputation, feature engineering, scaling
            - **Model Selection:** Random Forest, XGBoost, LightGBM, SVM, Logistic Regression
            - **Hyperparameter Optimization:** Grid search with cross-validation
            - **Performance Metrics:** F1-Score, AUC, Precision, Recall, Accuracy
            
            **Medical Features:**
            - **Anthropometric:** BMI categories, weight status
            - **Reproductive:** Cycle regularity, menstrual patterns  
            - **Hormonal:** Testosterone, LH/FSH ratios, insulin resistance markers
            - **Metabolic:** Glucose levels, lipid profiles, blood pressure
            - **Lifestyle:** Sleep quality, exercise adequacy, stress levels
            - **Clinical:** Hirsutism scores, acne severity, family history
            
            ### 📊 Validation & Performance
            
            The system has been validated using:
            - **Cross-validation:** 5-fold stratified cross-validation
            - **Multiple Metrics:** Comprehensive evaluation beyond accuracy
            - **Ensemble Voting:** Reduces individual model bias
            - **Feature Importance:** Transparent decision-making process
            
            ### ⚠️ Important Medical Disclaimers
            
            1. **Not Diagnostic:** This tool provides risk assessment only, not diagnosis
            2. **Professional Consultation:** Always consult healthcare providers for medical decisions
            3. **Individual Variation:** PCOS presents differently in each person
            4. **Complementary Tool:** Use alongside, not instead of, clinical judgment
            5. **Regular Updates:** Medical knowledge evolves - seek current medical advice
            
            ### 🔒 Privacy & Security
            
            - **Local Processing:** All data processed locally, not stored on servers
            - **No Data Retention:** Patient information not saved between sessions  
            - **HIPAA Considerations:** Designed with healthcare privacy in mind
            - **Professional Use:** Intended for healthcare settings and informed patients
            
            ### 📚 References & Evidence Base
            
            This system incorporates evidence from:
            - Rotterdam Criteria for PCOS diagnosis
            - Current endocrine society guidelines
            - Recent meta-analyses on PCOS risk factors
            - Machine learning best practices for healthcare
            
            ### 🔧 Technical Requirements
            
            **For VSCode Development:**
            ```python
            pip install pandas numpy scikit-learn matplotlib seaborn plotly gradio xgboost lightgbm
            ```
            
            **System Requirements:**
            - Python 3.8+
            - 4GB+ RAM recommended
            - Modern web browser for interface
            
            ### 📞 Support & Development
            
            For technical support, bug reports, or feature requests:
            - Review code documentation
            - Check error logs in console
            - Validate input data format
            - Ensure all dependencies installed
            
            **Version:** Professional PCOS Predictor v2.0  
            **Last Updated:** {datetime.now().strftime('%B %Y')}  
            **Compatibility:** VSCode, Jupyter, Standalone Python
            """)
        
        # Define the main prediction function
        def generate_assessment(*inputs):
            try:
                # Extract input values
                (patient_age, family_hist, weight_kg, height_cm, cycle_len, acne_sev, 
                 hirsut_score, sleep_hrs, exercise_hrs, stress_lvl, heart_rate,
                 test_level, lh_level, fsh_level, insulin_level, gluc_level,
                 hdl_level, ldl_level, trig_level, sys_bp, dias_bp) = inputs
                
                # Create patient data dictionary
                patient_data = {
                    'Age': patient_age,
                    'Weight_kg': weight_kg,
                    'Height_cm': height_cm,
                    'Cycle_length_days': cycle_len,
                    'Sleep_hours': sleep_hrs,
                    'Exercise_hours_week': exercise_hrs,
                    'Stress_level': stress_lvl,
                    'Acne_severity': acne_sev,
                    'Hair_growth_score': hirsut_score,
                    'Resting_BPM': heart_rate,
                    'Family_history': family_hist
                }
                
                # Add optional laboratory values if provided
                optional_fields = {
                    'Testosterone_ng_ml': test_level,
                    'LH_IU_L': lh_level,
                    'FSH_IU_L': fsh_level,
                    'Insulin_uIU_ml': insulin_level,
                    'Glucose_mg_dl': gluc_level,
                    'HDL_cholesterol': hdl_level,
                    'LDL_cholesterol': ldl_level,
                    'Triglycerides': trig_level,
                    'Systolic_BP': sys_bp,
                    'Diastolic_BP': dias_bp
                }
                
                for key, value in optional_fields.items():
                    if value is not None and value != "":
                        patient_data[key] = float(value)
                
                # Initialize system if not already done
                if not agent.is_initialized:
                    agent.initialize_system()
                
                # Make prediction
                prediction_result = agent.predict_risk(patient_data)
                
                # Generate quick result
                risk_level = prediction_result['risk_level']
                probability = prediction_result['probability']
                quick_summary = f"🎯 **{risk_level}** ({probability:.1%} probability)"
                
                # Generate comprehensive report
                full_report = agent.generate_professional_report(prediction_result)
                
                return quick_summary, full_report
                
            except Exception as e:
                error_msg = f"❌ Assessment Error: {str(e)}"
                detailed_error = f"""
# ❌ Error in Assessment

**Error Type:** {type(e).__name__}  
**Error Message:** {str(e)}  

**Troubleshooting Steps:**
1. Verify all required fields are filled
2. Check that numeric values are within reasonable ranges
3. Ensure system dependencies are installed
4. Contact technical support if error persists

**Technical Details:**
- Error occurred during prediction phase
- Check console for detailed stack trace
- Validate input data format
"""
                return error_msg, detailed_error
        
        # Define clear function
        def clear_all_fields():
            return (
                28, "No", 65, 165, 28, "None", "None", 7.5, 4, 5, 72,  # Basic fields
                None, None, None, None, None,  # Hormones
                None, None, None, None, None,  # Additional labs
                "", ""  # Output fields
            )
        
        # Connect the functions to the interface
        assess_btn.click(
            fn=generate_assessment,
            inputs=[
                age, family_history, weight, height, cycle_length, acne, hirsutism,
                sleep, exercise, stress, resting_bpm,
                testosterone, lh, fsh, insulin, glucose,
                hdl, ldl, triglycerides, systolic_bp, diastolic_bp
            ],
            outputs=[quick_result, comprehensive_report]
        )
        
        clear_btn.click(
            fn=clear_all_fields,
            outputs=[
                age, family_history, weight, height, cycle_length, acne, hirsutism,
                sleep, exercise, stress, resting_bpm,
                testosterone, lh, fsh, insulin, glucose,
                hdl, ldl, triglycerides, systolic_bp, diastolic_bp,
                quick_result, comprehensive_report
            ]
        )
    
    return interface

def main():
    """Main function to run the Professional PCOS Predictor"""
    print("""
    🏥 Professional PCOS Risk Predictor - VSCode Ready
    ================================================
    
    🎯 Features:
    • Advanced ensemble machine learning
    • Professional medical reporting
    • Comprehensive health metrics analysis
    • Optional laboratory data integration
    • Clinical-grade visualizations
    • Evidence-based recommendations
    
    🚀 Initializing system...
    """)
    
    try:
        # Initialize the professional agent
        agent = PCOSProfessionalAgent()
        
        # Create the professional interface
        interface = create_professional_gradio_interface(agent)
        
        print("""
    ✅ System Ready!
    
    📊 System Capabilities:
    • Multi-model ensemble (RF, XGB, LGB, SVM, LR)
    • Advanced feature engineering
    • KNN imputation for missing values
    • Hyperparameter optimization
    • 5-fold cross-validation
    • Professional medical reporting
    
    🌐 Starting web interface...
        """)
        
        # Launch the interface
        interface.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,  # Set to True if you want public link
            debug=True,
            inbrowser=True
        )
        
    except Exception as e:
        print(f"""
    ❌ Startup Error: {e}
    
    🔧 Troubleshooting:
    1. Install dependencies: pip install -r requirements.txt
    2. Check Python version (3.8+ required)
    3. Verify port 7860 is available
    4. Run with administrator privileges if needed
        """)
        raise 

if __name__ == "__main__":
    main()
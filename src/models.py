import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold
)

from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from .features import create_heart_disease_features

__all__ = [
    'CascadedLogisticRegression',
    'ThresholdedCascadedLogisticRegression',
    'generate_submission',
    'run_complete_pipeline',
    'run_model_comparison'
]

class CascadedLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Two-step hierarchical model:
    1. Binary model: 0 (Healthy) vs (1, 2, 3, 4) (Diseased).
    2. Multiclass model: 1 vs 2 vs 3 vs 4 (trained only on diseased samples).
    """
    def __init__(self, C_zero=1.0, C_multi=1.0, solver='lbfgs', max_iter=1000, random_state=42):
        self.C_zero = C_zero
        self.C_multi = C_multi
        self.solver = solver
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        y = np.asarray(y)
        y_binary = (y > 0).astype(int)
        self.binary_model_ = LogisticRegression(
            C=self.C_zero, solver=self.solver, 
            max_iter=self.max_iter, random_state=self.random_state
        )
        self.binary_model_.fit(X, y_binary)

        mask_pos = y > 0
        if np.any(mask_pos):
            self.multi_model_ = LogisticRegression(
                C=self.C_multi, solver=self.solver, 
                max_iter=self.max_iter, random_state=self.random_state
            )
            self.multi_model_.fit(X[mask_pos], y[mask_pos])
        return self

    def predict(self, X):
        y_bin_pred = self.binary_model_.predict(X)
        y_pred = np.zeros(len(X), dtype=int)
        
        idx_pos = np.where(y_bin_pred == 1)[0]
        if len(idx_pos) > 0:
            y_pred[idx_pos] = self.multi_model_.predict(X[idx_pos])
        return y_pred

class ThresholdedCascadedLogisticRegression(CascadedLogisticRegression):
    """
    Hierarchical model that uses a confidence threshold to prioritize the 'Healthy' (0) class.
    """
    def __init__(self, C_zero=1.0, C_multi=1.0, solver='lbfgs', max_iter=1000, 
                 random_state=42, zero_threshold=0.7):
        super().__init__(C_zero, C_multi, solver, max_iter, random_state)
        self.zero_threshold = zero_threshold

    def predict(self, X):
        proba_bin = self.binary_model_.predict_proba(X)
        idx_zero = np.where(self.binary_model_.classes_ == 0)[0][0]
        p_zero = proba_bin[:, idx_zero]

        y_pred = super().predict(X)
        
        mask_force_zero = p_zero >= self.zero_threshold
        y_pred[mask_force_zero] = 0
        return y_pred

def generate_submission(predictions, filename="./data/processed/submission.csv"):
    """
    Trains the pipeline on the full training set and exports CSV predictions.
    """

    submission = pd.DataFrame({'id': range(len(predictions)), 'label': predictions})
    submission.to_csv(filename, index=False)
    print(f"Submission saved to {filename}")
    return submission

def run_complete_pipeline(
    train_df,
    test_df,
    impute_minus_nine_func,
    impute_question_func,
    label_vars,
    onehot_vars,
    param_grid,
    use_feature_engineering=False,
    cv_folds=5,
    submission=False
):
    """
    Execute the complete ML pipeline: imputation, feature engineering, training, and evaluation.
    """
    train_imputed, test_imputed = impute_minus_nine_func(train_df.copy(), test_df.copy())
    
    train_imputed, test_imputed = impute_question_func(train_imputed, test_imputed)

    if use_feature_engineering:
        train_imputed = create_heart_disease_features(train_imputed)
        test_imputed = create_heart_disease_features(test_imputed)
    
    X_train = train_imputed.drop('label', axis=1)
    y_train = train_imputed['label']
    X_test = test_imputed

    for col in label_vars:
        if col in X_train.columns:
            le = LabelEncoder()
            all_values = pd.concat([X_train[col], X_test[col]]).unique()
            le.fit(all_values)
            X_train[col] = le.transform(X_train[col])
            X_test[col] = le.transform(X_test[col])
    
    for col in onehot_vars:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype(str)
            X_test[col] = X_test[col].astype(str)
    
    X_train = pd.get_dummies(X_train, columns=onehot_vars, prefix=onehot_vars)
    X_test = pd.get_dummies(X_test, columns=onehot_vars, prefix=onehot_vars)
    
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    X_train = X_train.reindex(columns=X_test.columns, fill_value=0)
    X_test = X_test[X_train.columns]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(max_iter=10000, random_state=42)
    
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=skf,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train_scaled, y_train)

    best_index = grid_search.best_index_
    mean_accuracy = grid_search.cv_results_['mean_test_score'][best_index]
    std_accuracy = grid_search.cv_results_['std_test_score'][best_index]
    
    model = grid_search.best_estimator_

    predictions = model.predict(X_test_scaled)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Mean CV Accuracy: {mean_accuracy:.4f}")
    print(f"Performance Range (CV): {mean_accuracy:.4f} +/- {std_accuracy:.4f}")
    print()

    if submission:
        generate_submission(predictions)
        return X_train_scaled, y_train, model, X_train.columns, scaler, grid_search, grid_search.best_score_
    else:
        return X_train_scaled, y_train, model, X_train.columns, scaler, grid_search, grid_search.best_score_

def run_model_comparison(X, y, cv_folds=5):
    """
    Runs a competitive leaderboard comparing multiple algorithms via GridSearch.
    """
    model_params = {
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'params': {'C': [0.1, 1, 10], 'solver': ['lbfgs', 'liblinear']}
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}
        },
        'SVM': {
            'model': SVC(random_state=42, probability=True),
            'params': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {'n_estimators': [100], 'learning_rate': [0.01, 0.1]}
        }
    }

    scores = []
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    for model_name, mp in model_params.items():
        clf = GridSearchCV(mp['model'], mp['params'], cv=skf, scoring='accuracy', n_jobs=-1)
        clf.fit(X, y)
        scores.append({
            'model': model_name, 
            'best_score': clf.best_score_, 
            'best_params': clf.best_params_
        })

    return pd.DataFrame(scores).sort_values(by='best_score', ascending=False)
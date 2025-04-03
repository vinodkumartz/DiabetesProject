from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import numpy as np

def train_and_evaluate_models(X, y, cat_features, num_features):
    """
    Train and evaluate multiple models for diabetes readmission prediction.
    Returns:
        results (dict): Model evaluation metrics.
        preprocessor (ColumnTransformer): Preprocessing pipeline.
        trained_models (dict): Dictionary containing trained models.
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

    # Create preprocessing pipeline
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, num_features),
            ('cat', categorical_transformer, cat_features)
        ])

    # Preprocess data
    print("Preprocessing data...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print(f"Processed training data shape: {X_train_processed.shape}")

    # Create a dictionary to store model performances
    results = {}
    trained_models = {}  # ✅ Store trained models

    # 1. Logistic Regression
    print("\nTraining Logistic Regression...")
    log_reg = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    log_reg.fit(X_train_processed, y_train)

    y_pred_lr = log_reg.predict(X_test_processed)
    y_prob_lr = log_reg.predict_proba(X_test_processed)[:, 1]

    results['Logistic Regression'] = {
        'accuracy': accuracy_score(y_test, y_pred_lr),
        'f1_score': f1_score(y_test, y_pred_lr),
        'roc_auc': roc_auc_score(y_test, y_prob_lr),
        'confusion_matrix': confusion_matrix(y_test, y_pred_lr),
        'classification_report': classification_report(y_test, y_pred_lr)
    }

    trained_models['Logistic Regression'] = log_reg  # ✅ Save trained model

    print(f"Logistic Regression - Accuracy: {results['Logistic Regression']['accuracy']:.4f}, "
          f"F1: {results['Logistic Regression']['f1_score']:.4f}, "
          f"ROC AUC: {results['Logistic Regression']['roc_auc']:.4f}")

    # 2. Decision Tree
    print("\nTraining Decision Tree...")
    dt = DecisionTreeClassifier(random_state=42, class_weight='balanced')

    param_grid_dt = {
        'max_depth': [5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_dt = GridSearchCV(
        dt, param_grid_dt, cv=StratifiedKFold(5),
        scoring='roc_auc', n_jobs=-1
    )

    grid_dt.fit(X_train_processed, y_train)
    best_dt = grid_dt.best_estimator_

    y_pred_dt = best_dt.predict(X_test_processed)
    y_prob_dt = best_dt.predict_proba(X_test_processed)[:, 1]

    results['Decision Tree'] = {
        'best_params': grid_dt.best_params_,
        'accuracy': accuracy_score(y_test, y_pred_dt),
        'f1_score': f1_score(y_test, y_pred_dt),
        'roc_auc': roc_auc_score(y_test, y_prob_dt),
        'confusion_matrix': confusion_matrix(y_test, y_pred_dt),
        'classification_report': classification_report(y_test, y_pred_dt)
    }

    trained_models['Decision Tree'] = best_dt  # ✅ Save trained model

    print(f"Decision Tree - Accuracy: {results['Decision Tree']['accuracy']:.4f}, "
          f"F1: {results['Decision Tree']['f1_score']:.4f}, "
          f"ROC AUC: {results['Decision Tree']['roc_auc']:.4f}")
    print(f"Best parameters: {results['Decision Tree']['best_params']}")

    # ✅ Return results, preprocessor, and trained models
    return results, preprocessor, trained_models

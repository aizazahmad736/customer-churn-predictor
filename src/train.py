"""
ChurnGuard AI - Model Training & Evaluation Engine
Trains multiple competitive machine learning classifiers, calculates
comprehensive performance metrics, and serializes production assets.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

from .config import (
    DATASET_PATH,
    MODEL_PATH,
    METRICS_PATH,
    FEATURE_IMPORTANCE_PATH,
    MODEL_COMPARISON_PATH,
    TARGET_COLUMN,
    ID_COLUMN,
    ALL_FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES
)
from .preprocessing import clean_dataset, build_preprocessor, get_feature_names_from_preprocessor

def train_and_evaluate_models():
    """Executes the full model training and selection workflow."""
    print("=" * 65)
    print("  ChurnGuard AI - Enterprise Model Training Pipeline")
    print("=" * 65)

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}. Please run data_generator.py first.")

    print(f"\n[1/5] Loading and preparing dataset from: {DATASET_PATH}")
    df_raw = pd.read_csv(DATASET_PATH)
    df_clean = clean_dataset(df_raw)

    # Encode Target
    y = (df_clean[TARGET_COLUMN] == "Yes").astype(int)
    X = df_clean[ALL_FEATURE_COLUMNS].copy()

    print(f"Total samples: {len(X):,} | Features: {X.shape[1]}")
    print(f"Target distribution: {y.value_counts().to_dict()} (Churn rate: {y.mean():.1%})")

    # Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train size: {len(X_train):,} | Test size: {len(X_test):,}")

    # Model candidates
    candidate_models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            C=1.0,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_split=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=4,
            subsample=0.85,
            random_state=42
        )
    }

    print("\n[2/5] Training and cross-evaluating benchmark models...")
    comparison_results = {}
    fitted_pipelines = {}

    for model_name, classifier in candidate_models.items():
        preprocessor = build_preprocessor()
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred))
        auc = float(roc_auc_score(y_test, y_proba))

        comparison_results[model_name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4)
        }
        fitted_pipelines[model_name] = pipeline

        print(f"  -> {model_name:<20} | ROC-AUC: {auc:.4f} | F1: {f1:.4f} | Acc: {acc:.4f} | Recall: {rec:.4f}")

    # Select best model based on combination of ROC-AUC and F1-Score
    best_model_name = max(
        comparison_results.keys(),
        key=lambda m: (comparison_results[m]["roc_auc"] * 0.6 + comparison_results[m]["f1_score"] * 0.4)
    )
    best_pipeline = fitted_pipelines[best_model_name]
    print(f"\n[3/5] Optimal Production Model Selected: >>> {best_model_name} <<<")

    # Detailed metrics for best model
    best_y_pred = best_pipeline.predict(X_test)
    best_y_proba = best_pipeline.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, best_y_pred)
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    fpr, tpr, _ = roc_curve(y_test, best_y_proba)
    p_curve, r_curve, _ = precision_recall_curve(y_test, best_y_proba)

    # Downsample curve points for lightweight JSON serialization
    sample_step = max(1, len(fpr) // 50)
    detailed_metrics = {
        "best_model": best_model_name,
        "test_samples": int(len(y_test)),
        "accuracy": comparison_results[best_model_name]["accuracy"],
        "precision": comparison_results[best_model_name]["precision"],
        "recall": comparison_results[best_model_name]["recall"],
        "f1_score": comparison_results[best_model_name]["f1_score"],
        "roc_auc": comparison_results[best_model_name]["roc_auc"],
        "confusion_matrix": {
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp
        },
        "roc_curve": {
            "fpr": [round(float(v), 4) for v in fpr[::sample_step]],
            "tpr": [round(float(v), 4) for v in tpr[::sample_step]]
        }
    }

    print("\n[4/5] Computing global feature importance rankings...")
    preprocessor = best_pipeline.named_steps["preprocessor"]
    clf = best_pipeline.named_steps["classifier"]
    feature_names = get_feature_names_from_preprocessor(preprocessor)

    if hasattr(clf, "feature_importances_"):
        raw_importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        raw_importances = np.abs(clf.coef_[0])
    else:
        raw_importances = np.ones(len(feature_names))

    # Normalize to percentage
    sum_imp = float(np.sum(raw_importances))
    norm_importances = (raw_importances / (sum_imp if sum_imp > 0 else 1.0)) * 100.0

    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance": norm_importances
    }).sort_values(by="importance", ascending=False)

    top_features = fi_df.head(20).to_dict(orient="records")

    print("\nTop 7 Global Predictive Factors:")
    for i, row in enumerate(top_features[:7], 1):
        clean_feat = row['feature'].replace('cat__', '').replace('num__', '')
        print(f"  {i}. {clean_feat:<35} : {row['importance']:.2f}%")

    print(f"\n[5/5] Serializing model and metadata assets to {os.path.dirname(MODEL_PATH)}...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    # Save Pipeline
    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"  -> Model Pipeline: {MODEL_PATH}")

    # Save Metrics
    with open(METRICS_PATH, "w") as f:
        json.dump(detailed_metrics, f, indent=2)
    print(f"  -> Best Metrics: {METRICS_PATH}")

    # Save Comparison
    with open(MODEL_COMPARISON_PATH, "w") as f:
        json.dump(comparison_results, f, indent=2)
    print(f"  -> Model Comparison: {MODEL_COMPARISON_PATH}")

    # Save Feature Importance
    with open(FEATURE_IMPORTANCE_PATH, "w") as f:
        json.dump(top_features, f, indent=2)
    print(f"  -> Feature Importance: {FEATURE_IMPORTANCE_PATH}")

    print("\n" + "=" * 65)
    print("  Training Complete! Production Model Ready for Deployment.")
    print("=" * 65)

if __name__ == "__main__":
    train_and_evaluate_models()

"""
train_model.py – Cognitive Drift Detection ML Training Pipeline for LearnFlow AI.

This standalone script:
  1. Loads the synthetic student dataset.
  2. Cleans and engineers features.
  3. Trains three classifiers: Random Forest (primary), Decision Tree, Logistic Regression.
  4. Evaluates each with accuracy, precision, recall, F1-score, and confusion matrix.
  5. Shows feature importance for Random Forest.
  6. Saves the best model as models/cognitive_drift_model.pkl.

Usage:
    python train_model.py

Run this BEFORE starting app.py for the first time.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib

# Ensure UTF-8 output encoding on Windows terminals if supported
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, 'dataset', 'student_learning_data.csv')
MODEL_DIR   = os.path.join(BASE_DIR, 'models')
MODEL_PATH  = os.path.join(MODEL_DIR, 'cognitive_drift_model.pkl')

os.makedirs(MODEL_DIR, exist_ok=True)

LABEL_MAP = {0: 'Stable', 1: 'Mild Drift', 2: 'High Drift'}

# ─────────────────────────────────────────────
# Step 1 – Load Dataset
# ─────────────────────────────────────────────

print("=" * 60)
print("  LearnFlow AI - Cognitive Drift Detection Model Trainer")
print("=" * 60)
print()
print("[Step 1] Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"  Loaded {len(df)} records, {df.shape[1]} columns.")
print(f"  Columns: {list(df.columns)}")
print()

# ─────────────────────────────────────────────
# Step 2 – Data Cleaning
# ─────────────────────────────────────────────

print("[Step 2] Data cleaning...")
print(f"  Missing values before cleaning:\n{df.isnull().sum()}")
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
print(f"  Clean dataset: {len(df)} records.")
print()

# ─────────────────────────────────────────────
# Step 3 – Feature Engineering
# ─────────────────────────────────────────────

print("[Step 3] Feature check...")
print(f"  Using base features only - no derived features added.")
print("  (Ensures consistency with the live predict_drift() pipeline.)")
print()

# ─────────────────────────────────────────────
# Step 4 – Feature Selection
# ─────────────────────────────────────────────

FEATURES = [
    'quiz_score', 'assignment_score', 'attendance', 'study_hours',
    'response_time', 'login_frequency', 'previous_performance',
    'engagement_level', 'learning_consistency',
]
TARGET = 'drift_label'

print("[Step 4] Feature selection...")
print(f"  Selected features: {FEATURES}")
print()

X = df[FEATURES].values
y = df[TARGET].values

# Label distribution
print("  Drift label distribution:")
for label, name in LABEL_MAP.items():
    count = (y == label).sum()
    print(f"    {name} ({label}): {count} records")
print()

# ─────────────────────────────────────────────
# Step 5 – Train / Test Split
# ─────────────────────────────────────────────

print("[Step 5] Train/Test split (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Training samples: {len(X_train)}")
print(f"  Testing  samples: {len(X_test)}")
print()

# Scale features for Logistic Regression
scaler  = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# Step 6 – Model Training
# ─────────────────────────────────────────────

print("[Step 6] Training models...")
print()

models = {
    'Random Forest':      RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5),
    'Decision Tree':      DecisionTreeClassifier(random_state=42, max_depth=4),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"  Training {name}...")
    if name == 'Logistic Regression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    results[name] = {
        'model':    model,
        'accuracy': acc,
        'cv_mean':  cv_scores.mean(),
        'y_pred':   y_pred
    }

    print(f"    Accuracy:  {acc * 100:.1f}%")
    print(f"    Precision: {prec * 100:.1f}%")
    print(f"    Recall:    {rec * 100:.1f}%")
    print(f"    F1-Score:  {f1 * 100:.1f}%")
    print(f"    CV Mean:   {cv_scores.mean() * 100:.1f}% (+/- {cv_scores.std() * 100:.1f}%)")
    print()

# ─────────────────────────────────────────────
# Step 7 – Model Evaluation (Best Model Details)
# ─────────────────────────────────────────────

best_name   = max(results, key=lambda k: results[k]['accuracy'])
best_result = results[best_name]
best_model  = best_result['model']
y_pred_best = best_result['y_pred']

print("=" * 60)
print(f"[Step 7] Best model -> {best_name}")
print("=" * 60)
print()
print("  Classification Report:")
print(classification_report(
    y_test, y_pred_best,
    target_names=[LABEL_MAP[i] for i in sorted(LABEL_MAP)],
    zero_division=0
))

print("  Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred_best)
labels = [LABEL_MAP[i] for i in sorted(LABEL_MAP)]
print(f"  {'':20s}" + "  ".join(f"{l:12s}" for l in labels))
for i, row in enumerate(cm):
    print(f"  {labels[i]:20s}" + "  ".join(f"{v:12d}" for v in row))
print()

# ─────────────────────────────────────────────
# Step 8 – Feature Importance (Random Forest)
# ─────────────────────────────────────────────

if best_name == 'Random Forest':
    rf_model = best_model
else:
    rf_model = results['Random Forest']['model']

print("[Step 8] Feature Importance (Random Forest):")
importances = rf_model.feature_importances_
sorted_idx  = np.argsort(importances)[::-1]
for i in sorted_idx:
    bar = '#' * int(importances[i] * 40)
    print(f"  {FEATURES[i]:25s} {importances[i]:.4f}  {bar}")
print()

# ─────────────────────────────────────────────
# Step 9 – Save Best Model
# ─────────────────────────────────────────────

# Always save the Random Forest model (primary model as per project spec)
print(f"[Step 9] Saving Random Forest model -> {MODEL_PATH}")
joblib.dump(results['Random Forest']['model'], MODEL_PATH)
print("  Model saved successfully!")
print()
print("=" * 60)
print("  Training complete. You can now start the Flask app:")
print("  $ python app.py")
print("=" * 60)

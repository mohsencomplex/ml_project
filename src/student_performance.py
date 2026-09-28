import pandas as pd
import os
import joblib

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV
)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

DATA_PATH = "data/student_performance/files/student-mat.csv"
MODEL_PATH = "models/student_performance_model.joblib"

df = pd.read_csv(DATA_PATH, sep=";")

X = df.drop(columns=["G1", "G2", "G3"])
y = (df["G3"] >= 10).astype(int)

print("=" * 50)
print("STUDENT PERFORMANCE PREDICTION")
print("=" * 50)

print("\nDataset shape:", df.shape)
print("Features shape:", X.shape)
print("Target shape:", y.shape)

print("\nTarget distribution:")
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

categorical_features = X.select_dtypes(
    include=["str", "object"]
).columns

numeric_features = X.select_dtypes(
    exclude=["str", "object"]
).columns

print("\nCategorical features:", len(categorical_features))
print("Numeric features:", len(numeric_features))

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            "passthrough",
            numeric_features
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                random_state=42
            )
        )
    ]
)

param_distributions = {
    "classifier__n_estimators": [100, 200, 300, 500],
    "classifier__max_depth": [None, 5, 10, 15, 20],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4],
    "classifier__max_features": ["sqrt", "log2", None],
    "classifier__class_weight": [None, "balanced"]
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("\n" + "=" * 50)
print("HYPERPARAMETER TUNING")
print("=" * 50)

search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,
    n_iter=30,
    scoring="f1",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

search.fit(X_train, y_train)

print("\nBest Parameters:")
print(search.best_params_)

print(f"\nBest Cross-Validation F1: {search.best_score_:.2%}")

best_model = search.best_estimator_

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n" + "=" * 50)
print("TUNED MODEL TEST PERFORMANCE")
print("=" * 50)

print(f"\nAccuracy:  {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall:    {recall:.2%}")
print(f"F1 Score:  {f1:.2%}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

os.makedirs("models", exist_ok=True)

joblib.dump(best_model, MODEL_PATH)

print("\nModel saved successfully!")
print("Model path:", MODEL_PATH)

print("\n" + "=" * 50)
print("FINAL DATA SPLIT")
print("=" * 50)

print("\nTraining set:", X_train.shape)
print("Testing set:", X_test.shape)

print("\n" + "=" * 50)
print("TRAINING COMPLETED")
print("=" * 50)
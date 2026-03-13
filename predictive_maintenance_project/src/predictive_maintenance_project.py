import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance


def load_dataset(file_path: str) -> pd.DataFrame:
    """Load dataset from CSV."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {file_path}. Place your dataset in the same folder as this script "
            "or update the file path."
        )
    return pd.read_csv(path)


TARGET_COLUMN = "failure"
CANDIDATE_NUMERIC_COLUMNS = [
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
    "operating_hours",
    "vibration",
    "pressure",
]
CANDIDATE_CATEGORICAL_COLUMNS = ["machine_type"]


def prepare_features(df: pd.DataFrame):
    """Pick available feature columns from the dataset."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found. "
            "Update TARGET_COLUMN in the script to match your dataset."
        )

    numeric_features = [col for col in CANDIDATE_NUMERIC_COLUMNS if col in df.columns]
    categorical_features = [col for col in CANDIDATE_CATEGORICAL_COLUMNS if col in df.columns]

    if not numeric_features and not categorical_features:
        raise ValueError(
            "No candidate feature columns were found. Update the candidate column lists "
            "to match your dataset."
        )

    X = df[numeric_features + categorical_features].copy()
    y = df[TARGET_COLUMN].copy()

    return X, y, numeric_features, categorical_features


def build_preprocessor(numeric_features, categorical_features):
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def evaluate_model(name: str, model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print(f"\n{'=' * 60}")
    print(f"{name}")
    print(f"{'=' * 60}")
    print("Confusion Matrix")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification Report")
    print(classification_report(y_test, predictions, digits=3))
    print(f"ROC AUC: {roc_auc_score(y_test, probabilities):.3f}")


if __name__ == "__main__":
    data = load_dataset("data/predictive_maintenance.csv")

    X, y, numeric_features, categorical_features = prepare_features(data)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    logistic_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )

    random_forest_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=8,
                    min_samples_split=10,
                    min_samples_leaf=4,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    logistic_model.fit(X_train, y_train)
    random_forest_model.fit(X_train, y_train)

    evaluate_model("Logistic Regression", logistic_model, X_test, y_test)
    evaluate_model("Random Forest", random_forest_model, X_test, y_test)

    print(f"\n{'=' * 60}")
    print("Feature Importance")
    print(f"{'=' * 60}")
    best_model = random_forest_model
    importance = permutation_importance(
        best_model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
        scoring="roc_auc",
    )

    feature_names = list(X.columns)
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": importance.importances_mean[: len(feature_names)],
        }
    ).sort_values("importance_mean", ascending=False)

    print(importance_df.to_string(index=False))
    print("\nProject complete. Use the results to write up your findings for GitHub and your CV.")

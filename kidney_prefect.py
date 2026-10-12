import pandas as pd
import joblib

from prefect import flow, task
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


@task
def load_data():
    print("\nTASK 1: Loading dataset")
    df = pd.read_csv("kidney_disease.csv")

    # Remove unnecessary ID column
    df = df.drop(columns=["id"], errors="ignore")

    # Clean target labels
    df["classification"] = (
        df["classification"].astype(str).str.strip().str.lower()
        .str.replace(r"\s+", "", regex=True)
        .str.replace(".", "", regex=False)
    )

    df = df[df["classification"].isin(["ckd", "notckd"])]
    print("Dataset shape:", df.shape)
    print("Target distribution:\n", df["classification"].value_counts())
    return df


@task
def preprocess_data(df):
    print("\nTASK 2: Preprocessing data")

    X = df.drop(columns=["classification"])
    y = df["classification"].map({"ckd": 1, "notckd": 0})

    # Convert numeric-looking columns to numeric values
    for col in X.columns:
        converted = pd.to_numeric(X[col], errors="coerce")
        if converted.notna().sum() > 0:
            X[col] = converted

    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_cols),
        ("categorical", categorical_pipeline, categorical_cols)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    return X_train, X_test, y_train, y_test, preprocessor


@task
def train_model(X_train, y_train, preprocessor):
    print("\nTASK 3: Training Random Forest model")

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100, random_state=42
        ))
    ])

    model.fit(X_train, y_train)
    joblib.dump(model, "kidney_disease_model.joblib")

    print("Model trained successfully.")
    print("Saved model: kidney_disease_model.joblib")
    return model


@task
def evaluate_model(model, X_test, y_test):
    print("\nTASK 4: Evaluating model")

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["notckd", "ckd"],
        zero_division=0
    ))

    return accuracy


@flow(name="Kidney-Disease-ML-Workflow")
def kidney_disease_workflow():
    print("Starting Prefect workflow")

    df = load_data()
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
    model = train_model(X_train, y_train, preprocessor)
    accuracy = evaluate_model(model, X_test, y_test)

    print("\nWorkflow completed successfully!")
    print(f"Final accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    kidney_disease_workflow()

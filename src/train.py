# working with paths
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.validate_data import validate_data


FEATURES = [
    "area_sqft",
    "bedrooms",
    "age_years",
    "distance_km",
]

TARGET = "price_lakh"

R2_THRESHOLD = 0.85


def train():

    # --------------------------------------------------
    # Validate data
    # --------------------------------------------------

    df = validate_data()

    # --------------------------------------------------
    # Train / Test Split
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES],
        df[TARGET],
        test_size=0.2,
        random_state=42,
    )

    # --------------------------------------------------
    # MLflow configuration
    # --------------------------------------------------

    # Always create MLflow database in the current
    # project directory.
    project_root = Path.cwd()

    mlflow_db = project_root / "mlflow.db"

    mlflow.set_tracking_uri(
        f"sqlite:///{mlflow_db}"
    )

    mlflow.set_experiment(
        "house-price-regression"
    )

    # --------------------------------------------------
    # Start MLflow run
    # --------------------------------------------------

    with mlflow.start_run() as run:

        # --------------------------------------------------
        # Model
        # --------------------------------------------------

        model = LinearRegression()

        model.fit(
            X_train,
            y_train,
        )

        # --------------------------------------------------
        # Predictions
        # --------------------------------------------------

        predictions = model.predict(X_test)

        # --------------------------------------------------
        # Metrics
        # --------------------------------------------------

        mae = mean_absolute_error(
            y_test,
            predictions,
        )

        mse = mean_squared_error(
            y_test,
            predictions,
        )

        rmse = mse ** 0.5

        r2 = r2_score(
            y_test,
            predictions,
        )

        # --------------------------------------------------
        # MLflow parameters
        # --------------------------------------------------

        mlflow.log_param(
            "model_type",
            "LinearRegression",
        )

        mlflow.log_param(
            "random_state",
            42,
        )

        # --------------------------------------------------
        # MLflow metrics
        # --------------------------------------------------

        mlflow.log_metric(
            "mae",
            mae,
        )

        mlflow.log_metric(
            "mse",
            mse,
        )

        mlflow.log_metric(
            "rmse",
            rmse,
        )

        mlflow.log_metric(
            "r2",
            r2,
        )

        # --------------------------------------------------
        # Log model to MLflow
        # --------------------------------------------------

        mlflow.sklearn.log_model(
            model,
            name="house-price-model",
        )

        # --------------------------------------------------
        # Display results
        # --------------------------------------------------

        print(
            f"MAE={mae:.4f}, "
            f"RMSE={rmse:.4f}, "
            f"R2={r2:.4f}"
        )

        print(
            f"MLflow Run ID: {run.info.run_id}"
        )

        # --------------------------------------------------
        # Model Quality Gate
        # --------------------------------------------------

        if r2 < R2_THRESHOLD:

            raise RuntimeError(
                f"Model quality gate failed: "
                f"R2={r2:.4f} < {R2_THRESHOLD}"
            )

        print(
            f"Model quality gate passed: "
            f"R2={r2:.4f} >= {R2_THRESHOLD}"
        )

        # --------------------------------------------------
        # Save model
        # --------------------------------------------------

        model_directory = project_root / "model"

        model_directory.mkdir(
            exist_ok=True
        )

        model_path = (
            model_directory / "model.pkl"
        )

        joblib.dump(
            model,
            model_path,
        )

        print(
            f"Model saved to {model_path}"
        )


if __name__ == "__main__":
    train()
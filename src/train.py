from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from src.validate_data import validate_data

FEATURES = ["area_sqft", "bedrooms", "age_years", "distance_km"]
TARGET = "price_lakh"
R2_THRESHOLD = 0.85

def train():
    df = validate_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2, random_state=42
    )

    
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("house-price-regression")

    with mlflow.start_run() as run:
        model = LinearRegression()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, pred)
        mse = mean_squared_error(y_test, pred)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, pred)

        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("random_state", 42)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(model, "model")

        print(f"MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
        print(f"MLflow Run ID: {run.info.run_id}")

        if r2 < R2_THRESHOLD:
            raise RuntimeError(f"Model quality gate failed: R2={r2:.4f} < {R2_THRESHOLD}")

        Path("model").mkdir(exist_ok=True)
        joblib.dump(model, "model/model.pkl")
        print("Model saved to model/model.pkl")

if __name__ == "__main__":
    train()

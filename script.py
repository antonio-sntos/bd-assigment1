import os

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def read_csv_with_time_index(path):
    """Helper to read CSVs with a datetime index."""
    df = pd.read_csv(path, parse_dates=["time"], index_col="time")
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df


def create_eda_plots(joined_dfs):
    """
    Create exploratory data analysis plots for wind power data
    """
    fig, ax = plt.subplots(1,3, figsize=(25,4))

    # Speed and Power for the last 7 days
    ax[0].plot(joined_dfs["Speed"].tail(int(7*24/3)), label="Speed", color="blue")
    ax[0].plot(joined_dfs["Total"].tail(int(7*24/3)), label="Power", color="tab:red")
    ax[0].set_title("Windspeed & Power Generation over last 7 days")
    ax[0].set_xlabel("Time")
    ax[0].tick_params(axis='x', labelrotation = 45)
    ax[0].set_ylabel("Windspeed [m/s], Power [MW]")
    ax[0].legend()

    # Speed vs Total (Power Curve nature)
    ax[1].scatter(joined_dfs["Speed"], joined_dfs["Total"])
    power_curve = joined_dfs.groupby("Speed").median(numeric_only=True)["Total"]
    ax[1].plot(power_curve.index, power_curve.values, "k:", label="Power Curve")
    ax[1].legend()
    ax[1].set_title("Windspeed vs Power")
    ax[1].set_ylabel("Power [MW]")
    ax[1].set_xlabel("Windspeed [m/s]")

    # Speed and Power per Wind Direction
    if "Direction" in joined_dfs.columns:
        wind_grouped_by_direction = joined_dfs.groupby("Direction").mean(numeric_only=True).reset_index()
        bar_width = 0.5
        x = np.arange(len(wind_grouped_by_direction.index))
        ax[2].bar(x, wind_grouped_by_direction.Total, width=0.5, label="Power", color="tab:red")
        ax[2].bar(x + bar_width, wind_grouped_by_direction.Speed, width=0.5, label="Speed", color="blue")
        ax[2].legend()
        ax[2].set_xticks(x)
        ax[2].set_xticklabels(wind_grouped_by_direction.Direction)
        ax[2].tick_params(axis='x', labelrotation = 45)
        ax[2].set_title("Speed and Power per Direction")
    else:
        ax[2].axis("off")

    plt.tight_layout()
    return fig


# Connect to the local MLflow server.
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Orkney Wind Power")
mlflow.sklearn.autolog()

# MLflow Projects creates an outer run. We create separate runs for our models.
os.environ.pop("MLFLOW_RUN_ID", None)

print("Loading data")

power_df = read_csv_with_time_index("data/power.csv")
wind_df = read_csv_with_time_index("data/weather.csv")
future_df = read_csv_with_time_index("data/future.csv")

# Keep the target column and the two weather features used by the model.
power_df = power_df[["Total"]]
wind_df = wind_df.drop(columns=["Lead_hours", "Source_time"])

# Resample power to the same 3-hour interval as the weather data.
power_3h = power_df.resample("3h").mean()
joined_dfs = wind_df.join(power_3h, how="left")
joined_dfs = joined_dfs.dropna(subset=["Total"])

os.makedirs("plots", exist_ok=True)
eda_figure = create_eda_plots(joined_dfs)
eda_figure.savefig("plots/eda_plots.png")
plt.close(eda_figure)

# Create the preprocessing pipeline.
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

direction_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("speed", numeric_transformer, ["Speed"]),
    ("direction", direction_transformer, ["Direction"])
])

# Make a chronological train and test split.
X = joined_dfs[["Speed", "Direction"]]
y = joined_dfs["Total"]

split_index = int(len(joined_dfs) * 0.8)
X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
}

results = []
best_mae = np.inf
best_model_name = None
best_pipeline = None

for model_name, model in models.items():
    current_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    with mlflow.start_run(run_name=model_name):
        print(f"Training {model_name}")

        current_pipeline.fit(X_train, y_train)
        predictions = current_pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        mlflow.log_metrics({"MAE": mae, "RMSE": rmse, "R2": r2})
        mlflow.log_artifact("plots/eda_plots.png")

        prediction_file = f"plots/{model_name.replace(' ', '_').lower()}_predictions.png"
        plt.figure(figsize=(15, 4))
        plt.plot(y_test.index, y_test, label="Truth")
        plt.plot(y_test.index, predictions, label="Predictions")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(prediction_file)
        plt.close()
        mlflow.log_artifact(prediction_file)

        results.append({
            "Model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

        if mae < best_mae:
            best_mae = mae
            best_model_name = model_name
            best_pipeline = current_pipeline

results_df = pd.DataFrame(results).sort_values("MAE")
print("\nModel comparison")
print(results_df)
print(f"\nBest model: {best_model_name}")

# Use the best model to predict the future weather data.
X_future = future_df[["Speed", "Direction"]]
future_predictions = best_pipeline.predict(X_future)

predictions_df = future_df[["Speed", "Direction"]].copy()
predictions_df["Predicted_Power"] = future_predictions
predictions_df.to_csv("plots/future_predictions.csv")

print("\nFirst 10 future predictions")
print(predictions_df.head(10))

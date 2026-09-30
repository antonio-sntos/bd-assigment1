# Forecasting Wind Power Production in Orkney

**Big Data Assignment 1**  
**Student:** [YOUR NAME]  
**ITU username:** [YOUR ITU USERNAME]  
**Repository:** [INSERT PRIVATE GITHUB REPOSITORY LINK]

## 1. Introduction

The purpose of this project was to build a reproducible machine learning pipeline for forecasting wind power production in Orkney. The solution covers the complete workflow from data alignment and preprocessing to experiment tracking, model registration, and model serving. The objective was not only to obtain accurate predictions, but also to make the training process repeatable and the selected model available through an HTTP endpoint.

## 2. Data alignment and exploration

The project uses historical power production from `power.csv`, weather forecasts from `weather.csv`, and unseen weather observations from `future.csv`. The power dataset contains 109,026 high-frequency observations between 11 December 2021 and 11 March 2022. The weather dataset contains 716 observations recorded every three hours over approximately the same period.

Because the two datasets have different time frequencies, I resampled the `Total` power production to three-hour intervals using the mean. I then used the weather timestamps as the reference and performed a left join with the resampled power data. This approach produces one training observation for each available weather forecast and makes the target correspond to average production during the same three-hour interval. Forty joined rows did not have a target value and were removed, leaving 676 usable observations.

Averaging the power data reduces short-term noise and makes the sources temporally compatible. However, it also hides variation within each three-hour interval. Dropping observations without a target is appropriate for supervised training, although it slightly reduces the amount of data available.

The exploratory plots show a clear relationship between wind speed and power production, but it is not completely linear. Production generally increases with wind speed before approaching the wind farm's maximum output. Average production also differs between wind directions. These observations support using both wind speed and direction as model features and comparing a linear baseline with a nonlinear model.

**[Insert Figure 1: `plots/eda_plots.png`]**  
*Figure 1. Wind speed and production over time, the observed wind-power relationship, and average production by wind direction.*

## 3. Preprocessing and data splitting

The target variable is `Total` power production, while the model inputs are `Speed` and `Direction`. All preprocessing is implemented inside a scikit-learn `Pipeline` and `ColumnTransformer`. Missing wind speeds are replaced with the median and then standardized. Missing wind directions are replaced with the most frequent category and transformed using one-hot encoding. Unknown direction categories are ignored during prediction so that the pipeline can also process new data.

Standardizing wind speed is particularly useful for Linear Regression because it puts the numerical input on a consistent scale. One-hot encoding allows the categorical wind direction to be used without incorrectly treating compass labels as ordered numerical values. Keeping preprocessing and prediction in one pipeline also ensures that training and serving apply exactly the same transformations.

I used a chronological 80/20 split instead of a random split because this is a forecasting problem. The first 540 observations, from 11 December 2021 to 21 February 2022, were used for training. The final 136 observations, from 21 February to 11 March 2022, were used for testing. This strategy better represents the real task of predicting later production from earlier observations and prevents future observations from leaking into training.

## 4. Model training and evaluation

I compared Linear Regression and Random Forest Regression. Linear Regression was used as a simple and interpretable baseline. Random Forest was selected as the second model because it can represent nonlinear relationships and interactions between wind speed and direction. The Random Forest used 100 trees, a maximum depth of 10, and `random_state=42` for repeatability.

The models were evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and the coefficient of determination (R²). MAE measures the average absolute prediction error in MW and was used as the main selection metric. RMSE gives additional weight to large errors, while R² measures how much of the variation in production is explained by the model.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 5.218 | 6.315 | 0.667 |
| Random Forest | **4.085** | **5.888** | **0.711** |

Random Forest performed best on all three metrics and was selected as the final model. Its MAE was approximately 1.13 MW lower than the Linear Regression MAE. The prediction plots show that both models reproduce the broad production pattern, although errors remain during some rapid changes and low-production periods.

**[Insert Figure 2: `plots/random_forest_predictions.png`]**  
*Figure 2. Random Forest predictions compared with observed production on the chronological test set.*

The selected pipeline was also applied to the 35 observations in `future.csv`, covering 11–15 March 2022. The predictions were saved to `plots/future_predictions.csv`. For example, wind with a speed of 15.19936 m/s from the southeast produced a forecast of approximately 31.77 MW. This confirms that the complete pipeline can process unseen weather data.

## 5. Experiment tracking and model selection

MLflow was used to organize the work in an experiment named `Orkney Wind Power`. Linear Regression and Random Forest were stored as separately named runs. Scikit-learn autologging recorded model parameters and the fitted pipeline, while MAE, RMSE, and R² were logged explicitly. The exploratory plot and a prediction plot for each model were logged as artifacts. This made it possible to compare both alternatives in one interface and select Random Forest using the lowest MAE.

**[Insert Screenshot 1: MLflow experiment comparison]**  
*Figure 3. MLflow experiment showing the Linear Regression and Random Forest runs with MAE, RMSE, and R².*

The selected logged model was registered as `OrkneyWindPowerModel`. Registering the model gives it a stable name and version that can be used for deployment instead of depending directly on a run or artifact path.

**[Insert Screenshot 2: registered model version]**  
*Figure 4. Version 1 of `OrkneyWindPowerModel` in the MLflow Model Registry.*

## 6. Model serving and reproducibility

The registered model was served locally with MLflow on port 5001, while the tracking server remained available on port 5000. The endpoint accepts weather data using MLflow's `dataframe_records` format. The following request was used to test the service:

```json
{"dataframe_records":[{"Speed":10.5,"Direction":"SW"}]}
```

The service returned the following successful prediction:

```json
{"predictions":[19.809576552934928]}
```

**[Insert Screenshot 3: terminal containing the curl command and prediction response]**  
*Figure 5. Successful request to the served model through the `/invocations` endpoint.*

The repository contains `script.py` as the training entry point, an `MLproject` file, `python_env.yaml`, and `requirements.txt`. The complete training workflow can be started with `python -m mlflow run . --env-manager local`. The script loads and aligns the raw data, trains both candidates, logs the results, selects the best pipeline, and produces future predictions. Packaging the workflow this way reduces the number of manual steps and makes the experiment easier to repeat on another machine.

## 7. Reflection and limitations

The available training window is approximately 90 days. This is enough to demonstrate the complete machine learning lifecycle and contains several different wind conditions. It is also recent relative to the test period. However, a 90-day window covers only part of one winter and cannot represent annual seasonality, long-term changes, maintenance periods, or the full range of rare weather events. A production system should use a longer historical window, evaluate different window lengths, and retrain on a rolling basis as new observations arrive.

The model uses only wind speed and categorical wind direction. Forecast accuracy could potentially be improved by adding weather variables such as gusts, air pressure, temperature, and forecast lead time, as well as lagged power production. One-hot encoding treats compass directions as separate categories and does not represent their circular relationship. Converting direction to sine and cosine components would preserve the fact that north-northeast is close to both north and northeast.

The current comparison contains only two models and one fixed Random Forest configuration. Time-series cross-validation and systematic hyperparameter tuning would provide a more reliable comparison. The three-hour aggregation also removes short-term variation, and the model does not quantify prediction uncertainty. Finally, the local endpoint has no authentication, monitoring, drift detection, or automated retraining. These would be necessary before using the service in a real electricity-grid setting.

---

## Checklist before exporting to PDF

- Replace the name, ITU username, and repository URL.
- Reconcile `python_env.yaml` and `requirements.txt` with the final tested Python environment.
- Insert the five figures/screenshots at the marked locations.
- Ensure the MLflow screenshots clearly show experiment/run names and metric columns.
- Ensure the serving screenshot includes both the request and returned prediction.
- Export as `[ITU_USERNAME]-A1-report.pdf`.

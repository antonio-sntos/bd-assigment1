# Forecasting Wind Power Production in Orkney

This project builds a reproducible machine learning pipeline that predicts wind power production in Orkney from weather forecasts. It covers data alignment, preprocessing, model comparison, MLflow experiment tracking, future predictions, model registration, and local model serving.


## Installation

After donwload or clone the repo:
Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the training project

### 1. Start the MLflow server

Run this command in the first terminal:

```bash
python -m mlflow server --host 127.0.0.1 --port 5000 --workers 1
```

Keep this terminal open. The MLflow interface is available at:

```text
http://127.0.0.1:5000
```

### 2. Run the packaged project

Open a second terminal in the repository directory.

```
export MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
python -m mlflow run . --env-manager local
```

The training script can also be run directly:

```bash
python script.py
```

## Serve the registered model

Keep the MLflow tracking server running. In a new terminal, set the tracking URI and serve the registered model on port 5001.

I had to use different commands that the one prupose in class, because I had to run it locally.

```powershell
$env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
python -m mlflow models serve -m "models:/OrkneyWindPowerModel/1" --env-manager local --host 127.0.0.1 --port 5001 --workers 1
```

Alternatively insteead of the "$env:" you can also use export. Like this.

```bash
export MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
python -m mlflow models serve -m "models:/OrkneyWindPowerModel/1" --env-manager local --host 127.0.0.1 --port 5001 --workers 1
```

## Test the prediction endpoint

The served model expects "Speed" and "Direction" as inputs.

I tried to put just the "curl" and i wasnt able to do it, with some search I was able to find an altrenative that would work for the powershell of Windows

```powershell
curl.exe --% -X POST -H "Content-Type: application/json" -d "{\"dataframe_records\":[{\"Speed\":10.5,\"Direction\":\"SW\"}]}" http://127.0.0.1:5001/invocations
```

Example:

```json
{"predictions": [19.809576552934928]}
```

Port 5000 is used by the MLflow tracking server and port 5001 is used by the model prediction endpoint.

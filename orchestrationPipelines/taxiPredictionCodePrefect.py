#!/usr/bin/env python
# coding: utf-8

# The pipeline orchestration contains various steps
# 
# Data ingestion --> Data transformation (filtering and removing outliners) --> Preparing data for ML (extracting X, y vaules) --> hyperparameter tuning --> train the model --> save the model in the model registry


import pandas as pd
import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error

import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")


import xgboost as xgb

from pathlib import Path
models_folder = Path('models')
models_folder.mkdir(exist_ok=True)

from prefect import flow, task

# Data ingestion & tranformation

@task(retries=3, retry_delay_seconds=2)
def read_dataframe(year, month):
    """Read data into Dataframe"""
    df = pd.read_parquet(f"https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet")

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

    return df 


# preparing the data for ML
@task
def create_X(df, dv=None):

    categorical = ['PU_DO'] #'PULocationID', 'DOLocationID']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')

    if dv is None:
    
        dv = DictVectorizer()
        X = dv.fit_transform(dicts)

    else:
        X = dv.transform(dicts)

    return X, dv

# Training the model
@task(log_prints=True)
def train_model(X_train, y_train, X_val, y_val, dv):

    mlflow.set_experiment("taxi-prediction-prefect")
    with mlflow.start_run():
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        best_params = {
            'learning_rate': 0.09585355369315604,
            'max_depth': 30,
            'min_child_weight': 1.060597050922164,
            'objective': 'reg:linear',
            'reg_alpha': 0.018060244040060163,
            'reg_lambda': 0.011658731377413597,
            'seed': 42
        }

        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=30,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )

        y_pred = booster.predict(valid)
        rmse = mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")


        # Save run metadata to JSON
        import json
        run = mlflow.active_run()
        run_info = {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "experiment_name": mlflow.get_experiment(run.info.experiment_id).name,
            "run_name": run.data.tags.get("mlflow.runName"),
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            "metrics": {"rmse": rmse}
        }
        with open("models/mlflow_run_metadata.json", "w") as f:
            json.dump(run_info, f, indent=4)

# main calling function
@flow
def run(year, month):
    df_train = read_dataframe(year=year, month=month)
    df_val   = read_dataframe(year=year, month=month + 1)

    X_train, dv = create_X(df=df_train)
    X_val, _= create_X(df=df_val, dv=dv)

    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values

    train_model(X_train, y_train, X_val, y_val, dv)
if __name__ == "__main__":
    # use argparse to get user input
    import argparse
    parser = argparse.ArgumentParser(description="Run model training for a given year and month")
    parser.add_argument("--year", type=int, required=True, help="Year of data")
    parser.add_argument("--month", type=int, required=True, help="Month of data")
    args = parser.parse_args()

    run(year=args.year, month=args.month)


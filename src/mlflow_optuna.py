import pandas as pd
import numpy as np
import json
import pickle
from prefect import flow, task
from xgboost import XGBClassifier
import optuna
import mlflow
from sklearn.model_selection import RepeatedKFold
import xgboost as xgb
import mlflow.xgboost
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    classification_report,
)
from sklearn.model_selection import train_test_split
import mlflow.pyfunc
from mlflow import MlflowClient
from util import parse_cfg


@task
def split_data(df, split_size, random_state):
    X, y = df.drop("Churn", axis=1), df.filter(items=["Churn"])
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=split_size, random_state=random_state
    )
    return X_train, y_train


@task
def objective(X_train, y_train, trial, _best_auc=0):

    try:
        EXPERIMENT_ID = mlflow.create_experiment("xgboost-optuna")
    except:
        EXPERIMENT_ID = dict(mlflow.get_experiment_by_name("xgboost-optuna"))[
            "experiment_id"
        ]

    with mlflow.start_run(experiment_id=EXPERIMENT_ID, nested=True):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "booster": "gbtree",
            "lambda": trial.suggest_float("lambda", 1e-3, 10.0),
            "alpha": trial.suggest_float("alpha", 1e-3, 10.0),
            "eta": trial.suggest_float("eta", 1e-8, 1.0),
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0),
            "grow_policy": trial.suggest_categorical(
                "grow_policy", ["depthwise", "lossguide"]
            ),
            "scale_pos_weight": trial.suggest_int("scale_pos_weight", 2, 50),
        }
        xgb_cl = XGBClassifier(**params)

        rkf = RepeatedKFold(n_splits=2, n_repeats=2, random_state=42)
        aucpr_scores = []

        X_values = X_train.values
        y_values = y_train.values
        for train_index, valid_index in rkf.split(X_train):
            X_train_sub, y_train_sub = X_values[train_index], y_values[train_index]
            X_valid_sub, y_valid_sub = X_values[valid_index], y_values[valid_index]

            xgb_cl.fit(
                X_train_sub,
                y_train_sub,
                eval_set=[(X_valid_sub, y_valid_sub)],
                verbose=0,
            )
            y_pred = xgb_cl.predict(X_valid_sub)
            aucpr = roc_auc_score(y_valid_sub, y_pred)
            aucpr_scores.append(aucpr)

        mean_aucpr = np.mean(aucpr_scores)

        mlflow.xgboost.log_model(xgb_cl, "xgboost_model")
        mlflow.log_param("Optuna_trial_num", trial.number)
        mlflow.log_params(params)

        if mean_aucpr > _best_auc:
            _best_auc = mean_aucpr
            with open("./output/trial_%d.pkl" % trial.number, "wb") as f:
                pickle.dump(xgb_cl, f)
            mlflow.log_artifact("./output/trial_%d.pkl" % trial.number)
        mlflow.log_metric("VAL_AUC", mean_aucpr)
    return mean_aucpr


@flow(name="Hyperparameter-tuning-n-serving")
def xgb_tuning():
    client = MlflowClient()
    cfg = parse_cfg()
    split_size, RS = cfg["split_size"], cfg["RS"]
    df = pd.read_csv("./data/df_cleaned.csv")

    X_train, y_train = split_data(df, split_size, RS)

    model_name = "telco-churn-xgb"
    model_version = 1

    study = optuna.create_study(study_name="test", direction="maximize")
    study.optimize(
        lambda trial: objective(X_train, y_train, trial, _best_auc=0), n_trials=30
    )
    best_trial = study.best_trial
    best_params = best_trial.params
    with open("./output/best_param.json", "w") as outfile:
        json.dump(best_params, outfile)

    runs_df = mlflow.search_runs(
        experiment_ids=dict(mlflow.get_experiment_by_name("xgboost-optuna"))[
            "experiment_id"
        ],
        order_by=["metrics.validation_aucroc DESC"],
    )
    best_run = runs_df.iloc[0]
    best_run_id = best_run["run_id"]
    # best_artifact_uri = best_run['artifact_uri']
    # best_model = mlflow.xgboost.load_model('runs:/' + best_run_id + '/xgboost_model')
    _ = mlflow.register_model("runs:/" + best_run_id + "/xgboost_model", model_name)

    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
    print(model)
    client.transition_model_version_stage(
        name=model_name, version=model_version, stage="Production"
    )


if __name__ == "__main__":
    xgb_tuning()

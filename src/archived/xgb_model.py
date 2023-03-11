import pandas as pd
import xgboost as xgb
import json
from prefect import flow, task
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
)
from sklearn.model_selection import train_test_split


@task
def df_splitting(df: pd.DataFrame, split_size: float, RS: int) -> pd.DataFrame:
    X, y = df.drop("Churn", axis=1), df.filter(items=["Churn"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split_size, random_state=RS
    )
    return X_train, X_test, y_train, y_test


@task
def xgb_fitting(feature: pd.DataFrame, response: pd.DataFrame) -> xgb.core.Booster:
    xgb_cl = XGBClassifier()
    xgb_cl.fit(feature, response)
    return xgb_cl


@task
def eval_metrics_extraction(
    model: xgb.core.Booster, feature: pd.DataFrame, response: pd.DataFrame
):
    y_pred = model.predict(feature)
    accuracy = accuracy_score(response, y_pred)
    precision = precision_score(response, y_pred)
    recall = recall_score(response, y_pred)
    f1 = f1_score(response, y_pred)
    roc_auc = roc_auc_score(response, y_pred)
    logloss = log_loss(response, y_pred)
    return accuracy, precision, recall, f1, roc_auc, logloss


@flow(name="xgb-modelling")
def xgb_modelling_flow():
    X_train, X_test, y_train, y_test = df_splitting(df_final, split_size, RS)
    xgb_cl = xgb_fitting(X_train, y_train)
    accuracy, precision, recall, f1, roc_auc, logloss = eval_metrics_extraction(
        xgb_cl, X_test, y_test
    )
    result_dict = {
        'accuracy':accuracy,
        'precision':precision,
        'recall':recall,
        'f1':f1,
        'roc_auc':roc_auc,
        'logloss':logloss
    }
    with open("./output/sample_metrics.json", "w") as outfile:
        json.dump(result_dict, outfile)

    return accuracy, precision, recall, f1, roc_auc, logloss


if __name__ == "__main__":
    df_final = pd.read_csv("./data/df_cleaned.csv")
    split_size = 0.2
    RS = 42
    xgb_modelling_flow()

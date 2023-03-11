from prefect import flow
from load_data import etl_flow
from transform import data_cleaning
from mlflow_optuna import xgb_tuning




@flow(name='end-to-end-development')
def development():
    etl_flow()
    data_cleaning()
    xgb_tuning()





if __name__ == "__main__":
    development()
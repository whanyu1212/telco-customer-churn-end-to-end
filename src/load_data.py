import pandas as pd
import pandera as pa
from prefect import flow, task
from jinjasql import JinjaSql
from google.cloud import bigquery
from typing import Dict
from pandera import check_output
from query_template import BQ_QUERY_TEMPLATE
from util import out_schema, parse_cfg


@task
@check_output(out_schema)
def fetch_bq_data(cfg: Dict) -> pd.DataFrame:
    """parse the key parameters within the config and parse into the the jinjasql template

    Args:
        cfg (dict): config file

    Returns:
        pd.DataFrame: bigquery table turned dataframe
    """
    query_param_dict = {k: cfg[k] for k in ("project_id", "dataset_id", "table_id")}
    client = bigquery.Client.from_service_account_json("./config/sacred-garden-369506-870f85c5921d.json")
    j = JinjaSql()
    query, _ = j.prepare_query(BQ_QUERY_TEMPLATE, query_param_dict)
    query_job = client.query(query).to_dataframe().query("TotalCharges!=' '").reset_index(drop=True)
    query_job.to_csv('./data/telco_customer_churn_coerced.csv',index=False)
    return query_job


@flow(name="etl-process")
def etl_flow():
    cfg = parse_cfg()
    fetch_bq_data(cfg)


if __name__ == "__main__":
    etl_flow()

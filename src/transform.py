import pandas as pd
from prefect import flow, task
from typing import Dict
from sklearn import preprocessing

@task
def generate_dummy_variables(df:pd.DataFrame, cat_columns: list) -> pd.DataFrame:
    """ one hot encode the categorical variables

    Args:
        df (pd.DataFrame): coerced data generated from etl.py
        cat_columns (list): names of the categorical columns

    Returns:
        pd.DataFrame: a dataframe consists of all the one hot encoded categorical variables
    """    
    dummy_frames = []
    for col in cat_columns:
        dummy_frame = pd.get_dummies(df[col], prefix=col)
        dummy_frames.append(dummy_frame)
        df_dummies = pd.concat(dummy_frames, axis=1)

    return df_dummies


@task
def combine_dummy_n_numeric(df_dummies:pd.DataFrame,df:pd.DataFrame) -> pd.DataFrame:
    df_final = pd.concat([df_dummies,df['tenure'],df['SeniorCitizen'].astype(int),df['MonthlyCharges'],
        df['TotalCharges'],df['Churn']],axis=1)
    le = preprocessing.LabelEncoder()
    le.fit(df_final.Churn)
    df_final.Churn = le.transform(df_final.Churn)
    df_final.to_csv('./data/df_cleaned.csv',index=False)
    return df_final

@flow(name="data-cleaning")
def data_cleaning():
    df_dummies = generate_dummy_variables(df,cat_columns)
    df_final = combine_dummy_n_numeric(df_dummies,df)
    return df_final










if __name__ == "__main__":
    df = pd.read_csv('./data/telco_customer_churn_coerced.csv')
    cat_columns = ['gender', 'Partner', 'Dependents',
    'PhoneService','MultipleLines','InternetService',
    'OnlineSecurity','OnlineBackup','DeviceProtection',
    'TechSupport','StreamingTV','StreamingMovies',
    'Contract','PaperlessBilling','PaymentMethod']
    data_cleaning()

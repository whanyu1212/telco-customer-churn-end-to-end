import pandera as pa
from pandera import Column, DataFrameSchema, Check

out_schema = DataFrameSchema(
    {
        "customerID": Column(str, nullable=False),
        "gender": Column(str, Check.isin(["Male", "Female"]), nullable=False),
        "SeniorCitizen": Column(pa.Category, Check.isin([0, 1]), nullable=False,coerce=True),
        "Partner": Column("boolean", nullable=False),
        "Dependents": Column("boolean", nullable=False),
        "tenure": Column(int, Check.less_than(100), nullable=False),
        "PhoneService": Column("boolean", nullable=False),
        "MultipleLines": Column(str, Check.isin(["Yes", "No","No phone service"]), nullable=False),
        "InternetService": Column(str, Check.isin(["No", "DSL", "Fiber optic"]), nullable=False),
        "OnlineSecurity": Column(
            str, Check.isin(["No internet service", "No", "Yes"]), nullable=False
        ),
        "OnlineBackup": Column(
            str, Check.isin(["No internet service", "No", "Yes"]), nullable=False
        ),
        "DeviceProtection": Column(
            str, Check.isin(["No internet service", "No", "Yes"]), nullable=False
        ),
        "TechSupport": Column(
            str, Check.isin(["No internet service", "No", "Yes"]), nullable=False
        ),
        "StreamingTV": Column(
            str, Check.isin(["No internet service", "No", "Yes"]), nullable=False
        ),
        "StreamingMovies": Column(
            str, Check.isin(["No internet service", "No", "Yes"]), nullable=False
        ),
        "Contract": Column(
            str,Check.isin(["One year", "Two year", "Month-to-month"]), nullable=False
        ),
        "PaperlessBilling": Column("boolean", nullable=False),
        "PaymentMethod": Column(
            str,
            Check.isin(
                [
                    "Credit card (automatic)",
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                ]
            ),
            nullable=False,
        ),
        "MonthlyCharges": Column(float, nullable=False, coerce=True),
        "TotalCharges": Column(float, nullable=False, coerce=True),
        "Churn": Column("boolean", nullable=False),
    }
)

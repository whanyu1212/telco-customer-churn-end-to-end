from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd

app = Dash(__name__)

df = pd.read_csv('./data/telco_customer_churn_coerced.csv').assign(SeniorCitizen = lambda x:x.SeniorCitizen.astype(str))

cat_list = df.select_dtypes(exclude=['float64','Int64']).columns.tolist()
cat_list = [e for e in cat_list if e not in ('customerID', 'Churn')]


app.layout = html.Div([
    html.H1('Categorical Variables from Telco Customer Churn dataset'),
    dcc.Dropdown(
        id="dropdown",
        options=cat_list,
        value="gender",
        clearable=False,
    ),
    dcc.Graph(id="graph"),
])

@app.callback(
    Output("graph", "figure"), 
    Input("dropdown", "value"))

def update_bar_chart(value):
    df_update = df.groupby(['Churn',value]).size().reset_index(name='count')
    fig = px.bar(df_update,x='Churn', y='count', 
                 color=value)
    return fig

app.run_server(debug=True)
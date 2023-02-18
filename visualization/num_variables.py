from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd

app = Dash(__name__)

df = pd.read_csv('./data/telco_customer_churn_coerced.csv').assign(SeniorCitizen = lambda x:x.SeniorCitizen.astype(str))


num_list = df.select_dtypes(include=['float64','Int64']).columns.tolist()


app.layout = html.Div([
    html.H1('Numerical Variables from Telco Customer Churn dataset'),
    dcc.Dropdown(
        id="dropdown",
        options=num_list,
        value="tenure",
        clearable=False,
    ),
    dcc.Graph(id="graph"),
])

@app.callback(
    Output("graph", "figure"), 
    Input("dropdown", "value"))

def update_bar_chart(value):
    fig = px.violin(df,x='Churn', y=value, 
                 color='Churn',box=True,points='all')
    return fig

app.run_server(debug=True)
import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px

# ======================
# 1. Load CSV files
# ======================
df_corr = pd.read_csv("delivery_insights.csv")   # from TensorFlow pipeline
df_raw = pd.read_csv("C:/Users/shank/Desktop/amazon_delivery.csv")

# Ensure Delivery_Time is numeric
df_raw['Delivery_Time'] = pd.to_numeric(df_raw['Delivery_Time'], errors='coerce')

# ======================
# 2. Dropdown options
# ======================
feature_options = [
    {"label": "Agent Age", "value": "Agent_Age"},
    {"label": "Agent Rating", "value": "Agent_Rating"},
    {"label": "Store Latitude", "value": "Store_Latitude"},
    {"label": "Store Longitude", "value": "Store_Longitude"},
    {"label": "Drop Latitude", "value": "Drop_Latitude"},
    {"label": "Drop Longitude", "value": "Drop_Longitude"},
    {"label": "Weather", "value": "Weather"},
    {"label": "Traffic", "value": "Traffic"},
    {"label": "Vehicle", "value": "Vehicle"},
    {"label": "Area", "value": "Area"},
    {"label": "Category", "value": "Category"}
]

# ======================
# 3. Build static correlation table (no callback)
# ======================
corr_table = html.Div([
    html.Table([
        html.Thead(html.Tr([html.Th(col) for col in df_corr.columns])),
        html.Tbody([
            html.Tr([html.Td(str(df_corr.iloc[i][col])) for col in df_corr.columns])
            for i in range(len(df_corr))
        ])
    ], style={'border': '1px solid black', 'borderCollapse': 'collapse', 'width': '80%', 'margin': 'auto'})
])

# ======================
# 4. Initialize Dash app
# ======================
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Logistics Delivery Insights", style={'textAlign': 'center'}),

    html.H3("Feature vs Delivery Time"),
    dcc.Dropdown(
        id='feature-dropdown',
        options=feature_options,
        value='Traffic',  # default selection
        clearable=False
    ),

    dcc.Graph(id='feature-vs-delivery'),

    html.H3("Correlation Table (from TensorFlow pipeline)"),
    corr_table
])

# ======================
# 5. Graph callback only
# ======================
@app.callback(
    Output('feature-vs-delivery', 'figure'),
    [Input('feature-dropdown', 'value')]
)
def update_chart(selected_feature):
    dff = df_raw[[selected_feature, 'Delivery_Time']].dropna()

    # Decide chart type
    if dff[selected_feature].dtype in ['float64', 'int64']:
        fig = px.scatter(
            dff, x=selected_feature, y="Delivery_Time",
            title=f"{selected_feature} vs Delivery Time",
            opacity=0.7
        )
        fig.update_traces(marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')))
    else:
        dff_grouped = dff.groupby(selected_feature)['Delivery_Time'].mean().reset_index()
        fig = px.bar(
            dff_grouped, x=selected_feature, y="Delivery_Time",
            title=f"{selected_feature} vs Avg Delivery Time"
        )
        fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))  # no 'size' for bars

    return fig

# ======================
# 6. Run app
# ======================
if __name__ == '__main__':
    app.run(debug=True)
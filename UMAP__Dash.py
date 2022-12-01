from dash import Dash, html, dcc, dash_table, Input, Output, ctx
import plotly.express as px
import pandas as pd
from sklearn.cluster import KMeans
import umap.umap_ as umap

app = Dash(__name__)
#---------------------------------------------------------------------------------------------------
path = 'C:/Studia/CMS/Magisterka/Saptarshi/Jan_Sal/KMean_AML_prepared_data_83features.csv'
df = pd.read_csv(path, na_values='NULL')
df.set_index('Pat', inplace=True)

path2 = "C:/Studia/CMS/Mastersthesis/test_data.csv"
df2 = pd.read_csv(path2, na_values='NULL')
df2.set_index('Pat', inplace=True)
i = 0
#---------------------------------------------------------------------------------------------------
app.layout = html.Div([

    html.H1("UMAP Projection for original data", style={'text-align': 'center'}),

    dcc.Checklist(id="select_feature",
                  options = [{'label':x, 'value':x, 'disabled':False}
                             for x in df],
                  value = ['AGE', 'KIT', 'PBB', 'DNMT3A', 'PLT',
                            'BMB', 'NPM1', 'CGCX', 'CEBPADM', 'CEBPA.bZIP',
                            'FLT3I', 'NRAS', 'TET2', 'FLT3T', 'FLT3R']
                  ),

    html.Div(id='output_container', children=[]),
    html.Br(),

    html.Div(children=[
        html.Div(
            dcc.Graph(id='Umap_projection2d',
                figure={}
            ), style={'display': 'inline-block'}),
        html.Div(
            dcc.Graph(id='Umap_projection3d',
                figure={}
            ), style={'display': 'inline-block'})
    ], style={'width': '100%', 'display': 'inline-block'}),

    html.H1("Pick the list of features", style={'text-align': 'center'}),

    dcc.Checklist(id="select_feature2",
                  options=[{'label': x, 'value': x, 'disabled': False}
                           for x in df],
                  value=['AGE', 'KIT', 'PBB', 'DNMT3A', 'PLT',
                            'BMB', 'NPM1', 'CGCX', 'CEBPADM', 'CEBPA.bZIP',
                            'FLT3I', 'NRAS', 'TET2', 'FLT3T', 'FLT3R']

                  ),
    html.Div(id='output_container2', children=[]),

    html.Button('Add new point', id='btn-nclicks-1', n_clicks=0),

    html.Div(dcc.Graph(id='Umap_projection2d_live_adding',
                figure={})),
])

@app.callback(
    [Output(component_id='output_container', component_property='children'),
     Output(component_id='Umap_projection2d', component_property='figure'),
     Output(component_id='Umap_projection3d', component_property='figure')],
    [Input(component_id='select_feature', component_property='value')]
)
def update_graph(option_selected):
    print(option_selected)

    dff = df.copy()
    dff = dff[option_selected]
    container = "Shape of the dataframe: {}".format(dff.shape)
    kmeans = KMeans(4)
    kmeans.fit(dff)
    clusters = dff.copy()
    clusters['clusters_pred'] = kmeans.fit_predict(dff)
    proj_2d = umap.UMAP(random_state=42).fit_transform(dff)

    fig_2d = px.scatter(
        proj_2d, x=proj_2d[:,0], y=proj_2d[:,1],
        color=clusters['clusters_pred'],
        labels={'clusters_pred'}
    )

    proj_3d = umap.UMAP(n_components=3, random_state=42).fit_transform(dff)
    fig_3d = px.scatter_3d(
        proj_3d, x=proj_3d[:, 0], y=proj_3d[:, 1], z=proj_3d[:, 2],
        color=clusters['clusters_pred'], labels={'color': 'clusters_pred'}
    )
    fig_3d.update_traces(marker_size=5)

    return container, fig_2d, fig_3d

@app.callback(
    [Output(component_id='output_container2', component_property='children'),
     Output(component_id='Umap_projection2d_live_adding', component_property='figure')],
    [Input(component_id='btn-nclicks-1', component_property='n_clicks'),
     Input(component_id='select_feature2', component_property='value')]
)
def live_graph(btn1, option_selected):
    global i
    global df

    df = df[option_selected]
    dff2 = df2.copy()
    dff2 = dff2[option_selected]
    container2 = "Shape of the original dataframe: {}".format(df.shape) + " Shape of the 2nd dataframe: {}".format((dff2.shape))

    if "btn-nclicks-1" == ctx.triggered_id:
        df = df.append(dff2.iloc[i])
        i = i+1
        print(df.shape, i)

    kmeans = KMeans(4)
    kmeans.fit(df)
    clusters = df.copy()
    clusters['clusters_pred'] = kmeans.fit_predict(df)
    standard_embedding = umap.UMAP(random_state=42).fit_transform(df)

    fig1 = px.scatter(df, x=standard_embedding[:, 0], y=standard_embedding[:, 1],
                      color=clusters['clusters_pred'])
    fig1.add_scatter(x=[standard_embedding[0][-1]], y=[standard_embedding[1][-1]],
                     mode='markers + text',
                     marker={'color': 'red', 'size': 14})

    return container2, fig1

if __name__ == '__main__':
    app.run_server(debug=True)

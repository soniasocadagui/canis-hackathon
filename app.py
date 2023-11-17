import dash
import requests

from dash import html
from dash import dcc
from dash.dependencies import Input, Output

import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import os

# Creating Data for my app

# weatwaves
disaster_path = "https://raw.githubusercontent.com/soniasocadagui/canis-hackathon-app/main/data/disaster_data.xlsx"
colslist = ['Year', 'Disaster Type','Disaster Subtype','Country','Start Month','Start Day','End Month','End Day']
disaster_data = pd.read_excel(disaster_path, usecols = colslist)
heatWaves_data = disaster_data[disaster_data['Disaster Subtype'] == 'Heat wave']
# total number disasters
number_disasters = disaster_data[disaster_data['Disaster Type'] == 'Extreme temperature'].groupby(['Country', 'Disaster Subtype']).count().reset_index()[["Country", 'Disaster Subtype', 'Year']]


# temperatures
df_total = pd.read_csv("https://raw.githubusercontent.com/soniasocadagui/canis-hackathon-app/main/data/data_total.csv", 
                           sep=",", low_memory=False)
meanChange = df_total['max_temperature_change'].mean()

weather_path = "https://raw.githubusercontent.com/soniasocadagui/canis-hackathon-app/main/data/weather_data.csv"
weather_data = pd.read_csv(weather_path, sep = ";")


# maps
url = "https://raw.githubusercontent.com/soniasocadagui/canis-hackathon-app/main/data/low_res_custom.geo.json"
f = requests.get(url)
countries = f.json()


# predictions
predict_path = "https://raw.githubusercontent.com/soniasocadagui/canis-hackathon-app/main/data/predictions.xlsx"
prediction_data = pd.read_excel(predict_path)
prediction_data = pd.melt(prediction_data, id_vars=["country", 'item'],var_name='year', value_name='gross_per_capita_pin')
prediction_data['type'] = 'Prediction'

dict_prod = {'Agriculture': "Agriculture",
             'Cereals, Total': "Cereals",
             'Food': "Food",
             'Livestock': "Livestock",
             'Milk, Total': "Milk",
             'Vegetables and Fruit Primary': "Vegetables and Fruits"}
prediction_data['item'] = prediction_data['item'].map(dict_prod)


counttr = ["Austria", "Bulgaria", "France", "Germany", "Greece", "Hungary", "Italy", "Portugal", "Romania", "Spain", "Switzerland"]
df_totalA = df_total.loc[df_total['country'].isin(counttr),
                         ['country', 'item', 'year', 'gross_per_capita_pin']].drop_duplicates().reset_index(drop=True)
df_totalA = df_totalA.loc[df_totalA['year'] > 2000].drop_duplicates().reset_index(drop=True)
df_totalA = df_totalA.loc[df_totalA['year'] < 2019 ].drop_duplicates().reset_index(drop=True)
df_totalA['type'] = 'Real'


prediction_data = pd.concat([prediction_data,df_totalA])


# creating the app
app = dash.Dash(__name__,)
server = app.server

app.layout = html.Div([
    html.Div([

        # Logo and title
        html.Div([
            html.Img(src=app.get_asset_url('heat-logo-1.jpg'),
            id = 'heat-image',
            style={'height': '100px',
            'width': 'auto',
            'margin-bottom': '25px'})

        ], className='one-third column'),
        html.Div([
            html.Div([
                html.H1('MDA PROJECT', style={'margin-bottom': '0px', 'color': 'white'}),
                html.H5('', style={'margin-bottom': '0px', 'color': 'white'})
            ])

        ], className='one-half column', id = 'title')
        

    ], id = 'header', className='row flex-display', style={'margin-bottom': '25px'}),

    # Counting
    html.Div([
        html.Div([
            html.H6(children='Number Heat Waves*',
                    style={'textAlign': 'center',
                            'color': 'white'}),
            html.P(f"{heatWaves_data.count().iloc[0]:,.0f}",
                    style={'textAlign': 'center',
                            'color': '#DC143C',
                            'fontSize': 40}),
            html.P("*From 1951 to 2021 worldwide",
                   style={'textAlign': 'center',
                          'color': 'white'}
                   )
        ], className='card_container three columns'),
        html.Div([
            html.H6(children='Minimum temperature change*',
                    style={'textAlign': 'center',
                            'color': 'white'}),
            html.P(f"{df_total[df_total['max_temperature_change'] == df_total['max_temperature_change'].min()]['max_temperature_change'].reset_index()['max_temperature_change'].iloc[0]:,.1f}°C",
                style={'textAlign': 'center',
                       'color': 'blue',
                       'fontSize': 40}),
            html.P("*Ecuador (1971)",
                   style={'textAlign': 'center',
                          'color': 'white'}
                   ),
            html.P("*Baseline climatology period: 1951 - 1980",
                   style={'textAlign': 'center',
                          'color': 'white'}
                   )
        ], className='card_container three columns'),
        html.Div([
            html.H6(children='Maximum temperature change*',
                    style={'textAlign': 'center',
                            'color': 'white'}),
             html.P(f"{df_total[df_total['max_temperature_change'] == df_total['max_temperature_change'].max()]['max_temperature_change'].reset_index()['max_temperature_change'].iloc[0]:,.1f}°C",
                style={'textAlign': 'center',
                       'color': '#DC143C',
                       'fontSize': 40}),
            html.P("*Finland (1990)",
                   style={'textAlign': 'center',
                          'color': 'white'}
                   ),
            html.P("*Baseline climatology period: 1951 - 1980",
                   style={'textAlign': 'center',
                          'color': 'white'}
                   )
        ], className='card_container three columns'),
        html.Div([
            html.H6(children='Average temperature change*',
                    style={'textAlign': 'center',
                            'color': 'white'}),
            html.P(f"{meanChange:,.1f}°C",
                    style={'textAlign': 'center',
                            'color': '#900C3F',
                            'fontSize': 40}),
            html.P("*From 1961 to 2020 worldwide",
                   style={'textAlign': 'center',
                          'color': 'white'}
                   )
        ], className='card_container three columns'),
    # First panel
        html.Div([
        
            html.Div([
                html.P('Place the cursor over the country:', className='fix_label', style={'color': 'white'}),
                dcc.Graph(
                    id='temp_map',
                    hoverData={'points': [{'customdata': 'Belgium'}]}
                )
            ], className='create_container seven columns'),#, style={'width': '60%', 'display': 'inline-block', 'padding': '0 20'}),
        
            html.Div([
                html.P('Select the product:', className='fix_label', style={'color': 'white'}),
                dcc.Dropdown(id="slct_item",
                        options=[
                            {"label": "Agriculture", "value": 'Agriculture'},
                            {"label": "Cereals", "value": 'Cereals'},
                            {"label": "Crops", "value": 'Crops'},
                            {"label": "Food", "value": 'Food'},
                            {"label": "Livestock", "value": 'Livestock'},
                            {"label": "Meat Indigenous", "value": 'Meat Indigenous'},
                            {"label": "Milk", "value": 'Milk'},
                            {"label": "Non-Food", "value": 'Non-Food'},
                            {"label": "Oilcrops", "value": 'Oilcrops'},
                            {"label": "Roots and Tubers", "value": 'Roots and Tubers'},
                            {"label": "Sugar Crops", "value": 'Sugar Crops'},
                            {"label": "Vegetables and Fruits", "value": 'Vegetables and Fruits'}],
                        multi=False,
                        value="Agriculture",
                        style={'width': "60%"}
                        ),
                
                dcc.Graph(id='x-time-series'),
                dcc.Graph(id='y-time-series'),
                
            ], className='create_container five columns')# style={'display': 'inline-block', 'width': '40%'})
        
    ]),

        html.Div([
            html.H1(children='ANALYSIS BY COUNTRY',
                    style={'textAlign': 'center',
                           'color': 'white'})
        ], style={'display': 'inline-block', 'width': '100%'}),

        # Second panel
    html.Div([
        html.Div([
            html.P('Select Country:', className='fix_label', style={'color': 'white'}),
            dcc.Dropdown(id = 'w_countries',
                         multi = False,
                         searchable = True,
                         value = 'Spain',
                         placeholder='Select Country',
                         options=[{'label': c, 'value': c}
                                  for c in (df_totalA["country"].unique())], className='dcc_compon')  ,
            html.P('Select the product:', className='fix_label', style={'color': 'white'}),
            dcc.Dropdown(id = 'w_product',
                         multi = False,
                         searchable = True,
                         value = 'Agriculture',
                         placeholder='Select product',
                         options=[{'label': c, 'value': c}
                                  for c in (prediction_data["item"].unique())], className='dcc_compon')

        ], className='create_container three columns'),
    
        html.Div([
            dcc.Graph(id = 'pie_chart', config={'displayModeBar': 'hover'})
            ], className='create_container four columns'),

        html.Div([
             dcc.Graph(id = 'heat_chart', config={'displayModeBar': 'hover'})
             ], className='create_container five columns')       

    ], className='row flex-display'),
     html.Div([
         html.Div([
             dcc.Graph(id = 'pred_chart', config={'displayModeBar': 'hover'})
             ], className='create_container twelve columns')
    ], className='row flex-display'),
        
    ],className='row flex display')    

    

], id='mainContainer', style={'display': 'flex', 'flex-direction': 'column'})


@app.callback(Output('pie_chart', 'figure'),
              Input('w_countries', 'value'))

def disasters(w_countries):
    number_disastersA = number_disasters.pivot(index='Country',columns='Disaster Subtype',values='Year').reset_index()

    if number_disastersA[number_disastersA['Country'] == w_countries].shape[0] == 0:
        value_heat = 0
        value_cold = 0
        value_seve = 0
    else:
        value_heat = number_disastersA[number_disastersA['Country'] == w_countries]['Heat wave'].iloc[0]
        value_cold = number_disastersA[number_disastersA['Country'] == w_countries]['Cold wave'].iloc[0]
        value_seve = number_disastersA[number_disastersA['Country'] == w_countries]['Severe winter conditions'].iloc[0]
    colors = ['#FDE725FF', '#2A788EFF', '#440154FF']
    return {
        'data': [go.Pie(
            labels=['Heat waves', 'Cold waves', 'Severe winter conditions'],
            values=[value_heat, value_cold, value_seve],
            marker=dict(colors=colors),
            hoverinfo='label+value+percent',
            textinfo='label+value',
            hole=.5
        )],

        'layout':go.Layout(
            title={'text': 'Extreme Temp. Disasters: ' + (w_countries),
                   'y':0.93,
                   'x': 0.5,
                   'xanchor': 'center',
                   'yanchor': 'top'},
            titlefont={'color': 'white',
                       'size':15},
            hovermode='closest',
            paper_bgcolor = '#8e9fbc', 
            plot_bgcolor = '#8e9fbc',
            legend={'orientation': 'h',
                    'bgcolor': '#8e9fbc',
                    'xanchor': 'center', 'x': 0.5, 'y': -0.7}

        )
    }


@app.callback(Output('heat_chart', 'figure'),
              [Input('w_countries', 'value')])

def temperature_heat(w_countries):
    average_country = weather_data[weather_data['Area'] == w_countries]
    fig = go.Figure(go.Heatmap(
                    z=average_country['Value'],
                    x=average_country['Year'],
                    y=average_country['Months'],
                    colorscale='viridis'))

    fig.update_layout(
        title='Average variation per quarter: '+w_countries)

    return fig


@app.callback(Output('pred_chart', 'figure'),
              Input('w_countries', 'value'),
              Input('w_product', 'value'))

def prediction_chart(w_countries, w_product):
    data_country = prediction_data.loc[prediction_data['country'] == w_countries].drop_duplicates().reset_index(drop=True)
    data_item = data_country[data_country['item'] == w_product]

    fig = px.line(data_item, x='year', y='gross_per_capita_pin', color='type', markers=True,
                  color_discrete_sequence=["#440154", "#21918c"],
                  labels={
                      "year": "Year",
                      "gross_per_capita_pin": "Gross per capita PIN"
                  })

    fig.update_yaxes(range=[40, 160])
    fig.update_layout(title=w_product+' - '+w_countries+': Prediction and Real data', legend_title="")

    return fig


@app.callback(
    Output('temp_map', 'figure'),
    Input('slct_item', 'value'))

def update_figure(selected_item):
    cols = ['country', 'year', 'max_temperature_change', 'code', 'population', 'continent']
    years = [1961, 1970, 1980, 1990, 2000, 2010, 2020]
    dff = df_total.copy()
    dff = dff.loc[dff['year'].isin(years), cols].drop_duplicates().reset_index(drop=True)

    fig = px.choropleth_mapbox(dff,
                               geojson=countries,
                               featureidkey="properties.admin",
                               color="max_temperature_change",
                               color_continuous_scale="Viridis",
                               range_color=(0, 9),
                               animation_frame="year",
                               locations="country",
                               hover_name="country",
                               hover_data=["country"],
                               center={"lat": 45.866667, "lon": 10.566667},
                               mapbox_style="open-street-map",
                               zoom=0.1,
                               width=670,
                               height=580,
                               labels={'max_temperature_change': 'Maximum temperature change'}
                               )
    fig.update_layout(title="Maximum Temperature Change by year")
    fig.update_layout(coloraxis_colorbar_title_text='')

    return fig

###############################################

def create_time_series_x(df_dev, title):
    fig = px.line(df_dev, x='year', y='gross_per_capita_pin', markers=True,
                  labels={
                      "year": "Year",
                      "gross_per_capita_pin": "Gross per capita PIN"
                  })

    fig.update_traces(line_color="#440154")
    fig.update_layout(title_text=title, title_x=0.5)
    fig.update_layout(height=270, margin={'l': 20, 'b': 30, 'r': 10, 't': 40})

    return fig

def create_time_series_y(df_dev):
    fig = px.scatter(df_dev,
                     x="gross_per_capita_pin",
                     y="max_temperature_change",
                     size="population",
                     size_max=10,
                     labels={
                         "gross_per_capita_pin": "Gross per capita PIN",
                         "max_temperature_change": "Max. Temperature Change"},
                     color="max_temperature_change",
                     color_continuous_scale="Viridis",
                     hover_name="year"
                     )

    year = min(df_dev["year"])
    fig.add_annotation(x=df_dev.loc[df_dev["year"] == year, "gross_per_capita_pin"].values[0],
                       y=df_dev.loc[df_dev["year"] == year, "max_temperature_change"].values[0] - 0.15,
                       text=str(year),
                       showarrow=False)

    year = max(df_dev["year"])
    fig.add_annotation(x=df_dev.loc[df_dev["year"] == year, "gross_per_capita_pin"].values[0],
                       y=df_dev.loc[df_dev["year"] == year, "max_temperature_change"].values[0] - 0.15,
                       text=str(year),
                       showarrow=False)

    fig.update_layout(coloraxis_colorbar_title_text='')
    fig.update_layout(coloraxis_showscale = False)
    fig.update_layout(height=270, margin={'l': 20, 'b': 30, 'r': 10, 't': 40})

    return fig

@app.callback(
    Output('x-time-series', 'figure'),
    Output('y-time-series', 'figure'),
    Input('temp_map', 'hoverData'),
    Input('slct_item', 'value'))
def update_x_timeseries(hoverData, item_name):
    country = hoverData['points'][0]['customdata']

    if country != "Belgium":
        country_name = country[0]
    else:
        country_name = country

    dff = df_total[df_total['country'] == country_name]
    dff = dff[dff['item'] == item_name]
    title = '<b>{}</b>'.format(country_name)
    return create_time_series_x(dff, title), create_time_series_y(dff)


if __name__ == '__main__':
    app.run(debug=False, port=(os.getenv("PORT", "1010")))
